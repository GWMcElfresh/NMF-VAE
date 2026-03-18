"""
NMF-VAE: main model combining Weibull encoder and non-negative decoder.

Module-level API functions mirror a scikit-learn-style interface for
convenience when using the model interactively.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .decoder import NNDecoder
from .distributions import WeibullDistribution, kl_weibull_gamma, nb_log_likelihood
from .encoder import WeibullEncoder


class NMFVAE(nn.Module):
    """
    NMF-like Variational Autoencoder for single-cell RNA-seq data.

    Uses a Weibull approximate posterior and a Gamma prior to encourage
    non-negative, NMF-interpretable latent factors.

    Args:
        input_dim: Number of input genes.
        latent_dim: Number of latent factors.
        hidden_dims: Hidden layer sizes for the encoder.
        gamma_alpha: Shape of the Gamma prior.
        gamma_beta: Rate of the Gamma prior.
        use_nb: If True use Negative Binomial likelihood, else Poisson.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        gamma_alpha: float = 1.0,
        gamma_beta: float = 1.0,
        use_nb: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.use_nb = use_nb

        self.encoder = WeibullEncoder(input_dim, latent_dim, hidden_dims)
        self.decoder = NNDecoder(latent_dim, input_dim)

        # Track training loss history
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Core model methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Weibull parameters (k, λ) for the posterior."""
        return self.encoder(x)

    def reparameterize(
        self, k: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample z ~ Weibull(k, λ) using the reparameterization trick.

        During evaluation, returns the Weibull mean: λ * Γ(1 + 1/k).
        """
        if self.training:
            q = WeibullDistribution(k, lam)
            return q.rsample()
        else:
            # Deterministic mean for evaluation
            return lam * torch.exp(torch.lgamma(1.0 + 1.0 / k))

    def decode(
        self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, theta) from the decoder."""
        return self.decoder(z, library_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: Count matrix, shape (batch, genes).

        Returns:
            mu: Predicted means, shape (batch, genes).
            theta: Overdispersion, shape (genes,).
            k: Weibull shape, shape (batch, latent).
            lam: Weibull scale, shape (batch, latent).
        """
        # Compute per-cell library size for scaling
        library_size = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Normalize to mean 1 so decoder weights are on a common scale
        library_size = library_size / library_size.mean()

        k, lam = self.encode(x)
        z = self.reparameterize(k, lam)
        mu, theta = self.decode(z, library_size)

        return mu, theta, k, lam

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def elbo_loss(
        self,
        x: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute -ELBO = reconstruction_loss + kl_weight * kl_loss.

        Args:
            x: Count matrix, shape (batch, genes).
            kl_weight: Annealing weight on the KL term.

        Returns:
            loss: Scalar -ELBO.
            recon_loss: Scalar reconstruction loss.
            kl_loss: Scalar KL divergence.
        """
        mu, theta, k, lam = self.forward(x)

        # Reconstruction loss
        if self.use_nb:
            # Broadcast theta: (genes,) → (batch, genes)
            theta_b = theta.unsqueeze(0).expand_as(mu)
            log_p = nb_log_likelihood(x, mu, theta_b)
        else:
            # Poisson log-likelihood
            eps = 1e-8
            log_p = x * torch.log(mu + eps) - mu - torch.lgamma(x + 1.0)

        recon_loss = -log_p.sum(dim=1).mean()

        # KL divergence (MC)
        kl = kl_weibull_gamma(
            k, lam, self.gamma_alpha, self.gamma_beta, n_samples=5
        )
        kl_loss = kl.sum(dim=1).mean()

        loss = recon_loss + kl_weight * kl_loss

        return loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
        kl_warmup_epochs: int = 10,
        device: Optional[str] = None,
    ) -> List[float]:
        """
        Train the model.

        Args:
            dataloader: DataLoader yielding count tensors.
            epochs: Number of training epochs.
            lr: Learning rate.
            kl_weight: Final KL annealing weight.
            kl_warmup_epochs: Number of epochs for linear KL warmup.
            device: 'cpu', 'cuda', or None (auto-detect).

        Returns:
            List of per-epoch losses.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            # Linear KL warmup
            if kl_warmup_epochs > 0:
                warmup_weight = min(1.0, (epoch + 1) / kl_warmup_epochs) * kl_weight
            else:
                warmup_weight = kl_weight

            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)

                optimizer.zero_grad()
                loss, _, _ = self.elbo_loss(x, kl_weight=warmup_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

        return self.loss_history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transform(
        self,
        x: Union[torch.Tensor, DataLoader, np.ndarray],
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode data to latent representations.

        Args:
            x: Count matrix (tensor, numpy array) or DataLoader.
            device: Compute device.

        Returns:
            Z: Latent matrix, shape (cells, latent).
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if isinstance(x, torch.Tensor):
            x = x.to(device)
            k, lam = self.encode(x)
            z = self.reparameterize(k, lam)
            return z.cpu().numpy()

        # DataLoader path
        zs = []
        for batch in x:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            k, lam = self.encode(batch)
            z = self.reparameterize(k, lam)
            zs.append(z.cpu().numpy())
        return np.concatenate(zs, axis=0)

    def get_gene_programs(self) -> np.ndarray:
        """
        Return decoder weight matrix W (genes × latent).

        Each column is a gene program (factor).

        Returns:
            W: Non-negative weight matrix, shape (genes, latent).
        """
        return self.decoder.W.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

_global_model: Optional[NMFVAE] = None


def fit_model(
    count_matrix,
    metadata=None,
    config: Optional[Dict] = None,
) -> NMFVAE:
    """
    Create and train an NMFVAE model.

    Args:
        count_matrix: np.ndarray of shape (cells, genes) OR anndata.AnnData.
        metadata: Optional cell metadata (unused in training, stored for export).
        config: Dict with keys: latent_dim, hidden_dims, epochs, batch_size,
                lr, kl_weight, gamma_alpha, gamma_beta, use_nb.

    Returns:
        Trained NMFVAE model.
    """
    global _global_model

    if config is None:
        config = {}

    # Handle AnnData input
    try:
        import anndata
        if isinstance(count_matrix, anndata.AnnData):
            X = count_matrix.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            count_matrix = np.array(X, dtype=np.float32)
    except ImportError:
        pass

    if not isinstance(count_matrix, np.ndarray):
        count_matrix = np.array(count_matrix, dtype=np.float32)

    n_cells, n_genes = count_matrix.shape

    latent_dim = config.get("latent_dim", 10)
    hidden_dims = config.get("hidden_dims", [256, 128])
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 256)
    lr = config.get("lr", 1e-3)
    kl_weight = config.get("kl_weight", 1.0)
    gamma_alpha = config.get("gamma_alpha", 1.0)
    gamma_beta = config.get("gamma_beta", 1.0)
    use_nb = config.get("use_nb", True)

    # Inline dataloader creation to avoid cross-package import issues
    tensor = torch.tensor(count_matrix, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NMFVAE(
        input_dim=n_genes,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        gamma_alpha=gamma_alpha,
        gamma_beta=gamma_beta,
        use_nb=use_nb,
    )
    model.fit(dataloader, epochs=epochs, lr=lr, kl_weight=kl_weight)

    _global_model = model
    return model


def transform(count_matrix, model: Optional[NMFVAE] = None) -> np.ndarray:
    """
    Encode count_matrix to latent representations.

    Args:
        count_matrix: np.ndarray (cells, genes).
        model: NMFVAE model. Uses global model if None.

    Returns:
        Z: Latent matrix (cells, latent).
    """
    if model is None:
        if _global_model is None:
            raise RuntimeError("No model available. Call fit_model first.")
        model = _global_model

    if not isinstance(count_matrix, np.ndarray):
        count_matrix = np.array(count_matrix, dtype=np.float32)

    x = torch.tensor(count_matrix, dtype=torch.float32)
    return model.transform(x)


def get_gene_programs(model: Optional[NMFVAE] = None) -> np.ndarray:
    """
    Return decoder weight matrix.

    Args:
        model: NMFVAE model. Uses global model if None.

    Returns:
        W: Shape (genes, latent).
    """
    if model is None:
        if _global_model is None:
            raise RuntimeError("No model available. Call fit_model first.")
        model = _global_model
    return model.get_gene_programs()


def plot_latent_space(
    count_matrix=None,
    Z: Optional[np.ndarray] = None,
    metadata=None,
    color_by: Optional[str] = None,
    method: str = "umap",
    save_path: Optional[str] = None,
    model: Optional[NMFVAE] = None,
):
    """Plot latent space embedding. Delegates to plot_utils."""
    from utils.plot_utils import plot_latent_space as _plot

    if Z is None:
        if count_matrix is None:
            raise ValueError("Either count_matrix or Z must be provided.")
        Z = transform(count_matrix, model=model)

    _plot(Z, metadata=metadata, color_by=color_by, method=method, save_path=save_path)


def export_results(
    output_dir: str,
    count_matrix=None,
    Z: Optional[np.ndarray] = None,
    metadata=None,
    model: Optional[NMFVAE] = None,
):
    """
    Save latent matrix Z, weight matrix W, and diagnostics.

    Args:
        output_dir: Directory to write outputs to.
        count_matrix: Input data (used to compute Z if Z not given).
        Z: Pre-computed latent matrix.
        metadata: Cell metadata DataFrame.
        model: NMFVAE model. Uses global model if None.
    """
    if model is None:
        if _global_model is None:
            raise RuntimeError("No model available.")
        model = _global_model

    if Z is None:
        if count_matrix is None:
            raise ValueError("Either count_matrix or Z must be provided.")
        Z = transform(count_matrix, model=model)

    W = get_gene_programs(model=model)

    # Inline output writing to avoid cross-package import issues
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    z_df = pd.DataFrame(Z, columns=[f"latent_{i}" for i in range(Z.shape[1])])
    if metadata is not None:
        try:
            metadata = metadata.reset_index(drop=True)
            z_df = pd.concat([metadata, z_df], axis=1)
        except Exception:
            pass
    z_df.to_csv(os.path.join(output_dir, "latent_Z.csv"), index=False)

    w_df = pd.DataFrame(W, columns=[f"factor_{i}" for i in range(W.shape[1])])
    w_df.to_csv(os.path.join(output_dir, "decoder_W.csv"), index=False)

    loss_df = pd.DataFrame({"epoch": range(len(model.loss_history)), "loss": model.loss_history})
    loss_df.to_csv(os.path.join(output_dir, "loss_history.csv"), index=False)
