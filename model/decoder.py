"""
Non-negative linear decoder for NMF-VAE.

The decoder weight matrix W is constrained to be non-negative via softplus,
giving the model NMF-like interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NNDecoder(nn.Module):
    """
    Non-negative linear decoder.

    Computes: μ = softplus(W_raw) @ z * library_scale
              θ = exp(log_theta)

    W_raw is an unconstrained parameter; softplus enforces non-negativity.

    Args:
        latent_dim: Dimensionality of latent factors.
        output_dim: Number of output genes.
        hidden_dims: Ignored (reserved for future non-linear decoder).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims=None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Raw (unconstrained) weight matrix; softplus applied in forward
        self.W_raw = nn.Parameter(torch.randn(output_dim, latent_dim) * 0.01)

        # Per-gene log overdispersion
        self.log_theta = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        z: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent factors to gene expression parameters.

        Args:
            z: Latent factors, shape (batch, latent).
            library_size: Per-cell library sizes, shape (batch, 1) or None.

        Returns:
            mu: Predicted gene means, shape (batch, genes).
            theta: Per-gene overdispersion, shape (genes,) broadcast to (batch, genes).
        """
        W = F.softplus(self.W_raw)  # (genes, latent)

        # μ = z W^T  (batch, genes)
        mu = z @ W.t()

        if library_size is not None:
            # Scale by library size (broadcast over genes)
            mu = mu * library_size

        # Ensure positivity
        mu = F.softplus(mu) + 1e-8

        theta = torch.exp(self.log_theta)  # (genes,)

        return mu, theta

    @property
    def W(self) -> torch.Tensor:
        """Non-negative weight matrix (genes × latent)."""
        return F.softplus(self.W_raw)
