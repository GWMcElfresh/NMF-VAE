"""
Tests for NMF-VAE model components and API.

All tests use small synthetic data (50 cells, 100 genes).
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

N_CELLS = 50
N_GENES = 100
LATENT_DIM = 5


@pytest.fixture
def synthetic_counts():
    """Return a (50, 100) random integer count matrix."""
    rng = np.random.default_rng(42)
    return rng.negative_binomial(5, 0.5, size=(N_CELLS, N_GENES)).astype(np.float32)


@pytest.fixture
def count_tensor(synthetic_counts):
    return torch.tensor(synthetic_counts, dtype=torch.float32)


@pytest.fixture
def dataloader(synthetic_counts):
    tensor = torch.tensor(synthetic_counts, dtype=torch.float32)
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=16, shuffle=True)


# --------------------------------------------------------------------------
# Distribution tests
# --------------------------------------------------------------------------


def test_distributions():
    """Test WeibullDistribution rsample, log_prob and kl_weibull_gamma."""
    from model.distributions import WeibullDistribution, kl_weibull_gamma

    batch, latent = 8, 5
    k = torch.ones(batch, latent) * 2.0
    lam = torch.ones(batch, latent) * 1.5

    dist = WeibullDistribution(k, lam)

    # rsample
    z = dist.rsample()
    assert z.shape == (batch, latent), f"Expected {(batch, latent)}, got {z.shape}"
    assert (z > 0).all(), "Weibull samples should be positive"

    # log_prob
    log_p = dist.log_prob(z)
    assert log_p.shape == (batch, latent)
    assert torch.isfinite(log_p).all(), "log_prob should be finite"

    # KL
    kl = kl_weibull_gamma(k, lam, gamma_alpha=1.0, gamma_beta=1.0)
    assert kl.shape == (batch, latent)
    assert torch.isfinite(kl).all(), "KL should be finite"


# --------------------------------------------------------------------------
# Encoder tests
# --------------------------------------------------------------------------


def test_encoder(count_tensor):
    """Test WeibullEncoder forward pass."""
    from model.encoder import WeibullEncoder

    encoder = WeibullEncoder(N_GENES, LATENT_DIM, hidden_dims=[64, 32])
    encoder.eval()

    with torch.no_grad():
        k, lam = encoder(count_tensor)

    assert k.shape == (N_CELLS, LATENT_DIM), f"k shape: {k.shape}"
    assert lam.shape == (N_CELLS, LATENT_DIM), f"lam shape: {lam.shape}"
    assert (k > 0).all(), "k should be positive"
    assert (lam > 0).all(), "lam should be positive"


# --------------------------------------------------------------------------
# Decoder tests
# --------------------------------------------------------------------------


def test_decoder():
    """Test NNDecoder forward pass."""
    from model.decoder import NNDecoder

    decoder = NNDecoder(LATENT_DIM, N_GENES)
    decoder.eval()

    z = torch.abs(torch.randn(N_CELLS, LATENT_DIM))  # positive inputs

    with torch.no_grad():
        mu, theta = decoder(z)

    assert mu.shape == (N_CELLS, N_GENES), f"mu shape: {mu.shape}"
    assert theta.shape == (N_GENES,), f"theta shape: {theta.shape}"
    assert (mu > 0).all(), "mu should be positive"
    assert (theta > 0).all(), "theta should be positive"

    # Test W property
    W = decoder.W
    assert W.shape == (N_GENES, LATENT_DIM)
    assert (W >= 0).all(), "W should be non-negative"


# --------------------------------------------------------------------------
# VAE forward pass
# --------------------------------------------------------------------------


def test_vae_forward(count_tensor):
    """Test full NMFVAE forward pass."""
    from model.vae import NMFVAE

    model = NMFVAE(N_GENES, LATENT_DIM, hidden_dims=[64, 32])
    model.eval()

    with torch.no_grad():
        mu, theta, k, lam = model(count_tensor)

    assert mu.shape == (N_CELLS, N_GENES), f"mu: {mu.shape}"
    assert theta.shape == (N_GENES,), f"theta: {theta.shape}"
    assert k.shape == (N_CELLS, LATENT_DIM), f"k: {k.shape}"
    assert lam.shape == (N_CELLS, LATENT_DIM), f"lam: {lam.shape}"


# --------------------------------------------------------------------------
# VAE training
# --------------------------------------------------------------------------


def test_vae_training(dataloader):
    """Train for 5 epochs and check loss doesn't NaN."""
    from model.vae import NMFVAE

    model = NMFVAE(N_GENES, LATENT_DIM, hidden_dims=[64, 32])

    losses = model.fit(dataloader, epochs=5, lr=1e-3, kl_weight=0.1, kl_warmup_epochs=2)

    assert len(losses) == 5
    assert all(np.isfinite(l) for l in losses), f"Non-finite losses: {losses}"


# --------------------------------------------------------------------------
# Data utils
# --------------------------------------------------------------------------


def test_data_utils(synthetic_counts, tmp_path):
    """Test tensor conversion, dataloader creation, and write_outputs."""
    from utils.data_utils import to_tensor, create_dataloader, write_outputs

    # to_tensor
    tensor = to_tensor(synthetic_counts)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (N_CELLS, N_GENES)

    # create_dataloader
    dl = create_dataloader(synthetic_counts, batch_size=16)
    batch = next(iter(dl))[0]
    assert batch.shape[1] == N_GENES

    # write_outputs
    Z = np.random.randn(N_CELLS, LATENT_DIM).astype(np.float32)
    W = np.random.rand(N_GENES, LATENT_DIM).astype(np.float32)
    loss_history = [1.0, 0.9, 0.8]
    out = str(tmp_path / "results")
    write_outputs(out, Z, W, None, loss_history)

    import os
    assert os.path.exists(os.path.join(out, "latent_Z.csv"))
    assert os.path.exists(os.path.join(out, "decoder_W.csv"))
    assert os.path.exists(os.path.join(out, "loss_history.csv"))


# --------------------------------------------------------------------------
# High-level API tests
# --------------------------------------------------------------------------


def test_fit_model_api(synthetic_counts):
    """Test high-level fit_model function."""
    from model.vae import fit_model

    config = {
        "latent_dim": LATENT_DIM,
        "hidden_dims": [64, 32],
        "epochs": 3,
        "batch_size": 16,
        "lr": 1e-3,
        "kl_weight": 0.1,
    }
    model = fit_model(synthetic_counts, config=config)
    assert model is not None
    assert len(model.loss_history) == 3


def test_transform_api(synthetic_counts):
    """Test high-level transform function."""
    from model.vae import fit_model, transform

    config = {
        "latent_dim": LATENT_DIM,
        "hidden_dims": [64, 32],
        "epochs": 2,
        "batch_size": 16,
        "lr": 1e-3,
    }
    model = fit_model(synthetic_counts, config=config)
    Z = transform(synthetic_counts, model=model)

    assert Z.shape == (N_CELLS, LATENT_DIM), f"Z shape: {Z.shape}"
    assert np.isfinite(Z).all(), "Z contains non-finite values"


def test_get_gene_programs(synthetic_counts):
    """Test get_gene_programs returns correct shape."""
    from model.vae import fit_model, get_gene_programs

    config = {
        "latent_dim": LATENT_DIM,
        "hidden_dims": [64, 32],
        "epochs": 2,
        "batch_size": 16,
        "lr": 1e-3,
    }
    model = fit_model(synthetic_counts, config=config)
    W = get_gene_programs(model=model)

    assert W.shape == (N_GENES, LATENT_DIM), f"W shape: {W.shape}"
    assert (W >= 0).all(), "W should be non-negative"
