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


# --------------------------------------------------------------------------
# Graph utility tests
# --------------------------------------------------------------------------


def test_resolve_lambda_presets():
    """Test that named presets resolve to the expected float values."""
    from utils.graph_utils import resolve_lambda, LAMBDA_PRESETS

    for name, expected in LAMBDA_PRESETS.items():
        assert resolve_lambda(name) == expected, f"Preset '{name}' mismatch"

    # Case insensitive
    assert resolve_lambda("WEAK") == LAMBDA_PRESETS["weak"]
    assert resolve_lambda("None") == 0.0


def test_resolve_lambda_float():
    """Test that a numeric lambda value passes through unchanged."""
    from utils.graph_utils import resolve_lambda

    assert resolve_lambda(0.5) == 0.5
    assert resolve_lambda(0) == 0.0
    assert resolve_lambda(1e-3) == pytest.approx(1e-3)


def test_resolve_lambda_invalid():
    """Test that invalid inputs raise ValueError."""
    from utils.graph_utils import resolve_lambda

    with pytest.raises(ValueError):
        resolve_lambda("unknown_preset")

    with pytest.raises(ValueError):
        resolve_lambda(-0.1)


def test_build_laplacian_unnormalized():
    """Test unnormalized Laplacian L = D - A."""
    from utils.graph_utils import build_laplacian_from_adjacency

    # Simple two-node graph with weight 1
    A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    L = build_laplacian_from_adjacency(A, normalized=False)

    expected = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(L, expected, atol=1e-6)


def test_build_laplacian_normalized():
    """Test symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    from utils.graph_utils import build_laplacian_from_adjacency

    # 3-node path graph: 0 -- 1 -- 2
    A = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    L = build_laplacian_from_adjacency(A, normalized=True)

    # L should be symmetric
    np.testing.assert_allclose(L, L.T, atol=1e-6)

    # Diagonal should be 1 for nodes with edges
    assert L[0, 0] == pytest.approx(1.0, abs=1e-6)

    # L should be positive semi-definite: all eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(L)
    assert (eigvals >= -1e-6).all(), f"Non-PSD eigenvalues: {eigvals}"


def test_build_laplacian_isolated_nodes():
    """Isolated nodes (zero degree) are handled gracefully."""
    from utils.graph_utils import build_laplacian_from_adjacency

    # One connected pair + one isolated node
    A = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    L = build_laplacian_from_adjacency(A, normalized=True)
    assert np.isfinite(L).all(), "Laplacian should be finite with isolated nodes"


def test_laplacian_penalty_zero_for_constant_genes():
    """Penalty should be zero when connected genes have identical weight rows."""
    from model.vae import NMFVAE

    # Two genes connected with weight 1 (unnormalized Laplacian)
    L = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    # Both genes have the same weight vector → penalty must be 0
    W = torch.tensor([[2.0, 3.0], [2.0, 3.0]])  # rows identical
    penalty = NMFVAE.laplacian_penalty(W, L)
    assert penalty.item() == pytest.approx(0.0, abs=1e-5), (
        f"Penalty should be 0 for constant gene weights, got {penalty.item()}"
    )


def test_laplacian_penalty_positive_for_divergent_genes():
    """Penalty should increase when connected genes have different weight rows."""
    from model.vae import NMFVAE

    # Two genes connected with weight 1 (unnormalized Laplacian)
    L = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    # Identical weights → penalty = 0
    W_const = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    penalty_const = NMFVAE.laplacian_penalty(W_const, L)

    # Divergent weights → penalty > 0
    W_div = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    penalty_div = NMFVAE.laplacian_penalty(W_div, L)

    assert penalty_div.item() > penalty_const.item() + 1e-6, (
        "Penalty should be larger for divergent gene weights"
    )
    assert penalty_div.item() > 0.0


def test_vae_training_with_graph_penalty(dataloader):
    """Model should train successfully when graph Laplacian penalty is active."""
    from model.vae import NMFVAE
    from utils.graph_utils import build_laplacian_from_adjacency

    # Build a trivial chain-graph Laplacian over N_GENES genes
    A = np.zeros((N_GENES, N_GENES), dtype=np.float32)
    for i in range(N_GENES - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    L = build_laplacian_from_adjacency(A, normalized=False)
    L_tensor = torch.tensor(L, dtype=torch.float32)

    model = NMFVAE(
        N_GENES,
        LATENT_DIM,
        hidden_dims=[64, 32],
        lambda_graph="weak",
        graph_laplacian=L_tensor,
    )
    loss_history = model.fit(dataloader, epochs=5, lr=1e-3, kl_weight=0.1, kl_warmup_epochs=2)

    assert len(loss_history) == 5
    assert all(np.isfinite(l) for l in loss_history), f"Non-finite losses: {loss_history}"


def test_vae_lambda_preset_none_disables_penalty():
    """lambda_graph='none' should disable the penalty (same as 0.0)."""
    from model.vae import NMFVAE

    model = NMFVAE(N_GENES, LATENT_DIM, hidden_dims=[64, 32], lambda_graph="none")
    assert model.lambda_graph == 0.0


def test_set_graph_laplacian():
    """set_graph_laplacian should update the buffer."""
    from model.vae import NMFVAE

    model = NMFVAE(N_GENES, LATENT_DIM, hidden_dims=[64, 32], lambda_graph=0.1)
    assert model._graph_laplacian is None

    L = torch.eye(N_GENES)
    model.set_graph_laplacian(L)
    assert model._graph_laplacian is not None
    assert model._graph_laplacian.shape == (N_GENES, N_GENES)


def test_build_coexpression_laplacian(synthetic_counts):
    """build_coexpression_laplacian should return a valid PSD Laplacian."""
    from utils.graph_utils import build_coexpression_laplacian

    L = build_coexpression_laplacian(synthetic_counts, k=5, normalized=True)

    assert isinstance(L, torch.Tensor)
    assert L.shape == (N_GENES, N_GENES)
    assert torch.isfinite(L).all()

    # Symmetry
    torch.testing.assert_close(L, L.t(), atol=1e-5, rtol=0)

    # PSD: all eigenvalues >= 0
    eigvals = torch.linalg.eigvalsh(L)
    assert (eigvals >= -1e-4).all(), f"Non-PSD eigenvalues: {eigvals.min().item()}"


def test_build_hybrid_laplacian():
    """build_hybrid_laplacian should interpolate correctly."""
    from utils.graph_utils import build_hybrid_laplacian

    L1 = torch.eye(4)
    L2 = torch.zeros(4, 4)

    # alpha=1.0 → all L1
    L_hybrid = build_hybrid_laplacian(L1, L2, alpha=1.0)
    torch.testing.assert_close(L_hybrid, L1)

    # alpha=0.0 → all L2
    L_hybrid = build_hybrid_laplacian(L1, L2, alpha=0.0)
    torch.testing.assert_close(L_hybrid, L2)

    # alpha=0.5 → average
    L_hybrid = build_hybrid_laplacian(L1, L2, alpha=0.5)
    torch.testing.assert_close(L_hybrid, 0.5 * L1)


def test_fit_model_api_with_graph_penalty(synthetic_counts):
    """fit_model should accept lambda_graph and graph_laplacian in config."""
    from model.vae import fit_model
    from utils.graph_utils import build_laplacian_from_adjacency

    A = np.eye(N_GENES, dtype=np.float32)  # trivial identity-graph Laplacian → zero
    L = build_laplacian_from_adjacency(A, normalized=False)
    L_tensor = torch.tensor(L, dtype=torch.float32)

    config = {
        "latent_dim": LATENT_DIM,
        "hidden_dims": [64, 32],
        "epochs": 2,
        "batch_size": 16,
        "lr": 1e-3,
        "kl_weight": 0.1,
        "lambda_graph": "moderate",
        "graph_laplacian": L_tensor,
    }
    model = fit_model(synthetic_counts, config=config)
    assert model is not None
    assert model.lambda_graph == pytest.approx(0.1)
    assert len(model.loss_history) == 2

