"""
Tests for NMF-VAE model components and API.

All tests use small synthetic data (50 cells, 100 genes).
"""

import numpy as np
import os
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


# --------------------------------------------------------------------------
# Signed Laplacian tests
# --------------------------------------------------------------------------


def test_build_signed_laplacian_unnormalized():
    """Unnormalized signed Laplacian: L = D_{|A|} - A."""
    from utils.graph_utils import build_signed_laplacian_from_adjacency

    # One positive and one negative edge
    A = np.array([[0.0, 1.0, -0.5], [1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=np.float32)
    L = build_signed_laplacian_from_adjacency(A, normalized=False)

    # Diagonal should be sum of |A| row
    assert L[0, 0] == pytest.approx(1.5, abs=1e-6)
    assert L[1, 1] == pytest.approx(1.0, abs=1e-6)
    assert L[2, 2] == pytest.approx(0.5, abs=1e-6)

    # Off-diagonal: L = D - A, so L[0,1] = 0 - A[0,1] = -1.0
    assert L[0, 1] == pytest.approx(-1.0, abs=1e-6)
    assert L[0, 2] == pytest.approx(0.5, abs=1e-6)  # -(-0.5) = 0.5


def test_build_signed_laplacian_normalized():
    """Normalized signed Laplacian should be symmetric and finite."""
    from utils.graph_utils import build_signed_laplacian_from_adjacency

    A = np.array([[0.0, 0.8, -0.3], [0.8, 0.0, 0.4], [-0.3, 0.4, 0.0]], dtype=np.float32)
    L = build_signed_laplacian_from_adjacency(A, normalized=True)

    assert L.shape == (3, 3)
    assert np.isfinite(L).all()
    np.testing.assert_allclose(L, L.T, atol=1e-6)


def test_build_signed_laplacian_isolated_nodes():
    """Isolated nodes (zero absolute degree) are handled without NaNs."""
    from utils.graph_utils import build_signed_laplacian_from_adjacency

    A = np.array(
        [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    L = build_signed_laplacian_from_adjacency(A, normalized=True)
    assert np.isfinite(L).all()


# --------------------------------------------------------------------------
# Correlation Laplacian tests (using a small synthetic pkl)
# --------------------------------------------------------------------------


def _make_correlation_pkl(path, genes):
    """Write a tiny synthetic correlation DataFrame as a pickle."""
    import pandas as pd
    import pickle

    n = len(genes)
    rng = np.random.default_rng(0)
    corr = rng.uniform(-1, 1, size=(n, n)).astype(np.float32)
    # Make symmetric and zero diagonal
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 0.0)
    df = pd.DataFrame(corr, index=genes, columns=genes)
    with open(path, "wb") as fh:
        pickle.dump(df, fh)
    return df


def test_build_correlation_laplacian_basic(tmp_path):
    """build_correlation_laplacian should return a valid tensor for matched genes."""
    from utils.graph_utils import build_correlation_laplacian

    genes = [f"GENE{i}" for i in range(10)]
    pkl_path = str(tmp_path / "corr.pkl")
    _make_correlation_pkl(pkl_path, genes)

    L, matched = build_correlation_laplacian(
        genes,
        pkl_path=pkl_path,
        correlation_threshold=0.3,
        normalized=True,
        convert_ncbi=False,  # skip network call in tests
    )

    assert isinstance(L, torch.Tensor)
    assert L.shape == (10, 10)
    assert torch.isfinite(L).all()
    assert len(matched) == 10
    assert all(matched), "All genes should match the synthetic pkl"


def test_build_correlation_laplacian_unmatched_genes(tmp_path):
    """Unmatched genes should receive the weak_prior_diagonal value."""
    from utils.graph_utils import build_correlation_laplacian

    pkl_genes = [f"GENE{i}" for i in range(8)]
    query_genes = pkl_genes + ["LOC123456", "NOVEL99"]  # 2 unmatched
    pkl_path = str(tmp_path / "corr.pkl")
    _make_correlation_pkl(pkl_path, pkl_genes)

    weak_val = 0.05
    L, matched = build_correlation_laplacian(
        query_genes,
        pkl_path=pkl_path,
        correlation_threshold=0.3,
        normalized=True,
        weak_prior_diagonal=weak_val,
        convert_ncbi=False,
    )

    assert L.shape == (10, 10)
    assert matched.count(False) == 2
    # Unmatched gene indices 8 and 9 should have L[i,i] == weak_val
    assert L[8, 8].item() == pytest.approx(weak_val, abs=1e-6)
    assert L[9, 9].item() == pytest.approx(weak_val, abs=1e-6)


def test_build_correlation_laplacian_threshold(tmp_path):
    """High threshold should zero out most edges, leaving a sparser graph."""
    from utils.graph_utils import build_correlation_laplacian
    import pandas as pd
    import pickle

    genes = [f"G{i}" for i in range(6)]
    # Craft a correlation matrix with known values
    corr = np.zeros((6, 6), dtype=np.float32)
    corr[0, 1] = corr[1, 0] = 0.9   # strong positive
    corr[2, 3] = corr[3, 2] = -0.8  # strong negative
    corr[4, 5] = corr[5, 4] = 0.2   # below threshold
    df = pd.DataFrame(corr, index=genes, columns=genes)
    pkl_path = str(tmp_path / "corr_thresh.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(df, fh)

    L_high, _ = build_correlation_laplacian(
        genes, pkl_path=pkl_path, correlation_threshold=0.5,
        normalized=False, convert_ncbi=False,
    )
    # Edge (4,5) is below 0.5 → should be zero in L
    assert L_high[4, 5].item() == pytest.approx(0.0, abs=1e-6)
    # Edge (0,1) is 0.9 → should be non-zero
    assert L_high[0, 1].item() != pytest.approx(0.0, abs=1e-6)


def test_build_correlation_laplacian_file_not_found():
    """FileNotFoundError should be raised for a missing pkl path."""
    from utils.graph_utils import build_correlation_laplacian

    with pytest.raises(FileNotFoundError):
        build_correlation_laplacian(
            ["GENE1"], pkl_path="/nonexistent/path.pkl", convert_ncbi=False
        )


# --------------------------------------------------------------------------
# save_laplacian tests
# --------------------------------------------------------------------------


def test_save_laplacian(tmp_path):
    """save_laplacian should write npy and optionally csv."""
    from utils.graph_utils import save_laplacian

    L = torch.eye(5)
    W = np.random.rand(5, 3).astype(np.float32)
    prefix = str(tmp_path / "test_output")

    save_laplacian(L, prefix, W=W, gene_names=["A", "B", "C", "D", "E"])

    assert os.path.exists(f"{prefix}_laplacian.npy")
    assert os.path.exists(f"{prefix}_W.csv")

    L_loaded = np.load(f"{prefix}_laplacian.npy")
    np.testing.assert_allclose(L_loaded, np.eye(5), atol=1e-6)


def test_save_laplacian_no_w(tmp_path):
    """save_laplacian without W should only write the npy file."""
    from utils.graph_utils import save_laplacian

    L = torch.eye(4)
    prefix = str(tmp_path / "lap_only")
    save_laplacian(L, prefix)

    assert os.path.exists(f"{prefix}_laplacian.npy")
    assert not os.path.exists(f"{prefix}_W.csv")


# --------------------------------------------------------------------------
# VAE training with signed Laplacian (end-to-end)
# --------------------------------------------------------------------------


def test_vae_training_with_signed_laplacian(dataloader):
    """Model should train when a signed graph Laplacian (with negative edges) is active."""
    from model.vae import NMFVAE
    from utils.graph_utils import build_signed_laplacian_from_adjacency

    # Build a signed adjacency with alternating positive/negative edges
    A = np.zeros((N_GENES, N_GENES), dtype=np.float32)
    for i in range(N_GENES - 1):
        sign = 1.0 if i % 2 == 0 else -1.0
        A[i, i + 1] = sign * 0.5
        A[i + 1, i] = sign * 0.5
    L = build_signed_laplacian_from_adjacency(A, normalized=True)
    L_tensor = torch.tensor(L, dtype=torch.float32)

    model = NMFVAE(
        N_GENES, LATENT_DIM, hidden_dims=[64, 32],
        lambda_graph="weak", graph_laplacian=L_tensor,
    )
    loss_history = model.fit(dataloader, epochs=3, lr=1e-3, kl_weight=0.1)

    assert len(loss_history) == 3
    assert all(np.isfinite(lo) for lo in loss_history), f"Non-finite losses: {loss_history}"

