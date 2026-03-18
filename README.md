# NMF-VAE

A **Variational Autoencoder (VAE)** for single-cell RNA-seq data with a non-negative, sparse, interpretable latent space that approximates **Gamma-Poisson (probabilistic NMF) factorization**.

## Overview

NMF-VAE learns latent **gene programs** from single-cell RNA-seq count data:

- **Non-negative latent factors** – each cell is a non-negative combination of programs
- **Sparse representations** – KL divergence with a Gamma prior encourages sparsity
- **Interpretable decoder** – non-negative weight matrix W links factors to genes
- **Negative Binomial / Poisson likelihood** – models overdispersed count data
- **Weibull approximate posterior** – provides a reparameterizable Gamma approximation

The model most closely resembles a VAE generalization of **scHPF / Poisson NMF**.

## Repository Structure

```
NMF-VAE/
├── model/
│   ├── distributions.py   # Weibull distribution, KL, NB log-likelihood
│   ├── encoder.py         # Weibull encoder network
│   ├── decoder.py         # Non-negative (NMF-like) decoder
│   └── vae.py             # NMFVAE model + scikit-learn-style API
├── utils/
│   ├── data_utils.py      # Data loading (h5ad, mtx, csv), DataLoader utils
│   └── plot_utils.py      # Latent space plots, ELBO curves, gene loadings
├── scripts/
│   ├── train.py           # CLI training script
│   └── preprocess.py      # CLI preprocessing / QC script
├── tests/
│   └── test_model.py      # pytest unit tests
├── data/                  # Place input data here (gitignored by default)
├── Dockerfile
├── singularity.def
└── .github/workflows/ci.yml
```

## Installation

### Local (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

### Docker

```bash
docker build -t nmfvae .
docker run --rm nmfvae
```

### Singularity (HPC)

```bash
singularity build nmfvae.sif singularity.def
singularity run nmfvae.sif python scripts/train.py --help
```

## Quick Start

### Python API

```python
import numpy as np
from model.vae import fit_model, transform, get_gene_programs, export_results

# Simulate count data
X = np.random.negative_binomial(5, 0.5, size=(500, 2000)).astype(np.float32)

# Train model
model = fit_model(
    X,
    config={
        "latent_dim": 15,
        "epochs": 100,
        "batch_size": 128,
        "lr": 1e-3,
    }
)

# Get latent representations (cells × latent)
Z = transform(X, model=model)

# Get gene programs (genes × latent)
W = get_gene_programs(model=model)

# Save results
export_results("results/", count_matrix=X, model=model)
```

### AnnData / Scanpy

```python
import scanpy as sc
from model.vae import fit_model, transform

adata = sc.read_h5ad("my_data.h5ad")
model = fit_model(adata, config={"latent_dim": 20, "epochs": 200})
Z = transform(adata.X, model=model)
adata.obsm["X_nmfvae"] = Z
sc.pl.umap(adata, color="cell_type")
```

### CLI

```bash
# Preprocess
python scripts/preprocess.py \
    --input data/raw.h5ad \
    --output data/processed.npz \
    --min-genes 200 --min-cells 10 --normalize --log1p

# Train
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --latent-dim 20 \
    --epochs 200 \
    --batch-size 256
```

## Model Details

### Generative model

```
z_ik ~ Gamma(alpha, beta)          # Sparse non-negative latent factors
mu_i = softplus(W) @ z_i           # Non-negative reconstruction
x_ij ~ NB(mu_ij, theta_j)          # Negative Binomial counts
```

### Inference

```
q(z_i | x_i) = Weibull(k_i, lambda_i)   # Approximate posterior
(k_i, lambda_i) = encoder(x_i)           # Neural network encoder
```

### ELBO

```
L = E_q[log p(x|z)] - KL[q(z|x) || p(z)]
```

## Tests

```bash
pytest tests/ -v
```

All 9 unit tests cover:
- WeibullDistribution (rsample, log_prob)
- Encoder / Decoder forward passes
- Full model forward + ELBO
- End-to-end training (5 epochs on synthetic data)
- Data utilities
- High-level API (fit_model, transform, get_gene_programs)

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:

1. **test** job – installs CPU PyTorch, runs pytest
2. **docker** job – builds Docker image, verifies size < 14 GB, pushes to GHCR on `main`

## References

- Lopez et al. (2018) – scVI
- Levitin et al. (2019) – scHPF
- Carbonetto et al. (2022) – LDVAE
- Zhang et al. (2020) – ZINB-WaVE
