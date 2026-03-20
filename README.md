# NMF-VAE

A **Variational Autoencoder (VAE)** for single-cell RNA-seq data with a non-negative, sparse, interpretable latent space that approximates **Gamma-Poisson (probabilistic NMF) factorization**.

## Overview

NMF-VAE learns latent **gene programs** from single-cell RNA-seq count data:

- **Non-negative latent factors** – each cell is a non-negative combination of programs
- **Sparse representations** – KL divergence with a Gamma prior encourages sparsity
- **Interpretable decoder** – non-negative weight matrix W links factors to genes
- **Negative Binomial / Poisson likelihood** – models overdispersed count data
- **Weibull approximate posterior** – provides a reparameterizable Gamma approximation
- **Graph Laplacian regularization** – optional STRING or co-expression network prior on decoder weights

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
│   ├── graph_utils.py     # Graph Laplacian utilities (STRING, co-expression, hybrid)
│   └── plot_utils.py      # Latent space plots, ELBO curves, gene loadings
├── scripts/
│   ├── train.py           # CLI training script
│   └── preprocess.py      # CLI preprocessing / QC script
├── notebooks/
│   └── graph_laplacian_tutorial.ipynb  # Interactive tutorial for graph regularization
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
# Build locally (GPU-capable image using CUDA 12.1)
docker build -t nmfvae .

# Run with GPU (requires nvidia-container-toolkit)
docker run --rm --gpus all nmfvae python scripts/train.py --help

# Run on CPU only (falls back automatically)
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

# Train (no graph prior)
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --latent-dim 20 \
    --epochs 200 \
    --batch-size 256

# Train with STRING graph Laplacian regularization
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --latent-dim 20 \
    --epochs 200 \
    --lambda-graph moderate \
    --use-string-graph \
    --genes-file data/gene_names.txt \
    --confidence-threshold 0.7 \
    --use-normalized-laplacian

# Train with hybrid graph (STRING + co-expression)
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --latent-dim 20 \
    --epochs 200 \
    --lambda-graph 0.05 \
    --use-string-graph \
    --genes-file data/gene_names.txt \
    --use-coexpression-graph \
    --use-hybrid-graph \
    --alpha-graph-mix 0.7
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

---

## Graph Laplacian Regularization

NMF-VAE supports an optional **graph Laplacian penalty** on the decoder weight matrix:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ELBO}} + \lambda \cdot \mathrm{Tr}(W^\top L W)$$

where $W \in \mathbb{R}_+^{G \times K}$ is the decoder weight matrix and $L$ is a graph Laplacian derived from a gene interaction network (e.g. STRING).

The penalty encourages **connected genes** (large $A_{ij}$) to have **similar decoder weight rows**, producing gene programs that align with known biological pathways.

### λ presets

| Preset | λ value | Behavior |
|--------|---------|----------|
| `"none"` | 0.0 | Pure data-driven (disabled) |
| `"weak"` | 0.01 | Soft STRING nudge |
| `"moderate"` | 0.10 | Aligned with STRING pathways |
| `"strong"` | 1.00 | Strongly constrained by STRING |

You may also pass any non-negative float directly (e.g. `lambda_graph=0.05`).

### Python API

```python
from utils.graph_utils import build_string_laplacian
from model.vae import fit_model

# Build STRING Laplacian (requires internet + requests)
L = build_string_laplacian(
    genes                = gene_names,   # list of gene symbols matching count matrix columns
    confidence_threshold = 0.7,
    species_id           = 9606,         # 9606 = human, 10090 = mouse
    normalized           = True,
)

# Train with graph prior
model = fit_model(X, config={
    "latent_dim"      : 20,
    "epochs"          : 200,
    "lambda_graph"    : "moderate",    # or e.g. lambda_graph=0.05
    "graph_laplacian" : L,
})
```

### Data-driven (co-expression) prior

```python
from utils.graph_utils import build_coexpression_laplacian

L = build_coexpression_laplacian(X, k=15, normalized=True)
model = fit_model(X, config={"lambda_graph": "weak", "graph_laplacian": L, ...})
```

### Hybrid graph (STRING + co-expression)

```python
from utils.graph_utils import build_string_laplacian, build_coexpression_laplacian, build_hybrid_laplacian

L_string = build_string_laplacian(gene_names)
L_data   = build_coexpression_laplacian(X, k=15)
L_hybrid = build_hybrid_laplacian(L_string, L_data, alpha=0.7)  # 70% STRING, 30% data

model = fit_model(X, config={"lambda_graph": "moderate", "graph_laplacian": L_hybrid, ...})
```

### Updating the Laplacian at runtime

```python
model.set_graph_laplacian(L_new)   # swap prior without rebuilding model
```

### Interactive tutorial

See **`notebooks/graph_laplacian_tutorial.ipynb`** for an end-to-end walkthrough covering:
- Synthetic dataset construction and baseline training
- Mathematical intuition behind the Laplacian penalty
- λ sweep: `none → weak → moderate → strong`
- Building and visualising the STRING adjacency
- Co-expression kNN graph
- Hybrid graph α sweep (STRING vs data)
- Correlation-based signed Laplacian from a pre-computed pkl file
- Real scRNA-seq workflow reference

## Gene-Gene Correlation Prior (signed Laplacian)

The `human_correlation_v2.4.pkl` file (or similar pre-computed correlation matrices) contains **both positive and negative** gene-gene correlations.  NMF-VAE supports these through a *signed graph Laplacian*:

$$L_s = D_{|A|} - A$$

where $D_{|A|} = \mathrm{diag}(|A|\mathbf{1})$ is the absolute-value degree matrix.  This formulation:

- **Penalises dissimilar** decoder weight rows for *positively* correlated gene pairs.
- **Penalises similar** decoder weight rows for *negatively* correlated gene pairs.

### Downloading the ARCHS4 correlation matrix

```python
from utils.graph_utils import fetch_archs4_correlation

# Download once (~6 GB); subsequent calls return the cached path instantly
pkl_path = fetch_archs4_correlation(
    dest_path="~/.cache/nmfvae/human_correlation_v2.4.pkl"  # optional; default location
)
```

The file is fetched from `https://s3.amazonaws.com/mssm-data/human_correlation_v2.4.pkl` and
cached locally at `~/.cache/nmfvae/human_correlation_v2.4.pkl` by default.  Pass
`force=True` to force a re-download.

### Python API

```python
from utils.graph_utils import build_correlation_laplacian, save_laplacian
from model.vae import fit_model, get_gene_programs

# Build signed Laplacian from pre-computed correlations
L, matched = build_correlation_laplacian(
    genes                = gene_names,          # list matching count matrix columns
    pkl_path             = "human_correlation_v2.4.pkl",
    correlation_threshold= 0.5,                 # discard |corr| < 0.5
    normalized           = True,
    weak_prior_diagonal  = 0.1,                 # for unmatched (LOC) genes
    convert_ncbi         = True,                # auto-convert to NCBI symbols
    species_id           = 9606,                # 9606 = human
)
print(f"{sum(matched)}/{len(gene_names)} genes matched the correlation matrix")

model = fit_model(X, config={
    "latent_dim"      : 20,
    "epochs"          : 200,
    "lambda_graph"    : "moderate",
    "graph_laplacian" : L,
})

# Save tuned Laplacian and W matrix to disk
W = get_gene_programs(model=model)
save_laplacian(L, "results/corr_prior", W=W, gene_names=gene_names)
# Writes: results/corr_prior_laplacian.npy
#         results/corr_prior_W.csv
```

### Gene name conversion

Gene names in single-cell data may not match the NCBI reference exactly (e.g. aliased symbols, species-specific `LOC*` identifiers).  `convert_to_ncbi_gene_names` uses the [MyGene.info](https://mygene.info) REST API to map names to official NCBI symbols:

```python
from utils.graph_utils import convert_to_ncbi_gene_names

ncbi_names, matched = convert_to_ncbi_gene_names(gene_names, species_id=9606)
unmatched = [g for g, m in zip(gene_names, matched) if not m]
print(f"Unmatched genes (will use weak prior): {unmatched}")
```

Unmatched genes receive a small diagonal value (`weak_prior_diagonal`) in the Laplacian so they are weakly regularised but primarily data-driven.

### CLI

```bash
# Auto-download ARCHS4 correlation matrix and train in one step
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --genes-file data/gene_names.txt \
    --lambda-graph moderate \
    --fetch-archs4 \
    --correlation-threshold 0.5 \
    --weak-prior-diagonal 0.1 \
    --save-laplacian results/my_laplacian

# Train with a manually-downloaded correlation pkl
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --genes-file data/gene_names.txt \
    --lambda-graph moderate \
    --correlation-pkl human_correlation_v2.4.pkl \
    --correlation-threshold 0.5 \
    --weak-prior-diagonal 0.1 \
    --save-laplacian results/my_laplacian

# Combine with STRING network (70% correlation, 30% STRING)
python scripts/train.py \
    --input data/processed.h5ad \
    --output results/ \
    --genes-file data/gene_names.txt \
    --lambda-graph moderate \
    --fetch-archs4 \
    --use-string-graph \
    --use-hybrid-graph \
    --alpha-graph-mix 0.7 \
    --save-laplacian results/hybrid_laplacian

# Disable NCBI name conversion (use gene names as-is)
python scripts/train.py \
    --correlation-pkl human_correlation_v2.4.pkl \
    --no-ncbi-convert \
    ...
```

## Tests

```bash
pytest tests/ -v
```

Unit tests cover:
- WeibullDistribution (rsample, log_prob)
- Encoder / Decoder forward passes
- Full model forward + ELBO
- End-to-end training (5 epochs on synthetic data)
- Data utilities
- High-level API (fit_model, transform, get_gene_programs)
- Graph utilities: `resolve_lambda` (presets + float + invalid), Laplacian math (unnormalized, normalized, isolated nodes)
- Laplacian penalty properties (zero for constant genes, positive for divergent)
- Training with graph Laplacian penalty enabled
- `set_graph_laplacian`, co-expression Laplacian, hybrid Laplacian, API with graph config
- **Signed Laplacian** (unnormalized, normalized, isolated nodes)
- **Correlation Laplacian** (`build_correlation_laplacian`: matched genes, unmatched genes with weak prior, thresholding, missing file)
- **`save_laplacian`** (with and without W matrix)
- **`fetch_archs4_correlation`** (cache hit, streaming download, force re-download, missing requests package)
- **End-to-end training** with a signed graph Laplacian

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:

1. **test** job – installs CPU PyTorch, runs pytest (fast, no Docker)
2. **docker** job – delegates to `GWMcElfresh/dockerDependencies/docker-cache.yml`:
   - Pulls the current month's base-deps image from GHCR (built once/month by `monthly-base.yml`)
   - Hashes `requirements.txt` → pulls or builds the incremental deps image
   - Builds the GPU-capable runtime image and runs tests inside it
   - Pushes the runtime to GHCR on `main`/`master` merges

`.github/workflows/monthly-base.yml` rebuilds the base dependency image on the 1st of each month (or on manual trigger) via `GWMcElfresh/dockerDependencies/build-base-image.yml`.

## References

- Lopez et al. (2018) – scVI
- Levitin et al. (2019) – scHPF
- Carbonetto et al. (2022) – LDVAE
- Zhang et al. (2020) – ZINB-WaVE
