# NMF-VAE — Project Brief

## Mission
NMF-VAE is a **Variational Autoencoder for single-cell RNA-seq data** that produces non-negative, sparse, interpretable latent representations. It approximates Gamma-Poisson (probabilistic NMF) factorization using deep learning.

## Core Identity
- **What**: A VAE where the latent space behaves like NMF (non-negative factors, sparse activations, interpretable decoder weights)
- **Domain**: Single-cell transcriptomics (scRNA-seq count data)
- **Key innovation**: Weibull approximate posterior with reparameterization trick for Gamma prior, plus optional graph Laplacian regularization on the decoder weight matrix

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Weibull→Gamma KL | Weibull is reparameterizable and approximates Gamma; enables gradient-based training |
| Non-negative decoder (`softplus(W_raw)`) | Makes W interpretable as gene programs (like NMF/Poisson-NMF) |
| Graph Laplacian penalty (`Tr(Wᵀ L W)`) | Biologically-informed regularization using STRING, co-expression, or hybrid networks |
| Signed Laplacian for correlations | Handles both positive and negative gene-gene correlations |
| Module-level singleton API | scikit-learn-style `fit_model()`, `transform()`, `get_gene_programs()` for simplicity |
| GPU-capable (CUDA 12.1) with CPU fallback | Broad deployment: local workstations, HPC, cloud |

## Architecture (High-Level)

```
Input: Count matrix (cells × genes)
  │
  ▼
WeibullEncoder ──► k, λ (shape, scale parameters)
  │
  ▼
WeibullDistribution.rsample() ──► z (non-negative latent factors)
  │
  ▼
NNDecoder ──► μ = softplus(W) @ z * lib_size, θ = exp(log_θ)
  │
  ▼
Loss: -NB_log_likelihood(x|μ,θ) + KL(Weibull||Gamma) + λ·Tr(Wᵀ L W)
```

## Project Status
- **Version**: 0.1.0
- **Testing**: pytest with synthetic data (50 cells × 100 genes)
- **CI/CD**: GitHub Actions with multi-stage Docker caching (GWMcElfresh/dockerDependencies)
- **License**: MIT (see LICENSE file)