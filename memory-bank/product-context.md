# NMF-VAE — Product Context

## Why This Exists

Single-cell RNA-seq produces massive count matrices (thousands to millions of cells × tens of thousands of genes). Researchers need to:
1. **Reduce dimensionality** while preserving biological signal
2. **Discover interpretable gene programs** — groups of co-expressed genes that define cell states
3. **Integrate prior biological knowledge** (protein interaction networks, gene co-expression databases)

NMF-VAE addresses all three by providing a deep learning factor model that:
- Produces **non-negative latent factors** interpretable as gene program usage per cell
- Learns **non-negative decoder weights** interpretable as gene loadings per program
- Supports **graph Laplacian regularization** from STRING, co-expression, or correlation matrices

### Comparison to Alternatives

| Method | Interpretable | Non-negative | Graph Prior | Deep/Nonlinear |
|---|---|---|---|---|
| PCA | No | No | No | No |
| NMF | Yes | Yes | Yes (graph-NMF) | No |
| scVI | No (deep) | No | No | Yes |
| scHPF | Yes | Yes | No | No |
| LDVAE | Partial | Yes | No | Yes |
| **NMF-VAE** | **Yes** | **Yes** | **Yes** | **Yes** |

## Target Users
- **Bioinformatics researchers** studying gene programs in single-cell data
- **Computational biologists** integrating network prior knowledge with data-driven factorization
- **HPC pipeline operators** needing reproducible GPU-accelerated factor analysis

## Key Capabilities

### 1. Sparse Non-Negative Factorization
The Gamma prior with small shape (α=1, β=1) encourages sparsity — most latent dimensions are near zero for each cell, making each cell a sparse combination of a few gene programs.

### 2. Graph-Regularized Gene Programs
The optional graph Laplacian penalty `Tr(Wᵀ L W)` penalizes the decoder weight matrix so that:
- Genes connected in STRING have **similar** gene program loadings
- Positively correlated genes have **similar** decoder weight rows
- Negatively correlated genes have **dissimilar** decoder weight rows

### 3. Multiple Graph Sources
- **STRING protein-protein interaction** (REST API, requires internet)
- **Co-expression kNN graph** (computed from input data)
- **Hybrid** (linear interpolation of STRING + co-expression)
- **ARCHS4 correlation matrix** (downloadable ~6GB pickle, signed Laplacian support)

### 4. Flexible I/O
Input formats: `.h5ad` (AnnData), `.mtx` (Matrix Market), `.csv`/`.tsv`, `.npz` (scipy sparse)
Output: CSV tables of latent Z, decoder W, loss history; PNG plots; optional NPY Laplacian

## Limitations
- Linear decoder only (no nonlinear interactions between latent factors and genes)
- No built-in batch correction (unlike scVI)
- STRING API requires internet at runtime (no offline cache)
- ARCHS4 download is ~6GB (impractical for many HPC environments)
- No multi-GPU support