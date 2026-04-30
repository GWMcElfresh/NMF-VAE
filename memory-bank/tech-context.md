# NMF-VAE — Technical Context

## Environment

| Aspect | Details |
|---|---|
| **Language** | Python 3.8+ |
| **Framework** | PyTorch 2.0+ |
| **GPU** | CUDA 12.1 (cuDNN 8), CPU fallback automatic |
| **Package manager** | pip + setuptools |
| **Test runner** | pytest 7.0+ |
| **OS** | Linux (Docker/Singularity), macOS (local dev), Windows (WSL) |

## Dependencies

### Core (required for all functionality)
```
torch>=2.0.0         # Deep learning framework
numpy>=1.21.0        # Array operations
scipy>=1.7.0         # Sparse matrix I/O
pandas>=1.3.0        # DataFrame I/O
```

### Single-cell (required for data I/O)
```
anndata>=0.8.0       # AnnData container (.h5ad)
scanpy>=1.9.0        # Single-cell analysis toolkit
```

### Visualization (required for plots)
```
matplotlib>=3.4.0    # Base plotting
seaborn>=0.11.0      # Heatmaps
umap-learn>=0.5.0    # UMAP dimensionality reduction
```

### Testing (required for development)
```
pytest>=7.0.0        # Test framework
```

### Optional (required for graph Laplacian features)
```
requests>=2.28.0     # STRING API, ARCHS4 download
mygene>=3.2.2        # NCBI gene name conversion
```

### Implicit (brought in by above, not pinned)
- `scikit-learn` (via `PCA` in plot_utils, implicit dependency from umap-learn)

## Container Images

### Docker
- **Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Multi-stage**: `deps` (heavy pip installs) → `runtime` (source code + editable install)
- **Build args**: `BASE_IMAGE`, `SKIP_BASE_DEPS`, `DEPS_IMAGE` (for CI caching protocol)
- **Default CMD**: `pytest tests/`

### Singularity
- **Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **%post**: Installs Python, CUDA PyTorch, and hardcoded pip packages
- **%runscript**: `exec python "$@"`

### CI/CD
- GitHub Actions with `GWMcElfresh/dockerDependencies`:
  - `monthly-base.yml`: Rebuilds base deps image on 1st of each month
  - `docker-cache.yml`: Hashes `requirements.txt` for incremental layer caching
  - Pushes to GHCR on main/master merges

## File Format Support

### Input Formats
| Format | Loader | Notes |
|---|---|---|
| `.h5ad` | `anndata.read_h5ad()` | Dense and sparse X supported |
| `.mtx` | `scipy.io.mmread()` | Matrix Market format |
| `.csv` | `pandas.read_csv(index_col=0)` | Comma-delimited |
| `.tsv` | `pandas.read_csv(sep='\t', index_col=0)` | Tab-delimited |
| `.npz` | `scipy.sparse.load_npz()` | SciPy sparse |

### Output Formats
| File | Format | Content |
|---|---|---|
| `latent_Z.csv` | CSV | cells × latent_dim |
| `decoder_W.csv` | CSV | genes × latent_dim |
| `loss_history.csv` | CSV | epoch, loss |
| `loss.png` | PNG | Training curve plot |
| `*_laplacian.npy` | NPY | Graph Laplacian matrix |

### Graph Input Formats
| Source | Format | Size |
|---|---|---|
| STRING API | REST JSON | Lightweight |
| Co-expression | Computed from input data | O(G²/2) |
| ARCHS4 correlation | Pickled DataFrame | ~6 GB |
| Custom correlation | Pickled DataFrame | Variable |

## Model Hyperparameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `latent_dim` | 10 | 2–100 | Number of gene programs |
| `hidden_dims` | [256, 128] | List[int] | Encoder hidden layer sizes |
| `gamma_alpha` | 1.0 | >0 | Gamma prior shape |
| `gamma_beta` | 1.0 | >0 | Gamma prior rate |
| `use_nb` | True | bool | NB vs Poisson likelihood |
| `epochs` | 100 | 1–1000 | Training epochs |
| `batch_size` | 256 | 16–1024 | Mini-batch size |
| `lr` | 1e-3 | 1e-5–1e-2 | Learning rate |
| `kl_weight` | 1.0 | 0–10 | Final KL weight |
| `kl_warmup_epochs` | 10 | 0–100 | KL annealing duration |
| `lambda_graph` | 0.0 | 0–1.0 | Graph Laplacian strength |

### Lambda Presets
| Name | Value |
|---|---|
| `"none"` | 0.0 |
| `"weak"` | 0.01 |
| `"moderate"` | 0.10 |
| `"strong"` | 1.00 |

## Supported Species (STRING/NCBI)
| Species ID | Name |
|---|---|
| 9606 | Human |
| 10090 | Mouse |
| 10116 | Rat |
| 7955 | Zebrafish |

## Module Inventory

```
model/
├── __init__.py          # Re-exports: NMFVAE, fit_model, transform, get_gene_programs, encoder, decoder, distributions
├── vae.py               # 576 lines — NMFVAE class + module-level API
├── encoder.py           #  76 lines — WeibullEncoder
├── decoder.py           #  81 lines — NNDecoder
└── distributions.py     # 176 lines — WeibullDistribution, KL, NB log-likelihood

utils/
├── __init__.py          # Empty (package marker)
├── data_utils.py        # 139 lines — load_data, to_tensor, create_dataloader, write_outputs
├── graph_utils.py       # 761 lines — STRING, co-expression, correlation Laplacians, NCBI conversion, fetch_archs4, save_laplacian
└── plot_utils.py        # 152 lines — UMAP/PCA, ELBO curve, gene loading heatmaps

scripts/
├── train.py             # 349 lines — CLI training with full graph Laplacian support
└── preprocess.py        #  69 lines — CLI preprocessing (QC filter, normalize, log1p)

tests/
├── __init__.py          # Empty (package marker)
└── test_model.py        # 800 lines — 25+ tests covering distributions, encoder, decoder, VAE, API, all graph types

notebooks/
└── graph_laplacian_tutorial.ipynb  # Interactive tutorial

data/
└── .gitkeep             # Placeholder for input data (gitignored)