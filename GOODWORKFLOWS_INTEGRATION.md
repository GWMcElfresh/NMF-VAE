# NMF-VAE — GoodWorkflows Integration Summary

> **Target audience**: A downstream LLM or developer implementing NMF-VAE as a Nextflow DSL2 module in the GoodWorkflows containerized pipeline system.

---

## 1. What NMF-VAE Does (for a Pipeline Operator)

NMF-VAE is a GPU-accelerated deep learning model that ingests a **single-cell RNA-seq count matrix** (cells × genes) and produces:

| Output | Shape | Meaning |
|---|---|---|
| `Z` (latent matrix) | cells × latent_dim | Non-negative, sparse representation of each cell as a combination of gene programs |
| `W` (decoder weights) | genes × latent_dim | Non-negative gene loadings — each column is a "gene program" |
| Loss curve | 1D | Per-epoch ELBO for diagnostics |

Optionally, a **graph Laplacian penalty** can be applied to W using STRING protein interaction networks, gene co-expression, or pre-computed correlation matrices (e.g., ARCHS4 ~6GB).

---

## 2. How It Fits in the GoodWorkflows DAG

GoodWorkflows currently has this integration pipeline:

```
ch_samples ──► [INGEST/INGEST_URL] ──► ch_ingested_rds
                                              │
                                              ▼
                                      [EXPORT_COUNTS] ──► ch_all_count_dirs
                                              │
                                              ▼
                                      [GENE_HARMONIZE] ──► harmonized counts
                                              │
                                              ▼
                                      [SCMODAL_INTEGRATE] (GPU)
```

**NMF-VAE adds a new, parallel GPU branch** that can run independently or alongside scMODAL:

```
ch_all_count_dirs (from EXPORT_COUNTS)
        │
        ├──► [GENE_HARMONIZE] ──► [SCMODAL_INTEGRATE]
        │
        └──► [NMFVAE_FACTORIZE] (GPU, new module)
                 │
                 ├── latent_Z.csv     → downstream clustering/UMAP
                 ├── decoder_W.csv    → gene program analysis
                 └── loss.png         → QC dashboard
```

Or, post-harmonization:

```
[GENE_HARMONIZE] ──► [NMFVAE_FACTORIZE]
```

---

## 3. Input/Output Contract

### Input (from upstream Nextflow channels)

```
tuple val(meta), path(count_matrix)
```
Where:
- `meta` = Map with at least `meta.id` (sample identifier) and optionally `meta.species`
- `count_matrix` = Path to a **single .h5ad file** (AnnData format) with:
  - `.X` = dense or sparse count matrix, shape (cells, genes)
  - `.var_names` = gene identifiers (optional but needed for STRING/correlation graph)

**Alternative**: If the upstream module emits a **directory of count files** (as `EXPORT_COUNTS` does with `counts_dir`), the module must find the `.h5ad` file within:
```bash
# Inside the process script:
H5AD_FILE=$(find ${counts_dir} -name "*.h5ad" | head -1)
```

### Optional Input (for graph regularization)

```
path gene_names_file
```
Plain text file with one gene symbol per line, matching the columns of the count matrix in order.

### Output

```
tuple val(meta), path("nmfvae_output/latent_Z.csv")
tuple val(meta), path("nmfvae_output/decoder_W.csv")
path "nmfvae_output/loss.png"
path "nmfvae_output/loss_history.csv"
```

---

## 4. Container Requirements

### Docker Image
```
ghcr.io/gwmcelfresh/nmfvae:latest
```
Built from the `Dockerfile` in this repo using multi-stage (`deps` → `runtime`) with:
- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- PyTorch: `torch>=2.0.0` with CUDA 12.1 support
- All deps from `requirements.txt`
- Source code installed via `pip install -e .`

**GPU requirement**: The process label MUST be `process_gpu`.

### Singularity SIF
```
nmfvae.sif
```
Built from `singularity.def`. **NOTE**: The current `singularity.def` is missing critical dependencies (see BUG_REPORT.md, BUG-5). The recommended fixed version:

```singularity
Bootstrap: docker
From: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev gcc g++ \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir \
        numpy scipy pandas \
        anndata scanpy \
        matplotlib seaborn umap-learn \
        pytest requests mygene scikit-learn
    pip install --no-cache-dir -e /app

%runscript
    exec python "$@"
```

---

## 5. Nextflow DSL2 Module Definition

```groovy
// modules/local/nmfvae_factorize/main.nf

process NMFVAE_FACTORIZE {
    tag "$meta.id"
    label 'process_gpu'

    container "${params.nmfvae_container ?: 'ghcr.io/gwmcelfresh/nmfvae:latest'}"
    publishDir "${params.outdir}/nmfvae/${meta.id}", mode: 'copy'

    input:
    tuple val(meta), path(count_matrix)   // .h5ad or directory containing .h5ad
    path gene_names                        // optional: text file, one gene per line

    output:
    tuple val(meta), path("nmfvae_output/latent_Z.csv"),  emit: latent
    tuple val(meta), path("nmfvae_output/decoder_W.csv"), emit: gene_programs
    path "nmfvae_output/loss.png",                         emit: loss_plot
    path "nmfvae_output/loss_history.csv",                 emit: loss_history

    script:
    // Resolve the count matrix if a directory is passed
    def input_arg = count_matrix
    if (count_matrix.isDirectory()) {
        input_arg = "${count_matrix}/*.h5ad"
    }

    // Build graph Laplacian CLI args if gene names and lambda are configured
    def graph_args = " --lambda-graph ${params.nmfvae_lambda_graph ?: 'none'}"
    if (gene_names && params.nmfvae_lambda_graph != "none") {
        graph_args += """
            --genes-file ${gene_names} \\
            --use-string-graph
        """
    }
    if (params.nmfvae_fetch_archs4) {
        graph_args += " --fetch-archs4"
    }

    """
    mkdir -p nmfvae_output
    python /app/scripts/train.py \\
        --input ${input_arg} \\
        --output nmfvae_output/ \\
        --latent-dim ${params.nmfvae_latent_dim ?: 20} \\
        --epochs ${params.nmfvae_epochs ?: 200} \\
        --batch-size ${params.nmfvae_batch_size ?: 256} \\
        --lr ${params.nmfvae_lr ?: 1e-3} \\
        ${graph_args}
    """

    stub:
    """
    mkdir -p nmfvae_output
    touch nmfvae_output/latent_Z.csv
    touch nmfvae_output/decoder_W.csv
    touch nmfvae_output/loss.png
    touch nmfvae_output/loss_history.csv
    """
}
```

---

## 6. Suggested Nextflow Configuration

```groovy
// configs/modules/nmfvae.config

params {
    // NMF-VAE model parameters
    nmfvae_container    = 'ghcr.io/gwmcelfresh/nmfvae:latest'
    nmfvae_latent_dim   = 20          // number of gene programs
    nmfvae_epochs       = 200         // training epochs
    nmfvae_batch_size   = 256         // mini-batch size
    nmfvae_lr           = 0.001       // learning rate (1e-3)
    nmfvae_lambda_graph = 'none'      // "none", "weak", "moderate", "strong", or float
    nmfvae_fetch_archs4 = false       // download 6GB ARCHS4 correlation matrix?
    nmfvae_use_string   = false       // query STRING API for PPI network?
    nmfvae_confidence_threshold = 0.7 // STRING edge confidence cutoff
    nmfvae_species_id   = 9606        // 9606=human, 10090=mouse
}

process {
    withLabel: process_gpu {
        // GPU resource requests (adjust for your cluster)
        queue = 'gpu'
        clusterOptions = '--gpus=1'
        // or for SLURM:
        // clusterOptions = '--partition=gpu --gres=gpu:1 --mem=32G --time=24:00:00'
    }
}
```

---

## 7. Workflow Composition Example

```groovy
// subworkflows/local/nmfvae_analysis/main.nf

include { NMFVAE_FACTORIZE } from '../../../modules/local/nmfvae_factorize/main'

workflow NMFVAE_ANALYSIS {
    take:
    ch_count_dirs    // channel: tuple val(meta), path(counts_dir)
    ch_gene_names    // channel: path(gene_names.txt), optional

    main:
    NMFVAE_FACTORIZE(ch_count_dirs, ch_gene_names)

    emit:
    latent        = NMFVAE_FACTORIZE.out.latent
    gene_programs = NMFVAE_FACTORIZE.out.gene_programs
    loss_plot     = NMFVAE_FACTORIZE.out.loss_plot
    loss_history  = NMFVAE_FACTORIZE.out.loss_history
}
```

---

## 8. Python API Integration (Alternative to CLI)

If the GoodWorkflows container needs to call NMF-VAE directly from Python (e.g., a wrapper script instead of the CLI), the API is:

```python
import numpy as np
from model.vae import fit_model, transform, get_gene_programs, export_results

# Load data (handles .h5ad, .mtx, .csv, .tsv, .npz)
from utils.data_utils import load_data
X = load_data("input.h5ad")  # → np.ndarray (cells, genes), float32

# Optional: build graph Laplacian
from utils.graph_utils import build_string_laplacian, resolve_lambda
L = build_string_laplacian(
    genes=gene_names_list,     # must match X columns
    confidence_threshold=0.7,
    species_id=9606,           # human
    normalized=True,
)
lambda_val = resolve_lambda("moderate")  # → 0.1

# Train
model = fit_model(X, config={
    "latent_dim": 20,
    "epochs": 200,
    "batch_size": 256,
    "lr": 1e-3,
    "lambda_graph": lambda_val,
    "graph_laplacian": L,   # torch.Tensor (genes, genes)
})

# Extract results
Z = transform(X, model=model)        # (cells, 20)
W = get_gene_programs(model=model)   # (genes, 20)

# Save
export_results("nmfvae_output/", count_matrix=X, model=model)
# Writes: latent_Z.csv, decoder_W.csv, loss_history.csv
```

---

## 9. Key Files Reference

| File | Path in container | Purpose |
|---|---|---|
| Train script | `/app/scripts/train.py` | CLI entry point for training |
| Preprocess script | `/app/scripts/preprocess.py` | CLI entry point for QC/normalization |
| Model class | `/app/model/vae.py` | `NMFVAE` nn.Module |
| Data utils | `/app/utils/data_utils.py` | `load_data()`, `create_dataloader()`, `write_outputs()` |
| Graph utils | `/app/utils/graph_utils.py` | All Laplacian builders, STRING fetch, ARCHS4 download |
| Plot utils | `/app/utils/plot_utils.py` | UMAP/PCA, ELBO curve, gene loading heatmaps |
| Tests | `/app/tests/test_model.py` | pytest test suite |
| Requirements | `/app/requirements.txt` | All pip dependencies |

---

## 10. Important Notes for the Implementing LLM

### Execution Model
- NMF-VAE uses **PyTorch** with automatic GPU detection (`torch.cuda.is_available()`)
- The `NMFVAE.fit()` method handles the training loop internally (no external training orchestrator needed)
- One `.h5ad` file → one trained model → one set of outputs

### Resource Estimates
- **GPU memory**: ~2-4 GB for typical single-cell datasets (10K-50K cells × 5K-20K genes)
- **Training time**: ~5-10 minutes for 200 epochs on a single GPU (V100/A100) with 10K cells
- **CPU-only**: ~20-30 minutes for same dataset (automatic fallback)
- **ARCHS4 download**: ~6 GB, only needed once per cache lifetime

### Error Handling
- If STRING API is unreachable, `build_string_laplacian` raises `RuntimeError` — wrap in try/catch
- If `mygene` or `requests` are missing, functions raise `ImportError` with install instructions
- The model returns `NaN` losses if the learning rate is too high — monitor `loss_history`
- UMAP failures silently fall back to PCA (no error propagation)

### Stub Support
The module includes a `stub:` block. For CI/test profiles, this generates empty output files without running the model.

### Parameter Overlap with Existing Modules
- `params.species_id` — already used by `GENE_HARMONIZE`. NMF-VAE's species ID should match.
- `params.outdir` — standard output root; NMF-VAE writes to `${params.outdir}/nmfvae/`