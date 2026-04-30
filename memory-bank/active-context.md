# NMF-VAE — Active Context

## Current State
- **Version**: 0.1.0
- **Date**: April 30, 2026
- **Last commit**: `765ae536843fd185fb08937dde23b3df2bcc3cf2`
- **Remote**: `origin: https://github.com/GWMcElfresh/NMF-VAE.git`

## What Just Happened
A comprehensive code review was performed on the entire repository. All source files in `model/`, `utils/`, `scripts/`, and `tests/` were read and analyzed. The GoodWorkflows MCP server was queried to understand the Nextflow DSL2 integration context.

## Known Bugs (See BUG_REPORT.md for full details)

### Critical (8 bugs)
1. **`laplacian_penalty` sparse tensor incompatibility** — `torch.sparse.mm(L, W)` may silently fail on older PyTorch if L is not CSR format
2. **`train.py` passes raw string `lambda_graph` to constructor** — re-resolves already-resolved value
3. **`transform()` silently returns mean during training mode** — unexpected behavior if not in eval mode
4. **`preprocess.py` `.clip(1)` truncation** — minor but semantically unclear
5. **`singularity.def` missing dependencies** — `requests`, `mygene`, `scikit-learn` not installed
6. **`plot_utils.py` sets global matplotlib backend at import time** — breaks Jupyter notebooks
7. **Null `_graph_laplacian` with `lambda_graph > 0` is silently ignored** — no user warning
8. **`test_model.py` monkeypatches `requests.get` globally** — can affect other tests

### Minor (4 bugs)
9. **No console_scripts entry points** — no `nmfvae-train` command
10. **`requirements.txt` has `torch>=2.0.0` but Dockerfile installs torch separately** — version conflict risk
11. **No `.github/workflows/ci.yml` found on disk** — README references it but file absent
12. **`_pca()` imports `sklearn` at call time** — late import pattern but inconsistent with other imports

## Pending Tasks
- Fix the 12 bugs listed above (see BUG_REPORT.md for recommended fixes)
- Add missing `.github/workflows/ci.yml` or confirm it exists in another branch
- Add `console_scripts` entry points to `setup.py`
- Improve `singularity.def` to match `Dockerfile` dependencies exactly
- Consider making `matplotlib.use("Agg")` conditional (e.g., only in non-interactive environments)

## Recent Decisions
- **Memory bank created** — This is the first memory bank for this repo. All context has been extracted from code analysis.
- **GoodWorkflows integration summary written** — A companion document (`GOODWORKFLOWS_INTEGRATION.md`) provides full examples for another LLM to implement NMF-VAE as a Nextflow DSL2 module.

## Active Constraints
- Python 3.8+ required
- CUDA 12.1 for GPU (CPU fallback works)
- Graph Laplacian must be `torch.Tensor` of shape `(n_genes, n_genes)`
- STRING API requires internet at runtime
- ARCHS4 download is ~6GB (cache at `~/.cache/nmfvae/`)
- Tests use 50×100 synthetic data only (no real data in test suite)

## Next Steps (for future contributors)
1. Fix bugs from BUG_REPORT.md
2. Implement NMF-VAE as a Nextflow DSL2 module using the patterns in GOODWORKFLOWS_INTEGRATION.md
3. Add batch correction capability (major feature)
4. Add multi-GPU support via `DataParallel` or `DistributedDataParallel`
5. Containerize and push to GHCR for GoodWorkflows consumption