# NMF-VAE — System Patterns

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Module-Level API                       │
│  fit_model() → transform() → get_gene_programs()         │
│  export_results()  plot_latent_space()                   │
│         │ (uses _global_model singleton)                  │
├──────────────────────────────────────────────────────────┤
│                    NMFVAE (nn.Module)                     │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐ │
│  │WeibullEncoder│ → │reparameterize│ → │  NNDecoder    │ │
│  │  (FC+BN+LR) │   │(WeibullDist) │   │ (softplus(W)@z)│ │
│  └─────────────┘   └──────────────┘   └───────────────┘ │
│                                              │           │
│         ┌────────────────────────────────────┘           │
│         ▼                                                │
│  ┌──────────────────┐                                    │
│  │ _graph_laplacian │ (registered buffer, optional)      │
│  │ laplacian_penalty│ = Tr(Wᵀ L W)                       │
│  └──────────────────┘                                    │
├──────────────────────────────────────────────────────────┤
│                    Utilities Layer                        │
│  data_utils  │  graph_utils  │  plot_utils               │
│  (load/save) │ (Laplacians)  │ (UMAP/PCA/loss plots)     │
└──────────────────────────────────────────────────────────┘
```

## Key Design Patterns

### 1. Singleton Module-Level API
**Pattern**: A module-level global variable `_global_model` caches the last-trained model. Functions `transform()`, `get_gene_programs()`, `export_results()` use it as default.

**Location**: `model/vae.py`, lines 383–576

**Why**: Enables scikit-learn-style API where the user doesn't need to pass the model object around. After `fit_model()`, all other functions "just work."

**Trade-off**: Not thread-safe. Not suitable for multi-model workflows. The `model` parameter override is available for explicit control.

```python
# Global state
_global_model: Optional[NMFVAE] = None

def fit_model(count_matrix, config=None) -> NMFVAE:
    global _global_model
    # ... train ...
    _global_model = model
    return model

def transform(count_matrix, model=None) -> np.ndarray:
    if model is None:
        model = _global_model  # fallback to singleton
    # ...
```

### 2. Buffer-Registered Laplacian
**Pattern**: The graph Laplacian tensor is registered as a PyTorch buffer (not a parameter). This ensures it moves to the correct device with `.to(device)` and is included in `state_dict()`.

**Location**: `model/vae.py`, lines 88–101

```python
if graph_laplacian is not None:
    self.register_buffer("_graph_laplacian", graph_laplacian.float())
else:
    self.register_buffer("_graph_laplacian", None)
```

**Why**: Buffers are part of model state (for serialization/device movement) but don't receive gradients.

### 3. Reparameterization Trick with Mode Switch
**Pattern**: During training, `reparameterize()` returns stochastic samples from Weibull. During eval, it returns the deterministic mean.

**Location**: `model/vae.py`, lines 114–127

```python
def reparameterize(self, k, lam):
    if self.training:
        q = WeibullDistribution(k, lam)
        return q.rsample()
    else:
        return lam * torch.exp(torch.lgamma(1.0 + 1.0 / k))  # mean
```

### 4. Softplus-Constrained Parameters
**Pattern**: All positivity constraints are enforced via `softplus(x) + epsilon` rather than clamping or absolute value.

Locations:
- `model/encoder.py` line 73-74: `k = F.softplus(self.fc_k(h)) + 1e-4`
- `model/decoder.py` line 62: `W = F.softplus(self.W_raw)`
- `model/decoder.py` line 72: `mu = F.softplus(mu) + 1e-8`

**Why**: Softplus is smooth (differentiable everywhere), unlike `ReLU` which has a dead zone, or `exp` which can explode.

### 5. Monte Carlo KL Divergence
**Pattern**: KL(Weibull || Gamma) has no closed form, so it's estimated via Monte Carlo sampling.

**Location**: `model/distributions.py`, lines 90–121

```python
def kl_weibull_gamma(k, lam, alpha, beta, n_samples=10):
    q = WeibullDistribution(k, lam)
    kl_sum = torch.zeros_like(k)
    for _ in range(n_samples):
        z = q.rsample()
        kl_sum += (q.log_prob(z) - gamma_log_prob(z, alpha, beta))
    return kl_sum / n_samples
```

### 6. Library-Size Scaling in Forward Pass
**Pattern**: The forward pass computes per-cell library sizes and normalizes them to mean 1.0. The decoder output is then scaled by library size.

**Location**: `model/vae.py`, lines 150–153

```python
library_size = x.sum(dim=1, keepdim=True).clamp(min=1.0)
library_size = library_size / library_size.mean()
mu, theta = self.decode(z, library_size)
```

**Why**: Keeps decoder weights on a common scale across cells with vastly different total counts.

### 7. Linear KL Warmup
**Pattern**: During training, the KL weight linearly increases from 0 to `kl_weight` over `kl_warmup_epochs`.

**Location**: `model/vae.py`, lines 293–296

```python
if kl_warmup_epochs > 0:
    warmup_weight = min(1.0, (epoch + 1) / kl_warmup_epochs) * kl_weight
```

**Why**: Prevents the KL term from dominating early training when the encoder produces poor posterior approximations.

### 8. CLI with sys.path Manipulation
**Pattern**: Scripts in `scripts/` insert the repo root into `sys.path` to allow running without `pip install -e .`.

**Location**: `scripts/train.py` line 11, `scripts/preprocess.py` line 10

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Data Flow

```
Input File (.h5ad/.mtx/.csv/.npz)
    │
    ▼ load_data()
numpy array (cells × genes)
    │
    ▼ create_dataloader() or inline TensorDataset
DataLoader → batches of (batch × genes)
    │
    ▼ NMFVAE.forward()
mu (batch × genes), theta (genes,), k (batch × latent), lam (batch × latent)
    │
    ▼ NMFVAE.elbo_loss()
    ├── recon_loss: -NB_log_likelihood(x, mu, theta)
    ├── kl_loss: kl_weibull_gamma(k, lam, α, β)
    └── lap_penalty: λ·Tr(Wᵀ L W) [if λ > 0 and L registered]
    │
    ▼ optimizer.step()
    │
    ▼ (after training) NMFVAE.transform()
Z: numpy array (cells × latent)
    │
    ▼ NMFVAE.get_gene_programs()
W: numpy array (genes × latent)
    │
    ▼ export_results() / write_outputs()
CSV files: latent_Z.csv, decoder_W.csv, loss_history.csv
```

## Error Handling Patterns
- `AssertionError` style: tests use plain `assert` statements (no custom exceptions)
- `ValueError` for invalid parameters (negative lambda, unknown preset)
- `FileNotFoundError` for missing correlation pkl files
- `ImportError` with helpful `pip install` messages for optional dependencies (requests, mygene)
- `RuntimeError` for API failures (STRING API, ARCHS4 download)
- Silent fallback: PCA if UMAP fails, pass if metadata concat fails
- `warnings.warn()` for unmatched genes in correlation matrix