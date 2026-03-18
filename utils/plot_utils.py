"""
Plotting utilities for NMF-VAE.
"""

from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_latent_space(
    Z: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None,
    method: str = "umap",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize latent space using UMAP or PCA.

    Args:
        Z: Latent matrix (cells × latent).
        metadata: Optional DataFrame with cell annotations.
        color_by: Column in metadata to use for colouring.
        method: 'umap' or 'pca'.
        save_path: If given, save figure to this path.
    """
    embedding = _embed(Z, method=method)

    colors = None
    if metadata is not None and color_by is not None:
        if color_by in metadata.columns:
            colors = metadata[color_by].values

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap="tab20",
        s=5,
        alpha=0.7,
    )
    if colors is not None:
        plt.colorbar(scatter, ax=ax, label=color_by)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"Latent space ({method.upper()})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_elbo(
    loss_history: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training ELBO / loss curve.

    Args:
        loss_history: Per-epoch losses.
        save_path: If given, save figure here.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (-ELBO)")
    ax.set_title("Training loss")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gene_loadings(
    W: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of top gene loadings per latent factor.

    Args:
        W: Weight matrix (genes × latent).
        gene_names: Optional list of gene names.
        top_n: Number of top genes per factor to show.
        save_path: If given, save figure here.
    """
    n_genes, n_factors = W.shape

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Select union of top-n genes across all factors
    top_idx = set()
    for j in range(n_factors):
        idx = np.argsort(W[:, j])[-top_n:]
        top_idx.update(idx.tolist())
    top_idx = sorted(top_idx)

    W_sub = W[top_idx, :]
    row_labels = [gene_names[i] for i in top_idx]
    col_labels = [f"F{j}" for j in range(n_factors)]

    fig, ax = plt.subplots(
        figsize=(max(4, n_factors * 0.5 + 2), max(4, len(top_idx) * 0.25 + 2))
    )
    sns.heatmap(
        W_sub,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.0,
    )
    ax.set_title("Gene loadings")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _embed(Z: np.ndarray, method: str = "umap") -> np.ndarray:
    """Return 2-D embedding of Z."""
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(Z)
        except Exception:
            # Fall back to PCA
            return _pca(Z)
    elif method == "pca":
        return _pca(Z)
    else:
        raise ValueError(f"Unknown embedding method: {method}")


def _pca(Z: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    return pca.fit_transform(Z)
