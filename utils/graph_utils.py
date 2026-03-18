"""
Graph utility functions for graph Laplacian regularization in NMF-VAE.

Provides tools for:
- Building graph Laplacians from STRING protein interaction networks
- Building graph Laplacians from gene co-expression data
- Computing hybrid Laplacians mixing STRING and co-expression graphs
- Resolving the lambda (regularization strength) parameter from named presets
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lambda preset registry
# ---------------------------------------------------------------------------

#: Named regularization-strength presets for the graph Laplacian penalty.
LAMBDA_PRESETS: Dict[str, float] = {
    "none": 0.0,
    "weak": 0.01,
    "moderate": 0.1,
    "strong": 1.0,
}


def resolve_lambda(lambda_graph: Union[float, str]) -> float:
    """
    Resolve a regularization-strength value to a float.

    A numeric value is returned as-is (after validation).  A string is looked
    up in :data:`LAMBDA_PRESETS`.  The user-provided numeric value always takes
    priority over any preset.

    Args:
        lambda_graph: Either a non-negative float **or** one of the preset
            names ``"none"``, ``"weak"``, ``"moderate"``, ``"strong"``.

    Returns:
        Float value for lambda (≥ 0).

    Raises:
        ValueError: If *lambda_graph* is an unknown preset name or a negative
            number.
    """
    if isinstance(lambda_graph, str):
        key = lambda_graph.lower()
        if key not in LAMBDA_PRESETS:
            raise ValueError(
                f"Unknown lambda preset '{lambda_graph}'. "
                f"Valid presets: {list(LAMBDA_PRESETS)}"
            )
        return LAMBDA_PRESETS[key]
    value = float(lambda_graph)
    if value < 0.0:
        raise ValueError(f"lambda_graph must be non-negative, got {value}")
    return value


# ---------------------------------------------------------------------------
# Laplacian construction from adjacency matrix
# ---------------------------------------------------------------------------


def build_laplacian_from_adjacency(
    A: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """
    Compute the graph Laplacian from a weighted adjacency matrix.

    Args:
        A: Symmetric adjacency matrix of shape ``(n, n)`` with non-negative
            edge weights.
        normalized: If ``True`` (default), return the *symmetric normalized*
            Laplacian ``L = I - D^{-1/2} A D^{-1/2}``.  If ``False``, return
            the *unnormalized* Laplacian ``L = D - A``.

    Returns:
        Laplacian matrix as a ``float32`` numpy array of shape ``(n, n)``.
    """
    A = np.array(A, dtype=np.float32)
    n = A.shape[0]
    d = A.sum(axis=1)  # degree vector

    if not normalized:
        L = np.diag(d) - A
    else:
        # Symmetric normalized: L = I - D^{-1/2} A D^{-1/2}
        d_safe = np.where(d > 0, d, 1.0)  # avoid division by zero for isolated nodes
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d_safe), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt

    return L.astype(np.float32)


# ---------------------------------------------------------------------------
# STRING network integration
# ---------------------------------------------------------------------------


def fetch_string_interactions(
    genes: Sequence[str],
    species_id: int = 9606,
    confidence_threshold: float = 0.7,
) -> List[Tuple[str, str, float]]:
    """
    Retrieve protein–protein interactions from the STRING REST API.

    Requires internet access and the ``requests`` package.

    Args:
        genes: Gene identifiers (symbols or STRING IDs) to query.
        species_id: NCBI taxonomy ID (default ``9606`` = human).
        confidence_threshold: Minimum combined confidence score in ``[0, 1]``.
            Interactions with lower scores are excluded.

    Returns:
        List of ``(gene_a, gene_b, score)`` tuples, where *score* is the
        STRING combined confidence value in ``[0, 1]``.

    Raises:
        ImportError: If the ``requests`` package is not installed.
        RuntimeError: If the STRING API request fails.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required to fetch STRING interactions. "
            "Install it with: pip install requests"
        ) from exc

    url = "https://string-db.org/api/json/network"
    payload = {
        "identifiers": "\n".join(genes),
        "species": species_id,
        "required_score": int(confidence_threshold * 1000),
        "caller_identity": "nmf_vae",
    }

    try:
        response = requests.post(url, data=payload, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"STRING API request failed: {exc}") from exc

    interactions: List[Tuple[str, str, float]] = []
    for item in response.json():
        gene_a = item.get("preferredName_A") or item.get("stringId_A", "")
        gene_b = item.get("preferredName_B") or item.get("stringId_B", "")
        score = float(item.get("score", 0.0))
        if gene_a and gene_b:
            interactions.append((gene_a, gene_b, score))

    return interactions


def build_string_laplacian(
    genes: Sequence[str],
    confidence_threshold: float = 0.7,
    species_id: int = 9606,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Build a graph Laplacian from the STRING protein interaction network.

    Genes that are not found in STRING or that have no interactions above
    *confidence_threshold* are treated as isolated nodes (zero row/column in
    the adjacency matrix).

    Args:
        genes: Ordered list of gene identifiers matching the model's gene
            input dimension.
        confidence_threshold: Minimum STRING confidence score in ``[0, 1]``.
        species_id: NCBI taxonomy species ID (default ``9606`` = human).
        normalized: If ``True``, return the symmetric normalized Laplacian.

    Returns:
        Laplacian as a ``float32`` :class:`torch.Tensor` of shape
        ``(n_genes, n_genes)``.
    """
    n = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    interactions = fetch_string_interactions(
        genes,
        species_id=species_id,
        confidence_threshold=confidence_threshold,
    )

    A = np.zeros((n, n), dtype=np.float32)
    for gene_a, gene_b, score in interactions:
        i = gene_to_idx.get(gene_a)
        j = gene_to_idx.get(gene_b)
        if i is not None and j is not None:
            A[i, j] = score
            A[j, i] = score

    L = build_laplacian_from_adjacency(A, normalized=normalized)
    return torch.tensor(L, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Co-expression graph
# ---------------------------------------------------------------------------


def build_coexpression_laplacian(
    X: np.ndarray,
    k: int = 10,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Build a gene co-expression kNN-graph Laplacian.

    Computes pairwise Pearson correlations between genes, constructs a
    *k*-nearest-neighbour graph based on absolute correlation magnitude, and
    returns the corresponding graph Laplacian.

    Args:
        X: Cell × gene count (or normalized) matrix of shape
            ``(n_cells, n_genes)``.
        k: Number of nearest-neighbour genes per gene.
        normalized: If ``True``, return the symmetric normalized Laplacian.

    Returns:
        Laplacian as a ``float32`` :class:`torch.Tensor` of shape
        ``(n_genes, n_genes)``.
    """
    X = np.array(X, dtype=np.float32)
    n_genes = X.shape[1]
    X_t = X.T  # (n_genes, n_cells)

    # Pearson correlation matrix
    mean = X_t.mean(axis=1, keepdims=True)
    std = X_t.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X_t - mean) / std  # (n_genes, n_cells)
    corr = X_norm @ X_norm.T / X_t.shape[1]  # (n_genes, n_genes)

    # Build kNN adjacency using absolute correlation as similarity
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)  # exclude self-loops

    A = np.zeros((n_genes, n_genes), dtype=np.float32)
    k_clamped = min(k, n_genes - 1)
    for i in range(n_genes):
        top_k = np.argpartition(abs_corr[i], -k_clamped)[-k_clamped:]
        for j in top_k:
            w = max(0.0, float(corr[i, j]))  # clamp negative correlations to zero
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)

    L = build_laplacian_from_adjacency(A, normalized=normalized)
    return torch.tensor(L, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Hybrid Laplacian
# ---------------------------------------------------------------------------


def build_hybrid_laplacian(
    L_string: torch.Tensor,
    L_data: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combine STRING and co-expression Laplacians via linear interpolation.

    .. math::

        L = \\alpha \\, L_{\\text{STRING}} + (1 - \\alpha) \\, L_{\\text{data}}

    Args:
        L_string: Laplacian derived from the STRING network, shape
            ``(n_genes, n_genes)``.
        L_data: Laplacian derived from co-expression/kNN graph, shape
            ``(n_genes, n_genes)``.
        alpha: Mixing coefficient in ``[0, 1]``.  ``alpha=1`` gives a
            STRING-only prior; ``alpha=0`` gives a data-only prior.

    Returns:
        Hybrid Laplacian tensor of shape ``(n_genes, n_genes)``.

    Raises:
        ValueError: If *alpha* is outside ``[0, 1]``.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * L_string + (1.0 - alpha) * L_data
