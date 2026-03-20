"""
Graph utility functions for graph Laplacian regularization in NMF-VAE.

Provides tools for:
- Building graph Laplacians from STRING protein interaction networks
- Building graph Laplacians from gene co-expression data
- Building signed graph Laplacians from pre-computed gene-gene correlation matrices
- Computing hybrid Laplacians mixing STRING and co-expression graphs
- Converting gene names to official NCBI symbols via mygene
- Fetching the ARCHS4 human gene-gene correlation matrix from S3
- Resolving the lambda (regularization strength) parameter from named presets
- Saving computed Laplacians and decoder weight matrices to disk
"""

from __future__ import annotations

import os
import warnings
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

#: Mapping from NCBI taxonomy species IDs to mygene species name strings.
_NCBI_SPECIES_MAP: Dict[int, str] = {
    9606: "human",
    10090: "mouse",
    10116: "rat",
    7955: "zebrafish",
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


# ---------------------------------------------------------------------------
# Signed graph Laplacian (for positive and negative edge weights)
# ---------------------------------------------------------------------------


def build_signed_laplacian_from_adjacency(
    A: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """
    Compute a signed graph Laplacian from a signed adjacency matrix.

    Extends the standard graph Laplacian to handle both positive and negative
    edge weights (e.g. from gene-gene correlation matrices):

    .. math::

        L_s = D_{|A|} - A

    where :math:`D_{|A|} = \\mathrm{diag}(|A| \\mathbf{1})` is the degree
    matrix built from absolute edge weights.  This formulation:

    - **Penalises dissimilar** decoder weight rows for *positively* correlated
      gene pairs (positive :math:`A_{ij}`).
    - **Penalises similar** decoder weight rows for *negatively* correlated
      gene pairs (negative :math:`A_{ij}`).

    Unlike the standard Laplacian, the signed Laplacian is **not** guaranteed
    to be positive semi-definite.

    Args:
        A: Signed symmetric adjacency matrix of shape ``(n, n)``.  May
            contain negative edge weights.
        normalized: If ``True`` (default), return the *symmetric normalized*
            signed Laplacian
            :math:`L_s = I - D_{|A|}^{-1/2} A D_{|A|}^{-1/2}`.
            If ``False``, return :math:`L_s = D_{|A|} - A`.

    Returns:
        Signed Laplacian matrix as a ``float32`` numpy array of shape
        ``(n, n)``.
    """
    A = np.array(A, dtype=np.float32)
    n = A.shape[0]
    abs_A = np.abs(A)
    d = abs_A.sum(axis=1)  # absolute-value degree vector

    if not normalized:
        L = np.diag(d) - A
    else:
        d_safe = np.where(d > 0, d, 1.0)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d_safe), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt

    return L.astype(np.float32)


# ---------------------------------------------------------------------------
# NCBI gene name conversion
# ---------------------------------------------------------------------------


def convert_to_ncbi_gene_names(
    genes: Sequence[str],
    species_id: int = 9606,
) -> Tuple[List[str], List[bool]]:
    """
    Attempt to map gene names to official NCBI gene symbols via mygene.

    Queries the MyGene.info REST API for each input name (searching gene
    symbols, aliases, and full names).  Genes that cannot be resolved —
    e.g. species-specific ``LOC*`` identifiers — are returned unchanged and
    flagged as unmatched.

    Requires internet access and the ``mygene`` package
    (``pip install mygene``).

    Args:
        genes: Input gene names to convert.
        species_id: NCBI taxonomy ID (default ``9606`` = human).

    Returns:
        Tuple of:

        - ``ncbi_names``: List of converted symbols; falls back to the
          original name for unmatched genes.
        - ``matched``: Boolean list, ``True`` when the gene was found in
          the NCBI database.

    Raises:
        ImportError: If the ``mygene`` package is not installed.
    """
    try:
        import mygene  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'mygene' package is required for NCBI gene name conversion. "
            "Install it with: pip install mygene"
        ) from exc

    species = _NCBI_SPECIES_MAP.get(species_id, str(species_id))

    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        list(genes),
        scopes=["symbol", "alias", "name"],
        fields=["symbol"],
        species=species,
        returnall=True,
        verbose=False,
    )

    gene_map: Dict[str, str] = {}
    for hit in results.get("out", []):
        query = hit.get("query", "")
        symbol = hit.get("symbol", "")
        if query and symbol and "notfound" not in hit:
            if query not in gene_map:
                gene_map[query] = symbol

    ncbi_names: List[str] = []
    matched: List[bool] = []
    for gene in genes:
        if gene in gene_map:
            ncbi_names.append(gene_map[gene])
            matched.append(True)
        else:
            ncbi_names.append(gene)
            matched.append(False)

    return ncbi_names, matched


# ---------------------------------------------------------------------------
# Correlation-matrix-based Laplacian (pkl file)
# ---------------------------------------------------------------------------


def build_correlation_laplacian(
    genes: Sequence[str],
    pkl_path: str,
    correlation_threshold: float = 0.5,
    normalized: bool = True,
    weak_prior_diagonal: float = 0.1,
    convert_ncbi: bool = True,
    species_id: int = 9606,
) -> Tuple[torch.Tensor, List[bool]]:
    """
    Build a signed graph Laplacian from a pre-computed gene-gene correlation
    matrix stored in a pickle file.

    The correlation matrix may contain **both positive and negative values**.
    Positive correlations encourage similar decoder weight rows; negative
    correlations discourage similarity via the signed Laplacian formulation
    (see :func:`build_signed_laplacian_from_adjacency`).

    Genes that cannot be located in the correlation matrix (e.g. species-
    specific ``LOC*`` identifiers not present in the human reference) receive
    a small diagonal entry (``weak_prior_diagonal``) rather than a zero row,
    so that their decoder weight rows are weakly regularised but primarily
    determined by the data.

    Args:
        genes: Ordered list of gene names matching the model's input
            dimension.
        pkl_path: Path to a ``.pkl`` file whose contents are either a
            :class:`pandas.DataFrame` or something convertible to one, with
            gene names as both index and columns.
        correlation_threshold: Absolute correlation values below this
            threshold are set to zero to reduce noise (default ``0.5``).
            Higher values yield sparser graphs.
        normalized: If ``True`` (default), return the symmetric normalized
            signed Laplacian.  If ``False``, return the unnormalized version.
        weak_prior_diagonal: Small diagonal value assigned to genes not found
            in the correlation matrix, providing a weakly informative
            regularization prior (default ``0.1``).  Set to ``0.0`` to treat
            unmatched genes as isolated nodes.
        convert_ncbi: If ``True`` (default), attempt to convert gene names to
            official NCBI symbols using :func:`convert_to_ncbi_gene_names`
            before looking them up in the correlation matrix.  Requires the
            ``mygene`` package; falls back silently if unavailable.
        species_id: NCBI taxonomy ID for gene name conversion (default
            ``9606`` = human).

    Returns:
        Tuple of:

        - ``L``: Signed graph Laplacian as a ``float32``
          :class:`torch.Tensor` of shape ``(n_genes, n_genes)``.
        - ``matched``: Boolean list indicating which input genes were found
          in the correlation matrix (after any name conversion).

    Raises:
        FileNotFoundError: If *pkl_path* does not exist.
        ValueError: If the pickle file cannot be interpreted as a gene-gene
            correlation matrix.
    """
    import pickle  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Correlation pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as fh:
        corr_data = pickle.load(fh)

    if isinstance(corr_data, pd.DataFrame):
        corr_df = corr_data
    elif isinstance(corr_data, dict):
        corr_df = pd.DataFrame(corr_data)
    else:
        try:
            corr_df = pd.DataFrame(corr_data)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert pkl content to a DataFrame: {exc}"
            ) from exc

    # Optionally convert to NCBI symbols
    query_genes: List[str] = list(genes)
    matched: List[bool] = [True] * len(genes)

    if convert_ncbi:
        try:
            query_genes, matched = convert_to_ncbi_gene_names(
                genes, species_id=species_id
            )
        except ImportError:
            warnings.warn(
                "mygene not installed; skipping NCBI name conversion. "
                "Install with: pip install mygene",
                RuntimeWarning,
                stacklevel=2,
            )

    n = len(genes)
    A_signed = np.zeros((n, n), dtype=np.float32)

    corr_index_set = set(corr_df.index)
    gene_to_row: Dict[str, int] = {g: i for i, g in enumerate(query_genes)}

    # Find which query genes are present in the correlation matrix
    common_genes = [g for g in query_genes if g in corr_index_set]

    if common_genes:
        # Vectorised extraction of the relevant submatrix (.copy() ensures mutability)
        sub_corr = corr_df.loc[common_genes, common_genes].to_numpy(
            dtype=np.float32, copy=True
        )
        np.fill_diagonal(sub_corr, 0.0)
        # Threshold spurious correlations
        sub_corr[np.abs(sub_corr) < correlation_threshold] = 0.0

        model_indices = np.array([gene_to_row[g] for g in common_genes])
        A_signed[np.ix_(model_indices, model_indices)] = sub_corr

    # Update matched list for genes absent from the correlation matrix
    for i, g in enumerate(query_genes):
        if g not in corr_index_set:
            matched[i] = False

    n_matched = sum(matched)
    n_unmatched = n - n_matched
    if n_unmatched > 0:
        warnings.warn(
            f"{n_unmatched} of {n} genes were not found in the correlation "
            "matrix and will use a weakly informative diagonal prior.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Build signed Laplacian
    L_np = build_signed_laplacian_from_adjacency(A_signed, normalized=normalized)
    L = torch.tensor(L_np, dtype=torch.float32)

    # Apply weakly informative diagonal for unmatched genes
    if weak_prior_diagonal > 0.0:
        for i, is_matched in enumerate(matched):
            if not is_matched:
                L[i, i] = float(weak_prior_diagonal)

    return L, matched


# ---------------------------------------------------------------------------
# ARCHS4 correlation matrix download
# ---------------------------------------------------------------------------

#: Public S3 URL for the ARCHS4 v2.4 human gene-gene correlation matrix.
ARCHS4_CORRELATION_URL: str = (
    "https://s3.amazonaws.com/mssm-data/human_correlation_v2.4.pkl"
)


def fetch_archs4_correlation(
    dest_path: Optional[str] = None,
    url: str = ARCHS4_CORRELATION_URL,
    chunk_size: int = 1024 * 1024,
    force: bool = False,
) -> str:
    """
    Download the ARCHS4 human gene-gene correlation matrix from S3.

    The file is approximately **6 GB** and is only downloaded once; subsequent
    calls return the cached path without re-downloading unless *force=True*.

    The correlation matrix is a :class:`pandas.DataFrame` pickled with Python's
    ``pickle`` protocol, with gene symbols as both the row index and column
    names.  Pass the returned path directly to
    :func:`build_correlation_laplacian`.

    Requires internet access and the ``requests`` package
    (``pip install requests``).

    Args:
        dest_path: Local filesystem path where the pkl file should be saved.
            Defaults to ``~/.cache/nmfvae/human_correlation_v2.4.pkl``.
        url: Download URL.  Defaults to
            :data:`ARCHS4_CORRELATION_URL`.
        chunk_size: Number of bytes per streaming chunk during download
            (default 1 MiB).  Larger values reduce syscall overhead on fast
            connections.
        force: If ``True``, re-download even if the file already exists
            (default ``False``).

    Returns:
        Absolute path to the downloaded (or already-cached) pkl file.

    Raises:
        ImportError: If the ``requests`` package is not installed.
        RuntimeError: If the HTTP request fails or returns a non-200 status.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required to download the ARCHS4 "
            "correlation matrix.  Install it with: pip install requests"
        ) from exc

    if dest_path is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "nmfvae"
        )
        dest_path = os.path.join(cache_dir, "human_correlation_v2.4.pkl")

    dest_path = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and not force:
        print(f"Using cached ARCHS4 correlation matrix: {dest_path}")
        return dest_path

    print(f"Downloading ARCHS4 correlation matrix from {url}")
    print(f"  Destination: {dest_path}")
    print("  File size: ~6 GB — this may take several minutes.")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download ARCHS4 correlation matrix: {exc}"
        ) from exc

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    report_every = 100 * 1024 * 1024  # report every 100 MiB
    next_report = report_every

    # Write to a temporary file first, then rename on success to avoid
    # leaving a partial file behind if the download is interrupted.
    tmp_path = dest_path + ".part"
    try:
        with open(tmp_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        if total:
                            pct = 100.0 * downloaded / total
                            print(
                                f"  {downloaded / 1e9:.2f} GB / "
                                f"{total / 1e9:.2f} GB ({pct:.0f}%)"
                            )
                        else:
                            print(f"  {downloaded / 1e9:.2f} GB downloaded")
                        next_report += report_every
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(
        f"  Download complete: {downloaded / 1e9:.2f} GB saved to {dest_path}"
    )
    return dest_path


# ---------------------------------------------------------------------------
# Saving Laplacian and W matrix to disk
# ---------------------------------------------------------------------------


def save_laplacian(
    L: torch.Tensor,
    path: str,
    W: Optional[np.ndarray] = None,
    gene_names: Optional[Sequence[str]] = None,
) -> None:
    """
    Save a graph Laplacian (and optionally a decoder weight matrix) to disk.

    Writes:

    - ``<path>_laplacian.npy`` – the Laplacian as a NumPy ``.npy`` array.
    - ``<path>_W.csv`` – the decoder weight matrix as a CSV (only when *W*
      is provided).

    Args:
        L: Graph Laplacian tensor to save, shape ``(n_genes, n_genes)``.
        path: Output path *prefix* (without extension).  Parent directory is
            created automatically.
        W: Optional decoder weight matrix of shape ``(n_genes, n_latent)``.
            When provided it is written to ``<path>_W.csv``.
        gene_names: Optional gene name labels for the CSV row index.  When
            ``None``, integer indices are used.
    """
    import pandas as pd  # noqa: PLC0415

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    np.save(f"{path}_laplacian.npy", L.detach().cpu().numpy())

    if W is not None:
        index = gene_names if gene_names is not None else range(W.shape[0])
        w_df = pd.DataFrame(
            W,
            index=index,
            columns=[f"factor_{i}" for i in range(W.shape[1])],
        )
        w_df.to_csv(f"{path}_W.csv")
