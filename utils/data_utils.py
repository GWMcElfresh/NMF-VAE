"""
Data loading and preprocessing utilities for NMF-VAE.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(path: str) -> np.ndarray:
    """
    Auto-detect format and load count matrix.

    Supported formats:
        .h5ad  – AnnData (via anndata)
        .mtx   – Market Exchange Format (via scipy)
        .csv / .tsv  – delimited text (via pandas)
        .npz   – scipy sparse (via scipy)

    Args:
        path: Path to data file.

    Returns:
        count_matrix: Dense numpy array (cells × genes), dtype float32.
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".h5ad":
        import anndata
        adata = anndata.read_h5ad(path)
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        return np.array(X, dtype=np.float32)

    elif ext == ".mtx":
        mat = sp.io.mmread(path).tocsr()
        return np.array(mat.toarray(), dtype=np.float32)

    elif ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
        return df.values.astype(np.float32)

    elif ext == ".npz":
        mat = sp.load_npz(path)
        return np.array(mat.toarray(), dtype=np.float32)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def to_tensor(matrix: Union[np.ndarray, "sp.spmatrix"]) -> torch.Tensor:
    """
    Convert numpy array or scipy sparse matrix to a float32 PyTorch tensor.

    Args:
        matrix: Input array or sparse matrix.

    Returns:
        Float32 tensor.
    """
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return torch.tensor(np.array(matrix, dtype=np.float32), dtype=torch.float32)


def create_dataloader(
    count_matrix: Union[np.ndarray, torch.Tensor],
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a count matrix.

    Args:
        count_matrix: (cells × genes) numpy array or tensor.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle on each epoch.

    Returns:
        PyTorch DataLoader.
    """
    if not isinstance(count_matrix, torch.Tensor):
        tensor = to_tensor(count_matrix)
    else:
        tensor = count_matrix.float()

    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def write_outputs(
    output_dir: str,
    Z: np.ndarray,
    W: np.ndarray,
    metadata: Optional[pd.DataFrame],
    loss_history: List[float],
) -> None:
    """
    Save latent matrix Z, weight matrix W, and loss history to CSV files.

    Args:
        output_dir: Directory path (created if necessary).
        Z: Latent representations (cells × latent).
        W: Decoder weights (genes × latent).
        metadata: Optional cell metadata DataFrame.
        loss_history: List of per-epoch loss values.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Latent matrix
    z_df = pd.DataFrame(
        Z, columns=[f"latent_{i}" for i in range(Z.shape[1])]
    )
    if metadata is not None:
        try:
            metadata = metadata.reset_index(drop=True)
            z_df = pd.concat([metadata, z_df], axis=1)
        except Exception:
            pass
    z_df.to_csv(os.path.join(output_dir, "latent_Z.csv"), index=False)

    # Decoder weights
    w_df = pd.DataFrame(
        W, columns=[f"factor_{i}" for i in range(W.shape[1])]
    )
    w_df.to_csv(os.path.join(output_dir, "decoder_W.csv"), index=False)

    # Loss history
    loss_df = pd.DataFrame({"epoch": range(len(loss_history)), "loss": loss_history})
    loss_df.to_csv(os.path.join(output_dir, "loss_history.csv"), index=False)
