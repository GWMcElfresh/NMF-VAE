#!/usr/bin/env python
"""
Preprocessing script for scRNA-seq count matrices.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq data")
    parser.add_argument("--input", required=True, help="Input count matrix path")
    parser.add_argument("--output", required=True, help="Output path (.npz)")
    parser.add_argument(
        "--min-genes", type=int, default=200, help="Min genes per cell"
    )
    parser.add_argument(
        "--min-cells", type=int, default=3, help="Min cells per gene"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Library-size normalize to 1e4 counts",
    )
    parser.add_argument("--log1p", action="store_true", help="Apply log1p transform")
    return parser.parse_args()


def main():
    args = parse_args()

    from utils.data_utils import load_data
    X = load_data(args.input)
    print(f"Input shape: {X.shape}")

    # Filter cells by min genes
    cell_mask = (X > 0).sum(axis=1) >= args.min_genes
    X = X[cell_mask, :]
    print(f"After min-genes filter ({args.min_genes}): {X.shape}")

    # Filter genes by min cells
    gene_mask = (X > 0).sum(axis=0) >= args.min_cells
    X = X[:, gene_mask]
    print(f"After min-cells filter ({args.min_cells}): {X.shape}")

    if args.normalize:
        lib_sizes = X.sum(axis=1, keepdims=True).clip(1)
        X = X / lib_sizes * 1e4
        print("Library-size normalization applied")

    if args.log1p:
        X = np.log1p(X)
        print("Log1p transform applied")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sp.save_npz(args.output, sp.csr_matrix(X))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
