#!/usr/bin/env python
"""
Command-line training script for NMF-VAE.
"""

import argparse
import sys
import os

# Allow running without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.vae import NMFVAE
from utils.data_utils import load_data, create_dataloader, write_outputs
from utils.graph_utils import (
    LAMBDA_PRESETS,
    ARCHS4_CORRELATION_URL,
    build_string_laplacian,
    build_coexpression_laplacian,
    build_hybrid_laplacian,
    build_correlation_laplacian,
    fetch_archs4_correlation,
    save_laplacian,
    resolve_lambda,
)
from utils.plot_utils import plot_elbo


def parse_args():
    parser = argparse.ArgumentParser(description="Train NMF-VAE on scRNA-seq data")
    parser.add_argument("--input", required=True, help="Path to count matrix")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=10, help="Latent dimensionality")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes",
    )
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL weight")
    parser.add_argument(
        "--use-poisson",
        action="store_true",
        help="Use Poisson likelihood instead of NB",
    )

    # Graph Laplacian arguments
    graph_group = parser.add_argument_group("Graph Laplacian regularization")
    graph_group.add_argument(
        "--lambda-graph",
        default="none",
        help=(
            "Strength of graph Laplacian penalty.  Accepts a non-negative "
            "float or one of the named presets: "
            + ", ".join(f"'{k}' ({v})" for k, v in LAMBDA_PRESETS.items())
            + ".  Default: 'none' (disabled)."
        ),
    )
    graph_group.add_argument(
        "--genes-file",
        default=None,
        help=(
            "Path to a plain-text file with one gene symbol per line, "
            "matching the columns of the count matrix.  Required when using "
            "STRING-based graph regularization (--use-string-graph)."
        ),
    )
    graph_group.add_argument(
        "--use-string-graph",
        action="store_true",
        help="Build graph Laplacian from the STRING protein interaction network.",
    )
    graph_group.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="STRING edge confidence threshold in [0, 1] (default: 0.7).",
    )
    graph_group.add_argument(
        "--species-id",
        type=int,
        default=9606,
        help="NCBI taxonomy species ID for STRING queries (default: 9606 = human).",
    )
    graph_group.add_argument(
        "--use-normalized-laplacian",
        action="store_true",
        help="Use the symmetric normalized Laplacian (default: unnormalized).",
    )
    graph_group.add_argument(
        "--use-coexpression-graph",
        action="store_true",
        help="Build a gene co-expression kNN graph Laplacian from the data.",
    )
    graph_group.add_argument(
        "--k-data-graph",
        type=int,
        default=10,
        help="Number of nearest neighbors for the co-expression kNN graph (default: 10).",
    )
    graph_group.add_argument(
        "--use-hybrid-graph",
        action="store_true",
        help=(
            "Combine STRING and co-expression Laplacians "
            "(requires --use-string-graph and --use-coexpression-graph)."
        ),
    )
    graph_group.add_argument(
        "--alpha-graph-mix",
        type=float,
        default=0.5,
        help=(
            "Mixing coefficient for the hybrid graph: "
            "alpha * L_STRING + (1 - alpha) * L_data (default: 0.5)."
        ),
    )
    graph_group.add_argument(
        "--correlation-pkl",
        default=None,
        help=(
            "Path to a .pkl file of pre-computed gene-gene correlations "
            "(e.g. human_correlation_v2.4.pkl).  Builds a signed graph "
            "Laplacian that handles both positive and negative correlations. "
            "Requires --genes-file."
        ),
    )
    graph_group.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.5,
        help=(
            "Absolute correlation threshold below which edges are removed "
            "from the correlation graph (default: 0.5).  Higher values yield "
            "sparser, less noisy graphs."
        ),
    )
    graph_group.add_argument(
        "--weak-prior-diagonal",
        type=float,
        default=0.1,
        help=(
            "Diagonal value for genes not found in the correlation matrix, "
            "providing a weakly informative regularization prior (default: 0.1). "
            "Set to 0.0 to treat unmatched genes as isolated nodes."
        ),
    )
    graph_group.add_argument(
        "--no-ncbi-convert",
        action="store_true",
        help=(
            "Disable automatic conversion of gene names to NCBI symbols "
            "when building the correlation graph (conversion is on by default)."
        ),
    )
    graph_group.add_argument(
        "--save-laplacian",
        default=None,
        help=(
            "Path prefix for saving the computed Laplacian and W matrix to disk. "
            "Writes <prefix>_laplacian.npy and <prefix>_W.csv."
        ),
    )
    graph_group.add_argument(
        "--fetch-archs4",
        action="store_true",
        help=(
            "Download the ARCHS4 human gene-gene correlation matrix (~6 GB) "
            f"from {ARCHS4_CORRELATION_URL} and use it as the correlation pkl. "
            "The file is cached at ~/.cache/nmfvae/human_correlation_v2.4.pkl "
            "so subsequent runs are instant.  Requires --genes-file and "
            "--lambda-graph > 0."
        ),
    )
    graph_group.add_argument(
        "--archs4-cache-path",
        default=None,
        help=(
            "Local path to cache the downloaded ARCHS4 pkl file "
            "(default: ~/.cache/nmfvae/human_correlation_v2.4.pkl). "
            "Only used together with --fetch-archs4."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.input}")
    X = load_data(args.input)
    print(f"  Data shape: {X.shape}")

    dataloader = create_dataloader(X, batch_size=args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Resolve graph Laplacian
    # ------------------------------------------------------------------
    graph_laplacian = None
    lambda_val = resolve_lambda(args.lambda_graph)

    if lambda_val > 0.0:
        normalized = args.use_normalized_laplacian

        L_string = None
        L_data = None

        if args.use_string_graph:
            if args.genes_file is None:
                parser_err = (
                    "--genes-file is required when using --use-string-graph."
                )
                raise SystemExit(f"Error: {parser_err}")
            with open(args.genes_file) as fh:
                genes = [line.strip() for line in fh if line.strip()]
            print(
                f"Building STRING Laplacian for {len(genes)} genes "
                f"(confidence ≥ {args.confidence_threshold})…"
            )
            L_string = build_string_laplacian(
                genes,
                confidence_threshold=args.confidence_threshold,
                species_id=args.species_id,
                normalized=normalized,
            )
            print("  STRING Laplacian built.")

        if args.use_coexpression_graph:
            print(f"Building co-expression Laplacian (k={args.k_data_graph})…")
            L_data = build_coexpression_laplacian(
                X, k=args.k_data_graph, normalized=normalized
            )
            print("  Co-expression Laplacian built.")

        if args.use_hybrid_graph:
            if L_string is None or L_data is None:
                raise SystemExit(
                    "Error: --use-hybrid-graph requires both "
                    "--use-string-graph and --use-coexpression-graph."
                )
            graph_laplacian = build_hybrid_laplacian(
                L_string, L_data, alpha=args.alpha_graph_mix
            )
            print(
                f"  Using hybrid Laplacian (alpha={args.alpha_graph_mix})."
            )
        elif L_string is not None:
            graph_laplacian = L_string
        elif L_data is not None:
            graph_laplacian = L_data

        # --fetch-archs4 downloads/caches the pkl and sets correlation_pkl
        correlation_pkl = args.correlation_pkl
        if args.fetch_archs4:
            if args.genes_file is None:
                raise SystemExit(
                    "Error: --genes-file is required when using --fetch-archs4."
                )
            correlation_pkl = fetch_archs4_correlation(
                dest_path=args.archs4_cache_path
            )

        if correlation_pkl is not None:
            if args.genes_file is None:
                raise SystemExit(
                    "Error: --genes-file is required when using "
                    "--correlation-pkl."
                )
            with open(args.genes_file) as fh:
                genes = [line.strip() for line in fh if line.strip()]
            print(
                f"Building correlation Laplacian for {len(genes)} genes "
                f"from {correlation_pkl} "
                f"(threshold={args.correlation_threshold})…"
            )
            corr_laplacian, matched = build_correlation_laplacian(
                genes,
                pkl_path=correlation_pkl,
                correlation_threshold=args.correlation_threshold,
                normalized=normalized,
                weak_prior_diagonal=args.weak_prior_diagonal,
                convert_ncbi=not args.no_ncbi_convert,
                species_id=args.species_id,
            )
            n_matched = sum(matched)
            print(
                f"  Correlation Laplacian built "
                f"({n_matched}/{len(genes)} genes matched)."
            )
            # Correlation graph replaces or combines with any existing prior
            if graph_laplacian is not None:
                graph_laplacian = build_hybrid_laplacian(
                    corr_laplacian, graph_laplacian, alpha=args.alpha_graph_mix
                )
                print(
                    f"  Merged with existing Laplacian "
                    f"(alpha={args.alpha_graph_mix})."
                )
            else:
                graph_laplacian = corr_laplacian

    model = NMFVAE(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        use_nb=not args.use_poisson,
        lambda_graph=args.lambda_graph,
        graph_laplacian=graph_laplacian,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    if lambda_val > 0.0:
        print(f"Graph Laplacian penalty: lambda={lambda_val}")

    print("Training...")
    loss_history = model.fit(
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
    )
    print(f"Final loss: {loss_history[-1]:.4f}")

    # Save outputs
    import torch
    model.eval()
    Z = model.transform(X)
    W = model.get_gene_programs()

    write_outputs(args.output, Z, W, None, loss_history)
    plot_elbo(loss_history, save_path=os.path.join(args.output, "loss.png"))
    print(f"Results saved to {args.output}")

    if args.save_laplacian is not None and graph_laplacian is not None:
        gene_names_for_save = None
        if args.genes_file is not None:
            with open(args.genes_file) as fh:
                gene_names_for_save = [line.strip() for line in fh if line.strip()]
        save_laplacian(graph_laplacian, args.save_laplacian, W=W, gene_names=gene_names_for_save)
        print(f"Laplacian saved to {args.save_laplacian}_laplacian.npy")


if __name__ == "__main__":
    main()
