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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.input}")
    X = load_data(args.input)
    print(f"  Data shape: {X.shape}")

    dataloader = create_dataloader(X, batch_size=args.batch_size, shuffle=True)

    model = NMFVAE(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        use_nb=not args.use_poisson,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

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


if __name__ == "__main__":
    main()
