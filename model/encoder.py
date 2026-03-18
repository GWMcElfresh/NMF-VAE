"""
Weibull encoder for NMF-VAE.

Maps gene expression profiles to Weibull distribution parameters (k, λ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class WeibullEncoder(nn.Module):
    """
    Amortized Weibull encoder.

    Architecture: FC → BN → LeakyReLU (repeated) → two output heads for k, λ.

    Both k and λ are passed through softplus to ensure positivity.

    Args:
        input_dim: Number of input genes.
        latent_dim: Dimensionality of the latent space (= number of factors).
        hidden_dims: Sizes of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build shared trunk
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h

        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.fc_k = nn.Linear(in_dim, latent_dim)
        self.fc_lam = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode gene expression to Weibull parameters.

        Args:
            x: Raw count matrix, shape (batch, genes).

        Returns:
            k: Weibull shape parameters, shape (batch, latent).
            lam: Weibull scale parameters, shape (batch, latent).
        """
        # Log1p transform for numerical stability
        x = torch.log1p(x)

        h = self.trunk(x)

        # softplus ensures strict positivity; add small offset for stability
        k = F.softplus(self.fc_k(h)) + 1e-4
        lam = F.softplus(self.fc_lam(h)) + 1e-4

        return k, lam
