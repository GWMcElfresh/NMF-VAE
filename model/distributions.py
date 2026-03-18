"""
Custom probability distributions for NMF-VAE.

Includes Weibull distribution with reparameterization trick and
KL divergence / log-likelihood utilities.
"""

import torch
import torch.nn as nn
from torch.distributions import constraints
import numpy as np


class WeibullDistribution:
    """
    Weibull distribution with reparameterization support.

    Parameterized by shape k > 0 and scale λ > 0.

    PDF: f(z) = (k/λ) * (z/λ)^(k-1) * exp(-(z/λ)^k)
    """

    def __init__(self, k: torch.Tensor, lam: torch.Tensor):
        """
        Args:
            k: Shape parameter (>0), shape (...,)
            lam: Scale parameter (>0), shape (...,)
        """
        self.k = k
        self.lam = lam

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Draw reparameterized samples using the inverse-CDF trick.

        z = λ * (-log(1 - u))^(1/k),  u ~ Uniform(0, 1)

        Args:
            sample_shape: Extra batch dimensions.

        Returns:
            Samples of shape (*sample_shape, *k.shape)
        """
        shape = sample_shape + self.k.shape
        u = torch.zeros(shape, dtype=self.k.dtype, device=self.k.device).uniform_()
        # Clamp to avoid log(0) or log(1) = -inf
        u = u.clamp(1e-6, 1 - 1e-6)
        z = self.lam * (-torch.log(1.0 - u)).pow(1.0 / self.k)
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log probability of samples under the Weibull distribution.

        log p(z) = log(k/λ) + (k-1)*log(z/λ) - (z/λ)^k

        Args:
            z: Samples (>0), shape matching (k, lam).

        Returns:
            Log probabilities, same shape as z.
        """
        z = z.clamp(min=1e-8)
        ratio = z / self.lam
        log_p = (
            torch.log(self.k)
            - torch.log(self.lam)
            + (self.k - 1.0) * torch.log(ratio)
            - ratio.pow(self.k)
        )
        return log_p

    def entropy(self) -> torch.Tensor:
        """
        Analytic entropy of the Weibull distribution.

        H = γ * (1 - 1/k) + log(λ/k) + 1
        where γ is the Euler–Mascheroni constant.
        """
        euler_mascheroni = 0.5772156649
        return (
            euler_mascheroni * (1.0 - 1.0 / self.k)
            + torch.log(self.lam / self.k)
            + 1.0
        )


def kl_weibull_gamma(
    weibull_k: torch.Tensor,
    weibull_lambda: torch.Tensor,
    gamma_alpha: float = 1.0,
    gamma_beta: float = 1.0,
    n_samples: int = 10,
) -> torch.Tensor:
    """
    Monte-Carlo estimate of KL(Weibull(k, λ) || Gamma(α, β)).

    KL = E_q[log q(z) - log p(z)]

    Args:
        weibull_k: Weibull shape, shape (batch, latent).
        weibull_lambda: Weibull scale, shape (batch, latent).
        gamma_alpha: Gamma shape hyperprior.
        gamma_beta: Gamma rate hyperprior.
        n_samples: Number of MC samples.

    Returns:
        KL divergence per sample, shape (batch, latent).
    """
    q = WeibullDistribution(weibull_k, weibull_lambda)

    kl_sum = torch.zeros_like(weibull_k)
    for _ in range(n_samples):
        z = q.rsample()  # (batch, latent)
        log_q = q.log_prob(z)
        log_p = _gamma_log_prob(z, gamma_alpha, gamma_beta)
        kl_sum = kl_sum + (log_q - log_p)

    return kl_sum / n_samples


def _gamma_log_prob(
    z: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    """
    Log probability under Gamma(alpha, beta) where beta is the rate parameter.

    log p(z) = alpha * log(beta) - lgamma(alpha) + (alpha-1)*log(z) - beta*z
    """
    import math

    z = z.clamp(min=1e-8)
    log_p = (
        alpha * math.log(beta)
        - math.lgamma(alpha)
        + (alpha - 1.0) * torch.log(z)
        - beta * z
    )
    return log_p


def nb_log_likelihood(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """
    Log likelihood under the Negative Binomial distribution.

    NB parameterized by mean μ and overdispersion θ:
      p(x | μ, θ) = Γ(x+θ) / (Γ(θ) * x!) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x

    Numerically stable implementation using lgamma.

    Args:
        x: Observed counts, shape (batch, genes).
        mu: Predicted means (>0), shape (batch, genes).
        theta: Overdispersion parameters (>0), shape (batch, genes) or (genes,).

    Returns:
        Log likelihoods, shape (batch, genes).
    """
    eps = 1e-8
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)

    log_p = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
    )
    return log_p
