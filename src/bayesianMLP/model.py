"""
src/BayesMolchanov.py
----------------------
Variational Dropout MLP (Molchanov et al. 2017) with PyTorch autograd.

Public API:
    VariationalDropoutMolchanov
    VarMLP
    make_mlp

Handles the creation of a VariationalMLP
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

# Molchanov constants (Molchanov et al. 2017)
K1, K2, K3 = 0.63576, 1.87320, 1.48695

# Molchanov et al. (2017) : weights with log_alpha >= this are considered pruned
SPARSITY_THRESHOLD = 3.0


# Variational layer

class VariationalDropoutMolchanov(nn.Module):
    """
    Single fully-connected variational layer.
    Parameters θ (mean) and logσ² are nn.Parameter so autograd tracks them.

    Args:

    in_features : features the layer takes in
    out_features: features the layer takes out
    theta : the weight mean
    logσ2 : the weights log squared variance 
    bias : the bias
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # weight mean and log-variance
        self.theta = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        self.logσ2 = nn.Parameter(
            torch.full((out_features, in_features), -3.0)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic forward pass via reparameterisation trick from Molchanov et al (2017)."""
        ε = torch.randn_like(self.theta)
        W = self.theta + torch.exp(0.5 * self.logσ2) * ε
        return W @ x + self.bias.unsqueeze(-1) if x.dim() == 2 else W @ x + self.bias

    def kl(self) -> torch.Tensor:
        """KL divergence — Molchanov et al. (2017) eq. 14."""
        log_alpha = self.logσ2 - 2.0 * torch.log(torch.abs(self.theta) + 1e-8)
        neg_kl = (
            -K1
            + K1 * torch.sigmoid(K2 + K3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
        )
        return -neg_kl.sum()

    
    # notice this is a magic number, we keep sparisity at log alpha 3
    def sparsity(self, threshold: float = SPARSITY_THRESHOLD) -> float:
        log_alpha = self.logσ2 - 2.0 * torch.log(torch.abs(self.theta) + 1e-8)
        return float((log_alpha >= threshold).float().mean())


# VarMLP with one hidden layer

class VarMLP(nn.Module):
    """
    Variational MLP: stack of VariationalDropoutMolchanov layers
    with per-layer activations.
    """

    def __init__(self, layers: list, activations: list):
        super().__init__()
        assert len(activations) == len(layers)
        self.vdm_layers  = nn.ModuleList(layers)
        self.activations = activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, act in zip(self.vdm_layers, self.activations):
            out = layer(out)
            if act is not None:
                out = act(out)
        return out

    def kl(self) -> torch.Tensor:
        return sum(l.kl() for l in self.vdm_layers)


# method to make an MLP
def make_mlp(sizes: list[int], activations: list = None) -> VarMLP:
    """
    Build a VDM MLP with arbitrary depth.
    sizes      : [in, hidden1, hidden2, ..., out]
    activations: per-layer activations, length = len(sizes) - 1
                 defaults to [torch.sin, ..., None] (sin hidden, identity output)
    """
    assert len(sizes) >= 2
    if activations is None:
        activations = [torch.sin] * (len(sizes) - 2) + [None]
    assert len(activations) == len(sizes) - 1. ## activations sit between layers, always one less

    layers = [
        VariationalDropoutMolchanov(sizes[i], sizes[i+1])
        for i in range(len(sizes) - 1)
    ]
    return VarMLP(layers=layers, activations=activations)



