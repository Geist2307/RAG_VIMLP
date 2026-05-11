"""
src/BayesMolchanov.py
----------------------
Variational Dropout MLP (Molchanov et al. 2017) with PyTorch autograd.
Replaces finite-difference gradients with proper backpropagation.

Public API (unchanged):
    VariationalDropoutMolchanov
    VarMLP
    make_mlp
    train_mlp
    posterior_predictive
    fit_trend
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

# ── Molchanov constants ────────────────────────────────────────────────────────
K1, K2, K3 = 0.63576, 1.87320, 1.48695


# ── Variational layer ──────────────────────────────────────────────────────────

class VariationalDropoutMolchanov(nn.Module):
    """
    Single fully-connected variational layer.
    Parameters θ (mean) and logσ² are nn.Parameter so autograd tracks them.
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
        """Stochastic forward pass via reparameterisation trick."""
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

    def sparsity(self, threshold: float = 3.0) -> float:
        log_alpha = self.logσ2 - 2.0 * torch.log(torch.abs(self.theta) + 1e-8)
        return float((log_alpha >= threshold).float().mean())


# ── VarMLP ─────────────────────────────────────────────────────────────────────

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


def make_mlp(hidden: int = 64) -> VarMLP:
    """
    Build [1 → hidden → 1] VDM network with sin hidden activation.
    Mirrors Julia: make_model([1, 64, 1]; activations=[sin], final_activation=identity)
    """
    l1 = VariationalDropoutMolchanov(in_features=1,  out_features=hidden)
    l2 = VariationalDropoutMolchanov(in_features=hidden, out_features=1)
    return VarMLP(layers=[l1, l2], activations=[torch.sin, None])


# ── Loss ───────────────────────────────────────────────────────────────────────

def energy_loss(model: VarMLP, x: torch.Tensor, y: torch.Tensor,
                kl_scale: float = 1.0) -> torch.Tensor:
    N    = y.shape[0]
    yhat = model(x).squeeze()
    nll  = N * torch.mean((yhat - y) ** 2)
    return nll + kl_scale * model.kl() / N


# ── Training with warmup + annealing ──────────────────────────────────────────

def train_mlp(model: VarMLP,
              x: np.ndarray,
              y: np.ndarray,
              warmup_epochs: int = 150,
              anneal_epochs: int = 150,
              lr_warmup: float = 0.1,
              lr_anneal: float = 0.01) -> list:
    """
    Adam training with warmup + KL annealing schedule.

    Phase 1 — warmup  (epochs 0 → warmup_epochs):
        KL weight = 0,   lr = lr_warmup

    Phase 2 — annealing (epochs warmup_epochs → warmup + anneal):
        KL weight = 0 → 1 linearly,   lr = lr_anneal
    """
    x_t = torch.tensor(x, dtype=torch.float32)   # (1, N)
    y_t = torch.tensor(y, dtype=torch.float32)   # (N,)

    # two param groups so we can switch lr mid-training
    optimiser = torch.optim.Adam(model.parameters(), lr=lr_warmup)

    losses      = []
    lr_switched = False

    for epoch in range(warmup_epochs + anneal_epochs):

        # ── schedule ──────────────────────────────────────────────────
        if epoch < warmup_epochs:
            kl_scale = 0.0
        else:
            progress = (epoch - warmup_epochs) / max(anneal_epochs - 1, 1)
            kl_scale = float(progress)

            # switch lr once at the annealing boundary
            if not lr_switched:
                for pg in optimiser.param_groups:
                    pg["lr"] = lr_anneal
                lr_switched = True

        # ── forward + backward ────────────────────────────────────────
        optimiser.zero_grad()
        loss = energy_loss(model, x_t, y_t, kl_scale=kl_scale)
        loss.backward()
        optimiser.step()

        losses.append(float(loss.detach()))

    return losses


# ── Posterior predictive ───────────────────────────────────────────────────────

def posterior_predictive(model: VarMLP,
                         x_grid: np.ndarray,
                         n_samples: int = 200) -> tuple:
    """
    n_samples stochastic forward passes → (mean, std) in normalised space.
    x_grid : (1, N_grid) numpy array
    """
    x_t   = torch.tensor(x_grid, dtype=torch.float32)
    model.eval()   # keeps dropout behaviour — VDM samples at every forward pass

    with torch.no_grad():
        preds = torch.stack([
            model(x_t).squeeze() for _ in range(n_samples)
        ])                          # (n_samples, N_grid)

    model.train()
    mean = preds.mean(dim=0).numpy()
    std  = preds.std(dim=0).numpy()
    return mean, std


# Load and predict
def load_and_predict(series_id: str,
                     values: list,
                     n_future: int = 30,
                     n_samples: int = 200,
                     registry_path: str = "models/registry.json") -> dict:
    """
    Load a pretrained VarMLP from the model registry and run
    posterior predictive inference on observed + future time points.
    No training happens here.
    """
    import json, torch

    with open(registry_path) as f:
        registry = json.load(f)

    if series_id not in registry:
        raise KeyError(f"No pretrained model found for {series_id} in registry.")

    entry = registry[series_id]

    # reconstruct architecture and load weights
    model = make_mlp(hidden=entry["hidden"])
    model_path = os.path.join(os.path.dirname(registry_path), entry["model_path"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # normalise fresh observations using saved constants
    y_raw   = np.array(values, dtype=np.float32)
    N       = len(y_raw)
    x_raw   = np.arange(N, dtype=np.float32)
    x_mean  = entry["x_mean"]
    x_std   = entry["x_std"]
    y_mean  = entry["y_mean"]
    y_std   = entry["y_std"]

    x_norm = ((x_raw - x_mean) / x_std).reshape(1, -1)

    # extend grid into future
    x_future_max = x_norm.max() + n_future / x_std
    x_grid_norm  = np.linspace(x_norm.min(), x_future_max, 300).reshape(1, -1)
    x_grid_orig  = x_grid_norm.flatten() * x_std + x_mean

    # posterior predictive
    pred_mean_norm, pred_std_norm = posterior_predictive(
        model, x_grid_norm, n_samples=n_samples
    )
    pred_mean = pred_mean_norm * y_std + y_mean
    pred_std  = pred_std_norm  * y_std


    return {
        "x_obs_orig":   x_raw.tolist(),
        "y_obs_orig":   y_raw.tolist(),
        "x_grid_orig":  x_grid_orig.tolist(),
        "y_mean":       pred_mean.tolist(),
        "y_std":        pred_std.tolist(),
        "last_obs_idx": float(x_raw[-1]),
        "n_future":     n_future,
        "n_samples":    n_samples,           # passed to chart for alert
        "trained_on":   entry["trained_on"],
    }