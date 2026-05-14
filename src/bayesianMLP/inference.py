from src.bayesianMLP.model import VarMLP, make_mlp
import numpy as np
import torch 
from src.bayesianMLP.training import energy_loss
import os

# path to saved models
DEFAULT_REGISTRY = "models/registry.json"  # relative to project root where app runs



# Posterior predictive
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
    mean = preds.mean(dim=0).numpy() # compute mean of preds
    std  = preds.std(dim=0).numpy() # compute std of preds
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
    No training happens here, just inference.
    """
    import json, torch

    with open(registry_path) as f:
        registry = json.load(f)

    if series_id not in registry:
        raise KeyError(f"No pretrained model found for {series_id} in registry.")

    entry = registry[series_id]

    # reconstruct architecture and load weights
    model = make_mlp([1, entry["hidden"], 1]) # built for one input, one output
    model_path = os.path.join(os.path.dirname(registry_path), entry["model_path"])
    model.load_state_dict(torch.load(model_path, weights_only=True)) # we only load the weights
    model.eval() # this keeps in into inference mode

    # normalise fresh observations using saved constants
    y_raw   = np.array(values, dtype=np.float32)
    N       = len(y_raw)
    x_raw   = np.arange(N, dtype=np.float32)

    # parameters from saved model (but should we recalculate)
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
    pred_mean = pred_mean_norm * y_std + y_mean # use predicted mean + actual std
    pred_std  = pred_std_norm  * y_std # keeps standard dev


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