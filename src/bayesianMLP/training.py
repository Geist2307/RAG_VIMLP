from src.bayesianMLP.model import VarMLP 
import numpy as np
import torch 



def energy_loss(model: VarMLP, x: torch.Tensor, y: torch.Tensor,
                kl_scale: float = 1.0) -> torch.Tensor:
    N    = y.shape[0]
    yhat = model(x).squeeze()
    nll  = N * torch.mean((yhat - y) ** 2) # Negative Log Likelihood (NLL)
    return nll + kl_scale * model.kl() / N 

# Training with warmup + annealing 

def train_mlp(model: VarMLP,
              x: np.ndarray,
              y: np.ndarray,
              warmup_epochs: int = 150,
              anneal_epochs: int = 150,
              lr_warmup: float = 0.1,
              lr_anneal: float = 0.01) -> list:
    """
    Adam training with warmup + KL annealing schedule.

    Phase 1 — warmup  (epochs 0 -> warmup_epochs):
        KL weight = 0,   lr = lr_warmup

    Phase 2 — annealing (epochs warmup_epochs → warmup + anneal):
        KL weight = 0 -> 1 linearly,   lr = lr_anneal
    """
    x_t = torch.tensor(x, dtype=torch.float32)   # (1, N)
    y_t = torch.tensor(y, dtype=torch.float32)   # (N,)

    # two param groups so we can switch lr mid-training
    optimiser = torch.optim.Adam(model.parameters(), lr=lr_warmup)

    losses = []
    lr_switched = False # we will switch at the end of annealing

    for epoch in range(warmup_epochs + anneal_epochs):

        #  schedule 
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

        #  forward + backward 
        optimiser.zero_grad()
        loss = energy_loss(model, x_t, y_t, kl_scale=kl_scale)
        loss.backward()
        optimiser.step()

        losses.append(float(loss.detach()))

    return losses