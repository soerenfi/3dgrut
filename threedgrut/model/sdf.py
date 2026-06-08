import torch
import torch.nn as nn


class SDFMLP(nn.Module):
    """Small MLP that maps 3D positions to signed distance values.

    Trained alongside Gaussians via:
      - pull loss: SDF≈0 at high-opacity Gaussian centers (learns where surfaces are)
      - eikonal loss: |∇SDF|=1 everywhere (enforces valid, smooth distance field)
      - push loss: moves Gaussian centers toward the SDF zero level-set (after warmup)

    No external depth or normal supervision — the SDF learns purely from the
    Gaussian opacity distribution and the eikonal constraint.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Softplus(beta=100),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=100),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
