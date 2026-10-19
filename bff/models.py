"""Deterministic encoder and latent-conditioned velocity field (paper §2)."""

import math

import torch
from torch import nn

from .ode import euler_solve


class VelocityField(nn.Module):
    """v(t, x, z) with smooth activations and no cross-sample interactions."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + input_dim + latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor):
        time = torch.stack((t, torch.sin(2 * math.pi * t),
                            torch.cos(2 * math.pi * t)), dim=-1)
        return self.net(torch.cat((time, x, z), dim=-1))


class BFF(nn.Module):
    """Encoder E(x) and decoder F(z) = X(1), with one fixed source X(0).

    Data are centered using the training mean. The zero source therefore
    corresponds to the training centroid in observation coordinates.
    The model architecture is identical for Rec + FM, BFF and no-separation.
    """

    def __init__(self, input_dim=100, latent_dim=2, hidden_dim=128, n_steps=16):
        super().__init__()
        if not 1 <= latent_dim <= input_dim:
            raise ValueError("Require 1 <= latent_dim <= input_dim.")
        if hidden_dim < 1 or n_steps < 1:
            raise ValueError("hidden_dim and n_steps must be positive.")
        self.n_steps = n_steps
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.velocity = VelocityField(input_dim, latent_dim, hidden_dim)
        self.register_buffer("source", torch.zeros(input_dim))

    def decode(self, z: torch.Tensor):
        source = self.source.expand(z.shape[0], -1)
        return euler_solve(self.velocity, z, source, self.n_steps)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.decode(z)

