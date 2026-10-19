"""Proposition 3: perfect local spectra can coexist with endpoint collapse."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bff.losses import endpoint_separation, latent_singular_values
from bff.ode import euler_solve


def rotating_velocity(t, x, z):
    """v(t,x,z) = R(2*pi*t)z; dv/dx = 0 and both singular values are 1."""
    angle = 2 * math.pi * t
    c, s = angle.cos(), angle.sin()
    return torch.stack((c * z[:, 0] - s * z[:, 1],
                        s * z[:, 0] + c * z[:, 1]), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/cancellation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    z = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=torch.float64)
    source = torch.zeros_like(z)
    endpoints = euler_solve(rotating_velocity, z, source, n_steps=128)
    times = torch.linspace(0, 1, len(z), dtype=z.dtype)
    spectrum = latent_singular_values(rotating_velocity, times, source, z)
    separation = endpoint_separation(endpoints, z, margin=0.5,
                                      generator=torch.Generator().manual_seed(0))
    result = {"local_sigma_min": float(spectrum.min()),
              "local_sigma_max": float(spectrum.max()),
              "largest_endpoint_norm": float(endpoints.norm(dim=-1).max()),
              "endpoint_separation_loss": float(separation)}

    # Analytic integrated paths: S_t = integral_0^t R(2*pi*s) ds.
    t = np.linspace(0, 1, 401)
    angle = 2 * np.pi * t
    sensitivity = np.stack((np.sin(angle), np.cos(angle) - 1,
                            1 - np.cos(angle), np.sin(angle)), axis=-1).reshape(-1, 2, 2)
    sensitivity /= 2 * np.pi
    paths = np.einsum("tij,bj->bti", sensitivity, z.numpy())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    colors = ["#176c72", "#ce6a32", "#6978b8", "#ba5e7b"]
    for code, path, color in zip(z.numpy(), paths, colors):
        axes[0].plot(path[:, 0], path[:, 1], color=color, linewidth=2,
                     label=f"z = ({code[0]:.0f}, {code[1]:.0f})")
        axes[0].scatter(*path[len(t) // 2], color=color, s=20)
    axes[0].scatter([0], [0], color="#172c40", marker="X", s=80, zorder=5,
                    label="Shared start and endpoint")
    axes[0].set(xlabel="$X_1$ coordinate", ylabel="$X_2$ coordinate", aspect="equal",
                title="Different codes, the same endpoint")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    axes[1].plot(t, np.ones_like(t), color="#ce6a32", linestyle="--", linewidth=2,
                 label=r"Local $\sigma_{\min}(\partial_z v)=1$")
    axes[1].plot(t, np.abs(np.sin(np.pi * t)) / np.pi, color="#176c72", linewidth=2.5,
                 label=r"Integrated $\sigma_{\min}(S_t)$")
    axes[1].set(xlabel="Time t", ylabel="Smallest singular value", ylim=(-0.04, 1.14),
                title="Local control does not prevent cancellation")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "cancellation.png", dpi=180)
    plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved figure to {(output / 'cancellation.png').resolve()}")


if __name__ == "__main__":
    main()
