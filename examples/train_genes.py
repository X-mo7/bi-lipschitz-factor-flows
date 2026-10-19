"""Train, evaluate and plot one configuration. Run from the repository root."""

import argparse
from dataclasses import asdict, replace
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bff.metrics import evaluate
from bff.synthetic import make_genes
from bff.train import load_config, train


def plot_run(output, z, truth, predicted, history, name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    if z.shape[1] == 2:
        points = axes[0].scatter(z[:, 0], z[:, 1], c=truth[:, 1], s=7,
                                 cmap="viridis", alpha=0.8, rasterized=True)
        axes[0].set(xlabel="$z_1$", ylabel="$z_2$", title="Held-out codes")
    else:
        points = axes[0].scatter(predicted[:, 1], predicted[:, 0], c=truth[:, 1], s=7,
                                 cmap="viridis", alpha=0.8, rasterized=True)
        axes[0].set(xlabel="Predicted a", ylabel="Predicted t", title="Training-fitted OLS projections")
    fig.colorbar(points, ax=axes[0], label="Factor a")
    axes[1].scatter(truth[:, 1], predicted[:, 1], s=7, alpha=0.5, color="#176c72")
    axes[1].plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="True a", ylabel="Predicted a", title="Probe fitted on training cells")
    axes[2].plot([r["epoch"] for r in history], [r["rec"] for r in history], color="#176c72")
    axes[2].set(xlabel="Epoch", ylabel="Mean squared L2 error", title="Training reconstruction")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{name} — results from this run", fontsize=13)
    fig.savefig(output / "diagnostics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bff.yaml")
    parser.add_argument("--output", default=None, help="New or empty directory; defaults to outputs/<name>_seed<seed>.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads.")
    parser.add_argument("--smoke", action="store_true", help="Tiny two-epoch execution check, not a benchmark.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.smoke:
        config = replace(config, n_train=64, n_test=32, hidden_dim=16, n_steps=4,
                         epochs=2, batch_size=32, geometry_batch=8)
    config.validate()
    suffix = "_smoke" if args.smoke else ""
    output = Path(args.output or f"outputs/{config.name}_seed{config.seed}{suffix}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}. Choose a new --output path.")
    output.mkdir(parents=True, exist_ok=True)
    data = make_genes(config.n_train, config.n_test, config.n_genes,
                      genome_seed=config.genome_seed, data_seed=config.data_seed)

    def report(row):
        if row["epoch"] == 1 or row["epoch"] % 10 == 0 or row["epoch"] == config.epochs:
            print(f"epoch {row['epoch']:4d} | total {row['total']:.5f} | rec {row['rec']:.5f}", flush=True)

    model, history = train(config, data.train, device=args.device, callback=report)
    metrics, z, predictions = evaluate(model, data)
    config_payload = {**asdict(config), "device": args.device, "smoke": args.smoke,
                      "python_version": platform.python_version(), "torch_version": torch.__version__,
                      "numpy_version": np.__version__, "cpu_threads": args.threads}
    for filename, payload in [("config.json", config_payload), ("history.json", history),
                              ("metrics.json", metrics)]:
        (output / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    torch.save({"state_dict": model.cpu().state_dict(),
                "model_config": {"input_dim": config.n_genes, "latent_dim": config.latent_dim,
                                 "hidden_dim": config.hidden_dim, "n_steps": config.n_steps}},
               output / "model.pt")
    np.savez(output / "preprocessing.npz", mean=data.mean, scale=data.scale)
    np.savez(output / "latent_test.npz", z=z, factors=data.factors_test, predictions=predictions)
    plot_run(output, z, data.factors_test, predictions, history, config.name)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved run to {output.resolve()}")


if __name__ == "__main__":
    main()

