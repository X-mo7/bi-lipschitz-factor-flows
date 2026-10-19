"""One training path; the three configurations differ only in loss weights."""

from dataclasses import dataclass, fields
import math
from pathlib import Path

import torch
import yaml

from .losses import (
    endpoint_separation, flow_matching, latent_early_lower_loss,
    latent_upper_loss, state_jacobian_loss,
)
from .models import BFF


@dataclass
class Config:
    name: str = "bff"
    seed: int = 0
    data_seed: int = 0
    genome_seed: int = 42
    n_train: int = 5000
    n_test: int = 2000
    n_genes: int = 100
    latent_dim: int = 2
    hidden_dim: int = 128
    n_steps: int = 16
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.002
    weight_decay: float = 0.00001
    grad_clip: float = 1.0
    geometry_batch: int = 16
    n_probes: int = 1
    n_pairs: int = 256
    lambda_fm: float = 1.0
    lambda_x: float = 100.0
    lambda_upper: float = 1.0
    lambda_early: float = 1.0
    lambda_sep: float = 50.0
    ceiling: float = 1.0
    floor: float = 0.5
    margin: float = 0.5
    epsilon: float = 0.1
    delta: float = 0.0

    def validate(self):
        integer_fields = (
            "n_train", "n_test", "n_genes", "latent_dim", "hidden_dim", "n_steps",
            "epochs", "batch_size", "geometry_batch", "n_probes", "n_pairs",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("seed", "data_seed", "genome_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32).")
        for field in fields(self):
            if field.type is float:
                value = getattr(self, field.name)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{field.name} must be finite and non-negative.")
        if min(self.n_train, self.n_test, self.n_genes, self.batch_size) < 2:
            raise ValueError("Data splits, gene count and batch size must be >= 2.")
        if self.latent_dim > self.n_genes or not 0 < self.epsilon <= 1:
            raise ValueError("Require latent_dim <= n_genes and 0 < epsilon <= 1.")
        if self.learning_rate == 0 or self.grad_clip == 0:
            raise ValueError("learning_rate and grad_clip must be positive.")
        if self.floor > self.ceiling:
            raise ValueError("The local floor must not exceed the ceiling.")
        return self


def load_config(path):
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    unknown = payload.keys() - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return Config(**payload).validate()


def train(config, x_train, *, device="cpu", callback=None):
    """Fit using observations only. Ground-truth factors are not an argument.

    Independent RNG streams keep initialization, batch ordering and FM times
    paired across configurations even when geometric losses are skipped.
    Local penalties use a uniformly sampled subset of each training batch.
    They are evaluated on straight training interpolants, not every ODE state.
    """
    config.validate()
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda; this implementation does not target MPS.")
    if x_train.shape != (config.n_train, config.n_genes):
        raise ValueError("Training observations do not match the configuration.")
    torch.manual_seed(config.seed)
    model = BFF(config.n_genes, config.latent_dim, config.hidden_dim, config.n_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    order_rng = torch.Generator().manual_seed(config.seed + 11)
    streams = {
        name: torch.Generator(device=device).manual_seed(config.seed + offset)
        for name, offset in [("fm", 23), ("early", 37), ("hutch", 41),
                             ("pairs", 53), ("geometry", 67)]
    }
    weights = {"rec": 1.0, "fm": config.lambda_fm, "jx": config.lambda_x,
               "upper": config.lambda_upper, "early": config.lambda_early,
               "sep": config.lambda_sep}
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = dict.fromkeys([*weights, "total"], 0.0)
        order = torch.randperm(len(x_train), generator=order_rng)
        for indices in order.split(config.batch_size):
            x = x_train[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            z, x_hat = model(x)  # Exactly one ODE solve for rec and sep.
            terms = {key: x.new_zeros(()) for key in weights}
            terms["rec"] = (x_hat - x).square().sum(-1).mean()
            t = torch.rand(len(x), device=device, generator=streams["fm"])
            if config.lambda_fm:
                terms["fm"] = flow_matching(model.velocity, x, z, t, model.source)
            if config.lambda_sep:
                terms["sep"] = endpoint_separation(
                    x_hat, z, config.margin, config.delta, config.n_pairs,
                    generator=streams["pairs"],
                )
            if config.lambda_x or config.lambda_upper or config.lambda_early:
                subset = torch.randperm(len(x), device=device, generator=streams["geometry"])
                subset = subset[:config.geometry_batch]
                x_g, z_g, t_g = x[subset], z[subset], t[subset]
                x_t = (1 - t_g[:, None]) * model.source + t_g[:, None] * x_g
                if config.lambda_x:
                    terms["jx"] = state_jacobian_loss(
                        model.velocity, t_g, x_t, z_g, n_probes=config.n_probes,
                        generator=streams["hutch"],
                    )
                if config.lambda_upper:
                    terms["upper"] = latent_upper_loss(
                        model.velocity, t_g, x_t, z_g, config.ceiling,
                    )
                if config.lambda_early:
                    t_e = config.epsilon * torch.rand(
                        len(subset), device=device, generator=streams["early"],
                    )
                    x_e = (1 - t_e[:, None]) * model.source + t_e[:, None] * x_g
                    terms["early"] = latent_early_lower_loss(
                        model.velocity, t_e, x_e, z_g, config.floor,
                    )
            total = sum(weights[key] * term for key, term in terms.items())
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
            for key, value in {**terms, "total": total}.items():
                sums[key] += float(value.detach()) * len(x)
        record = {"epoch": epoch, **{key: value / len(x_train) for key, value in sums.items()}}
        history.append(record)
        if callback is not None:
            callback(record)
    return model, history

