"""Evaluation only: normalized reconstruction error, factor probes, distortion."""

import numpy as np
import torch

from .losses import sample_pairs


def fit_probe(z_train, factors_train):
    """OLS with intercept, fitted exclusively on training observations."""
    design = np.column_stack((np.asarray(z_train, dtype=np.float64), np.ones(len(z_train))))
    return np.linalg.lstsq(design, factors_train, rcond=None)[0]


def predict_probe(z, coefficients):
    return np.column_stack((z, np.ones(len(z)))) @ coefficients


def factor_r2(target, predicted):
    residual = ((target - predicted) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - residual / np.maximum(total, 1e-30)


def empirical_distortion(z, x_hat, n_pairs=20000, seed=123, delta=1e-8):
    """kappa_emp^95 = p95(r)/p05(r), r = ||dx_hat||/||dz||.

    Endpoints must share one source. A fixed random pair sample makes memory
    O(K), instead of O(N²). Undefined values are None (strict JSON null).
    A collapsed decoder is reported as degenerate, never as well conditioned.
    """
    if not torch.isfinite(z).all() or not torch.isfinite(x_hat).all():
        raise ValueError("Non-finite codes or endpoints in distortion evaluation.")
    rng = torch.Generator(device=z.device).manual_seed(seed)
    i, j = sample_pairs(len(z), n_pairs, device=z.device, generator=rng)
    dz = (z[i] - z[j]).norm(dim=-1)
    dx = (x_hat[i] - x_hat[j]).norm(dim=-1)
    valid = dz > delta
    ratios = dx[valid] / dz[valid]
    info = {"pairs_sampled": n_pairs, "pairs_valid": int(valid.sum()),
            "r_p05": None, "r_p95": None, "kappa95": None, "degenerate": True}
    if len(ratios):
        low, high = torch.quantile(ratios, ratios.new_tensor([0.05, 0.95])).tolist()
        info.update(r_p05=low, r_p95=high)
        if low > 0:
            info.update(kappa95=high / low, degenerate=False)
    return info


@torch.no_grad()
def encode_decode(model, x, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    codes, endpoints = [], []
    for batch in x.split(batch_size):
        z, x_hat = model(batch.to(device))
        codes.append(z.cpu())
        endpoints.append(x_hat.cpu())
    return torch.cat(codes), torch.cat(endpoints)


def evaluate(model, data):
    """All scores on held-out cells; factors only enter the post-hoc probe."""
    z_train, _ = encode_decode(model, data.train)
    z_test, x_hat = encode_decode(model, data.test)
    coefficients = fit_probe(z_train.numpy(), data.factors_train)
    predictions = predict_probe(z_test.numpy(), coefficients)
    r2_t, r2_a = factor_r2(data.factors_test, predictions)
    metrics = {
        "mse_per_feature": float((x_hat - data.test).square().mean()),
        "r2_t": float(r2_t), "r2_a": float(r2_a),
        "distortion": empirical_distortion(z_test, x_hat),
    }
    return metrics, z_test.numpy(), predictions

