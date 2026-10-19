"""Two-factor synthetic expression data (paper Appendix G.3).

The generator is adapted from the research workspace. Ground-truth factors
are returned separately and are never passed to the model or training loss.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GeneData:
    train: torch.Tensor
    test: torch.Tensor
    factors_train: np.ndarray
    factors_test: np.ndarray
    mean: np.ndarray
    scale: float


def make_genes(n_train=5000, n_test=2000, n_genes=100, *, genome_seed=42, data_seed=0):
    """Poisson counts -> log1p -> global training scale -> training centering.

    The genome stays fixed across model seeds. A second, independent random
    stream draws factors and count noise. No per-cell normalization is used.
    """
    if min(n_train, n_test, n_genes) < 2:
        raise ValueError("Need at least two training cells, test cells and genes.")
    genome = np.random.RandomState(genome_seed)
    base = genome.uniform(0.5, 2.0, n_genes)
    amplitude = genome.uniform(200, 400, n_genes)
    center_t = genome.uniform(0, 1, n_genes)
    width_t = genome.uniform(0.05, 0.20, n_genes)
    modulation = genome.uniform(-0.5, 0.5, n_genes)
    n_primary = round(0.6 * n_genes)
    amplitude_a = np.concatenate((
        genome.uniform(0, 10, n_primary),
        genome.uniform(100, 400, n_genes - n_primary),
    ))
    center_a = genome.uniform(0, 1, n_genes)
    width_a = genome.uniform(0.05, 0.20, n_genes)

    rng = np.random.RandomState(data_seed)
    n = n_train + n_test
    factors = np.column_stack((rng.uniform(size=n), rng.uniform(size=n)))
    t, a = factors[:, :1], factors[:, 1:]
    bump_t = np.exp(-(t - center_t) ** 2 / (2 * width_t ** 2))
    bump_a = np.exp(-(a - center_a) ** 2 / (2 * width_a ** 2))
    rates = base + amplitude * bump_t * (1 + modulation * (2 * a - 1))
    rates += amplitude_a * bump_a
    observations = np.log1p(rng.poisson(rates).astype(np.float64))
    scale = max(float(observations[:n_train].std()), 1e-12)
    observations /= scale
    mean = observations[:n_train].mean(axis=0)
    observations -= mean
    x = torch.from_numpy(observations.astype(np.float32))
    return GeneData(x[:n_train], x[n_train:], factors[:n_train],
                    factors[n_train:], mean, scale)

