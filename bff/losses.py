"""The flow-matching objective and four geometric penalties of paper §4.

All functions return differentiable scalars. Jacobian penalties require
sample-independent networks (no BatchNorm or other batch mixing).
"""

import torch
from torch.func import jacfwd, jvp, vmap


def flow_matching(velocity, x, z, t, source):
    """L_FM on straight paths from the same fixed source as the decoder."""
    interpolant = (1 - t[:, None]) * source + t[:, None] * x
    target = x - source
    return (velocity(t, interpolant, z) - target).square().sum(-1).mean()


def state_jacobian_loss(velocity, t, x, z, *, n_probes=1, generator=None):
    """L_Jx = E ||dv/dx||_F², estimated without forming an N x N matrix.

    For Rademacher probes r, E ||Jr||² = ||J||_F². Forward-mode JVPs
    remain differentiable with respect to network parameters and codes.
    """
    if n_probes < 1:
        raise ValueError("n_probes must be positive.")
    estimates = []
    for _ in range(n_probes):
        probe = torch.randint(0, 2, x.shape, device=x.device, generator=generator)
        probe = probe.to(x.dtype) * 2 - 1
        _, tangent = jvp(lambda state: velocity(t, state, z), (x,), (probe,))
        estimates.append(tangent.square().sum(-1).mean())
    return torch.stack(estimates).mean()


def latent_singular_values(velocity, t, x, z):
    """Singular values of dv/dz for each sample, in descending order.

    Forward-mode AD builds only the thin N x d Jacobian (d is small).
    Direct SVD avoids squaring the condition number via a Gram matrix.
    """
    def spectrum(time, state, code):
        def field(local_code):
            return velocity(time[None], state[None], local_code[None])[0]
        return torch.linalg.svdvals(jacfwd(field)(code))

    return vmap(spectrum)(t, x, z)


def latent_upper_loss(velocity, t, x, z, ceiling):
    """L_Jz^u: penalize excessive local expansion in latent directions."""
    largest = latent_singular_values(velocity, t, x, z)[:, 0]
    return torch.relu(largest - ceiling).square().mean()


def latent_early_lower_loss(velocity, t_early, x_early, z, floor):
    """L_Jz,early^ell; callers sample t_early uniformly on [0, epsilon]."""
    smallest = latent_singular_values(velocity, t_early, x_early, z)[:, -1]
    return torch.relu(floor - smallest).square().mean()


def sample_pairs(n_samples, n_pairs, *, device, generator=None):
    """Uniform ordered non-self pairs, sampled WITH replacement in O(K).

    Repeated pairs are allowed. The sample mean estimates the same pairwise
    expectation without allocating all B(B-1) possible pairs.
    """
    if n_samples < 2 or n_pairs < 1:
        raise ValueError("Need at least two samples and one pair.")
    i = torch.randint(n_samples, (n_pairs,), device=device, generator=generator)
    j = torch.randint(n_samples - 1, (n_pairs,), device=device, generator=generator)
    return i, j + (j >= i).long()


def endpoint_separation(x_hat, z, margin, delta=0.0, n_pairs=256, *, generator=None):
    """L_sep, Eq. (5). Reuse endpoints from the reconstruction forward pass.

    Both endpoints MUST have been decoded from the same source. Average
    over all sampled pairs, including pairs zeroed by the latent-distance
    gate. Gradients flow through both distances, as in the paper.
    """
    if len(z) < 2:
        return x_hat.sum() * 0 + z.sum() * 0
    i, j = sample_pairs(len(z), n_pairs, device=z.device, generator=generator)
    dz = (z[i] - z[j]).norm(dim=-1)
    dx = (x_hat[i] - x_hat[j]).norm(dim=-1)
    return (torch.relu(margin * dz - dx).square() * (dz > delta)).mean()

