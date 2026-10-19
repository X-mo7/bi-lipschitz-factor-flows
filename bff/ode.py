"""Fixed-step differentiable integration; no external ODE dependency."""


def euler_solve(velocity, z, source, n_steps: int = 16):
    """Integrate dX/dt = v(t, X, z) on [0, 1] using forward Euler.

    Gradients propagate through every step. This is a discrete approximation
    of the continuous decoder in the paper, not an exact ODE solution.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    dt = 1.0 / n_steps
    x = source
    for step in range(n_steps):
        t = x.new_full((z.shape[0],), step * dt)
        x = x + dt * velocity(t, x, z)
    return x

