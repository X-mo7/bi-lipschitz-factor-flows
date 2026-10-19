# From the paper to the code

## 1. A decoder defined by integration

For one fixed source, the map $F:z\mapsto X_1$ is deterministic. Its sensitivity
$S_t=\partial_z X_t$ satisfies

$$
\dot S_t=A(t)S_t+B(t),\quad S_0=0,
\qquad A=\partial_xv,\quad B=\partial_zv.
$$

[`BFF.decode`](../bff/models.py) and [`euler_solve`](../bff/ode.py) implement
the state dynamics. Autograd differentiates the discrete integration when
optimizing the endpoint objectives; training does not explicitly construct
the full endpoint Jacobian.

## 2. The asymmetry between expansion and collapse

The variation-of-constants formula gives

$$
S_1=\int_0^1\Psi(1,s)B(s)\,ds.
$$

If the state and latent Jacobians satisfy uniform spectral bounds $C_x$ and
$C_z$, then for $C_x>0$,

$$
\|S_1\|_2\le\frac{C_z}{C_x}(e^{C_x}-1).
$$

The limit for $C_x=0$ is $C_z$. Pointwise bounds on the smallest singular
value of $B(t)$ do not give the analogous endpoint lower bound: contributions
may cancel. The [rotation example](../examples/cancellation.py) exhibits this
failure without training a network.

## 3. The objective

Training minimizes the batch mean of squared L2 reconstruction errors plus
flow matching and four optional penalties. Reconstruction and flow matching
sum over features; the **reported MSE metric** divides by feature count.

| Quantity | Formula / sampling |
|---|---|
| Reconstruction | Batch mean of squared distance between x and its endpoint |
| Flow matching | t uniform on [0,1], straight interpolant from the fixed source, target velocity x minus source |
| State penalty | Hutchinson estimate of squared Frobenius norm of the state Jacobian |
| Latent upper penalty | Mean squared positive part of sigma_max minus `ceiling` |
| Early lower penalty | Mean squared positive part of `floor` minus sigma_min, with t uniform on [0,`epsilon`] |
| Endpoint separation | Eq. (5), averaged over sampled non-self pairs with the latent-distance gate |

The local losses use interpolants built from a uniformly sampled subset of
the training batch. This is a Monte Carlo objective, not uniform control of
the learned ODE trajectories. Both distances in separation retain gradients,
including the latent distance. A source shared by all decoded pairs is
essential to its interpretation as a property of one map F.

The public full BFF configuration activates all four penalties. The baseline
sets their four weights to zero; the ablation changes only the separation
weight. No encoder regularizer or endpoint-sensitivity oracle is added.

## 4. Data and evaluation

[`make_genes`](../bff/synthetic.py) follows the two-program expression model
in Appendix G.3. Gaussian response bumps in t are amplitude-modulated by a;
a second family of bumps supplies an additional a-dependent contribution.
Poisson observations are transformed by log1p, divided by one scalar fitted
on the training split, and centered by the training mean.

The positive modulation avoids the obvious collapse of the entire a=0 edge
in a purely multiplicative generator. The code does not claim to prove global
injectivity of every randomly drawn finite gene map.

Training receives observations only. The OLS factor probe is fitted on
training codes, then evaluated on held-out cells. Empirical distortion is
measured between decoded endpoints from the same fixed source, not between
noisy observations. The ratio of percentiles is invariant to uniform latent
rescaling; it is not a global bi-Lipschitz certificate.

## 5. Cost and numerical choices

State-Jacobian Hutchinson estimates use Rademacher probes and JVPs. Latent
Jacobians have shape N x d with d small; `jacfwd` and `vmap` build them and a
direct SVD avoids the conditioning loss of forming B-transpose-B. The
geometric sub-batch bounds peak memory. Endpoint separation needs no second
ODE solve, and pair sampling allocates only K indices per side.

The Euler decoder is deliberately transparent. More accurate integration,
alternative architectures or a direct estimator of transported-sensitivity
spread are possible extensions, not claims implemented by this release.

