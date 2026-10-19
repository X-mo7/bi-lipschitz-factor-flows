# Scope and provenance

This is a curated public implementation of the method in
**Bi-Lipschitz Factor Flows: Regularizing the Latent Geometry of Flow-Matching
Decoders**. It was assembled from the available local research workspace,
using the supplied manuscript as the specification for the method.

## Included

- Fixed-source, latent-conditioned ODE decoder with an ELU MLP encoder and
  SiLU velocity network.
- Reconstruction, linear-path flow matching and all four penalties from §4.
- Synthetic gene data and test-set evaluation of reconstruction, factor
  recovery and decoder distance distortion.
- Paired configurations and an independent theoretical counterexample.
- Original manuscript and clearly labeled results/figure from that manuscript.

## Adaptations made for this release

| Research workspace | Public implementation |
|---|---|
| An encoder-geodesic penalty enabled by some defaults | Removed to match the stated BFF objective |
| Early lower penalty disabled in some historical scripts | Enabled explicitly in the full BFF example |
| Multiple model variants, diagnostics and oracle losses | One architecture and the terms in §4 |
| Full pair permutations and pairwise distance matrices | O(K) pair sampling with replacement |
| Gram-matrix eigendecomposition for latent spectra | Direct SVD of the thin Jacobian |
| Several normalizations and metrics across experiments | Explicit squared L2 training losses and per-feature test MSE |
| Environment-specific launchers | One portable example script and a package manifest |

Hyperparameters, model width, integration steps and geometric sub-batch size
are explicit public example choices, not reconstructed cluster settings.
Some historical code and configurations are unavailable. The repository has
not been verified to regenerate the numerical table in the paper.

## Experimental coverage

| Experiment | Public executable pipeline | Paper results documented |
|---|---|---|
| Synthetic genes | Yes | Yes |
| Synthetic images | No | Yes |
| Tabula Muris Senis | No | Yes |
| Rotation counterexample | Yes; analytic and numerical | Proposition 3 |

The available historical image trainer samples different source images per
observation. That is a different setting from the fixed-source decoder in
the manuscript, so it is not exported as an implementation of that method.
Image support would require aligning decoding and pairwise evaluation around
one fixed source.

The TMS work includes substantial cohort selection, fold-specific
preprocessing, calibration and cluster orchestration. It is outside this
minimal release. An extension should preserve the manuscript's separation by
animal groups, training-only preprocessing and post-hoc age evaluation. No
TMS data, frozen splits or reconstructed training artifacts are implied to be
available here.

## How to read the reported results

The numerical results in the supplied manuscript and the extracted image
comparison come from the original research experiments. Outputs from the public
examples are written to `outputs/` and are labeled as results from that run.
Smoke-test scores should not be interpreted as benchmark performance.

The public metric is explicitly named `mse_per_feature`. Some historical
scripts used a mean of summed squared errors under an MSE label; comparisons
with historical artifacts require checking their normalization and split.
The manuscript table is reproduced as reported, without silently rescaling
or recalculating its values.

## Source map

The compact modules were adapted from these research components:

- `models/flow_autoencoder.py`, `models/velocity_field.py`, and the MLP encoder
  in `models/coders.py` -> `bff/models.py`.
- The forward Euler routine in `utils/ode_solvers.py` -> `bff/ode.py`.
- Selected functions in `utils/flow_regs.py` and `utils/jacobian_utils.py`
  -> `bff/losses.py`.
- The training loop in `trainers/flow_trainer.py` -> `bff/train.py`.
- Synthetic single-cell generation in `data/flow_datasets.py`
  -> `bff/synthetic.py`.
- The OLS probe in `run_singlecell_flow.py` and decoder distortion in
  `utils/measures.py` -> `bff/metrics.py`.

These are provenance references, not runtime dependencies. The new folder
does not import from or require the surrounding research workspace.

