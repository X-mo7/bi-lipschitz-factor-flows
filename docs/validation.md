# Validation of the public implementation

Checked locally on 2026-10-19, using CPU execution:

| Component | Version |
|---|---|
| Python | 3.11.7 |
| PyTorch | 2.2.2 |
| NumPy | 1.26.4 |
| Matplotlib | 3.8.0 |
| PyYAML | 6.0.1 |
| pytest | 7.4.0 |

## Checks performed

- Editable package build and installation in a temporary virtual environment
  using already-installed dependencies.
- 19 passing pytest cases, covering analytic endpoint derivatives, local
  singular values, penalty gradients, the rotation counterexample,
  non-self pair sampling, degeneracy reporting, factor probes, data
  preprocessing, sample independence and repeatable tiny training runs.
- Successful CLI runs of all three configurations with `--smoke`, including
  evaluation, strict JSON output, checkpoint serialization and plotting.
- Reloading all three saved checkpoints reproduces their evaluation metrics.
- Successful cancellation demonstration: local singular values equal to one
  up to floating-point precision, maximum endpoint norm below 1e-15, and a
  positive endpoint separation penalty.
- Visual inspection of the repository figures and a generated training plot.
- All local Markdown links resolve; the manuscript copy is byte-identical
  to the supplied source PDF.

The smoke runs use 64 training cells, 32 test cells, hidden width 16, four
Euler steps and two epochs. Their scores are not benchmark results and are
not used to support the paper's empirical claims. Full-length public training,
historical result reproduction, fresh dependency downloads and GPU execution
have not been validated here.

The installed Matplotlib stack emitted Pyparsing deprecation warnings during
test collection; all tests completed successfully.

## Repeat the checks

After installing the project from its root directory:

```bash
python -m pytest
python examples/cancellation.py
python examples/train_genes.py --config configs/rec_fm.yaml --smoke
python examples/train_genes.py --config configs/bff.yaml --smoke
python examples/train_genes.py --config configs/bff_no_sep.yaml --smoke
```

For a repeated CLI run, choose a new `--output` directory.
