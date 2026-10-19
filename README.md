# Bi-Lipschitz Factor Flows

PyTorch implementation accompanying **[Bi-Lipschitz Factor Flows: Regularizing the Latent Geometry of Flow-Matching Decoders](docs/paper.pdf)**.

See the paper for the method, theoretical analysis and experimental results.
This repository includes the core model, a synthetic gene-expression example and the cancellation counterexample.
Configurations are illustrative and are not intended to reproduce the paper's numerical results exactly.

```bash
python -m pip install -e .
python examples/train_genes.py --config configs/bff.yaml --smoke
```

[Implementation notes](docs/scope.md) · [License](LICENSE) · [Credits](NOTICE)
