# intro-to-vit

Compare a Vision Transformer (ViT) with a convolutional neural network (CNN) on CIFAR-10.

Both models are small (<1M parameters), written from scratch in PyTorch, and trained with the
same recipe, so the architecture is the only thing that differs.

## Quickstart

```bash
make install           # uv sync
make quick             # 1-epoch smoke run on 10% of the data
make compare           # 20 epochs on full CIFAR-10 → results/full/
make data-efficiency   # 10%, 25% and 100% of the data → results/data-efficiency/
```

CIFAR-10 is downloaded to `data/` on the first run. For custom runs see
`uv run compare-models --help`.

**In short:** trained from scratch for 20 epochs, the CNN reaches 90.3% test accuracy and the ViT
77.3%. The gap widens when training data is reduced. See
[docs/experiment.md](docs/experiment.md#sample-results) for the full results.

## Docs

- [docs/models.md](docs/models.md): how the two architectures work and how they differ
- [docs/experiment.md](docs/experiment.md): experiment setup, metrics, outputs and code layout
- [docs/how-vit-works.md](docs/how-vit-works.md): high-level intro to Vision Transformers and how images become tokens
