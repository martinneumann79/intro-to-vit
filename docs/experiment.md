# The experiment

`compare-models` trains each model on each requested fraction of the CIFAR-10 training set,
evaluates it on the full 10,000-image test set, and writes a summary table.

## Keeping it fair

Everything except the architecture is shared:

| Setting | Value |
|---|---|
| Data | CIFAR-10, normalised with dataset mean/std |
| Augmentation | Random crop (padding 4) + horizontal flip |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 |
| Schedule | 1 epoch linear warmup, then cosine decay to 0 (per step) |
| Loss | Cross-entropy with label smoothing 0.1 |
| Batch size / epochs | 128 / 20 by default |
| Seed | Same seed for weights and for picking the data subset |
| Model size | ~0.81M (ViT) vs ~0.70M (CNN) parameters |

## Running it

```bash
make quick             # 1 epoch on 10% of the data — checks everything works
make compare           # both models, full training set   → results/full/
make data-efficiency   # 10%, 25% and 100% of the data    → results/data-efficiency/
make compare EPOCHS=40 # override the number of epochs
```

Or call the CLI directly (`uv run compare-models --help`):

```bash
uv run compare-models --models vit cnn --fractions 0.1 0.5 1.0 --epochs 30 --out-dir results/custom
```

CIFAR-10 (~160 MB) is downloaded to `data/` on the first run. Training uses CUDA or Apple MPS when
available, otherwise the CPU.

## Output

Each run writes to its `--out-dir`:

- `results.md` — the summary table (also printed at the end)
- `results.json` — the same numbers plus per-epoch history (`train_loss`, `train_acc`, `test_acc`,
  `epoch_seconds`) for plotting learning curves

| Column | Meaning |
|---|---|
| best test acc | Highest test accuracy over all epochs |
| final train acc | Training accuracy in the last epoch, measured with augmentation and dropout active; a big gap to test accuracy means overfitting |
| train time (s) | Wall-clock time spent in training epochs (excludes evaluation) |
| inference img/s | Forward-pass throughput on random 256-image batches on the same device |

## What to look for

- **Accuracy gap** between the models on the full data.
- **Data efficiency**: how fast each model's accuracy drops as the training set shrinks. The
  CNN's built-in assumptions about images should make it degrade more gracefully.
- **Train vs. test accuracy**: which model memorises the training set and which generalises.
- **Speed**: parameters are similar, but the work per image is not.

## Sample results

`make data-efficiency` (20 epochs, seed 0) on an Apple M5 Pro (MPS), October 2026. Accuracy is the
best test accuracy over the 20 epochs:

| Training data | CNN | ViT | Gap |
|---|---|---|---|
| 100% (50,000 images) | 90.3% | 77.3% | 13.0 pts |
| 25% (12,500 images) | 81.1% | 61.6% | 19.5 pts |
| 10% (5,000 images) | 68.3% | 49.7% | 18.6 pts |

- **The CNN wins at every data size**, and the gap grows from 13 to about 19 points as training data
  shrinks. That is what its built-in assumptions about images predict.
- **The ViT learns slowly; it does not overfit.** On the full data its train and test accuracy stay
  close (78.5% vs 77.3%) and test accuracy was still rising at epoch 20. The CNN passes 76% test
  accuracy by epoch 5; the ViT is at 75.7% after 15 epochs. It has to learn from data what the CNN
  gets from its architecture, for example that neighbouring pixels belong together.
- **Speed is similar:** 9.4 s (ViT) vs 8.6 s (CNN) per full-data epoch, and about 20k vs 24k
  images/s at inference. With only 65 tokens, the quadratic cost of attention does not matter yet.

## Caveats

- Epochs are fixed across data fractions, so smaller fractions also get fewer optimisation steps.
  Part of a model's drop at 10% data can be under-training rather than lack of data.
- Results come from a single seed; differences of a fraction of a percent are noise.
- These are small models trained from scratch. Large pre-trained ViTs behave very differently.

## Code layout

| File | Contents |
|---|---|
| [models/vit.py](../src/intro_to_vit/models/vit.py) | `PatchEmbedding`, `MultiHeadSelfAttention`, `TransformerBlock`, `VisionTransformer` |
| [models/cnn.py](../src/intro_to_vit/models/cnn.py) | `ResidualBlock`, `SmallResNet` |
| [models/\_\_init\_\_.py](../src/intro_to_vit/models/__init__.py) | `build_model(name)` factory |
| [data.py](../src/intro_to_vit/data.py) | Transforms, CIFAR-10 loaders, reproducible subsetting |
| [training.py](../src/intro_to_vit/training.py) | Device selection, warmup + cosine schedule, train/eval loops, `fit` |
| [metrics.py](../src/intro_to_vit/metrics.py) | Parameter count, inference throughput |
| [compare.py](../src/intro_to_vit/compare.py) | Experiment grid, results table, saving |
| [cli.py](../src/intro_to_vit/cli.py) | `compare-models` command-line entry point |
