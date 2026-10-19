# intro-to-vit

Material for a talk on Vision Transformers (ViT) versus convolutional neural networks (CNN), with two
experiments that show both sides of the trade-off:

1. **Trained from scratch on little data, the CNN wins.** A small ViT and a small CNN (<1M
   parameters each), written from scratch in PyTorch, are trained the same way on CIFAR-10.
2. **Pretrained and equally sized, the ViT copes better with noisy images.** ViT-S/16 and ResNet-50,
   both trained on the same ImageNet images and equally accurate on clean images, are tested on
   images they never saw while noise, blur or occlusion is turned up. A browser demo lets you drag
   the noise level live.

## Quickstart

```bash
make install           # uv sync

# Part 1: training from scratch on CIFAR-10
make quick             # 1-epoch smoke run on 10% of the data
make compare           # 20 epochs on full CIFAR-10 → results/full/
make data-efficiency   # 10%, 25% and 100% of the data → results/data-efficiency/

# Part 2: pretrained models under corruption
make download          # test images and model weights; afterwards everything runs offline
make robustness        # five corruptions, ViT vs CNN → results/robustness/ (~5 min)
make demo              # interactive demo on http://127.0.0.1:7860
```

CIFAR-10, Imagenette and the pretrained weights are downloaded on first use. For custom runs see
`uv run compare-models --help` and `uv run robustness-benchmark --help`.

**In short:** trained from scratch for 20 epochs, the CNN reaches 90.3% test accuracy on CIFAR-10
and the ViT 77.3%. Pretrained on ImageNet, both models recognise 81% of clean test images, but with
Gaussian noise of σ = 0.3 the ViT still gets 62% right and the ResNet-50 43% (39% vs 26% on
ImageNetV2's 10,000 images in all 1000 classes). A modern CNN
(ConvNeXt) copes with noise as well as the ViT, though not with blur, and the training recipe
matters as much as the architecture.
See [docs/experiment.md](docs/experiment.md#sample-results) and
[docs/robustness-demo.md](docs/robustness-demo.md#results).

## Docs

Presentation material (in Swedish):

- [docs/presentation-info.md](docs/presentation-info.md): Transformers, CNN and ViT, how they differ,
  pros and cons, when to use which, and the demo results
- [docs/presentation-plan.md](docs/presentation-plan.md): a 45-minute plan, slide ideas, demo
  script and answers to likely questions
- [docs/vit-in-practice.md](docs/vit-in-practice.md): real projects built on ViT, with links, and
  whether ViT is state of the art

Technical docs:

- [docs/how-vit-works.md](docs/how-vit-works.md): high-level intro to Vision Transformers and how
  images become tokens
- [docs/models.md](docs/models.md): the two from-scratch models and how they differ
- [docs/experiment.md](docs/experiment.md): the CIFAR-10 experiment: setup, metrics, outputs
- [docs/robustness-demo.md](docs/robustness-demo.md): the robustness benchmark and browser demo:
  models, test data, corruptions, results and code layout
