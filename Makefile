.PHONY: install compare data-efficiency quick download robustness robustness-models robustness-v2 demo clean

EPOCHS ?= 20
ROBUSTNESS_PERTURBATIONS = noise salt-pepper blur occlusion occlusion-shifted

# Install dependencies into .venv
install:
	uv sync

# --- Part 1: small ViT vs CNN trained from scratch on CIFAR-10 ---

# Train ViT and CNN on full CIFAR-10 and compare them (results/full/)
compare:
	uv run compare-models --epochs $(EPOCHS) --out-dir results/full

# Same comparison on 10%, 25% and 100% of the training data (results/data-efficiency/)
data-efficiency:
	uv run compare-models --epochs $(EPOCHS) --fractions 0.1 0.25 1.0 --out-dir results/data-efficiency

# One-epoch smoke run on 10% of the data (results/quick/)
quick:
	uv run compare-models --epochs 1 --fractions 0.1 --out-dir results/quick

# --- Part 2: pretrained ViT vs CNN under noise, blur and occlusion ---

# Download the test images and the two demo models, so the demo also works offline
download:
	uv run robustness-download

# ViT-S/16 vs ResNet-50 on 500 unseen ImageNet images, five corruptions (results/robustness/, ~5 min)
robustness:
	uv run robustness-benchmark --perturbations $(ROBUSTNESS_PERTURBATIONS)

# Same with six models: two ViTs, Swin-T, ConvNeXt-T and two ResNet-50 recipes (results/robustness-models/, ~25 min)
robustness-models:
	uv run robustness-benchmark --models vit cnn vit-base convnext swin resnet50-2015 \
		--perturbations $(ROBUSTNESS_PERTURBATIONS) --out-dir results/robustness-models --examples 0

# Noise on ImageNetV2: 10,000 new images in all 1000 classes; downloads 1.3 GB (results/robustness/imagenetv2/, ~15 min)
robustness-v2:
	uv run robustness-benchmark --dataset imagenetv2 --models vit cnn convnext --perturbations noise --examples 0

# Interactive browser demo on http://127.0.0.1:7860 (run `make robustness` first to show the graphs)
demo:
	uv run robustness-demo

# Remove experiment outputs (keeps the downloaded datasets in data/)
clean:
	rm -rf results
