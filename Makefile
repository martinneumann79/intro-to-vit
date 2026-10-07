.PHONY: install compare data-efficiency quick clean

EPOCHS ?= 20

# Install dependencies into .venv
install:
	uv sync

# Train ViT and CNN on full CIFAR-10 and compare them (results/full/)
compare:
	uv run compare-models --epochs $(EPOCHS) --out-dir results/full

# Same comparison on 10%, 25% and 100% of the training data (results/data-efficiency/)
data-efficiency:
	uv run compare-models --epochs $(EPOCHS) --fractions 0.1 0.25 1.0 --out-dir results/data-efficiency

# One-epoch smoke run on 10% of the data (results/quick/)
quick:
	uv run compare-models --epochs 1 --fractions 0.1 --out-dir results/quick

# Remove experiment outputs (keeps the downloaded dataset in data/)
clean:
	rm -rf results
