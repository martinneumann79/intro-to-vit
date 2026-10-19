"""Land-use mapping with a Vision Transformer on Copernicus Sentinel-2 images.

A pretrained ViT is fine-tuned on EuroSAT (Sentinel-2 patches in ten land-use classes) and then
classifies a fresh Sentinel-2 scene, fetched through Sentinel Hub on the Copernicus Data Space
Ecosystem, patch by patch.
"""
