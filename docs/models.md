# The two models

Both models are written from scratch in PyTorch, sized for 32×32 CIFAR-10 images, and kept in the
same size class (under 1M parameters) so the comparison is about *architecture*, not capacity.

## CNN baseline — `SmallResNet` ([cnn.py](../src/intro_to_vit/models/cnn.py))

```
image 3×32×32
  └─ stem: 3×3 conv → BN → ReLU                      32×32×32
  └─ stage 1: 2 × ResidualBlock(32)                  32×32×32
  └─ stage 2: 2 × ResidualBlock(64),  stride 2       64×16×16
  └─ stage 3: 2 × ResidualBlock(128), stride 2      128×8×8
  └─ global average pool → linear                    10 logits
```

A `ResidualBlock` is two 3×3 convolutions with batch norm plus a skip connection
(`relu(body(x) + skip(x))`). Each 3×3 kernel only sees a small neighbourhood and the *same* kernel
slides over the whole image, so the network is built to detect local patterns anywhere. Stacking
layers and downsampling gradually widens what each unit can see. ~0.70M parameters.

## Vision Transformer — `VisionTransformer` ([vit.py](../src/intro_to_vit/models/vit.py))

```
image 3×32×32
  └─ PatchEmbedding: cut into 4×4 patches, project each to 128-d     64 tokens × 128
  └─ prepend learned [CLS] token, add learned position embeddings    65 tokens × 128
  └─ 6 × TransformerBlock (pre-norm):
        x = x + MultiHeadSelfAttention(LayerNorm(x))    4 heads
        x = x + MLP(LayerNorm(x))                       128 → 256 → 128, GELU
  └─ LayerNorm on the [CLS] token → linear                           10 logits
```

The image is treated as a *sequence* of 64 patches. In self-attention each patch computes
query/key/value vectors and gathers information from **every** other patch, weighted by
`softmax(q·kᵀ / √d)`. Nothing in the architecture says that neighbouring patches are related — the
model has to learn spatial structure from the data through the position embeddings. ~0.81M
parameters.

`VisionTransformer.attention_maps()` returns the attention weights of each block from the most recent
forward pass, shaped `(batch, heads, 65, 65)`; row 0 shows what the `[CLS]` token attends to.

## Key differences

| | CNN | ViT |
|---|---|---|
| Core operation | Convolution: fixed local filter, same weights everywhere | Self-attention: input-dependent weights between all patch pairs |
| Built-in assumptions (inductive bias) | Locality, translation equivariance, hierarchy | Almost none beyond the patch grid |
| Receptive field | Grows layer by layer | Global from the first layer |
| Cost as images grow | Linear in number of pixels | Quadratic in number of patches |
| Normalisation | BatchNorm | LayerNorm |
| Summary for classification | Global average pool | `[CLS]` token |
| Data appetite | Learns well from small datasets | Needs lots of data, strong augmentation or pre-training |

The strong built-in assumptions are why the CNN usually wins when training from scratch on a small
dataset like CIFAR-10. The ViT's flexibility pays off at large scale (e.g. pre-training on hundreds
of millions of images), where learned relationships beat hand-designed ones. The
[experiment](experiment.md) lets you see the small-data side of that trade-off directly.
