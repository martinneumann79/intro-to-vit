# How Vision Transformers work

A Vision Transformer (ViT) is the standard text Transformer applied, nearly unchanged, to images.
The only real trick is turning an image into something that looks like a sentence.

## What a Transformer expects

A text Transformer never sees raw characters. Its input is a **sequence of vectors**:

1. Text is split into tokens (words or sub-words).
2. Each token id is looked up in an embedding table → a vector of size `d`.
3. A position embedding is added so the model knows token order.
4. A stack of blocks (self-attention + MLP) lets every token exchange information with every other token.

Nothing in steps 3–4 knows the vectors came from text. Any input you can turn into a
sequence of `d`-dimensional vectors can go through the same layers.

## Turning an image into tokens

ViT changes only step 1–2, the tokenizer and embedding:

| Text Transformer                     | Vision Transformer                                    |
|--------------------------------------|-------------------------------------------------------|
| Split a sentence into words          | Cut the image into fixed-size patches (e.g. 4×4 px)   |
| Vocabulary: a finite set of word ids | No vocabulary: patch pixel values are continuous      |
| Embedding = table lookup by id       | Embedding = linear projection of the flattened pixels |
| Position = index in the sentence    | Position = index of the patch in the grid             |

Concretely, for a 32×32 RGB image with 4×4 patches (the setup in this repo):

- The image becomes an 8×8 grid = **64 patches**.
- Each patch has 4·4·3 = 48 numbers; flatten it and multiply by a learned 48×128 matrix → a
  **128-d token**. (This is usually implemented as a convolution with kernel size = stride = patch
  size, which is mathematically the same thing.)
- The result is 64 tokens of size 128 — to the Transformer, a 64-"word" sentence.

Because there is no discrete vocabulary, the projection plays the role of the embedding table: it
learns which pixel patterns matter and maps them into the same kind of vector space a word
embedding lives in.

## Position and the [CLS] token

Self-attention treats its input as an unordered set, so the patch grid's layout would be lost.
A **learned position embedding** (one vector per patch slot) is added to each token, exactly as in
text. The model is not told the patches form a 2-D grid; it has to learn which positions are
neighbours from data.

Like BERT, ViT prepends an extra learned **[CLS] token**. It carries no image content; through
attention it gathers information from all patches, and its final vector is fed to a small
classification head. (An alternative is to average all patch tokens.)

## Inside the encoder

From here on it is a plain Transformer encoder: each block applies

```
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

In self-attention every patch can look at every other patch from the very first layer, so the
model can relate distant parts of the image immediately. A CNN, by contrast, only sees a small
neighbourhood per layer and widens its view gradually.

## The trade-off

CNNs have built-in assumptions (*inductive biases*): nearby pixels are related, and the same
pattern means the same thing anywhere in the image. ViT drops most of these and must learn them
from data. That makes ViTs data-hungry: on small datasets they usually lose to CNNs, but with
large-scale pre-training (or heavy augmentation) they match or beat them.

## Images in multimodal (text + image) models

The same idea is what lets language models "see". A vision encoder (typically a ViT) turns an
image into patch tokens, a small projection maps them into the language model's embedding space,
and they are inserted into the text token sequence. The language model then attends over image
tokens and word tokens together, with no change to its architecture.
