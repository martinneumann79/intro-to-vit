"""A minimal Vision Transformer (Dosovitskiy et al., 2020) written from scratch."""

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project each to a vector."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        """Create the patch projection.

        A strided convolution with kernel == stride == patch_size is exactly a
        linear projection of each flattened patch.
        """
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, C, H, W) images to (B, num_patches, dim) patch tokens."""
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention that also exposes its attention weights."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        """Create query/key/value and output projections."""
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend every token to every other token; returns the same shape as x."""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        self.last_attention = attn.detach()
        out = self.dropout(attn) @ v
        return self.out(out.transpose(1, 2).reshape(b, n, d))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block: attention + MLP, each with a residual."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        """Create the attention and MLP sub-layers."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """ViT classifier: patch tokens + [CLS] token + learned positions -> transformer."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        """Build the ViT. Defaults give a ~0.8M parameter model for CIFAR-10."""
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (B, num_classes)."""
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.blocks(self.dropout(tokens))
        return self.head(self.norm(tokens[:, 0]))

    def attention_maps(self) -> list[torch.Tensor]:
        """Return the attention weights from the most recent forward pass, per block."""
        return [blk.attn.last_attention for blk in self.blocks]
