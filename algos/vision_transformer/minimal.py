import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_dim=d_model * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )
        num_tokens = self.patch_embed.num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_state = x[:, 0]
        logits = self.head(cls_state)
        return logits


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        d_model=64,
        num_heads=4,
        num_layers=3,
    )

    images = torch.randn(2, 3, 32, 32)

    patch_tokens = model.patch_embed(images)
    logits = model(images)

    print("image shape:", tuple(images.shape))
    print("patch tokens shape:", tuple(patch_tokens.shape))
    print("logits shape:", tuple(logits.shape))
    print(logits)


if __name__ == "__main__":
    main()
