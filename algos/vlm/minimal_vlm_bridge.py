import torch
import torch.nn as nn


def build_prefix_causal_mask(num_visual_tokens, num_text_tokens, device):
    total_tokens = num_visual_tokens + num_text_tokens
    mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)

    # Visual tokens are observed conditions, so they can see each other freely.
    mask[:num_visual_tokens, :num_visual_tokens] = False

    # Text tokens can attend to all visual tokens.
    mask[num_visual_tokens:, :num_visual_tokens] = False

    # Text tokens remain causal among themselves.
    text_causal = torch.triu(
        torch.ones(num_text_tokens, num_text_tokens, dtype=torch.bool, device=device),
        diagonal=1,
    )
    mask[num_visual_tokens:, num_visual_tokens:] = text_causal
    return mask


class PatchEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, width=64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, visual_tokens):
        return self.net(visual_tokens)


class TinyVLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        image_size=224,
        patch_size=16,
        vision_width=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        max_len=512,
    ):
        super().__init__()
        self.vision_encoder = PatchEncoder(
            in_channels=3,
            patch_size=patch_size,
            width=vision_width,
        )
        self.projector = MLPProjector(in_dim=vision_width, out_dim=d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode_image(self, images):
        visual_tokens = self.vision_encoder(images)
        return self.projector(visual_tokens)

    def forward(self, images, input_ids):
        visual_tokens = self.encode_image(images)
        text_tokens = self.token_embedding(input_ids)
        x = torch.cat([visual_tokens, text_tokens], dim=1)

        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        mask = build_prefix_causal_mask(
            num_visual_tokens=visual_tokens.shape[1],
            num_text_tokens=text_tokens.shape[1],
            device=x.device,
        )
        x = self.decoder(x, mask=mask)

        text_hidden = x[:, visual_tokens.shape[1] :]
        text_hidden = self.ln_f(text_hidden)
        return self.lm_head(text_hidden)


if __name__ == "__main__":
    model = TinyVLM(vocab_size=1000)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 16))
    logits = model(images, input_ids)
    print("logits shape:", logits.shape)
