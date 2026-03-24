import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        return x + self.pe[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        assert head_dim % 2 == 0

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        position = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(position, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class RoPEDemoAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(head_dim=d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        cos, sin = self.rope(seq_len=x.shape[-2], device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return output, weights


def set_identity_weights(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features))


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    x = torch.tensor(
        [
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    print("Input X shape:", tuple(x.shape))
    print(x)

    pe = SinusoidalPositionalEncoding(d_model=4, max_len=16)
    x_with_pe = pe(x)
    print("\nSinusoidal positional encoding result:")
    print(x_with_pe)

    rope = RotaryPositionEmbedding(head_dim=4)
    cos, sin = rope(seq_len=x.shape[0], device=x.device)
    print("\nRoPE cos:")
    print(cos)
    print("RoPE sin:")
    print(sin)

    attn = RoPEDemoAttention(d_model=4)
    set_identity_weights(attn.w_q)
    set_identity_weights(attn.w_k)
    set_identity_weights(attn.w_v)

    rope_out, rope_weights = attn(x)
    print("\nRoPE attention weights:")
    print(rope_weights)
    print("RoPE attention output:")
    print(rope_out)


if __name__ == "__main__":
    main()
