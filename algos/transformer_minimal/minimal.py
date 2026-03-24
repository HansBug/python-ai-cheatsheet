import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self_attn_out)

        cross_attn_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        d_ff: int = 64,
        num_layers: int = 2,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_mask = build_causal_mask(tgt_tokens.shape[1], device=tgt_tokens.device)

        src_x = self.src_embedding(src_tokens)
        src_x = self.pos_encoding(src_x)
        memory = self.encoder(src_x, src_key_padding_mask=src_key_padding_mask)

        tgt_x = self.tgt_embedding(tgt_tokens)
        tgt_x = self.pos_encoding(tgt_x)
        hidden = self.decoder(
            tgt_x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_proj(hidden)
        return logits


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    model = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=30,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        max_len=32,
    )

    src_tokens = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [5, 6, 7, 0, 0],
        ],
        dtype=torch.long,
    )
    tgt_tokens = torch.tensor(
        [
            [1, 8, 9, 0],
            [1, 10, 0, 0],
        ],
        dtype=torch.long,
    )

    src_key_padding_mask = src_tokens.eq(0)
    tgt_key_padding_mask = tgt_tokens.eq(0)

    logits = model(
        src_tokens,
        tgt_tokens,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
    )

    print("src_tokens shape:", tuple(src_tokens.shape))
    print("tgt_tokens shape:", tuple(tgt_tokens.shape))
    print("logits shape:", tuple(logits.shape))
    print(logits[0, :2, :6])


if __name__ == "__main__":
    main()
