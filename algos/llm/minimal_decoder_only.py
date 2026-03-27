from __future__ import annotations

import torch
import torch.nn as nn


class SimpleTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        tokens = text.strip().split()
        ids = [self.token_to_id[token] for token in tokens]
        if add_bos:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_token[idx] for idx in ids)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = build_causal_mask(seq_len=seq_len, device=x.device)

        h = self.ln_1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        h = self.ln_2(x)
        x = x + self.ffn(h)
        return x


class TinyDecoderOnlyLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        d_ff: int = 64,
        num_layers: int = 2,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

        # Weight tying is common in LLMs: input embedding and output projection
        # share the same vocabulary matrix.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def print_topk(logits: torch.Tensor, tokenizer: SimpleTokenizer, k: int = 5) -> None:
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=k)
    for token_id, prob in zip(indices.tolist(), values.tolist()):
        print(f"  {tokenizer.id_to_token[token_id]:>8s} : prob={prob:.4f}")


def generate_greedy(
    model: TinyDecoderOnlyLM,
    input_ids: torch.Tensor,
    tokenizer: SimpleTokenizer,
    max_new_tokens: int,
) -> torch.Tensor:
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token_id], dim=1)

        print(f"\nStep {step + 1}:")
        print("  full logits shape:", tuple(logits.shape))
        print("  last-token logits shape:", tuple(next_token_logits.shape))
        print("  chosen token:", tokenizer.decode(next_token_id[0].tolist()))
        print("  generated so far:", tokenizer.decode(generated[0].tolist()))

        if int(next_token_id.item()) == tokenizer.eos_token_id:
            break

    return generated


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    tokenizer = SimpleTokenizer(
        [
            "<bos>",
            "<eos>",
            "I",
            "like",
            "eat",
            "apples",
            "bananas",
            "today",
            "you",
            ".",
        ]
    )

    model = TinyDecoderOnlyLM(
        vocab_size=len(tokenizer),
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        max_len=32,
    )

    prompt = "I like"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    logits = model(input_ids)

    print("This is an untrained toy model. The numbers are only for data-flow demo.")
    print("Prompt ids:", input_ids.tolist())
    print("Prompt tokens:", tokenizer.decode(input_ids[0].tolist()))
    print("Input shape:", tuple(input_ids.shape))
    print("Logits shape:", tuple(logits.shape))

    for pos in range(input_ids.shape[1]):
        prefix = tokenizer.decode(input_ids[0, : pos + 1].tolist())
        print(f"\nPosition {pos}:")
        print("  prefix:", prefix)
        print("  this position predicts the token after the prefix above")
        print_topk(logits[0, pos], tokenizer, k=5)

    generated = generate_greedy(
        model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        max_new_tokens=4,
    )
    print("\nFinal generated tokens:")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
