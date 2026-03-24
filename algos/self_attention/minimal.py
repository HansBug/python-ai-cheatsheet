import torch
import torch.nn as nn


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return output, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, d_model = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        num_heads, seq_len, head_dim = x.shape
        return x.permute(1, 0, 2).reshape(seq_len, num_heads * head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        all_outputs = []
        all_weights = []
        for q_i, k_i, v_i in zip(q, k, v):
            scores = q_i @ k_i.transpose(-1, -2) / (q_i.shape[-1] ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)

            weights = torch.softmax(scores, dim=-1)
            all_outputs.append(weights @ v_i)
            all_weights.append(weights)

        concat = self.combine_heads(torch.stack(all_outputs, dim=0))
        output = self.w_o(concat)
        return output, torch.stack(all_weights, dim=0)


def build_causal_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def set_identity_weights(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features))


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    print("Input X shape:", tuple(x.shape))
    print(x)

    d_model = x.shape[-1]
    num_heads = 2

    single_attn = SingleHeadSelfAttention(d_model=d_model)
    set_identity_weights(single_attn.w_q)
    set_identity_weights(single_attn.w_k)
    set_identity_weights(single_attn.w_v)

    single_out, single_weights = single_attn(x)
    print("\nSingle-head attention weights:")
    print(single_weights)
    print("Single-head output:")
    print(single_out)

    causal_mask = build_causal_mask(seq_len=x.shape[0])
    masked_out, masked_weights = single_attn(x, mask=causal_mask)
    print("\nSingle-head causal attention weights:")
    print(masked_weights)
    print("Single-head causal output:")
    print(masked_out)

    multi_attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    set_identity_weights(multi_attn.w_q)
    set_identity_weights(multi_attn.w_k)
    set_identity_weights(multi_attn.w_v)
    set_identity_weights(multi_attn.w_o)

    multi_out, multi_weights = multi_attn(x)
    print("\nMulti-head attention weights shape:", tuple(multi_weights.shape))
    print(multi_weights)
    print("Multi-head output:")
    print(multi_out)


if __name__ == "__main__":
    main()
