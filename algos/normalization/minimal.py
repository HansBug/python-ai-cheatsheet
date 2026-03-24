import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 0.0, -10.0, 0.0],
        ],
        dtype=torch.float32,
    )
    print("Input X shape:", tuple(x.shape))
    print(x)

    layer_norm = LayerNorm(d_model=x.shape[-1])
    rms_norm = RMSNorm(d_model=x.shape[-1])

    ln_out = layer_norm(x)
    rms_out = rms_norm(x)

    print("\nLayerNorm output:")
    print(ln_out)
    print("LayerNorm mean over hidden dim:")
    print(ln_out.mean(dim=-1))

    print("\nRMSNorm output:")
    print(rms_out)
    print("RMSNorm mean over hidden dim:")
    print(rms_out.mean(dim=-1))


if __name__ == "__main__":
    main()
