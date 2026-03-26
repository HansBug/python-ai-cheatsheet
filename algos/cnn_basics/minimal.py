import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))

    batch_size, _, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h = (in_h - kernel_h) // stride_h + 1
    out_w = (in_w - kernel_w) // stride_w + 1

    out = torch.zeros(
        batch_size,
        out_channels,
        out_h,
        out_w,
        device=x.device,
        dtype=x.dtype,
    )

    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    w_start = j * stride_w
                    window = x[
                        b,
                        :,
                        h_start : h_start + kernel_h,
                        w_start : w_start + kernel_w,
                    ]
                    out[b, oc, i, j] = (window * weight[oc]).sum()
                    if bias is not None:
                        out[b, oc, i, j] += bias[oc]

    return out


class LeNetStyleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class AlexNetStem(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZFNetStem(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")

        branch_channels = out_channels // 4
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch_pool(x),
        ]
        return torch.cat(outputs, dim=1)


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False)

    x = torch.randn(1, 2, 5, 5)
    weight = torch.randn(3, 2, 3, 3)
    bias = torch.randn(3)
    naive_out = naive_conv2d(x, weight, bias=bias, stride=1, padding=1)
    torch_out = F.conv2d(x, weight, bias=bias, stride=1, padding=1)

    print("naive conv output shape:", tuple(naive_out.shape))
    print("torch conv output shape:", tuple(torch_out.shape))
    print("max difference:", float((naive_out - torch_out).abs().max()))

    gray_images = torch.randn(2, 1, 28, 28)
    lenet = LeNetStyleCNN(num_classes=10)
    print("LeNet-style logits shape:", tuple(lenet(gray_images).shape))

    images = torch.randn(2, 3, 64, 64)
    alex_stem = AlexNetStem()
    zf_stem = ZFNetStem()
    vgg_block = VGGBlock(in_channels=64, out_channels=128, num_convs=2)
    inception = InceptionBlock(in_channels=128, out_channels=128)

    stem_out = alex_stem(images)
    zf_out = zf_stem(images)
    vgg_out = vgg_block(stem_out)
    inception_out = inception(vgg_out)

    print("AlexNet stem output shape:", tuple(stem_out.shape))
    print("ZFNet stem output shape:", tuple(zf_out.shape))
    print("VGG block output shape:", tuple(vgg_out.shape))
    print("Inception block output shape:", tuple(inception_out.shape))


if __name__ == "__main__":
    main()
