from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyYOLOHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.box_conv = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.obj_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.cls_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat = F.silu(self.stem(feat))
        box_reg = self.box_conv(feat)
        objectness = self.obj_conv(feat)
        class_logits = self.cls_conv(feat)

        pred = torch.cat([box_reg, objectness, class_logits], dim=1)
        return pred.permute(0, 2, 3, 1).contiguous()


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0.0)
    inter_h = (y2 - y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou(boxes[current : current + 1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    all_keep = []
    for class_id in labels.unique(sorted=True):
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
        class_keep = nms(boxes[class_indices], scores[class_indices], iou_threshold)
        all_keep.append(class_indices[class_keep])

    keep = torch.cat(all_keep)
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return keep


def decode_predictions(
    raw_pred: torch.Tensor,
    stride: int,
    conf_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width, channels = raw_pred.shape
    num_classes = channels - 5

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)

    box_offsets = raw_pred[..., 0:2]
    box_scales = raw_pred[..., 2:4]
    objectness = torch.sigmoid(raw_pred[..., 4])
    class_probs = torch.sigmoid(raw_pred[..., 5 : 5 + num_classes])

    centers = (torch.sigmoid(box_offsets) + grid) * stride
    wh = torch.exp(box_scales) * stride
    boxes = cxcywh_to_xyxy(torch.cat([centers, wh], dim=-1))

    best_class_probs, labels = class_probs.max(dim=-1)
    scores = objectness * best_class_probs

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    objectness = objectness.reshape(-1)

    keep = scores >= conf_threshold
    return boxes[keep], scores[keep], labels[keep], objectness[keep]


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    num_classes = 2
    class_names = ["car", "person"]
    stride = 32

    head = TinyYOLOHead(in_channels=16, hidden_channels=16, num_classes=num_classes)
    feat = torch.randn(1, 16, 4, 4)
    raw_head_output = head(feat)
    print("Tiny YOLO head output shape [B, H, W, 5 + C]:", tuple(raw_head_output.shape))

    toy_raw = torch.full((4, 4, 5 + num_classes), fill_value=-8.0)

    # Two nearby car predictions that both describe the same object.
    toy_raw[1, 1, 0:4] = torch.tensor([0.2, 0.3, 0.8, 0.1])
    toy_raw[1, 1, 4] = 6.0
    toy_raw[1, 1, 5:] = torch.tensor([5.5, -5.0])

    toy_raw[1, 2, 0:4] = torch.tensor([-2.6, 0.3, 0.8, 0.1])
    toy_raw[1, 2, 4] = 5.2
    toy_raw[1, 2, 5:] = torch.tensor([5.0, -4.0])

    # A person prediction in another region.
    toy_raw[2, 0, 0:4] = torch.tensor([0.4, -0.2, 0.2, 0.0])
    toy_raw[2, 0, 4] = 5.8
    toy_raw[2, 0, 5:] = torch.tensor([-5.0, 5.5])

    # Another distant car prediction that should survive NMS.
    toy_raw[3, 3, 0:4] = torch.tensor([0.1, 0.2, 0.6, -0.1])
    toy_raw[3, 3, 4] = 5.0
    toy_raw[3, 3, 5:] = torch.tensor([4.8, -5.0])

    boxes, scores, labels, objectness = decode_predictions(
        toy_raw,
        stride=stride,
        conf_threshold=0.25,
    )
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)

    print("\nDecoded predictions before NMS:")
    for i in range(boxes.shape[0]):
        print(
            f"  idx={i}, class={class_names[int(labels[i])]}, "
            f"obj={objectness[i]:.3f}, score={scores[i]:.3f}, box={boxes[i].tolist()}"
        )

    print("\nPredictions kept after class-wise NMS:")
    for idx in keep.tolist():
        print(
            f"  idx={idx}, class={class_names[int(labels[idx])]}, "
            f"score={scores[idx]:.3f}, box={boxes[idx].tolist()}"
        )


if __name__ == "__main__":
    main()
