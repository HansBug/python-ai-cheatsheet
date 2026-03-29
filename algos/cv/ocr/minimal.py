import numpy as np
import torch
import torch.nn as nn


class TinyTextDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, images):
        return torch.sigmoid(self.backbone(images))


def score_map_to_boxes(score_map, threshold=0.5):
    boxes = []
    binary = (score_map.squeeze(1) > threshold).cpu().numpy()
    for sample in binary:
        ys, xs = np.where(sample)
        if len(xs) == 0:
            boxes.append([])
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        boxes.append([(x1, y1, x2, y2)])
    return boxes


def crop_boxes(images, batch_boxes, out_h=32, out_w=128):
    crops = []
    for image, boxes in zip(images, batch_boxes):
        for x1, y1, x2, y2 in boxes:
            crop = image[:, y1:y2, x1:x2].unsqueeze(0)
            crop = nn.functional.interpolate(
                crop,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            crops.append(crop)

    if not crops:
        return torch.empty(0, 3, out_h, out_w, device=images.device)
    return torch.cat(crops, dim=0)


class TinyTextRecognizer(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_chars)

    def forward(self, text_crops):
        x = self.features(text_crops)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        return self.classifier(x)


def ctc_greedy_decode(logits, blank_id=0):
    best_ids = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    for sequence in best_ids:
        tokens = []
        prev = None
        for token in sequence:
            if token != blank_id and token != prev:
                tokens.append(token)
            prev = token
        decoded.append(tokens)
    return decoded


if __name__ == "__main__":
    detector = TinyTextDetector()
    recognizer = TinyTextRecognizer(num_chars=38)

    images = torch.randn(2, 3, 128, 128)
    score_map = detector(images)
    batch_boxes = score_map_to_boxes(score_map, threshold=0.5)
    crops = crop_boxes(images, batch_boxes)

    if len(crops) > 0:
        logits = recognizer(crops)
        decoded = ctc_greedy_decode(logits)
        print("decoded token ids:", decoded)
    else:
        print("no text regions found")
