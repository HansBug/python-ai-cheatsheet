import torch
import torch.nn as nn


def uniform_sample(video, num_frames):
    total_frames = video.shape[0]
    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long()
    return video[indices]


def stride_sample(video, stride):
    return video[::stride]


class FrameEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, video):
        batch_size, num_frames, channels, height, width = video.shape
        x = video.reshape(batch_size * num_frames, channels, height, width)
        x = self.net(x).flatten(1)
        return x.reshape(batch_size, num_frames, -1)


class TinyVideoClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=64):
        super().__init__()
        self.frame_encoder = FrameEncoder(out_dim=feature_dim)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, video):
        frame_features = self.frame_encoder(video)
        frame_features = frame_features.transpose(1, 2)
        video_feature = self.temporal_pool(frame_features).squeeze(-1)
        return self.head(video_feature)


if __name__ == "__main__":
    video = torch.randn(24, 3, 224, 224)
    sampled = uniform_sample(video, num_frames=8)

    model = TinyVideoClassifier(num_classes=10)
    batched_video = sampled.unsqueeze(0)
    logits = model(batched_video)
    print("sampled video shape:", sampled.shape)
    print("logits shape:", logits.shape)
