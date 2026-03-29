import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, images):
        x = self.features(images)
        x = x.flatten(1)
        return self.proj(x)


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = x.mean(dim=1)
        return self.proj(x)


class TinyCLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.image_encoder = SimpleImageEncoder(embed_dim=embed_dim)
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        image_features = self.image_encoder(images)
        return F.normalize(image_features, dim=-1)

    def encode_text(self, token_ids):
        text_features = self.text_encoder(token_ids)
        return F.normalize(text_features, dim=-1)

    def forward(self, images, token_ids):
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)
        return self.logit_scale.exp() * image_features @ text_features.T


def clip_loss(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.T, labels)
    return 0.5 * (image_to_text + text_to_image)


if __name__ == "__main__":
    model = TinyCLIP(vocab_size=5000, embed_dim=128)
    images = torch.randn(4, 3, 224, 224)
    token_ids = torch.randint(0, 5000, (4, 16))
    logits = model(images, token_ids)
    loss = clip_loss(logits)
    print("logits shape:", logits.shape)
    print("loss:", float(loss))
