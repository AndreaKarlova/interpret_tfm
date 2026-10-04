"""A small CNN feature extractor for the MNIST-C experiment (needs torch).

Kept out of kernel_louis/__init__.py so the rest of the package works without torch.
"""

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    """Conv(1->32) -> pool -> Conv(32->64) -> pool -> Linear(3136->128) -> ReLU -> Linear(128->10).

    The 128-d ReLU output is the feature vector used by every predictor.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.body(x))

    def features(self, x):
        return self.body(x)


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 images (n, 28, 28) -> float tensor (n, 1, 28, 28) with pixels divided by 255."""
    return torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1)


def train_cnn(images: np.ndarray, labels: np.ndarray, seed: int, epochs: int = 5,
              lr: float = 1e-3, batch_size: int = 128, holdout_frac: float = 0.1,
              device: str = "cpu"):
    """Train SmallCNN with Adam and cross-entropy on (clean) images; no augmentation.

    A random holdout_frac of the images is held out to monitor accuracy.
    Returns (frozen model in eval mode, held-out accuracy).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    n_hold = int(round(holdout_frac * len(labels)))
    hold, train = order[:n_hold], order[n_hold:]

    X = images_to_tensor(images)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    model = SmallCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        shuffled = train[rng.permutation(len(train))]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(X[batch].to(device)), y[batch].to(device))
            loss.backward()
            optimiser.step()
        print(f"  epoch {epoch + 1}/{epochs}: last batch loss {loss.item():.3f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        pred = predict_logits(model, images[hold], device).argmax(axis=1)
    return model, float(np.mean(pred == np.asarray(labels)[hold]))


def predict_logits(model: SmallCNN, images: np.ndarray, device: str = "cpu",
                   batch_size: int = 1024) -> np.ndarray:
    """Class logits (n, 10) of the CNN's own head, computed in batches."""
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images_to_tensor(images[start:start + batch_size]).to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def extract_features(model: SmallCNN, images: np.ndarray, device: str = "cpu",
                     batch_size: int = 1024) -> np.ndarray:
    """128-d penultimate-layer features (float64, shape (n, 128)), computed in batches."""
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images_to_tensor(images[start:start + batch_size]).to(device)
            out.append(model.features(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
