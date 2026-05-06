"""Train the 13-class symbol CNN used by the equation decoder.

The classifier is trained on MNIST training digits plus deterministic rendered
operator glyphs. The equation-level evaluation set is generated separately by
``dataset.py`` from MNIST test digits.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset

from project_paths import PROJECT_ROOT


SEED = 0
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "symbol_classifier.pth"
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
NUM_OPERATOR_COPIES = 100


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_operator_image(symbol: str, size: tuple[int, int] = (28, 28)) -> np.ndarray:
    """Render one operator symbol as a 28x28 grayscale image."""
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    if symbol == "+":
        draw.line((14, 6, 14, 22), fill=255, width=2)
        draw.line((6, 14, 22, 14), fill=255, width=2)
    elif symbol == "-":
        draw.line((6, 14, 22, 14), fill=255, width=2)
    elif symbol == "=":
        draw.line((6, 12, 22, 12), fill=255, width=2)
        draw.line((6, 16, 22, 16), fill=255, width=2)
    else:
        raise ValueError(f"unsupported operator symbol: {symbol}")
    return np.asarray(image)


class SymbolClassifier(nn.Module):
    """Compact CNN that maps one 28x28 crop to 13 symbol classes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 13)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def build_training_tensors() -> TensorDataset:
    """Build the symbol-classification training dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transform,
    )

    digit_images = mnist_train.data.numpy().astype(np.float32) / 255.0
    digit_labels = mnist_train.targets.numpy()

    plus_img = create_operator_image("+").astype(np.float32) / 255.0
    minus_img = create_operator_image("-").astype(np.float32) / 255.0
    equals_img = create_operator_image("=").astype(np.float32) / 255.0

    op_images = []
    op_labels = []
    for _ in range(NUM_OPERATOR_COPIES):
        op_images.extend([plus_img, minus_img, equals_img])
        op_labels.extend([10, 11, 12])

    all_images = np.concatenate([digit_images, np.asarray(op_images)], axis=0)
    all_labels = np.concatenate([digit_labels, np.asarray(op_labels)], axis=0)

    image_tensor = torch.tensor(all_images).unsqueeze(1)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    return TensorDataset(image_tensor, label_tensor)


def train_model(seed: int = SEED) -> SymbolClassifier:
    """Train and return the symbol classifier."""
    set_seed(seed)
    dataset = build_training_tensors()
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = SymbolClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        last_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, loss={last_loss:.6f}")

    return model


def main() -> None:
    """Train the classifier and write the checkpoint."""
    model = train_model()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved {MODEL_PATH} (seed={SEED})")


if __name__ == "__main__":
    main()
