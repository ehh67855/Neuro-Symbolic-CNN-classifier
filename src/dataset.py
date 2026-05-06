"""Generate the fixed-seed synthetic equation evaluation set.

The saved artifact, ``synthetic_expressions.npz``, contains 5,000 expressions
with the fixed layout ``d1 op d2 = d3``. Digits are sampled from the MNIST test
split so the evaluation expressions are held out from classifier training.
Operators are deterministic rendered glyphs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from project_paths import PROJECT_ROOT


SEED = 0
NUM_SAMPLES = 5000
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "synthetic_expressions.npz"
MNIST_SPLIT = "test"
SYMBOL_TO_CLASS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "-": 11,
    "=": 12,
}


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_digit_pool(split: str = MNIST_SPLIT) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST digits from the requested split."""
    transform = transforms.Compose([transforms.ToTensor()])
    train = split == "train"
    mnist = torchvision.datasets.MNIST(
        root=str(DATA_ROOT),
        train=train,
        download=True,
        transform=transform,
    )
    return mnist.data.numpy(), mnist.targets.numpy()


def generate_expressions(
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
    split: str = MNIST_SPLIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate expression images and five-symbol labels deterministically."""
    set_seed(seed)
    digit_images, digit_labels = load_digit_pool(split)
    plus_img = create_operator_image("+")
    minus_img = create_operator_image("-")
    equals_img = create_operator_image("=")

    expressions = []
    labels = []
    for _ in range(num_samples):
        op = random.choice(["+", "-"])
        if op == "+":
            left = random.randint(0, 4)
            right = random.randint(0, 4)
            result = left + right
        else:
            left = random.randint(5, 9)
            right = random.randint(0, 4)
            result = left - right

        left_indices = np.where(digit_labels == left)[0]
        right_indices = np.where(digit_labels == right)[0]
        result_indices = np.where(digit_labels == result)[0]

        left_img = digit_images[random.choice(left_indices)]
        right_img = digit_images[random.choice(right_indices)]
        result_img = digit_images[random.choice(result_indices)]
        op_img = plus_img if op == "+" else minus_img

        expression_img = np.concatenate(
            [left_img, op_img, right_img, equals_img, result_img],
            axis=1,
        )
        expressions.append(expression_img)
        labels.append([left, SYMBOL_TO_CLASS[op], right, SYMBOL_TO_CLASS["="], result])

    return np.asarray(expressions), np.asarray(labels, dtype=np.int64)


def main() -> None:
    """Write the reproducible held-out equation dataset."""
    expressions, labels = generate_expressions()
    np.savez(OUTPUT_PATH, images=expressions, labels=labels)
    print(
        f"Saved {OUTPUT_PATH} with {len(labels)} samples "
        f"(seed={SEED}, mnist_split={MNIST_SPLIT})"
    )


if __name__ == "__main__":
    main()
