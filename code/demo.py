from __future__ import annotations

import numpy as np
import torch

from beam_decoder import BeamSearchDecoder
from classifier import SymbolClassifier
from project_paths import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "symbol_classifier.pth"
DATASET_PATH = PROJECT_ROOT / "synthetic_expressions.npz"
SYMBOLS = list(range(10)) + ["+", "-", "="]


def load_demo_model() -> SymbolClassifier:
    model = SymbolClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.size and image.max() > 1.0:
        image = image / 255.0
    return image


def expression_probabilities(model: SymbolClassifier, expression: np.ndarray) -> list[np.ndarray]:
    probabilities = []
    for position in range(5):
        crop = expression[:, position * 28 : (position + 1) * 28]
        tensor = torch.tensor(crop).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(tensor)
            probabilities.append(torch.softmax(logits, dim=1).squeeze(0).numpy())
    return probabilities


def main() -> None:
    rng = np.random.default_rng(0)
    data = np.load(DATASET_PATH)
    image = normalize_image(data["images"][0])
    truth = data["labels"][0]

    sigma = 0.4
    noisy_image = np.clip(image + rng.normal(0.0, sigma, image.shape), 0.0, 1.0)

    model = load_demo_model()
    probabilities = expression_probabilities(model, noisy_image)
    baseline = [int(np.argmax(probability)) for probability in probabilities]

    decoder = BeamSearchDecoder(beam_width=20)
    neuro_symbolic, trace = decoder.decode(probabilities)

    print("True expression:", truth.tolist())
    print("Noise level:", sigma)
    print("Greedy CNN prediction:", baseline)
    print("Greedy correct:", bool(np.array_equal(baseline, truth)))
    print("Beam prediction:", neuro_symbolic)
    print("Beam correct:", bool(np.array_equal(neuro_symbolic, truth)))
    print("Beam trace:")
    for step in trace:
        print(
            f"  position {step['position']}: "
            f"kept={step['kept_count']}, pruned={step['pruned_count']}"
        )

    print("First-symbol probabilities above 0.01:")
    for index, probability in enumerate(probabilities[0]):
        if probability > 0.01:
            print(f"  {SYMBOLS[index]}: {probability:.3f}")


if __name__ == "__main__":
    main()
