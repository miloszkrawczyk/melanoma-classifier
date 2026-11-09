"""Measures what averaging over rotations at inference buys, on validation.

A lesion photographed under a dermatoscope has no top or bottom, so the same
photo turned by a quarter is an equally fair reading of it. The network does
not know that: convolution is not rotation invariant, so it gives four
slightly different answers. Averaging them is test-time augmentation, and it
costs nothing but compute, because no retraining is involved.

Run it to reproduce the table whose chosen row lives in config.TTA_QUARTER_TURNS:

    uv run python -m scripts.measure_tta

Validation, never test. The test set has been used once and this would be a
second time, for a decision that validation can make perfectly well.

The variants here are deliberately wider than the one the project settled on,
because the point is to show what was rejected and why, not only what was kept.
"""

import time

import torch
from sklearn.metrics import average_precision_score

from src.config import CLASSES, MALIGNANT_CLASSES, MALIGNANT_THRESHOLD, MODELS
from src.data import build_dataloaders
from src.device import pick_device
from src.evaluate import is_malignant
from src.train import build_model
from src.weights import load_weights

MALIGNANT_INDEXES = [CLASSES.index(dx) for dx in MALIGNANT_CLASSES]

# Each variant is a list of (quarter turns, mirror) readings to average over.
# Quarter turns and mirrors are exact permutations of pixels: no interpolation,
# no empty corners, nothing invented.
VARIANTS = {
    "no TTA": [(0, False)],
    "four turns": [(turns, False) for turns in range(4)],
    "four turns + mirror": [(turns, m) for turns in range(4) for m in (False, True)],
}

device = pick_device()
settings = MODELS["resnet"]

model = build_model("resnet").to(device)
model.load_state_dict(load_weights(settings["weights"]))
model.eval()

_, val_loader, _ = build_dataloaders(mean=settings["mean"], std=settings["std"])

print(f"resnet, validation set, threshold {MALIGNANT_THRESHOLD}")
print()
print(
    f"{'variant':22}{'recall':>9}{'precision':>11}{'missed':>8}"
    f"{'false alarms':>14}{'accuracy':>10}{'PR-AUC':>9}{'seconds':>9}"
)

for name, readings in VARIANTS.items():
    started = time.perf_counter()
    collected = []
    truths = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            total = None

            for turns, mirror in readings:
                turned = torch.rot90(images, k=turns, dims=[2, 3])
                if mirror:
                    turned = torch.flip(turned, dims=[3])
                probabilities = torch.softmax(model(turned), dim=1)
                total = probabilities if total is None else total + probabilities

            collected.append((total / len(readings)).cpu())
            truths.append(labels)

    seconds = time.perf_counter() - started
    probabilities = torch.cat(collected)
    actual = torch.cat(truths)

    score = probabilities[:, MALIGNANT_INDEXES].sum(dim=1)
    truth = is_malignant(actual)
    flagged = score >= MALIGNANT_THRESHOLD

    caught = int((flagged & truth).sum())
    missed = int((~flagged & truth).sum())
    false_alarms = int((flagged & ~truth).sum())
    accuracy = float((probabilities.argmax(dim=1) == actual).float().mean())
    pr_auc = average_precision_score(truth.numpy(), score.numpy())

    print(
        f"{name:22}{caught / (caught + missed):>8.1%}"
        f"{caught / (caught + false_alarms):>11.1%}{missed:>8}"
        f"{false_alarms:>14}{accuracy:>10.1%}{pr_auc:>9.3f}{seconds:>8.1f}s"
    )
