"""Finds a better rule for calling a lesion malignant than "pick the top class".

Taking whichever of the seven scores is highest treats every mistake as
costing the same. It does not. Sending someone to a dermatologist who did not
need to go costs an appointment; telling someone with a melanoma that their
mole is fine can cost them their life.

So instead of the top class, this adds up the probability of the three
malignant diagnoses and compares that total against a threshold. Lower the
threshold and the model catches more cancers while raising more false alarms.
This script prints the whole trade-off so the number can be chosen on
evidence rather than taste.

    uv run python -m scripts.tune_threshold cnn
"""

import sys

import torch

from src.config import CLASSES, MALIGNANT_CLASSES, MODELS
from src.data import build_dataloaders
from src.train import build_model, pick_device

MALIGNANT_INDEXES = [CLASSES.index(dx) for dx in MALIGNANT_CLASSES]


def collect_probabilities(model, loader, device):
    """Run the validation set and keep the probabilities, not just the winner."""
    model.eval()
    probabilities = []
    actual = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            # softmax turns the seven raw scores into seven probabilities that
            # add up to 1. The training loss did this internally; here it has
            # to be done by hand, because the numbers themselves are the point.
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            actual.append(labels)

    return torch.cat(probabilities), torch.cat(actual)


def malignant_probability(probabilities):
    """How likely the lesion is malignant, whichever of the three it is.

    A photo scoring 0.3 melanoma, 0.3 basal cell carcinoma and 0.4 mole is
    called a mole by the top-class rule, even though it is 60% malignant.
    Adding the three up is the honest reading.
    """
    return probabilities[:, MALIGNANT_INDEXES].sum(dim=1)


def score_at(threshold, malignant_scores, was_malignant):
    """Recall, precision and the raw counts if this threshold were used."""
    called_malignant = malignant_scores >= threshold

    caught = int((called_malignant & was_malignant).sum())
    missed = int((~called_malignant & was_malignant).sum())
    false_alarms = int((called_malignant & ~was_malignant).sum())

    recall = caught / (caught + missed) if caught + missed else 0.0
    precision = caught / (caught + false_alarms) if caught + false_alarms else 0.0

    return recall, precision, missed, false_alarms


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    model = build_model(name)
    settings = MODELS[name]

    device = pick_device()
    _, val_loader, _ = build_dataloaders(mean=settings["mean"], std=settings["std"])

    model = model.to(device)
    model.load_state_dict(torch.load(settings["weights"], map_location=device))

    probabilities, actual = collect_probabilities(model, val_loader, device)
    malignant_scores = malignant_probability(probabilities)
    was_malignant = torch.tensor([int(i) in MALIGNANT_INDEXES for i in actual])

    print(f"{name}, validation set, {len(actual)} images")
    print(f"of which malignant: {int(was_malignant.sum())}")
    print()
    print(f"{'threshold':>10}{'recall':>9}{'precision':>11}{'missed':>9}{'false alarms':>14}")

    for step in range(1, 20):
        threshold = step / 20
        recall, precision, missed, false_alarms = score_at(
            threshold, malignant_scores, was_malignant
        )
        print(
            f"{threshold:>10.2f}{recall:>8.1%}{precision:>11.1%}"
            f"{missed:>9}{false_alarms:>14}"
        )
