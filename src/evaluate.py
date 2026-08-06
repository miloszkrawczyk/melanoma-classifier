"""Scores a trained model properly, which here means not with accuracy.

Two thirds of the dataset are ordinary moles, so a model answering "nv" every
time scores 67% and is worth nothing. What matters is how many of the actual
malignant lesions it caught, and that only shows up per class.

Defaults to the validation set. The test set is for the very end, once there
is nothing left to tune, so pass "test" deliberately:

    uv run python -m src.evaluate          # while still working on the model
    uv run python -m src.evaluate test     # once, for the README
"""

import sys

import torch

from src.config import CLASSES, CNN_WEIGHTS, MALIGNANT_CLASSES
from src.data import build_dataloaders
from src.models.cnn import SkinLesionCNN
from src.train import pick_device

MALIGNANT_INDEXES = [CLASSES.index(dx) for dx in MALIGNANT_CLASSES]


def collect_predictions(model, loader, device):
    """Run the whole loader and return what the model said next to the truth."""
    model.eval()
    predicted = []
    actual = []

    with torch.no_grad():
        for images, labels in loader:
            guesses = model(images.to(device)).argmax(dim=1).cpu()
            predicted.append(guesses)
            actual.append(labels)

    return torch.cat(predicted), torch.cat(actual)


def confusion_matrix(predicted, actual):
    """Rows are the real diagnosis, columns are what the model guessed."""
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int)
    for guess, truth in zip(predicted, actual):
        matrix[truth, guess] += 1
    return matrix


def print_confusion_matrix(matrix):
    print("rows: real diagnosis, columns: what the model guessed")
    print(f"{'':7}" + "".join(f"{name:>7}" for name in CLASSES))
    for i, name in enumerate(CLASSES):
        print(f"{name:7}" + "".join(f"{int(count):>7}" for count in matrix[i]))


def print_per_class(matrix):
    """Recall and precision for each of the seven diagnoses.

    Recall answers "of the real melanomas, how many did it find". Precision
    answers "when it says melanoma, how often is it right". Recall is the one
    that matters here, because a missed cancer is worse than a false alarm.
    """
    print(f"{'class':7}{'support':>9}{'recall':>9}{'precision':>11}")
    for i, name in enumerate(CLASSES):
        correct = int(matrix[i, i])
        real = int(matrix[i].sum())
        guessed = int(matrix[:, i].sum())

        recall = correct / real if real else 0.0
        precision = correct / guessed if guessed else 0.0

        print(f"{name:7}{real:>9}{recall:>8.1%}{precision:>11.1%}")


def print_binary(predicted, actual):
    """Collapse the seven diagnoses into the question people actually ask."""
    is_malignant = torch.tensor([int(i) in MALIGNANT_INDEXES for i in predicted])
    was_malignant = torch.tensor([int(i) in MALIGNANT_INDEXES for i in actual])

    caught = int((is_malignant & was_malignant).sum())
    missed = int((~is_malignant & was_malignant).sum())
    false_alarms = int((is_malignant & ~was_malignant).sum())

    print(f"malignant lesions found:  {caught}")
    print(f"malignant lesions missed: {missed}")
    print(f"false alarms:             {false_alarms}")
    print(f"recall on malignant:      {caught / (caught + missed):.1%}")
    print(f"precision on malignant:   {caught / (caught + false_alarms):.1%}")


split = sys.argv[1] if len(sys.argv) > 1 else "val"
if split not in ("val", "test"):
    raise SystemExit(f'expected "val" or "test", got "{split}"')

device = pick_device()
train_loader, val_loader, test_loader = build_dataloaders()
loader = test_loader if split == "test" else val_loader

model = SkinLesionCNN().to(device)
model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))

predicted, actual = collect_predictions(model, loader, device)
matrix = confusion_matrix(predicted, actual)

print(f"scoring on the {split} set, {len(actual)} images")
print()
print_confusion_matrix(matrix)
print()
print_per_class(matrix)
print()
accuracy = int((predicted == actual).sum()) / len(actual)
print(f"accuracy: {accuracy:.1%}  (a model always answering nv would get 67.2%)")
print()
print_binary(predicted, actual)
