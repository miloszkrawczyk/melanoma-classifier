"""Scores a trained model properly, which here means not with accuracy.

Two thirds of the dataset are ordinary moles, so a model answering "nv" every
time scores 67% and is worth nothing. What matters is how many of the actual
malignant lesions it caught, and that only shows up per class.

Takes the model to score and, optionally, which split to score it on. The
test set is for the very end, once there is nothing left to tune, so it has
to be asked for deliberately:

    uv run python -m src.evaluate cnn           # validation, day to day
    uv run python -m src.evaluate resnet        # validation
    uv run python -m src.evaluate resnet test   # once, for the README
"""

import sys

import torch

from src.config import CLASSES, MALIGNANT_CLASSES, MODELS
from src.data import build_dataloaders
from src.device import pick_device
from src.train import build_model

MALIGNANT_INDEXES = [CLASSES.index(dx) for dx in MALIGNANT_CLASSES]


def parse_arguments(argv):
    """Read the model name and the split off the command line.

    The split defaults to validation, not test. The test set is for the very
    end, so asking for it has to be deliberate: there is no way to spend it by
    forgetting an argument. Takes argv rather than reading sys.argv, so a test
    can hand it a list.

    Both values are checked here, so a typo stops the run with a readable
    message. Without the name check, MODELS[name] in the block below raises a
    bare KeyError and build_model's own, clearer rejection is never reached.
    """
    name = argv[1] if len(argv) > 1 else "cnn"
    split = argv[2] if len(argv) > 2 else "val"

    if name not in MODELS:
        expected = " or ".join(f'"{known}"' for known in MODELS)
        raise SystemExit(f'expected {expected}, got "{name}"')

    if split not in ("val", "test"):
        raise SystemExit(f'expected "val" or "test", got "{split}"')

    return name, split


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


def print_report(predicted, actual, name, split):
    """Everything the command prints, in the order it prints it."""
    matrix = confusion_matrix(predicted, actual)

    print(f"scoring {name} on the {split} set, {len(actual)} images")
    print()
    print_confusion_matrix(matrix)
    print()
    print_per_class(matrix)
    print()
    accuracy = int((predicted == actual).sum()) / len(actual)
    print(f"accuracy: {accuracy:.1%}  (a model always answering nv would get 67.2%)")
    print()
    print_binary(predicted, actual)


# Guarded for the same reason train.py is. Without it, importing this module
# built a network, read 43 MB of weights off disk and scored a whole split,
# which is why it was the only file in src/ with no tests of its own.
if __name__ == "__main__":
    name, split = parse_arguments(sys.argv)
    settings = MODELS[name]

    device = pick_device()

    # The same normalisation the model was trained with. Feeding it the other
    # pair would quietly cost accuracy with nothing to show why.
    _, val_loader, test_loader = build_dataloaders(
        mean=settings["mean"], std=settings["std"]
    )
    loader = test_loader if split == "test" else val_loader

    model = build_model(name).to(device)
    model.load_state_dict(torch.load(settings["weights"], map_location=device))

    predicted, actual = collect_predictions(model, loader, device)
    print_report(predicted, actual, name, split)
