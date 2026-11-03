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
from math import sqrt

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    CLASSES,
    INTERVAL_PERCENT,
    INTERVAL_Z,
    MALIGNANT_CLASSES,
    MODELS,
)
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


def collect_probabilities(model, loader, device):
    """Run the whole loader and return the model's probabilities next to the truth.

    The whole distribution rather than the winning class, because the
    threshold-free scores further down need it and fetching it in a second
    pass would double the time spent on the split. Nothing is lost: softmax
    does not change which class is largest, so argmax gives the same answer.
    """
    model.eval()
    probabilities = []
    actual = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            actual.append(labels)

    return torch.cat(probabilities), torch.cat(actual)


def confusion_matrix(predicted, actual):
    """Rows are the real diagnosis, columns are what the model guessed."""
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int)
    for guess, truth in zip(predicted, actual):
        matrix[truth, guess] += 1
    return matrix


def majority_class_baseline(actual):
    """What a model answering the single commonest class every time would score.

    Accuracy means nothing here without this number to sit next to, because two
    thirds of HAM10000 are ordinary moles. It differs per split: 67.2% on
    validation against 64.1% on the test set. Hard-coding it printed the
    validation figure for every split, including the one that went to the README.
    """
    counts = torch.bincount(actual, minlength=len(CLASSES))
    commonest = int(counts.argmax())

    return CLASSES[commonest], int(counts[commonest]) / len(actual)


def wilson_interval(successes, total):
    """The point estimate and the range it could plausibly have landed in.

    Wilson rather than the textbook p +/- z*sqrt(p(1-p)/n), which falls apart
    exactly where this project needs it. df has 12 test images: at 1 of 12 the
    textbook formula puts the lower bound at -7.3%, and at 0 of 10 it returns a
    range of zero width, claiming certainty from no evidence. Wilson gives
    1.5% and 0 to 27.8%.

    Clamped to [0, 1], because a proportion cannot leave it and the formula
    returns a hair below zero at 0 successes from rounding alone.
    """
    if not total:
        return 0.0, 0.0, 0.0

    point = successes / total
    divisor = 1 + INTERVAL_Z**2 / total
    centre = (point + INTERVAL_Z**2 / (2 * total)) / divisor
    half = (
        INTERVAL_Z
        * sqrt(point * (1 - point) / total + INTERVAL_Z**2 / (4 * total**2))
        / divisor
    )

    return point, max(0.0, centre - half), min(1.0, centre + half)


def is_malignant(labels):
    """True wherever the label is one of the three malignant diagnoses."""
    return torch.tensor([int(i) in MALIGNANT_INDEXES for i in labels])


def malignant_auc(probabilities, actual):
    """Two scores for the malignant call that do not depend on the threshold.

    What gets ranked is the sum of the three malignant probabilities, the same
    number predict.py compares against MALIGNANT_THRESHOLD. So these measure
    the decision the app actually makes, not a neighbouring one.

    PR-AUC is the one worth quoting. ROC-AUC counts true negatives, and with
    four benign lesions for every malignant one there are so many of them that
    it flatters anything. PR-AUC ignores them and asks only how well the
    positives were found, which is the question when a miss is what costs.
    """
    scores = probabilities[:, MALIGNANT_INDEXES].sum(dim=1).numpy()
    truth = is_malignant(actual).numpy()

    return roc_auc_score(truth, scores), average_precision_score(truth, scores)


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
    heading = f"recall {INTERVAL_PERCENT}% CI"
    print(f"{'class':7}{'support':>9}{'recall':>9}{heading:>18}{'precision':>11}")

    for i, name in enumerate(CLASSES):
        correct = int(matrix[i, i])
        real = int(matrix[i].sum())
        guessed = int(matrix[:, i].sum())

        recall, low, high = wilson_interval(correct, real)
        precision = correct / guessed if guessed else 0.0
        interval = f"{low:.1%} - {high:.1%}"

        print(f"{name:7}{real:>9}{recall:>8.1%}{interval:>18}{precision:>11.1%}")


def print_binary(predicted, actual):
    """Collapse the seven diagnoses into the question people actually ask."""
    called_malignant = is_malignant(predicted)
    really_malignant = is_malignant(actual)

    caught = int((called_malignant & really_malignant).sum())
    missed = int((~called_malignant & really_malignant).sum())
    false_alarms = int((called_malignant & ~really_malignant).sum())

    print(f"malignant lesions found:  {caught}")
    print(f"malignant lesions missed: {missed}")
    print(f"false alarms:             {false_alarms}")
    print(f"recall on malignant:      {caught / (caught + missed):.1%}")
    print(f"precision on malignant:   {caught / (caught + false_alarms):.1%}")


def print_report(probabilities, actual, name, split):
    """Everything the command prints, in the order it prints it."""
    predicted = probabilities.argmax(dim=1)
    matrix = confusion_matrix(predicted, actual)

    print(f"scoring {name} on the {split} set, {len(actual)} images")
    print()
    print_confusion_matrix(matrix)
    print()
    print_per_class(matrix)
    print()
    accuracy = int((predicted == actual).sum()) / len(actual)
    commonest, baseline = majority_class_baseline(actual)
    print(
        f"accuracy: {accuracy:.1%}  "
        f"(a model always answering {commonest} would get {baseline:.1%})"
    )
    print()
    print_binary(predicted, actual)
    print()
    roc_auc, pr_auc = malignant_auc(probabilities, actual)
    print(f"ROC-AUC on malignant:     {roc_auc:.3f}")
    print(f"PR-AUC on malignant:      {pr_auc:.3f}")


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

    probabilities, actual = collect_probabilities(model, loader, device)
    print_report(probabilities, actual, name, split)
