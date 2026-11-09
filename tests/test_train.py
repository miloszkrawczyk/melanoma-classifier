"""Tests that the training run draws from pinned generators.

None of these need the dataset or any weights: they check the sources of
randomness rather than the training loop, which is the only way to test this
in seconds instead of an hour. Two full runs landing on identical weights is
not something a test can assert anyway, because some MPS kernels are not
deterministic.
"""

import re

import pytest
import torch
from PIL import Image

from src.config import MODELS, PROJECT_ROOT
from src.data import build_transform
from src.models.cnn import SkinLesionCNN
from src.models.resnet import build_resnet18
from src.train import set_seed


def a_photo():
    """A plain picture, enough for a transform to have something to turn."""
    return Image.new("RGB", (600, 450), (195, 140, 145))


def first_weight(model):
    return next(model.parameters()).detach().clone()


def test_the_cnn_starts_from_the_same_place_twice():
    """Its whole 400k parameters are random at birth, so this is the big one."""
    set_seed()
    first = first_weight(SkinLesionCNN())

    set_seed()
    second = first_weight(SkinLesionCNN())

    assert torch.equal(first, second)


def test_the_cnn_starts_somewhere_else_without_the_seed():
    """Guards the test above against passing for the wrong reason.

    If initialisation were somehow deterministic on its own, the comparison
    would hold with or without seeding and prove nothing.
    """
    torch.manual_seed(1)
    first = first_weight(SkinLesionCNN())

    torch.manual_seed(2)
    second = first_weight(SkinLesionCNN())

    assert not torch.equal(first, second)


def test_the_resnet_starts_from_the_same_place_twice():
    set_seed()
    first = first_weight(build_resnet18(pretrained=False))

    set_seed()
    second = first_weight(build_resnet18(pretrained=False))

    assert torch.equal(first, second)


def test_augmentation_repeats_after_seeding():
    """Flips, rotations and colour jitter all draw from the global generator.

    Unseeded they are the largest remaining difference between two runs, since
    every photo in every epoch goes through them.
    """
    augment = build_transform(augment=True)
    photo = a_photo()

    set_seed()
    first = augment(photo)

    set_seed()
    second = augment(photo)

    assert torch.equal(first, second)


def test_augmentation_actually_varies_between_photos():
    """Otherwise the test above would pass on a transform that does nothing."""
    augment = build_transform(augment=True)
    photo = a_photo()

    set_seed()
    first = augment(photo)
    second = augment(photo)

    assert not torch.equal(first, second)


def recorded_val_losses(results_file):
    """The validation losses of a run this project actually did.

    Read from the file rather than pasted in, because results/ holds the
    record of what happened and is never rewritten. Forty numbers would be a
    lot to copy, and copying them would let the two drift apart.
    """
    text = (PROJECT_ROOT / "results" / results_file).read_text()

    return [float(loss) for loss in re.findall(r"val loss (\d+\.\d+)", text)]


def epoch_early_stopping_would_keep(losses, patience):
    """Which epoch's weights a given patience ends up keeping."""
    best, best_epoch, barren = float("inf"), 0, 0

    for epoch, loss in enumerate(losses, start=1):
        if loss < best:
            best, best_epoch, barren = loss, epoch, 0
        else:
            barren += 1
            if barren >= patience:
                break

    return best_epoch


@pytest.mark.parametrize("name", ["cnn", "resnet"])
def test_the_learning_rate_drops_before_training_is_abandoned(name):
    """Ordering between the two patiences, and getting it wrong is silent.

    If early stopping were the quicker of the two, training would be given up
    before the smaller learning rate was ever tried, and the scheduler would
    be decoration. Seen once for real: with both set to two, a run stopped at
    epoch four and the rate never moved.
    """
    settings = MODELS[name]

    assert settings["lr_patience"] < settings["early_stopping_patience"]


def test_the_resnet_patience_survives_its_own_barren_stretch():
    """Replays results/04, the run the ResNet's patience was measured on.

    It went three epochs without improving, 4 to 6, before its best at 7. A
    patience of 2 or 3, which is what most tutorials use, would have stopped
    before it and kept a model 6.9% worse on its own loss.
    """
    losses = recorded_val_losses("04-resnet18.txt")
    patience = MODELS["resnet"]["early_stopping_patience"]

    assert epoch_early_stopping_would_keep(losses, patience) == 7


def test_the_cnn_patience_survives_its_own_barren_stretch():
    """Replays results/03, and this is the one that caught a real mistake.

    The CNN's patience was first set to the ResNet's five. That run went
    thirteen epochs without improving, 14 to 26, and then found its best at
    37, so five stopped at epoch 13 and kept a model 8.3% worse. A network
    learning from scratch has a much noisier curve than a pretrained one
    being nudged, and one pair of patiences cannot serve both.
    """
    losses = recorded_val_losses("03-cnn-40-epochs.txt")
    patience = MODELS["cnn"]["early_stopping_patience"]

    assert epoch_early_stopping_would_keep(losses, patience) == 37


def test_a_shared_patience_would_have_failed_one_of_them():
    """Why the two numbers are per model rather than one constant.

    Neither model's value works for the other: the ResNet's five stops the
    CNN twenty-four epochs early, and the CNN's fifteen would have run the
    ResNet to its ceiling rather than stopping it.
    """
    cnn_losses = recorded_val_losses("03-cnn-40-epochs.txt")
    resnet_patience = MODELS["resnet"]["early_stopping_patience"]

    kept = epoch_early_stopping_would_keep(cnn_losses, resnet_patience)

    assert kept == 13, (
        f"the ResNet's patience of {resnet_patience} keeps epoch {kept} of the "
        "CNN's run, not its best at 37"
    )
