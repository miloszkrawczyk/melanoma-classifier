"""Tests that the training run draws from pinned generators.

None of these need the dataset or any weights: they check the sources of
randomness rather than the training loop, which is the only way to test this
in seconds instead of an hour. Two full runs landing on identical weights is
not something a test can assert anyway, because some MPS kernels are not
deterministic.
"""

import torch
from PIL import Image

from src.config import EARLY_STOPPING_PATIENCE, LR_PATIENCE
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


def test_the_learning_rate_drops_before_training_is_abandoned():
    """Ordering between two patiences, and getting it wrong is silent.

    If early stopping were the quicker of the two, training would be given up
    before the smaller learning rate was ever tried, and the scheduler would
    be decoration.
    """
    assert LR_PATIENCE < EARLY_STOPPING_PATIENCE


def test_the_stopping_patience_survives_a_barren_stretch():
    """Four is the measured minimum for this dataset, not a round number.

    Replaying the validation losses recorded in results/04 through early
    stopping:

        patience   stops after   best epoch   best loss
           1, 2, 3     4, 5, 6            3      0.6815
           4                11            7      0.6373

    That run went three epochs without improving (4 to 6) before its best one
    at 7. A patience of 2 or 3, which is what most tutorials use, would have
    stopped before it and kept a model 6.9% worse on its own loss. This test
    exists so nobody tidies the number down to a more familiar one.
    """
    recorded_val_losses = [
        1.1445,
        0.7399,
        0.6815,
        0.7957,
        0.8313,
        0.7443,
        0.6373,
        0.7396,
        0.6393,
        0.6961,
        0.6457,
        0.6400,
        0.7634,
        0.7345,
        0.6849,
    ]

    best = float("inf")
    best_epoch = 0
    barren = 0

    for epoch, loss in enumerate(recorded_val_losses, start=1):
        if loss < best:
            best, best_epoch, barren = loss, epoch, 0
        else:
            barren += 1
            if barren >= EARLY_STOPPING_PATIENCE:
                break

    assert best_epoch == 7, (
        f"patience {EARLY_STOPPING_PATIENCE} would have kept epoch {best_epoch} "
        f"at loss {best:.4f} instead of epoch 7 at 0.6373"
    )
