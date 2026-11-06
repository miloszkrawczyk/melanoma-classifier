"""Tests that the training run draws from pinned generators.

None of these need the dataset or any weights: they check the sources of
randomness rather than the training loop, which is the only way to test this
in seconds instead of an hour. Two full runs landing on identical weights is
not something a test can assert anyway, because some MPS kernels are not
deterministic.
"""

import torch
from PIL import Image

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
