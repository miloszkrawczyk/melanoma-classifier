"""Tests for the metadata handling, mostly guarding the train/test split.

The leak these check for is invisible at runtime: nothing crashes, the
accuracy just comes out too high. A test is the only thing that catches it.
"""

import pytest

from src.config import CLASSES, IMAGE_SIZE
from src.data import (
    SkinLesionDataset,
    build_transform,
    load_metadata,
    split_metadata,
    to_category,
    to_class_index,
)


def test_class_index_matches_position_in_classes():
    assert to_class_index("akiec") == 0
    assert to_class_index("mel") == 4
    assert CLASSES[to_class_index("nv")] == "nv"


def test_unknown_diagnosis_is_rejected():
    with pytest.raises(ValueError):
        to_class_index("xyz")


def test_category_covers_every_class():
    for dx in CLASSES:
        assert to_category(dx) in ("benign", "malignant")


def test_no_lesion_ends_up_in_two_sets():
    train, val, test = split_metadata(load_metadata())

    train_lesions = set(train["lesion_id"])
    val_lesions = set(val["lesion_id"])
    test_lesions = set(test["lesion_id"])

    assert train_lesions.isdisjoint(val_lesions)
    assert train_lesions.isdisjoint(test_lesions)
    assert val_lesions.isdisjoint(test_lesions)


def test_split_keeps_every_row():
    metadata = load_metadata()
    train, val, test = split_metadata(metadata)

    assert len(train) + len(val) + len(test) == len(metadata)


def test_dataset_returns_a_tensor_and_a_label():
    train, _, _ = split_metadata(load_metadata())
    dataset = SkinLesionDataset(train, build_transform())

    image, label = dataset[0]

    assert tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert label in range(len(CLASSES))


def test_dataset_length_matches_the_metadata():
    train, _, _ = split_metadata(load_metadata())
    dataset = SkinLesionDataset(train, build_transform())

    assert len(dataset) == len(train)


def test_split_is_repeatable():
    metadata = load_metadata()

    first_train, _, _ = split_metadata(metadata)
    second_train, _, _ = split_metadata(metadata)

    assert list(first_train["image_id"]) == list(second_train["image_id"])
