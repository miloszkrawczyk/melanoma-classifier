"""Tests for the metadata handling, mostly guarding the train/test split.

The leak these check for is invisible at runtime: nothing crashes, the
accuracy just comes out too high. A test is the only thing that catches it.
"""

import pytest
import torch

from src.config import CLASSES, IMAGE_SIZE, METADATA_CSV
from src.data import (
    SkinLesionDataset,
    build_dataloaders,
    build_transform,
    load_metadata,
    split_metadata,
    to_category,
    to_class_index,
)

# The dataset is 2.9 GB and gitignored, so on a fresh checkout, and on the
# machine that runs the tests after a push, it simply is not there. Skipping
# beats failing: these tests are about the split and the loading, and there is
# nothing to split when there is no data.
pytestmark = pytest.mark.skipif(
    not METADATA_CSV.exists(),
    reason="needs the HAM10000 dataset, see the README for how to fetch it",
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


def test_dataloader_stacks_images_into_a_batch():
    # No workers: this only wants one batch, and starting them would cost
    # more than everything else in this file put together.
    train_loader, _, _ = build_dataloaders(batch_size=8, num_workers=0)

    images, labels = next(iter(train_loader))

    assert tuple(images.shape) == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(labels.shape) == (8,)


def test_the_training_loader_hands_out_the_same_order_twice():
    """The loader shuffles, and its generator is pinned so the order repeats.

    This one needs the dataset, which is why it lives here rather than with
    the other seeding tests. Its own generator rather than the global one, so
    the order does not depend on how many draws the weight initialisation and
    the augmentation happened to take first.

    It takes about twelve seconds, nearly all of it starting worker
    processes, and it has to. Loading now runs in two of them, and whether
    seeding survives that is exactly what this checks. Passing num_workers=0
    here would make it quick and prove nothing.
    """
    first, _, _ = build_dataloaders(batch_size=8)
    second, _, _ = build_dataloaders(batch_size=8)

    # Two batches is plenty: a different order shows up immediately.
    first_labels = [labels for _, labels in [next(iter(first))]]
    second_labels = [labels for _, labels in [next(iter(second))]]

    assert torch.equal(first_labels[0], second_labels[0])


def test_split_is_repeatable():
    metadata = load_metadata()

    first_train, _, _ = split_metadata(metadata)
    second_train, _, _ = split_metadata(metadata)

    assert list(first_train["image_id"]) == list(second_train["image_id"])


def shares_of_each_set(metadata, diagnosis):
    """What fraction of one diagnosis's photos landed in each of the three sets."""
    train, val, test = split_metadata(metadata)
    counts = [int((part["dx"] == diagnosis).sum()) for part in (train, val, test)]

    return [count / sum(counts) for count in counts]


def test_every_class_keeps_its_share_of_each_set():
    """The point of the whole step, and the thing the old split got wrong.

    df is 115 photos in total. The old deal gave the test set twelve of them,
    10.4% instead of 15%, and its recall there carried a 95% interval 44
    points wide. One point of tolerance is loose enough for the arithmetic of
    whole lesions and tight enough to fail on the old behaviour.
    """
    metadata = load_metadata()

    for diagnosis in CLASSES:
        train, val, test = shares_of_each_set(metadata, diagnosis)

        assert abs(train - 0.70) < 0.01, diagnosis
        assert abs(val - 0.15) < 0.01, diagnosis
        assert abs(test - 0.15) < 0.01, diagnosis


def test_no_class_disappears_from_a_set():
    """A class with a handful of photos could otherwise miss a set entirely.

    vasc has 142 photos and df 115. Scoring a class that is absent from
    validation would divide by zero, and scoring one absent from test would
    quietly report nothing at all.
    """
    metadata = load_metadata()
    train, val, test = split_metadata(metadata)

    for diagnosis in CLASSES:
        for part in (train, val, test):
            assert int((part["dx"] == diagnosis).sum()) > 0, diagnosis


def test_the_sets_hold_comparably_many_photos_per_lesion():
    """Guards an ordering choice that was measured and then undone.

    Dealing the lesions with most photos first also holds the proportions, but
    it sweeps the multi-photo lesions into training: 4597 distinct lesions
    there instead of 5227, and 1.05 photos per lesion in test against 1.53 in
    training. Evaluation wants independent lesions, so the sets should look
    alike in this respect.
    """
    metadata = load_metadata()
    parts = split_metadata(metadata)

    per_lesion = [len(part) / part["lesion_id"].nunique() for part in parts]

    assert max(per_lesion) - min(per_lesion) < 0.1
