"""Reading HAM10000 metadata, turning a diagnosis code into the number the
model is trained on, and splitting the rows into train, validation and test.
"""

import random

import pandas as pd

from src.config import (
    BENIGN,
    BENIGN_CLASSES,
    CLASS_NAMES,
    CLASSES,
    MALIGNANT,
    MALIGNANT_CLASSES,
    METADATA_CSV,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)


def to_class_index(dx):
    """Turn a diagnosis code like "mel" into its position in CLASSES."""
    return CLASSES.index(dx)


def to_class_name(dx):
    """Turn a diagnosis code into something readable, like "Melanoma (mel)"."""
    return CLASS_NAMES[dx]


def to_category(dx):
    """Say whether a diagnosis is malignant or benign."""
    if dx in MALIGNANT_CLASSES:
        return MALIGNANT
    if dx in BENIGN_CLASSES:
        return BENIGN
    raise ValueError(f"unknown diagnosis: {dx}")


def load_metadata():
    """Read the csv and add a label column holding the number to train on."""
    metadata = pd.read_csv(METADATA_CSV)
    metadata["label"] = metadata["dx"].apply(to_class_index)
    return metadata


def split_lesion_ids(metadata):
    """Deal the lesion ids into three groups: train, validation and test.

    Splitting lesions rather than rows matters, because the dataset
    photographs some lesions more than once. Split the rows and the same
    lesion lands on both sides, so the test score would be measured on
    pictures the model has already learned.
    """
    lesion_ids = sorted(metadata["lesion_id"].unique())
    random.Random(SEED).shuffle(lesion_ids)

    train_end = int(len(lesion_ids) * TRAIN_RATIO)
    val_end = train_end + int(len(lesion_ids) * VAL_RATIO)

    return lesion_ids[:train_end], lesion_ids[train_end:val_end], lesion_ids[val_end:]


def split_metadata(metadata):
    """Cut the metadata into three tables, one per set."""
    train_ids, val_ids, test_ids = split_lesion_ids(metadata)

    train = metadata[metadata["lesion_id"].isin(train_ids)]
    val = metadata[metadata["lesion_id"].isin(val_ids)]
    test = metadata[metadata["lesion_id"].isin(test_ids)]

    return train, val, test
