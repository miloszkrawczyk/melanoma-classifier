"""Reading HAM10000 metadata and turning a diagnosis code into the number
the model is trained on.
"""

import pandas as pd

from src.config import (
    BENIGN,
    BENIGN_CLASSES,
    CLASS_NAMES,
    CLASSES,
    MALIGNANT,
    MALIGNANT_CLASSES,
    METADATA_CSV,
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
