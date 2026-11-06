"""Everything between the raw dataset and the network: reading the metadata,
turning a diagnosis code into a number, splitting the rows into train,
validation and test, and handing out images one at a time as tensors.
"""

import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    BENIGN,
    BENIGN_CLASSES,
    CLASS_NAMES,
    CLASSES,
    HAM10000_MEAN,
    HAM10000_STD,
    IMAGE_SIZE,
    IMAGES_DIR,
    MALIGNANT,
    MALIGNANT_CLASSES,
    METADATA_CSV,
    ROTATION_DEGREES,
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

    Two things have to hold at once, and they pull against each other.

    No lesion may land in two sets. 1956 lesions are photographed more than
    once, 4501 photos in all, so dealing rows instead of lesions would put the
    same mole in training and in test and the score would come out flattering
    and useless. That is what the grouping is for.

    And every diagnosis has to keep its share of each set. The old deal did
    not watch that: df is 115 photos in total and came out 82 / 21 / 12, so
    its recall on the test set rested on twelve photos and carried a 95%
    interval 44 points wide.

    So the deal runs one class at a time and greedily: shuffle that class's
    lesions, then hand each in turn to whichever set is furthest behind its
    share of that class's photos.

    What is counted is photos, not lesions, and that is the whole point. nv
    averages 1.24 photos per lesion and mel 1.81, so an even deal of lesions
    is not an even deal of photos. It is also why sklearn's
    StratifiedGroupKFold was not used here: it balances groups, and measured
    on this data it left df with 18 test photos but only 12 validation ones,
    moving the problem rather than fixing it. It is built for groups that span
    several classes, and no lesion here has more than one diagnosis.

    Greedy partitioning usually wants the largest items dealt first, so the
    small ones are left to close the gaps. Measured, it makes no difference to
    the proportions here and costs something real: no lesion holds more than 6
    photos, which is tiny against the smallest target in the table (17 photos,
    df's share of a set), so ordering cannot strand anything. Dealing biggest
    first did hold the proportions, but it swept the multi-photo lesions into
    training and left it 4597 distinct lesions instead of 5227.

    Measured: the worst a class sits from its target was 4.6 points before and
    0.4 after. df goes from 82 / 21 / 12 photos to 81 / 17 / 17, and the
    photos per lesion stay even across the three sets, 1.34 / 1.33 / 1.35.
    """
    photos_per_lesion = metadata.groupby("lesion_id").size()
    diagnosis_of = metadata.groupby("lesion_id")["dx"].first()
    shares = (TRAIN_RATIO, VAL_RATIO, 1 - TRAIN_RATIO - VAL_RATIO)

    dealt = ([], [], [])
    shuffler = random.Random(SEED)

    for diagnosis in sorted(metadata["dx"].unique()):
        lesion_ids = sorted(diagnosis_of[diagnosis_of == diagnosis].index)
        shuffler.shuffle(lesion_ids)

        photos = int(photos_per_lesion[lesion_ids].sum())
        given = [0, 0, 0]

        for lesion in lesion_ids:
            behind = [share * photos - count for share, count in zip(shares, given)]
            furthest = behind.index(max(behind))

            dealt[furthest].append(lesion)
            given[furthest] += int(photos_per_lesion[lesion])

    return dealt


def split_metadata(metadata):
    """Cut the metadata into three tables, one per set."""
    train_ids, val_ids, test_ids = split_lesion_ids(metadata)

    train = metadata[metadata["lesion_id"].isin(train_ids)]
    val = metadata[metadata["lesion_id"].isin(val_ids)]
    test = metadata[metadata["lesion_id"].isin(test_ids)]

    return train, val, test


def build_transform(augment=False, mean=HAM10000_MEAN, std=HAM10000_STD):
    """The recipe every photo goes through on its way into the network.

    With augment=True the photo is flipped and turned at random first, which
    only ever belongs on the training set. Validation and test have to stay
    identical between runs, otherwise two scores cannot be compared.

    Defaults to the statistics of our own training images, which is what the
    network trained from scratch wants. The pretrained ResNet gets the
    ImageNet pair passed in instead, because that is what it was built on.
    """
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if augment:
        # A lesion has no top or bottom, so a flipped or turned photo is still
        # an honest example of the same diagnosis. Faces or text could not be
        # treated this way.
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(ROTATION_DEGREES),
            # Dermatoscopes and lighting differ between clinics, so nudging
            # brightness and contrast keeps the network from leaning on the
            # look of one particular camera.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(steps)


class SkinLesionDataset(Dataset):
    """Hands out one image at a time, already resized and turned into a tensor.

    PyTorch only asks a dataset two things: how many examples it holds, and
    what example number i looks like. Batching and shuffling are the
    DataLoader's job, built on top of those two answers.

    Nothing is read up front. All 10015 photos decoded at once would take
    around 8 GB of memory and most of them would sit there unused, since the
    network works through 32 at a time.
    """

    def __init__(self, metadata, transform):
        # Plain lists rather than the DataFrame. The split leaves gaps in the
        # row numbers, so metadata["image_id"][0] can raise a KeyError even
        # though the table is not empty.
        self.image_ids = list(metadata["image_id"])
        self.labels = list(metadata["label"])
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_path = IMAGES_DIR / f"{self.image_ids[index]}.jpg"
        # convert("RGB") so a stray grayscale file still comes out 3 channels.
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.labels[index]


def build_dataloaders(batch_size=BATCH_SIZE, augment=False, mean=HAM10000_MEAN, std=HAM10000_STD):
    """One loader per split, each handing out batches instead of single images.

    Only the training loader shuffles, and only it ever gets augmented. The
    other two are read end to end for a score, so a fixed order and a fixed
    recipe are what make two runs comparable.
    """
    train_transform = build_transform(augment=augment, mean=mean, std=std)
    plain_transform = build_transform(augment=False, mean=mean, std=std)

    train, val, test = split_metadata(load_metadata())

    # Its own generator rather than the global one. The order of an epoch then
    # depends on this seed alone, instead of on how many draws the weight
    # initialisation and the augmentation happened to take first.
    train_loader = DataLoader(
        SkinLesionDataset(train, train_transform),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        SkinLesionDataset(val, plain_transform), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        SkinLesionDataset(test, plain_transform), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
