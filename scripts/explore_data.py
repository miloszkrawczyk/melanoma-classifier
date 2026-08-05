"""Manual look at HAM10000 before any model code gets written: what the
metadata holds, how skewed the classes are once the seven diagnoses fold
into benign vs malignant, and whether every row actually has a photo.

Also counts unique lesions, because the dataset photographs some lesions
more than once and that matters when splitting into train and test.
"""

import pandas as pd
from PIL import Image

from src.config import BENIGN_CLASSES, IMAGES_DIR, MALIGNANT_CLASSES, METADATA_CSV

metadata = pd.read_csv(METADATA_CSV)

print(f"rows: {len(metadata)}")
print(f"columns: {list(metadata.columns)}")
print()

print("diagnoses in the dx column:")
print(metadata["dx"].value_counts())
print()

benign_count = metadata["dx"].isin(BENIGN_CLASSES).sum()
malignant_count = metadata["dx"].isin(MALIGNANT_CLASSES).sum()

print(f"benign: {benign_count}")
print(f"malignant: {malignant_count}")
print(f"malignant share: {malignant_count / len(metadata) * 100:.1f}%")
print()

# One lesion can appear in several rows, photographed from different angles.
# Splitting on rows would put the same lesion in train and in test, so the
# test score would be flattered by pictures the model has effectively seen.
print(f"unique lesions: {metadata['lesion_id'].nunique()}")
print(f"rows per lesion: {len(metadata) / metadata['lesion_id'].nunique():.2f}")
print()

# The metadata stores image_id without the extension, the files are .jpg.
missing = [
    image_id
    for image_id in metadata["image_id"]
    if not (IMAGES_DIR / f"{image_id}.jpg").exists()
]
print(f"rows with no image file: {len(missing)}")

first_id = metadata["image_id"].iloc[0]
image = Image.open(IMAGES_DIR / f"{first_id}.jpg")
print(f"first image {first_id}.jpg: size {image.size}, mode {image.mode}")
