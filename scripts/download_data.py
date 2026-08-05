"""Downloads HAM10000 from Kaggle and gathers every photo into one folder.

Kaggle serves the images split across two folders, part_1 and part_2. That
split means nothing for training, so this script merges them into
data/images/ and the rest of the project only ever sees a single folder.

Needs a Kaggle API token in ~/.kaggle/kaggle.json, see the README.
"""

import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import DATA_DIR, IMAGES_DIR, KAGGLE_DATASET

api = KaggleApi()
api.authenticate()

DATA_DIR.mkdir(exist_ok=True)

print(f"downloading {KAGGLE_DATASET}, about 2.5 GB, this takes a few minutes")
api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)

IMAGES_DIR.mkdir(exist_ok=True)

for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
    part_dir = DATA_DIR / part
    if not part_dir.exists():
        continue

    for image_path in part_dir.glob("*.jpg"):
        shutil.move(image_path, IMAGES_DIR / image_path.name)

    part_dir.rmdir()
    print(f"moved images out of {part}")

image_count = len(list(IMAGES_DIR.glob("*.jpg")))
print(f"done: {image_count} images in {IMAGES_DIR}")
