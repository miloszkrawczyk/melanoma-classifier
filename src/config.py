"""Project-wide constants: where the data lives and how HAM10000's seven
diagnoses collapse into the two classes this project actually predicts.
"""

from pathlib import Path

# *** Paths ***
# Anchored to the project root so the scripts work no matter which
# directory I happen to run them from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
IMAGES_DIR: Path = DATA_DIR / "images"
METADATA_CSV: Path = DATA_DIR / "HAM10000_metadata.csv"

# *** Kaggle ***
KAGGLE_DATASET: str = "kmader/skin-cancer-mnist-ham10000"

# *** Labels ***
# Every image in HAM10000 carries one of seven diagnoses in the "dx" column.
# I fold them into benign vs malignant, because that is the question the
# whole project is built around.
#   akiec - actinic keratoses, pre-cancerous
#   bcc   - basal cell carcinoma
#   mel   - melanoma
MALIGNANT_CLASSES: tuple = ("akiec", "bcc", "mel")
#   bkl   - benign keratosis
#   df    - dermatofibroma
#   nv    - melanocytic nevus, an ordinary mole
#   vasc  - vascular lesion
BENIGN_CLASSES: tuple = ("bkl", "df", "nv", "vasc")

# Numbers, not strings, because that is what the loss function expects.
BENIGN: int = 0
MALIGNANT: int = 1
