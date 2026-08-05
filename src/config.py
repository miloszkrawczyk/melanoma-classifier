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

# *** Classes ***
# Every image in HAM10000 carries one of seven diagnoses in the "dx" column,
# and the model predicts all seven. A position in this tuple is the number
# the network learns, so the order has to stay fixed. Once weights are
# trained against it, reordering would silently relabel every prediction.
# Alphabetical, so there is no arbitrary decision to remember.
CLASSES: tuple = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

# What a person should see instead of a four-letter code. The code stays in
# the name because it is what the dataset and every paper on it use.
CLASS_NAMES: dict = {
    "akiec": "Actinic keratosis (akiec)",
    "bcc": "Basal cell carcinoma (bcc)",
    "bkl": "Benign keratosis (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic nevus (nv)",
    "vasc": "Vascular lesion (vasc)",
}

# The model never predicts these two directly. They are read off whichever
# diagnosis it picked, so a prediction can answer both "what is it" and
# "should I worry". akiec counts as malignant because it is pre-cancerous.
MALIGNANT_CLASSES: tuple = ("akiec", "bcc", "mel")
BENIGN_CLASSES: tuple = ("bkl", "df", "nv", "vasc")

MALIGNANT: str = "malignant"
BENIGN: str = "benign"


# *** Splitting the data ***
# A fixed seed keeps the same lesions in the same set on every run. Without
# it each training run would be scored against a different test set, and
# comparing two runs would tell me nothing.
SEED: int = 42

# Test gets whatever is left, so these two ratios decide all three sizes.
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15


# *** Preparing the images ***
# The photos are 600x450, which is more detail than the network needs and
# slow to train on. 224x224 is the size ImageNet models were built around.
IMAGE_SIZE: int = 224

# Channel means and standard deviations measured across ImageNet. A model
# pretrained there was fed input scaled exactly this way, so handing it
# anything else costs accuracy for no reason.
IMAGENET_MEAN: tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple = (0.229, 0.224, 0.225)
