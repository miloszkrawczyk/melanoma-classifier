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

# Channel means and deviations of the training images, measured by
# scripts/compute_stats.py over all 6986 of them. Skin close-ups come out
# much brighter, much redder and far more alike than internet photos, so
# the ImageNet figures below would centre the data in the wrong place.
# These belong to the network trained from scratch.
HAM10000_MEAN: tuple = (0.764, 0.546, 0.571)
HAM10000_STD: tuple = (0.140, 0.152, 0.169)

# The same two figures measured across ImageNet. A model pretrained there
# was fed input scaled exactly this way, so handing it anything else costs
# accuracy for no reason. These belong to the ResNet.
IMAGENET_MEAN: tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple = (0.229, 0.224, 0.225)

# How far a training photo may be turned. Any angle is a fair example of the
# same lesion, but a full 180 leaves large empty corners after the rotation,
# so this stays modest.
ROTATION_DEGREES: int = 20


# *** Training ***
# How many images the network looks at before it updates its weights. Small
# enough to fit in a laptop's memory, large enough that one odd photo cannot
# drag the whole update off course.
BATCH_SIZE: int = 32

# How many times the network works through the whole training set. Started at
# 15, the same as the MNIST project. Augmentation made the training task hard
# enough that the best epoch was the last one, so the run was being cut short
# rather than converging. Only the best epoch is kept, so overshooting costs
# time and nothing else.
EPOCHS: int = 40

# How big a step the optimiser takes. 0.001 is Adam's own default and a
# sensible place to start; too high and the loss bounces, too low and the
# training crawls.
LEARNING_RATE: float = 0.001

# Trained weights land here, one file per model so training the ResNet does
# not quietly wipe out the CNN. Gitignored, since running the training
# rebuilds them and a binary blob is not worth versioning.
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
CNN_WEIGHTS: Path = WEIGHTS_DIR / "cnn.pth"
