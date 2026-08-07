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

# A lesion counts as malignant when the three malignant classes add up to at
# least this much, rather than when one of them happens to score highest.
# Taking the top class treats both mistakes as costing the same, and they do
# not: a false alarm costs an appointment, a miss can cost a life.
#
# Chosen from the sweep in scripts/tune_threshold.py against the ResNet. On
# the validation set 0.15 misses 18 of 293 malignant lesions and raises 394
# false alarms out of 1521 images. Taking the top class instead would miss 66.
MALIGNANT_THRESHOLD: float = 0.15

# Below this much confidence in the winning diagnosis, the answer is not worth
# naming. Measured on the validation set: the model is right 53% of the time
# when it scores 0.50-0.60 and 93% when it scores above 0.90, so the number
# means something. At 0.60 it declines to name 17% of images and the diagnoses
# it does give are right 84.2% of the time, against 78.2% for answering
# everything.
#
# This gates the seven-class diagnosis only. The malignant/benign call stands
# either way, since it adds three probabilities up and survives the model
# being unsure which of the three it is looking at.
CONFIDENCE_THRESHOLD: float = 0.60


# *** Turning away photos that are not skin lesions ***
# A classifier can only ever answer with one of the classes it was taught. Ask
# it about a checkerboard and it names a diagnosis, confidently. Measured on
# random noise it returned "bkl" at 86.5% certainty, so the confidence gate
# above does not catch this on its own.
#
# These bounds come from 153 validation images. Roughness is the average
# brightness change between neighbouring pixels: noise sits at 0.22, a flat
# colour at 0.0, real photos between 0.0040 and 0.0494. The margins are wide
# on purpose, since turning away a real lesion is the worse mistake.
SKIN_MIN_ROUGHNESS: float = 0.002
SKIN_MAX_ROUGHNESS: float = 0.08

# Skin close-ups are red-dominant: red was the strongest channel in 98% of the
# sample, and never averaged below 0.461. The tolerance leaves room for the
# other 2% rather than turning them away.
SKIN_MIN_RED: float = 0.35
SKIN_RED_TOLERANCE: float = 0.05

# Saturation is the average gap between a pixel's brightest and dimmest
# channel. A greyscale image sits at 0.0, real skin between 0.0509 and
# 0.4232 in the sample. This is what turns away a grey checkerboard, which
# slips past the red check by having every channel equal.
SKIN_MIN_SATURATION: float = 0.03


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

# Trained weights land here, one file per model so training the ResNet does
# not quietly wipe out the CNN. Gitignored, since running the training
# rebuilds them and a binary blob is not worth versioning.
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"

# The two networks need different treatment, so every setting that differs
# between them lives here rather than being edited by hand between runs.
MODELS: dict = {
    "cnn": {
        "weights": WEIGHTS_DIR / "cnn.pth",
        # Its own statistics, because nothing constrains a network built here.
        "mean": HAM10000_MEAN,
        "std": HAM10000_STD,
        # Adam's own default. Starting from noise, the first steps have a long
        # way to travel.
        "learning_rate": 0.001,
        # Augmentation made the task hard enough that 15 epochs cut the run
        # short. Only the best epoch is kept, so overshooting costs time only.
        "epochs": 40,
    },
    "resnet": {
        "weights": WEIGHTS_DIR / "resnet.pth",
        # ImageNet's statistics, because that is what its filters were fed
        # during pretraining.
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        # Ten times smaller. Its weights are already good, so the point is to
        # nudge them, not to knock them out of place.
        "learning_rate": 0.0001,
        # Fewer epochs too: it starts most of the way there.
        "epochs": 15,
    },
}
