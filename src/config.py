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


# *** Reporting uncertainty ***
# Every metric here is a proportion measured on a finite sample, so each one
# carries a range it could plausibly have landed in. 95% is what medical
# papers quote, and picking anything else invites the question why.
INTERVAL_PERCENT: int = 95

# The normal quantile matching INTERVAL_PERCENT above. These two have to move
# together: 90% is 1.645, 99% is 2.576. Written out rather than taken from
# scipy.stats, because one number with a comment is cheaper than a dependency
# for one number.
#
# Named INTERVAL rather than CONFIDENCE on purpose. CONFIDENCE_THRESHOLD above
# is how sure the model says it is, which has nothing to do with this.
INTERVAL_Z: float = 1.96

# How many times the bootstrap redraws the split. Measured: 10000 resamples of
# 1521 images cost 0.1 s, which is nothing against the 40 s the scoring pass
# itself takes. 1000 would be ten times quicker and leave the bounds wobbling
# in the third digit between runs, which defeats the point of reporting them.
BOOTSTRAP_RESAMPLES: int = 10000


# *** Limits on what the service accepts ***
# Measured over the dataset: HAM10000 photos run 83 KB to 479 KB, averaging
# 270 KB, and the examples shipped with the interface sit between 231 and
# 333 KB. A photo straight off a current phone reaches a few megabytes. 8 MB
# clears all of that and turns away anything that is not a photograph.
#
# This is not a guard against the bytes arriving. Starlette has already
# buffered the whole body before a handler runs, spooling to a temporary file
# past 1 MB, and caps nothing. It is a guard against handing them to the model.
# A real byte cap belongs in the reverse proxy in front of the app.
MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024

# The byte limit above does not catch a decompression bomb: a PNG of a few tens
# of kilobytes can declare 10000x10000, which is 400 MB once decoded. PIL waves
# that through, since its own MAX_IMAGE_PIXELS is 89.5 Mpx and it only raises
# above twice that. Neither limit contains the other, because an 8 MB JPEG can
# easily be 50 Mpx.
#
# 50 Mpx is what the best phone cameras produce today, so anything lower would
# start refusing real photographs. Decoded it is around 150 MB, which the
# service survives. Named UPLOAD rather than IMAGE to stay clearly apart from
# PIL.Image.MAX_IMAGE_PIXELS, which is a different number for a different job.
MAX_UPLOAD_PIXELS: int = 50_000_000

# How often one caller may ask. Measured against a running server: a single
# client pushes 81 requests a second through /predict on one thread and 295 on
# eight, which is 17715 a minute. A person looking at the demo, examples and
# own photos included, manages maybe 15 a minute.
#
# So 60 a minute leaves a person four times the room they need while capping a
# script at a third of a percent of what the machine can do. The limit is not
# here to protect the CPU from a human; it is here so one script cannot take
# the service away from everybody else.
RATE_LIMIT_REQUESTS: int = 60
RATE_LIMIT_WINDOW_SECONDS: int = 60


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
# scripts/compute_stats.py over all 7009 of them. Skin close-ups come out
# much brighter, much redder and far more alike than internet photos, so
# the ImageNet figures below would centre the data in the wrong place.
# These belong to the network trained from scratch.
#
# Remeasured after the split started keeping class proportions, because these
# are properties of whichever photos ended up in training. The answer barely
# moved: the means came back identical and one digit of one deviation changed,
# 0.140 to 0.141. Worth knowing, since it says the old numbers were never the
# problem, and worth redoing anyway so the count above is not a lie.
HAM10000_MEAN: tuple = (0.764, 0.546, 0.571)
HAM10000_STD: tuple = (0.141, 0.152, 0.169)

# The same two figures measured across ImageNet. A model pretrained there
# was fed input scaled exactly this way, so handing it anything else costs
# accuracy for no reason. These belong to the ResNet.
IMAGENET_MEAN: tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple = (0.229, 0.224, 0.225)

# How far a training photo may be turned. Any angle is a fair example of the
# same lesion, but a full 180 leaves large empty corners after the rotation,
# so this stays modest.
ROTATION_DEGREES: int = 20


# *** Test-time augmentation ***
# At inference the same photo is shown to the network four times, turned by a
# quarter each time, and the four sets of probabilities are averaged. A lesion
# has no top or bottom, so all four are equally fair readings of it, and the
# answers differ only because convolution is not rotation invariant.
#
# Quarter turns rather than the ROTATION_DEGREES used in training: a turn by a
# right angle is an exact permutation of pixels, with no interpolation and none
# of the empty corners the comment above worries about.
#
# Measured on validation, 1521 images, same weights, threshold 0.15:
#
#                       recall  precision  missed  false alarms  PR-AUC
#   no TTA               93.9%      41.1%      18           394   0.721
#   four turns           95.2%      39.4%      14           430   0.742
#   four turns + mirror  95.6%      39.8%      13           423   0.742
#
# Four misses fewer out of eighteen, and PR-AUC rises, which says the ranking
# genuinely improved rather than sliding along the same curve. The price is 36
# more false alarms. Mirroring adds 0.3 points of recall at identical PR-AUC,
# so it buys nothing but time. Reproduce with scripts/measure_tta.py.
TTA_QUARTER_TURNS: tuple = (0, 1, 2, 3)


# *** Training ***
# How many images the network looks at before it updates its weights. Small
# enough to fit in a laptop's memory, large enough that one odd photo cannot
# drag the whole update off course.
BATCH_SIZE: int = 32

# How many processes decode and augment photos alongside the training loop.
# Measured on the real command, three ResNet epochs over 7009 photos:
#
#   workers                      per epoch   speedup
#     0                             54 s       1.0x
#     2, restarted each epoch       48 s       1.14x
#     2, kept alive between epochs  34 s       1.6x
#
# Two is the whole win on the loading side. In isolation the loaders reach 312
# photos a second with none, 698 with two, 2616 with eight, while the model
# consumes about 290, so once loading clears that, more processes have nothing
# to do: four and eight were measurably no faster.
#
# The reason persistent_workers matters so much is macOS. It starts processes
# by spawning rather than forking, so each worker re-imports torch, and both
# the training and validation loaders rebuild theirs every epoch. Measured in
# isolation with fork the epoch came out at 24 s, which the real command never
# delivers: that number was 1.6x too good and is the reason these figures come
# from "python -m src.train" and not from a snippet.
#
# This is this machine's number, not a universal one. On faster hardware the
# model stops being the bottleneck and the threshold moves up.
#
# pin_memory is deliberately not set alongside this. PyTorch answers it with
# "'pin_memory' argument is set as true but not supported on MPS now", and the
# timings with and without are identical. It earns its place on CUDA, which is
# neither this machine nor the CPU-only container.
NUM_WORKERS: int = 2

# *** Regularisation and when to stop ***
# Weight decay pulls every weight slightly towards zero on every step, which
# keeps the network from fitting the training set in detail it cannot
# generalise. It is needed here, and that part is measured: in results/04 the
# training loss fell to 0.2505 while the validation loss bottomed at 0.6373 in
# epoch 7 and then drifted up to 0.6849. A 2.7x gap is overfitting that has
# already started.
#
# The value itself is convention, not measurement: 1e-4 is the usual starting
# point for AdamW at these learning rates. Saying otherwise would be inventing
# a measurement. The retraining run is what will show whether it earned its
# place.
WEIGHT_DECAY: float = 1e-4

# How far the learning rate drops when it is cut. A tenth is what
# ReduceLROnPlateau uses by default, written out because a number that decides
# something does not belong hidden inside a library call.
#
# The two patiences that go with it are NOT here. They live per model in MODELS
# below, because measuring them on one model and applying them to both is a
# mistake I made and then caught: see the tables there.
LR_DROP: float = 0.1

# Trained weights land here, one file per model so training the ResNet does
# not quietly wipe out the CNN. Gitignored, since running the training
# rebuilds them and a binary blob is not worth versioning.
#
# Named .safetensors because that is what training writes and what the image
# ships. load_weights still reads a .pth of the same name if that is all a
# checkout has, so naming the preferred format here breaks nothing.
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"

# The two networks need different treatment, so every setting that differs
# between them lives here rather than being edited by hand between runs.
MODELS: dict = {
    "cnn": {
        "weights": WEIGHTS_DIR / "cnn.safetensors",
        # Its own statistics, because nothing constrains a network built here.
        "mean": HAM10000_MEAN,
        "std": HAM10000_STD,
        # Adam's own default. Starting from noise, the first steps have a long
        # way to travel.
        "learning_rate": 0.001,
        # Its own log says 40 was still too few: results/03 found its best at
        # epoch 37 of 40 and was still improving when it ran out. 60 is a
        # ceiling, and early stopping decides when to stop.
        "epochs": 60,
        # Measured on the losses recorded in results/03. That run went
        # thirteen epochs without improving, epochs 14 to 26, and then found
        # its best at 37:
        #
        #   patience   stops after   best epoch   best loss
        #      5, 8, 10, 12   18-25           13      1.0269
        #     15                 40           37      0.9478
        #
        # So anything under 14 keeps a model 8.3% worse on its own loss. A
        # network learning from scratch has a far noisier validation curve
        # than a pretrained one being nudged: standard deviation 0.169 across
        # its epochs against the ResNet's 0.126.
        "early_stopping_patience": 15,
        # Five, not the ResNet's two. Two-epoch dips happen constantly on this
        # curve, three times in the first ten epochs of results/03, and
        # cutting the rate on each one leaves the model learning at 1e-5 while
        # its training loss is still falling. Measured once with two: the run
        # stopped at epoch 13, underfit.
        "lr_patience": 5,
    },
    "resnet": {
        "weights": WEIGHTS_DIR / "resnet.safetensors",
        # ImageNet's statistics, because that is what its filters were fed
        # during pretraining.
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        # Ten times smaller. Its weights are already good, so the point is to
        # nudge them, not to knock them out of place.
        "learning_rate": 0.0001,
        # A ceiling now, not a count: early stopping decides when to stop and
        # this only says when to give up waiting. 15 was too low to be a
        # ceiling, since the old run was still finding near-ties at epoch 12.
        "epochs": 30,
        # Measured on the losses recorded in results/04. That run went three
        # epochs without improving, 4 to 6, before its best at 7:
        #
        #   patience   stops after   best epoch   best loss
        #      1, 2, 3       4, 5, 6            3      0.6815
        #      4, 5           11, 12            7      0.6373
        #
        # Four is the minimum, five leaves a margin. Far smaller than the
        # CNN's fifteen, because a pretrained network's curve is much calmer.
        "early_stopping_patience": 5,
        "lr_patience": 2,
    },
}
