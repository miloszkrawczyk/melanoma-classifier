"""Measures the mean and standard deviation of each colour channel across
the training images.

Normalisation needs those two numbers, and the usual ImageNet values are a
poor fit here. These photos are close-ups of skin, so they come out brighter,
pinker and far more alike than a random pile of internet pictures.

Run it once and copy the result into config.py. Recomputing it on every
training run would cost minutes for numbers that never change.
"""

import torch
from PIL import Image
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGES_DIR
from src.data import load_metadata, split_metadata

# Training images only. Measuring across validation or test would let facts
# about them seep into training, the same leak the lesion split guards against.
train, _, _ = split_metadata(load_metadata())
image_ids = list(train["image_id"])

to_tensor = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

# Running totals, one entry per colour channel. Summing as I go keeps memory
# flat, whereas holding 6986 tensors at once would not fit.
channel_sums = torch.zeros(3)
channel_squares = torch.zeros(3)
pixels = 0

for i, image_id in enumerate(image_ids, start=1):
    image = Image.open(IMAGES_DIR / f"{image_id}.jpg").convert("RGB")
    tensor = to_tensor(image)

    channel_sums += tensor.sum(dim=[1, 2])
    channel_squares += (tensor**2).sum(dim=[1, 2])
    pixels += IMAGE_SIZE * IMAGE_SIZE

    if i % 1000 == 0:
        print(f"{i}/{len(image_ids)} images")

mean = channel_sums / pixels
# Variance is the mean of the squares minus the square of the mean.
std = (channel_squares / pixels - mean**2).sqrt()

print()
print(f"mean: ({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f})")
print(f"std:  ({std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f})")
