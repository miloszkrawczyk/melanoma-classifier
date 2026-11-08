"""A small convolutional network written from scratch, the first of the two
models in this project.

Four identical blocks, each halving the picture and doubling the number of
patterns it hunts for, then one linear layer that picks a diagnosis. Writing
it by hand first is the point: the pretrained ResNet that comes later is the
same idea, only deeper and already trained.
"""

import torch.nn as nn

from src.config import CLASSES


def conv_block(in_channels, out_channels):
    """Find patterns, level the signal, drop the negatives, halve the map.

    bias=False because BatchNorm comes next and subtracts the batch mean,
    which removes any constant the convolution added. Checked rather than
    assumed: adding 12.345 to every bias leaves the output after BatchNorm
    unchanged to within floating point noise.

    This is not a saving worth having on its own, at 480 numbers out of
    391175. It is here so the model does not carry parameters that provably
    cannot affect what it answers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SkinLesionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 224 -> 112 -> 56 -> 28 -> 14 pixels, 3 -> 32 -> 64 -> 128 -> 256
        # channels. The picture shrinks while the number of patterns grows,
        # which is the usual shape of a convolutional network.
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )

        # Squash each of the 256 maps down to one number: how strongly that
        # filter fired anywhere in the picture. Flattening 256x14x14 instead
        # would hand the last layer 50176 inputs and ample room to memorise
        # 6986 training photos.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, len(CLASSES))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        # Drop the two leftover 1x1 dimensions, keeping the batch dimension.
        x = x.flatten(1)
        return self.classifier(x)
