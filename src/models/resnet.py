"""ResNet18 pretrained on ImageNet, with its last layer swapped for ours.

Its convolutional layers already know edges, textures and shapes, learned
from 1.28 million photos of dogs, cars and mushrooms. None of that is about
skin, but the low-level features are the same ones our own CNN had to work
out from scratch on 6986 images.

Only the final classifier gets replaced, because ImageNet's 1000 categories
are no use here.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import CLASSES


def build_resnet18(pretrained=True):
    """Build the network and give it a seven-class head.

    pretrained=False skips the 45 MB download of ImageNet weights. That is what
    serving wants: the very next thing it does is load our own fine-tuned
    weights over the top, so fetching someone else's first is 45 MB and a
    writable cache directory spent on numbers that get thrown away. It also
    means the container starts without needing the network at all.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # model.fc is the final Linear(512, 1000). Everything in front of it stays
    # exactly as trained; only the answer sheet at the end changes.
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    return model
