"""Shape checks for the pretrained ResNet, same idea as the CNN's tests.

There is one extra thing worth asserting here that does not apply to the CNN:
that the head really was swapped. A ResNet still answering with 1000 numbers
would train without complaint and produce nonsense.
"""

import torch

from src.config import CLASSES, IMAGE_SIZE
from src.models.resnet import build_resnet18


def test_resnet_output_shape():
    model = build_resnet18()
    fake_batch = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE)

    output = model(fake_batch)

    assert output.shape == torch.Size([4, len(CLASSES)])


def test_head_was_replaced():
    model = build_resnet18()

    assert model.fc.out_features == len(CLASSES)
    assert model.fc.in_features == 512
