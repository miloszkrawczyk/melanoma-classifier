"""Shape checks only. The weights are random until training runs, so there is
nothing about the answers worth asserting yet.
"""

import torch

from src.config import BATCH_SIZE, CLASSES, IMAGE_SIZE
from src.model import SkinLesionCNN


def test_model_output_shape():
    model = SkinLesionCNN()
    fake_batch = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    output = model(fake_batch)

    assert output.shape == torch.Size([BATCH_SIZE, len(CLASSES)])


def test_model_handles_a_single_image():
    model = SkinLesionCNN()
    fake_batch = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    output = model(fake_batch)

    assert output.shape == torch.Size([1, len(CLASSES)])
