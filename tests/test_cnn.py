"""Shape checks only. The weights are random until training runs, so there is
nothing about the answers worth asserting yet.
"""

import torch
import torch.nn as nn

from src.config import BATCH_SIZE, CLASSES, IMAGE_SIZE
from src.models.cnn import SkinLesionCNN


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


def test_no_convolution_carries_a_bias():
    """BatchNorm follows every convolution and removes any constant it adds.

    Checked rather than reasoned about: adding 12.345 to every bias leaves the
    output after BatchNorm unchanged to within floating point noise, so those
    480 numbers out of 391175 cannot affect an answer. Small enough not to
    matter for memory, which is the point of testing it rather than trusting
    anyone to notice it drift back.
    """
    model = SkinLesionCNN()

    with_bias = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Conv2d) and layer.bias is not None
    ]

    assert with_bias == []


def test_a_bias_before_batchnorm_would_change_nothing():
    """The reason the test above is allowed to be strict.

    If this ever fails, the assumption behind bias=False has broken and the
    biases should come back rather than the other test being relaxed.
    """
    torch.manual_seed(0)
    convolution = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    normalise = nn.BatchNorm2d(8)
    photos = torch.randn(4, 3, 16, 16)

    before = normalise(convolution(photos)).clone()
    with torch.no_grad():
        convolution.bias += 12.345
    after = normalise(convolution(photos))

    assert torch.allclose(before, after, atol=1e-4)
