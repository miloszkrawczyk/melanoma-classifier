"""Shape and sanity checks for the attention map.

Uses an untrained ResNet, so these run without the dataset or saved weights.
An untrained network produces a meaningless map, but a map of the right shape
and range, which is all these check.
"""

import torch
from PIL import Image

from src.gradcam import GradCAM, last_conv_layer, overlay
from src.models.resnet import build_resnet18


def test_heatmap_has_the_layers_own_resolution():
    model = build_resnet18()
    cam_tool = GradCAM(model, last_conv_layer(model, "resnet"))

    cam = cam_tool.heatmap(torch.rand(1, 3, 224, 224), class_index=4)

    # 224 halved five times on the way down the network.
    assert cam.shape == torch.Size([7, 7])


def test_heatmap_is_scaled_between_zero_and_one():
    model = build_resnet18()
    cam_tool = GradCAM(model, last_conv_layer(model, "resnet"))

    cam = cam_tool.heatmap(torch.rand(1, 3, 224, 224), class_index=0)

    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_overlay_keeps_the_original_size():
    image = Image.new("RGB", (300, 200), (200, 150, 150))
    cam = torch.rand(7, 7)

    assert overlay(image, cam).size == (300, 200)
