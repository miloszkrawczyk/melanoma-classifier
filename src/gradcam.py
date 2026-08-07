"""Grad-CAM: a heatmap of where the network was looking when it answered.

The idea is short. The last convolutional layer holds a stack of small maps,
each one saying "my pattern showed up here". Which of those patterns mattered
for a given diagnosis is exactly what the gradient of that diagnosis with
respect to the maps measures. Weight the maps by that and add them up, and
you get a picture of what the answer rested on.

Worth having for two reasons. It is the honest answer to "why should I trust
this", and it catches a specific failure: HAM10000 contains photos with
rulers, ink marks and hairs in frame, and a network that learned to key on
those would score well and be useless. If the heat lands on the lesion, the
network is looking at the right thing.
"""

import numpy as np
import torch
from PIL import Image

# The blue-to-red ramp everyone recognises from heatmaps, written out rather
# than imported. matplotlib carries it, but pulling 40 MB of plotting library
# into a web service for six colours is a poor trade.
COLOUR_STOPS = (
    (0.0, (0, 0, 143)),
    (0.125, (0, 0, 255)),
    (0.375, (0, 255, 255)),
    (0.625, (255, 255, 0)),
    (0.875, (255, 0, 0)),
    (1.0, (128, 0, 0)),
)


class GradCAM:
    """Records what the chosen layer saw, and how much it mattered."""

    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Hooks are callbacks PyTorch runs as data passes through a layer. The
        # forward one catches what the layer produced, the backward one catches
        # how much each of those values pulled on the final answer. There is no
        # other way to reach inside a network mid-pass.
        layer.register_forward_hook(self._save_activations)
        layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def heatmap(self, batch, class_index):
        """A 0-1 map the size of the layer's own maps, usually 7x7."""
        self.model.zero_grad()

        logits = self.model(batch)
        # Backward from one class only. The heatmap answers "why this
        # diagnosis", so a different class gives a different picture.
        logits[0, class_index].backward()

        # One number per map: how strongly it pushed the answer, averaged over
        # its own width and height. This is what "gradient-weighted" means.
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        # Negative values are evidence against the class, and drawing them
        # would just muddy the picture.
        cam = torch.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu()


def colourise(values):
    """Turn a 0-1 array into RGB along the blue-to-red ramp."""
    positions = [position for position, _ in COLOUR_STOPS]
    channels = [
        np.interp(values, positions, [colour[i] for _, colour in COLOUR_STOPS])
        for i in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def overlay(image, cam, alpha=0.5):
    """Paint the heatmap over the photo, red where it mattered most."""
    size = image.size
    # The map is 7x7 for a 224x224 input, since the network shrank the picture
    # five times on the way down. Stretching it back is why Grad-CAM looks
    # blurry: the detail genuinely is not there.
    heat = Image.fromarray((cam.numpy() * 255).astype(np.uint8), mode="L")
    heat = heat.resize(size, Image.BICUBIC)

    coloured = Image.fromarray(colourise(np.array(heat) / 255.0), mode="RGB")

    return Image.blend(image.convert("RGB"), coloured, alpha)


def last_conv_layer(model, name):
    """The deepest convolutional layer, which is the one worth looking at.

    Earlier layers see edges and textures everywhere; only the last one has
    combined them into something that maps onto a diagnosis.
    """
    if name == "resnet":
        return model.layer4
    if name == "cnn":
        return model.features
    raise ValueError(f"no known conv layer for {name}")
