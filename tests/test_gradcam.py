"""Shape and sanity checks for the attention map, plus its thread safety.

Uses an untrained ResNet, so these run without the dataset or saved weights.
An untrained network produces a meaningless map, but a map of the right shape
and range, which is all the first few check.

The concurrency tests matter more than they look. The service runs its
handlers in worker threads, so one model answers several requests at once, and
a hook reaching inside that model is shared by all of them. Nothing crashes
when that goes wrong: the caller just gets a heatmap drawn from somebody
else's photo.
"""

import threading

import torch
from PIL import Image

from src.gradcam import heatmap, last_conv_layer, overlay
from src.models.resnet import build_resnet18


def untrained_resnet():
    torch.manual_seed(0)
    model = build_resnet18(pretrained=False)
    model.eval()

    return model, last_conv_layer(model, "resnet")


def test_heatmap_has_the_layers_own_resolution():
    model, layer = untrained_resnet()

    cam = heatmap(model, layer, torch.rand(1, 3, 224, 224), class_index=4)

    # 224 halved five times on the way down the network.
    assert cam.shape == torch.Size([7, 7])


def test_heatmap_is_scaled_between_zero_and_one():
    model, layer = untrained_resnet()

    cam = heatmap(model, layer, torch.rand(1, 3, 224, 224), class_index=0)

    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_overlay_keeps_the_original_size():
    image = Image.new("RGB", (300, 200), (200, 150, 150))
    cam = torch.rand(7, 7)

    assert overlay(image, cam).size == (300, 200)


def test_heatmap_leaves_no_hook_behind():
    """Registering per call is only safe because the hook comes off again.

    Without the removal this would climb by one on every request until the
    process ran out of memory, which is why the hook used to be registered
    once and its result shared.
    """
    model, layer = untrained_resnet()

    heatmap(model, layer, torch.rand(1, 3, 224, 224), class_index=4)

    assert len(layer._forward_hooks) == 0


def test_heatmap_does_not_touch_the_parameter_gradients():
    """autograd.grad rather than backward(), for 43 MB of reasons.

    backward() would write a gradient onto all 11.2 million parameters, shared
    by every request, none of which Grad-CAM reads.
    """
    model, layer = untrained_resnet()

    heatmap(model, layer, torch.rand(1, 3, 224, 224), class_index=4)

    assert all(parameter.grad is None for parameter in model.parameters())


def test_concurrent_heatmaps_match_the_sequential_ones():
    """Eight threads, eight different photos, one shared model.

    This is the test the whole step exists for. A hook belongs to the layer,
    not to the call, so before the fix a thread would capture whatever forward
    happened to pass through next and draw its map from that.
    """
    model, layer = untrained_resnet()
    batches = [torch.rand(1, 3, 224, 224) for _ in range(8)]

    sequential = [heatmap(model, layer, batch, 4) for batch in batches]

    concurrent = [None] * len(batches)

    def run(index):
        concurrent[index] = heatmap(model, layer, batches[index], 4)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(batches))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for one, other in zip(sequential, concurrent):
        assert torch.equal(one, other)
