"""Shape checks for the pretrained ResNet, same idea as the CNN's tests.

There is one extra thing worth asserting here that does not apply to the CNN:
that the head really was swapped. A ResNet still answering with 1000 numbers
would train without complaint and produce nonsense.
"""

import torch
import torch.nn as nn

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


def trainable(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


def forward_passes_in_train_mode(model, how_many=5):
    """Run the model the way an epoch does, and hand back bn1's statistics."""
    model.train()
    with torch.no_grad():
        for _ in range(how_many):
            model(torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE))

    return model.bn1.running_mean.clone(), model.bn1.running_var.clone()


def test_nothing_is_frozen_by_default():
    """Full fine-tuning is what this project does and the switch must not change it."""
    model = build_resnet18(pretrained=False)

    assert len(trainable(model)) == len(list(model.named_parameters()))


def test_freezing_leaves_only_the_head():
    """3591 numbers out of 11180103, which is what the comparison is about."""
    model = build_resnet18(pretrained=False, freeze=True)

    assert sorted(trainable(model)) == ["fc.bias", "fc.weight"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3591


def test_freezing_also_pins_the_batchnorm_statistics():
    """The part that is easy to miss, and the reason momentum is set to zero.

    BatchNorm folds each batch into its running mean and variance whenever the
    model is in train mode, whether or not gradients are on. Measured: one
    forward pass moved bn1's running mean by 0.0379, and an epoch does 220 of
    them. A backbone drifting like that is not frozen.
    """
    model = build_resnet18(pretrained=False, freeze=True)
    before_mean = model.bn1.running_mean.clone()
    before_var = model.bn1.running_var.clone()

    after_mean, after_var = forward_passes_in_train_mode(model)

    assert torch.equal(before_mean, after_mean)
    assert torch.equal(before_var, after_var)


def test_the_statistics_do_move_when_nothing_is_frozen():
    """Keeps the test above from passing for the wrong reason.

    If BatchNorm never updated its statistics, pinning them would be a no-op
    and that test would hold whether or not the momentum line existed.
    """
    model = build_resnet18(pretrained=False)
    before_mean = model.bn1.running_mean.clone()

    after_mean, _ = forward_passes_in_train_mode(model)

    assert not torch.equal(before_mean, after_mean)


def test_weight_decay_does_not_shrink_a_frozen_weight():
    """A frozen backbone slowly decaying towards zero would be hard to spot.

    AdamW skips parameters with no gradient, so it does not happen, but the
    cost of finding that out the slow way is a wasted training run.
    """
    model = build_resnet18(pretrained=False, freeze=True)
    frozen = dict(model.named_parameters())["layer4.1.conv2.weight"]
    before = frozen.clone()

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(3):
        optimiser.zero_grad()
        output = model(torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE))
        nn.CrossEntropyLoss()(output, torch.tensor([0, 1, 2, 3])).backward()
        optimiser.step()

    assert torch.equal(before, frozen)
