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


def build_resnet18(pretrained=True, freeze=False):
    """Build the network and give it a seven-class head.

    pretrained=False skips the 45 MB download of ImageNet weights. That is what
    serving wants: the very next thing it does is load our own fine-tuned
    weights over the top, so fetching someone else's first is 45 MB and a
    writable cache directory spent on numbers that get thrown away. It also
    means the container starts without needing the network at all.

    freeze=True trains the new head and nothing else, which is 3591 of the
    11180103 parameters, 0.032% of the network. It is off by default: full
    fine-tuning is what this project has always done, and the point of the
    switch is to measure whether that was the right call rather than to change
    it. Both arms of that comparison have to differ in this one thing only.

    freeze=True with pretrained=False is a combination that runs and means
    nothing: it would fit a 3591-parameter head on top of frozen random
    filters. Useful in a test that wants the shape without the download,
    useless for training anything.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # model.fc is the final Linear(512, 1000). Everything in front of it stays
    # exactly as trained; only the answer sheet at the end changes.
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if freeze:
        # After the head is replaced, so the new one stays trainable.
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")

        # Switching gradients off is not enough, and this is the part that is
        # easy to miss. BatchNorm folds each batch into its running mean and
        # variance whenever the model is in train mode, gradients or no
        # gradients: measured, a single forward pass moved bn1's running mean
        # by 0.0379, and training does 220 of those per epoch. A backbone
        # drifting like that is not frozen, and comparing it against full
        # fine-tuning would be measuring something else entirely.
        #
        # momentum=0 stops it, since the update is
        # running = (1 - momentum) * running + momentum * batch. Unlike
        # calling .eval() on the backbone it survives the model.train() at the
        # top of every epoch, because momentum is an attribute and not a mode.
        # Unlike track_running_stats=False it leaves inference alone: the
        # service keeps using these statistics rather than computing them from
        # a batch of one photograph.
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = 0.0

    return model
