"""Training loop shared by both models: forward pass, loss, backward pass,
optimiser step, repeated over batches and epochs.

    uv run python -m src.train cnn       # the network built here
    uv run python -m src.train resnet    # the pretrained one

The loop is the same either way. What differs between the two, the learning
rate, the epoch count and the normalisation, lives in config.MODELS.

Two things here that the MNIST project did not need. The loss carries class
weights, because nv accounts for two thirds of the data and without them the
network learns to answer "ordinary mole" and stop thinking. And every epoch
ends with a pass over the validation set, so the weights that get saved are
the ones that did best on photos the network never trained on.
"""

import random
import sys

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    CLASSES,
    EARLY_STOPPING_PATIENCE,
    LR_DROP,
    LR_PATIENCE,
    MODELS,
    SEED,
    WEIGHT_DECAY,
    WEIGHTS_DIR,
)
from src.data import build_dataloaders, load_metadata, split_metadata
from src.device import pick_device
from src.models.cnn import SkinLesionCNN
from src.models.resnet import build_resnet18
from src.weights import safetensors_beside, save_weights


def set_seed(seed=SEED):
    """Pin every generator the training run draws from.

    The split was already repeatable: data.py shuffles lesion ids through
    random.Random(SEED). Nothing else was. Four things draw from the global
    generators and all four change the weights: the CNN's initialisation, the
    new head on the ResNet, the order the training loader hands out batches,
    and the flips, rotations and colour jitter of augmentation.

    Without this, two runs of the same command produce different weights and
    different numbers in results/, which makes comparing two runs meaningless.
    That is about to matter more than usual: the frozen-backbone comparison
    further along is supposed to differ in one thing only.

    numpy is seeded although training never calls it directly, because pandas
    and scikit-learn both sit on it and the stratified split uses sklearn.
    Seeding it now is cheaper than working out later why a split will not come
    out the same twice.

    What this does not buy is bit-for-bit identical runs. Some MPS kernels are
    not deterministic, so two runs land close rather than equal. Seeding
    removes the avoidable difference, not the hardware's.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Covers the CPU generator and the MPS or CUDA one, whichever is in use.
    torch.manual_seed(seed)


def build_model(name):
    """Build the network asked for on the command line."""
    if name == "cnn":
        return SkinLesionCNN()
    if name == "resnet":
        return build_resnet18()
    raise SystemExit(f'expected "cnn" or "resnet", got "{name}"')


def class_weights(metadata):
    """How much a mistake on each class should hurt.

    Inversely proportional to how common the class is, so getting one of the
    82 dermatofibromas wrong costs far more than one of the 4716 moles.
    """
    counts = torch.tensor(
        [(metadata["dx"] == name).sum() for name in CLASSES], dtype=torch.float
    )
    return counts.sum() / (len(CLASSES) * counts)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training set, updating the weights as it goes."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    """One pass with the weights frozen, just to see how well it is doing."""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            total_loss += loss_fn(predictions, labels).item()
            correct += (predictions.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


# Guarded, because evaluate.py imports build_model from here. Without the
# guard, that import would run the whole training again.
if __name__ == "__main__":
    # Before the model is built, because building it is the first thing that
    # draws from the generators.
    set_seed()

    name = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    model = build_model(name)
    settings = MODELS[name]

    WEIGHTS_DIR.mkdir(exist_ok=True)

    device = pick_device()
    print(f"training {name} on {device}")

    train_loader, val_loader, _ = build_dataloaders(
        augment=True, mean=settings["mean"], std=settings["std"]
    )
    train_metadata, _, _ = split_metadata(load_metadata())

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_metadata).to(device))
    # AdamW rather than Adam with weight_decay set. Adam adds the L2 penalty
    # to the gradient before the adaptive step rescales it, so how hard a
    # weight is pulled towards zero ends up depending on how large its
    # gradients have been. AdamW applies the decay separately, which is the
    # whole reason it exists and the only reason to prefer it here.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=WEIGHT_DECAY,
    )

    # Cuts the learning rate when the validation loss stops improving, rather
    # than on a timetable. It watches the same number early stopping watches,
    # so the two cannot disagree about whether progress has stalled.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_DROP, patience=LR_PATIENCE
    )

    # Validation loss decides which weights are worth keeping. Training loss
    # keeps falling even once the network has started memorising, so it cannot
    # be the one to judge.
    best_val_loss = float("inf")
    epochs_since_best = 0

    for epoch in range(1, settings["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        line = (
            f"epoch {epoch:2}: train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val accuracy {val_accuracy:.2%}, "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
            save_weights(model.state_dict(), settings["weights"])
            line += "  <- saved"
        else:
            epochs_since_best += 1

        print(line)

        if epochs_since_best >= EARLY_STOPPING_PATIENCE:
            # Said out loud, because a run that just ends looks like a crash
            # and a log that stops early needs to say it meant to.
            print(
                f"stopping early: {EARLY_STOPPING_PATIENCE} epochs "
                "without a better validation loss"
            )
            break

    # The path the config names is the old .pth; save_weights writes the
    # safetensors sibling, and this has to say which file actually appeared.
    saved_to = safetensors_beside(settings["weights"])
    print(f"best validation loss: {best_val_loss:.4f}, weights in {saved_to}")
