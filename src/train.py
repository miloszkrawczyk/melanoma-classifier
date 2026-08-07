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

import sys

import torch
import torch.nn as nn

from src.config import CLASSES, MODELS, WEIGHTS_DIR
from src.data import build_dataloaders, load_metadata, split_metadata
from src.device import pick_device
from src.models.cnn import SkinLesionCNN
from src.models.resnet import build_resnet18


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
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

    # Validation loss decides which weights are worth keeping. Training loss
    # keeps falling even once the network has started memorising, so it cannot
    # be the one to judge.
    best_val_loss = float("inf")

    for epoch in range(1, settings["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        line = (
            f"epoch {epoch:2}: train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val accuracy {val_accuracy:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), settings["weights"])
            line += "  <- saved"

        print(line)

    print(f"best validation loss: {best_val_loss:.4f}, weights in {settings['weights']}")
