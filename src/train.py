"""Training loop for SkinLesionCNN: forward pass, loss, backward pass,
optimiser step, repeated over batches and epochs.

Two things here that the MNIST project did not need. The loss carries class
weights, because nv accounts for two thirds of the data and without them the
network learns to answer "ordinary mole" and stop thinking. And every epoch
ends with a pass over the validation set, so the weights that get saved are
the ones that did best on photos the network never trained on.
"""

import torch
import torch.nn as nn

from src.config import CLASSES, EPOCHS, LEARNING_RATE, WEIGHTS_PATH
from src.data import build_dataloaders, load_metadata, split_metadata
from src.model import SkinLesionCNN


def pick_device():
    """Use the Mac's graphics chip if it is there, otherwise the processor."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


# Guarded, because evaluate.py imports pick_device and class_weights from
# here. Without the guard, that import would run the whole training again.
if __name__ == "__main__":
    device = pick_device()
    print(f"training on {device}")

    train_loader, val_loader, _ = build_dataloaders()
    train_metadata, _, _ = split_metadata(load_metadata())

    model = SkinLesionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_metadata).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Validation loss decides which weights are worth keeping. Training loss
    # keeps falling even once the network has started memorising, so it cannot
    # be the one to judge.
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        line = (
            f"epoch {epoch:2}: train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val accuracy {val_accuracy:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            line += "  <- saved"

        print(line)

    print(f"best validation loss: {best_val_loss:.4f}, weights in {WEIGHTS_PATH.name}")
