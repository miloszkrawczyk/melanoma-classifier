"""Draws the training curves out of a results file.

Reads the epoch lines a training run printed, then writes a png next to the
file and prints a rough text version to paste back into it. Keeping the text
version means the results file still makes sense on its own, without opening
an image.

    uv run python -m scripts.plot_training results/01-cnn-baseline.txt
"""

import re
import sys
from pathlib import Path

import matplotlib

# Draw straight to a file instead of opening a window, since nothing here
# needs a screen.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_LINE = re.compile(
    r"epoch\s+(\d+): train loss ([\d.]+), val loss ([\d.]+), val accuracy ([\d.]+)%"
)


def read_epochs(path):
    """Pull the numbers out of the log lines and ignore everything else."""
    epochs, train_losses, val_losses, accuracies = [], [], [], []

    for line in path.read_text().splitlines():
        found = EPOCH_LINE.search(line)
        if found:
            epochs.append(int(found.group(1)))
            train_losses.append(float(found.group(2)))
            val_losses.append(float(found.group(3)))
            accuracies.append(float(found.group(4)))

    return epochs, train_losses, val_losses, accuracies


def save_png(path, epochs, train_losses, val_losses, accuracies):
    """Losses on the left, accuracy on the right."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.plot(epochs, train_losses, marker="o", label="train loss")
    left.plot(epochs, val_losses, marker="o", label="val loss")
    best = val_losses.index(min(val_losses))
    left.axvline(epochs[best], color="grey", linestyle="--", label="saved weights")
    left.set_xlabel("epoch")
    left.set_ylabel("loss")
    left.legend()
    left.grid(alpha=0.3)

    right.plot(epochs, accuracies, marker="o", color="green", label="val accuracy")
    # A model answering nv every time would land here, so anything below this
    # line is worse than not thinking at all.
    right.axhline(67.2, color="red", linestyle="--", label="always nv (67.2%)")
    right.set_xlabel("epoch")
    right.set_ylabel("accuracy %")
    right.legend()
    right.grid(alpha=0.3)

    figure.suptitle(path.stem)
    figure.tight_layout()

    png_path = path.with_suffix(".png")
    figure.savefig(png_path, dpi=120)
    return png_path


def print_ascii(epochs, train_losses, val_losses, height=14):
    """A rough plot made of characters, for pasting into the results file."""
    lowest = min(min(train_losses), min(val_losses))
    highest = max(max(train_losses), max(val_losses))

    def row_of(value):
        """Which line from the top this value belongs on."""
        share = (value - lowest) / (highest - lowest)
        return round((1 - share) * (height - 1))

    for row in range(height):
        value = highest - row * (highest - lowest) / (height - 1)
        line = f"{value:5.2f} |"
        for i in range(len(epochs)):
            if row_of(train_losses[i]) == row and row_of(val_losses[i]) == row:
                line += "  *"
            elif row_of(train_losses[i]) == row:
                line += "  t"
            elif row_of(val_losses[i]) == row:
                line += "  v"
            else:
                line += "   "
        print(line)

    print(" " * 6 + "+" + "---" * len(epochs))
    print(" " * 7 + "".join(f"{epoch:>3}" for epoch in epochs))
    print(" " * 7 + "epoch")
    print()
    print("t = train loss, v = val loss, * = both on the same line")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("give me a results file, e.g. results/01-cnn-baseline.txt")

    path = Path(sys.argv[1])
    epochs, train_losses, val_losses, accuracies = read_epochs(path)

    if not epochs:
        raise SystemExit(f"no epoch lines found in {path}")

    png_path = save_png(path, epochs, train_losses, val_losses, accuracies)
    print(f"saved {png_path}")
    print()
    print_ascii(epochs, train_losses, val_losses)
