"""Rewrites the trained weights from the old pickle into safetensors.

A .pth file is a Python pickle, so reading one runs whatever it says to run.
src/weights.py already refuses to unpickle anything but tensors, which closes
that, but the format itself is still a pickle. safetensors has nothing to
execute: a JSON header of names, shapes and dtypes, then the raw bytes. For
weights that get published that is the format to use, and it is what the
Hugging Face Hub expects.

Run it once, after which training writes the new format by itself. The old
.pth files are left where they are rather than deleted: they are what an
August checkout has, the reader still understands them, and nothing is gained
by making the migration irreversible.

    uv run python -m scripts.convert_weights

Every tensor is read back and compared afterwards. A silent change in a weight
would not break anything that fails loudly, it would just shift the metrics,
and this is the only moment where catching that is cheap.
"""

import torch

from src.config import MODELS
from src.weights import (
    load_weights,
    pickle_beside,
    safetensors_beside,
    save_weights,
)

for name, settings in MODELS.items():
    source = pickle_beside(settings["weights"])
    destination = safetensors_beside(settings["weights"])

    if not source.exists():
        print(f"{name}: no {source.name}, nothing to convert")
        continue

    original = torch.load(source, map_location="cpu", weights_only=True)
    save_weights(original, settings["weights"])
    restored = load_weights(settings["weights"])

    assert restored.keys() == original.keys(), f"{name}: tensor names changed"
    for key, tensor in original.items():
        assert torch.equal(restored[key], tensor), f"{name}: {key} changed"
        assert restored[key].dtype == tensor.dtype, f"{name}: {key} changed dtype"

    print(
        f"{name}: {source.name} {source.stat().st_size / 1e6:.1f} MB "
        f"-> {destination.name} {destination.stat().st_size / 1e6:.1f} MB, "
        f"{len(original)} tensors verified identical"
    )
