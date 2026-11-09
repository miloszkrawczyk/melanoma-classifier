"""Reads and writes trained weights, preferring safetensors over the pickle.

A .pth file is a Python pickle. Step 8a made loading one safe by refusing
anything that is not a tensor, which closes the hole but leaves the format.
safetensors has nothing to execute in the first place: a JSON header giving
every tensor its name, shape and dtype, then the raw bytes. For weights that
get published it is the right format, and it is what the Hugging Face Hub
expects, which is where this project tells people to fetch them from.

Both formats are read, on purpose. A checkout from before this change, or a
container image built against the old weights, has only the .pth and has to
keep working.

Nothing here takes a device. Measured: a state dict read onto the CPU loads
into a model sitting on MPS, the parameters stay on MPS, and the network's
output is identical either way. load_state_dict copies into parameters that
already live on the device, so where the file was read to does not matter.
"""

import torch
from safetensors.torch import load_file, save_file


def safetensors_beside(path):
    """The safetensors file belonging next to a given weights path."""
    return path.with_suffix(".safetensors")


def pickle_beside(path):
    """The legacy .pth file belonging next to a given weights path."""
    return path.with_suffix(".pth")


def load_weights(path):
    """Read a state dict, from safetensors if there is one, else from the pickle.

    The extension on `path` is not used. Both candidates are derived from it,
    so the config can name whichever format it prefers and callers never have
    to know which one is on disk. That is what keeps an August checkout, with
    only the .pth, working against a config that asks for safetensors.
    """
    safe = safetensors_beside(path)
    if safe.exists():
        return load_file(safe)

    # weights_only=True because this file is a pickle and unpickling runs what
    # the file says to run. See the comment in predict.py for why that matters
    # here rather than in theory.
    return torch.load(pickle_beside(path), map_location="cpu", weights_only=True)


def save_weights(state_dict, path):
    """Write a state dict as safetensors, whatever extension the path carries.

    Only ever writes the new format. Nothing in the project produces a pickle
    any more; the reader keeps understanding one.
    """
    save_file(state_dict, safetensors_beside(path))
