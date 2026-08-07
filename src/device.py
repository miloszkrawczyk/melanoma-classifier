"""Where the maths gets done.

Its own file because both training and serving need it, and serving should not
have to import the training code to get at it. That import pulled pandas, the
dataset splitting and the from-scratch CNN into the web service, none of which
it has any use for.
"""

import torch


def pick_device():
    """Use the Mac's graphics chip if it is there, otherwise the processor."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
