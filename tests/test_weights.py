"""Guards the one keyword that stops a weights file from running code.

A .pth file is a Python pickle, and unpickling executes whatever the file says
to execute. torch.load takes weights_only=True to read tensors and refuse
everything else. The README tells people to download resnet.pth from the
Hugging Face Hub, so the service reads a file that arrived over the network,
which is what makes this worth a test rather than a comment.

This one is unusual: it reads the project's own source instead of calling it.
There is no way to write the ordinary version. Checking that our weights load
safely would need the weights, which CI does not have, and it would be testing
torch rather than this project. The real risk is that someone adds a fourth
load later, or drops the keyword while tidying, and nothing else in the suite
would notice. Parsing the source is what notices.

ast rather than a text search, because the calls are split across lines and
grepping would both miss them and match comments mentioning torch.load.

What it does not catch: `from torch import load` followed by a bare `load(...)`
would slip past, since this looks for the attribute spelling. Everything here
writes torch.load, which is also what PyTorch's own documentation does, so the
gap is worth knowing about rather than worth code.
"""

import ast
from pathlib import Path

import torch

from src.config import PROJECT_ROOT
from src.device import pick_device
from src.models.resnet import build_resnet18
from src.weights import (
    load_weights,
    pickle_beside,
    safetensors_beside,
    save_weights,
)


def python_files():
    """Every file in the project's own code, tests excluded.

    Tests are left out on purpose: a test is allowed to load a pickle it wrote
    itself a line earlier. api/ and frontend/ have no torch.load today and are
    swept anyway, because the point is to catch the one somebody adds later.
    """
    return sorted(
        path
        for folder in ("src", "scripts", "api", "frontend")
        for path in PROJECT_ROOT.glob(f"{folder}/**/*.py")
    )


def torch_load_calls(path):
    """Every torch.load(...) in one file, as (line number, keyword names)."""
    tree = ast.parse(path.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func
        called_torch_load = (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id == "torch"
        )
        if called_torch_load:
            yield node.lineno, {keyword.arg for keyword in node.keywords}


def test_the_project_has_some_weight_loading_to_check():
    """Keeps the test below from passing by finding nothing at all."""
    found = [
        (path, line) for path in python_files() for line, _ in torch_load_calls(path)
    ]

    assert found, "no torch.load anywhere, so the check below proves nothing"


def test_every_weight_load_refuses_to_unpickle():
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{line}"
        for path in python_files()
        for line, keywords in torch_load_calls(path)
        if "weights_only" not in keywords
    ]

    assert not offenders, (
        "these torch.load calls would run whatever is in the file: "
        + ", ".join(offenders)
    )


def a_state_dict():
    """Two tensors of the dtypes a real state dict holds.

    float32 for the weights, int64 for BatchNorm's num_batches_tracked, which
    is the pair that both weight files in this project actually contain.
    """
    torch.manual_seed(0)

    return {
        "layer.weight": torch.rand(4, 3, 3, 3),
        "layer.num_batches_tracked": torch.tensor(7),
    }


def test_a_saved_state_dict_comes_back_unchanged(tmp_path):
    """Bit for bit, not close enough. Weights that shift are weights that lie."""
    original = a_state_dict()

    save_weights(original, tmp_path / "model.pth")
    restored = load_weights(tmp_path / "model.pth")

    assert restored.keys() == original.keys()
    for name, tensor in original.items():
        assert torch.equal(restored[name], tensor)
        assert restored[name].dtype == tensor.dtype


def test_saving_writes_the_safetensors_sibling(tmp_path):
    """Whatever extension the caller names, only the new format is written."""
    save_weights(a_state_dict(), tmp_path / "model.pth")

    assert (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "model.pth").exists()


def test_safetensors_wins_when_both_formats_are_there(tmp_path):
    """A converted checkout has both, and the new one has to be the one read."""
    torch.save({"which": torch.tensor([1.0])}, tmp_path / "model.pth")
    save_weights({"which": torch.tensor([2.0])}, tmp_path / "model.pth")

    assert load_weights(tmp_path / "model.pth")["which"].item() == 2.0


def test_the_pickle_is_still_read_when_there_is_no_safetensors(tmp_path):
    """The reason the fallback exists: an August checkout has only the .pth."""
    torch.save({"which": torch.tensor([1.0])}, tmp_path / "model.pth")

    assert not safetensors_beside(tmp_path / "model.pth").exists()
    assert load_weights(tmp_path / "model.pth")["which"].item() == 1.0


def test_a_loaded_state_dict_goes_into_a_model_on_the_device(tmp_path):
    """Why load_weights takes no device.

    The file is read onto the CPU and load_state_dict copies into parameters
    that already live wherever the model was put, so the parameters stay there.
    """
    device = pick_device()
    model = build_resnet18(pretrained=False).to(device)

    save_weights(model.state_dict(), tmp_path / "model.pth")
    model.load_state_dict(load_weights(tmp_path / "model.pth"))

    assert next(model.parameters()).device.type == device.type


def test_a_safetensors_path_still_finds_an_old_pickle(tmp_path):
    """The case the config change created, and the reason for the fallback.

    MODELS now names weights/resnet.safetensors. An August checkout has only
    weights/resnet.pth, and asking for the one has to find the other. Reading
    the named path directly would have handed safetensors to the unpickler.
    """
    torch.save({"which": torch.tensor([1.0])}, tmp_path / "model.pth")
    asked_for = tmp_path / "model.safetensors"

    assert not asked_for.exists()
    assert load_weights(asked_for)["which"].item() == 1.0


def test_the_two_siblings_are_derived_from_either_extension():
    """Both helpers ignore the extension they are handed, on purpose."""
    for named in (Path("w/model.pth"), Path("w/model.safetensors")):
        assert safetensors_beside(named).name == "model.safetensors"
        assert pickle_beside(named).name == "model.pth"
