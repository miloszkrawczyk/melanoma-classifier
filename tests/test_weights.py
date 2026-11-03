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

from src.config import PROJECT_ROOT


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
