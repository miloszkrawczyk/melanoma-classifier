"""Tests for the scoring module, which until now had none.

The reason it had none is the thing these guard: the file used to run its
whole report at import time, so collecting it cost a model build, 43 MB of
weights off disk and a pass over a split. None of the tests below need the
dataset or the weights, which is only true because the report now sits behind
a __main__ guard.
"""

import pytest
import torch

import src.evaluate as evaluate
from src.evaluate import confusion_matrix, parse_arguments, print_report


def test_importing_the_module_runs_nothing():
    """The import at the top of this file is the real test.

    There is no way to assert "nothing happened", so this checks the module
    merely loaded. Before the guard, pytest could not even collect this file
    without the trained weights on disk.
    """
    assert callable(evaluate.print_report)


def test_arguments_default_to_the_cnn_on_validation():
    assert parse_arguments(["src.evaluate"]) == ("cnn", "val")


def test_arguments_read_the_model_and_the_split():
    assert parse_arguments(["src.evaluate", "resnet", "test"]) == ("resnet", "test")


def test_an_unknown_split_is_rejected():
    """A typo has to stop the run, not silently score validation.

    Scoring the wrong split would print a plausible report for the wrong
    numbers, which is worse than failing.
    """
    with pytest.raises(SystemExit):
        parse_arguments(["src.evaluate", "resnet", "tset"])


def test_confusion_matrix_puts_the_truth_in_rows():
    # Two lesions that really were class 0, one of them called class 1.
    actual = torch.tensor([0, 0])
    predicted = torch.tensor([0, 1])

    matrix = confusion_matrix(predicted, actual)

    assert int(matrix[0, 0]) == 1
    assert int(matrix[0, 1]) == 1
    assert int(matrix[1, 0]) == 0
    assert int(matrix.sum()) == len(actual)


def test_report_names_the_split_and_counts_the_images(capsys):
    actual = torch.tensor([0, 1, 2])
    print_report(actual, actual, "resnet", "val")

    out = capsys.readouterr().out

    assert "scoring resnet on the val set, 3 images" in out
    assert "accuracy: 100.0%" in out


def test_an_unknown_model_is_rejected():
    """A bare KeyError out of the config dict is not a usable error message."""
    with pytest.raises(SystemExit, match='expected "cnn" or "resnet"'):
        parse_arguments(["src.evaluate", "rsnet"])
