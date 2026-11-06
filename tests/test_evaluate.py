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
from src.config import CLASSES, METADATA_CSV
from src.data import load_metadata, split_metadata
from src.evaluate import (
    bootstrap_malignant_intervals,
    confusion_matrix,
    format_interval,
    is_malignant,
    majority_class_baseline,
    malignant_auc,
    parse_arguments,
    print_report,
    wilson_interval,
)


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


def one_hot(labels):
    """Probabilities that put everything on the given class, as a perfect model would."""
    return torch.nn.functional.one_hot(torch.tensor(labels), len(CLASSES)).float()


def test_report_names_the_split_and_counts_the_images(capsys):
    # One of each malignant class and one benign, so the AUC lines have both
    # sides of the question to work with.
    labels = [CLASSES.index(dx) for dx in ("akiec", "bcc", "mel", "nv")]
    print_report(one_hot(labels), torch.tensor(labels), "resnet", "val")

    out = capsys.readouterr().out

    assert "scoring resnet on the val set, 4 images" in out
    assert "accuracy: 100.0%" in out
    assert "PR-AUC on malignant:      1.000" in out


def test_an_unknown_model_is_rejected():
    """A bare KeyError out of the config dict is not a usable error message."""
    with pytest.raises(SystemExit, match='expected "cnn" or "resnet"'):
        parse_arguments(["src.evaluate", "rsnet"])


def test_majority_class_baseline_names_the_commonest_class():
    # Four moles and one melanoma: answering nv every time scores 80%.
    actual = torch.tensor([CLASSES.index("nv")] * 4 + [CLASSES.index("mel")])

    assert majority_class_baseline(actual) == ("nv", 0.8)


# Skipped per test rather than through a module-level pytestmark, because every
# other test in this file runs without the dataset and should keep doing so.
@pytest.mark.skipif(
    not METADATA_CSV.exists(),
    reason="needs the HAM10000 metadata, see the README for how to fetch it",
)
def test_majority_class_baseline_matches_the_documented_numbers():
    """Guards the two baselines quoted in results/ and in the README.

    Reads the label column and nothing else: no model, no images, no
    predictions. The share of a class within a split is a property of the
    split, so this does not count as touching the test set.

    These moved when the split started keeping class proportions. They used to
    be 67.2% and 64.1%, and the 3.1 point gap between them was the symptom:
    the two sets did not hold the same mix. They now agree to a tenth of a
    point, which is the fix showing up in the one number that measures it.
    """
    _, val, test = split_metadata(load_metadata())

    val_name, val_share = majority_class_baseline(torch.tensor(val["label"].to_numpy()))
    test_name, test_share = majority_class_baseline(
        torch.tensor(test["label"].to_numpy())
    )

    assert (val_name, f"{val_share:.1%}") == ("nv", "66.9%")
    assert (test_name, f"{test_share:.1%}") == ("nv", "67.0%")


def test_is_malignant_marks_the_three_malignant_diagnoses():
    labels = torch.tensor([CLASSES.index(dx) for dx in ("akiec", "bcc", "mel", "nv")])

    assert is_malignant(labels).tolist() == [True, True, True, False]


def test_malignant_auc_is_perfect_when_the_scores_separate_the_classes():
    actual = torch.tensor([CLASSES.index("mel"), CLASSES.index("nv")])
    # The melanoma gets all its mass on a malignant class, the mole none of it.
    probabilities = one_hot([CLASSES.index("mel"), CLASSES.index("nv")])

    assert malignant_auc(probabilities, actual) == (1.0, 1.0)


def test_malignant_auc_is_a_coin_flip_when_every_score_is_the_same():
    """A model that cannot tell the two apart has to score 0.5, not higher.

    This is the test that would catch the scores being ranked the wrong way
    round, which inverts the measure instead of breaking it.
    """
    actual = torch.tensor([CLASSES.index("mel"), CLASSES.index("nv")])
    probabilities = torch.full((2, len(CLASSES)), 1 / len(CLASSES))

    roc_auc, _ = malignant_auc(probabilities, actual)

    assert roc_auc == 0.5


def test_wilson_interval_matches_the_recorded_test_recall():
    """289 of 314 malignant lesions caught, the headline number in results/05."""
    point, low, high = wilson_interval(289, 314)

    assert f"{point:.1%}" == "92.0%"
    assert f"{low:.1%} - {high:.1%}" == "88.5% - 94.5%"


def test_wilson_interval_stays_inside_zero_and_one():
    """The textbook formula does not, which is the reason for using Wilson.

    At 0 of 10 it returns a range of zero width, and at 1 of 12 a negative
    lower bound. Neither is a proportion.
    """
    _, none_low, none_high = wilson_interval(0, 10)
    _, all_low, all_high = wilson_interval(10, 10)

    assert none_low == 0.0
    assert 0.0 < none_high < 1.0
    assert 0.0 < all_low < 1.0
    assert all_high == 1.0


def test_wilson_interval_survives_a_class_with_no_images():
    """print_per_class walks all seven classes, including any the split lacks."""
    assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_interval_is_far_wider_for_a_small_class():
    """The whole point: 75% off 12 images means much less than 83% off 967.

    Both are real numbers from results/05, df against nv, and the table gave
    them equal weight until now.
    """
    _, df_low, df_high = wilson_interval(9, 12)
    _, nv_low, nv_high = wilson_interval(806, 967)

    assert (df_high - df_low) > 0.40
    assert (nv_high - nv_low) < 0.06


def malignant_calls(size, truly_malignant, correctly_called):
    """A split of the given size with a known recall, built by hand.

    Avoids needing a model: these are just the two boolean columns that recall
    and precision are computed from.
    """
    really = torch.zeros(size, dtype=torch.bool)
    really[:truly_malignant] = True

    called = torch.zeros(size, dtype=torch.bool)
    called[:correctly_called] = True

    return called, really


def test_format_interval_names_the_confidence_level():
    assert format_interval(0.723, 0.824) == "(95% CI 72.3% - 82.4%)"


def test_bootstrap_interval_brackets_the_measured_recall():
    called, really = malignant_calls(1000, truly_malignant=200, correctly_called=150)

    (recall_low, recall_high), _ = bootstrap_malignant_intervals(called, really)

    assert recall_low < 0.75 < recall_high


def test_bootstrap_intervals_are_repeatable():
    """Two runs of the same command have to print the same range.

    The generator is seeded for this reason. Without it the bounds would
    wander between runs and nobody could quote them.
    """
    called, really = malignant_calls(500, truly_malignant=100, correctly_called=80)

    assert bootstrap_malignant_intervals(called, really) == (
        bootstrap_malignant_intervals(called, really)
    )


def test_bootstrap_interval_narrows_on_a_bigger_split():
    """Same recall, ten times the images, so the range has to tighten."""
    small = malignant_calls(200, truly_malignant=40, correctly_called=30)
    large = malignant_calls(2000, truly_malignant=400, correctly_called=300)

    (small_low, small_high), _ = bootstrap_malignant_intervals(*small)
    (large_low, large_high), _ = bootstrap_malignant_intervals(*large)

    assert (large_high - large_low) < (small_high - small_low) / 2
