"""Tests for the guard that turns away photos which are not skin lesions.

These build their tensors by hand, so they need neither the dataset nor
trained weights and will run anywhere.
"""

import pytest
import torch
from PIL import Image

from src.config import MODELS, PROJECT_ROOT, TTA_QUARTER_TURNS
from src.predict import Predictor, looks_like_skin, roughness


def skin_like_tensor():
    """Something with roughly the colour and texture of a HAM10000 photo.

    The blur matters: raw noise is far rougher than any photograph, so it has
    to be smoothed until neighbouring pixels sit close together. These numbers
    land at roughness 0.0100, inside the 0.0040 to 0.0494 range measured on
    real images.
    """
    torch.manual_seed(0)
    base = torch.tensor([0.76, 0.55, 0.57]).view(3, 1, 1)
    texture = torch.randn(3, 224, 224) * 0.08
    texture = torch.nn.functional.avg_pool2d(texture, 3, stride=1, padding=1)
    return (base + texture).clamp(0, 1)


def test_roughness_separates_noise_from_flat_colour():
    torch.manual_seed(0)
    noise = torch.rand(3, 224, 224)
    flat = torch.full((3, 224, 224), 0.5)

    assert roughness(noise) > 0.1
    assert roughness(flat) == 0.0


def test_skin_like_image_is_accepted():
    assert looks_like_skin(skin_like_tensor())


def test_answer_is_a_plain_bool():
    """`is True` rather than a truthiness check, on purpose.

    A torch tensor passes a plain assert, so the earlier version of this test
    went green while the function returned tensor(True). FastAPI could not
    turn that into JSON and served an empty object instead.
    """
    assert looks_like_skin(skin_like_tensor()) is True
    assert looks_like_skin(torch.full((3, 224, 224), 0.5)) is False


def test_noise_is_rejected():
    torch.manual_seed(0)
    assert not looks_like_skin(torch.rand(3, 224, 224))


def test_flat_colour_is_rejected():
    assert not looks_like_skin(torch.full((3, 224, 224), 0.5))


def test_grey_image_is_rejected():
    """Grey passes the red check, since every channel is equal. Saturation
    is what catches it."""
    grey = skin_like_tensor().mean(dim=0, keepdim=True).repeat(3, 1, 1)

    assert not looks_like_skin(grey)


def test_green_image_is_rejected():
    green = torch.zeros(3, 224, 224)
    green[1] = 0.6

    assert not looks_like_skin(green)


def test_four_quarter_turns_return_to_the_start():
    """Why quarter turns and not the angles training uses.

    Turning by a right angle is a permutation of pixels: do it four times and
    the original is back, bit for bit. An arbitrary angle interpolates, so it
    invents pixel values and leaves empty corners, and four of them would not
    come back to anything.
    """
    original = skin_like_tensor()

    turned = original
    for _ in range(4):
        turned = torch.rot90(turned, k=1, dims=[1, 2])

    assert torch.equal(turned, original)


def test_a_quarter_turn_actually_changes_the_picture():
    """Guards against the averaging being four copies of one reading.

    If rot90 were a no-op here, TTA would silently average the same answer
    four times and the measured gain would be impossible.
    """
    original = skin_like_tensor()

    assert not torch.equal(torch.rot90(original, k=1, dims=[1, 2]), original)


def test_averaging_rotations_keeps_a_probability_distribution():
    """Four distributions averaged are still one distribution.

    The malignant call adds three of these together and compares against a
    threshold, so they have to keep summing to one.
    """
    torch.manual_seed(0)
    four = torch.softmax(torch.randn(len(TTA_QUARTER_TURNS), 7), dim=1)

    averaged = four.mean(dim=0)

    assert averaged.sum().item() == pytest.approx(1.0)
    assert (averaged >= 0).all()


def test_the_configured_turns_are_the_four_distinct_ones():
    """Averaging the same reading twice would weight it twice, not help."""
    assert sorted(TTA_QUARTER_TURNS) == [0, 1, 2, 3]


# Needs the trained weights, so it skips where they are not there, the same
# way the dataset tests do. Everything above this line runs anywhere.
@pytest.mark.skipif(
    not MODELS["resnet"]["weights"].exists(),
    reason="needs the trained weights, see the README for how to fetch them",
)
def test_passing_the_diagnosis_does_not_change_the_heatmap():
    """The shortcut must be a shortcut and nothing more.

    explain() can work out which diagnosis to draw for, or be told. Being told
    saves a second four-rotation batch, which the interface was paying for
    every analysis. It must not change the picture: an off-by-one in the class
    index would quietly draw the map for a neighbouring diagnosis.
    """
    predictor = Predictor()
    photo = Image.open(PROJECT_ROOT / "frontend" / "examples" / "ISIC_0025303.jpg")

    diagnosis = predictor.predict(photo)["diagnosis"]

    worked_out = predictor.explain(photo)
    told = predictor.explain(photo, diagnosis)

    assert told.tobytes() == worked_out.tobytes()
