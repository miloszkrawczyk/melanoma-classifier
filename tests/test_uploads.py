"""Tests for the limits the service puts on what it will accept.

None of these need the dataset, the weights or a web server: open_upload takes
a plain file object and answers with its own exceptions. That is the reason it
lives in src/ rather than next to the Predictor, where importing it would have
cost 8 seconds and 43 MB of weights that CI does not have.
"""

import io

import pytest
from PIL import Image

from src.config import MAX_UPLOAD_BYTES
from src.uploads import NotAnImage, TooLarge, open_upload


def photo(width=600, height=450, payload_size=None):
    """A JPEG the shape of a HAM10000 photo, optionally padded to a given size."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (195, 140, 145)).save(buffer, format="JPEG")
    data = buffer.getvalue()

    if payload_size is not None:
        # Trailing bytes a JPEG decoder ignores, so the file grows while the
        # picture stays the same. Cheaper than generating a genuinely huge one.
        data += b"\0" * (payload_size - len(data))

    return io.BytesIO(data)


def test_a_normal_photo_is_accepted():
    image = open_upload(photo())

    assert (image.width, image.height) == (600, 450)


def test_a_file_that_is_not_an_image_is_refused():
    with pytest.raises(NotAnImage):
        open_upload(io.BytesIO(b"this is a text file, not a photograph"))


def test_a_file_over_the_byte_limit_is_refused():
    with pytest.raises(TooLarge) as refusal:
        open_upload(photo(payload_size=MAX_UPLOAD_BYTES + 1))

    # The message has to say how big it was, or the caller is left guessing.
    assert "MB" in str(refusal.value)


def test_an_image_over_the_pixel_limit_is_refused(monkeypatch):
    """The decompression bomb the byte limit misses.

    The limit is lowered rather than building a 50 megapixel image, which
    would allocate 150 MB to prove a comparison.
    """
    monkeypatch.setattr("src.uploads.MAX_UPLOAD_PIXELS", 1000)

    with pytest.raises(TooLarge) as refusal:
        open_upload(photo(width=200, height=200))

    assert "megapixel" in str(refusal.value)


def test_the_byte_limit_is_checked_before_the_header():
    """A file can fail both tests, and the cheaper answer is the useful one.

    Size is known without reading a byte. Dimensions need the header parsed,
    which means handing a file already known to be too big to a decoder.
    """
    monkeypatch_free = photo(payload_size=MAX_UPLOAD_BYTES + 1)

    with pytest.raises(TooLarge) as refusal:
        open_upload(monkeypatch_free)

    assert "MB" in str(refusal.value)


def test_the_stream_is_rewound_for_the_caller():
    """open_upload seeks to the end to measure, so it has to seek back.

    Without that, PIL would be handed a stream sitting at EOF and every
    upload would look like it was not an image.
    """
    image = open_upload(photo())

    assert image.width == 600


def test_a_truncated_photo_is_refused():
    """The header parses and the pixels do not, which used to reach the model.

    Measured: half of a valid JPEG opens without complaint and raises OSError
    on decode. Before this, that OSError surfaced inside the Predictor and the
    caller got a 500 for a file they had cut short themselves.
    """
    whole = photo().getvalue()

    with pytest.raises(NotAnImage):
        open_upload(io.BytesIO(whole[: len(whole) // 2]))


def test_an_accepted_photo_is_decoded_and_usable():
    """Forcing the decode must not leave the image unusable afterwards."""
    image = open_upload(photo())

    assert image.convert("RGB").resize((224, 224)).size == (224, 224)
