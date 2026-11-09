"""Decides whether an uploaded file is a picture this service is willing to decode.

Kept out of api/main.py for the same reason predict.py is: nothing in here
knows about HTTP, so it can be called from a test or a script without a web
server, and without the 43 MB of weights that importing the app pulls in.
It answers with its own two exceptions and lets the caller pick status codes.

Both limits exist because the service answers requests in parallel now, so one
expensive upload is no longer alone with the machine.
"""

import os

from PIL import Image, UnidentifiedImageError

from src.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_PIXELS


class NotAnImage(Exception):
    """The file was not a picture at all."""


class TooLarge(Exception):
    """A picture, but more of one than this service will take."""


def open_upload(stream):
    """Open a file object as a picture, or say why it was turned away.

    The returned image is not decoded yet. PIL reads the header and stops,
    which is what lets the size checks happen before any memory is committed.
    """
    # Free to ask, because whatever called this has already buffered the body,
    # so the length is known. Which is also the limit of what this can do:
    # nothing here stops the bytes arriving or being spooled to a temporary
    # file, only from reaching the model. A real byte cap belongs in the proxy.
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)

    if size > MAX_UPLOAD_BYTES:
        # The numbers go in the message so the caller can see by how much they
        # went over instead of guessing.
        raise TooLarge(
            f"that file is {size / 1024 / 1024:.1f} MB, "
            f"the limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB"
        )

    try:
        image = Image.open(stream)
    except UnidentifiedImageError:
        raise NotAnImage("that file is not an image")

    # The dimensions come from the header, so this refuses the picture while it
    # is still a few tens of kilobytes on disk rather than a few hundred
    # megabytes in memory. That is the whole value of checking here, and the
    # reason it has to happen before the decode below.
    pixels = image.width * image.height
    if pixels > MAX_UPLOAD_PIXELS:
        raise TooLarge(
            f"that image is {image.width}x{image.height}, "
            f"{pixels // 10**6} megapixels against a limit of "
            f"{MAX_UPLOAD_PIXELS // 10**6}"
        )

    # Image.open has only read the header, so a file whose header is fine and
    # whose pixels are cut short gets this far happily. Measured: half of a
    # valid JPEG opens without complaint and raises OSError on decode. Left
    # alone, that OSError surfaces later inside the Predictor, where nothing
    # can tell whose fault it was, and the caller gets a 500 for their own
    # truncated upload. An upload cut short by a dropped connection does this.
    #
    # So the decode happens here, where it can still be answered for. Caught
    # separately from the open above on purpose: UnidentifiedImageError is
    # itself a subclass of OSError, and keeping the two apart says which stage
    # reports what.
    try:
        image.load()
    except OSError:
        raise NotAnImage("that image is damaged or was cut short")

    return image
