"""Counts how often each caller has asked lately, and says when it is too often.

Kept out of api/main.py for the same reason src/uploads.py is: importing the
app builds a Predictor, which costs eight seconds and 43 MB of weights that CI
does not have. Nothing in here knows about HTTP, so it can be tested with a
dictionary and a made-up clock.

The state comes in as an argument rather than living in this module, so two
tests cannot leak into each other and there is no hidden thing to reset.
"""

from src.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


class TooManyRequests(Exception):
    """This caller has asked more often than the service is willing to answer."""


def note_request(recent, client, now):
    """Record one request, or refuse it because this client has had its share.

    `recent` is one deque of (client, when) pairs covering every caller, not a
    deque per caller. That keeps the memory tied to how many requests arrived
    in the last window rather than to how many addresses have ever been seen,
    so nothing has to be swept up later. Counting means walking the window,
    which at sixty requests a minute is a few dozen comparisons.
    """
    while recent and now - recent[0][1] > RATE_LIMIT_WINDOW_SECONDS:
        recent.popleft()

    if sum(1 for who, _ in recent if who == client) >= RATE_LIMIT_REQUESTS:
        raise TooManyRequests(
            f"too many requests, the limit is {RATE_LIMIT_REQUESTS} "
            f"every {RATE_LIMIT_WINDOW_SECONDS} seconds"
        )

    recent.append((client, now))
