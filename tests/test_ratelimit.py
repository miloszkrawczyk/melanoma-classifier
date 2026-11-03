"""Tests for the request counter behind the 429.

No threads, no HTTP and no weights: note_request takes the state and the
current time as arguments, so each test brings its own deque and its own
made-up clock. That is the reason the state is passed in rather than kept in
the module.
"""

from collections import deque

import pytest

from src.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
from src.ratelimit import TooManyRequests, note_request


def test_requests_up_to_the_limit_are_allowed():
    recent = deque()

    for i in range(RATE_LIMIT_REQUESTS):
        note_request(recent, "1.2.3.4", now=i * 0.1)

    assert len(recent) == RATE_LIMIT_REQUESTS


def test_one_request_over_the_limit_is_refused():
    recent = deque()
    for i in range(RATE_LIMIT_REQUESTS):
        note_request(recent, "1.2.3.4", now=i * 0.1)

    with pytest.raises(TooManyRequests) as refusal:
        note_request(recent, "1.2.3.4", now=10.0)

    # The message says what the limit is, or the caller cannot tell what to
    # change about how often they ask.
    assert str(RATE_LIMIT_REQUESTS) in str(refusal.value)


def test_the_budget_comes_back_after_the_window():
    recent = deque()
    for i in range(RATE_LIMIT_REQUESTS):
        note_request(recent, "1.2.3.4", now=i * 0.1)

    # Far enough past the window that every one of those has aged out.
    note_request(recent, "1.2.3.4", now=RATE_LIMIT_WINDOW_SECONDS + 100)

    assert len(recent) == 1


def test_two_callers_have_separate_budgets():
    """One client spending its share must not refuse anybody else."""
    recent = deque()
    for i in range(RATE_LIMIT_REQUESTS):
        note_request(recent, "1.2.3.4", now=i * 0.1)

    note_request(recent, "5.6.7.8", now=10.0)

    with pytest.raises(TooManyRequests):
        note_request(recent, "1.2.3.4", now=10.0)


def test_old_entries_are_dropped_so_the_deque_stays_bounded():
    """One deque for every caller only works if it is pruned as it goes.

    Without this it would grow for as long as the process lives, which is the
    reason not to keep a deque per address instead.
    """
    recent = deque()

    for minute in range(200):
        note_request(
            recent,
            f"10.0.0.{minute}",
            now=minute * (RATE_LIMIT_WINDOW_SECONDS + 1),
        )

    # Two hundred callers went through and the deque holds the newest alone.
    assert len(recent) == 1


def test_an_entry_exactly_one_window_old_still_counts():
    """The pruning asks for strictly older than the window, not as old.

    Which direction that rounds matters for a limit: counting the request on
    the boundary is the careful way round, since it refuses one early rather
    than allowing one too many.
    """
    recent = deque()
    note_request(recent, "1.2.3.4", now=0.0)

    note_request(recent, "1.2.3.4", now=RATE_LIMIT_WINDOW_SECONDS)

    assert len(recent) == 2
