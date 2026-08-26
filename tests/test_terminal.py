"""Colour in the terminal, and the three promises that make it safe.

A decorated CLI that breaks a pipe is worse than a plain one, and this
project has already broken one pipe once - `spiyweb lint --json | jq` got a
progress line on stdout before it got JSON. Escape codes would break it
again, more subtly, so the guarantees are tested rather than trusted:

1. not a terminal, no colour;
2. `NO_COLOR` wins over everything;
3. the drawing itself is correct - a bar's length is the value's share, and
   a value that is not zero never draws as nothing.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

from spiyweb.terminal import (
    MIN_WIDTH,
    RESET,
    bar,
    columns,
    hop_color,
    paint,
    paint_hop,
    rule,
    supports_color,
    terminal_width,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.session import SpiywebIndex


class FakeTTY(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


# --- when colour is allowed at all -----------------------------------------


def test_a_pipe_gets_no_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing rule: `| jq` and `> file` must receive clean text."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert supports_color(FakeTTY(tty=False)) is False


def test_no_color_wins_over_a_real_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://no-color.org - any value, including an empty one."""
    monkeypatch.setenv("NO_COLOR", "")
    assert supports_color(FakeTTY(tty=True)) is False


def test_a_dumb_terminal_gets_no_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert supports_color(FakeTTY(tty=True)) is False


def test_a_stream_that_cannot_answer_gets_no_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed or exotic stream is a reason to be plain, not to raise."""
    monkeypatch.delenv("NO_COLOR", raising=False)

    class Hostile:
        def isatty(self) -> bool:
            raise ValueError("no")

    assert supports_color(Hostile()) is False  # type: ignore[arg-type]


# --- painting ---------------------------------------------------------------


def test_painting_is_a_no_op_when_disabled() -> None:
    assert paint("text", "good", enabled=False) == "text"


def test_painting_wraps_and_always_resets() -> None:
    painted = paint("text", "good", enabled=True)
    assert painted.startswith("\x1b[")
    assert painted.endswith(RESET)
    assert "text" in painted


def test_an_unknown_style_costs_a_colour_not_a_crash() -> None:
    """A typo in a style name must not abort somebody's report."""
    assert paint("text", "not-a-style") == "text"


def test_empty_text_is_never_wrapped() -> None:
    """Escape codes around nothing are invisible bytes in a pipe."""
    assert paint("", "good") == ""


def test_the_hop_gradient_saturates_rather_than_overflowing() -> None:
    assert hop_color(0) != hop_color(1)
    assert hop_color(4) == hop_color(99)
    assert hop_color(-1) == hop_color(0)


def test_hop_painting_respects_the_switch() -> None:
    assert paint_hop("x", 0, enabled=False) == "x"
    assert paint_hop("x", 0, enabled=True).endswith(RESET)


# --- the drawing ------------------------------------------------------------


def test_a_full_value_fills_the_bar() -> None:
    assert bar(10.0, 10.0, 8) == "█" * 8


def test_a_half_value_fills_half() -> None:
    assert bar(5.0, 10.0, 8) == "█" * 4


def test_a_tiny_value_still_draws_something() -> None:
    """Activated-at-all and not-activated are different facts."""
    drawn = bar(0.001, 10.0, 8)
    assert drawn, "a node that activated was drawn as nothing"


def test_zero_draws_nothing() -> None:
    assert bar(0.0, 10.0, 8) == ""
    assert bar(5.0, 0.0, 8) == ""
    assert bar(5.0, 10.0, 0) == ""


def test_a_value_over_the_maximum_does_not_overflow_the_width() -> None:
    assert len(bar(50.0, 10.0, 8)) == 8


def test_a_rule_fills_the_width_it_was_given() -> None:
    drawn = rule("title", 40, enabled=False)
    assert drawn.startswith("TITLE ")
    assert len(drawn) == 40


def test_columns_align_on_printed_width_not_byte_length() -> None:
    """Escape codes have no width on screen and plenty in `len()`."""
    lines = columns(
        [[paint("aa", "good"), "x"], ["bbbb", "y"]],
        gap=1,
    )
    assert lines[0].endswith(" x") or lines[0].endswith("  x")
    # The painted cell is padded to the same PRINTED width as "bbbb".
    assert lines[1].index("y") == len("bbbb") + 1


def test_the_width_never_collapses_below_the_readable_minimum() -> None:
    assert terminal_width(fallback=10) >= MIN_WIDTH


# --- the promise that matters, end to end -----------------------------------


def test_json_output_stays_parseable_when_the_cli_is_decorated(
    capsys: pytest.CaptureFixture[str], tiny_index_root: Path
) -> None:
    """The regression this whole module is most likely to cause."""
    from spiyweb.cli import main

    assert main(["lint", str(tiny_index_root), "--json"]) == 0
    printed = capsys.readouterr().out
    assert "\x1b[" not in printed, "escape codes leaked into --json output"
    json.loads(printed)


def test_a_decorated_answer_carries_no_escape_codes_into_a_pipe(
    capsys: pytest.CaptureFixture[str],
    open_tiny: Callable[..., SpiywebIndex],
    tiny_index_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiyweb.cli import main

    index = open_tiny()
    monkeypatch.setattr("spiyweb.cli._open", lambda path, **options: index)
    assert main(["query", str(tiny_index_root), "who raised the tower"]) == 0
    printed = capsys.readouterr().out
    assert "\x1b[" not in printed
    assert "█" in printed, "the bar is drawing, not colouring"


# --- the palette is a system, not a preference ------------------------------


def test_every_style_is_a_distinct_colour() -> None:
    """Two names that paint the same thing cannot be told apart on screen."""
    from spiyweb.terminal import _CODES

    colours = [code for name, code in _CODES.items() if code.startswith("38;5;")]
    assert len(colours) == len(set(colours)), "two styles share a colour"


def test_the_two_families_never_overlap() -> None:
    """Blue means the web working; red means a finding. A shared code would
    make a healthy report and a broken one look the same at a glance."""
    from spiyweb.terminal import _CODES

    blue = {_CODES[name] for name in ("heading", "accent", "good")}
    red = {_CODES[name] for name in ("warn", "bad")}
    assert not (blue & red)


def test_the_hop_ramp_is_monotonic_and_all_blue() -> None:
    """Five separable steps, and none of them borrows the finding colour."""
    from spiyweb.terminal import _CODES, _HOP_COLORS

    assert len(set(_HOP_COLORS)) == len(_HOP_COLORS)
    assert _CODES["bad"] not in _HOP_COLORS
    assert _CODES["warn"] not in _HOP_COLORS
    # 256-colour blues run 25..51 in this ramp; brightest first.
    numbers = [int(code.rsplit(";", 1)[1]) for code in _HOP_COLORS]
    assert numbers == sorted(numbers, reverse=True), numbers
