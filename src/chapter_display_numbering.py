"""Shared chapter-number sequencing for spine-ordered user interfaces."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable


def filename_chapter_number(filename, *, is_special=False) -> int:
    """Return the rightmost filename number, or zero for special/unnumbered files."""
    if is_special:
        return 0
    stem = os.path.splitext(os.path.basename(str(filename or "")))[0]
    matches = re.findall(r"\d+", stem)
    if not matches:
        return 0
    try:
        return max(0, int(matches[-1]))
    except (TypeError, ValueError):
        return 0


def nonreset_chapter_display_numbers(raw_numbers: Iterable[object]) -> list[int]:
    """Return spine-ordered numbers that never reset after becoming positive.

    Leading zeroes remain zero. Once a positive chapter number has appeared,
    any later value below the current display number is treated as a filename
    sequence restart and advances to the next display number instead.
    """
    result: list[int] = []
    positive_started = False
    previous_display = 0

    for raw_number in raw_numbers:
        try:
            number = max(0, int(raw_number))
        except (TypeError, ValueError, OverflowError):
            number = 0

        if positive_started and number < previous_display:
            number = previous_display + 1

        result.append(number)
        previous_display = number
        if number > 0:
            positive_started = True

    return result

