"""Minimal pyplot shim for headless educational environments.

This module implements the handful of functions needed by the notebook to
produce a rudimentary bar chart without depending on the real matplotlib
package.  The ``show`` routine prints an ASCII representation of the plot so
students can still interpret the result in text-based workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class _Figure:
    figsize: Tuple[float, float] = (6.0, 4.0)
    bars: List[Tuple[float, float]] = field(default_factory=list)
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    color: str = "#000000"


_current_figure: _Figure | None = None


def figure(figsize: Tuple[float, float] | None = None) -> None:
    """Create a new figure and make it the current drawing target."""

    global _current_figure
    _current_figure = _Figure(figsize=figsize or (6.0, 4.0))


def bar(x: Sequence[float], height: Sequence[float], width: float = 0.8, color: str | None = None) -> None:
    """Add a bar series to the current figure."""

    if _current_figure is None:
        figure()
    if len(x) != len(height):
        raise ValueError("x and height must have the same length")
    _current_figure.color = color or "#5976ba"
    _current_figure.bars.extend(zip(x, height))


def xlabel(label: str) -> None:
    if _current_figure is None:
        figure()
    _current_figure.xlabel = label


def ylabel(label: str) -> None:
    if _current_figure is None:
        figure()
    _current_figure.ylabel = label


def title(text: str) -> None:
    if _current_figure is None:
        figure()
    _current_figure.title = text


def tight_layout() -> None:
    """No-op placeholder to mirror matplotlib's API."""

    return None


def show() -> None:
    """Render the stored bar chart as simple ASCII art."""

    if _current_figure is None or not _current_figure.bars:
        print("<empty plot>")
        return

    bars = _current_figure.bars
    max_height = max(value for _, value in bars)
    if max_height <= 0:
        scale = 0.0
    else:
        scale = 40.0 / max_height

    print(_current_figure.title or "Bar chart")
    print(f"x-axis: {_current_figure.xlabel}")
    print(f"y-axis: {_current_figure.ylabel}")
    print("".ljust(12) + "+" + "-" * 42)
    for x_value, height in bars:
        bar_length = int(round(height * scale))
        bar = "#" * bar_length
        print(f"{x_value:>10.1f} | {bar}")
    print("".ljust(12) + "+" + "-" * 42)
    print(f"Max height: {max_height:.3f}")


__all__ = [
    "figure",
    "bar",
    "xlabel",
    "ylabel",
    "title",
    "tight_layout",
    "show",
]
