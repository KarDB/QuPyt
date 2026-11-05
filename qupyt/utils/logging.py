"""Consistent completion and value entries for QuPyt's measurement logs.

Callers describe an action or report named values without choosing a padding
width. The CLI formatter adds dotted leaders using the configured line width and
the entry's actual prefix. Other handlers receive ordinary readable messages.
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass


def _render_field(
    label: str,
    value: object,
    *,
    width: int,
    first_indent: int,
    continuation_indent: int,
) -> list[str]:
    """Wrap a label before filling the final line with dots and its value."""
    text = str(value).expandtabs()
    reserve_value = "\n" not in text and width - len(text) - 3 > continuation_indent
    # Reserving the suffix keeps completion markers and short values intact.
    wrap_width = width - len(text) - 3 if reserve_value else width
    content = label if reserve_value else f"{label}...{text}"
    lines: list[str] = []
    for paragraph in content.expandtabs().split("\n"):
        indent = first_indent if not lines else continuation_indent
        if indent >= wrap_width:
            # The header can consume all the room left for this field.
            lines.append("")
            indent = continuation_indent
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=wrap_width,
                initial_indent=" " * indent,
                subsequent_indent=" " * continuation_indent,
                expand_tabs=False,
                replace_whitespace=False,
                break_on_hyphens=False,
            )
            or [" " * indent]
        )
    if reserve_value:
        dots = "." * max(3, width - len(lines[-1]) - len(text))
        lines[-1] += dots + text
    return lines


@dataclass(frozen=True)
class LogDetails:
    """Named values rendered with dotted leaders by the CLI formatter."""

    summary: str | None
    fields: tuple[tuple[str, object], ...]

    def render(self, *, width: int, first_indent: int, continuation_indent: int) -> str:
        """Fit value rows to the available width without truncating their content."""
        lines = [] if self.summary is None else [self.summary]
        for label, value in self.fields:
            first_field = not lines
            field_lines = _render_field(
                label,
                value,
                width=width,
                first_indent=first_indent if first_field else continuation_indent,
                continuation_indent=continuation_indent,
            )
            if first_field:
                # The logging formatter supplies the first line's header itself.
                field_lines[0] = field_lines[0][first_indent:]
            lines.extend(field_lines)
        return "\n".join(lines)


def log_done(message: str, *, level: int = logging.INFO) -> None:
    """Report a completed action with a right-aligned [done] marker."""
    logging.log(
        level,
        "%s [done]",
        message,
        extra={"qupyt_details": LogDetails(None, ((message, "[done]"),))},
        stacklevel=2,
    )


def log_value(label: str, value: object, *, level: int = logging.INFO) -> None:
    """Report a named value with its label and dotted leaders."""
    logging.log(
        level,
        "%s=%s",
        label,
        value,
        extra={"qupyt_details": LogDetails(None, ((label, value),))},
        stacklevel=2,
    )


def log_values(
    message: str, values: Mapping[str, object], *, level: int = logging.INFO
) -> None:
    """Report related values on separate aligned rows below a summary."""
    arguments: list[object] = [message]
    for label, value in values.items():
        arguments.extend((label, value))
    logging.log(
        level,
        "%s" + "\n%s=%s" * len(values),
        *arguments,
        extra={"qupyt_details": LogDetails(message, tuple(values.items()))},
        stacklevel=2,
    )
