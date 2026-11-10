"""Parsing for the single fenced command used by the OfficeQA agent."""

import re

from officeqa_rl.types import FormatError


_FENCED_COMMAND_RE = re.compile(
    r"```[A-Za-z0-9_-]*\s*\n(.*?)\n```",
    re.DOTALL,
)


def parse_fenced_command(response_text: str) -> str:
    """Return the only non-empty fenced command or raise ``FormatError``."""
    matches = _FENCED_COMMAND_RE.findall(response_text)
    if len(matches) != 1:
        raise FormatError(
            f"Expected exactly one fenced command block, got {len(matches)}."
        )

    command = matches[0].strip()
    if not command:
        raise FormatError("Command block was empty.")
    return command


__all__ = ["parse_fenced_command"]
