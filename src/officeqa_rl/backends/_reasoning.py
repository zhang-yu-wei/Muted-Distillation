"""Split generated text into reasoning blocks and the visible remainder.

Gemma parser ported from the history-render fix a750a5c.
"""

from __future__ import annotations

import re

GEMMA_OPEN = "<|channel>thought"
GEMMA_CLOSE = "<channel|>"
_GEMMA_BOUNDARY = re.compile(f"{re.escape(GEMMA_OPEN)}|{re.escape(GEMMA_CLOSE)}")


def strip_gemma_reasoning(content: str) -> tuple[str | None, str]:
    """Remove every thought block, including malformed/repeated boundaries.

    Preserve all visible text outside the blocks. Reasoning is saved separately
    rather than replayed as visible assistant prose in the next prompt.

    An unmatched closing marker makes the preceding segment ambiguous: it may
    contain continued deliberation (including candidate commands) whose opener
    was omitted. Keep that segment as reasoning, not executable visible text.
    Do not simply delete the closing marker and expose the candidate commands.
    """
    reasoning: list[str] = []
    visible: list[str] = []
    pos = 0
    in_thought = False
    for boundary in _GEMMA_BOUNDARY.finditer(content):
        segment = content[pos:boundary.start()]
        closes = boundary.group() == GEMMA_CLOSE
        if in_thought or closes:
            reasoning.append(segment.strip())
        else:
            visible.append(segment)
        in_thought = not closes
        pos = boundary.end()
    if in_thought:
        reasoning.append(content[pos:].strip())
    else:
        visible.append(content[pos:])
    joined = "\n".join(part for part in reasoning if part)
    return (joined if reasoning else None), "".join(visible).strip()
