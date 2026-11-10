"""Shared generation-error classification helpers."""

from __future__ import annotations


def is_context_length_error(error: BaseException | str) -> bool:
    """Return whether a provider rejected a request for exceeding context.

    Providers use several phrasings.  In particular, SGLang reports
    ``Input length (...) exceeds the maximum allowed length (...)`` without
    using the words ``context length``.
    """
    detail = str(error).lower()
    if "context length" in detail or "context window" in detail:
        return True
    if "maximum sequence length" in detail or "max sequence length" in detail:
        return True
    return (
        ("input length" in detail or "prompt length" in detail)
        and ("maximum allowed length" in detail or "max allowed length" in detail)
        and ("exceed" in detail or "too long" in detail)
    )
