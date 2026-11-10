"""Protocol implemented by all OfficeQA generation backends."""

from typing import Any, Protocol

from officeqa_rl.types import GeneratedTurn, Message


class GenerationBackend(Protocol):
    """Generate one model turn without parsing or executing its action."""

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        """Return generated text and any available token provenance."""
        ...


__all__ = ["GenerationBackend"]
