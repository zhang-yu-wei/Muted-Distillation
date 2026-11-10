"""Per-assistant-turn SFT message construction shared by the offline dataset
builder (``scripts/build_sft_dataset.py``) and the online teacher rollout
(``officeqa_rl.training.teacher_sft_rollout``).

Each assistant turn of an episode becomes one training sample: the earlier
messages are context, and only the target turn is supervised via slime's
per-message ``step_loss_mask`` (honored by the ``qwen3_5`` loss-mask type).
"""

from __future__ import annotations

from typing import Any


REASONING_STYLES = ("qwen", "gemma")


def reasoning_style_for_loss_mask(loss_mask_type: str) -> str:
    """Map a slime ``--loss-mask-type`` onto the reasoning rendering it expects."""
    return "gemma" if str(loss_mask_type).startswith("gemma") else "qwen"


def per_turn_training_messages(
    cleaned: list[dict[str, Any]],
    reasoning_style: str = "qwen",
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Build one slime SFT message list per assistant turn.

    ``cleaned`` is an episode's message list (roles system/user/assistant, no
    terminal ``exit`` message). Returns ``(target_message, messages)`` pairs in
    episode order, where ``messages`` supervises exactly the target turn with
    ``step_loss_mask=1``, while earlier assistant turns keep only their content
    with ``step_loss_mask=0``.

    ``reasoning_style`` says how the target turn's reasoning reaches the chat
    template: ``qwen`` inlines it as ``<think>reasoning</think>`` + content
    (Qwen3.5's template keeps inline tags verbatim); ``gemma`` passes it as the
    message's ``reasoning_content`` field, which gemma-4's template renders as
    a ``<|channel>thought ... <channel|>`` block for the last assistant turn --
    the same template STRIPS thought blocks written inline in ``content``, so
    the qwen style would leave literal ``<think>`` tags in a gemma target.
    """
    if reasoning_style not in REASONING_STYLES:
        raise ValueError(
            f"reasoning_style must be one of {REASONING_STYLES}, got {reasoning_style!r}"
        )
    output: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for target_index, target in enumerate(cleaned):
        if target["role"] != "assistant":
            continue
        messages: list[dict[str, Any]] = []
        for index, message in enumerate(cleaned[: target_index + 1]):
            item: dict[str, Any] = {
                "role": message["role"],
                "content": message["content"],
            }
            if message["role"] == "assistant":
                if index == target_index:
                    reasoning = message.get("reasoning_content") or ""
                    reasoning = reasoning.strip()
                    if reasoning and reasoning_style == "gemma":
                        item["reasoning_content"] = reasoning
                    elif reasoning:
                        item["content"] = f"<think>{reasoning}</think>\n{message['content']}"
                    item["step_loss_mask"] = 1
                else:
                    # Qwen3.5 does not replay earlier reasoning at RL time.
                    item["step_loss_mask"] = 0
            messages.append(item)
        output.append((target, messages))
    return output


__all__ = ["REASONING_STYLES", "per_turn_training_messages", "reasoning_style_for_loss_mask"]
