"""vLLM generation backend for OfficeQA inference."""

from __future__ import annotations

import asyncio
import importlib
import threading
from itertools import count
from typing import Any

from officeqa_rl.types import GeneratedTurn, Message


class _VLLMEpisodeBackend:
    """Bind all turns of one agent episode to one vLLM DP rank."""

    def __init__(self, backend: "VLLMGenerationBackend", data_parallel_rank: int):
        self._backend = backend
        self._data_parallel_rank = data_parallel_rank

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        return await self._backend._generate_on_rank(
            messages,
            sampling_params,
            data_parallel_rank=self._data_parallel_rank,
        )


class VLLMGenerationBackend:
    """Generate OfficeQA turns with an in-process async vLLM engine.

    The async engine (``AsyncLLMEngine``) continuously batches whatever
    requests are in flight, so concurrent trajectories share GPU batches
    instead of taking turns. Action parsing and sandbox execution
    intentionally remain in :class:`officeqa_rl.agent.OfficeQAAgent`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_tokens: int,
        serialize_calls: bool = False,
        enable_thinking: bool | None = None,
        **vllm_kwargs: Any,
    ) -> None:
        try:
            vllm = importlib.import_module("vllm")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "vLLM is required for VLLMGenerationBackend; install the "
                "'officeqa-rl[inference]' optional dependencies"
            ) from exc

        self.model_path = model_path
        self.max_tokens = max_tokens
        # This is a chat-template kwarg, not a vLLM engine kwarg. In
        # particular, Gemma-4 otherwise closes its thought channel in the
        # generation prompt because its template defaults this to false.
        self.enable_thinking = enable_thinking
        self.chat_template_kwargs = (
            {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        )
        self._vllm = vllm
        self._generate_lock = asyncio.Lock() if serialize_calls else None
        self._stats_lock = threading.Lock()
        self._n_calls = 0
        self._cost = 0.0
        self._request_ids = count()
        self._data_parallel_size = int(vllm_kwargs.get("data_parallel_size", 1))
        self._next_data_parallel_rank = count()
        self.llm = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(model=model_path, **vllm_kwargs)
        )
        self.tokenizer = self.llm.get_tokenizer()

    def for_episode(
        self, affinity_key: str
    ) -> VLLMGenerationBackend | _VLLMEpisodeBackend:
        """Return a view whose requests always use the same local DP rank.

        New episodes rotate across ranks for balanced placement. The returned
        wrapper retains its assigned rank for every turn of that episode.
        """
        if self._data_parallel_size == 1:
            return self
        del affinity_key
        rank = next(self._next_data_parallel_rank) % self._data_parallel_size
        return _VLLMEpisodeBackend(self, rank)

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        return await self._generate_on_rank(messages, sampling_params)

    async def _generate_on_rank(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> GeneratedTurn:
        """Render messages and await the async engine natively."""
        if self._generate_lock is None:
            return await self._generate_async(
                messages,
                sampling_params,
                data_parallel_rank=data_parallel_rank,
            )
        async with self._generate_lock:
            return await self._generate_async(
                messages,
                sampling_params,
                data_parallel_rank=data_parallel_rank,
            )

    async def _generate_async(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> GeneratedTurn:
        with self._stats_lock:
            self._n_calls += 1
            request_id = f"officeqa-{next(self._request_ids)}"

        chat_messages, images = self._to_chat_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        params = {
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            **sampling_params,
        }
        model_input: dict[str, Any] = {"prompt": prompt}
        if images:
            model_input["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
        request_output = None
        generate_kwargs = (
            {"data_parallel_rank": data_parallel_rank}
            if data_parallel_rank is not None
            else {}
        )
        async for output in self.llm.generate(
            model_input,
            self._vllm.SamplingParams(**params),
            request_id,
            **generate_kwargs,
        ):
            request_output = output
        if request_output is None or not request_output.outputs:
            raise RuntimeError("vLLM returned no completion for the OfficeQA request")

        completion = request_output.outputs[0]
        prompt_ids = getattr(request_output, "prompt_token_ids", None)
        output_ids = getattr(completion, "token_ids", None)
        # num_cached_tokens is the prefix-cache hit count for this call, which
        # is the local analogue of a provider's cached-input token class.
        cached_tokens = getattr(request_output, "num_cached_tokens", None)
        usage = {
            "prompt_tokens": len(prompt_ids) if prompt_ids is not None else 0,
            "completion_tokens": len(output_ids) if output_ids is not None else 0,
            "cached_tokens": int(cached_tokens) if cached_tokens is not None else 0,
        }
        return GeneratedTurn(
            text=completion.text,
            prompt_ids=list(prompt_ids) if prompt_ids is not None else None,
            output_ids=list(output_ids) if output_ids is not None else None,
            finish_reason=getattr(completion, "finish_reason", None),
            usage=usage,
        )

    @staticmethod
    def _to_chat_messages(messages: list[Message]) -> tuple[list[Message], list[Any]]:
        chat_messages: list[Message] = []
        images: list[Any] = []
        for message in messages:
            if message["role"] == "exit":
                continue
            content = message["content"]
            if isinstance(content, list):
                chat_content: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        raise TypeError(
                            "Multimodal message items must be dictionaries"
                        )
                    if item.get("type") == "text":
                        chat_content.append(
                            {"type": "text", "text": str(item.get("text", ""))}
                        )
                    elif item.get("type") == "image":
                        image = item.get("image")
                        if image is None:
                            raise TypeError("Multimodal image item has no image")
                        images.append(image)
                        # Qwen's chat template emits its vision placeholder for
                        # this item; the matching pixels travel separately in
                        # multi_modal_data below.
                        chat_content.append({"type": "image"})
                    else:
                        raise TypeError(
                            f"Unsupported multimodal item type: {item.get('type')!r}"
                        )
                content = chat_content
            elif not isinstance(content, str):
                raise TypeError(
                    "Message content must be a string or multimodal content list"
                )
            role = message["role"]
            if role not in {"system", "user", "assistant"}:
                role = "user"
            chat_messages.append({"role": role, "content": content})
        return chat_messages, images

    def get_stats(self) -> dict[str, int | float]:
        """Return thread-safe inference counters."""
        with self._stats_lock:
            return {"n_calls": self._n_calls, "cost": self._cost}


__all__ = ["VLLMGenerationBackend"]
