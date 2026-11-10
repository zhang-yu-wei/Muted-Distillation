"""Generation backends for OfficeQA agents."""

from officeqa_rl.backends.gemini import GeminiGenerationBackend
from officeqa_rl.backends.kimi import KimiGenerationBackend
from officeqa_rl.backends.openai_server import OpenAIServerGenerationBackend
from officeqa_rl.backends.protocol import GenerationBackend
from officeqa_rl.backends.vllm import VLLMGenerationBackend

__all__ = [
    "GeminiGenerationBackend",
    "GenerationBackend",
    "KimiGenerationBackend",
    "OpenAIServerGenerationBackend",
    "VLLMGenerationBackend",
]
