"""Shared fakes for exercising the SGLang training path without a GPU server."""

from aiohttp import web


class ByteTokenizer:
    """Deterministic fake actor tokenizer: one token per rendered character."""

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return [ord(ch) for ch in rendered + "<assistant>"]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]


def sglang_response(text: str, *, finish_reason: str = "stop") -> dict:
    """Build a /generate response whose token ids decode back to ``text``."""
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": [
                [-0.5 * (index + 1), ord(ch)] for index, ch in enumerate(text)
            ],
            "finish_reason": {"type": finish_reason},
        },
    }


class FakeSGLangServer:
    """Local /generate + /abort_request endpoint recording every payload."""

    def __init__(self, responder):
        self.responder = responder
        self.generate_payloads = []
        self.generate_headers = []
        self.abort_payloads = []

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/generate", self._generate)
        app.router.add_post("/abort_request", self._abort)
        self._runner = web.AppRunner(app, shutdown_timeout=0.5)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.host = "127.0.0.1"
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self._runner.cleanup()
        return False

    async def _generate(self, request):
        payload = await request.json()
        self.generate_payloads.append(payload)
        self.generate_headers.append(dict(request.headers))
        result = await self.responder(payload)
        if isinstance(result, web.Response):
            return result
        return web.json_response(result)

    async def _abort(self, request):
        self.abort_payloads.append(await request.json())
        return web.json_response({})
