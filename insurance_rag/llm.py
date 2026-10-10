"""Chat client for any OpenAI-compatible server; used with a vLLM server (`vllm serve <model>`)."""
from __future__ import annotations

import httpx


class LLMError(RuntimeError):
    pass


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "microsoft/Phi-3.5-mini-instruct",
                 api_key: str | None = None, timeout: float = 120.0, max_tokens: int = 300,
                 transport: httpx.BaseTransport | None = None):
        self.model, self.max_tokens = model, max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers, transport=transport)

    def chat(self, messages: list[dict]) -> str:
        try:
            r = self.client.post("/v1/chat/completions", json={
                "model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": self.max_tokens})
        except httpx.HTTPError as e:
            raise LLMError(f"cannot reach the LLM server: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"LLM server returned {r.status_code}: {r.text[:200]}")
        try:
            text = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected LLM response: {r.text[:200]}") from e
        if not text or not text.strip():
            raise LLMError("LLM returned an empty answer")
        return text.strip()
