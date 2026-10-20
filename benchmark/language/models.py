"""
benchmark/language/models.py
============================
Which LLM parses the utterance. Every entry returns a
``core.llm_inference.CommandParser``, so all models see the same prompt and
pass the same validation; only the request to the provider differs.
"""

from __future__ import annotations

import json
import os
import urllib.request

from core.landmark_kb import LandmarkKnowledgeBase
from core.llm_inference import CommandParser


class QwenCommandParser(CommandParser):
    """Qwen via an OpenAI-compatible endpoint (QWEN_BASE_URL, QWEN_API_KEY);
    JSON output, temperature 0."""

    provider = "qwen"

    def __init__(self, kb: LandmarkKnowledgeBase, model: str):
        super().__init__(kb, model)
        self.url = os.environ["QWEN_BASE_URL"].rstrip("/") + "/chat/completions"
        self.key = os.environ["QWEN_API_KEY"]
        # Extra JSON body fields, e.g. '{"reasoning_effort": "none"}'.
        self.extra = json.loads(os.environ.get("QWEN_EXTRA_BODY") or "{}")

    def complete(self, system_prompt: str, text: str) -> str:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": text}],
                "response_format": {"type": "json_object"},
                "temperature": 0.0, **self.extra}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        # HTTP errors propagate; the caller retries.
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        usage = data.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("prompt_tokens"),
                           "output_tokens": usage.get("completion_tokens")}
        return data["choices"][0]["message"].get("content") or ""


def make_parser(provider: str, model: str, kb: LandmarkKnowledgeBase):
    if provider == "gemini":
        from core.llm_inference import GeminiCommandParser
        return GeminiCommandParser(kb, model=model)
    if provider == "qwen":
        return QwenCommandParser(kb, model=model)
    raise ValueError(f"unknown provider {provider!r}; available: gemini, qwen")
