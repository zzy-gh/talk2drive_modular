"""
benchmark/language/models.py
============================
Which LLM parses the utterance. Every entry returns a
``core.llm_inference.CommandParser``, so all models see the same prompt and
pass the same validation; only the request to the provider differs.
"""

from __future__ import annotations

from core.landmark_kb import LandmarkKnowledgeBase


def make_parser(provider: str, model: str, kb: LandmarkKnowledgeBase):
    if provider == "gemini":
        from core.llm_inference import GeminiCommandParser
        return GeminiCommandParser(kb, model=model)
    raise ValueError(f"unknown provider {provider!r}; available: gemini")
