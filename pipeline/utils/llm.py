"""
llm.py – Thin LLM client wrapper.

Supports:
  ``openai``    – OpenAI Chat Completions API  (``pip install openai``)
  ``anthropic`` – Anthropic Messages API        (``pip install anthropic``)
  ``stub``      – No network calls; returns deterministic fake responses.
                  Used in unit tests and offline development.

Usage::

    from pipeline.utils.llm import LLMClient
    client = LLMClient(provider="openai", model="gpt-4o",
                       api_key_env="OPENAI_API_KEY")
    text = client.complete(system="You are a researcher.", user="Give me an idea.")
    data = client.complete_json(system="...", user="Return JSON only.")
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict


class LLMClient:
    """
    Unified LLM completion interface.

    Parameters
    ----------
    provider:
        ``"openai"`` | ``"anthropic"`` | ``"stub"``
    model:
        Model identifier (e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet-20241022"``).
    api_key_env:
        Name of the environment variable that holds the API key.
        Ignored when *provider* is ``"stub"``.
    temperature:
        Sampling temperature (0–2 for OpenAI; 0–1 for Anthropic).
    max_tokens:
        Maximum output tokens per request.
    """

    def __init__(
        self,
        provider: str = "stub",
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""

        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "stub":
            pass  # no setup needed
        else:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                "Choose 'openai', 'anthropic', or 'stub'."
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key)

    def _init_anthropic(self) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        """
        Return the model's text response to (*system*, *user*) messages.

        Raises
        ------
        RuntimeError
            If the API call fails.
        """
        if self.provider == "stub":
            return self._stub_complete(system, user)
        if self.provider == "openai":
            return self._openai_complete(system, user)
        if self.provider == "anthropic":
            return self._anthropic_complete(system, user)
        raise RuntimeError(f"Unknown provider: {self.provider}")

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        Return the model's response parsed as a JSON object or array.

        The method strips Markdown code fences (triple-backtick json blocks).

        Raises
        ------
        json.JSONDecodeError
            If the response cannot be parsed as JSON.
        """
        raw = self.complete(system, user)
        extracted = _extract_json(raw)
        return json.loads(extracted)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _openai_complete(self, system: str, user: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

    def _anthropic_complete(self, system: str, user: str) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

    def _stub_complete(self, system: str, user: str) -> str:
        """
        Return a deterministic fake response.

        If *system* or *user* contains the word ``json``, return a minimal
        valid JSON object so that :meth:`complete_json` works in tests.
        """
        combined = (system + " " + user).lower()
        if "json" in combined:
            return json.dumps({
                "stub": True,
                "title": "Stub Idea",
                "slug": "stub-idea",
                "description": "A stub description.",
                "h0": "No effect.",
                "h1": "Significant effect.",
                "acceptance_criteria": {"p_value": 0.05},
                "experiment_code": "print('stub experiment')",
                "requirements": [],
                "validation_result": {"passed": True, "p_value": 0.01},
                "revision_target": None,
                "paper_md": "# Stub Paper\n",
                "paper_tex": "\\documentclass{article}\\begin{document}Stub.\\end{document}",
                "references_bib": "",
            })
        return f"Stub LLM response for: {user[:120]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """
    Extract a JSON value from *text*.

    Tries, in order:
    1. Strip a triple-backtick json or plain code block.
    2. Find the first ``{...}`` or ``[...]`` in the text.
    3. Return the text as-is (let the caller's ``json.loads`` raise).
    """
    # Code-fence extraction
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    # Bare JSON object / array
    bare = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if bare:
        return bare.group(1).strip()
    return text.strip()
