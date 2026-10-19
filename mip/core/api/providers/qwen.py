"""Qwen — the qwen API columns, served by Alibaba's DashScope platform.

The provider is named after the model family; the platform behind it is
DashScope, so the key variables keep the platform's name and recognition goes
by api_base: compatible-mode requests look exactly like OpenAI's (model ids are
``openai/qwen…``), so the endpoint domain is the only thing that reveals the
vendor, and this provider must be tried before the openai/ prefix match.

Only the international site is supported. The site-specific variable
(``DASHSCOPE_INTL_API_KEY``) wins; the generic ``DASHSCOPE_API_KEY`` serves a
single-site shell; ``OPENAI_API_KEY`` is the last fallback, for a shell that
loads the DashScope key under that name.
"""

from __future__ import annotations

from .base import BaseProvider

INTL_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenProvider(BaseProvider):
    name = "qwen"
    key_env = "DASHSCOPE_API_KEY"
    key_fallbacks = ("OPENAI_API_KEY",)

    def key_chain(self, extra: dict | None = None) -> tuple[str, ...]:
        return ("DASHSCOPE_INTL_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")

    def matches(self, model_id: str, extra: dict | None = None) -> bool:
        api_base = str((extra or {}).get("api_base", "")).lower()
        return "dashscope" in api_base or "maas.aliyuncs.com" in api_base
