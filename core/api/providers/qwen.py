"""Qwen — the qwen API columns, served by Alibaba's DashScope platform.

Provider named after the model family (user decision 2026-08-21, aligned
with how the other vendors read); the PLATFORM behind it is DashScope,
which is why the key variables keep the platform's own name and recognition
goes by api_base: compatible-mode makes the requests look exactly like
OpenAI's (model ids are ``openai/qwen…``), so the endpoint domain is the
only thing that reveals the vendor — and this provider must be tried BEFORE
the openai/ prefix match.

International site ONLY (user decision 2026-08-23: mainland 百炼 permanently
dropped — its key chain, workspace domain and DASHSCOPE_CN_* variables are
gone; 2026-08-21..23 cn runs were deleted, any surviving cn-baked api_base
fails loud here on auth). The site-specific variable wins; the generic
``DASHSCOPE_API_KEY`` serves a single-site shell. The OPENAI_API_KEY
fallback keeps the historical "load the DashScope key into OPENAI_API_KEY
for the run shell" recipe working byte-identically when nothing else is set.
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
