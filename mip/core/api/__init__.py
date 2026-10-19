"""mip.core.api: the API layer, abstracted out of the harnesses.

Folder rule: caller shallow, callee deep.
The ROOT holds what requests actually hit and the machinery that makes those
endpoints exist — wire gateways (``anthropic_wire`` / ``openai_wire`` over
``service.LoopbackService``) and local model serving (``ollama`` / ``vllm``).
The bottom subfolder ``providers/`` holds pure knowledge of who is being
called (vendor identity, keys, wire, default endpoints). Imports point
strictly downward.

Entry points — the only import surface for outside code:

- ``provider_of(model_id, extra)`` — vendor resolution.
- ``route_of(model_id, extra)`` — how a seat's model is reached, resolved once:
  the provider, the endpoint, the key a caller passes, whether a vendor key is
  checked at all (``ModelRoute``).
- ``gateway_for(spec, wire, tag)`` — cross-API seats: a harness speaks its
  NATIVE wire to a foreign vendor through a wire gateway. Engaged only for
  cells whose extra carries ``("api_gateway", "litellm")``; native pairings
  keep their own call paths.
- ``fake_gateway_for(spec, tag)`` — the scripted endpoint of an ``api=fake``
  run (``fake_wire``): a real harness, no model.
- ``ensure_serving`` / ``serving_log_offset`` / ``serving_audit`` — local
  model serving, dispatched by provider identity. The dispatch lives HERE,
  not on the providers: the bottom layer stays declarative.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anthropic_wire import AnthropicWireGateway
from .fake_wire import FakeWireGateway
from .ollama import OllamaServer
from .openai_wire import CODEX_KEY_ENV, OpenAIWireGateway, Route
from .providers import PROVIDERS, provider_of
from .providers.base import BaseProvider, CallSpec
from .service import LoopbackService
from .vllm import VllmServer
from .vllm import probe as _vllm_probe

_OLLAMA = OllamaServer()

#: route prefixes whose endpoint a caller names itself (an OpenAI-compatible server) or serves
#: locally — no vendor key is checked for them
KEYLESS_PREFIXES = ("ollama", "hosted_vllm/", "openai/")


@dataclass(frozen=True)
class ModelRoute:
    """How one seat's model is reached, resolved once from the model id and the model row's
    extra (``ModelSpec.extra``): who serves it, where, with which key — the harness reads
    this instead of the row's keys."""

    model_id: str
    provider: BaseProvider | None
    #: the cell's own api_base; else a local vendor's default endpoint; else None (the vendor's)
    api_base: str | None
    #: the canonical key, passed explicitly beside the cell's own api_base (a key found in a
    #: fallback variable reaches the harness through the environment instead)
    api_key: str | None
    #: no vendor key is checked: the cell names its endpoint, or the id is a local /
    #: OpenAI-compatible route — litellm's price table is not expected to know it either
    keyless: bool

    @property
    def local(self) -> bool:
        """Served on our own machines (ollama, vLLM): the serving stack is ours to ensure."""
        return self.provider is not None and self.provider.local

    def require_key(self) -> None:
        """Refuse a keyed route whose vendor key is not set — before episode 0."""
        if self.keyless or self.provider is None or not self.provider.key_env:
            return
        if self.provider.resolve_key()[0] is None:
            raise RuntimeError(
                f"none of {' / '.join(self.provider.key_chain())} is set — "
                "litellm needs it (API billing)"
            )

    def litellm_kwargs(self) -> dict:
        """The endpoint kwargs a litellm caller passes: the api_base, and the key beside it."""
        out: dict = {}
        if self.api_base:
            out["api_base"] = self.api_base
        if self.api_key:
            out["api_key"] = self.api_key
        return out


def route_of(model_id: str, extra: dict | None = None) -> ModelRoute:
    """The route of a harness-facing model id under its row's extra (``api_base`` …)."""
    extra = dict(extra or {})
    provider = provider_of(model_id, extra)
    own_base = extra.get("api_base")
    api_key = None
    if own_base and provider is not None and provider.key_env:
        api_key = provider.prepare(model_id, extra).api_key
    return ModelRoute(
        model_id=model_id,
        provider=provider,
        api_base=own_base or (provider.default_api_base if provider is not None else None),
        api_key=api_key,
        keyless=bool(own_base) or model_id.startswith(KEYLESS_PREFIXES),
    )


def fake_gateway_for(spec, tag: str):
    """The scripted endpoint of an ``api=fake`` run (``spec.extra``
    carries ``api_gateway: fake`` + ``gateway_task``, optionally
    ``gateway_script``): started, or None for every other cell. It speaks
    both wires, so any harness can point at it — and it is the one gateway
    the mini harness takes (mini speaks litellm itself; the litellm gateway
    below has nothing to add for it)."""
    extra = dict(spec.extra)
    if extra.get("api_gateway") != "fake":
        return None
    gw = FakeWireGateway(str(extra.get("gateway_task") or ""), extra.get("gateway_script"), tag=tag)
    gw.start()
    return gw


def gateway_for(spec, wire: str, tag: str):
    """Start a wire gateway when the model asks for one (``spec`` is the
    run's mip.core.episode.ModelSpec): the scripted fake endpoint
    (``api_gateway: fake``, see fake_gateway_for), or the litellm gateway
    of a cross-API seat.

    A cross-API model carries ``api_gateway: litellm`` plus
    ``gateway_route: <litellm slug>``: the slug the mini harness uses for
    the same model key. The alias the harness asks for is the model's own id,
    so the harness-side model knob needs no special-casing.

    ``wire`` is the harness's NATIVE wire format: anthropic-messages gets
    our own shim endpoint (litellm proxy's /v1/messages mangles
    openai/responses/* routes — see anthropic_wire.py), openai-chat gets the
    litellm proxy. Vendor knowledge (which key variable) is resolved HERE
    and handed down — the gateways stay vendor-blind.
    """
    fake = fake_gateway_for(spec, tag)
    if fake is not None:
        return fake
    extra = dict(spec.extra)
    if extra.get("api_gateway") != "litellm":
        return None
    route = str(extra.get("gateway_route") or "")
    provider = provider_of(route)
    if not route or provider is None or not provider.key_env:
        raise RuntimeError(f"api_gateway cell without a resolvable gateway_route: {route!r}")
    if wire == "anthropic-messages":
        gw = AnthropicWireGateway({spec.model_id: route}, tag=tag)
    elif wire == "openai-chat":
        gw = OpenAIWireGateway([Route(spec.model_id, route, provider.key_env)], tag=tag)
    else:
        raise RuntimeError(f"no wire gateway for {wire!r}")
    gw.start()
    return gw


def ensure_serving(model_id: str, extra: dict | None = None) -> dict | None:
    """Bring a local model's serving stack up and verify it; returns the
    describe dict for harness_inherent["serving"], or None for remote APIs."""
    provider = provider_of(model_id, extra)
    if provider is None:
        return None
    if provider.name == "ollama":
        return _OLLAMA.ensure(model_id.split("/", 1)[1])
    if provider.name == "hosted_vllm":
        return _vllm_probe((extra or {}).get("api_base") or provider.default_api_base)
    return None


def serving_log_offset(model_id: str, extra: dict | None = None) -> int:
    """Marker for the serving audit's slice; 0 when the stack has no audit."""
    provider = provider_of(model_id, extra)
    if provider is not None and provider.name == "ollama":
        return _OLLAMA.log_offset()
    return 0


def serving_audit(model_id: str, offset: int, extra: dict | None = None) -> dict:
    """Post-run serving-side accounting (ollama: exact prompt-token counts
    from the serve log); {} when the stack has none."""
    provider = provider_of(model_id, extra)
    if provider is not None and provider.name == "ollama":
        return _OLLAMA.audit(offset)
    return {}
