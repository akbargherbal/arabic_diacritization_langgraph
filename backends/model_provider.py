"""
backends/model_provider.py
============================
Provider-agnostic model construction for the orchestrator and its
subagents.

Why this exists
----------------
main.py used to hardcode:

    MODEL = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", ...) or deepseek-v4-pro

That's fine for DeepSeek (an OpenAI-compatible endpoint) but breaks the
moment you point this at a provider with a different wire schema, e.g.
Anthropic. `get_model()` replaces that hardcoding with LangChain's
`init_chat_model`, which already knows how to build the right chat model
class for a given provider — you switch providers via `MODEL_PROVIDER`
(and optionally `MODEL_NAME`) instead of editing code.

NVIDIA is the one exception to "just call init_chat_model": see the NVIDIA
section below for why it's special-cased and built via `ChatNVIDIA`
directly (requires `pip install langchain-nvidia-ai-endpoints`).

NOTE: verify current provider support and package names against
docs.langchain.com before relying on this in production. init_chat_model's
provider-string conventions and the specific langchain-<provider> packages
required have moved before and will move again.

Retry behavior
---------------
Deep agent runs are long: a single batch can be dozens of sequential
model calls across the orchestrator and several subagents. A transient
network hiccup or a provider's rate limiter tripping on step 15 of 20
should not blow away the whole run. `_with_retry` / `_awith_retry`
implement bounded exponential backoff with jitter, retrying only on
errors that look transient (timeouts, connection resets, rate limits,
5xx/"overloaded" style server errors) and re-raising everything else
(auth failures, bad request shape, etc.) immediately.

Rather than requiring every callsite in this codebase (and inside
deepagents' own internals, which we don't control) to know about this
retry policy, `get_model()` monkeypatches the constructed model
*instance's* `invoke`/`ainvoke` bound methods in place via
`object.__setattr__`. That bypasses pydantic's `BaseModel.__setattr__`
(chat models are pydantic models, and `invoke`/`ainvoke` aren't declared
fields, so a plain `model.invoke = ...` would raise) while leaving the
class and every other instance untouched. This applies uniformly to every
provider, including NVIDIA, so there's a single retry policy for the whole
module rather than one behavior for init_chat_model-backed providers and
a different one (e.g. `Runnable.with_retry()`) for NVIDIA.

Timeout
-------
Every provider gets a 300s (5 min) request timeout by default. 60s (the
langchain-nvidia-ai-endpoints default, and often the underlying SDK
default for other providers too) is too short for reasoning models or long
tool-calling agent turns -- it's a client-side default, not a real ceiling
imposed by any of these APIs, so raising it here is safe. Override
globally with `MODEL_TIMEOUT_SECONDS`, or per-call with `get_model(timeout=...)`.
"""

from __future__ import annotations

import inspect
import logging
import os
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

import requests
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "deepseek"

DEFAULT_MODELS: dict[str, str] = {
    "deepseek": "deepseek-v4-flash",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1",
    # NVIDIA NIM / API Catalog. Default pinned to the model exercised in
    # the reference notebook.
    #
    # NOTE: client.get_available_models() showed "z-ai/glm-5.2" is NOT in
    # the catalog's listing (it still worked, with a "type is unknown"
    # warning -- likely a very-recently-added model not yet fully
    # indexed). "z-ai/glm-5.1" is the closest catalog entry confirmed
    # non-deprecated as of that same listing. Override via the
    # NVIDIA_MODEL env var (or model_name=) if "z-ai/glm-5.2" starts
    # failing outright rather than just warning.
    "nvidia": "z-ai/glm-5.2",
}

# Which env vars to read for api_key / base_url, per provider. Only set on
# the init_chat_model() call if the env var is actually present, so
# providers that don't need a base_url override (e.g. Anthropic normally)
# aren't forced to have one. NVIDIA doesn't go through init_chat_model at
# all (see _build_nvidia_model), but it still declares its required env
# var here so get_model()'s "what does this provider need" story stays in
# one place.
_PROVIDER_ENV: dict[str, dict[str, str]] = {
    "deepseek": {"api_key": "DEEPSEEK_API_KEY", "base_url": "DEEPSEEK_BASE_URL"},
    "anthropic": {"api_key": "ANTHROPIC_API_KEY", "base_url": "ANTHROPIC_BASE_URL"},
    "openai": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_BASE_URL"},
    "nvidia": {"api_key": "NVIDIA_API_KEY"},
}

# Provider-specific defaults to suppress reasoning/"thinking" tokens for
# calls that don't need them (e.g. short, deterministic JSON-output tasks
# like diacritization). Applied via setdefault in get_model(), so any
# caller that explicitly wants reasoning can still override these by
# passing its own extra_body / reasoning_effort kwarg.
#
# - deepseek: thinking mode defaults to ON for deepseek-v4-pro/flash; the
#   only real suppression switch is extra_body={"thinking": {"type": "disabled"}}
#   (confirmed against current DeepSeek API docs -- "thinking_budget" is not
#   a recognized field and is silently ignored if sent).
# - openai: reasoning-capable models (o-series, gpt-5-class) read
#   reasoning_effort; non-reasoning models (e.g. the gpt-4.1 default here)
#   ignore it harmlessly.
# - anthropic: extended thinking is opt-in and OFF by default -- nothing
#   to suppress; deliberately absent from this table.
# - nvidia: no generalizable suppression field exists across the different
#   models NVIDIA fronts (Qwen3 uses enable_thinking, some DeepSeek-backed
#   endpoints use thinking, Granite inverts the default) -- deliberately
#   left unset rather than guessing wrong for whichever model is live.
_REASONING_SUPPRESSION: dict[str, dict[str, Any]] = {
    "deepseek": {"extra_body": {"thinking": {"type": "disabled"}}},
    "openai": {"reasoning_effort": "minimal"},
}

# Every provider gets this request timeout by default unless overridden
# via MODEL_TIMEOUT_SECONDS or a per-call timeout= kwarg. See module
# docstring.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "300"))

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 30.0

# Matched against the *exception class name*, not an isinstance check,
# deliberately: which SDK (openai/anthropic/deepseek/requests/...) raises
# these lives in whichever provider package happens to be installed for
# the active MODEL_PROVIDER, and we don't want this module to import all
# of them just to catch their errors.
_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        # ChatNVIDIA is built on `requests`, which raises this for any
        # non-2xx response -- including transient gateway failures like a
        # 502/503/504 (seen in practice as NVIDIA's gateway timing out
        # mid-generation). Name-matching means a non-transient 401/403
        # HTTPError will also retry a couple of times before failing
        # identically; that's a wasted few seconds, not a correctness
        # issue, so it's left simple rather than parsing status codes out
        # of every provider's exception shape.
        "HTTPError",
    }
)

_RETRYABLE_MESSAGE_TOKENS = (
    "rate limit",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "gateway timeout",
    "503",
    "502",
    "504",
    "429",
)


def _is_retryable(exc: BaseException) -> bool:
    """Best-effort transient-error classification (see module docstring)."""
    if type(exc).__name__ in _RETRYABLE_ERROR_NAMES:
        return True
    msg = str(exc).lower()
    return any(token in msg for token in _RETRYABLE_MESSAGE_TOKENS)


def _with_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap a synchronous bound method with bounded exponential backoff."""

    def _wrapped(*args: Any, **kwargs: Any) -> T:
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - filtered by _is_retryable below
                attempt += 1
                if attempt > MAX_RETRIES or not _is_retryable(exc):
                    raise
                delay = min(
                    MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                )
                delay += random.uniform(0, delay * 0.25)
                logger.warning(
                    "Transient model error (attempt %d/%d): %s: %s -- retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)

    return _wrapped


def _awith_retry(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Wrap an async bound method with bounded exponential backoff."""

    async def _wrapped(*args: Any, **kwargs: Any) -> T:
        import asyncio

        attempt = 0
        while True:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - filtered by _is_retryable below
                attempt += 1
                if attempt > MAX_RETRIES or not _is_retryable(exc):
                    raise
                delay = min(
                    MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                )
                delay += random.uniform(0, delay * 0.25)
                logger.warning(
                    "Transient model error (attempt %d/%d): %s: %s -- retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    return _wrapped


def _patch_retry(model: BaseChatModel) -> BaseChatModel:
    """Monkeypatch `invoke`/`ainvoke` on this model *instance* to retry
    transient errors transparently. See module docstring for why
    `object.__setattr__` is used instead of a plain attribute assignment.
    """
    for method_name in ("invoke", "ainvoke"):
        original = getattr(model, method_name, None)
        if original is None:
            continue
        wrapper = (
            _awith_retry(original)
            if inspect.iscoroutinefunction(original)
            else _with_retry(original)
        )
        object.__setattr__(model, method_name, wrapper)
    return model


# ---------------------------------------------------------------------------
# NVIDIA NIM / API Catalog
# ---------------------------------------------------------------------------
def _patch_nvidia_error_masking(client_cls: type) -> None:
    """
    ChatNVIDIA's internal client `_try_raise(response)` unconditionally
    calls `response.json()` to build its error message. A gateway-level
    failure (e.g. a 504 with an empty body) has no JSON body to parse, so
    the *real* error ("504 Server Error: Gateway Timeout") gets masked by
    a confusing `JSONDecodeError: Expecting value` instead. Patched here to
    fall back to the real HTTPError when the body isn't JSON.

    Patched against `type(llm._client)` (the real runtime class) rather
    than a hardcoded `langchain_nvidia_ai_endpoints._common.SomeClassName`
    import -- only the method name (`_try_raise`) is documented/observed
    behavior here, not that module's exact source layout, so guessing an
    exact class name risks a silent no-op or an AttributeError if it's
    wrong. Re-verify this still applies if langchain-nvidia-ai-endpoints is
    upgraded.
    """
    if not hasattr(client_cls, "_try_raise") or getattr(
        client_cls, "_patched_for_readable_errors", False
    ):
        return

    _orig_try_raise = client_cls._try_raise

    def _patched_try_raise(self, response, *args, **kwargs):
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            try:
                response.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                raise http_err from None
        return _orig_try_raise(self, response, *args, **kwargs)

    client_cls._try_raise = _patched_try_raise
    client_cls._patched_for_readable_errors = True


def _build_nvidia_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    """
    Build a `ChatNVIDIA` instance for NVIDIA NIM / API Catalog models.

    `init_chat_model()` has no documented, guaranteed "nvidia:" provider
    string the way it does for openai/anthropic/deepseek, so this
    instantiates `langchain_nvidia_ai_endpoints`'s `ChatNVIDIA` directly
    (mirrors the reference notebook). Everything else about get_model()'s
    contract -- env var required, kwargs passed through, retry-wrapped
    instance returned -- stays identical to every other provider; the
    caller in get_model() doesn't need to know this branch works
    differently under the hood.

    Requires: pip install langchain-nvidia-ai-endpoints
    Env vars:
        NVIDIA_API_KEY -- required, your NVIDIA API Catalog key (starts "nvapi-")
        NVIDIA_MODEL   -- optional, overrides model_name / DEFAULT_MODELS["nvidia"]
    """
    # Local import so the rest of the app doesn't hard-require this
    # package just because it's installed for one provider.
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    api_key_env = _PROVIDER_ENV["nvidia"]["api_key"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Provider 'nvidia' requires {api_key_env} to be set.")

    model_id = os.environ.get("NVIDIA_MODEL", model_name)

    # Sensible defaults mirrored from the reference notebook; any of these
    # can still be overridden by callers via **kwargs.
    init_kwargs: dict[str, Any] = {
        "temperature": 1,
        "top_p": 1,
        "max_completion_tokens": 64000,
        "max_tokens": 512,
        "seed": 42,
    }
    init_kwargs.update(kwargs)
    timeout_seconds = init_kwargs.pop("timeout", DEFAULT_TIMEOUT_SECONDS)

    llm = ChatNVIDIA(model=model_id, api_key=api_key, **init_kwargs)

    # ChatNVIDIA's public constructor has no `timeout=` kwarg -- anything
    # passed in gets silently absorbed into model_kwargs (and sent as a
    # bogus request field) instead of configuring the HTTP client. The
    # underlying client defaults to a 60s read timeout, which a reasoning
    # model asked for max_completion_tokens=64000 can easily exceed
    # mid-generation. Set it directly on the (mutable) private client
    # attributes instead.
    llm._client.timeout = timeout_seconds
    llm._async_client.timeout = timeout_seconds

    _patch_nvidia_error_masking(type(llm._client))

    return llm


def get_model(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Build a retry-wrapped chat model for the configured provider.

    Provider/model selection precedence: explicit args > env vars
    (`MODEL_PROVIDER` / `MODEL_NAME`, or `NVIDIA_MODEL` specifically for
    the nvidia provider) > module defaults (DeepSeek, to preserve this
    project's prior default behavior).

    Extra `**kwargs` are forwarded to `init_chat_model` (and from there to
    the underlying provider class's `__init__`) for every provider except
    NVIDIA, which is built directly via `ChatNVIDIA` (see
    `_build_nvidia_model`) since `init_chat_model` doesn't know about it.
    Either way, provider-specific tuning (e.g. `temperature=`) works
    without this module knowing about it, and every returned model is
    retry-wrapped identically via `_patch_retry`.
    """
    resolved_provider = (
        provider or os.environ.get("MODEL_PROVIDER") or DEFAULT_PROVIDER
    ).lower()
    resolved_model_name = (
        model_name
        or os.environ.get("MODEL_NAME")
        or DEFAULT_MODELS.get(resolved_provider)
    )
    if resolved_model_name is None:
        msg = (
            f"No default model configured for provider '{resolved_provider}'; "
            f"pass model_name= explicitly or set the MODEL_NAME env var."
        )
        raise ValueError(msg)

    if resolved_provider == "nvidia":
        return _patch_retry(_build_nvidia_model(resolved_model_name, **kwargs))

    env_names = _PROVIDER_ENV.get(resolved_provider, {})
    init_kwargs: dict[str, Any] = dict(kwargs)

    api_key_env = env_names.get("api_key")
    if api_key_env and os.environ.get(api_key_env):
        init_kwargs.setdefault("api_key", os.environ[api_key_env])

    base_url_env = env_names.get("base_url")
    if base_url_env and os.environ.get(base_url_env):
        init_kwargs.setdefault("base_url", os.environ[base_url_env])
    elif resolved_provider == "deepseek":
        # Preserve this project's original hardcoded DeepSeek endpoint as
        # the default so behavior is unchanged when DEEPSEEK_BASE_URL
        # isn't set.
        init_kwargs.setdefault("base_url", "https://api.deepseek.com")

    # init_chat_model's `timeout` kwarg is passed straight through to the
    # underlying SDK client (OpenAI/Anthropic/DeepSeek all honor it
    # directly), so every non-NVIDIA provider gets the same 5-minute
    # default as NVIDIA instead of whatever short default its SDK ships
    # with. See module docstring.
    init_kwargs.setdefault("timeout", DEFAULT_TIMEOUT_SECONDS)

    # We own retry via _with_retry/_awith_retry below; keep the SDK's own
    # internal retry at 0 so a transient failure surfaces to our retry
    # loop immediately instead of two independent retry/backoff layers
    # silently compounding total wait time.
    init_kwargs.setdefault("max_retries", 0)

    # Suppress reasoning/thinking tokens by provider default (see
    # _REASONING_SUPPRESSION above). setdefault only -- a caller that
    # explicitly passes its own extra_body/reasoning_effort wins.
    for key, value in _REASONING_SUPPRESSION.get(resolved_provider, {}).items():
        init_kwargs.setdefault(key, value)

    model = init_chat_model(
        model=resolved_model_name, model_provider=resolved_provider, **init_kwargs
    )
    return _patch_retry(model)
