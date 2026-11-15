"""Provider-neutral chat client with an on-disk cache.

A model is named ``provider:model_id``:

  gemini:<id>      Google Generative Language REST API (GEMINI_API_KEY)
  anthropic:<id>   Anthropic SDK (credentials resolved by the SDK)
  openai:<id>      Any OpenAI-compatible chat-completions endpoint
                   (OPENAI_API_KEY, OPENAI_BASE_URL; e.g. a local vLLM server)
  openrouter:<id>  OpenRouter (OPENROUTER_API_KEY); reports the cost of each call
  mock:<name>      Scripted responses for tests (see ``MockBackend``)

Every call is cached on (model, system, messages, tag). ``tag`` lets a caller
ask for independent samples of the same prompt. Set ``MAX_COST_USD`` to stop a
run once the providers' reported spend passes a limit. Evaluated models are never
given a fallback model: a refusal is returned as a failed call so that results
always describe the model that was named.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class LLMError(RuntimeError):
    pass


class Refusal(LLMError):
    pass


class BudgetExceeded(RuntimeError):
    """Deliberately not an LLMError, so nothing downstream swallows it."""


_lock = threading.Lock()
USAGE: dict[str, dict[str, int]] = {}


MAX_COST_USD: float | None = None


def total_cost():
    with _lock:
        return sum(u["cost_usd"] for u in USAGE.values())


def _count(model, calls=0, cached=0, input_tokens=0, output_tokens=0, cost=0.0):
    with _lock:
        u = USAGE.setdefault(model, {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        u["calls"] += calls
        u["cached"] += cached
        u["input_tokens"] += input_tokens or 0
        u["output_tokens"] += output_tokens or 0
        u["cost_usd"] += cost or 0.0


def _post(url, body, headers, timeout=300, retries=5):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (408, 429) or e.code >= 500:
                time.sleep(min(60, 4 * 2**attempt))
                continue
            raise LLMError(f"HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise LLMError(f"network error: {e}") from None
            time.sleep(min(60, 4 * 2**attempt))
    raise LLMError("request failed after retries")


def _gemini(model, system, messages, max_tokens):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    body = {
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]} for m in messages],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    d = _post(url, body, {"x-goog-api-key": key})
    cands = d.get("candidates") or []
    if not cands:
        raise Refusal(f"no candidates: {json.dumps(d.get('promptFeedback'))}")
    parts = cands[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        raise Refusal(f"empty response, finishReason={cands[0].get('finishReason')}")
    u = d.get("usageMetadata", {})
    out = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
    return text, u.get("promptTokenCount", 0), out


def _chat_completions(base, key, body):
    d = _post(f"{base}/chat/completions", body, {"Authorization": f"Bearer {key}"} if key else {})
    if "choices" not in d:
        raise LLMError(f"unexpected response: {json.dumps(d)[:300]}")
    text = d["choices"][0]["message"].get("content") or ""
    if not text:
        raise Refusal(f"empty response, finish_reason={d['choices'][0].get('finish_reason')}")
    u = d.get("usage") or {}
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost") or 0.0


def _openai(model, system, messages, max_tokens):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    return _chat_completions(base, os.environ.get("OPENAI_API_KEY", ""),
                             {"model": model, "messages": msgs, "max_completion_tokens": max_tokens})


def _openrouter(model, system, messages, max_tokens):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    return _chat_completions("https://openrouter.ai/api/v1", key,
                             {"model": model, "messages": msgs, "max_tokens": max_tokens, "usage": {"include": True}})


_anthropic_client = None


def _anthropic(model, system, messages, max_tokens):
    global _anthropic_client
    try:
        import anthropic
    except ImportError:
        raise LLMError("pip install anthropic to use anthropic: models") from None
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(max_retries=5)
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    try:
        # Streaming keeps long generations under the HTTP timeout.
        with _anthropic_client.messages.stream(**kwargs) as stream:
            r = stream.get_final_message()
    except anthropic.APIStatusError as e:
        raise LLMError(f"anthropic {e.status_code}: {e.message}") from None
    except anthropic.APIConnectionError as e:
        raise LLMError(f"anthropic connection error: {e}") from None
    if r.stop_reason == "refusal":
        category = getattr(r.stop_details, "category", None) if r.stop_details else None
        raise Refusal(f"refusal (category={category})")
    text = "".join(b.text for b in r.content if b.type == "text")
    if not text:
        raise Refusal(f"empty response, stop_reason={r.stop_reason}")
    return text, r.usage.input_tokens, r.usage.output_tokens


class MockBackend:
    """Route prompts to scripted handlers: ``handler(system, messages) -> str``."""

    handlers: dict[str, object] = {}

    @classmethod
    def call(cls, model, system, messages, max_tokens):
        if model not in cls.handlers:
            raise LLMError(f"no mock handler registered for {model!r}")
        return cls.handlers[model](system, messages), 0, 0


_BACKENDS = {"gemini": _gemini, "openai": _openai, "openrouter": _openrouter, "anthropic": _anthropic, "mock": MockBackend.call}

CACHE_DIR: Path | None = None  # set by the pipeline; None disables caching


def chat(model, messages, *, system="", max_tokens=16000, tag="", cache=True):
    """Return the model's text reply. ``messages`` is a list of role/content dicts
    or a single user string."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    provider, _, model_id = model.partition(":")
    if provider not in _BACKENDS or not model_id:
        raise LLMError(f"model must be 'provider:model_id', got {model!r}")
    path = None
    if cache and CACHE_DIR is not None and provider != "mock":
        key = hashlib.sha256(json.dumps([model, system, messages, tag], sort_keys=True).encode()).hexdigest()
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if path.exists():
            _count(model, cached=1)
            return json.loads(path.read_text())["text"]
    if MAX_COST_USD is not None and total_cost() >= MAX_COST_USD:
        raise BudgetExceeded(f"spend of ${total_cost():.2f} has reached the limit of ${MAX_COST_USD:.2f}")
    text, tin, tout, *cost = _BACKENDS[provider](model_id, system, messages, max_tokens)
    _count(model, calls=1, input_tokens=tin, output_tokens=tout, cost=cost[0] if cost else 0.0)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"model": model, "tag": tag, "text": text}))
        tmp.replace(path)
    return text


def parse_json(text):
    """Extract the first JSON object or array from a model reply."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return obj


def chat_json(model, messages, *, system="", max_tokens=16000, tag="", validate=None, attempts=3):
    """Call ``chat`` and parse JSON. ``validate(obj)`` returns a list of problems;
    on a parse failure or any problem the model is shown the error and asked to
    correct its reply."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    messages = list(messages)
    last = "no attempt made"
    for i in range(attempts):
        reply = chat(model, messages, system=system, max_tokens=max_tokens, tag=f"{tag}#{i}" if i else tag)
        try:
            obj = parse_json(reply)
            problems = validate(obj) if validate else []
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            problems = [f"reply could not be read as the requested JSON: {e}"]
            obj = None
        if not problems:
            return obj
        last = "; ".join(problems[:8])
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Your reply had these problems:\n- " + "\n- ".join(problems[:12])
                      + "\nReturn the complete corrected JSON only."}]
    raise LLMError(f"no valid JSON after {attempts} attempts: {last}")
