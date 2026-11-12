"""Single LLM layer used by every agent.

Supports AWS Bedrock (Converse API) and OpenAI, chosen by LLM_PROVIDER in .env.
All callers use `complete()`, so changing provider needs no code changes.

Callers must treat failures as non-fatal: wrap `complete()` in try/except and
fall back to rule-based behaviour. `is_available()` is a cheap pre-check.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import config

_JSON_RULE = (
    "Respond with a single valid JSON object only. No prose and no markdown fences."
)

# Cached state so a broken setup does not slow down every step of a claim.
_client_lock = threading.Lock()
_bedrock_clients: Dict[str, Any] = {}
_model_cache: Dict[str, str] = {}
_availability: Dict[str, Any] = {"ok": None, "at": 0.0, "error": ""}
_AVAILABILITY_TTL = 60  # seconds


# --- Availability ----------------------------------------------------------
def provider() -> str:
    return config.LLM_PROVIDER


def is_available() -> bool:
    """True if the configured provider looks usable (cached for a minute)."""
    if provider() == "openai":
        return bool(config.OPENAI_API_KEY)
    now = time.time()
    if _availability["ok"] is not None and now - _availability["at"] < _AVAILABILITY_TTL:
        return bool(_availability["ok"])
    try:
        _session().client("sts", region_name=config.AWS_REGION).get_caller_identity()
        _availability.update(ok=True, at=now, error="")
    except Exception as exc:
        _availability.update(ok=False, at=now, error=f"{type(exc).__name__}: {exc}")
    return bool(_availability["ok"])


def last_error() -> str:
    return _availability.get("error", "")


# --- Bedrock helpers -------------------------------------------------------
def _session():
    import boto3

    if config.AWS_PROFILE:
        return boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
    return boto3.Session(region_name=config.AWS_REGION)


def _client(kind: str):
    # Locked so documents analysed in parallel share one client instead of
    # racing to create several. boto3 clients are safe to share once created.
    with _client_lock:
        if kind not in _bedrock_clients:
            _bedrock_clients[kind] = _session().client(kind, region_name=config.AWS_REGION)
        return _bedrock_clients[kind]


def _discover_model() -> str:
    """Pick an available Claude model/inference profile (Haiku preferred: cheap)."""
    if "auto" in _model_cache:
        return _model_cache["auto"]
    ids: List[str] = []
    try:
        pager = _client("bedrock").get_paginator("list_inference_profiles")
        for page in pager.paginate(typeEquals="SYSTEM_DEFINED"):
            ids += [p["inferenceProfileId"] for p in page.get("inferenceProfileSummaries", [])]
    except Exception:
        pass
    if not ids:
        resp = _client("bedrock").list_foundation_models(byProvider="Anthropic")
        ids = [m["modelId"] for m in resp.get("modelSummaries", [])]

    claude = [i for i in ids if "anthropic.claude" in i]
    region_prefix = config.AWS_REGION.split("-")[0]  # us / eu / ap
    # Prefer region-matching profiles, then Haiku, then Sonnet; skip legacy ids.
    def rank(i: str):
        return (
            0 if i.startswith(region_prefix + ".") or i.startswith("global.") else 1,
            0 if "haiku" in i else 1 if "sonnet" in i else 2,
            0 if "4" in i or "5" in i else 1,
            i,
        )
    if not claude:
        raise RuntimeError(
            "No Claude models are available in this account/region. Enable model "
            "access in the Bedrock console or set BEDROCK_MODEL_ID."
        )
    _model_cache["auto"] = sorted(claude, key=rank)[0]
    return _model_cache["auto"]


def model_in_use(vision: bool = False) -> str:
    if provider() == "openai":
        return config.OPENAI_VISION_MODEL if vision else config.OPENAI_MODEL
    pinned = config.BEDROCK_VISION_MODEL_ID if vision else config.BEDROCK_MODEL_ID
    return pinned or config.BEDROCK_MODEL_ID or _discover_model()


def _parse_data_url(url: str):
    m = re.match(r"data:(image/[\w.+-]+);base64,(.*)", url, re.S)
    if not m:
        raise ValueError("Unsupported image data URL")
    mime, b64 = m.group(1), m.group(2)
    fmt = {"image/jpg": "jpeg"}.get(mime, mime.split("/")[1])
    if fmt == "tif":
        fmt = "tiff"
    return fmt, base64.b64decode(b64)


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply (handles ``` fences / prose)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        t = t[start : end + 1]
    json.loads(t)  # validate; raises if still not JSON
    return t


def _normalize_turns(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Bedrock needs strictly alternating roles starting with 'user'."""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = m.get("content", "") or ""
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": role, "content": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


# --- Public API ------------------------------------------------------------
def complete(
    prompt: Optional[str] = None,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    system: Optional[str] = None,
    images: Optional[List[str]] = None,
    json_mode: bool = False,
    max_tokens: int = 1024,
) -> str:
    """Run one model call and return the reply text.

    prompt     single user message (or use `messages` for a conversation)
    system     system instructions
    images     list of base64 data URLs attached to the last user message
    json_mode  force a JSON object reply; the returned text is valid JSON
    """
    turns = list(messages) if messages else [{"role": "user", "content": prompt or ""}]
    if provider() == "openai":
        return _openai_complete(turns, system, images, json_mode, max_tokens)
    try:
        return _bedrock_complete(turns, system, images, json_mode, max_tokens)
    except Exception as exc:
        # Mark Bedrock unusable for a minute so a blocked account does not add a
        # failing network call to every step of every claim.
        _availability.update(
            ok=False, at=time.time(), error=f"{type(exc).__name__}: {exc}"
        )
        raise


def _bedrock_complete(turns, system, images, json_mode, max_tokens) -> str:
    turns = _normalize_turns(turns)
    if not turns:
        raise ValueError("No user message to send")
    sys_text = (system or "").strip()
    if json_mode:
        sys_text = (sys_text + "\n\n" + _JSON_RULE).strip()

    converse_msgs: List[Dict[str, Any]] = []
    for i, t in enumerate(turns):
        blocks: List[Dict[str, Any]] = []
        if images and i == len(turns) - 1 and t["role"] == "user":
            for url in images:
                fmt, raw = _parse_data_url(url)
                blocks.append({"image": {"format": fmt, "source": {"bytes": raw}}})
        blocks.append({"text": t["content"] or " "})
        converse_msgs.append({"role": t["role"], "content": blocks})

    kwargs: Dict[str, Any] = {
        "modelId": model_in_use(vision=bool(images)),
        "messages": converse_msgs,
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if sys_text:
        kwargs["system"] = [{"text": sys_text}]

    resp = _client("bedrock-runtime").converse(**kwargs)
    parts = resp["output"]["message"]["content"]
    text = "".join(p.get("text", "") for p in parts).strip()
    return _extract_json(text) if json_mode else text


def _openai_complete(turns, system, images, json_mode, max_tokens) -> str:
    client = config.get_openai_client()
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for i, t in enumerate(turns):
        if images and i == len(turns) - 1 and t["role"] == "user":
            content: Any = [{"type": "text", "text": t["content"]}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": t["role"], "content": t["content"]})
    kwargs: Dict[str, Any] = {
        "model": model_in_use(vision=bool(images)),
        "messages": msgs,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()
