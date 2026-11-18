"""Client for the Shield agents hosted on Amazon Bedrock AgentCore Runtime.

When AGENTCORE_RUNTIME_ARN is set, the AI steps (document reading, policy
coverage reasoning, chatbot replies and review notes) run on AgentCore instead
of in this process. The website, API, rules and SQLite database stay local.

Leave AGENTCORE_RUNTIME_ARN empty to run everything locally as before. The
AgentCore runtime itself never sets it, so the hosted code runs the local path.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

RUNTIME_ARN = os.getenv("AGENTCORE_RUNTIME_ARN", "").strip()
QUALIFIER = os.getenv("AGENTCORE_QUALIFIER", "DEFAULT").strip() or "DEFAULT"

_client_lock = threading.Lock()
_client_cache: Dict[str, Any] = {}


def enabled() -> bool:
    return bool(RUNTIME_ARN)


def _client():
    with _client_lock:
        if "c" not in _client_cache:
            import boto3
            from botocore.config import Config

            # Reading a scanned document with the vision model can take a while.
            cfg = Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 2})
            session = (
                boto3.Session(profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION)
                if config.AWS_PROFILE
                else boto3.Session(region_name=config.AWS_REGION)
            )
            _client_cache["c"] = session.client("bedrock-agentcore", config=cfg)
        return _client_cache["c"]


def invoke(action: str, payload: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """Call one action on the hosted runtime and return its JSON result.

    Raises RuntimeError if the runtime reports an error.
    """
    body = json.dumps({"action": action, **payload}).encode("utf-8")
    resp = _client().invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        qualifier=QUALIFIER,
        # AgentCore needs a session id of at least 33 characters. Each call is
        # independent, so a fresh one per call keeps sessions isolated.
        runtimeSessionId=session_id or f"shield-{action}-{uuid.uuid4().hex}",
        contentType="application/json",
        accept="application/json",
        payload=body,
    )
    raw = resp["response"].read()
    data = json.loads(raw.decode("utf-8") or "{}")
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"AgentCore {action} failed: {data['error']}")
    return data


# --- Typed helpers used by the agents -------------------------------------
def analyze_document(path: str, schema: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    return invoke(
        "analyze_document",
        {
            "filename": Path(path).name,
            "content_b64": base64.b64encode(data).decode("ascii"),
            "schema": schema,
        },
    )


def coverage_decision(policy_type: str, incident_type: str, claim: Dict[str, Any]) -> Dict[str, Any]:
    return invoke(
        "coverage_decision",
        {
            "policy_type": policy_type,
            "incident_type": incident_type,
            "claim": {
                "incident_date": claim.get("incident_date"),
                "description": claim.get("description"),
                "claimed_amount": claim.get("claimed_amount"),
            },
        },
    )


def chat_reply(
    message: str,
    history: List[Dict[str, str]],
    policy_type: str,
    detected_incident: Optional[str],
) -> Dict[str, Any]:
    return invoke(
        "chat",
        {
            "message": message,
            "history": history[-8:],
            "policy_type": policy_type,
            "detected_incident": detected_incident,
        },
    )


def narrate(prompt: str, max_tokens: int = 300) -> str:
    return str(invoke("narrate", {"prompt": prompt, "max_tokens": max_tokens}).get("text", ""))
