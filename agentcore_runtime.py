"""Entrypoint for the Shield agents on Amazon Bedrock AgentCore Runtime.

AgentCore calls POST /invocations with a JSON payload and GET /ping for health;
BedrockAgentCoreApp provides both on port 8080. Every request names an action:

  analyze_document   {filename, content_b64, schema?}  -> {document_type, raw_text, extracted}
  coverage_decision  {policy_type, incident_type, claim} -> {covered, reasoning}
  chat               {message, history, policy_type, detected_incident} -> chatbot reply
  narrate            {prompt, max_tokens?}              -> {text}

The runtime is stateless: it holds no database. The caller (api.py on the
user's machine) keeps policies, claims and uploads, and sends only what each
agent needs. Model calls use the runtime's IAM execution role.
"""
from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path

# Must be set before the shared modules import config.
os.environ.setdefault("BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0")
os.environ.setdefault("LLM_PROVIDER", "bedrock")
os.environ.setdefault("DATA_DIR", os.path.join(tempfile.gettempdir(), "shield"))
# Never call ourselves. Set (not removed) so a local .env cannot fill it back in.
os.environ["AGENTCORE_RUNTIME_ARN"] = ""

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402

import llm  # noqa: E402
import policy_rules  # noqa: E402
from tools import chatbot  # noqa: E402
from tools.document_analyzer import IMAGE_EXTS, PDF_EXTS, analyze_document  # noqa: E402
from tools.policy_agent import _llm_decide  # noqa: E402

app = BedrockAgentCoreApp()

MAX_DOC_BYTES = 10 * 1024 * 1024
ALLOWED_EXTS = IMAGE_EXTS | PDF_EXTS


def _analyze(payload: dict) -> dict:
    name = Path(str(payload.get("filename") or "document")).name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return {"error": f"Unsupported file type '{ext}'."}
    try:
        data = base64.b64decode(payload.get("content_b64") or "", validate=True)
    except Exception:
        return {"error": "content_b64 is not valid base64."}
    if not data or len(data) > MAX_DOC_BYTES:
        return {"error": "Document is empty or larger than 10 MB."}
    schema = payload.get("schema")
    if schema is not None and not (
        isinstance(schema, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in schema.items())
    ):
        return {"error": "schema must map field names to descriptions."}

    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(name).stem)[:40] or "doc"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{stem}{ext}"
        path.write_bytes(data)
        result = analyze_document(str(path), schema=schema)
    result["raw_text"] = (result.get("raw_text") or "")[:20000]
    return result


def _coverage(payload: dict) -> dict:
    policy_type = str(payload.get("policy_type") or "").strip().lower()
    policy_doc = policy_rules.load_policy_document(policy_type) if policy_type else None
    if not policy_doc:
        return {"error": f"No policy document for type '{policy_type}'."}
    claim = payload.get("claim") if isinstance(payload.get("claim"), dict) else {}
    incident = str(payload.get("incident_type") or "").strip().lower()
    return _llm_decide(policy_doc, policy_type, incident, claim)


def _chat(payload: dict) -> dict:
    policy_type = str(payload.get("policy_type") or "").strip().lower()
    if not policy_rules.load_policy_document(policy_type):
        return {"error": f"No policy document for type '{policy_type}'."}
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    history = [h for h in history if isinstance(h, dict)][-8:]
    return chatbot._llm_reply(
        str(payload.get("message") or ""),
        history,
        {"policy_type": policy_type},
        payload.get("detected_incident"),
    )


def _narrate(payload: dict) -> dict:
    prompt = str(payload.get("prompt") or "")[:8000]
    max_tokens = min(int(payload.get("max_tokens") or 300), 1000)
    return {"text": llm.complete(prompt, max_tokens=max_tokens)}


ACTIONS = {
    "analyze_document": _analyze,
    "coverage_decision": _coverage,
    "chat": _chat,
    "narrate": _narrate,
}


@app.entrypoint
def invoke(payload, context=None):
    if not isinstance(payload, dict):
        return {"error": "Payload must be a JSON object."}
    action = payload.get("action")
    handler = ACTIONS.get(action)
    if not handler:
        return {"error": f"Unknown action '{action}'. Use one of {sorted(ACTIONS)}."}
    try:
        return handler(payload)
    except Exception as exc:  # report, don't crash the runtime
        print(f"[shield] {action} failed: {type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    app.run()
