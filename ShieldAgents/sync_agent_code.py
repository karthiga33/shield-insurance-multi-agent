"""Copy the shared agent modules into the AgentCore runtime folder.

AgentCore packages only app/agent/ (CodeZip). The agents themselves live one
level up in claim-intake-agent/, so run this before `agentcore dev` or
`agentcore deploy`:

    python sync_agent_code.py

Only the files the hosted agents need are copied. The .env file, the SQLite
database and uploads are never copied, so no secrets or customer data ship.
"""
from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent                      # claim-intake-agent/
DEST = HERE / "app" / "agent"

FILES = [
    "agentcore_runtime.py",
    "agentcore_client.py",
    "config.py",
    "database.py",
    "llm.py",
    "policy_rules.py",
    "tools/__init__.py",
    "tools/chatbot.py",
    "tools/document_analyzer.py",
    "tools/policy_agent.py",
    "policies/auto_policy.md",
    "policies/health_policy.md",
    "policies/life_policy.md",
]


def main() -> None:
    for rel in FILES:
        src, dst = SRC / rel, DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {rel}")


if __name__ == "__main__":
    main()
