"""Central configuration and shared OpenAI client for the claim-intake agent.

All paths are resolved relative to this file so the app can be run from any
working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Where the database and uploaded files live. Locally this is the project
# folder; set DATA_DIR to keep them elsewhere (e.g. a persistent volume).
DATA_DIR = Path(os.getenv("DATA_DIR") or BASE_DIR).resolve()
DB_PATH = DATA_DIR / "claims.db"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
FRONTEND_DIR = BASE_DIR / "frontend"

for _dir in (UPLOAD_DIR, SAMPLE_DOCS_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # read-only code folder (e.g. on AgentCore Runtime)

# --- OpenAI ----------------------------------------------------------------
# Accept either name so an existing OPEN_AI_KEY (from the original notebook)
# keeps working.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

# --- LLM provider ----------------------------------------------------------
# "bedrock" (AWS) or "openai". All model calls go through llm.py.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").strip().lower()

# AWS Bedrock settings. Credentials come from the normal AWS chain (environment
# variables / .env, `aws configure`, an SSO profile, or an IAM role when hosted
# on AWS).
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_PROFILE = os.getenv("AWS_PROFILE")  # optional named profile / SSO profile
# Leave blank to auto-pick an available Claude model; set to pin a specific
# model or inference-profile ID (e.g. one copied from the Bedrock console).
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "").strip()
BEDROCK_VISION_MODEL_ID = os.getenv("BEDROCK_VISION_MODEL_ID", "").strip()


def _find_tesseract():
    """Use TESSERACT_CMD if set, else look in the usual Windows install spots."""
    explicit = os.getenv("TESSERACT_CMD")
    if explicit:
        return explicit
    candidates = [
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# Tesseract binary for the OCR fallback (None = rely on PATH).
TESSERACT_CMD = _find_tesseract()


def get_openai_client():
    """Return a configured OpenAI client.

    Imported lazily so the rest of the project (DB setup, tests) can run
    without the key being present.
    """
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def has_openai_key() -> bool:
    return bool(OPENAI_API_KEY)
