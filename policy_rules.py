"""Policy knowledge base.

Single source of truth for:
  * which incident types belong to which policy type
  * which supporting document each incident requires (and whether it is mandatory)
  * loading the human-readable policy document text for the coverage agent

The structured map below is kept in sync with the markdown policy documents in
`policies/`. The agent reads the markdown for its natural-language reasoning;
the structured map drives the UI (what document to ask the user for) and the
deterministic mandatory-document rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BASE_DIR

POLICIES_DIR = BASE_DIR / "policies"

# incident_type -> {policy_type, covered, document, document_required}
# document_required=True means the claim cannot proceed without it.
INCIDENT_RULES: Dict[str, Dict[str, Any]] = {
    # --- auto ---
    "collision": {
        "policy_type": "auto",
        "covered": True,
        "document_label": "Photographs of the damaged vehicle (and repair estimate if available)",
        "document_required": True,
    },
    "theft": {
        "policy_type": "auto/home",
        "covered": True,
        "document_label": "Police complaint / FIR copy",
        "document_required": True,
    },
    "vandalism": {
        "policy_type": "auto",
        "covered": True,
        "document_label": "Photographs of the damage (police complaint if filed)",
        "document_required": True,
    },
    "fire": {
        "policy_type": "auto/home",
        "covered": True,
        "document_label": "Photographs of the damage and fire-brigade/police report",
        "document_required": True,
    },
    "flood": {
        "policy_type": "auto/home",
        "covered": False,
        "document_label": "",
        "document_required": False,
    },
    "wear_and_tear": {
        "policy_type": "auto/home",
        "covered": False,
        "document_label": "",
        "document_required": False,
    },
    # --- home ---
    "water_damage": {
        "policy_type": "home",
        "covered": True,
        "document_label": "Photographs of the damage and plumber/repair report",
        "document_required": True,
    },
    "storm": {
        "policy_type": "home",
        "covered": True,
        "document_label": "Photographs of the damage",
        "document_required": True,
    },
    # --- health ---
    "hospitalization": {
        "policy_type": "health",
        "covered": True,
        "document_label": "Hospital discharge summary and itemized hospital bill",
        "document_required": True,
    },
    "surgery": {
        "policy_type": "health",
        "covered": True,
        "document_label": "Surgeon's report and itemized hospital bill",
        "document_required": True,
    },
    "emergency": {
        "policy_type": "health",
        "covered": True,
        "document_label": "Emergency-room record and itemized bill",
        "document_required": True,
    },
    "cosmetic": {
        "policy_type": "health",
        "covered": False,
        "document_label": "",
        "document_required": False,
    },
    # --- life ---
    "death": {
        "policy_type": "life",
        "covered": True,
        "document_label": "Death certificate and beneficiary identity proof",
        "document_required": True,
    },
    "terminal_illness": {
        "policy_type": "life",
        "covered": True,
        "document_label": "Medical certificate confirming terminal diagnosis",
        "document_required": True,
    },
}


def incident_requirement(incident_type: str) -> Dict[str, Any]:
    """Return the document requirement for an incident type (safe defaults)."""
    key = (incident_type or "").strip().lower()
    return INCIDENT_RULES.get(
        key,
        {
            "policy_type": "unknown",
            "covered": None,
            "document_label": "Any supporting document for your claim",
            "document_required": False,
        },
    )


def incident_types_for_policy(policy_type: str) -> List[str]:
    """List incident types that apply to a given policy type."""
    pt = (policy_type or "").lower()
    out = []
    for incident, rule in INCIDENT_RULES.items():
        if pt in rule["policy_type"].split("/"):
            out.append(incident)
    return out


def load_policy_document(policy_type: str) -> Optional[str]:
    """Load the markdown policy document text for a policy type."""
    path = POLICIES_DIR / f"{(policy_type or '').lower()}_policy.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def all_incident_requirements() -> Dict[str, Dict[str, Any]]:
    """Expose the full map (used by the frontend to render dynamic prompts)."""
    return INCIDENT_RULES
