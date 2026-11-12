"""Deterministic business-logic tools used by the claim-processing agents.

These are rule-based (no LLM) so decisions are auditable and reproducible.
The agents call them and then narrate / summarize the results.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import database as db
import underwriting_rules as uw


def _parse_date(value: Any):
    if not value:
        return None
    if isinstance(value, (date, datetime)):
        return value if isinstance(value, date) else value.date()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def validate_coverage(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Check policy exists, is active, in force on incident date, and that the
    incident type is covered. Returns a structured result."""
    issues: List[str] = []
    policy_number = claim.get("policy_number")
    policy = db.get_policy(policy_number) if policy_number else None

    if not policy:
        return {
            "passed": False,
            "issues": [f"Policy '{policy_number}' not found."],
            "policy": None,
        }

    if policy["status"] != "active":
        issues.append(f"Policy status is '{policy['status']}', not active.")

    incident_date = _parse_date(claim.get("incident_date"))
    eff = _parse_date(policy.get("effective_date"))
    exp = _parse_date(policy.get("expiry_date"))
    if incident_date and eff and exp:
        if not (eff <= incident_date <= exp):
            issues.append(
                f"Incident date {incident_date} is outside the policy period "
                f"({eff} to {exp})."
            )

    incident_type = (claim.get("incident_type") or "").strip().lower()

    # Health waiting period: only emergencies are payable in the first N days,
    # as stated in the quote, the policy page and the policy PDF.
    wait_days = uw.PRODUCTS.get(policy["policy_type"], {}).get("waiting_period_days")
    if wait_days and incident_type != "emergency" and incident_date and eff:
        cover_from = eff + timedelta(days=wait_days)
        if incident_date < cover_from:
            issues.append(
                f"Incident date {incident_date} is within the {wait_days}-day waiting "
                f"period; non-emergency claims are covered from {cover_from}."
            )
    covered = db.is_incident_covered(policy["policy_type"], incident_type)
    if covered is None:
        issues.append(
            f"Incident type '{incident_type}' is not recognized for a "
            f"{policy['policy_type']} policy."
        )
    elif covered is False:
        issues.append(
            f"Incident type '{incident_type}' is excluded under a "
            f"{policy['policy_type']} policy."
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "policy": policy,
        "coverage_limit": policy["coverage_limit"],
        "deductible": policy["deductible"],
    }


def check_fraud(claim: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based consistency / fraud scoring. Score 0.0 (clean) .. 1.0 (high risk)."""
    flags: List[str] = []
    score = 0.0

    claimed = claim.get("claimed_amount")
    extracted_amount = (extracted or {}).get("claimed_amount")

    # 1. Amount mismatch between what the user said and the document.
    try:
        if claimed is not None and extracted_amount is not None:
            if abs(float(claimed) - float(extracted_amount)) > 0.01 * max(
                float(claimed), 1
            ) + 1:
                flags.append(
                    f"Claimed amount ({claimed}) differs from document amount "
                    f"({extracted_amount})."
                )
                score += 0.3
    except (TypeError, ValueError):
        pass

    # 2. Incident type mismatch between user and document.
    user_incident = (claim.get("incident_type") or "").strip().lower()
    doc_incident = ((extracted or {}).get("incident_type") or "").strip().lower()
    if user_incident and doc_incident and user_incident != doc_incident:
        flags.append(
            f"Incident type from user ('{user_incident}') differs from document "
            f"('{doc_incident}')."
        )
        score += 0.25

    # 3. Future-dated incident.
    incident_date = _parse_date(claim.get("incident_date"))
    if incident_date and incident_date > date.today():
        flags.append("Incident date is in the future.")
        score += 0.4

    # 4. Implausibly large claim vs coverage limit.
    policy = db.get_policy(claim.get("policy_number"))
    if policy and claimed is not None:
        try:
            if float(claimed) > float(policy["coverage_limit"]) * 2:
                flags.append("Claimed amount is more than double the coverage limit.")
                score += 0.3
        except (TypeError, ValueError):
            pass

    # 5. Duplicate: same policy + incident date + incident type already filed.
    existing = db.list_claims()
    for c in existing:
        if (
            c.get("policy_number") == claim.get("policy_number")
            and c.get("incident_date") == claim.get("incident_date")
            and c.get("incident_type") == claim.get("incident_type")
            and c.get("claim_id") != claim.get("claim_id")
        ):
            flags.append(f"Possible duplicate of existing claim {c.get('claim_id')}.")
            score += 0.35
            break

    # 6. Missing document.
    if not claim.get("document_path"):
        flags.append("No supporting document was provided.")
        score += 0.15

    score = min(round(score, 2), 1.0)
    return {
        "fraud_score": score,
        "flags": flags,
        "risk": "high" if score >= 0.5 else "medium" if score >= 0.25 else "low",
    }


def assess_claim(
    claim: Dict[str, Any], validation: Dict[str, Any], fraud: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute the payable amount after applying deductible and coverage limit."""
    claimed = claim.get("claimed_amount") or 0.0
    try:
        claimed = float(claimed)
    except (TypeError, ValueError):
        claimed = 0.0

    deductible = float(validation.get("deductible") or 0.0)
    limit = float(validation.get("coverage_limit") or 0.0)

    after_deductible = max(claimed - deductible, 0.0)
    payable = min(after_deductible, limit) if limit else after_deductible

    breakdown = {
        "claimed_amount": round(claimed, 2),
        "deductible_applied": round(min(deductible, claimed), 2),
        "coverage_limit": round(limit, 2),
        "assessed_amount": round(payable, 2),
    }
    return breakdown
