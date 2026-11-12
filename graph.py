"""LangGraph workflow for autonomous insurance claim intake.

Pipeline (each step is a node):

  create_claim  ->  analyze_document  ->  validate_coverage
                                                 |
                              (fail)  <----------+----------> (pass)
                                 |                               |
                          reject_auto                      check_fraud
                                                                 |
                              (high risk) <--------+-------------+------> (ok)
                                   |                                       |
                              reject_auto                             assess_claim
                                                                           |
                                                                   submit_for_approval
                                                                           |
                                                                    (human gate)

The graph marks a claim 'pending_approval' once it passes every automated
check. A human then approves/rejects it from the frontend. Validation or
fraud failures end as 'rejected_auto' and never reach a human.

The agents add a short natural-language narration (via the LLM when a key is
available) so the stored decision trail reads like an analyst's notes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

import config
import database as db
import llm
from tools import claim_tools
from tools.document_analyzer import analyze_document
from tools.policy_agent import decide_coverage


class ClaimState(TypedDict, total=False):
    # Input
    claim_input: Dict[str, Any]      # claimant_name, policy_number, incident_type, etc.
    document_path: Optional[str]
    # Working fields
    claim_id: str
    document_type: Optional[str]
    extracted: Dict[str, Any]
    raw_text: str
    validation: Dict[str, Any]
    coverage: Dict[str, Any]
    fraud: Dict[str, Any]
    assessment: Dict[str, Any]
    # Outcome
    status: str
    status_reason: str
    trail: List[str]                 # step-by-step decision log


def _log(state: ClaimState, message: str) -> List[str]:
    trail = list(state.get("trail", []))
    trail.append(message)
    print(message)
    return trail


def _narrate(step: str, facts: Dict[str, Any]) -> str:
    """Optional LLM narration of a step. Falls back to a plain summary."""
    if not llm.is_available():
        return f"{step}: {facts}"
    try:
        return llm.complete(
            f"You are a claims analyst. In one or two sentences, summarize "
            f"this '{step}' result for an internal review note:\n{facts}",
            max_tokens=300,
        )
    except Exception as exc:
        return f"{step}: {facts} (narration unavailable: {type(exc).__name__})"


# --- Nodes -----------------------------------------------------------------
def node_create_claim(state: ClaimState) -> ClaimState:
    data = dict(state.get("claim_input", {}))
    data["document_path"] = state.get("document_path")
    claim_id = db.create_claim(data)
    trail = _log(state, f"[1/6] Created claim {claim_id} in draft status.")
    return {"claim_id": claim_id, "trail": trail, "status": db.STATUS_DRAFT}


def node_analyze_document(state: ClaimState) -> ClaimState:
    claim_id = state["claim_id"]
    path = state.get("document_path")
    if not path:
        trail = _log(state, "[2/6] No document supplied; skipping analysis.")
        db.update_claim(claim_id, {"extracted_data": {}})
        return {"extracted": {}, "raw_text": "", "document_type": None, "trail": trail}

    result = analyze_document(path)
    extracted = result.get("extracted", {}) or {}
    db.update_claim(
        claim_id,
        {
            "document_type": result.get("document_type"),
            "extracted_data": extracted,
        },
    )
    trail = _log(
        state,
        f"[2/6] Analyzed {result.get('document_type')} document. "
        f"Extracted fields: {list(extracted.keys())}",
    )
    return {
        "extracted": extracted,
        "raw_text": result.get("raw_text", ""),
        "document_type": result.get("document_type"),
        "trail": trail,
    }


def node_validate_coverage(state: ClaimState) -> ClaimState:
    claim = db.get_claim(state["claim_id"])
    validation = claim_tools.validate_coverage(claim)
    db.update_claim(state["claim_id"], {"validation_result": validation})
    status = "passed" if validation["passed"] else "FAILED"
    trail = _log(
        state,
        f"[3/7] Policy validation {status}. Issues: {validation['issues'] or 'none'}",
    )
    return {"validation": validation, "trail": trail}


def node_policy_coverage(state: ClaimState) -> ClaimState:
    """Read the actual policy document and decide coverage, with reasoning.

    Also enforces the policy's required-document rule as a hard gate: if the
    policy requires a document for this incident type and none was supplied,
    the claim cannot pass.
    """
    claim = db.get_claim(state["claim_id"])
    coverage = decide_coverage(claim)

    # Required-document gate.
    if coverage.get("document_required") and not coverage.get("document_satisfied"):
        coverage["covered"] = False
        gate_note = (
            f" Required document missing: {coverage.get('required_document')}."
        )
        coverage["reasoning"] = (coverage.get("reasoning", "") + gate_note).strip()

    db.update_claim(state["claim_id"], {"coverage_result": coverage})
    verdict = (
        "COVERED" if coverage.get("covered") is True
        else "NOT COVERED" if coverage.get("covered") is False
        else "UNDETERMINED"
    )
    trail = _log(
        state,
        f"[4/7] Policy coverage decision: {verdict}. {coverage.get('reasoning', '')}",
    )
    return {"coverage": coverage, "trail": trail}


def node_check_fraud(state: ClaimState) -> ClaimState:
    claim = db.get_claim(state["claim_id"])
    fraud = claim_tools.check_fraud(claim, state.get("extracted", {}))
    db.update_claim(
        state["claim_id"],
        {"fraud_result": fraud, "fraud_score": fraud["fraud_score"]},
    )
    trail = _log(
        state,
        f"[5/7] Fraud check: risk={fraud['risk']} score={fraud['fraud_score']}. "
        f"Flags: {fraud['flags'] or 'none'}",
    )
    return {"fraud": fraud, "trail": trail}


def node_assess(state: ClaimState) -> ClaimState:
    claim = db.get_claim(state["claim_id"])
    assessment = claim_tools.assess_claim(
        claim, state.get("validation", {}), state.get("fraud", {})
    )
    db.update_claim(
        state["claim_id"],
        {
            "assessment_result": assessment,
            "assessed_amount": assessment["assessed_amount"],
        },
    )
    trail = _log(
        state,
        f"[6/7] Assessed payable amount: ${assessment['assessed_amount']:,.2f} "
        f"(claimed ${assessment['claimed_amount']:,.2f}, "
        f"deductible ${assessment['deductible_applied']:,.2f}).",
    )
    return {"assessment": assessment, "trail": trail}


def node_submit_for_approval(state: ClaimState) -> ClaimState:
    summary = _narrate(
        "final claim decision",
        {
            "policy_coverage": state.get("coverage", {}).get("reasoning"),
            "validation": state.get("validation", {}).get("issues"),
            "fraud": state.get("fraud"),
            "assessment": state.get("assessment"),
        },
    )
    trail = _log(
        state,
        "[7/7] All checks passed. Submitting for human approval.",
    )
    db.update_claim(
        state["claim_id"],
        {
            "status": db.STATUS_PENDING_APPROVAL,
            "status_reason": "Passed policy, validation and fraud checks; awaiting human sign-off.",
            "agent_summary": summary + "\n\nTrail:\n" + "\n".join(trail),
        },
    )
    return {
        "status": db.STATUS_PENDING_APPROVAL,
        "status_reason": "Awaiting human approval.",
        "trail": trail,
    }


def node_reject_auto(state: ClaimState) -> ClaimState:
    reasons: List[str] = []
    reasons += state.get("validation", {}).get("issues", [])
    coverage = state.get("coverage", {})
    if coverage.get("covered") is False and coverage.get("reasoning"):
        reasons.append(coverage["reasoning"])
    reasons += state.get("fraud", {}).get("flags", [])
    reason_text = "; ".join(reasons) or "Failed automated checks."
    summary = _narrate("auto rejection", {"reasons": reasons})
    trail = _log(state, f"[x] Auto-rejected: {reason_text}")
    db.update_claim(
        state["claim_id"],
        {
            "status": db.STATUS_REJECTED_AUTO,
            "status_reason": reason_text,
            "agent_summary": summary + "\n\nTrail:\n" + "\n".join(trail),
        },
    )
    return {
        "status": db.STATUS_REJECTED_AUTO,
        "status_reason": reason_text,
        "trail": trail,
    }


# --- Conditional edges -----------------------------------------------------
def after_validation(state: ClaimState) -> str:
    return "policy_coverage" if state.get("validation", {}).get("passed") else "reject_auto"


def after_coverage(state: ClaimState) -> str:
    # Covered claims proceed to fraud; not-covered (incl. missing required
    # document) are auto-rejected.
    return "check_fraud" if state.get("coverage", {}).get("covered") is True else "reject_auto"


def after_fraud(state: ClaimState) -> str:
    # High fraud risk never auto-proceeds; a human-review-only path could be
    # added, but per the requirement anything suspicious is held back.
    return "reject_auto" if state.get("fraud", {}).get("risk") == "high" else "assess"


# --- Build graph -----------------------------------------------------------
def build_graph():
    g = StateGraph(ClaimState)

    g.add_node("create_claim", node_create_claim)
    g.add_node("analyze_document", node_analyze_document)
    g.add_node("validate_coverage", node_validate_coverage)
    g.add_node("policy_coverage", node_policy_coverage)
    g.add_node("check_fraud", node_check_fraud)
    g.add_node("assess", node_assess)
    g.add_node("submit_for_approval", node_submit_for_approval)
    g.add_node("reject_auto", node_reject_auto)

    g.set_entry_point("create_claim")
    g.add_edge("create_claim", "analyze_document")
    g.add_edge("analyze_document", "validate_coverage")
    g.add_conditional_edges(
        "validate_coverage",
        after_validation,
        {"policy_coverage": "policy_coverage", "reject_auto": "reject_auto"},
    )
    g.add_conditional_edges(
        "policy_coverage",
        after_coverage,
        {"check_fraud": "check_fraud", "reject_auto": "reject_auto"},
    )
    g.add_conditional_edges(
        "check_fraud",
        after_fraud,
        {"assess": "assess", "reject_auto": "reject_auto"},
    )
    g.add_edge("assess", "submit_for_approval")
    g.add_edge("submit_for_approval", END)
    g.add_edge("reject_auto", END)

    return g.compile()


# Compiled app (import this elsewhere).
app = build_graph()


def run_claim(claim_input: Dict[str, Any], document_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the full intake pipeline for one claim and return the final state."""
    initial: ClaimState = {
        "claim_input": claim_input,
        "document_path": document_path,
        "trail": [],
    }
    final_state = app.invoke(initial)
    return {
        "claim_id": final_state.get("claim_id"),
        "status": final_state.get("status"),
        "status_reason": final_state.get("status_reason"),
        "trail": final_state.get("trail", []),
    }
