"""LangGraph workflow for new insurance applications (stage 1).

  create_application -> check_eligibility --(fail)--> reject_auto
                               |
                            (pass)
                               v
                       analyze_documents -> verify_documents -> price_and_risk
                                                                     |
                                                            submit_for_approval -> [HUMAN GATE]

Hard eligibility failures (age, vehicle age, coverage limits) are rejected
automatically. Everything else goes to a human with the agent's findings:
document cross-checks, risk level and the premium quote. Approving issues the
policy (see database.set_application_decision).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

import agentcore_client
import database as db
import llm
import underwriting_rules as uw
from tools.document_analyzer import analyze_document


class AppState(TypedDict, total=False):
    application_input: Dict[str, Any]
    application_id: str
    eligibility: Dict[str, Any]
    documents: List[Dict[str, Any]]
    verification: List[Dict[str, Any]]
    pricing: Dict[str, Any]
    status: str
    status_reason: str
    trail: List[str]


def _log(state: AppState, message: str) -> List[str]:
    trail = list(state.get("trail", []))
    trail.append(message)
    print(message)
    return trail


def _narrate(facts: Dict[str, Any]) -> str:
    """Short underwriter-style note from the model; plain fallback if unavailable."""
    if not llm.is_available():
        return ""
    prompt = (
        "You are an insurance underwriter. In two or three sentences, summarize "
        "this application review for the approving manager. Mention any "
        f"document mismatches and the main risk factors:\n{facts}"
    )
    try:
        if agentcore_client.enabled():
            return agentcore_client.narrate(prompt, max_tokens=300)
        return llm.complete(prompt, max_tokens=300)
    except Exception:
        return ""


def node_create(state: AppState) -> AppState:
    app_id = db.create_application(state["application_input"])
    return {"application_id": app_id, "trail": _log(state, f"[1/5] Created application {app_id}.")}


def node_eligibility(state: AppState) -> AppState:
    app = db.get_application(state["application_id"])
    result = uw.check_eligibility(app)
    db.update_application(state["application_id"], {"eligibility_result": result})
    failed = [c for c in result["checks"] if not c["passed"]]
    msg = "passed" if result["passed"] else "FAILED: " + "; ".join(c["detail"] for c in failed)
    return {"eligibility": result, "trail": _log(state, f"[2/5] Eligibility {msg}")}


def _analyze_one(doc: Dict[str, Any]) -> Dict[str, Any]:
    result = analyze_document(doc["path"], schema=uw.APPLICATION_DOC_SCHEMA)
    return {**doc, "document_type": result["document_type"], "extracted": result["extracted"]}


def node_analyze_documents(state: AppState) -> AppState:
    app = db.get_application(state["application_id"])
    pending = app.get("documents") or []
    # Read documents in parallel: each one is two model calls, so doing them
    # one after another made applicants wait several minutes.
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(pending)))) as pool:
        docs = list(pool.map(_analyze_one, pending))
    db.update_application(state["application_id"], {"documents": docs})
    read = sum(1 for d in docs if d["extracted"])
    return {"documents": docs, "trail": _log(state, f"[3/5] Read {read} of {len(docs)} documents.")}


def node_verify(state: AppState) -> AppState:
    app = db.get_application(state["application_id"])
    checks = uw.verify_documents(app, state.get("documents", []))
    db.update_application(state["application_id"], {"verification_result": checks})
    mismatches = [c for c in checks if c["status"] == "mismatch"]
    note = "; ".join(f"{c['document']}: {c['detail']}" for c in mismatches) or "no mismatches"
    return {"verification": checks, "trail": _log(state, f"[4/5] Document checks: {note}")}


def node_price(state: AppState) -> AppState:
    app = db.get_application(state["application_id"])
    pricing = uw.price_and_risk(app, state.get("verification", []))
    db.update_application(
        state["application_id"],
        {
            "pricing_result": pricing,
            "annual_premium": pricing["annual_premium"],
            "instalment_amount": pricing["instalment"],
            "risk_level": pricing["risk_level"],
        },
    )
    msg = (
        f"[5/5] {pricing['term_years']}-year plan, {pricing['payment_frequency']} "
        f"instalment ${pricing['instalment']:,.2f} ({pricing['instalments_count']} payments, "
        f"total ${pricing['total_premium']:,.2f}); risk {pricing['risk_level']} "
        f"(score {pricing['risk_score']})."
    )
    return {"pricing": pricing, "trail": _log(state, msg)}


def node_submit(state: AppState) -> AppState:
    trail = _log(state, "Submitted for underwriter approval.")
    summary = _narrate(
        {
            "verification": state.get("verification"),
            "pricing": state.get("pricing"),
        }
    )
    db.update_application(
        state["application_id"],
        {
            "status": db.STATUS_PENDING_APPROVAL,
            "status_reason": "Passed eligibility checks; awaiting underwriter approval.",
            "agent_summary": (summary + "\n\n" if summary else "") + "Trail:\n" + "\n".join(trail),
        },
    )
    return {"status": db.STATUS_PENDING_APPROVAL, "trail": trail}


def node_reject(state: AppState) -> AppState:
    failed = [c for c in state.get("eligibility", {}).get("checks", []) if not c["passed"]]
    reason = "; ".join(f"{c['rule']}: {c['detail']}" for c in failed) or "Not eligible."
    trail = _log(state, f"[x] Auto-rejected: {reason}")
    db.update_application(
        state["application_id"],
        {
            "status": db.STATUS_REJECTED_AUTO,
            "status_reason": reason,
            "agent_summary": "Trail:\n" + "\n".join(trail),
        },
    )
    return {"status": db.STATUS_REJECTED_AUTO, "status_reason": reason, "trail": trail}


def build_graph():
    g = StateGraph(AppState)
    g.add_node("create", node_create)
    g.add_node("eligibility", node_eligibility)
    g.add_node("analyze_documents", node_analyze_documents)
    g.add_node("verify", node_verify)
    g.add_node("price", node_price)
    g.add_node("submit", node_submit)
    g.add_node("reject", node_reject)

    g.set_entry_point("create")
    g.add_edge("create", "eligibility")
    g.add_conditional_edges(
        "eligibility",
        lambda s: "analyze_documents" if s["eligibility"]["passed"] else "reject",
        {"analyze_documents": "analyze_documents", "reject": "reject"},
    )
    g.add_edge("analyze_documents", "verify")
    g.add_edge("verify", "price")
    g.add_edge("price", "submit")
    g.add_edge("submit", END)
    g.add_edge("reject", END)
    return g.compile()


app = build_graph()


def run_application(application_input: Dict[str, Any]) -> Dict[str, Any]:
    final = app.invoke({"application_input": application_input, "trail": []})
    return {
        "application_id": final.get("application_id"),
        "status": final.get("status"),
        "trail": final.get("trail", []),
    }
