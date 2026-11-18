"""Policy coverage chatbot.

Answers "is this covered under my policy?" using ONLY the policy document for
the user's policy type, tells the user which document they would need, and
points them to the claim form when the incident is covered.

Privacy: the bot only discloses policy terms. Personal data and the money
figures (limit, deductible) are shared only when the customer_id matches the
policy owner.

With an OpenAI key the reply is written by the model, grounded in the policy
text. Without a key a keyword-based fallback gives the same structured answer.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import agentcore_client
import config
import database as db
import llm
import policy_rules

# Plain-language words -> incident_type used by the policy rules.
KEYWORDS: Dict[str, List[str]] = {
    "theft": ["stolen", "steal", "theft", "robbed", "robbery", "burglar", "break-in", "break in"],
    "collision": ["accident", "crash", "collision", "collided", "hit", "rear-ended", "bumped"],
    "vandalism": ["vandal", "keyed", "graffiti", "smashed"],
    "fire": ["fire", "burnt", "burned", "smoke"],
    "flood": ["flood", "flooded", "submerged"],
    "water_damage": ["water damage", "leak", "pipe burst", "burst pipe", "seepage"],
    "storm": ["storm", "hail", "cyclone", "hurricane", "tornado", "lightning"],
    "wear_and_tear": ["wear and tear", "wear & tear", "old age", "breakdown", "worn out"],
    "hospitalization": ["hospital", "admitted", "hospitalised", "hospitalized"],
    "surgery": ["surgery", "operation", "operated"],
    "emergency": ["emergency", "ambulance", "er visit"],
    "cosmetic": ["cosmetic", "plastic surgery", "botox", "aesthetic"],
    "death": ["died", "death", "passed away", "deceased"],
    "terminal_illness": ["terminal", "terminally ill"],
}


def _detect_incident(text: str) -> Optional[str]:
    t = text.lower()
    # Check longer phrases first so "plastic surgery" beats "surgery".
    pairs = [(kw, inc) for inc, kws in KEYWORDS.items() for kw in kws]
    for kw, inc in sorted(pairs, key=lambda p: -len(p[0])):
        if kw in t:
            return inc
    return None


def _owner_match(policy: Optional[Dict[str, Any]], customer_id: Optional[str]) -> bool:
    return bool(policy and customer_id and policy.get("customer_id") == customer_id)


def chat(
    message: str,
    history: List[Dict[str, str]],
    policy_number: Optional[str],
    customer_id: Optional[str],
) -> Dict[str, Any]:
    """Return {reply, incident_type, covered, document, can_file_claim}."""
    policy = db.get_policy(policy_number) if policy_number else None
    if policy and not _owner_match(policy, customer_id):
        # Do not confirm that someone else's policy exists.
        policy = None

    if not policy:
        return {
            "reply": (
                "Please enter your policy number and the email registered on it "
                "above so I can check the right policy for you. I can only answer "
                "from your own policy terms."
            ),
            "incident_type": None,
            "covered": None,
            "document": None,
            "can_file_claim": False,
        }

    policy_type = policy["policy_type"]
    incident = _detect_incident(message)
    # Fall back to the latest incident mentioned earlier in the chat.
    if not incident:
        for turn in reversed(history):
            if turn.get("role") == "user":
                incident = _detect_incident(turn.get("content", ""))
                if incident:
                    break

    if llm.is_available():
        try:
            return _llm_reply(message, history, policy, incident)
        except Exception:
            pass  # fall through to the rule-based answer
    return _rule_reply(policy, incident)


def _rule_reply(policy: Dict[str, Any], incident: Optional[str]) -> Dict[str, Any]:
    ptype = policy["policy_type"]
    allowed = policy_rules.incident_types_for_policy(ptype)

    if not incident:
        listed = ", ".join(i.replace("_", " ") for i in allowed)
        return {
            "reply": (
                f"Your policy is a {ptype} policy. Tell me what happened and I will "
                f"check it against your policy. Incidents listed under this policy "
                f"type: {listed}."
            ),
            "incident_type": None,
            "covered": None,
            "document": None,
            "can_file_claim": False,
        }

    req = policy_rules.incident_requirement(incident)
    name = incident.replace("_", " ")

    if incident not in allowed:
        return {
            "reply": (
                f"Under your {ptype} policy, '{name}' is not a listed covered "
                f"incident, so it would not be covered."
            ),
            "incident_type": incident,
            "covered": False,
            "document": None,
            "can_file_claim": False,
        }
    if req.get("covered") is False:
        return {
            "reply": (
                f"Under your {ptype} policy, '{name}' is specifically excluded, so "
                f"it would not be covered."
            ),
            "incident_type": incident,
            "covered": False,
            "document": None,
            "can_file_claim": False,
        }

    doc = req.get("document_label") or ""
    reply = f"Under your {ptype} policy, '{name}' is covered."
    if doc:
        reply += f" To claim, you will need: {doc}."
    reply += " The payout is subject to your deductible and coverage limit."
    return {
        "reply": reply,
        "incident_type": incident,
        "covered": True,
        "document": doc or None,
        "can_file_claim": True,
    }


def _llm_reply(
    message: str,
    history: List[Dict[str, str]],
    policy: Dict[str, Any],
    detected_incident: Optional[str],
) -> Dict[str, Any]:
    ptype = policy["policy_type"]
    if agentcore_client.enabled():
        # Hosted on AgentCore; errors fall back to the rule-based reply in chat().
        return agentcore_client.chat_reply(message, history, ptype, detected_incident)
    policy_doc = policy_rules.load_policy_document(ptype) or ""
    allowed = policy_rules.incident_types_for_policy(ptype)

    system = (
        "You are a friendly insurance assistant helping a customer understand "
        f"their own {ptype} insurance policy. Answer using the policy document "
        "below. You may:\n"
        "- Explain what the policy covers and excludes, in plain language.\n"
        "- List the covered incidents, the required documents and the claim process.\n"
        "- Summarize or describe the whole policy when asked a general question "
        "(e.g. 'tell me about all policies' or 'what does this cover').\n"
        "- When the user describes an incident, say clearly whether it is covered, "
        "cite the relevant clause and name the document required.\n"
        "Do not invent coverage that is not in the document. If something is not "
        "covered by the document, say so and suggest contacting support. Keep "
        "replies short and helpful, and always actually answer the question asked "
        "rather than only asking what happened.\n"
        f"Valid incident_type values for this policy: {allowed}.\n"
        "Respond as JSON with keys: reply (string, your answer to the user), "
        "incident_type (one valid value if the user described a specific incident, "
        "else null), covered (true/false if you judged a specific incident, else "
        "null), document (string or null).\n\n"
        f"=== {ptype.upper()} POLICY DOCUMENT ===\n{policy_doc}"
    )
    msgs = []
    for turn in history[-8:]:
        if turn.get("role") in ("user", "assistant"):
            msgs.append({"role": turn["role"], "content": turn.get("content", "")})
    msgs.append({"role": "user", "content": message})

    data = json.loads(
        llm.complete(messages=msgs, system=system, json_mode=True, max_tokens=700)
    )
    incident = data.get("incident_type") or detected_incident
    if incident not in allowed and incident not in policy_rules.INCIDENT_RULES:
        incident = None
    covered = data.get("covered")
    return {
        "reply": data.get("reply", "").strip() or "Sorry, I could not answer that.",
        "incident_type": incident,
        "covered": covered,
        "document": data.get("document"),
        "can_file_claim": covered is True and incident in allowed,
    }
