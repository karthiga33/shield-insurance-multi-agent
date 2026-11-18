"""FastAPI backend for the claim-intake agent.

Two audiences:

USER (identified by policy number + the email registered on that policy)
  GET  /                                        User portal (submit + track own claims)
  GET  /api/incident-requirements               What document each incident type needs
  GET  /api/policy-info?policy_number=&email=   Incident types for the user's policy
  POST /api/chat                                Coverage chatbot
  POST /api/claims                              Submit a claim; returns a tracking_id
  GET  /api/track/{tracking_id}?email=          Track one claim
  GET  /api/my-claims?policy_number=&email=     List the customer's claims

  Customer IDs are internal. A wrong policy/email pair gets the same generic
  "not found" answer as a missing policy, so nothing leaks about other people.

APPLY FOR INSURANCE (stage 1, no policy yet)
  GET  /api/products                            Policy types, questions, documents
  POST /api/applications                        Apply; returns an APL- tracking id
  GET  /api/applications/track/{id}?email=      Track an application
  GET  /api/admin/applications?status=          (admin) all applications
  POST /api/admin/applications/{id}/decision    (admin) approve -> issues policy

  User-facing responses are SANITIZED: they expose status, assessed amount and a
  simple message, but NOT the fraud score, internal claim_id, or raw agent trail.

ADMIN (shared passcode via X-Admin-Token header)
  GET  /admin                            Redirects to the Admin tab of the main page
  GET  /api/admin/claims?status=...      All claims, full detail
  GET  /api/admin/claims/{claim_id}      One claim, full detail
  POST /api/admin/claims/{claim_id}/decision   Approve / reject

Run:  python -m uvicorn api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

import config
import database as db
import policy_rules
import underwriting_rules as uw
from application_graph import run_application
from graph import run_claim
from tools import chatbot

app = FastAPI(title="Insurance Claim Intake Agent")

# Shared admin passcode. Override with env ADMIN_TOKEN; default for local demo.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin123")

db.init_db()


# --- Helpers ---------------------------------------------------------------
def _require_admin(token: Optional[str]) -> None:
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


NO_MATCH = "We could not find a policy with that policy number and email."

ALLOWED_UPLOAD_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per file
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _save_upload(upload) -> str:
    """Save an uploaded file under a unique name and return its path.

    Unique names stop two customers' files with the same name (e.g. id.jpg)
    from overwriting each other. Only document/image types are accepted, and
    files over MAX_UPLOAD_BYTES are refused.
    """
    original = Path(upload.filename or "").name
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"'{original}' is not a supported file. Upload a PDF or an image.",
        )
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(original).stem)[:40] or "file"
    dest = Path(config.UPLOAD_DIR) / f"{uuid.uuid4().hex}_{stem}{ext}"
    size = 0
    with open(dest, "wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400, detail=f"'{original}' is larger than 10 MB."
                )
            fh.write(chunk)
    return str(dest)


def _owner_or_404(policy_number: str, email: str) -> Dict[str, Any]:
    customer = db.find_policy_owner(policy_number, email)
    if not customer:
        raise HTTPException(status_code=404, detail=NO_MATCH)
    return customer


def _user_view(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal fields so a customer only sees what they should."""
    status = claim.get("status")
    friendly = {
        db.STATUS_DRAFT: "We are processing your request.",
        db.STATUS_REJECTED_AUTO: "Your claim did not meet the policy requirements.",
        db.STATUS_PENDING_APPROVAL: "Your claim passed our checks and is awaiting final approval.",
        db.STATUS_APPROVED: "Your claim has been approved.",
        db.STATUS_REJECTED: "Your claim was reviewed and could not be approved.",
    }.get(status, "Status unavailable.")
    return {
        "tracking_id": claim.get("tracking_id"),
        "claimant_name": claim.get("claimant_name"),
        "policy_number": claim.get("policy_number"),
        "incident_type": claim.get("incident_type"),
        "incident_date": claim.get("incident_date"),
        "claimed_amount": claim.get("claimed_amount"),
        "status": status,
        "message": friendly,
        # The user may see the payout and the reason, but not fraud internals.
        "assessed_amount": claim.get("assessed_amount")
        if status in (db.STATUS_PENDING_APPROVAL, db.STATUS_APPROVED)
        else None,
        # After a human decision the pipeline note ("passed checks...") is out
        # of date, so show the reviewer's notes instead.
        "reason": (claim.get("approval_notes") or None)
        if status in (db.STATUS_APPROVED, db.STATUS_REJECTED)
        else claim.get("status_reason"),
        "created_at": claim.get("created_at"),
    }


# ===========================================================================
# USER ENDPOINTS
# ===========================================================================
@app.get("/api/incident-requirements")
async def incident_requirements():
    """Drives the dynamic 'which document do we need' prompt in the user form."""
    return {"requirements": policy_rules.all_incident_requirements()}


@app.get("/api/policy-info")
async def policy_info(policy_number: str, email: str):
    """Policy type + the incident types that apply, only for the policy owner."""
    _owner_or_404(policy_number, email)
    policy = db.get_policy(policy_number.strip())
    return {
        "policy_type": policy["policy_type"],
        "status": policy["status"],
        "incident_types": policy_rules.incident_types_for_policy(policy["policy_type"]),
    }


class ChatIn(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    policy_number: Optional[str] = None
    email: Optional[str] = None


@app.post("/api/chat")
async def chat_endpoint(body: ChatIn):
    # Resolve the owner from policy + email; a mismatch means "no policy" to
    # the chatbot, which then asks the user for their details.
    customer = db.find_policy_owner(body.policy_number or "", body.email or "")
    policy_number = body.policy_number.strip() if customer else None
    customer_id = customer["customer_id"] if customer else None
    return chatbot.chat(body.message, body.history, policy_number, customer_id)


@app.post("/api/claims")
async def submit_claim(
    claimant_name: str = Form(...),
    email: str = Form(...),
    policy_number: str = Form(...),
    incident_type: str = Form(...),
    incident_date: str = Form(...),
    description: str = Form(""),
    claimed_amount: float = Form(...),
    document: Optional[UploadFile] = File(None),
):
    """Accept a new claim and run the autonomous pipeline. Returns tracking_id."""
    # The email must match the one registered on this policy.
    customer = db.find_policy_owner(policy_number, email)
    if not customer:
        raise HTTPException(status_code=400, detail=NO_MATCH)
    customer_id = customer["customer_id"]
    policy_number = policy_number.strip()

    # Enforce the policy's required-document rule up front for a clear message.
    req = policy_rules.incident_requirement(incident_type)
    has_doc = document is not None and bool(document.filename)
    if req.get("document_required") and not has_doc:
        raise HTTPException(
            status_code=400,
            detail=f"This incident type requires a document: {req['document_label']}",
        )

    document_path = _save_upload(document) if has_doc else None

    claim_input = {
        "claimant_name": claimant_name,
        "customer_id": customer_id,
        "policy_number": policy_number,
        "incident_type": incident_type,
        "incident_date": incident_date,
        "description": description,
        "claimed_amount": claimed_amount,
    }
    result = run_claim(claim_input, document_path=document_path)
    claim = db.get_claim(result["claim_id"])
    return JSONResponse(
        {
            "tracking_id": claim["tracking_id"],
            "customer_id": customer_id,
            "status": claim["status"],
            "view": _user_view(claim),
        }
    )


@app.get("/api/track/{tracking_id}")
async def track(tracking_id: str, email: str):
    """Track one claim. The email must match the claimant's registered email."""
    claim = db.get_claim_by_tracking(tracking_id.strip().upper())
    customer = db.get_customer(claim["customer_id"]) if claim else None
    if not customer or (customer.get("email") or "").strip().lower() != email.strip().lower():
        raise HTTPException(
            status_code=404, detail="No claim found for that tracking ID and email."
        )
    return _user_view(claim)


@app.get("/api/my-claims")
async def my_claims(policy_number: str, email: str):
    customer = _owner_or_404(policy_number, email)
    claims = db.list_claims_for_customer(customer["customer_id"])
    return {"claims": [_user_view(c) for c in claims]}


# ===========================================================================
# INSURANCE APPLICATIONS (stage 1: apply for a new policy)
# ===========================================================================
def _application_view(app: Dict[str, Any]) -> Dict[str, Any]:
    """What an applicant may see: no risk score, checks or internal notes."""
    status = app.get("status")
    message = {
        db.STATUS_DRAFT: "We are reviewing your application.",
        db.STATUS_REJECTED_AUTO: "Your application does not meet our eligibility rules.",
        db.STATUS_PENDING_APPROVAL: "Your application passed our checks and is awaiting final approval.",
        db.STATUS_APPROVED: "Your application is approved and your policy is active.",
        db.STATUS_REJECTED: "Your application was reviewed and could not be approved.",
    }.get(status, "Status unavailable.")
    if status == db.STATUS_REJECTED_AUTO:
        reason = app.get("status_reason")
    elif status in (db.STATUS_APPROVED, db.STATUS_REJECTED):
        reason = app.get("approval_notes") or None
    else:
        reason = None
    product = uw.PRODUCTS.get(app.get("policy_type"), {})
    show_premium = status in (db.STATUS_PENDING_APPROVAL, db.STATUS_APPROVED)
    pricing = app.get("pricing_result") or {}
    view = {
        "tracking_id": app.get("tracking_id"),
        "kind": "application",
        "full_name": app.get("full_name"),
        "policy_type": app.get("policy_type"),
        "policy_label": product.get("label", app.get("policy_type")),
        "coverage_amount": app.get("coverage_amount"),
        "coverage_label": product.get("coverage_label", "Coverage"),
        "term_years": app.get("term_years"),
        "payment_frequency": app.get("payment_frequency"),
        "annual_premium": app.get("annual_premium") if show_premium else None,
        "instalment": pricing.get("instalment") if show_premium else None,
        "instalments_count": pricing.get("instalments_count") if show_premium else None,
        "total_premium": pricing.get("total_premium") if show_premium else None,
        "benefits": pricing.get("benefits") if show_premium else None,
        "status": status,
        "message": message,
        "reason": reason,
        "policy_number": None,
        "created_at": app.get("created_at"),
    }
    if status == db.STATUS_APPROVED and app.get("issued_policy_number"):
        pol_no = app["issued_policy_number"]
        view["policy_number"] = pol_no
        policy = db.get_policy(pol_no) or {}
        view["policy"] = {
            "start_date": policy.get("effective_date"),
            "end_date": policy.get("expiry_date"),
            "plan": policy.get("plan_details"),
        }
        view["policy_pdf_url"] = (
            f"/api/applications/{app['tracking_id']}/policy.pdf"
            f"?email={app.get('email', '')}"
        )
    return view


@app.get("/api/products")
async def products():
    """Policy types, the questions they ask and the documents they need."""
    return {"products": uw.public_products(), "payment_frequencies": uw.payment_frequencies()}


class QuoteIn(BaseModel):
    policy_type: str
    coverage_amount: float = 0
    term_years: int = 0
    payment_frequency: str = "yearly"
    date_of_birth: Optional[str] = None
    details: Dict[str, Any] = {}


@app.post("/api/quote")
async def quote(body: QuoteIn):
    """Live premium / benefit quote while the applicant fills the form."""
    if body.policy_type not in uw.PRODUCTS:
        raise HTTPException(status_code=400, detail="Unknown policy type.")
    return uw.compute_quote(
        body.policy_type, body.coverage_amount, body.term_years or None,
        body.payment_frequency, body.date_of_birth, body.details or {},
    )


@app.post("/api/applications")
async def submit_application(request: Request):
    """Apply for a new policy. Runs the underwriting pipeline; returns an APL- id."""
    form = await request.form()

    def text(key: str) -> str:
        value = form.get(key)
        return value.strip() if isinstance(value, str) else ""

    personal = {
        "full_name": text("full_name"),
        "email": text("email").lower(),
        "phone": text("phone"),
        "date_of_birth": text("date_of_birth"),
        "address": text("address"),
    }
    missing = [k.replace("_", " ") for k, v in personal.items() if not v]
    if missing:
        raise HTTPException(status_code=400, detail="Please fill in: " + ", ".join(missing) + ".")
    if not EMAIL_RE.match(personal["email"]):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    policy_type = text("policy_type")
    product = uw.PRODUCTS.get(policy_type)
    if not product:
        raise HTTPException(status_code=400, detail="Please choose a policy type.")

    try:
        raw_details = json.loads(text("details") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid application details.")
    if not isinstance(raw_details, dict):
        raise HTTPException(status_code=400, detail="Invalid application details.")
    # Keep only the fields this product asks for.
    allowed = {f["name"] for f in product["fields"]}
    details = {k: str(v).strip() for k, v in raw_details.items() if k in allowed}

    term_years = text("term_years")
    frequency = text("payment_frequency") or "yearly"
    problems = uw.validate_form(policy_type, text("coverage_amount"), details, term_years, frequency)
    if problems:
        raise HTTPException(status_code=400, detail=" ".join(problems))

    # Check every required document is present and of an allowed type before
    # saving anything, so a bad upload never leaves half-saved files behind.
    uploads = []
    for doc in product["documents"]:
        upload = form.get(f"doc_{doc['kind']}")
        has_file = bool(getattr(upload, "filename", ""))
        if not has_file:
            if doc["required"]:
                raise HTTPException(status_code=400, detail=f"Please upload: {doc['label']}.")
            continue
        if Path(upload.filename).suffix.lower() not in ALLOWED_UPLOAD_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' is not a supported file. Upload a PDF or an image.",
            )
        uploads.append((doc, upload))
    documents = [
        {"kind": doc["kind"], "label": doc["label"], "path": _save_upload(upload)}
        for doc, upload in uploads
    ]

    result = run_application(
        {
            **personal,
            "policy_type": policy_type,
            "coverage_amount": float(text("coverage_amount").replace(",", "")),
            "term_years": int(float(term_years)),
            "payment_frequency": frequency,
            "details": details,
            "documents": documents,
        }
    )
    app_row = db.get_application(result["application_id"])
    return {
        "tracking_id": app_row["tracking_id"],
        "status": app_row["status"],
        "view": _application_view(app_row),
    }


def _application_for(tracking_id: str, email: str) -> Dict[str, Any]:
    app_row = db.get_application_by_tracking(tracking_id.strip().upper())
    if not app_row or (app_row.get("email") or "").lower() != email.strip().lower():
        raise HTTPException(
            status_code=404, detail="No application found for that ID and email."
        )
    return app_row


@app.get("/api/applications/track/{tracking_id}")
async def track_application(tracking_id: str, email: str):
    return _application_view(_application_for(tracking_id, email))


@app.get("/api/applications/{tracking_id}/policy.pdf")
async def download_policy_pdf(tracking_id: str, email: str):
    """Download the policy document. Needs the applicant's email, and only
    works once the application is approved and a policy has been issued."""
    app_row = _application_for(tracking_id, email)
    if app_row["status"] != db.STATUS_APPROVED or not app_row.get("issued_policy_number"):
        raise HTTPException(status_code=400, detail="No policy has been issued yet.")
    from policy_pdf import generate_policy_pdf

    path = generate_policy_pdf(app_row["issued_policy_number"])
    return FileResponse(
        path, media_type="application/pdf",
        filename=f"policy_{app_row['issued_policy_number']}.pdf",
    )


# ===========================================================================
# ADMIN ENDPOINTS
# ===========================================================================
@app.get("/api/admin/login")
async def admin_login(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {"ok": True}


@app.get("/api/admin/claims")
async def admin_list(status: Optional[str] = None, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {"claims": db.list_claims(status)}


_PREVIEW_TYPES = {
    ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}


def _serve_upload(path: Optional[str]) -> FileResponse:
    """Serve a stored upload. Only paths recorded on a claim/application reach
    here, and they must sit inside the uploads folder."""
    if not path:
        raise HTTPException(status_code=404, detail="No document uploaded.")
    file = Path(path).resolve()
    if Path(config.UPLOAD_DIR).resolve() not in file.parents or not file.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    media = _PREVIEW_TYPES.get(file.suffix.lower(), "application/octet-stream")
    return FileResponse(str(file), media_type=media)


@app.get("/api/admin/claims/{claim_id}/document")
async def admin_claim_document(claim_id: str, x_admin_token: Optional[str] = Header(None)):
    """The document uploaded with a claim, for the reviewer to look at."""
    _require_admin(x_admin_token)
    claim = db.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return _serve_upload(claim.get("document_path"))


@app.get("/api/admin/applications/{application_id}/documents/{index}")
async def admin_application_document(
    application_id: str, index: int, x_admin_token: Optional[str] = Header(None)
):
    """One of the documents uploaded with an application (by position)."""
    _require_admin(x_admin_token)
    app_row = db.get_application(application_id)
    docs = (app_row or {}).get("documents") or []
    if not app_row or not (0 <= index < len(docs)):
        raise HTTPException(status_code=404, detail="Document not found")
    return _serve_upload(docs[index].get("path"))


@app.get("/api/admin/claims/{claim_id}")
async def admin_get(claim_id: str, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    claim = db.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


@app.post("/api/admin/claims/{claim_id}/decision")
async def admin_decide(
    claim_id: str,
    approved: bool = Form(...),
    approver: str = Form("admin"),
    notes: str = Form(""),
    x_admin_token: Optional[str] = Header(None),
):
    _require_admin(x_admin_token)
    claim = db.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    if claim["status"] != db.STATUS_PENDING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Claim is '{claim['status']}', not pending approval.",
        )
    db.set_human_decision(claim_id, approved, approver, notes)
    return {"claim_id": claim_id, "status": db.get_claim(claim_id)["status"]}


@app.get("/api/admin/applications")
async def admin_list_applications(
    status: Optional[str] = None, x_admin_token: Optional[str] = Header(None)
):
    _require_admin(x_admin_token)
    return {"applications": db.list_applications(status)}


@app.post("/api/admin/applications/{application_id}/decision")
async def admin_decide_application(
    application_id: str,
    approved: bool = Form(...),
    approver: str = Form("admin"),
    notes: str = Form(""),
    x_admin_token: Optional[str] = Header(None),
):
    """Approve (issues the policy number) or reject a pending application."""
    _require_admin(x_admin_token)
    app_row = db.get_application(application_id)
    if not app_row:
        raise HTTPException(status_code=404, detail="Application not found")
    if app_row["status"] != db.STATUS_PENDING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Application is '{app_row['status']}', not pending approval.",
        )
    updated = db.set_application_decision(application_id, approved, approver, notes)
    return {
        "application_id": application_id,
        "status": updated["status"],
        "policy_number": updated.get("issued_policy_number"),
        "customer_id": updated.get("issued_customer_id"),
    }


# ===========================================================================
# FRONTEND ROUTES
# ===========================================================================
@app.get("/health")
async def health():
    """Load-balancer health check: the app is up and the database answers."""
    conn = db.connect()
    try:
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()
    return {"status": "ok"}


@app.get("/")
async def user_portal():
    return FileResponse(str(config.FRONTEND_DIR / "user.html"))


@app.get("/admin")
async def admin_portal():
    # The admin console now lives in the main page's Admin tab.
    return RedirectResponse(url="/#/admin")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
