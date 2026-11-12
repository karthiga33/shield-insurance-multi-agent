"""SQLite data layer for the claim-intake agent.

Schema is based on the original multi-agent-system project (customers,
policies, claims) and extended so the agent can CREATE claims, attach the
analyzed document data, record validation / fraud / assessment results, and
drive a human-approval workflow.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import DB_PATH

# Claim lifecycle statuses used by the pipeline.
STATUS_DRAFT = "draft"                       # created, pipeline running
STATUS_REJECTED_AUTO = "rejected_auto"       # failed validation/fraud, never reaches human
STATUS_PENDING_APPROVAL = "pending_approval"  # passed all checks, waiting on human
STATUS_APPROVED = "approved"                 # human approved
STATUS_REJECTED = "rejected"                 # human rejected


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist. Does not drop existing data."""
    conn = connect()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id   TEXT PRIMARY KEY,
            first_name    TEXT,
            last_name     TEXT,
            email         TEXT,
            phone         TEXT,
            state         TEXT
        );

        CREATE TABLE IF NOT EXISTS policies (
            policy_number     TEXT PRIMARY KEY,
            customer_id       TEXT,
            policy_type       TEXT,          -- auto | home | health | life
            status            TEXT,          -- active | lapsed | cancelled
            premium_amount    REAL,
            coverage_limit    REAL,          -- max payout the policy covers
            deductible        REAL,          -- amount the claimant pays first
            effective_date    TEXT,
            expiry_date       TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        -- Covered incident types per policy type, used by coverage validation.
        CREATE TABLE IF NOT EXISTS coverage_rules (
            policy_type    TEXT,
            incident_type  TEXT,
            is_covered     INTEGER,          -- 1 covered, 0 excluded
            PRIMARY KEY (policy_type, incident_type)
        );

        CREATE TABLE IF NOT EXISTS claims (
            claim_id             TEXT PRIMARY KEY,
            tracking_id          TEXT UNIQUE,  -- short public reference the user tracks by
            policy_number        TEXT,
            customer_id          TEXT,
            claimant_name        TEXT,
            incident_type        TEXT,
            incident_date        TEXT,
            description          TEXT,
            claimed_amount       REAL,
            -- document analysis
            document_path        TEXT,
            document_type        TEXT,        -- pdf | image | scanned
            extracted_data       TEXT,        -- JSON from document analyzer
            -- pipeline results
            validation_result    TEXT,        -- JSON
            coverage_result      TEXT,        -- JSON (policy-document reasoning)
            fraud_result         TEXT,        -- JSON
            assessment_result    TEXT,        -- JSON
            assessed_amount      REAL,
            fraud_score          REAL,
            status               TEXT,        -- see STATUS_* constants
            status_reason        TEXT,
            agent_summary        TEXT,        -- human-readable decision trail
            -- human approval
            approver             TEXT,
            approval_notes       TEXT,
            created_at           TEXT,
            updated_at           TEXT,
            FOREIGN KEY (policy_number) REFERENCES policies(policy_number)
        );

        -- New insurance applications (stage 1, before a policy exists).
        CREATE TABLE IF NOT EXISTS applications (
            application_id       TEXT PRIMARY KEY,
            tracking_id          TEXT UNIQUE,   -- public reference, e.g. APL-7F3A2B
            full_name            TEXT,
            email                TEXT,
            phone                TEXT,
            date_of_birth        TEXT,
            address              TEXT,
            policy_type          TEXT,
            coverage_amount      REAL,
            details              TEXT,          -- JSON: type-specific answers
            documents            TEXT,          -- JSON: [{kind,label,path,document_type,extracted}]
            eligibility_result   TEXT,          -- JSON
            verification_result  TEXT,          -- JSON
            pricing_result       TEXT,          -- JSON
            annual_premium       REAL,
            risk_level           TEXT,
            status               TEXT,          -- same STATUS_* values as claims
            status_reason        TEXT,
            agent_summary        TEXT,
            approver             TEXT,
            approval_notes       TEXT,
            issued_customer_id   TEXT,          -- set when the policy is issued
            issued_policy_number TEXT,
            created_at           TEXT,
            updated_at           TEXT
        );
        """
    )
    # Columns added after the first release. ALTER TABLE keeps existing data.
    _ensure_columns(conn, "applications", {
        "term_years": "INTEGER",
        "payment_frequency": "TEXT",
        "instalment_amount": "REAL",
    })
    _ensure_columns(conn, "policies", {
        "term_years": "INTEGER",
        "payment_frequency": "TEXT",
        "instalment_amount": "REAL",
        "plan_details": "TEXT",      # JSON: quote, benefits, payment schedule
        "application_id": "TEXT",
    })
    conn.commit()
    conn.close()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


# --- Reference lookups -----------------------------------------------------
def get_policy(policy_number: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM policies WHERE policy_number = ?", (policy_number,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    policy = dict(row)
    if policy.get("plan_details"):
        try:
            policy["plan_details"] = json.loads(policy["plan_details"])
        except (TypeError, json.JSONDecodeError):
            policy["plan_details"] = None
    return policy


def get_customer(customer_id: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def find_policy_owner(policy_number: str, email: str) -> Optional[Dict[str, Any]]:
    """Return the customer who owns this policy if the email matches theirs.

    Email comparison ignores case and surrounding spaces. Returns None when the
    policy does not exist or the email does not match, without saying which.
    """
    if not policy_number or not email:
        return None
    conn = connect()
    row = conn.execute(
        """
        SELECT c.* FROM policies p
        JOIN customers c ON c.customer_id = p.customer_id
        WHERE p.policy_number = ? AND LOWER(TRIM(c.email)) = LOWER(TRIM(?))
        """,
        (policy_number.strip(), email),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def is_incident_covered(policy_type: str, incident_type: str) -> Optional[bool]:
    """Return True/False if a rule exists, else None (unknown)."""
    conn = connect()
    row = conn.execute(
        "SELECT is_covered FROM coverage_rules WHERE policy_type = ? AND incident_type = ?",
        (policy_type, incident_type),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return bool(row["is_covered"])


# --- Claim CRUD ------------------------------------------------------------
def _next_claim_id() -> str:
    conn = connect()
    row = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()
    conn.close()
    return f"CLM{str((row['n'] or 0) + 1).zfill(6)}"


def _new_tracking_id() -> str:
    """Short, human-friendly public reference, e.g. TRK-7F3A2B."""
    return "TRK-" + secrets.token_hex(3).upper()


def create_claim(data: Dict[str, Any]) -> str:
    """Insert a new claim in draft status. Returns the generated claim_id.

    A unique public tracking_id is generated so the customer can track the
    request without exposing the internal claim_id.
    """
    claim_id = _next_claim_id()
    tracking_id = _new_tracking_id()
    now = datetime.utcnow().isoformat()
    conn = connect()
    conn.execute(
        """
        INSERT INTO claims (
            claim_id, tracking_id, policy_number, customer_id, claimant_name,
            incident_type, incident_date, description, claimed_amount,
            document_path, document_type, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            claim_id,
            tracking_id,
            data.get("policy_number"),
            data.get("customer_id"),
            data.get("claimant_name"),
            data.get("incident_type"),
            data.get("incident_date"),
            data.get("description"),
            data.get("claimed_amount"),
            data.get("document_path"),
            data.get("document_type"),
            STATUS_DRAFT,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return claim_id


def update_claim(claim_id: str, fields: Dict[str, Any]) -> None:
    """Update arbitrary claim columns. Dict/list values are JSON-encoded."""
    if not fields:
        return
    encoded: Dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            encoded[key] = json.dumps(value)
        else:
            encoded[key] = value
    encoded["updated_at"] = datetime.utcnow().isoformat()

    cols = ", ".join(f"{k} = ?" for k in encoded)
    conn = connect()
    conn.execute(
        f"UPDATE claims SET {cols} WHERE claim_id = ?",
        (*encoded.values(), claim_id),
    )
    conn.commit()
    conn.close()


def _decode_claim(row: sqlite3.Row) -> Dict[str, Any]:
    claim = dict(row)
    for json_field in (
        "extracted_data",
        "validation_result",
        "coverage_result",
        "fraud_result",
        "assessment_result",
    ):
        raw = claim.get(json_field)
        if raw:
            try:
                claim[json_field] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                pass
    return claim


def get_claim(claim_id: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM claims WHERE claim_id = ?", (claim_id,)
    ).fetchone()
    conn.close()
    return _decode_claim(row) if row else None


def get_claim_by_tracking(tracking_id: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM claims WHERE tracking_id = ?", (tracking_id,)
    ).fetchone()
    conn.close()
    return _decode_claim(row) if row else None


def list_claims_for_customer(customer_id: str) -> List[Dict[str, Any]]:
    """All claims filed by one customer, newest first (for the user tracker)."""
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM claims WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    ).fetchall()
    conn.close()
    return [_decode_claim(r) for r in rows]


def list_claims(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = connect()
    if status:
        rows = conn.execute(
            "SELECT * FROM claims WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM claims ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [_decode_claim(r) for r in rows]


def set_human_decision(
    claim_id: str, approved: bool, approver: str, notes: str = ""
) -> None:
    update_claim(
        claim_id,
        {
            "status": STATUS_APPROVED if approved else STATUS_REJECTED,
            "approver": approver,
            "approval_notes": notes,
        },
    )


# --- Insurance applications ------------------------------------------------
_APP_JSON_FIELDS = (
    "details",
    "documents",
    "eligibility_result",
    "verification_result",
    "pricing_result",
)


def _next_numeric_id(conn: sqlite3.Connection, table: str, column: str, prefix: str, width: int) -> str:
    """Next id after the highest existing one, e.g. POL000080 -> POL000081."""
    row = conn.execute(
        f"SELECT MAX(CAST(SUBSTR({column}, {len(prefix) + 1}) AS INTEGER)) AS n "
        f"FROM {table} WHERE {column} LIKE ?",
        (prefix + "%",),
    ).fetchone()
    return f"{prefix}{str((row['n'] or 0) + 1).zfill(width)}"


def create_application(data: Dict[str, Any]) -> str:
    """Insert a new application in draft status. Returns the application_id."""
    now = datetime.utcnow().isoformat()
    conn = connect()
    application_id = _next_numeric_id(conn, "applications", "application_id", "APP", 6)
    tracking_id = "APL-" + secrets.token_hex(3).upper()
    conn.execute(
        """
        INSERT INTO applications (
            application_id, tracking_id, full_name, email, phone, date_of_birth,
            address, policy_type, coverage_amount, term_years, payment_frequency,
            details, documents, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            application_id,
            tracking_id,
            data.get("full_name"),
            (data.get("email") or "").strip().lower(),
            data.get("phone"),
            data.get("date_of_birth"),
            data.get("address"),
            data.get("policy_type"),
            data.get("coverage_amount"),
            data.get("term_years"),
            data.get("payment_frequency"),
            json.dumps(data.get("details") or {}),
            json.dumps(data.get("documents") or []),
            STATUS_DRAFT,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return application_id


def update_application(application_id: str, fields: Dict[str, Any]) -> None:
    if not fields:
        return
    encoded = {
        k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in fields.items()
    }
    encoded["updated_at"] = datetime.utcnow().isoformat()
    cols = ", ".join(f"{k} = ?" for k in encoded)
    conn = connect()
    conn.execute(
        f"UPDATE applications SET {cols} WHERE application_id = ?",
        (*encoded.values(), application_id),
    )
    conn.commit()
    conn.close()


def _decode_application(row: sqlite3.Row) -> Dict[str, Any]:
    app = dict(row)
    for field in _APP_JSON_FIELDS:
        raw = app.get(field)
        if raw:
            try:
                app[field] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                pass
    return app


def get_application(application_id: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM applications WHERE application_id = ?", (application_id,)
    ).fetchone()
    conn.close()
    return _decode_application(row) if row else None


def get_application_by_tracking(tracking_id: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM applications WHERE tracking_id = ?", (tracking_id,)
    ).fetchone()
    conn.close()
    return _decode_application(row) if row else None


def list_applications(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = connect()
    if status:
        rows = conn.execute(
            "SELECT * FROM applications WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [_decode_application(r) for r in rows]


def find_customer_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM customers WHERE LOWER(TRIM(email)) = LOWER(TRIM(?))",
        (email or "",),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def issue_policy(app: Dict[str, Any]) -> Dict[str, str]:
    """Create the customer (or reuse one with the same email) and an active
    policy for the chosen term from an approved application. One transaction."""
    from datetime import date

    import underwriting_rules as uw

    pricing = app.get("pricing_result") or {}
    today = date.today()
    term = int(app.get("term_years") or pricing.get("term_years") or 1)
    end = uw.add_years(today, term)
    plan = {
        **pricing,
        "start_date": today.isoformat(),
        "end_date": end.isoformat(),
        "schedule": uw.payment_schedule(today, pricing) if pricing.get("instalment") else [],
    }
    # Internal underwriting numbers do not belong on the customer's policy.
    for internal in ("risk_score", "risk_level", "base_rate", "total_multiplier"):
        plan.pop(internal, None)
    conn = connect()
    try:
        existing = conn.execute(
            "SELECT customer_id FROM customers WHERE LOWER(TRIM(email)) = LOWER(TRIM(?))",
            (app["email"],),
        ).fetchone()
        if existing:
            customer_id = existing["customer_id"]
        else:
            customer_id = _next_numeric_id(conn, "customers", "customer_id", "CUST", 5)
            parts = (app.get("full_name") or "").split()
            conn.execute(
                "INSERT INTO customers (customer_id, first_name, last_name, email, phone, state) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    customer_id,
                    parts[0] if parts else "",
                    " ".join(parts[1:]),
                    app["email"],
                    app.get("phone"),
                    None,
                ),
            )
        policy_number = _next_numeric_id(conn, "policies", "policy_number", "POL", 6)
        conn.execute(
            "INSERT INTO policies (policy_number, customer_id, policy_type, status, "
            "premium_amount, coverage_limit, deductible, effective_date, expiry_date, "
            "term_years, payment_frequency, instalment_amount, plan_details, application_id) "
            "VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                policy_number,
                customer_id,
                app["policy_type"],
                app.get("annual_premium") or pricing.get("annual_premium"),
                app.get("coverage_amount"),
                pricing.get("deductible", 0),
                today.isoformat(),
                end.isoformat(),
                term,
                app.get("payment_frequency") or pricing.get("payment_frequency") or "yearly",
                pricing.get("instalment"),
                json.dumps(plan),
                app.get("application_id"),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"customer_id": customer_id, "policy_number": policy_number}


def set_application_decision(
    application_id: str, approved: bool, approver: str, notes: str = ""
) -> Dict[str, Any]:
    """Approve (issues the policy) or reject an application."""
    app = get_application(application_id)
    fields: Dict[str, Any] = {
        "status": STATUS_APPROVED if approved else STATUS_REJECTED,
        "approver": approver,
        "approval_notes": notes,
    }
    if approved:
        issued = issue_policy(app)
        fields["issued_customer_id"] = issued["customer_id"]
        fields["issued_policy_number"] = issued["policy_number"]
    update_application(application_id, fields)
    return get_application(application_id)


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
