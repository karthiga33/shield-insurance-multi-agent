"""Insurance products and underwriting rules for new policy applications.

Single source of truth for:
  * what each policy type asks the applicant (form fields)
  * which documents each policy type requires
  * hard eligibility rules (failing one auto-rejects the application)
  * premium pricing and the risk score shown to the reviewer
  * cross-checking uploaded documents against what the applicant typed

All rules are deterministic so every decision is explainable. Change the
numbers here to change the product.
"""
from __future__ import annotations

import difflib
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# Field types used by the frontend: text, number, select, yesno.
PRODUCTS: Dict[str, Dict[str, Any]] = {
    "auto": {
        "label": "Auto insurance",
        "description": "Covers collision, theft, vandalism and fire damage to your vehicle.",
        "coverage_min": 2000,
        "coverage_max": 100000,
        "deductible": 500,
        "base_rate": 0.03,
        "terms": [1, 2, 3],
        "default_term": 1,
        "coverage_label": "Insured value of the vehicle",
        "fields": [
            {"name": "vehicle_number", "label": "Vehicle registration number", "type": "text", "required": True},
            {"name": "make_model", "label": "Vehicle make and model", "type": "text", "required": True},
            {"name": "vehicle_year", "label": "Year of manufacture", "type": "number", "required": True},
        ],
        "documents": [
            {"kind": "id_proof", "label": "ID proof (passport, national ID or similar)", "required": True},
            {"kind": "vehicle_registration", "label": "Vehicle registration certificate (RC)", "required": True},
            {"kind": "driving_licence", "label": "Driving licence", "required": True},
        ],
    },
    "home": {
        "label": "Home insurance",
        "description": "Covers fire, theft, water damage and storm damage to your home.",
        "coverage_min": 10000,
        "coverage_max": 1000000,
        "deductible": 1000,
        "base_rate": 0.004,
        "terms": [1, 3, 5],
        "default_term": 1,
        "coverage_label": "Sum insured for the home",
        "fields": [
            {"name": "property_address", "label": "Property address", "type": "text", "required": True},
            {"name": "property_type", "label": "Property type", "type": "select",
             "options": ["apartment", "house", "villa"], "required": True},
            {"name": "year_built", "label": "Year built", "type": "number", "required": True},
            {"name": "property_value", "label": "Property value ($)", "type": "number", "required": True},
        ],
        "documents": [
            {"kind": "id_proof", "label": "ID proof (passport, national ID or similar)", "required": True},
            {"kind": "property_document", "label": "Property ownership document", "required": True},
        ],
    },
    "health": {
        "label": "Health insurance",
        "description": "Covers hospitalization, surgery and emergency treatment.",
        "coverage_min": 5000,
        "coverage_max": 200000,
        "deductible": 250,
        "base_rate": 0.04,
        "terms": [1, 2, 3],
        "default_term": 1,
        "coverage_label": "Sum insured per policy year",
        # Claims for anything except emergencies are not payable in the first
        # days of the policy (enforced in tools/claim_tools.validate_coverage).
        "waiting_period_days": 30,
        "fields": [
            {"name": "height_cm", "label": "Height (cm)", "type": "number", "required": True},
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number", "required": True},
            {"name": "smoker", "label": "Do you smoke?", "type": "yesno", "required": True},
            {"name": "existing_conditions", "label": "Existing medical conditions (leave blank if none)",
             "type": "text", "required": False},
        ],
        "documents": [
            {"kind": "id_proof", "label": "ID proof (passport, national ID or similar)", "required": True},
            {"kind": "medical_declaration", "label": "Signed medical declaration", "required": True},
        ],
    },
    "life": {
        "label": "Life insurance (savings plan)",
        "description": "Pays you the sum assured plus bonus at the end of the term, or pays your "
        "nominee if you die during the term.",
        "coverage_min": 10000,
        "coverage_max": 1000000,
        "deductible": 0,
        # Yearly cost of the life cover itself, as a share of the sum assured.
        "base_rate": 0.008,
        "terms": [10, 15, 20, 25, 30],
        "default_term": 20,
        "coverage_label": "Sum assured",
        # Savings part: the sum assured spread over the term, plus a loading.
        "savings_loading": 1.05,
        # Illustrative simple bonus added each year, as a share of the sum assured.
        "bonus_rate": 0.03,
        "max_age_at_maturity": 75,
        "fields": [
            {"name": "occupation", "label": "Occupation", "type": "text", "required": True},
            {"name": "annual_income", "label": "Annual income ($)", "type": "number", "required": True},
            {"name": "smoker", "label": "Do you smoke?", "type": "yesno", "required": True},
            {"name": "nominee_name", "label": "Nominee (beneficiary) full name", "type": "text", "required": True},
        ],
        "documents": [
            {"kind": "id_proof", "label": "ID proof (passport, national ID or similar)", "required": True},
            {"kind": "income_proof", "label": "Income proof (salary slip or tax return)", "required": True},
        ],
    },
}

# Words we expect in the model's description of each document kind. Used to
# catch an obviously wrong upload (e.g. a selfie uploaded as a registration).
DOC_KIND_KEYWORDS: Dict[str, List[str]] = {
    "id_proof": ["id", "identity", "passport", "national", "aadhaar", "voter", "pan", "licence", "license", "card"],
    "vehicle_registration": ["registration", "rc", "vehicle", "certificate"],
    "driving_licence": ["licence", "license", "driving", "driver"],
    "property_document": ["property", "deed", "title", "ownership", "sale", "lease", "tax", "land"],
    "income_proof": ["salary", "payslip", "pay slip", "income", "tax", "bank", "statement", "form 16", "w-2", "w2"],
    "medical_declaration": ["medical", "health", "declaration", "doctor", "hospital", "report"],
}

# Labels too vague to judge the document type by.
GENERIC_DOC_KINDS = {"other", "unknown", "document", "n/a", "na", "none", "null", "image", "scan"}

# Fields the document analyzer pulls out of any application document.
APPLICATION_DOC_SCHEMA = {
    "full_name": "Full name of the person the document belongs to",
    "date_of_birth": "Date of birth in YYYY-MM-DD format, if shown",
    "id_number": "ID, licence or registration number, if shown",
    "vehicle_number": "Vehicle registration number, if shown",
    "vehicle_year": "Vehicle manufacture or registration year as a number, if shown",
    "address": "Address shown on the document",
    "annual_income": "Annual income as a number (no currency symbol), if shown",
    "document_kind": "What kind of document this is (ID card, passport, driving licence, "
    "vehicle registration certificate, property deed, salary slip, tax return, "
    "medical declaration, photo, other)",
}


# --- Helpers ---------------------------------------------------------------
def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def age_on(dob: Any, today: Optional[date] = None) -> Optional[int]:
    born = _parse_date(dob)
    if not born:
        return None
    today = today or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in ("yes", "y", "true", "1")


def _tokens(name: str) -> List[str]:
    return re.findall(r"[a-z]+", (name or "").lower())


def names_match(a: str, b: str) -> bool:
    """Lenient name comparison: shared name words, or near-identical spelling."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    shared = set(ta) & set(tb)
    if len(shared) >= min(2, len(ta), len(tb)):
        return True
    return difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() >= 0.85


def _alnum(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


# --- Validation of the submitted form --------------------------------------
def validate_form(
    policy_type: str,
    coverage: Any,
    details: Dict[str, Any],
    term_years: Any = None,
    frequency: Optional[str] = None,
) -> List[str]:
    """Return a list of problems with the submitted form (empty = OK)."""
    product = PRODUCTS.get(policy_type)
    if not product:
        return [f"Unknown policy type '{policy_type}'."]
    problems: List[str] = []
    term = _num(term_years)
    if term is None or int(term) not in product["terms"]:
        problems.append(
            f"Policy term for {product['label']} must be one of "
            f"{', '.join(str(t) for t in product['terms'])} years."
        )
    if frequency not in PAYMENT_FREQUENCIES:
        problems.append("Payment frequency must be yearly or monthly.")
    amount = _num(coverage)
    if amount is None:
        problems.append("Coverage amount must be a number.")
    elif not (product["coverage_min"] <= amount <= product["coverage_max"]):
        problems.append(
            f"Coverage for {product['label']} must be between "
            f"${product['coverage_min']:,} and ${product['coverage_max']:,}."
        )
    for field in product["fields"]:
        value = details.get(field["name"])
        if field["required"] and (value is None or str(value).strip() == ""):
            problems.append(f"'{field['label']}' is required.")
        elif field["type"] == "number" and value not in (None, "") and _num(value) is None:
            problems.append(f"'{field['label']}' must be a number.")
        elif field["type"] == "select" and value and value not in field["options"]:
            problems.append(f"'{field['label']}' must be one of {field['options']}.")
    return problems


# --- Eligibility (hard rules) ----------------------------------------------
def check_eligibility(app: Dict[str, Any]) -> Dict[str, Any]:
    """Hard rules. Any failed rule auto-rejects the application."""
    ptype = app.get("policy_type")
    details = app.get("details") or {}
    coverage = _num(app.get("coverage_amount")) or 0
    age = age_on(app.get("date_of_birth"))
    this_year = date.today().year
    checks: List[Dict[str, Any]] = []

    def rule(name: str, passed: bool, detail: str) -> None:
        checks.append({"rule": name, "passed": bool(passed), "detail": detail})

    if age is None:
        rule("Applicant age", False, "Date of birth is missing or invalid.")
    else:
        limits = {"health": (18, 65), "life": (18, 60)}.get(ptype, (18, 120))
        rule("Applicant age", limits[0] <= age <= limits[1],
             f"Age {age}; allowed {limits[0]}-{limits[1]}.")

    if ptype == "auto":
        year = _num(details.get("vehicle_year"))
        if year is None:
            rule("Vehicle age", False, "Year of manufacture is missing.")
        else:
            v_age = this_year - int(year)
            rule("Vehicle age", 0 <= v_age <= 15 and year <= this_year,
                 f"Vehicle is {v_age} years old; maximum 15.")
    elif ptype == "home":
        value = _num(details.get("property_value")) or 0
        rule("Coverage within property value", coverage <= value,
             f"Coverage ${coverage:,.0f} vs property value ${value:,.0f}.")
        built = _num(details.get("year_built"))
        rule("Year built", built is not None and 1900 <= built <= this_year,
             f"Built {int(built) if built else 'unknown'}; must be 1900 or later.")
    elif ptype == "life":
        income = _num(details.get("annual_income")) or 0
        rule("Coverage within 10x income", income > 0 and coverage <= income * 10,
             f"Coverage ${coverage:,.0f} vs 10x income ${income * 10:,.0f}.")
        term = int(_num(app.get("term_years")) or PRODUCTS["life"]["default_term"])
        max_age = PRODUCTS["life"]["max_age_at_maturity"]
        if age is not None:
            rule("Age at end of term", age + term <= max_age,
                 f"Age {age + term} at the end of a {term}-year term; maximum {max_age}.")

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


# --- Document cross-checks -------------------------------------------------
def verify_documents(app: Dict[str, Any], documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare what each document says with what the applicant typed.

    status is one of: ok, mismatch, unverified (nothing could be read).
    """
    details = app.get("details") or {}
    results: List[Dict[str, Any]] = []

    def add(doc: str, check: str, status: str, detail: str) -> None:
        results.append({"document": doc, "check": check, "status": status, "detail": detail})

    for doc in documents:
        kind, label = doc.get("kind"), doc.get("label", doc.get("kind"))
        ex = doc.get("extracted") or {}
        if not ex:
            add(label, "Readable", "unverified", "No details could be read from this document.")
            continue

        doc_kind_text = str(ex.get("document_kind") or "").strip().lower()
        keywords = DOC_KIND_KEYWORDS.get(kind, [])
        # A vague label ("other", "document") says nothing; only flag a clearly
        # different document type, e.g. a photo uploaded as a licence.
        vague = doc_kind_text in GENERIC_DOC_KINDS
        if doc_kind_text and not vague and keywords and not any(k in doc_kind_text for k in keywords):
            add(label, "Document type", "mismatch",
                f"This looks like '{ex.get('document_kind')}', not a {label.lower()}.")

        if ex.get("full_name"):
            ok = names_match(app.get("full_name", ""), ex["full_name"])
            add(label, "Name", "ok" if ok else "mismatch",
                f"Document name '{ex['full_name']}' vs applicant '{app.get('full_name')}'.")

        if kind == "id_proof" and ex.get("date_of_birth"):
            ok = _parse_date(ex["date_of_birth"]) == _parse_date(app.get("date_of_birth"))
            add(label, "Date of birth", "ok" if ok else "mismatch",
                f"Document DOB {ex['date_of_birth']} vs applicant {app.get('date_of_birth')}.")

        if kind == "vehicle_registration":
            if ex.get("vehicle_number"):
                ok = _alnum(ex["vehicle_number"]) == _alnum(details.get("vehicle_number"))
                add(label, "Vehicle number", "ok" if ok else "mismatch",
                    f"Document {ex['vehicle_number']} vs entered {details.get('vehicle_number')}.")
            doc_year, typed_year = _num(ex.get("vehicle_year")), _num(details.get("vehicle_year"))
            if doc_year and typed_year:
                ok = abs(doc_year - typed_year) <= 1
                add(label, "Vehicle year", "ok" if ok else "mismatch",
                    f"Document {int(doc_year)} vs entered {int(typed_year)}.")

        if kind == "income_proof":
            doc_income, typed_income = _num(ex.get("annual_income")), _num(details.get("annual_income"))
            if doc_income and typed_income:
                ok = doc_income >= typed_income * 0.8
                add(label, "Income", "ok" if ok else "mismatch",
                    f"Document shows ${doc_income:,.0f}; declared ${typed_income:,.0f}.")
    return results


# --- Pricing, payment schedule and benefits ----------------------------------
PAYMENT_FREQUENCIES: Dict[str, Dict[str, Any]] = {
    "yearly": {"label": "Yearly", "per_year": 1, "loading": 1.0},
    # Paying monthly costs 5% more in total (illustrative instalment loading).
    "monthly": {"label": "Monthly", "per_year": 12, "loading": 1.05},
}

ILLUSTRATIVE_NOTE = (
    "Premium, bonus and maturity figures are illustrative sample rates for this "
    "demo, not a guarantee."
)


def add_years(d: date, years: int) -> date:
    """Same day N years later (29 Feb falls back to 28 Feb)."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def premium_factors(ptype: str, dob: Any, details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Risk factors that raise the premium, each with its multiplier."""
    age = age_on(dob) or 0
    this_year = date.today().year
    factors: List[Dict[str, Any]] = []

    def factor(name: str, multiplier: float) -> None:
        factors.append({"factor": name, "multiplier": multiplier})

    if ptype == "auto":
        v_age = this_year - int(_num(details.get("vehicle_year")) or this_year)
        if v_age > 8:
            factor(f"Vehicle older than 8 years ({v_age})", 1.3)
        if age < 25:
            factor(f"Driver under 25 ({age})", 1.4)
    elif ptype == "home":
        built = int(_num(details.get("year_built")) or this_year)
        if this_year - built > 40:
            factor(f"Property older than 40 years (built {built})", 1.25)
    elif ptype == "health":
        if age > 45:
            factor(f"Age over 45 ({age})", 1.5)
        if is_yes(details.get("smoker")):
            factor("Smoker", 1.3)
        h, w = _num(details.get("height_cm")), _num(details.get("weight_kg"))
        if h and w:
            bmi = w / ((h / 100) ** 2)
            if bmi > 30:
                factor(f"BMI over 30 ({bmi:.1f})", 1.2)
        if str(details.get("existing_conditions") or "").strip():
            factor("Declared existing medical conditions", 1.25)
    elif ptype == "life":
        if age > 40:
            factor(f"Age over 40 ({age})", 1.6)
        if is_yes(details.get("smoker")):
            factor("Smoker", 1.5)
    return factors


def compute_quote(
    ptype: str,
    coverage: Any,
    term_years: Any,
    frequency: str,
    dob: Any,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    """Premium, instalments, total payable and the benefits for a plan choice."""
    product = PRODUCTS[ptype]
    coverage = _num(coverage) or 0.0
    term = int(_num(term_years) or product["default_term"])
    freq = PAYMENT_FREQUENCIES.get(frequency) or PAYMENT_FREQUENCIES["yearly"]
    factors = premium_factors(ptype, dob, details)
    multiplier = 1.0
    for f in factors:
        multiplier *= f["multiplier"]

    # Cost of the cover itself each year.
    risk_part = coverage * product["base_rate"] * multiplier
    # Life savings plan: the sum assured is built up over the term.
    savings_part = coverage / term * product["savings_loading"] if ptype == "life" else 0.0
    annual = max(round(risk_part + savings_part, 2), 100.0)
    instalment = round(annual / freq["per_year"] * freq["loading"], 2)
    count = term * freq["per_year"]

    return {
        "policy_type": ptype,
        "coverage": coverage,
        "coverage_label": product["coverage_label"],
        "term_years": term,
        "payment_frequency": frequency if frequency in PAYMENT_FREQUENCIES else "yearly",
        "payment_frequency_label": freq["label"],
        "annual_premium": annual,
        "instalment": instalment,
        "instalments_count": count,
        "total_premium": round(instalment * count, 2),
        "deductible": product["deductible"],
        "factors": factors,
        "total_multiplier": round(multiplier, 3),
        "benefits": benefits_for(ptype, coverage, term, instalment * freq["per_year"]),
        "note": ILLUSTRATIVE_NOTE,
    }


def benefits_for(ptype: str, coverage: float, term: int, paid_per_year: float) -> Dict[str, Any]:
    """What the policyholder gets, in plain terms, for each policy type."""
    product = PRODUCTS[ptype]
    if ptype == "life":
        bonus = coverage * product["bonus_rate"]
        # Year-by-year view: what has been paid, what the family would get, and
        # (final year) what the policyholder receives at maturity.
        marks = sorted({y for y in range(5, term + 1, 5)} | {1, term})
        projection = [
            {
                "year": y,
                "premiums_paid": round(paid_per_year * y, 2),
                "death_benefit": round(coverage + bonus * y, 2),
            }
            for y in marks
        ]
        maturity = round(coverage + bonus * term, 2)
        return {
            "summary": [
                f"Pay for {term} years, then receive ${maturity:,.0f} at the end of the term "
                f"(sum assured ${coverage:,.0f} plus bonus).",
                f"If the insured person dies during the term, the nominee receives the sum "
                f"assured plus the bonus added so far.",
                f"A simple bonus of {product['bonus_rate'] * 100:.0f}% of the sum assured "
                f"(${bonus:,.0f}) is added each completed year.",
            ],
            "maturity_benefit": maturity,
            "bonus_per_year": round(bonus, 2),
            "projection": projection,
        }
    if ptype == "health":
        wait = product["waiting_period_days"]
        return {
            "summary": [
                f"Covers hospitalization, surgery and emergency treatment up to "
                f"${coverage:,.0f} in each policy year, for {term} year(s).",
                f"Waiting period: claims (except emergencies) can be made {wait} days "
                f"after the policy starts. Emergencies are covered from day one.",
                f"You pay the first ${product['deductible']:,} of each claim (deductible).",
                "Cosmetic procedures are not covered.",
            ],
            "waiting_period_days": wait,
        }
    covered = {
        "auto": "collision, theft, vandalism and fire",
        "home": "fire, theft, water damage and storm",
    }[ptype]
    return {
        "summary": [
            f"Covers {covered} up to ${coverage:,.0f}, for {term} year(s). "
            f"Claims are payable from the day the policy starts.",
            f"You pay the first ${product['deductible']:,} of each claim (deductible).",
            "Flood and wear and tear are not covered.",
            "Renew before the end date to keep your cover.",
        ],
    }


def payment_schedule(start: date, quote: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per policy year: when payments fall due and how much."""
    rows = []
    per_year = PAYMENT_FREQUENCIES[quote["payment_frequency"]]["per_year"]
    paid = 0.0
    for year in range(1, quote["term_years"] + 1):
        year_start = add_years(start, year - 1)
        amount = round(quote["instalment"] * per_year, 2)
        paid += amount
        rows.append(
            {
                "policy_year": year,
                "from": year_start.isoformat(),
                "to": (add_years(start, year)).isoformat(),
                "first_due": year_start.isoformat(),
                "instalments": per_year,
                "instalment": quote["instalment"],
                "amount_for_year": amount,
                "total_paid_to_date": round(paid, 2),
            }
        )
    return rows


def price_and_risk(app: Dict[str, Any], verification: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quote for the chosen plan, plus a risk score for the reviewer."""
    quote = compute_quote(
        app["policy_type"],
        app.get("coverage_amount"),
        app.get("term_years"),
        app.get("payment_frequency") or "yearly",
        app.get("date_of_birth"),
        app.get("details") or {},
    )
    mismatches = [v for v in verification if v["status"] == "mismatch"]
    unverified = [v for v in verification if v["status"] == "unverified"]
    risk_points = len(quote["factors"]) + 2 * len(mismatches) + 0.5 * len(unverified)
    level = "high" if risk_points >= 4 else "medium" if risk_points >= 2 else "low"
    return {
        **quote,
        "base_rate": PRODUCTS[app["policy_type"]]["base_rate"],
        "risk_score": round(risk_points, 1),
        "risk_level": level,
    }


def public_products() -> Dict[str, Any]:
    """Product definitions for the application form (no pricing internals)."""
    return {
        key: {
            "label": p["label"],
            "description": p["description"],
            "coverage_min": p["coverage_min"],
            "coverage_max": p["coverage_max"],
            "coverage_label": p["coverage_label"],
            "deductible": p["deductible"],
            "terms": p["terms"],
            "default_term": p["default_term"],
            "fields": p["fields"],
            "documents": p["documents"],
        }
        for key, p in PRODUCTS.items()
    }


def payment_frequencies() -> Dict[str, str]:
    return {k: v["label"] for k, v in PAYMENT_FREQUENCIES.items()}
