"""Generate a policy document PDF for an issued policy.

Reads the policy row (and the plan_details JSON saved when it was issued) and
produces a proper, printable policy document with the schedule of benefits and
the payment schedule. All figures are the illustrative sample rates from
underwriting_rules.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config
import database as db
import underwriting_rules as uw

INSURER = "Shield Insurance (Demo)"
BLUE = colors.HexColor("#1e40af")


def _money(v: Any) -> str:
    try:
        return "$" + format(float(v), ",.2f")
    except (TypeError, ValueError):
        return "-"


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], textColor=BLUE, fontSize=20))
    s.add(ParagraphStyle("Center", parent=s["Normal"], alignment=TA_CENTER, fontSize=9,
                         textColor=colors.grey))
    s.add(ParagraphStyle("H", parent=s["Heading2"], textColor=BLUE, fontSize=12,
                         spaceBefore=12, spaceAfter=4))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=10, leading=15))
    s.add(ParagraphStyle("Small", parent=s["Normal"], fontSize=8, textColor=colors.grey))
    return s


def _kv_table(rows, styles, widths=(55 * mm, 110 * mm)):
    data = [[Paragraph(f"<b>{k}</b>", styles["Body"]), Paragraph(str(v), styles["Body"])] for k, v in rows]
    t = Table(data, colWidths=list(widths))
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]))
    return t


def _grid(header, rows, styles, widths):
    data = [[Paragraph(f"<b>{h}</b>", styles["Body"]) for h in header]]
    data += [[Paragraph(str(c), styles["Body"]) for c in r] for r in rows]
    t = Table(data, colWidths=list(widths), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def generate_policy_pdf(policy_number: str, out_path: Optional[str] = None) -> str:
    """Write the policy document PDF and return its path."""
    policy = db.get_policy(policy_number)
    if not policy:
        raise ValueError(f"Policy {policy_number} not found")
    customer = db.get_customer(policy["customer_id"]) or {}
    plan: Dict[str, Any] = policy.get("plan_details") or {}
    product = uw.PRODUCTS.get(policy["policy_type"], {})
    styles = _styles()

    out_path = out_path or str(Path(config.UPLOAD_DIR) / f"policy_{policy_number}.pdf")
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Policy {policy_number}",
    )
    story = []
    name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()

    story.append(Paragraph(INSURER, styles["TitleBig"]))
    story.append(Paragraph(product.get("label", policy["policy_type"]) + " — Policy Document", styles["Center"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Policy schedule", styles["H"]))
    story.append(_kv_table([
        ("Policy number", policy_number),
        ("Policyholder", name or "-"),
        ("Email", customer.get("email", "-")),
        ("Policy type", product.get("label", policy["policy_type"])),
        (product.get("coverage_label", "Coverage amount"), _money(policy.get("coverage_limit"))),
        ("Policy term", f"{policy.get('term_years', '-')} year(s)"),
        ("Start date", policy.get("effective_date", "-")),
        ("End date", policy.get("expiry_date", "-")),
        ("Deductible", _money(policy.get("deductible"))),
        ("Premium", f"{_money(policy.get('instalment_amount'))} "
                    f"{plan.get('payment_frequency_label', '')} "
                    f"(total {_money(plan.get('total_premium'))} over the term)"),
    ], styles))

    benefits = plan.get("benefits", {})
    if benefits.get("summary"):
        story.append(Paragraph("What this policy gives you", styles["H"]))
        for line in benefits["summary"]:
            story.append(Paragraph("• " + line, styles["Body"]))

    # Life savings projection.
    proj = benefits.get("projection")
    if proj:
        story.append(Paragraph("Illustrative benefit projection", styles["H"]))
        story.append(_grid(
            ["Policy year", "Premiums paid", "Benefit if insured dies"],
            [[p["year"], _money(p["premiums_paid"]), _money(p["death_benefit"])] for p in proj],
            styles, (45 * mm, 60 * mm, 60 * mm),
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<b>Maturity benefit at the end of year {policy.get('term_years')}: "
            f"{_money(benefits.get('maturity_benefit'))}</b>", styles["Body"]))

    schedule = plan.get("schedule")
    if schedule:
        story.append(Paragraph("Payment schedule", styles["H"]))
        story.append(_grid(
            ["Policy year", "From", "To", "Instalments", "Amount for year", "Paid to date"],
            [[r["policy_year"], r["from"], r["to"], r["instalments"],
              _money(r["amount_for_year"]), _money(r["total_paid_to_date"])] for r in schedule],
            styles, (22 * mm, 28 * mm, 28 * mm, 22 * mm, 33 * mm, 33 * mm),
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Important notes", styles["H"]))
    story.append(Paragraph(
        plan.get("note", uw.ILLUSTRATIVE_NOTE) +
        " This document is generated for a demonstration system and is not a real "
        "contract of insurance.", styles["Small"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Issued on {datetime.utcnow().strftime('%Y-%m-%d')} by {INSURER}.", styles["Small"]))

    doc.build(story)
    return out_path
