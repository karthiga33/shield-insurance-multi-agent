"""Populate the database with synthetic customers, policies, and coverage rules.

Run once before using the agent:  python seed_data.py
This drops and recreates the reference tables so re-runs are deterministic,
but it does NOT delete claims that were already filed.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from config import DB_PATH
from database import connect, init_db

random.seed(42)

FIRST_NAMES = ["Kevin", "Sarah", "Jane", "Aria", "Isabella", "Liam", "Noah", "Mia", "Ethan", "Olivia"]
LAST_NAMES = ["Lopez", "Martin", "Smith", "Rivera", "Wright", "Nguyen", "Patel", "Khan", "Brown", "Davis"]
STATES = ["TX", "NY", "OH", "PA", "CA", "FL", "WA", "IL"]
POLICY_TYPES = ["auto", "home", "health", "life"]

# incident_type -> covered for which policy types
COVERAGE = {
    "auto": {
        "collision": True,
        "theft": True,
        "vandalism": True,
        "fire": True,
        "flood": False,          # excluded for standard auto
        "wear_and_tear": False,  # excluded (maintenance)
    },
    "home": {
        "fire": True,
        "theft": True,
        "water_damage": True,
        "storm": True,
        "flood": False,          # typically needs separate flood policy
        "wear_and_tear": False,
    },
    "health": {
        "hospitalization": True,
        "surgery": True,
        "emergency": True,
        "cosmetic": False,       # excluded
    },
    "life": {
        "death": True,
        "terminal_illness": True,
    },
}

N_CUSTOMERS = 50
N_POLICIES = 80


def seed() -> None:
    init_db()
    conn = connect()
    cur = conn.cursor()

    # Reset reference tables only.
    cur.executescript(
        """
        DELETE FROM customers;
        DELETE FROM policies;
        DELETE FROM coverage_rules;
        """
    )

    # Customers
    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        cid = f"CUST{str(i).zfill(5)}"
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        customers.append(
            (cid, fn, ln, f"user{i}@example.com", f"555-{random.randint(100,999)}-{random.randint(1000,9999)}", random.choice(STATES))
        )
    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers
    )

    # Policies
    today = date.today()
    policies = []
    for i in range(1, N_POLICIES + 1):
        pnum = f"POL{str(i).zfill(6)}"
        cid = f"CUST{str(random.randint(1, N_CUSTOMERS)).zfill(5)}"
        ptype = random.choice(POLICY_TYPES)
        status = random.choices(["active", "lapsed", "cancelled"], weights=[85, 10, 5])[0]
        premium = round(random.uniform(400, 2500), 2)
        coverage_limit = round(random.uniform(10000, 100000), 2)
        deductible = round(random.choice([250, 500, 1000, 1500]), 2)
        effective = today - timedelta(days=random.randint(30, 700))
        expiry = effective + timedelta(days=365)
        policies.append(
            (pnum, cid, ptype, status, premium, coverage_limit, deductible, effective.isoformat(), expiry.isoformat())
        )
    # Name the columns: the table has extra plan columns that sample policies leave empty.
    cur.executemany(
        "INSERT INTO policies (policy_number, customer_id, policy_type, status, premium_amount, "
        "coverage_limit, deductible, effective_date, expiry_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        policies,
    )

    # Coverage rules
    rules = []
    for ptype, incidents in COVERAGE.items():
        for incident, covered in incidents.items():
            rules.append((ptype, incident, 1 if covered else 0))
    cur.executemany(
        "INSERT INTO coverage_rules VALUES (?, ?, ?)", rules
    )

    conn.commit()

    # Print a couple of ready-to-use active policies for testing.
    sample = cur.execute(
        "SELECT p.policy_number, c.email, p.policy_type, p.coverage_limit, p.deductible "
        "FROM policies p JOIN customers c ON c.customer_id = p.customer_id "
        "WHERE p.status = 'active' ORDER BY p.policy_type LIMIT 12"
    ).fetchall()
    conn.close()

    print(f"Seeded {len(customers)} customers, {len(policies)} policies, {len(rules)} coverage rules.")
    print(f"Database: {DB_PATH}")
    print("\nSample active policies you can test with:")
    for r in sample:
        print(f"  {r['policy_number']}  email={r['email']:20s}  type={r['policy_type']:7s}  limit=${r['coverage_limit']:,.0f}  deductible=${r['deductible']:,.0f}")


if __name__ == "__main__":
    seed()
