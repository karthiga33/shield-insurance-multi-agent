# Standard Life Insurance Policy

**Policy Type:** life
**Document version:** 1.0

## 1. Scope of Coverage

This policy pays the sum assured to the nominated beneficiary on the following
events, subject to the coverage limit (sum assured) stated on the policy
schedule. Life policies do not apply a deductible.

### 1.1 Covered Incidents

| Incident type | Covered | Required supporting document |
|---|---|---|
| `death` | Yes | Death certificate; beneficiary identity proof |
| `terminal_illness` | Yes | Medical certificate confirming terminal diagnosis |

### 1.2 Excluded Incidents

- Death by suicide within the first two years of the policy is excluded.
- Death occurring while committing a crime is excluded.

## 2. Conditions

1. The policy must be **active** on the date of the event.
2. The event date must fall within the policy effective and expiry dates.
3. The payable amount equals the coverage limit (sum assured). No deductible
   applies.
4. A `death` claim **must** include a death certificate.

## 3. Claim Payout Formula

```
payable = coverage_limit   (sum assured; no deductible)
```
