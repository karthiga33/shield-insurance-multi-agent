# Standard Health Insurance Policy

**Policy Type:** health
**Document version:** 1.0

## 1. Scope of Coverage

This policy reimburses medically necessary expenses incurred by the insured for
the following, subject to the deductible and the coverage limit stated on the
policy schedule.

### 1.1 Covered Incidents

| Incident type | Covered | Required supporting document |
|---|---|---|
| `hospitalization` | Yes | Hospital discharge summary; itemized hospital bill |
| `surgery` | Yes | Surgeon's report; itemized hospital bill |
| `emergency` | Yes | Emergency-room record; itemized bill |

### 1.2 Excluded Incidents

| Incident type | Covered | Reason |
|---|---|---|
| `cosmetic` | No | Elective cosmetic procedures are not medically necessary and are excluded |

## 2. Conditions

1. The policy must be **active** on the date of treatment.
2. The treatment date must fall within the policy effective and expiry dates.
3. The claimed amount is payable **after** deducting the policy deductible and is
   capped at the coverage limit.
4. All claims **must** include an itemized hospital bill.

## 3. Claim Payout Formula

```
payable = min( max(claimed_amount - deductible, 0), coverage_limit )
```
