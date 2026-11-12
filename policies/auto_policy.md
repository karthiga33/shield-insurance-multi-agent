# Standard Auto Insurance Policy

**Policy Type:** auto
**Document version:** 1.0

## 1. Scope of Coverage

This policy covers direct accidental physical loss or damage to the insured
vehicle arising from the following perils, subject to the deductible and the
coverage limit stated on the policy schedule.

### 1.1 Covered Incidents

| Incident type | Covered | Required supporting document |
|---|---|---|
| `collision` | Yes | Photographs of the damaged vehicle; repair estimate or invoice |
| `theft` | Yes | Police complaint / FIR copy; proof of ownership |
| `vandalism` | Yes | Photographs of the damage; police complaint |
| `fire` | Yes | Photographs of the damage; fire-brigade or police report |

### 1.2 Excluded Incidents

| Incident type | Covered | Reason |
|---|---|---|
| `flood` | No | Flood and water-ingress damage requires a separate add-on cover |
| `wear_and_tear` | No | Mechanical breakdown, ageing and routine maintenance are excluded |

## 2. Conditions

1. The policy must be **active** on the date of the incident.
2. The incident date must fall within the policy effective and expiry dates.
3. The claimed amount is payable **after** deducting the policy deductible and is
   capped at the coverage limit.
4. Claims for theft or fire **must** include the police / authority report named
   above, or they cannot be processed.

## 3. Claim Payout Formula

```
payable = min( max(claimed_amount - deductible, 0), coverage_limit )
```
