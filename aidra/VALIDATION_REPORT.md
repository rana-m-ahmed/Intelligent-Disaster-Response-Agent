# AIDRA Validation Report

## CCP Output Checklist

| Output | Where to find | Verified? |
| --- | --- | --- |
| Prioritized rescue order with justification | logs/decision_log.json (event_type: VICTIM_ORDER) | [x] |
| Selected route for each rescue trip with trade-off | logs/decision_log.json (event_type: ROUTE_SELECTION) | [x] |
| Resource allocation plan | logs/decision_log.json (event_type: ASSIGNMENT, assignment_plan) | [x] |
| Survival/risk estimate at rescue time | logs/decision_log.json (event_type: RESCUE_COMPLETE, updated_survival_prob) | [x] |
| Decision log for each replanning event | logs/decision_log.json (event_type: REPLAN) | [x] |
| Comparative performance report | results/kpi_table.csv and charts in results/ | [x] |

## Scenario Validation

- Scenario A: Completed, victims rescued = 5, priority order logged, routes include alpha trade-off justification.
- Scenario B: Road block triggers replanning with old/new costs and trigger_reason=road_block.
- Scenario C: Two road blocks and new victim handled, CSP reallocation logged, no crashes.

## Tests Summary

- Unit tests: 11 passed (pytest).

## Known Limitations

- Risk exposure is computed from ML risk predictions blended with area risk; it is indicative rather than calibrated to real-world risk.
- CSP backtrack counts are heuristic-based and intended for relative comparison, not absolute optimality proofs.

## Decision Log Snippet (Trade-off Justification)

```
{
  "event_type": "ROUTE_SELECTION",
  "victim_id": "V2",
  "alpha_used": 0.0,
  "justification_text": "Critical victim V2: alpha=0 for fastest rescue."
}
```

## Confirmation

- Scenarios A, B, and C executed without errors and generated outputs in results/ and logs/.
