from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_PATH = os.path.join(LOG_DIR, "decision_log.json")


class DecisionLogger:
    """Append decision events to a JSON log."""

    def __init__(self, reset: bool = False) -> None:
        """Create the log file, optionally resetting it."""
        os.makedirs(LOG_DIR, exist_ok=True)
        if reset or not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_event(self, payload: Dict[str, Any]) -> None:
        """Append a single event payload with a UTC timestamp."""
        payload = dict(payload)
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        payload = _apply_defaults(payload)
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            entries: List[Dict[str, Any]] = json.load(f)
        entries.append(payload)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)


def _apply_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "event_type": None,
        "victim_id": None,
        "chosen_path": None,
        "alpha_used": None,
        "time_cost": None,
        "risk_cost": None,
        "justification_text": None,
        "old_route_cost": None,
        "new_route_cost": None,
        "trigger_reason": None,
        "victim_priority_list": None,
        "updated_survival_prob": None,
        "assignment_plan": None,
        "scenario": None,
        "frontier_sizes": None,
    }
    for key, value in defaults.items():
        payload.setdefault(key, value)
    return payload
