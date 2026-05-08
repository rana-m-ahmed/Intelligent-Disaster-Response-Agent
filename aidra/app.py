from __future__ import annotations

import os
import sys
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from environment import CellType, Environment, Victim
from csp import priority_score, solve_csp, validate_assignment
from fuzzy import FuzzyRisk
from logger import DecisionLogger
from ml_model import MLModel
from search import compute_path_cost, manhattan, search

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "aidra-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

state_lock = Lock()

class GlobalState:
    def __init__(self) -> None:
        self.env: Optional[Environment] = None
        self.ml = MLModel()
        self.fuzzy = FuzzyRisk()
        self.logger = DecisionLogger(reset=True)
        self.algorithm = "astar"
        self.alpha = 1.0
        self.current_algorithm = "astar"
        self.current_alpha = 1.0
        self.current_algorithm_params: Dict[str, Any] = {}
        self.ambulance_routes: Dict[str, List[Tuple[int, int]]] = {}
        self.ambulance_progress: Dict[str, int] = {}
        self.rescued_victims: set = set()
        self.simulation_running = True
        self.step_count = 0
        self.simulation_step = 0
        self.rescue_step_durations: List[int] = []
        self.route_risk_scores: List[float] = []
        self.risk_steps = 0
        self.active_assignments: Dict[str, str] = {}
        self.route_costs: Dict[str, float] = {}
        self.assignment_started_at: Dict[str, int] = {}
        self.rescue_queues: Dict[str, List[str]] = {}
        self.ambulance_trip_victims: Dict[str, List[str]] = {}
        self.ambulance_trip_waypoints: Dict[str, List[Tuple[int, int]]] = {}
        self.ambulance_trip_stage_index: Dict[str, int] = {}
        self.ambulance_trip_dropoff: Dict[str, Tuple[int, int]] = {}
        self.ambulance_loads: Dict[str, int] = {}
        self.ambulance_capacity = 2
        self.detour_threshold = 1.5
        self.ml_report_emitted = False
        self.latest_csp_backtracks = 0
        self.committed_kit_victims: set = set()

    def log_event(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("step", self.step_count)
        self.logger.log_event(payload)

    def _victim_features(self, victim: Victim, time_since: float) -> List[float]:
        severity_map = {"critical": 2, "moderate": 1, "minor": 0}
        severity = severity_map.get(victim.severity, 1)
        distance = 0
        if self.env:
            distance = min(
                manhattan(ambulance.pos, victim.pos) for ambulance in self.env.ambulances
            )
        area_risk = 0.0
        if self.env and self.env.grid[victim.pos[0]][victim.pos[1]] == CellType.RISK:
            area_risk = 1.0
        return [float(severity), float(distance), float(area_risk), float(time_since)]

    def refresh_victim_survival_probabilities(self) -> None:
        if not self.env:
            return
        for victim in self.env.victims:
            if victim.victim_id in self.rescued_victims:
                continue
            victim.survival_prob = self.ml.predict_survival(
                self._victim_features(victim, float(self.step_count))
            )

    def _active_victims(self) -> List[Victim]:
        if not self.env:
            return []
        return [victim for victim in self.env.victims if victim.victim_id not in self.rescued_victims]

    def _full_rescue_priority(self, victim: Victim) -> float:
        severity_boost = {"critical": 0.24, "moderate": 0.08, "minor": 0.0}.get(victim.severity, 0.0)
        distance_bonus = 0.0
        if self.env and self.env.ambulances:
            nearest = min(manhattan(ambulance.pos, victim.pos) for ambulance in self.env.ambulances)
            distance_bonus = 1.0 / (1.0 + float(nearest))
        base_priority = priority_score(victim, self.ml, self.env, float(self.step_count))
        return base_priority + severity_boost + 0.08 * distance_bonus

    def _cleanup_rescue_queue(self, ambulance_id: str) -> None:
        queue = self.rescue_queues.get(ambulance_id, [])
        while queue and queue[0] in self.rescued_victims:
            queue.pop(0)
        self.rescue_queues[ambulance_id] = queue

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"critical": 3, "moderate": 2, "minor": 1}.get(severity, 1)

    def _nearest_med_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        if not self.env or not self.env.med_centers:
            return pos
        return min(self.env.med_centers, key=lambda center: manhattan(pos, center))

    @staticmethod
    def _stitch_paths(paths: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        stitched: List[Tuple[int, int]] = []
        for path in paths:
            if not path:
                continue
            if not stitched:
                stitched.extend(path)
            else:
                stitched.extend(path[1:])
        return stitched

    def _build_waypoint_route(
        self,
        start: Tuple[int, int],
        waypoints: List[Tuple[int, int]],
        algorithm: str,
        alpha: float,
    ) -> Dict[str, Any]:
        if not self.env or not waypoints:
            return {"path": [], "cost": 0.0, "risk": 0.0, "segments": []}

        current_pos = start
        segments: List[List[Tuple[int, int]]] = []
        total_cost = 0.0
        total_risk = 0.0

        for waypoint in waypoints:
            result = search(self.env, current_pos, waypoint, algorithm, self.ml, self.fuzzy, alpha, True, self.current_algorithm_params)
            segments.append(result.path)
            total_cost += result.total_cost
            total_risk += result.risk_score
            current_pos = waypoint

        return {
            "path": self._stitch_paths(segments),
            "cost": total_cost,
            "risk": total_risk,
            "segments": segments,
        }

    def _victim_queue_key(self, victim: Victim) -> Tuple[int, int, int, str]:
        nearest_distance = 0
        if self.env and self.env.ambulances:
            nearest_distance = min(
                manhattan(ambulance.pos, victim.pos) for ambulance in self.env.ambulances
            )
        return (-self._severity_rank(victim.severity), nearest_distance, victim.kits_needed, victim.victim_id)

    @staticmethod
    def _kits_for_severity(severity: str) -> int:
        return {"critical": 2, "moderate": 1, "minor": 0}.get(severity, 1)

    def _kits_needed_for_victim(self, victim: Victim) -> int:
        return self._kits_for_severity(victim.severity)

    def _log_resource_exhaustion(self, trigger_reason: str) -> None:
        if not self.env:
            return
        self.log_event(
            {
                "event_type": "CONSTRAINT_VIOLATION",
                "trigger_reason": trigger_reason,
                "module": "csp",
                "outcome": "resource_exhaustion",
                "justification_text": "Resource exhaustion constraint violation: medical kits are depleted (0).",
                "assignment_plan": dict(self.active_assignments),
                "victim_priority_list": self.get_priority_list(),
            }
        )

    def _log_insufficient_kits(self, victim_id: str, required: int, available: int, trigger_reason: str) -> None:
        self.log_event(
            {
                "event_type": "CONSTRAINT_VIOLATION",
                "victim_id": victim_id,
                "trigger_reason": trigger_reason,
                "module": "csp",
                "outcome": "insufficient_kits",
                "justification_text": (
                    f"Constraint violation: cannot assign {victim_id}; required kits={required}, "
                    f"available kits={available}."
                ),
                "assignment_plan": dict(self.active_assignments),
                "victim_priority_list": self.get_priority_list(),
            }
        )

    def _deduct_kits_for_victims(
        self,
        victim_ids: List[str],
        trigger_reason: str,
    ) -> Tuple[bool, List[str]]:
        if not self.env:
            return False, []
        victim_map = {victim.victim_id: victim for victim in self.env.victims}
        newly_committed = [victim_id for victim_id in victim_ids if victim_id not in self.committed_kit_victims]
        required = 0
        for victim_id in newly_committed:
            victim = victim_map.get(victim_id)
            if victim:
                required += self._kits_needed_for_victim(victim)

        if required > self.env.medical_kits:
            primary = newly_committed[0] if newly_committed else ""
            self._log_insufficient_kits(primary, required, self.env.medical_kits, trigger_reason)
            if self.env.medical_kits == 0:
                self._log_resource_exhaustion(trigger_reason)
            return False, []

        self.env.medical_kits -= required
        for victim_id in newly_committed:
            self.committed_kit_victims.add(victim_id)

        if self.env.medical_kits == 0:
            self._log_resource_exhaustion(trigger_reason)
        return True, newly_committed

    def solve_csp_with_tracking(
        self,
        use_mrv: bool = True,
        use_forward_checking: bool = True,
    ):
        if not self.env:
            return None
        if self.env.medical_kits == 0:
            self._log_resource_exhaustion("csp_cycle")
        result = solve_csp(self.env, self.ml, use_mrv=use_mrv, use_forward_checking=use_forward_checking)
        self.latest_csp_backtracks = result.backtracks
        return result

    def _reset_trip_state(self, ambulance_id: str) -> None:
        self.ambulance_trip_victims.pop(ambulance_id, None)
        self.ambulance_trip_waypoints.pop(ambulance_id, None)
        self.ambulance_trip_stage_index.pop(ambulance_id, None)
        self.ambulance_trip_dropoff.pop(ambulance_id, None)
        self.ambulance_loads[ambulance_id] = 0

    def _current_trip_target(self, ambulance_id: str, route: List[Tuple[int, int]], progress: int) -> Optional[Tuple[int, int]]:
        waypoints = self.ambulance_trip_waypoints.get(ambulance_id, [])
        stage_index = self.ambulance_trip_stage_index.get(ambulance_id, 0)
        if not route or not waypoints:
            return None
        if stage_index < len(waypoints):
            return waypoints[stage_index]
        return waypoints[-1]

    def _current_trip_target_victim(self, ambulance_id: str) -> Optional[Victim]:
        if not self.env:
            return None
        waypoints = self.ambulance_trip_waypoints.get(ambulance_id, [])
        stage_index = self.ambulance_trip_stage_index.get(ambulance_id, 0)
        if not waypoints or stage_index >= len(waypoints) - 1:
            return None
        target_pos = waypoints[stage_index]
        return next(
            (
                victim
                for victim in self.env.victims
                if victim.pos == target_pos and victim.victim_id not in self.rescued_victims
            ),
            None,
        )

    def _advance_trip_stage(self, ambulance_id: str, current_pos: Tuple[int, int]) -> None:
        waypoints = self.ambulance_trip_waypoints.get(ambulance_id, [])
        if not waypoints:
            return

        stage_index = self.ambulance_trip_stage_index.get(ambulance_id, 0)
        while stage_index < len(waypoints) and waypoints[stage_index] == current_pos:
            stage_index += 1
        self.ambulance_trip_stage_index[ambulance_id] = stage_index

    def _build_trip_route(
        self,
        ambulance_id: str,
        primary_victim: Victim,
        algorithm: str,
        alpha: float,
        queue: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.env:
            return {"path": [], "victim_ids": [], "dropoff": None, "load": 0, "cost": 0.0, "risk": 0.0, "segments": []}

        queue = list(queue or self.rescue_queues.get(ambulance_id, []))
        start = self.get_current_pos(ambulance_id)
        trip_victims = [primary_victim]
        consumed_ids = [primary_victim.victim_id]

        first_leg = search(self.env, start, primary_victim.pos, algorithm, self.ml, self.fuzzy, alpha, True, self.current_algorithm_params)
        segments: List[List[Tuple[int, int]]] = [first_leg.path]
        total_cost = first_leg.total_cost
        total_risk = first_leg.risk_score

        current_pos = primary_victim.pos
        dropoff = self._nearest_med_center(current_pos)

        if len(queue) > 1:
            next_victim_id = queue[1]
            next_victim = next(
                (victim for victim in self.env.victims if victim.victim_id == next_victim_id),
                None,
            )
            if next_victim and next_victim.victim_id not in self.rescued_victims:
                direct_dropoff = search(self.env, current_pos, dropoff, algorithm, self.ml, self.fuzzy, alpha, True, self.current_algorithm_params)
                detour_leg = search(self.env, current_pos, next_victim.pos, algorithm, self.ml, self.fuzzy, alpha, True, self.current_algorithm_params)
                if detour_leg.total_cost <= self.detour_threshold * max(direct_dropoff.total_cost, 1e-9):
                    trip_victims.append(next_victim)
                    consumed_ids.append(next_victim.victim_id)
                    segments.append(detour_leg.path)
                    total_cost += detour_leg.total_cost
                    total_risk += detour_leg.risk_score
                    current_pos = next_victim.pos
                    dropoff = self._nearest_med_center(current_pos)

        drop_leg = search(self.env, current_pos, dropoff, algorithm, self.ml, self.fuzzy, alpha, True, self.current_algorithm_params)
        segments.append(drop_leg.path)
        total_cost += drop_leg.total_cost
        total_risk += drop_leg.risk_score

        self.ambulance_trip_victims[ambulance_id] = consumed_ids
        self.ambulance_trip_waypoints[ambulance_id] = [victim.pos for victim in trip_victims] + [dropoff]
        self.ambulance_trip_stage_index[ambulance_id] = 0
        self.ambulance_trip_dropoff[ambulance_id] = dropoff
        self.ambulance_loads[ambulance_id] = len(consumed_ids)

        return {
            "path": self._stitch_paths(segments),
            "victim_ids": consumed_ids,
            "dropoff": dropoff,
            "load": len(consumed_ids),
            "cost": total_cost,
            "risk": total_risk,
            "segments": segments,
        }

    def _dispatch_next_queue_target(self, ambulance_id: str) -> Optional[Dict[str, Any]]:
        if not self.env:
            return None

        self._cleanup_rescue_queue(ambulance_id)
        queue = self.rescue_queues.get(ambulance_id, [])
        if not queue:
            self.ambulance_routes[ambulance_id] = []
            self.ambulance_progress[ambulance_id] = 0
            self.active_assignments.pop(ambulance_id, None)
            self.assignment_started_at.pop(ambulance_id, None)
            self.route_costs.pop(ambulance_id, None)
            self._reset_trip_state(ambulance_id)
            return None

        victim_map = {victim.victim_id: victim for victim in self.env.victims}
        victim_id = queue[0]
        victim = victim_map.get(victim_id)
        if not victim or victim.victim_id in self.rescued_victims:
            queue.pop(0)
            self.rescue_queues[ambulance_id] = queue
            return self._dispatch_next_queue_target(ambulance_id)

        trip = self._build_trip_route(
            ambulance_id,
            victim,
            self.current_algorithm,
            self.current_alpha,
            queue=queue,
        )
        accepted, _committed = self._deduct_kits_for_victims(
            list(trip.get("victim_ids", [])),
            trigger_reason="dispatch_assignment",
        )
        if not accepted:
            queue.pop(0)
            self.rescue_queues[ambulance_id] = queue
            return self._dispatch_next_queue_target(ambulance_id)

        consumed_ids = set(trip.get("victim_ids", []))
        self.rescue_queues[ambulance_id] = [victim_id for victim_id in queue if victim_id not in consumed_ids]

        self.ambulance_routes[ambulance_id] = trip["path"]
        self.ambulance_progress[ambulance_id] = 0
        self.active_assignments[ambulance_id] = victim.victim_id
        self.route_costs[ambulance_id] = trip["cost"]
        self.assignment_started_at[ambulance_id] = self.step_count
        self.route_risk_scores.append(trip["risk"])

        justification = (
            f"Auto-dispatched {ambulance_id} to {victim.victim_id} from the full rescue queue "
            f"with {self.current_algorithm.upper()} (alpha={self.current_alpha:.1f})."
        )
        self.log_event(
            {
                "event_type": "ROUTE_SELECTION",
                "victim_id": victim.victim_id,
                "chosen_path": trip["path"],
                "alpha_used": self.current_alpha,
                "time_cost": trip["cost"],
                "risk_cost": trip["risk"],
                "algorithm": self.current_algorithm,
                "justification_text": justification,
                "frontier_sizes": [],
                "optimality_ratio": 1.0,
                "fuzzy_risk_along_path": _path_fuzzy_risk(self.env, trip["path"]),
                "trigger_reason": "full_rescue_queue",
                "victim_priority_list": self.get_priority_list(),
                "assignment_plan": dict(self.active_assignments),
            }
        )
        emit(
            "route_planned",
            {
                "ambulance_id": ambulance_id,
                "victim_id": victim.victim_id,
                "algorithm": self.current_algorithm,
                "alpha": self.current_alpha,
                "path": trip["path"],
                "cost": trip["cost"],
                "risk_score": trip["risk"],
                "justification": justification,
                "is_replan": False,
                "optimality_ratio": 1.0,
                "fuzzy_risk_along_path": _path_fuzzy_risk(self.env, trip["path"]),
                "frontier_sizes": [],
                "load": trip.get("load", 1),
                "capacity": self.ambulance_capacity,
                "dropoff": trip.get("dropoff"),
            },
            broadcast=True,
        )
        return {
            "victim_id": victim.victim_id,
            "route_cost": round(trip["cost"], 3),
            "risk_cost": round(trip["risk"], 3),
            "priority": round(self._full_rescue_priority(victim), 4),
            "optimality_ratio": 1.0,
        }

    def build_full_rescue_plan(self, algorithm: str, alpha: float) -> Dict[str, Any]:
        if not self.env:
            return {"assignment_plan": {}, "route_summary": {}, "victim_priority_list": []}

        active_victims = sorted(self._active_victims(), key=self._victim_queue_key)
        queues: Dict[str, List[str]] = {amb.agent_id: [] for amb in self.env.ambulances}
        cursor_positions: Dict[str, Tuple[int, int]] = {amb.agent_id: amb.pos for amb in self.env.ambulances}
        route_summary: Dict[str, Dict[str, Any]] = {}

        for victim in active_victims:
            best_choice: Optional[Tuple[float, str, Any]] = None
            for amb in self.env.ambulances:
                amb_id = amb.agent_id
                result = search(self.env, cursor_positions[amb_id], victim.pos, algorithm, self.ml, self.fuzzy, alpha, True, self.current_algorithm_params)
                score = float(self._severity_rank(victim.severity)) * 100.0 - result.total_cost - manhattan(cursor_positions[amb_id], victim.pos)
                choice = (score, amb_id, result)
                if best_choice is None or choice[0] > best_choice[0]:
                    best_choice = choice

            assert best_choice is not None
            _, chosen_ambulance, chosen_result = best_choice
            queues[chosen_ambulance].append(victim.victim_id)
            cursor_positions[chosen_ambulance] = victim.pos
            if chosen_ambulance not in route_summary:
                route_summary[chosen_ambulance] = {
                    "victim_id": victim.victim_id,
                    "route_cost": round(chosen_result.total_cost, 3),
                    "risk_cost": round(chosen_result.risk_score, 3),
                    "priority": round(float(self._severity_rank(victim.severity)) + 1.0 / (1.0 + manhattan(cursor_positions[chosen_ambulance], victim.pos)), 4),
                    "optimality_ratio": round(chosen_result.optimality_ratio, 4),
                }

        self.rescue_queues = {ambulance_id: list(queue) for ambulance_id, queue in queues.items()}
        return {
            "assignment_plan": {ambulance_id: list(queue) for ambulance_id, queue in queues.items() if queue},
            "route_summary": route_summary,
            "victim_priority_list": [
                {
                    "victim_id": victim.victim_id,
                    "priority": round(float(self._severity_rank(victim.severity)) + 1.0 / (1.0 + min(manhattan(amb.pos, victim.pos) for amb in self.env.ambulances)), 4),
                }
                for victim in active_victims
            ],
        }

    def reallocate_full_rescue(self, algorithm: str, alpha: float) -> Dict[str, Any]:
        if not self.env:
            return {"assignment_plan": {}, "route_summary": {}, "victim_priority_list": []}
        full_plan = self.build_full_rescue_plan(algorithm, alpha)
        plan = full_plan["assignment_plan"]
        updated_queues: Dict[str, List[str]] = {}
        for amb in self.env.ambulances:
            amb_id = amb.agent_id
            queue = list(plan.get(amb_id, []))
            active = self.active_assignments.get(amb_id)
            if active and active not in self.rescued_victims:
                if active in queue:
                    queue = [active] + [victim_id for victim_id in queue if victim_id != active]
                else:
                    queue = [active] + queue
            updated_queues[amb_id] = queue
        self.rescue_queues = updated_queues
        return {
            "assignment_plan": {ambulance_id: list(queue) for ambulance_id, queue in updated_queues.items() if queue},
            "route_summary": full_plan["route_summary"],
            "victim_priority_list": full_plan["victim_priority_list"],
        }

    def get_ml_report_payload(self) -> Dict[str, Any]:
        return {
            "event_type": "ML_REPORT",
            "ml_report": self.ml.get_metrics_report(),
            "justification_text": "Model comparison report emitted at startup.",
            "trigger_reason": "startup_model_report",
        }

    def get_csp_assignment_snapshot(self) -> Dict[str, str]:
        if not self.env:
            return {}
        snapshot: Dict[str, str] = {}
        for amb in self.env.ambulances:
            route = self.ambulance_routes.get(amb.agent_id, [])
            if not route:
                continue
            target = self.active_assignments.get(amb.agent_id)
            if not target:
                target = next(
                    (
                        victim.victim_id
                        for victim in self.env.victims
                        if victim.pos == route[-1] and victim.victim_id not in self.rescued_victims
                    ),
                    "",
                )
            if target:
                snapshot[amb.agent_id] = target
        return snapshot

    def init_scenario(self, scenario: str) -> None:
        with state_lock:
            self.env = Environment(scenario)
            self.ambulance_routes = {}
            self.ambulance_progress = {}
            self.rescued_victims = set()
            self.simulation_running = True
            self.step_count = 0
            self.simulation_step = 0
            self.rescue_step_durations = []
            self.route_risk_scores = []
            self.risk_steps = 0
            self.active_assignments = {}
            self.route_costs = {}
            self.assignment_started_at = {}
            self.rescue_queues = {}
            self.ambulance_trip_victims = {}
            self.ambulance_trip_waypoints = {}
            self.ambulance_trip_stage_index = {}
            self.ambulance_trip_dropoff = {}
            self.ambulance_loads = {}
            self.ml_report_emitted = False
            self.latest_csp_backtracks = 0
            self.committed_kit_victims = set()
            for amb in self.env.ambulances:
                self.ambulance_routes[amb.agent_id] = []
                self.ambulance_progress[amb.agent_id] = 0
                self.ambulance_loads[amb.agent_id] = 0
            self.refresh_victim_survival_probabilities()

    def get_grid_for_json(self) -> List[List[str]]:
        if not self.env:
            return []
        return [[cell.value for cell in row] for row in self.env.grid]

    def get_victims_for_json(self) -> List[Dict[str, Any]]:
        if not self.env:
            return []
        return [
            {
                "id": v.victim_id,
                "pos": v.pos,
                "severity": v.severity,
                "survival_prob": v.survival_prob,
                "rescued": v.victim_id in self.rescued_victims,
            }
            for v in self.env.victims
        ]

    def get_ambulances_for_json(self) -> List[Dict[str, Any]]:
        if not self.env:
            return []
        result = []
        for amb in self.env.ambulances:
            route = self.ambulance_routes.get(amb.agent_id, [])
            progress = self.ambulance_progress.get(amb.agent_id, 0)
            current_pos = amb.pos
            if route and progress < len(route):
                current_pos = route[progress]
            # Send only the active remaining route segment to the UI.
            display_route = route[progress:] if route and progress < len(route) else []
            result.append(
                {
                    "id": amb.agent_id,
                    "pos": current_pos,
                    "route": display_route,
                    "progress": 0,
                    "load": self.ambulance_loads.get(amb.agent_id, 0),
                    "capacity": self.ambulance_capacity,
                }
            )
        return result

    def get_state_snapshot(self) -> Dict[str, Any]:
        total_victims = len(self.env.victims) if self.env else 0
        avg_rescue_steps = (
            sum(self.rescue_step_durations) / len(self.rescue_step_durations)
            if self.rescue_step_durations
            else 0.0
        )
        avg_rescue_time_sec = avg_rescue_steps * 0.5
        risk_exposure = float(self.risk_steps)
        return {
            "grid": self.get_grid_for_json(),
            "victims": self.get_victims_for_json(),
            "ambulances": self.get_ambulances_for_json(),
            "med_centers": self.env.med_centers if self.env else [],
            "rescue_team": {
                "id": self.env.rescue_team.agent_id,
                "pos": self.env.rescue_team.pos,
            }
            if self.env
            else None,
            "rescued_victims": len(self.rescued_victims),
            "total_victims": total_victims,
            "avg_rescue_time": avg_rescue_time_sec,
            "risk_exposure": risk_exposure,
            "csp_assignment": self.get_csp_assignment_snapshot(),
            "medical_kits": self.env.medical_kits if self.env else 0,
            "medical_kits_capacity": 10,
            "csp_backtracks": self.latest_csp_backtracks,
            "ambulance_loads": self.ambulance_loads if self.env else {},
            "ambulance_capacity": self.ambulance_capacity,
            "algorithm": self.current_algorithm,
            "alpha": self.current_alpha,
        }

    def get_current_pos(self, ambulance_id: str) -> Tuple[int, int]:
        if not self.env:
            return (0, 0)
        route = self.ambulance_routes.get(ambulance_id, [])
        progress = self.ambulance_progress.get(ambulance_id, 0)
        if route and progress < len(route):
            return route[progress]
        ambulance = next((a for a in self.env.ambulances if a.agent_id == ambulance_id), None)
        return ambulance.pos if ambulance else (0, 0)

    def get_priority_list(self) -> List[Dict[str, Any]]:
        if not self.env:
            return []
        active_victims = self._active_victims()
        sorted_victims = sorted(
            active_victims,
            key=self._victim_queue_key,
        )
        return [
            {
                "victim_id": v.victim_id,
                "priority": round(float(self._severity_rank(v.severity)) + 1.0 / (1.0 + min(manhattan(amb.pos, v.pos) for amb in self.env.ambulances)), 4),
            }
            for v in sorted_victims
        ]

    def get_assignment_plan(self) -> Dict[str, str]:
        if not self.env:
            return {}
        csp_result = self.solve_csp_with_tracking()
        if csp_result is None:
            return {}
        vars_to_ambulance: Dict[str, str] = {}
        for idx, amb in enumerate(self.env.ambulances, start=1):
            vars_to_ambulance[f"ambulance_{idx}"] = amb.agent_id
        plan: Dict[str, str] = {}
        for var_name, victims in csp_result.assignment.items():
            amb_id = vars_to_ambulance.get(var_name)
            if not amb_id:
                continue
            for victim_id in victims:
                if victim_id not in self.rescued_victims:
                    plan[amb_id] = victim_id
                    break
        return plan


def _neighborhood_stats(env: Environment, pos: Tuple[int, int]) -> Tuple[float, float]:
    rows = range(max(0, pos[0] - 1), min(env.size, pos[0] + 2))
    cols = range(max(0, pos[1] - 1), min(env.size, pos[1] + 2))
    total = 0
    blocked = 0
    risky = 0
    for row in rows:
        for col in cols:
            total += 1
            cell = env.grid[row][col]
            if cell == CellType.BLOCKED:
                blocked += 1
            elif cell == CellType.RISK:
                risky += 1
    total = max(total, 1)
    return blocked / total, risky / total


def _area_risk_for_cell(env: Environment, pos: Tuple[int, int]) -> float:
    return 1.0 if env.grid[pos[0]][pos[1]] == CellType.RISK else 0.0


def _victim_features(env: Environment, victim: Victim, time_since: float) -> List[float]:
    severity_map = {"critical": 2, "moderate": 1, "minor": 0}
    severity = severity_map.get(victim.severity, 1)
    distance = min(manhattan(amb.pos, victim.pos) for amb in env.ambulances)
    area_risk = _area_risk_for_cell(env, victim.pos)
    return [float(severity), float(distance), float(area_risk), float(time_since)]


def _path_fuzzy_risk(env: Environment, path: List[Tuple[int, int]]) -> float:
    if len(path) <= 1:
        return 0.0
    risks = []
    for pos in path[1:]:
        block_prob, hazard_rate = _neighborhood_stats(env, pos)
        risks.append(global_state.fuzzy.compute_risk_weight(block_prob, hazard_rate))
    return sum(risks) / len(risks)


def _ml_report_payload() -> Dict[str, Any]:
    return {
        "event_type": "ML_REPORT",
        "ml_report": global_state.ml.get_metrics_report(),
        "justification_text": "Startup model comparison report.",
        "trigger_reason": "startup_model_report",
        "step": 0,
    }


def _assignment_summary_for_event() -> Dict[str, List[str]]:
    if not global_state.env:
        return {}
    csp_result = global_state.solve_csp_with_tracking()
    if csp_result is None:
        return {}
    return {resource: list(victims) for resource, victims in csp_result.assignment.items() if victims}


global_state = GlobalState()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@socketio.on("init_scenario")
def handle_init_scenario(data: Dict[str, str]) -> None:
    scenario = data.get("scenario", "A")
    global_state.init_scenario(scenario)

    global_state.log_event(
        {
            "event_type": "SCENARIO_START",
            "scenario": scenario,
            "justification_text": f"Starting scenario {scenario}.",
            "trigger_reason": "scenario_init",
            "victim_priority_list": global_state.get_priority_list(),
            "assignment_plan": global_state.get_assignment_plan(),
        }
    )

    if not global_state.ml_report_emitted:
        ml_report_payload = global_state.get_ml_report_payload()
        global_state.log_event(ml_report_payload)
        emit("ml_report", ml_report_payload, broadcast=True)
        global_state.ml_report_emitted = True

    emit("state_update", global_state.get_state_snapshot(), broadcast=True)


@socketio.on("get_state")
def handle_get_state() -> None:
    emit("state_update", global_state.get_state_snapshot())


@socketio.on("plan_route")
def handle_plan_route(data: Dict[str, Any]) -> None:
    if not global_state.env:
        return

    victim_id = data.get("victim_id")
    algorithm = data.get("algorithm", "astar")
    alpha = data.get("alpha", 1.0)
    algorithm_params = data.get("algorithm_params", {})

    global_state.current_algorithm = algorithm
    global_state.current_alpha = alpha
    global_state.current_algorithm_params = algorithm_params or {}

    victim_map = {v.victim_id: v for v in global_state.env.victims}
    victim = victim_map.get(victim_id)
    if not victim or victim.victim_id in global_state.rescued_victims:
        return

    assignment_plan = global_state.get_assignment_plan()
    available_ambulances = [
        amb.agent_id
        for amb in global_state.env.ambulances
        if not global_state.ambulance_routes.get(amb.agent_id)
    ]
    preferred_ambulance = next(
        (amb_id for amb_id, vid in assignment_plan.items() if vid == victim_id), None
    )
    if preferred_ambulance and preferred_ambulance in available_ambulances:
        assigned_ambulance = preferred_ambulance
    else:
        pool = available_ambulances or [amb.agent_id for amb in global_state.env.ambulances]
        assigned_ambulance = min(
            pool,
            key=lambda amb_id: abs(global_state.get_current_pos(amb_id)[0] - victim.pos[0])
            + abs(global_state.get_current_pos(amb_id)[1] - victim.pos[1]),
        )

    result = global_state._build_trip_route(assigned_ambulance, victim, algorithm, alpha, queue=[victim_id])
    accepted, _committed = global_state._deduct_kits_for_victims(
        list(result.get("victim_ids", [])),
        trigger_reason="manual_assignment",
    )
    if not accepted:
        emit("state_update", global_state.get_state_snapshot(), broadcast=True)
        return

    justify = _generate_justification(victim, algorithm, alpha, result["cost"])

    global_state.ambulance_routes[assigned_ambulance] = result["path"]
    global_state.ambulance_progress[assigned_ambulance] = 0
    global_state.active_assignments[assigned_ambulance] = victim_id
    global_state.route_costs[assigned_ambulance] = result["cost"]
    global_state.assignment_started_at[assigned_ambulance] = global_state.simulation_step
    global_state.route_risk_scores.append(result["risk"])

    global_state.log_event(
        {
            "event_type": "ROUTE_SELECTION",
            "victim_id": victim_id,
            "chosen_path": result["path"],
            "alpha_used": alpha,
            "time_cost": result["cost"],
            "risk_cost": result["risk"],
            "algorithm": algorithm,
            "justification_text": justify,
            "frontier_sizes": [],
            "optimality_ratio": 1.0,
            "fuzzy_risk_along_path": _path_fuzzy_risk(global_state.env, result["path"]),
            "trigger_reason": "user_plan",
            "victim_priority_list": global_state.get_priority_list(),
            "assignment_plan": dict(global_state.active_assignments),
        }
    )

    emit(
        "route_planned",
        {
            "ambulance_id": assigned_ambulance,
            "victim_id": victim_id,
            "algorithm": algorithm,
            "alpha": alpha,
            "path": result["path"],
            "cost": result["cost"],
            "risk_score": result["risk"],
            "justification": justify,
            "is_replan": False,
            "optimality_ratio": 1.0,
            "fuzzy_risk_along_path": _path_fuzzy_risk(global_state.env, result["path"]),
            "frontier_sizes": [],
            "load": result.get("load", 1),
            "capacity": global_state.ambulance_capacity,
            "dropoff": result.get("dropoff"),
        },
        broadcast=True,
    )
    emit("state_update", global_state.get_state_snapshot(), broadcast=True)


@socketio.on("plan_full_rescue")
def handle_plan_full_rescue(data: Dict[str, Any]) -> None:
    if not global_state.env:
        return

    algorithm = data.get("algorithm", global_state.current_algorithm)
    alpha = float(data.get("alpha", global_state.current_alpha))
    algorithm_params = data.get("algorithm_params", {})
    global_state.current_algorithm = algorithm
    global_state.current_alpha = alpha
    global_state.current_algorithm_params = algorithm_params or {}

    # Keep CSP solve for telemetry/contract visibility, but execute full rescue using
    # a complete queue so the run continues beyond the first assigned victims.
    csp_result = global_state.solve_csp_with_tracking(use_mrv=True, use_forward_checking=True)
    if csp_result is None:
        return
    validate_assignment(csp_result.assignment, global_state.env, allow_partial=True)
    full_plan = global_state.build_full_rescue_plan(algorithm, alpha)
    full_assignment_plan = full_plan["assignment_plan"]
    summary: Dict[str, Dict[str, Any]] = full_plan["route_summary"]

    global_state.ambulance_routes = {amb.agent_id: [] for amb in global_state.env.ambulances}
    global_state.ambulance_progress = {amb.agent_id: 0 for amb in global_state.env.ambulances}
    global_state.active_assignments = {}
    global_state.route_costs = {}
    global_state.assignment_started_at = {}
    global_state.route_risk_scores = []
    global_state.rescue_queues = {
        ambulance_id: list(queue)
        for ambulance_id, queue in full_assignment_plan.items()
    }
    for amb in global_state.env.ambulances:
        global_state.rescue_queues.setdefault(amb.agent_id, [])
        global_state._dispatch_next_queue_target(amb.agent_id)

    priority_list = full_plan["victim_priority_list"]
    global_state.log_event(
        {
            "event_type": "ASSIGNMENT",
            "assignment_plan": full_assignment_plan,
            "victim_priority_list": priority_list,
            "trigger_reason": "full_rescue_plan",
            "justification_text": "Full rescue plan computed from CSP with ML-informed priorities.",
        }
    )
    emit(
        "full_rescue_planned",
        {
            "assignment_plan": full_assignment_plan,
            "victim_priority_list": priority_list,
            "route_summary": summary,
            "trigger_reason": "full_rescue_plan",
        },
        broadcast=True,
    )
    emit("state_update", global_state.get_state_snapshot(), broadcast=True)


def _route_victim_id(route: List[Tuple[int, int]]) -> Optional[str]:
    if not global_state.env or not route:
        return None
    goal = route[-1]
    victim = next(
        (v for v in global_state.env.victims if v.pos == goal and v.victim_id not in global_state.rescued_victims),
        None,
    )
    return victim.victim_id if victim else None


def _auto_replan_for_block(coords: Tuple[int, int], trigger_reason: str) -> None:
    if not global_state.env:
        return
    victims_by_id = {v.victim_id: v for v in global_state.env.victims}
    affected = []
    for amb_id, route in global_state.ambulance_routes.items():
        if not route:
            continue
        progress = global_state.ambulance_progress.get(amb_id, 0)
        remaining = route[progress:]
        if coords in remaining:
            affected.append((amb_id, route, progress))

    if not affected:
        global_state.log_event(
            {
                "event_type": "REPLAN",
                "trigger_reason": trigger_reason,
                "old_route_cost": 0.0,
                "new_route_cost": 0.0,
                "victim_priority_list": global_state.get_priority_list(),
                "assignment_plan": dict(global_state.active_assignments),
                "justification_text": f"Road blocked at {coords}; no active ambulance route crossed the blocked cell.",
            }
        )
        return

    for amb_id, old_route, progress in affected:
        old_remaining = old_route[progress:]
        old_route_cost, _ = compute_path_cost(
            global_state.env,
            old_remaining,
            global_state.ml,
            global_state.fuzzy,
            global_state.current_alpha,
        )
        victim_id = global_state.active_assignments.get(amb_id) or _route_victim_id(old_route)
        victim = victims_by_id.get(victim_id) if victim_id else None
        target_victim = global_state._current_trip_target_victim(amb_id)
        target_pos = global_state._current_trip_target(amb_id, old_route, progress)

        waypoints = list(global_state.ambulance_trip_waypoints.get(amb_id, []))
        stage_index = global_state.ambulance_trip_stage_index.get(amb_id, 0)
        remaining_waypoints = waypoints[stage_index:] if waypoints else []

        if remaining_waypoints:
            target_pos = remaining_waypoints[0]
            target_victim = next(
                (
                    v
                    for v in global_state.env.victims
                    if v.pos == target_pos and v.victim_id not in global_state.rescued_victims
                ),
                target_victim,
            )
        elif target_pos is None and target_victim:
            target_pos = target_victim.pos
        elif target_pos is None and victim:
            target_pos = victim.pos
        if target_pos is None:
            continue

        start = global_state.get_current_pos(amb_id)
        if remaining_waypoints:
            new_result = global_state._build_waypoint_route(
                start,
                remaining_waypoints,
                global_state.current_algorithm,
                global_state.current_alpha,
            )
        else:
            route_result = search(
                global_state.env,
                start,
                target_pos,
                global_state.current_algorithm,
                global_state.ml,
                global_state.fuzzy,
                global_state.current_alpha,
                True,
                global_state.current_algorithm_params,
            )
            new_result = {
                "path": route_result.path,
                "cost": route_result.total_cost,
                "risk": route_result.risk_score,
                "segments": [route_result.path],
            }

        target_label = target_victim.victim_id if target_victim else victim.victim_id if victim and target_pos == victim.pos else "medical_centre"

        global_state.ambulance_routes[amb_id] = new_result["path"]
        global_state.ambulance_progress[amb_id] = 0
        global_state.route_costs[amb_id] = new_result["cost"]
        global_state.route_risk_scores.append(new_result["risk"])

        justify = (
            f"Route for {amb_id} to {target_label} crossed blocked cell {coords}; "
            f"replanned with {global_state.current_algorithm} (α={global_state.current_alpha})."
        )
        global_state.log_event(
            {
                "event_type": "REPLAN",
                "trigger_reason": "road_blocked",
                "victim_id": target_label,
                "chosen_path": new_result["path"],
                "alpha_used": global_state.current_alpha,
                "time_cost": new_result["cost"],
                "risk_cost": new_result["risk"],
                "old_route_cost": old_route_cost,
                "new_route_cost": new_result["cost"],
                "victim_priority_list": global_state.get_priority_list(),
                "assignment_plan": dict(global_state.active_assignments),
                "frontier_sizes": [],
                "optimality_ratio": 1.0,
                "fuzzy_risk_along_path": _path_fuzzy_risk(global_state.env, new_result["path"]),
                "justification_text": justify,
            }
        )

        emit(
            "route_planned",
            {
                "ambulance_id": amb_id,
                "victim_id": target_label,
                "algorithm": global_state.current_algorithm,
                "alpha": global_state.current_alpha,
                "path": new_result["path"],
                "cost": new_result["cost"],
                "risk_score": new_result["risk"],
                "justification": justify,
                "is_replan": True,
                "optimality_ratio": 1.0,
                "fuzzy_risk_along_path": _path_fuzzy_risk(global_state.env, new_result["path"]),
                "frontier_sizes": [],
            },
            broadcast=True,
        )


@socketio.on("trigger_event")
def handle_trigger_event(data: Dict[str, Any]) -> None:
    if not global_state.env:
        return

    event_type = data.get("type")
    coords = tuple(data.get("coords", []))
    if len(coords) != 2:
        return

    if event_type == "block":
        global_state.env.trigger_road_block(coords)
        _auto_replan_for_block(coords, trigger_reason="user_block")
        emit("event_triggered", {"type": "block", "coords": coords}, broadcast=True)
    elif event_type == "new_victim":
        global_state.env.trigger_new_victim(coords, "moderate")
        new_victim = global_state.env.victims[-1]
        global_state.refresh_victim_survival_probabilities()
        full_plan = global_state.reallocate_full_rescue(global_state.current_algorithm, global_state.current_alpha)
        assignment_plan = full_plan["assignment_plan"]
        for amb in global_state.env.ambulances:
            if not global_state.ambulance_routes.get(amb.agent_id):
                global_state._dispatch_next_queue_target(amb.agent_id)
        justify = f"New victim {new_victim.victim_id} at {coords} ({new_victim.severity}) – CSP reallocation needed."
        global_state.log_event(
            {
                "event_type": "ASSIGNMENT",
                "trigger_reason": "new_victim_detected",
                "victim_id": new_victim.victim_id,
                "victim_priority_list": full_plan["victim_priority_list"],
                "assignment_plan": assignment_plan,
                "justification_text": justify,
            }
        )
        emit(
            "victim_added",
            {"victim": new_victim.victim_id, "assignment_plan": assignment_plan},
            broadcast=True,
        )
        emit(
            "full_rescue_planned",
            {
                "assignment_plan": assignment_plan,
                "victim_priority_list": full_plan["victim_priority_list"],
                "route_summary": full_plan["route_summary"],
                "trigger_reason": "new_victim_detected",
            },
            broadcast=True,
        )
    emit("state_update", global_state.get_state_snapshot(), broadcast=True)


@socketio.on("step_simulation")
def handle_step_simulation() -> None:
    if not global_state.env or not global_state.simulation_running:
        return

    global_state.step_count += 1
    global_state.simulation_step = global_state.step_count
    scheduled_events = global_state.env.update(global_state.simulation_step)
    global_state.refresh_victim_survival_probabilities()
    for event_name, payload in scheduled_events:
        if event_name == "block":
            coords = tuple(payload)
            _auto_replan_for_block(coords, trigger_reason="scenario_event")
            emit("event_triggered", {"type": "block", "coords": coords}, broadcast=True)
        elif event_name == "new_victim":
            coords, _severity = payload
            new_victim = global_state.env.victims[-1]
            global_state.refresh_victim_survival_probabilities()
            # Note: For scenario-scheduled events, do NOT auto-trigger full rescue plan.
            # User must explicitly request via "Plan Full Rescue" button.
            assignment_plan = global_state.get_assignment_plan()
            global_state.log_event(
                {
                    "event_type": "ASSIGNMENT",
                    "trigger_reason": "scenario_event",
                    "victim_id": new_victim.victim_id,
                    "victim_priority_list": global_state.get_priority_list(),
                    "assignment_plan": assignment_plan,
                    "justification_text": f"Scenario event added {new_victim.victim_id} at {coords}; user can manually replan.",
                }
            )
            emit(
                "victim_added",
                {"victim": new_victim.victim_id, "assignment_plan": assignment_plan},
                broadcast=True,
            )

    for amb_id, route in list(global_state.ambulance_routes.items()):
        if not route or amb_id not in global_state.ambulance_progress:
            continue
        progress = global_state.ambulance_progress[amb_id]
        if progress < len(route) - 1:
            next_pos = route[progress + 1]
            if global_state.env.grid[next_pos[0]][next_pos[1]] == CellType.BLOCKED:
                _auto_replan_for_block(next_pos, trigger_reason="pre_move_block_validation")
                route = global_state.ambulance_routes.get(amb_id, [])
                progress = global_state.ambulance_progress.get(amb_id, 0)
                if not route or progress >= len(route) - 1:
                    continue
                next_pos = route[progress + 1]
                if global_state.env.grid[next_pos[0]][next_pos[1]] == CellType.BLOCKED:
                    continue
            global_state.ambulance_progress[amb_id] += 1
            next_pos = route[global_state.ambulance_progress[amb_id]]
            if global_state.env.grid[next_pos[0]][next_pos[1]] == CellType.RISK:
                global_state.risk_steps += 1
            global_state._advance_trip_stage(amb_id, next_pos)

        progress = global_state.ambulance_progress[amb_id]
        trip_victims = global_state.ambulance_trip_victims.get(amb_id, [])
        dropoff = global_state.ambulance_trip_dropoff.get(amb_id)
        if route and progress == len(route) - 1 and dropoff and route[-1] == dropoff and trip_victims:
            elapsed_steps = max(
                1,
                global_state.simulation_step
                - global_state.assignment_started_at.get(amb_id, global_state.simulation_step - 1),
            )
            rescue_time_sec = elapsed_steps * 0.5
            assignment_plan = dict(global_state.active_assignments)
            for victim_id in trip_victims:
                victim_at_goal = next((v for v in global_state.env.victims if v.victim_id == victim_id), None)
                if not victim_at_goal or victim_at_goal.victim_id in global_state.rescued_victims:
                    continue
                global_state.rescued_victims.add(victim_at_goal.victim_id)
                updated_survival = global_state.ml.predict_survival(
                    global_state._victim_features(victim_at_goal, float(global_state.step_count))
                )
                victim_at_goal.survival_prob = updated_survival
                global_state.rescue_step_durations.append(elapsed_steps)
                global_state.log_event(
                    {
                        "event_type": "RESCUE_COMPLETE",
                        "trigger_reason": "ambulance_arrival_med_center",
                        "victim_id": victim_at_goal.victim_id,
                        "updated_survival_prob": updated_survival,
                        "assignment_plan": assignment_plan,
                        "victim_priority_list": global_state.get_priority_list(),
                        "justification_text": f"{amb_id} delivered {victim_at_goal.victim_id} to a medical centre; rescue completed in {rescue_time_sec:.1f}s.",
                    }
                )
                emit(
                    "rescue_complete",
                    {
                        "ambulance_id": amb_id,
                        "victim_id": victim_at_goal.victim_id,
                        "updated_survival_prob": updated_survival,
                        "rescue_time": rescue_time_sec,
                    },
                    broadcast=True,
                )

            global_state.ambulance_loads[amb_id] = 0
            global_state._reset_trip_state(amb_id)
            global_state.ambulance_routes[amb_id] = []
            global_state.ambulance_progress[amb_id] = 0
            global_state.active_assignments.pop(amb_id, None)
            global_state.assignment_started_at.pop(amb_id, None)
            global_state.route_costs.pop(amb_id, None)

            global_state.refresh_victim_survival_probabilities()
            if global_state.rescue_queues.get(amb_id):
                global_state._dispatch_next_queue_target(amb_id)
    emit("state_update", global_state.get_state_snapshot(), broadcast=True)


@socketio.on("pause")
def handle_pause() -> None:
    global_state.simulation_running = False


@socketio.on("resume")
def handle_resume() -> None:
    global_state.simulation_running = True


def _generate_justification(
    victim: Victim, algorithm: str, alpha: float, cost: float
) -> str:
    if alpha == 0.0:
        return f"{victim.victim_id} ({victim.severity}): selected fast route (α=0). Risk accepted to minimise rescue time."
    if alpha == float("inf"):
        return f"{victim.victim_id} ({victim.severity}): selected safe route (α=∞). Time cost accepted to avoid hazards."
    return f"{victim.victim_id} ({victim.severity}): balanced route (α=1, algo={algorithm}). Path cost {cost:.2f}."


if __name__ == "__main__":
    socketio.run(app, debug=False, host="127.0.0.1", port=5001, use_reloader=False)
