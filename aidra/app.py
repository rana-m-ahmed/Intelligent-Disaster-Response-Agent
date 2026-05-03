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
from csp import priority_score, solve_csp
from fuzzy import FuzzyRisk
from logger import DecisionLogger
from ml_model import MLModel
from search import compute_path_cost, search

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
        self.ambulance_routes: Dict[str, List[Tuple[int, int]]] = {}
        self.ambulance_progress: Dict[str, int] = {}
        self.rescued_victims: set = set()
        self.simulation_running = True
        self.simulation_step = 0
        self.rescue_step_durations: List[int] = []
        self.route_risk_scores: List[float] = []
        self.active_assignments: Dict[str, str] = {}
        self.route_costs: Dict[str, float] = {}
        self.assignment_started_at: Dict[str, int] = {}

    def init_scenario(self, scenario: str) -> None:
        with state_lock:
            self.env = Environment(scenario)
            self.ambulance_routes = {}
            self.ambulance_progress = {}
            self.rescued_victims = set()
            self.simulation_running = True
            self.simulation_step = 0
            self.rescue_step_durations = []
            self.route_risk_scores = []
            self.active_assignments = {}
            self.route_costs = {}
            self.assignment_started_at = {}
            for amb in self.env.ambulances:
                self.ambulance_routes[amb.agent_id] = []
                self.ambulance_progress[amb.agent_id] = 0

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
            result.append(
                {
                    "id": amb.agent_id,
                    "pos": current_pos,
                    "route": route,
                    "progress": progress,
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
        risk_exposure = (
            sum(self.route_risk_scores) / len(self.route_risk_scores)
            if self.route_risk_scores
            else 0.0
        )
        return {
            "grid": self.get_grid_for_json(),
            "victims": self.get_victims_for_json(),
            "ambulances": self.get_ambulances_for_json(),
            "med_centers": self.env.med_centers if self.env else [],
            "rescued_victims": len(self.rescued_victims),
            "total_victims": total_victims,
            "avg_rescue_time": avg_rescue_time_sec,
            "risk_exposure": risk_exposure,
            "csp_assignment": dict(self.active_assignments),
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
        active_victims = [v for v in self.env.victims if v.victim_id not in self.rescued_victims]
        sorted_victims = sorted(active_victims, key=priority_score, reverse=True)
        return [
            {
                "victim_id": v.victim_id,
                "priority": round(priority_score(v), 4),
            }
            for v in sorted_victims
        ]

    def get_assignment_plan(self) -> Dict[str, str]:
        if not self.env:
            return {}
        csp_result = solve_csp(self.env)
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


global_state = GlobalState()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@socketio.on("init_scenario")
def handle_init_scenario(data: Dict[str, str]) -> None:
    scenario = data.get("scenario", "A")
    global_state.init_scenario(scenario)

    global_state.logger.log_event(
        {
            "event_type": "SCENARIO_START",
            "scenario": scenario,
            "justification_text": f"Starting scenario {scenario}.",
            "trigger_reason": "scenario_init",
            "victim_priority_list": global_state.get_priority_list(),
            "assignment_plan": global_state.get_assignment_plan(),
        }
    )

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

    global_state.current_algorithm = algorithm
    global_state.current_alpha = alpha

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

    start = global_state.get_current_pos(assigned_ambulance)
    result = search(global_state.env, start, victim.pos, algorithm, global_state.ml, global_state.fuzzy, alpha)

    justify = _generate_justification(victim, algorithm, alpha, result.total_cost)

    global_state.ambulance_routes[assigned_ambulance] = result.path
    global_state.ambulance_progress[assigned_ambulance] = 0
    global_state.active_assignments[assigned_ambulance] = victim_id
    global_state.route_costs[assigned_ambulance] = result.total_cost
    global_state.assignment_started_at[assigned_ambulance] = global_state.simulation_step
    global_state.route_risk_scores.append(result.risk_score)

    global_state.logger.log_event(
        {
            "event_type": "ROUTE_SELECTION",
            "victim_id": victim_id,
            "chosen_path": result.path,
            "alpha_used": alpha,
            "time_cost": result.total_cost,
            "risk_cost": result.risk_score,
            "algorithm": algorithm,
            "justification_text": justify,
            "frontier_sizes": result.frontier_sizes,
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
            "path": result.path,
            "cost": result.total_cost,
            "risk_score": result.risk_score,
            "justification": justify,
            "is_replan": False,
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
        global_state.logger.log_event(
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
        if not victim or victim.victim_id in global_state.rescued_victims:
            continue

        start = global_state.get_current_pos(amb_id)
        new_result = search(
            global_state.env,
            start,
            victim.pos,
            global_state.current_algorithm,
            global_state.ml,
            global_state.fuzzy,
            global_state.current_alpha,
        )

        global_state.ambulance_routes[amb_id] = new_result.path
        global_state.ambulance_progress[amb_id] = 0
        global_state.route_costs[amb_id] = new_result.total_cost
        global_state.route_risk_scores.append(new_result.risk_score)

        justify = (
            f"Route for {amb_id} to {victim.victim_id} crossed blocked cell {coords}; "
            f"replanned with {global_state.current_algorithm} (α={global_state.current_alpha})."
        )
        global_state.logger.log_event(
            {
                "event_type": "REPLAN",
                "trigger_reason": "road_blocked",
                "victim_id": victim.victim_id,
                "chosen_path": new_result.path,
                "alpha_used": global_state.current_alpha,
                "time_cost": new_result.total_cost,
                "risk_cost": new_result.risk_score,
                "old_route_cost": old_route_cost,
                "new_route_cost": new_result.total_cost,
                "victim_priority_list": global_state.get_priority_list(),
                "assignment_plan": dict(global_state.active_assignments),
                "frontier_sizes": new_result.frontier_sizes,
                "justification_text": justify,
            }
        )

        emit(
            "route_planned",
            {
                "ambulance_id": amb_id,
                "victim_id": victim.victim_id,
                "algorithm": global_state.current_algorithm,
                "alpha": global_state.current_alpha,
                "path": new_result.path,
                "cost": new_result.total_cost,
                "risk_score": new_result.risk_score,
                "justification": justify,
                "is_replan": True,
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
        assignment_plan = global_state.get_assignment_plan()
        justify = f"New victim {new_victim.victim_id} at {coords} ({new_victim.severity}) – CSP reallocation needed."
        global_state.logger.log_event(
            {
                "event_type": "ASSIGNMENT",
                "trigger_reason": "new_victim_detected",
                "victim_id": new_victim.victim_id,
                "victim_priority_list": global_state.get_priority_list(),
                "assignment_plan": assignment_plan,
                "justification_text": justify,
            }
        )
        emit(
            "victim_added",
            {"victim": new_victim.victim_id, "assignment_plan": assignment_plan},
            broadcast=True,
        )
    emit("state_update", global_state.get_state_snapshot(), broadcast=True)


@socketio.on("step_simulation")
def handle_step_simulation() -> None:
    if not global_state.env or not global_state.simulation_running:
        return

    global_state.simulation_step += 1
    scheduled_events = global_state.env.update(global_state.simulation_step)
    for event_name, payload in scheduled_events:
        if event_name == "block":
            coords = tuple(payload)
            _auto_replan_for_block(coords, trigger_reason="scenario_event")
            emit("event_triggered", {"type": "block", "coords": coords}, broadcast=True)
        elif event_name == "new_victim":
            coords, _severity = payload
            new_victim = global_state.env.victims[-1]
            assignment_plan = global_state.get_assignment_plan()
            global_state.logger.log_event(
                {
                    "event_type": "ASSIGNMENT",
                    "trigger_reason": "scenario_event",
                    "victim_id": new_victim.victim_id,
                    "victim_priority_list": global_state.get_priority_list(),
                    "assignment_plan": assignment_plan,
                    "justification_text": f"Scenario event added {new_victim.victim_id} at {coords}; CSP reallocation computed.",
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
            global_state.ambulance_progress[amb_id] += 1

        progress = global_state.ambulance_progress[amb_id]
        if route and progress == len(route) - 1:
            victim_at_goal = next(
                (v for v in global_state.env.victims if v.pos == route[-1]),
                None,
            )
            if victim_at_goal and victim_at_goal.victim_id not in global_state.rescued_victims:
                global_state.rescued_victims.add(victim_at_goal.victim_id)
                elapsed_steps = max(
                    1,
                    global_state.simulation_step
                    - global_state.assignment_started_at.get(amb_id, global_state.simulation_step - 1),
                )
                global_state.rescue_step_durations.append(elapsed_steps)
                updated_survival = global_state.ml.predict_survival(
                    [1.0, float(elapsed_steps), 0.5, 5.0]
                )
                assignment_plan = dict(global_state.active_assignments)
                global_state.logger.log_event(
                    {
                        "event_type": "RESCUE_COMPLETE",
                        "trigger_reason": "ambulance_arrival",
                        "victim_id": victim_at_goal.victim_id,
                        "updated_survival_prob": updated_survival,
                        "assignment_plan": assignment_plan,
                        "victim_priority_list": global_state.get_priority_list(),
                        "justification_text": f"{amb_id} reached {victim_at_goal.victim_id}; rescue completed in {elapsed_steps * 0.5:.1f}s.",
                    }
                )
                emit(
                    "rescue_complete",
                    {
                        "ambulance_id": amb_id,
                        "victim_id": victim_at_goal.victim_id,
                        "updated_survival_prob": updated_survival,
                        "rescue_time": elapsed_steps * 0.5,
                    },
                    broadcast=True,
                )

            global_state.ambulance_routes[amb_id] = []
            global_state.ambulance_progress[amb_id] = 0
            global_state.active_assignments.pop(amb_id, None)
            global_state.assignment_started_at.pop(amb_id, None)
            global_state.route_costs.pop(amb_id, None)

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
    socketio.run(app, debug=False, host="0.0.0.0", port=5000, use_reloader=False)
