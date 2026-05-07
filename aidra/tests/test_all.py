import json
import os
import sys

import pytest

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from csp import compare_backtracks, solve_csp
from environment import CellType, Environment
from fuzzy import FuzzyRisk
from logger import LOG_PATH, DecisionLogger
from ml_model import MLModel
from search import dijkstra, edge_cost, search
from main import run_scenario
from app import GlobalState


def test_environment_initialization():
    env = Environment("A")
    assert env.size == 10
    assert len(env.victims) == 5
    assert env.grid[0][0] == CellType.MED_CENTER
    assert env.grid[9][9] == CellType.MED_CENTER


def test_environment_dynamic_events():
    env = Environment("C")
    events = env.update(2)
    assert any(event[0] == "block" for event in events)
    events = env.update(4)
    assert any(event[0] == "new_victim" for event in events)
    assert len(env.victims) == 6


def test_severity_to_kits_policy():
    env = Environment("A")
    victim_map = {v.victim_id: v for v in env.victims}
    assert victim_map["V1"].kits_needed == 2
    assert victim_map["V2"].kits_needed == 2
    assert victim_map["V3"].kits_needed == 1
    assert victim_map["V4"].kits_needed == 1
    assert victim_map["V5"].kits_needed == 0


def test_search_algorithms_return_valid_paths():
    env = Environment("A")
    ml = MLModel()
    fuzzy = FuzzyRisk()
    start = env.ambulances[0].pos
    goal = env.victims[0].pos
    for algo in ["bfs", "dfs", "greedy", "astar"]:
        result = search(env, start, goal, algo, ml, fuzzy, 1.0)
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert all(env.grid[p[0]][p[1]] != CellType.BLOCKED for p in result.path)


def test_edge_cost_alpha_behavior():
    env = Environment("A")
    ml = MLModel()
    fuzzy = FuzzyRisk()
    pos = env.victims[0].pos
    time_only = edge_cost(env, pos, ml, fuzzy, 0.0)
    assert time_only == 1.0
    risk_only = edge_cost(env, pos, ml, fuzzy, float("inf"))
    assert 0.0 <= risk_only <= 1.0


def test_alpha_zero_matches_dijkstra():
    env = Environment("A")
    ml = MLModel()
    fuzzy = FuzzyRisk()
    a_star = search(env, env.ambulances[0].pos, env.victims[0].pos, "astar", ml, fuzzy, 0.0)
    dij = dijkstra(env, env.ambulances[0].pos, env.victims[0].pos, ml, fuzzy)
    assert pytest.approx(a_star.total_cost, rel=1e-5) == dij.total_cost


def test_csp_constraints_and_backtracks():
    env = Environment("A")
    result = solve_csp(env, use_mrv=True, use_forward_checking=True)
    assignment = result.assignment
    assert len(assignment.get("ambulance_1", [])) <= 2
    assert len(assignment.get("ambulance_2", [])) <= 2
    assert len(assignment.get("rescue_team", [])) <= 1
    total_kits = 0
    victim_map = {v.victim_id: v for v in env.victims}
    for victims in assignment.values():
        for victim_id in victims:
            total_kits += victim_map[victim_id].kits_needed
    assert total_kits <= env.medical_kits

    results = compare_backtracks(env)
    assert results["mrv_fc"].backtracks <= results["no_heuristics"].backtracks


def test_ml_training_and_predictions():
    ml = MLModel()
    X, y_survival, y_risk = ml._generate_dataset()
    assert X.shape[0] == 500
    pred_survival = ml.predict_survival([1.0, 1.0, 0.5, 5.0])
    pred_risk = ml.predict_risk([1.0, 1.0, 0.5, 5.0])
    assert 0.0 <= pred_survival <= 1.0
    assert 0.0 <= pred_risk <= 1.0


def test_fuzzy_rules_and_output():
    fuzzy = FuzzyRisk()
    weight = fuzzy.compute_risk_weight(0.9, 0.9)
    assert 0.0 <= weight <= 1.0
    assert weight > 0.5


def test_logger_writes_required_fields(tmp_path):
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    logger = DecisionLogger(reset=True)
    logger.log_event(
        {
            "event_type": "ROUTE_SELECTION",
            "victim_id": "V1",
            "chosen_path": [(0, 0), (0, 1)],
            "alpha_used": 0.0,
            "time_cost": 2.0,
            "risk_cost": 0.2,
            "justification_text": "Test route selection",
            "frontier_sizes": [1, 2],
            "scenario": "A",
        }
    )
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert entries[0]["event_type"] == "ROUTE_SELECTION"
    assert "chosen_path" in entries[0]
    assert "justification_text" in entries[0]


def test_replanning_trigger_and_valid_route():
    env = Environment("A")
    env.trigger_road_block((0, 1))
    assert env.is_replan_needed()
    ml = MLModel()
    fuzzy = FuzzyRisk()
    result = search(env, env.ambulances[0].pos, env.victims[0].pos, "astar", ml, fuzzy, 0.0)
    assert (0, 1) not in result.path


def test_replan_log_contains_costs():
    run_scenario("B")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    replans = [e for e in entries if e.get("event_type") == "REPLAN"]
    assert replans
    assert replans[0].get("old_route_cost") is not None
    assert replans[0].get("new_route_cost") is not None
    assert replans[0].get("trigger_reason") == "road_block"


def test_assignment_deduction_critical_plus_moderate_uses_three_kits():
    gs = GlobalState()
    gs.init_scenario("A")
    assert gs.env is not None
    before = gs.env.medical_kits

    ok, committed = gs._deduct_kits_for_victims(["V1", "V3"], "test_assignment")
    assert ok
    assert set(committed) == {"V1", "V3"}
    assert gs.env.medical_kits == before - 3


def test_resource_exhaustion_logs_every_csp_cycle_when_zero_kits():
    gs = GlobalState()
    gs.logger = DecisionLogger(reset=True)
    gs.init_scenario("A")
    assert gs.env is not None
    gs.env.medical_kits = 0

    gs.solve_csp_with_tracking()
    gs.solve_csp_with_tracking()

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    violations = [
        e
        for e in entries
        if e.get("event_type") == "CONSTRAINT_VIOLATION"
        and e.get("outcome") == "resource_exhaustion"
    ]
    assert len(violations) >= 2


def test_state_snapshot_exposes_latest_csp_backtracks():
    gs = GlobalState()
    gs.init_scenario("A")
    result = gs.solve_csp_with_tracking(use_mrv=True, use_forward_checking=True)
    assert result is not None
    assert gs.latest_csp_backtracks == result.backtracks

    snapshot = gs.get_state_snapshot()
    assert snapshot["csp_backtracks"] == result.backtracks
