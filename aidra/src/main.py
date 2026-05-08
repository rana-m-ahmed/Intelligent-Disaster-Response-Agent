from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

from csp import compare_backtracks, priority_score, solve_csp
from environment import CellType, Environment, Victim
from fuzzy import FuzzyRisk
from logger import DecisionLogger
from ml_model import MLModel
from search import SearchResult, hill_climb_assignments, search

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def print_grid(env: Environment, agents: Dict[str, Tuple[int, int]]) -> None:
    """Print a terminal grid representation with agent markers."""
    legend = {
        CellType.SAFE: "S",
        CellType.RISK: "R",
        CellType.BLOCKED: "B",
        CellType.MED_CENTER: "M",
        CellType.VICTIM: "V",
    }
    grid_display = [[legend[cell] for cell in row] for row in env.grid]
    for agent_id, pos in agents.items():
        grid_display[pos[0]][pos[1]] = agent_id
    for row in grid_display:
        print(" ".join(row))


def route_visualization(env: Environment, path: List[Tuple[int, int]], scenario: str) -> None:
    """Render a static matplotlib overlay of the selected route."""
    if plt is None:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    color_map = {
        CellType.SAFE: (0.7, 0.9, 0.7),
        CellType.RISK: (1.0, 0.7, 0.2),
        CellType.BLOCKED: (0.9, 0.2, 0.2),
        CellType.MED_CENTER: (0.2, 0.6, 0.9),
        CellType.VICTIM: (0.9, 0.9, 0.1),
    }
    grid_colors = np.zeros((env.size, env.size, 3))
    for r in range(env.size):
        for c in range(env.size):
            grid_colors[r, c] = color_map[env.grid[r][c]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid_colors)
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color="blue", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"route_vis_{scenario}.png"))
    plt.close(fig)


def choose_alpha(severity: str, risk_level: float) -> float:
    """Select alpha based on severity and local risk level."""
    if severity == "critical":
        return 0.0
    if risk_level > 0.7:
        return float("inf")
    return 1.0


def _victim_features(env: Environment, victim: Victim, time_since: float) -> List[float]:
    severity_map = {"critical": 2, "moderate": 1, "minor": 0}
    severity = severity_map.get(victim.severity, 0)
    center = env.med_centers[0]
    distance = abs(victim.pos[0] - center[0]) + abs(victim.pos[1] - center[1])
    cell = env.grid[victim.pos[0]][victim.pos[1]]
    area_risk = 0.7 if cell == CellType.RISK else 0.1
    return [float(severity), float(distance), float(area_risk), float(time_since)]


def _priority_list(env: Environment, ml: MLModel) -> List[Victim]:
    for victim in env.victims:
        victim.survival_prob = ml.predict_survival(_victim_features(env, victim, 5.0))
    return sorted(env.victims, key=lambda victim: priority_score(victim, ml, env), reverse=True)


def _build_pending(env: Environment, ml: MLModel, rescued: set) -> List[Victim]:
    ordered = _priority_list(env, ml)
    return [victim for victim in ordered if victim.victim_id not in rescued]


def _write_kpis(path: str, kpis: Dict[str, float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kpi", "value"])
        for key, value in kpis.items():
            writer.writerow([key, value])


def _write_backtracks(path: str, backtracks: Dict[str, int]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "backtracks"])
        for key, value in backtracks.items():
            writer.writerow([key, value])


def _plot_bar(values: Dict[str, float], title: str, output_path: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(values.keys())
    ax.bar(labels, values.values(), color="#4c72b0")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_confusion_matrices(metrics: Dict[str, Dict[str, object]], output_path: str) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, (model_name, report) in zip(axes, metrics.items()):
        cm = np.array(report.confusion)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(model_name)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _route_justification(victim: Victim, alpha: float) -> str:
    # legacy shim - kept for compatibility but prefer _detailed_route_justification
    if alpha == 0.0:
        return f"Critical victim {victim.victim_id}: alpha=0 for fastest rescue."
    if alpha == float("inf"):
        return f"High risk near {victim.victim_id}: alpha=inf to avoid hazards."
    return f"Victim {victim.victim_id}: alpha=1 for balanced time and risk."


def _detailed_route_justification(victim: Victim, result: SearchResult, alpha: float) -> str:
    """Return justification text that includes numeric time and risk values."""
    alpha_label = "inf" if alpha == float("inf") else str(alpha)
    return (
        f"Rescue {victim.victim_id}: time={result.total_cost:.2f}, "
        f"risk={result.risk_score:.2f}, alpha={alpha_label}"
    )


def _neighborhood_stats(env: Environment, pos: Tuple[int, int]) -> Dict[str, float]:
    rows = range(max(0, pos[0] - 1), min(env.size, pos[0] + 2))
    cols = range(max(0, pos[1] - 1), min(env.size, pos[1] + 2))
    total = 0
    blocked = 0
    risky = 0
    for r in rows:
        for c in cols:
            total += 1
            cell = env.grid[r][c]
            if cell == CellType.BLOCKED:
                blocked += 1
            elif cell == CellType.RISK:
                risky += 1
    return {
        "block_prob": blocked / max(total, 1),
        "hazard_rate": risky / max(total, 1),
    }


def _assignment_plan(assignments: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: list(v) for k, v in assignments.items()}


def run_scenario(scenario: str) -> Dict[str, object]:
    """Run a single scenario and emit KPIs, charts, and logs."""
    env = Environment(scenario)
    ml = MLModel()
    fuzzy = FuzzyRisk()
    logger = DecisionLogger(reset=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.log_event(
        {
            "event_type": "ML_REPORT",
            "ml_report": ml.get_metrics_report(),
            "scenario": scenario,
            "step": 0,
            "justification_text": "Startup model comparison report.",
            "trigger_reason": "startup_model_report",
        }
    )

    comparison: Dict[str, SearchResult] = {}
    start = env.ambulances[0].pos
    goal = env.victims[0].pos
    for algo in ["bfs", "dfs", "greedy", "astar"]:
        for alpha in [0.0, 1.0, float("inf")]:
            key = f"{algo}_alpha_{alpha}"
            result = search(env, start, goal, algo, ml, fuzzy, alpha)
            comparison[key] = result

    assignments_result = solve_csp(env, ml, use_mrv=True, use_forward_checking=True)
    refined_assignment = hill_climb_assignments(
        assignments_result.assignment, env, ml, fuzzy
    )
    backtrack_comparison = compare_backtracks(env)

    victim_priority = _priority_list(env, ml)
    logger.log_event(
        {
            "event_type": "VICTIM_ORDER",
            "victim_priority_list": [v.victim_id for v in victim_priority],
            "justification_text": "Priority based on severity and survival probability.",
            "scenario": scenario,
            "step": 0,
        }
    )
    logger.log_event(
        {
            "event_type": "ASSIGNMENT",
            "assignment_plan": _assignment_plan(refined_assignment),
            "justification_text": "CSP assignment with MRV+FC and hill climbing refinement.",
            "scenario": scenario,
            "trigger_reason": "scenario_init",
            "victim_priority_list": [
                {"victim_id": victim.victim_id, "priority": round(priority_score(victim, ml, env), 4)}
                for victim in victim_priority
            ],
            "step": 0,
        }
    )

    victims_saved = 0
    total_time = 0.0
    risk_exposure = 0.0
    nodes_expanded = {k: v.nodes_expanded for k, v in comparison.items()}
    # track current positions per resource so we dispatch the correct vehicle
    positions = {
        "ambulance_1": env.ambulances[0].pos,
        "ambulance_2": env.ambulances[1].pos,
        "rescue_team": env.rescue_team.pos,
    }
    step = 0
    rescued: set = set()
    pending = _build_pending(env, ml, rescued)

    # create victim->resource mapping from refined CSP assignment
    victim_to_resource: Dict[str, str] = {}
    for resource, victims in refined_assignment.items():
        for v_id in victims:
            victim_to_resource[v_id] = resource

    while pending:
        victim = pending.pop(0)
        # determine assigned resource and start position
        assigned_resource = victim_to_resource.get(victim.victim_id, "ambulance_1")
        start_pos = positions.get(assigned_resource, env.ambulances[0].pos)

        # Compute three trade-off paths: fast (alpha=0), balanced (alpha=1), safe (alpha=inf)
        path_fast = search(env, start_pos, victim.pos, "astar", ml, fuzzy, 0.0)
        path_balanced = search(env, start_pos, victim.pos, "astar", ml, fuzzy, 1.0)
        path_safe = search(env, start_pos, victim.pos, "astar", ml, fuzzy, float("inf"))

        # determine fuzzy threshold from local neighborhood
        nb = _neighborhood_stats(env, victim.pos)
        fuzzy_threshold = fuzzy.compute_risk_weight(nb["block_prob"], nb["hazard_rate"])

        # selection logic: prefer fast if risk acceptable, otherwise safe if cost penalty is small, else balanced
        if path_fast.risk_score <= fuzzy_threshold:
            result = path_fast
            chosen_alpha = 0.0
        elif path_safe.total_cost <= 1.2 * path_fast.total_cost:
            result = path_safe
            chosen_alpha = float("inf")
        else:
            result = path_balanced
            chosen_alpha = 1.0

        step += 1
        events = env.update(step)
        trigger_reason = None
        for event in events:
            if event[0] == "block":
                trigger_reason = "road_block"
                print(f"ROAD BLOCKED at {event[1]} - REPLANNING")
            elif event[0] == "new_victim":
                trigger_reason = "new_victim"
                print(f"NEW VICTIM at {event[1][0]} - REPLANNING")
                pending = _build_pending(env, ml, rescued)
                pending = [v for v in pending if v.victim_id != victim.victim_id]
                logger.log_event(
                    {
                        "event_type": "ASSIGNMENT",
                        "assignment_plan": _assignment_plan(
                                solve_csp(env, ml, use_mrv=True, use_forward_checking=True).assignment
                        ),
                        "justification_text": "CSP reassigned due to new victim.",
                        "scenario": scenario,
                        "trigger_reason": "new_victim_detected",
                        "victim_priority_list": [
                            {"victim_id": victim.victim_id, "priority": round(priority_score(victim, ml, env), 4)}
                            for victim in _priority_list(env, ml)
                        ],
                        "step": step,
                    }
                )

        if env.is_replan_needed():
            replan_start = result.path[1] if len(result.path) > 1 else start_pos
            replanned = search(env, replan_start, victim.pos, "astar", ml, fuzzy, chosen_alpha)
            logger.log_event(
                {
                    "event_type": "REPLAN",
                    "algorithm": "astar",
                    "victim_id": victim.victim_id,
                    "old_route_cost": result.total_cost,
                    "new_route_cost": replanned.total_cost,
                    "trigger_reason": trigger_reason or "dynamic_event",
                    "scenario": scenario,
                    "chosen_path": replanned.path,
                    "alpha_used": chosen_alpha,
                    "time_cost": replanned.total_cost,
                    "risk_cost": replanned.risk_score,
                    "frontier_sizes": replanned.frontier_sizes,
                    "optimality_ratio": replanned.optimality_ratio,
                    "fuzzy_risk_along_path": 0.0,
                    "step": step,
                }
            )
            # if replanned is significantly more expensive (>20%), evaluate switching vehicles
            if replanned.total_cost > 1.2 * result.total_cost:
                # pick alternate ambulance if available
                alt = None
                if assigned_resource == "ambulance_1":
                    alt = "ambulance_2"
                elif assigned_resource == "ambulance_2":
                    alt = "ambulance_1"
                if alt is not None:
                    logger.log_event(
                        {
                            "event_type": "REPLAN_EVALUATE_SWITCH",
                            "from": assigned_resource,
                            "to": alt,
                            "victim_id": victim.victim_id,
                            "old_cost": result.total_cost,
                            "new_cost": replanned.total_cost,
                            "ratio": replanned.total_cost / max(result.total_cost, 1.0),
                            "step": step,
                            "scenario": scenario,
                        }
                    )
            result = replanned
            # update vehicle position to the replan start so the next search begins from correct cell
            positions[assigned_resource] = replan_start
            # also reflect on env agents
            if assigned_resource == "ambulance_1":
                env.ambulances[0].pos = positions[assigned_resource]
            elif assigned_resource == "ambulance_2":
                env.ambulances[1].pos = positions[assigned_resource]
            else:
                env.rescue_team.pos = positions[assigned_resource]
            env.clear_replan_flag()

        victims_saved += 1
        total_time += result.total_cost
        risk_exposure += result.risk_score

        # decrement kits when a victim is rescued and account kit usage
        env.medical_kits = max(0, env.medical_kits - victim.kits_needed)

        updated_survival = ml.predict_survival(_victim_features(env, victim, 10.0))
        logger.log_event(
            {
                "event_type": "ROUTE_SELECTION",
                    "algorithm": "astar",
                "victim_id": victim.victim_id,
                "chosen_path": result.path,
                "alpha_used": chosen_alpha,
                "time_cost": result.total_cost,
                "risk_cost": result.risk_score,
                "frontier_sizes": result.frontier_sizes,
                "optimality_ratio": result.optimality_ratio,
                "fuzzy_risk_along_path": 0.0,
                "justification_text": _detailed_route_justification(victim, result, chosen_alpha),
                "scenario": scenario,
                "step": step,
            }
        )
        logger.log_event(
            {
                "event_type": "RESCUE_COMPLETE",
                "victim_id": victim.victim_id,
                "updated_survival_prob": updated_survival,
                "scenario": scenario,
                "assignment_plan": _assignment_plan(refined_assignment),
                "step": step,
            }
        )

        rescued.add(victim.victim_id)
        # update the dispatched vehicle position to victim location
        positions[assigned_resource] = victim.pos
        if assigned_resource == "ambulance_1":
            env.ambulances[0].pos = victim.pos
        elif assigned_resource == "ambulance_2":
            env.ambulances[1].pos = victim.pos
        else:
            env.rescue_team.pos = victim.pos
        # display currently dispatched agent and rescue team
        print_grid(
            env,
            {
                "A": positions.get(assigned_resource, positions["ambulance_1"]),
                "T": positions.get("rescue_team", env.rescue_team.pos),
            },
        )

    avg_time = total_time / max(victims_saved, 1)

    comparison_rows = [
        (key, res.total_cost, res.nodes_expanded) for key, res in comparison.items()
    ]
    with open(
        os.path.join(RESULTS_DIR, f"comparison_{scenario}.csv"), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "total_cost", "nodes_expanded"])
        writer.writerows(comparison_rows)

    nodes_chart = {row[0]: row[2] for row in comparison_rows}
    cost_chart = {row[0]: row[1] for row in comparison_rows}
    _plot_bar(
        nodes_chart,
        "Nodes Expanded",
        os.path.join(RESULTS_DIR, f"nodes_{scenario}.png"),
    )
    _plot_bar(
        cost_chart,
        "Path Cost",
        os.path.join(RESULTS_DIR, f"cost_{scenario}.png"),
    )

    _plot_confusion_matrices(
        ml.metrics.get("survival", {}), os.path.join(RESULTS_DIR, "cm_survival.png")
    )
    _plot_confusion_matrices(
        ml.metrics.get("risk", {}), os.path.join(RESULTS_DIR, "cm_risk.png")
    )

    with open(os.path.join(RESULTS_DIR, "ml_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "model", "accuracy", "precision", "recall", "f1"])
        for task, reports in ml.metrics.items():
            for model_name, report in reports.items():
                writer.writerow(
                    [
                        task,
                        model_name,
                        report.accuracy,
                        report.precision,
                        report.recall,
                        report.f1,
                    ]
                )

    backtrack_counts = {
        key: value.backtracks for key, value in backtrack_comparison.items()
    }
    _write_backtracks(
        os.path.join(RESULTS_DIR, f"csp_backtracks_{scenario}.csv"),
        backtrack_counts,
    )
    _plot_bar(
        backtrack_counts,
        "CSP Backtracks",
        os.path.join(RESULTS_DIR, f"csp_backtracks_{scenario}.png"),
    )

    optimal_cost = comparison["astar_alpha_0.0"].total_cost
    avg_cost = sum(res.total_cost for res in comparison.values()) / len(comparison)
    path_opt_ratio = max(avg_cost / max(optimal_cost, 1.0), 1.0)
    victim_map = {v.victim_id: v for v in env.victims}
    kit_used = sum(victim_map[v_id].kits_needed for v_id in rescued)
    ambulances_dispatched = len(refined_assignment)
    teams_deployed = 1 if env else 0
    resources_used = ambulances_dispatched + kit_used + teams_deployed
    resources_available = len(env.ambulances) + 10 + 1
    resource_util = (resources_used / max(resources_available, 1)) * 100.0
    kpi = {
        "victims_saved": victims_saved,
        "average_rescue_time": avg_time,
        "path_optimality_ratio": path_opt_ratio,
        "resource_util_rate": resource_util,
        "risk_exposure_score": risk_exposure,
        "kit_usage": kit_used,
        "resources_used": resources_used,
        "resources_available": resources_available,
    }
    _write_kpis(os.path.join(RESULTS_DIR, f"kpis_{scenario}.csv"), kpi)

    kpi_table_path = os.path.join(RESULTS_DIR, "kpi_table.csv")
    with open(kpi_table_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(
                [
                    "Scenario",
                    "Algorithm",
                    "VictimsSaved",
                    "AvgRescueTime",
                    "PathOptimalityRatio",
                    "RiskExposureScore",
                    "NodesExpanded",
                    "CSP_Backtracks",
                ]
            )
        for algo, res in comparison.items():
            algo_ratio = res.total_cost / max(optimal_cost, 1.0)
            writer.writerow(
                [
                    scenario,
                    algo,
                    victims_saved,
                    round(avg_time, 3),
                    round(algo_ratio, 3),
                    round(risk_exposure, 3),
                    res.nodes_expanded,
                    backtrack_counts["mrv_fc"],
                ]
            )

    route_visualization(env, comparison["astar_alpha_1.0"].path, scenario)

    return {
        "victims_saved": victims_saved,
        "average_rescue_time": avg_time,
        "risk_exposure": risk_exposure,
        "nodes_expanded": nodes_expanded,
        "backtrack_comparison": backtrack_comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["A", "B", "C"], default="A")
    args = parser.parse_args()
    metrics = run_scenario(args.scenario)
    print(f"Scenario {args.scenario} complete.")
    print(metrics)


if __name__ == "__main__":
    main()
