from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from environment import CellType, Environment
from fuzzy import FuzzyRisk
from ml_model import MLModel

GridPos = Tuple[int, int]


@dataclass
class SearchResult:
    path: List[GridPos]
    total_cost: float
    risk_score: float
    nodes_expanded: int
    frontier_sizes: List[int]
    runtime_sec: float


def manhattan(a: GridPos, b: GridPos) -> int:
    """Compute Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(env: Environment, pos: GridPos) -> List[GridPos]:
    """Return valid, non-blocked neighbors for a grid position."""
    candidates = [
        (pos[0] + 1, pos[1]),
        (pos[0] - 1, pos[1]),
        (pos[0], pos[1] + 1),
        (pos[0], pos[1] - 1),
    ]
    result = []
    for r, c in candidates:
        if 0 <= r < env.size and 0 <= c < env.size:
            if env.grid[r][c] != CellType.BLOCKED:
                result.append((r, c))
    return result


def _cell_features(env: Environment, pos: GridPos) -> List[float]:
    """Build ML features for a grid cell."""
    center = env.med_centers[0]
    distance = manhattan(pos, center)
    cell = env.grid[pos[0]][pos[1]]
    if cell == CellType.RISK:
        area_risk = 0.7
        severity = 2.0
    elif cell == CellType.BLOCKED:
        area_risk = 1.0
        severity = 2.0
    else:
        area_risk = 0.1
        severity = 0.0
    return [severity, float(distance), area_risk, 1.0]


def edge_cost(
    env: Environment,
    pos: GridPos,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
) -> float:
    """Edge cost: travel_time * (1 + alpha * ML_risk * fuzzy_weight)."""
    travel_time = 1.0
    ml_risk = ml_model.predict_risk(_cell_features(env, pos))
    fuzzy_weight = fuzzy.compute_risk_weight(ml_risk, ml_risk)
    if math.isinf(alpha):
        return ml_risk * fuzzy_weight
    return travel_time * (1.0 + alpha * ml_risk * fuzzy_weight)


def compute_path_cost(
    env: Environment,
    path: List[GridPos],
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
) -> Tuple[float, float]:
    """Compute total cost and cumulative ML risk for a path."""
    total = 0.0
    risk = 0.0
    for pos in path[1:]:
        ml_risk = ml_model.predict_risk(_cell_features(env, pos))
        fuzzy_weight = fuzzy.compute_risk_weight(ml_risk, ml_risk)
        risk += ml_risk
        if math.isinf(alpha):
            total += ml_risk * fuzzy_weight
        else:
            total += 1.0 * (1.0 + alpha * ml_risk * fuzzy_weight)
    return total, risk


def search(
    env: Environment,
    start: GridPos,
    goal: GridPos,
    algorithm: str,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
) -> SearchResult:
    """Run BFS, DFS, Greedy Best-First, or A* and return a SearchResult."""
    algo = algorithm.lower()
    if algo not in {"bfs", "dfs", "greedy", "astar"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    start_time = time.perf_counter()
    frontier_sizes: List[int] = []
    nodes_expanded = 0

    if algo in {"bfs", "dfs"}:
        frontier: List[GridPos] = [start]
        came_from: Dict[GridPos, Optional[GridPos]] = {start: None}
        while frontier:
            current = frontier.pop(0) if algo == "bfs" else frontier.pop()
            nodes_expanded += 1
            if current == goal:
                break
            for nxt in _neighbors(env, current):
                if nxt not in came_from:
                    came_from[nxt] = current
                    frontier.append(nxt)
            # Track frontier size for traceability after each expansion.
            frontier_sizes.append(len(frontier))
    else:
        frontier = []
        heapq.heappush(frontier, (0.0, start))
        came_from = {start: None}
        g_costs = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            nodes_expanded += 1
            if current == goal:
                break
            for nxt in _neighbors(env, current):
                step_cost = edge_cost(env, nxt, ml_model, fuzzy, alpha)
                new_cost = g_costs[current] + step_cost
                if nxt not in g_costs or new_cost < g_costs[nxt]:
                    g_costs[nxt] = new_cost
                    came_from[nxt] = current
                    if algo == "greedy":
                        priority = manhattan(nxt, goal)
                    else:
                        priority = new_cost + manhattan(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
            # Track frontier size for traceability after each expansion.
            frontier_sizes.append(len(frontier))

    path = _reconstruct_path(came_from, start, goal)
    total_cost, risk_score = compute_path_cost(env, path, ml_model, fuzzy, alpha)
    runtime_sec = time.perf_counter() - start_time
    return SearchResult(
        path=path,
        total_cost=total_cost,
        risk_score=risk_score,
        nodes_expanded=nodes_expanded,
        frontier_sizes=frontier_sizes,
        runtime_sec=runtime_sec,
    )


def dijkstra(
    env: Environment,
    start: GridPos,
    goal: GridPos,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
) -> SearchResult:
    """Return time-only A* (alpha=0) as Dijkstra-equivalent."""
    return search(env, start, goal, "astar", ml_model, fuzzy, alpha=0.0)


def _reconstruct_path(
    came_from: Dict[GridPos, Optional[GridPos]],
    start: GridPos,
    goal: GridPos,
) -> List[GridPos]:
    """Rebuild the path from goal to start."""
    current = goal
    if current not in came_from:
        return [start]
    path = [current]
    while current != start:
        current = came_from[current]
        if current is None:
            break
        path.append(current)
    path.reverse()
    return path


def hill_climb_assignments(
    assignment: Dict[str, List[str]],
    env: Environment,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
) -> Dict[str, List[str]]:
    """Refine CSP assignments by swapping victims between ambulances."""
    best = {k: list(v) for k, v in assignment.items()}
    best_score = _average_rescue_time(best, env, ml_model, fuzzy)
    improved = True
    while improved:
        improved = False
        for v1 in list(best.get("ambulance_1", [])):
            for v2 in list(best.get("ambulance_2", [])):
                candidate = {k: list(v) for k, v in best.items()}
                candidate["ambulance_1"].remove(v1)
                candidate["ambulance_2"].remove(v2)
                candidate["ambulance_1"].append(v2)
                candidate["ambulance_2"].append(v1)
                score = _average_rescue_time(candidate, env, ml_model, fuzzy)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    return best


def _average_rescue_time(
    assignment: Dict[str, List[str]],
    env: Environment,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
) -> float:
    """Compute average rescue time for an assignment using A* (alpha=0)."""
    victim_map = {v.victim_id: v for v in env.victims}
    times = []
    for agent, victims in assignment.items():
        if not victims:
            continue
        if agent == "rescue_team":
            start = env.rescue_team.pos
        elif agent == "ambulance_2":
            start = env.ambulances[1].pos
        else:
            start = env.ambulances[0].pos
        for victim_id in victims:
            victim = victim_map[victim_id]
            result = search(env, start, victim.pos, "astar", ml_model, fuzzy, 0.0)
            times.append(result.total_cost)
    if not times:
        return 0.0
    return sum(times) / len(times)
