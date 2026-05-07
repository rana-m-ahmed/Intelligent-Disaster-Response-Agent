from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

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
    optimality_ratio: float
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
            if env.grid[r][c] == CellType.BLOCKED:
                continue
            result.append((r, c))
    return result


def _cell_features(env: Environment, pos: GridPos) -> List[float]:
    """Build ML features for a grid cell."""
    cell = env.grid[pos[0]][pos[1]]
    center = min(env.med_centers, key=lambda c: manhattan(pos, c))
    distance = manhattan(pos, center)
    neighborhood = _neighborhood_stats(env, pos)
    severity = 2.0 if cell == CellType.RISK else 0.0
    return [severity, float(distance), neighborhood["hazard_rate"], neighborhood["block_prob"]]


def _neighborhood_stats(env: Environment, pos: GridPos) -> Dict[str, float]:
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


def _cell_penalty(env: Environment, pos: GridPos, ml_model: MLModel, fuzzy: FuzzyRisk) -> float:
    cell = env.grid[pos[0]][pos[1]]
    if cell != CellType.RISK:
        return 0.0, 0.0
    ml_risk = ml_model.predict_risk(_cell_features(env, pos))
    neighborhood = _neighborhood_stats(env, pos)
    fuzzy_weight = fuzzy.compute_risk_weight(neighborhood["block_prob"], neighborhood["hazard_rate"])
    # return tuple-like info via a simple combined value for legacy callers
    # but primarily callers should compute the multiplicative factor
    return ml_risk, fuzzy_weight


def edge_cost(
    env: Environment,
    pos: GridPos,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
) -> float:
    """Edge cost: base move cost plus alpha-weighted risk penalty."""
    base_move_cost = 1.0
    if env.grid[pos[0]][pos[1]] == CellType.BLOCKED:
        return math.inf
    # obtain ml risk and fuzzy weight
    ml_risk, fuzzy_weight = _cell_penalty(env, pos, ml_model, fuzzy)
    # CCP formula: cost = travel_time * (1 + alpha * ML_risk * fuzzy_weight)
    if math.isinf(alpha):
        factor = 1.0 + (ml_risk * fuzzy_weight)
    else:
        factor = 1.0 + (alpha * ml_risk * fuzzy_weight)
    return base_move_cost * factor


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
        ml_risk, fuzzy_weight = _cell_penalty(env, pos, ml_model, fuzzy)
        risk += ml_risk
        total += edge_cost(env, pos, ml_model, fuzzy, alpha)
    return total, risk


def _hop_count(path: List[GridPos]) -> float:
    return float(max(len(path) - 1, 0))


def search(
    env: Environment,
    start: GridPos,
    goal: GridPos,
    algorithm: str,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
    use_heuristic: bool = True,
) -> SearchResult:
    """Run BFS, DFS, Greedy Best-First, or A* and return a SearchResult."""
    algo = algorithm.lower()
    if algo not in {"bfs", "dfs", "greedy", "astar"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    start_time = time.perf_counter()
    result = _run_search_core(env, start, goal, algo, ml_model, fuzzy, alpha, use_heuristic)
    if algo == "bfs":
        optimality_ratio = 1.0
    else:
        # Compare against A* baseline (heuristic-enabled) per CCP guidance
        astar_baseline = _run_search_core(env, start, goal, "astar", ml_model, fuzzy, alpha, True)
        optimality_ratio = result.total_cost / max(astar_baseline.total_cost, 1e-9)
    runtime_sec = time.perf_counter() - start_time
    return SearchResult(
        path=result.path,
        total_cost=result.total_cost,
        risk_score=result.risk_score,
        nodes_expanded=result.nodes_expanded,
        frontier_sizes=result.frontier_sizes,
        optimality_ratio=optimality_ratio,
        runtime_sec=runtime_sec,
    )


def _run_search_core(
    env: Environment,
    start: GridPos,
    goal: GridPos,
    algo: str,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
    use_heuristic: bool = True,
) -> SearchResult:
    frontier_sizes: List[int] = []
    nodes_expanded = 0
    came_from: Dict[GridPos, Optional[GridPos]] = {start: None}

    if algo == "bfs":
        frontier: Deque[GridPos] = deque([start])
        visited = {start}
        while frontier:
            current = frontier.popleft()
            nodes_expanded += 1
            if current == goal:
                break
            for nxt in _neighbors(env, current):
                if env.grid[nxt[0]][nxt[1]] == CellType.BLOCKED:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                came_from[nxt] = current
                frontier.append(nxt)
            frontier_sizes.append(len(frontier))
    elif algo == "dfs":
        frontier = [start]
        visited = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1
            if current == goal:
                break
            for nxt in reversed(_neighbors(env, current)):
                if env.grid[nxt[0]][nxt[1]] == CellType.BLOCKED:
                    continue
                if nxt in visited:
                    continue
                if nxt not in came_from:
                    came_from[nxt] = current
                frontier.append(nxt)
            frontier_sizes.append(len(frontier))
    elif algo == "greedy":
        frontier = []
        counter = 0
        heapq.heappush(frontier, (manhattan(start, goal), counter, start))
        visited = {start}
        while frontier:
            _, _, current = heapq.heappop(frontier)
            nodes_expanded += 1
            if current == goal:
                break
            ordered_neighbors = sorted(
                _neighbors(env, current),
                key=lambda nxt: (manhattan(nxt, goal), abs(nxt[1] - goal[1]), abs(nxt[0] - goal[0])),
            )
            for nxt in ordered_neighbors:
                if env.grid[nxt[0]][nxt[1]] == CellType.BLOCKED:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (manhattan(nxt, goal), counter, nxt))
            frontier_sizes.append(len(frontier))
    else:
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0.0, 0.0, counter, start))
        best_costs = {start: 0.0}
        closed = set()
        while frontier:
            _, current_g, _, current = heapq.heappop(frontier)
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1
            if current == goal:
                break
            for nxt in _neighbors(env, current):
                if env.grid[nxt[0]][nxt[1]] == CellType.BLOCKED:
                    continue
                step_cost = edge_cost(env, nxt, ml_model, fuzzy, alpha)
                new_cost = current_g + step_cost
                if nxt in best_costs and new_cost >= best_costs[nxt]:
                    continue
                best_costs[nxt] = new_cost
                came_from[nxt] = current
                counter += 1
                # include heuristic only when allowed (Dijkstra / alpha=0 should disable if requested)
                heuristic = manhattan(nxt, goal) if use_heuristic else 0.0
                priority = new_cost + heuristic
                heapq.heappush(frontier, (priority, new_cost, counter, nxt))
            frontier_sizes.append(len(frontier))

    path = _reconstruct_path(came_from, start, goal)
    if algo in {"bfs", "dfs", "greedy"}:
        total_cost = _hop_count(path)
        risk_score = 0.0
    else:
        total_cost, risk_score = compute_path_cost(env, path, ml_model, fuzzy, alpha)
    return SearchResult(
        path=path,
        total_cost=total_cost,
        risk_score=risk_score,
        nodes_expanded=nodes_expanded,
        frontier_sizes=frontier_sizes,
        optimality_ratio=1.0,
        runtime_sec=0.0,
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
