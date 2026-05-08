from __future__ import annotations

import heapq
import math
import time
import random
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
    algorithm_params: dict | None = None,
) -> SearchResult:
    """Run BFS, DFS, Greedy Best-First, or A* and return a SearchResult."""
    algo = algorithm.lower()
    if algo not in {"bfs", "dfs", "greedy", "astar", "simulated_annealing"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    start_time = time.perf_counter()
    if algo == "simulated_annealing":
        result = _simulated_annealing_search(env, start, goal, ml_model, fuzzy, alpha, algorithm_params or {})
    else:
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
    algorithm_params: dict | None = None,
) -> SearchResult:
    frontier_sizes: List[int] = []
    nodes_expanded = 0

    if algo == "bfs":
        frontier: Deque[Tuple[GridPos, List[GridPos]]] = deque([(start, [start])])
        visited = {start}
        while frontier:
            current, path = frontier.popleft()
            nodes_expanded += 1
            if current == goal:
                total_cost = _hop_count(path)
                return SearchResult(path, total_cost, 0.0, nodes_expanded, frontier_sizes, 1.0, 0.0)
            for nxt in _neighbors(env, current):
                if nxt in visited:
                    continue
                visited.add(nxt)
                frontier.append((nxt, path + [nxt]))
            frontier_sizes.append(len(frontier))

    elif algo == "dfs":
        frontier: List[Tuple[GridPos, List[GridPos]]] = [(start, [start])]
        visited = set()
        while frontier:
            current, path = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1
            if current == goal:
                total_cost = _hop_count(path)
                return SearchResult(path, total_cost, 0.0, nodes_expanded, frontier_sizes, 1.0, 0.0)
            for nxt in reversed(_neighbors(env, current)):
                if nxt in visited:
                    continue
                frontier.append((nxt, path + [nxt]))
            frontier_sizes.append(len(frontier))

    elif algo == "greedy":
        frontier: List[Tuple[float, int, GridPos, List[GridPos]]] = []
        counter = 0
        heapq.heappush(frontier, (manhattan(start, goal), counter, start, [start]))
        visited = {start}
        while frontier:
            _, _, current, path = heapq.heappop(frontier)
            nodes_expanded += 1
            if current == goal:
                total_cost = _hop_count(path)
                return SearchResult(path, total_cost, 0.0, nodes_expanded, frontier_sizes, 1.0, 0.0)
            ordered_neighbors = sorted(
                _neighbors(env, current),
                key=lambda nxt: (manhattan(nxt, goal), abs(nxt[1] - goal[1]), abs(nxt[0] - goal[0])),
            )
            for nxt in ordered_neighbors:
                if nxt in visited:
                    continue
                visited.add(nxt)
                counter += 1
                heapq.heappush(frontier, (manhattan(nxt, goal), counter, nxt, path + [nxt]))
            frontier_sizes.append(len(frontier))

    else:
        frontier: List[Tuple[float, float, int, GridPos, List[GridPos]]] = []
        counter = 0
        heapq.heappush(frontier, (0.0, 0.0, counter, start, [start]))
        best_costs = {start: 0.0}
        closed = set()
        while frontier:
            _, current_g, _, current, path = heapq.heappop(frontier)
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1
            if current == goal:
                total_cost, risk_score = compute_path_cost(env, path, ml_model, fuzzy, alpha)
                return SearchResult(path, total_cost, risk_score, nodes_expanded, frontier_sizes, 1.0, 0.0)
            for nxt in _neighbors(env, current):
                step_cost = edge_cost(env, nxt, ml_model, fuzzy, alpha)
                new_cost = current_g + step_cost
                if nxt in best_costs and new_cost >= best_costs[nxt]:
                    continue
                best_costs[nxt] = new_cost
                counter += 1
                heuristic = manhattan(nxt, goal) if use_heuristic else 0.0
                priority = new_cost + heuristic
                heapq.heappush(frontier, (priority, new_cost, counter, nxt, path + [nxt]))
            frontier_sizes.append(len(frontier))

    return SearchResult([start], 0.0, 0.0, nodes_expanded, frontier_sizes, 1.0, 0.0)


def _simulated_annealing_search(
    env: Environment,
    start: GridPos,
    goal: GridPos,
    ml_model: MLModel,
    fuzzy: FuzzyRisk,
    alpha: float,
    params: dict,
) -> SearchResult:
    """A practical Simulated Annealing search that perturbs A* subpaths.

    Strategy:
    - Seed with an A* path (heuristic-enabled).
    - Repeatedly select two interior indices i<j and replan the subpath between them
      using A*; accept according to Metropolis criterion.
    - Keep best path found and return its SearchResult.
    """
    temperature = float(params.get("temperature", 1.0))
    cooling_rate = float(params.get("cooling_rate", 0.995))
    max_iterations = int(params.get("max_iterations", 1000))

    start_time = time.perf_counter()
    # Baseline using A*
    baseline = _run_search_core(env, start, goal, "astar", ml_model, fuzzy, alpha, True, params)
    best_path = list(baseline.path)
    best_cost, best_risk = compute_path_cost(env, best_path, ml_model, fuzzy, alpha)
    current_path = list(best_path)
    current_cost = best_cost

    nodes_expanded = baseline.nodes_expanded
    frontier_sizes: List[int] = []

    if len(current_path) <= 2:
        runtime_sec = time.perf_counter() - start_time
        return SearchResult(best_path, best_cost, best_risk, nodes_expanded, frontier_sizes, 1.0, runtime_sec)

    for it in range(max_iterations):
        # pick two indices i<j avoiding endpoints
        i = random.randint(1, max(1, len(current_path) - 2))
        j = random.randint(i + 1, len(current_path) - 1)

        a = current_path[i - 1]
        b = current_path[j]

        # replan subpath from a to b
        subresult = _run_search_core(env, a, b, "astar", ml_model, fuzzy, alpha, True, params)
        nodes_expanded += subresult.nodes_expanded
        if not subresult.path or subresult.path[0] != a or subresult.path[-1] != b:
            # skip invalid perturbation
            temperature *= cooling_rate
            continue

        new_path = current_path[: i] + subresult.path + current_path[j + 1 :]
        new_cost, new_risk = compute_path_cost(env, new_path, ml_model, fuzzy, alpha)

        delta = new_cost - current_cost
        accept = False
        if delta <= 0:
            accept = True
        else:
            prob = math.exp(-delta / max(1e-12, temperature))
            if random.random() < prob:
                accept = True

        if accept:
            current_path = new_path
            current_cost = new_cost
            # update best if improved
            if new_cost < best_cost:
                best_cost = new_cost
                best_risk = new_risk
                best_path = list(new_path)

        temperature *= cooling_rate
        if temperature <= 1e-12:
            break

    runtime_sec = time.perf_counter() - start_time
    # Compare against baseline for optimality ratio
    optimality_ratio = best_cost / max(baseline.total_cost, 1e-9)
    return SearchResult(best_path, best_cost, best_risk, nodes_expanded, frontier_sizes, optimality_ratio, runtime_sec)


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
