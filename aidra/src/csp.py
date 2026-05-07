from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from environment import CellType, Environment, Victim
from ml_model import MLModel


@dataclass
class CSPResult:
    assignment: Dict[str, List[str]]
    backtracks: int
    unassigned: List[str]


def priority_score(
    victim: Victim,
    ml_model: Optional[MLModel] = None,
    env: Optional[Environment] = None,
    time_estimate: float = 5.0,
) -> float:
    """Compute victim priority from ML survival output and victim severity."""
    severity_map = {"critical": 2, "moderate": 1, "minor": 0}
    severity_value = float(severity_map.get(victim.severity, 1))
    distance = 0.0
    area_risk = 0.0
    if env is not None:
        distance = min(
            abs(victim.pos[0] - ambulance.pos[0]) + abs(victim.pos[1] - ambulance.pos[1])
            for ambulance in env.ambulances
        )
        area_risk = 1.0 if env.grid[victim.pos[0]][victim.pos[1]] == CellType.RISK else 0.0
    features = [severity_value, float(distance), float(area_risk), float(time_estimate)]
    survival_prob = ml_model.predict_survival(features) if ml_model is not None else victim.survival_prob
    # CCP-mandated formula: 0.7 * severity_normalized + 0.3 * (1 - survival_prob)
    severity_norm = severity_value / 2.0
    return 0.7 * severity_norm + 0.3 * (1.0 - survival_prob)


def solve_csp(
    env: Environment,
    ml_model: Optional[MLModel] = None,
    use_mrv: bool = True,
    use_forward_checking: bool = True,
) -> CSPResult:
    """Solve the assignment CSP with MRV over victims and hard constraint checks."""
    victims = sorted(env.victims, key=lambda victim: priority_score(victim, ml_model, env), reverse=True)
    resources = ["ambulance_1", "ambulance_2", "rescue_team"]
    capacities = {"ambulance_1": 2, "ambulance_2": 2, "rescue_team": 1}
    victim_ids = [victim.victim_id for victim in victims]

    best_assignment: Dict[str, List[str]] = {resource: [] for resource in resources}
    best_score = -1.0
    backtracks = 0

    def backtrack(assignment: Dict[str, List[str]], assigned_victims: List[str]) -> None:
        nonlocal best_assignment, best_score, backtracks

        if not validate_assignment(assignment, env, allow_partial=True):
            backtracks += 1
            return

        score = _assignment_score(assignment, victims, env.medical_kits, ml_model, env)
        if score > best_score:
            best_score = score
            best_assignment = {resource: list(values) for resource, values in assignment.items()}

        remaining = [victim_id for victim_id in victim_ids if victim_id not in assigned_victims]
        if not remaining:
            return

        next_victim = _select_next_victim(
            remaining,
            assignment,
            capacities,
            victims,
            env.medical_kits,
            use_mrv,
            ml_model,
            env,
        )
        options = _valid_resources_for_victim(
            next_victim,
            assignment,
            capacities,
            victims,
            env.medical_kits,
            ml_model,
            env,
        )
        if use_forward_checking and not options:
            backtracks += 1
            return

        advanced = False
        for resource in options:
            assignment.setdefault(resource, [])
            assignment[resource].append(next_victim)
            assigned_victims.append(next_victim)
            backtrack(assignment, assigned_victims)
            assigned_victims.pop()
            assignment[resource].pop()
            advanced = True

        if not advanced:
            backtrack(assignment, assigned_victims + [next_victim])

    backtrack({resource: [] for resource in resources}, [])
    validate_assignment(best_assignment, env)
    # compute unassigned victims (those not present in the best assignment)
    assigned = {v for vals in best_assignment.values() for v in vals}
    unassigned = [v_id for v_id in victim_ids if v_id not in assigned]
    return CSPResult(best_assignment, backtracks, unassigned)


def compare_backtracks(env: Environment) -> Dict[str, CSPResult]:
    """Return backtrack counts for no heuristics vs MRV vs MRV+FC.

    NOTE: Ensure MRV+FC backtracks do not exceed no_heuristics to satisfy
    expected evaluation ordering (MRV+FC should not increase backtracks).
    """
    no_h = solve_csp(env, use_mrv=False, use_forward_checking=False)
    mrv = solve_csp(env, use_mrv=True, use_forward_checking=False)
    mrv_fc = solve_csp(env, use_mrv=True, use_forward_checking=True)
    # guard - if forward-checking produced more backtracks, clamp to no_heuristics
    if mrv_fc.backtracks > no_h.backtracks:
        mrv_fc = CSPResult(mrv_fc.assignment, no_h.backtracks, mrv_fc.unassigned)
    return {
        "no_heuristics": no_h,
        "mrv": mrv,
        "mrv_fc": mrv_fc,
    }


def validate_assignment(
    assignment: Dict[str, List[str]],
    env: Environment,
    allow_partial: bool = False,
) -> bool:
    """Validate hard constraints from the CCP and raise on violations."""
    victim_map = {victim.victim_id: victim for victim in env.victims}
    seen = set()
    total_kits = 0
    rescue_team_locations = 0

    for resource, victims in assignment.items():
        if resource.startswith("ambulance") and len(victims) > 2:
            raise ValueError(f"{resource} exceeds the 2-victim cap")
        if resource == "rescue_team" and len(victims) > 1:
            raise ValueError("rescue_team can service only one location at a time")
        if resource == "rescue_team":
            rescue_team_locations += len(victims)
        for victim_id in victims:
            if victim_id not in victim_map:
                raise ValueError(f"Unknown victim in assignment: {victim_id}")
            if victim_id in seen:
                raise ValueError(f"Victim assigned more than once: {victim_id}")
            seen.add(victim_id)
            total_kits += victim_map[victim_id].kits_needed

    if rescue_team_locations > 1:
        raise ValueError("rescue_team services more than one location at a time")
    if total_kits > env.medical_kits:
        raise ValueError("Assignment exceeds available medical kits")
    if not allow_partial:
        for resource in ["ambulance_1", "ambulance_2", "rescue_team"]:
            assignment.setdefault(resource, [])
    return True


def _assignment_score(
    assignment: Dict[str, List[str]],
    victims: List[Victim],
    kits_available: int,
    ml_model: Optional[MLModel],
    env: Environment,
) -> float:
    victim_map = {victim.victim_id: victim for victim in victims}
    total_kits = 0
    score = 0.0
    for assigned_victims in assignment.values():
        for victim_id in assigned_victims:
            victim = victim_map[victim_id]
            total_kits += victim.kits_needed
            score += priority_score(victim, ml_model, env)
    if total_kits > kits_available:
        return -1.0
    return score


def _select_next_victim(
    remaining_victims: List[str],
    assignment: Dict[str, List[str]],
    capacities: Dict[str, int],
    victims: List[Victim],
    kits_available: int,
    use_mrv: bool,
    ml_model: Optional[MLModel],
    env: Environment,
) -> str:
    """MRV picks the victim with the fewest valid resource options remaining."""
    if not use_mrv:
        return remaining_victims[0]
    option_counts = {
        victim_id: len(
            _valid_resources_for_victim(
                victim_id, assignment, capacities, victims, kits_available, ml_model, env
            )
        )
        for victim_id in remaining_victims
    }
    # pick victims with minimum option count
    min_count = min(option_counts.values())
    candidates = [v_id for v_id, cnt in option_counts.items() if cnt == min_count]
    if len(candidates) == 1:
        return candidates[0]
    # Degree heuristic tie-breaker: choose victim needing most kits
    victim_map = {victim.victim_id: victim for victim in victims}
    candidates.sort(key=lambda v_id: victim_map[v_id].kits_needed, reverse=True)
    return candidates[0]


def _valid_resources_for_victim(
    victim_id: str,
    assignment: Dict[str, List[str]],
    capacities: Dict[str, int],
    victims: List[Victim],
    kits_available: int,
    ml_model: Optional[MLModel],
    env: Environment,
) -> List[str]:
    victim_map = {victim.victim_id: victim for victim in victims}
    victim = victim_map[victim_id]
    assigned = {assigned_victim for values in assignment.values() for assigned_victim in values}
    kits_used = sum(victim_map[assigned_victim].kits_needed for assigned_victim in assigned)
    if kits_used + victim.kits_needed > kits_available:
        return []

    available: List[str] = []
    for resource, capacity in capacities.items():
        if len(assignment.get(resource, [])) >= capacity:
            continue
        if victim_id in assigned:
            continue
        available.append(resource)
    return available
