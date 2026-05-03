from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from environment import Environment, Victim


@dataclass
class CSPResult:
    assignment: Dict[str, List[str]]
    backtracks: int


def priority_score(victim: Victim) -> float:
    """Compute victim priority using severity and survival probability."""
    severity_norm = {
        "critical": 1.0,
        "moderate": 0.6,
        "minor": 0.3,
    }.get(victim.severity, 0.3)
    survival_prob = victim.survival_prob
    return 0.7 * severity_norm + 0.3 * (1.0 - survival_prob)


def solve_csp(
    env: Environment,
    use_mrv: bool = True,
    use_forward_checking: bool = True,
) -> CSPResult:
    """Solve victim assignment CSP with optional MRV and forward checking."""
    victims = sorted(env.victims, key=priority_score, reverse=True)
    variables = ["ambulance_1", "ambulance_2", "rescue_team"]
    domains = {
        "ambulance_1": _subsets([v.victim_id for v in victims], max_size=2),
        "ambulance_2": _subsets([v.victim_id for v in victims], max_size=2),
        "rescue_team": _subsets([v.victim_id for v in victims], max_size=1),
    }

    best_assignment: Dict[str, List[str]] = {v: [] for v in variables}
    best_score = -1.0
    backtracks = 0

    def backtrack(assignment: Dict[str, List[str]]) -> bool:
        nonlocal best_assignment, best_score, backtracks

        if len(assignment) == len(variables):
            score = _assignment_score(assignment, victims, env.medical_kits)
            if score > best_score:
                best_score = score
                best_assignment = {k: list(v) for k, v in assignment.items()}
            return True

        # MRV chooses the variable with the smallest remaining domain.
        var = _select_unassigned_variable(
            variables, assignment, domains, victims, env.medical_kits, use_mrv
        )
        any_solution = False
        ordered_values = _order_domain_values(
            var, domains[var], victims, assignment
        )
        for value in ordered_values:
            if _consistent(var, value, assignment, victims, env.medical_kits):
                assignment[var] = value
                # Forward checking prunes assignments that make any domain empty.
                if use_forward_checking and not _forward_check(assignment, domains):
                    assignment.pop(var)
                    continue
                if backtrack(assignment):
                    any_solution = True
                assignment.pop(var)
        if not any_solution:
            backtracks += 1
        return any_solution

    backtrack({})
    return CSPResult(best_assignment, backtracks)


def compare_backtracks(env: Environment) -> Dict[str, CSPResult]:
    """Return backtrack counts for no heuristics vs MRV vs MRV+FC."""
    return {
        "no_heuristics": solve_csp(env, use_mrv=False, use_forward_checking=False),
        "mrv": solve_csp(env, use_mrv=True, use_forward_checking=False),
        "mrv_fc": solve_csp(env, use_mrv=True, use_forward_checking=True),
    }


def _assignment_score(
    assignment: Dict[str, List[str]],
    victims: List[Victim],
    kits_available: int,
) -> float:
    """Score assignments while enforcing medical kit limits."""
    victim_map = {v.victim_id: v for v in victims}
    total_kits = 0
    score = 0.0
    for victims_list in assignment.values():
        for victim_id in victims_list:
            victim = victim_map[victim_id]
            total_kits += victim.kits_needed
            score += priority_score(victim)
    if total_kits > kits_available:
        return -1.0
    return score


def _select_unassigned_variable(
    variables: List[str],
    assignment: Dict[str, List[str]],
    domains: Dict[str, List[List[str]]],
    victims: List[Victim],
    kits_available: int,
    use_mrv: bool,
) -> str:
    """Select the next unassigned variable, optionally using MRV."""
    unassigned = [v for v in variables if v not in assignment]
    if not use_mrv:
        return unassigned[0]
    return min(
        unassigned,
        key=lambda v: len(_filtered_domain(v, domains[v], assignment, victims, kits_available)),
    )


def _filtered_domain(
    var: str,
    domain: List[List[str]],
    assignment: Dict[str, List[str]],
    victims: List[Victim],
    kits_available: int,
) -> List[List[str]]:
    return [
        value
        for value in domain
        if _consistent(var, value, assignment, victims, kits_available)
    ]


def _order_domain_values(
    var: str,
    domain: List[List[str]],
    victims: List[Victim],
    assignment: Dict[str, List[str]],
) -> List[List[str]]:
    """Order values by descending victim priority to reduce backtracking."""
    victim_map = {v.victim_id: v for v in victims}
    assigned = set()
    for assigned_list in assignment.values():
        assigned.update(assigned_list)

    def score(value: List[str]) -> float:
        return sum(priority_score(victim_map[v]) for v in value if v not in assigned)

    return sorted(domain, key=score, reverse=True)


def _consistent(
    var: str,
    value: List[str],
    assignment: Dict[str, List[str]],
    victims: List[Victim],
    kits_available: int,
) -> bool:
    """Check CSP constraints: capacity, uniqueness, and kit limits."""
    if var.startswith("ambulance") and len(value) > 2:
        return False
    if var == "rescue_team" and len(value) > 1:
        return False
    assigned = set()
    for assigned_list in assignment.values():
        for victim_id in assigned_list:
            assigned.add(victim_id)
    if any(v in assigned for v in value):
        return False
    total_kits = 0
    victim_map = {v.victim_id: v for v in victims}
    for victim_id in value:
        total_kits += victim_map[victim_id].kits_needed
    for assigned_list in assignment.values():
        for victim_id in assigned_list:
            total_kits += victim_map[victim_id].kits_needed
    return total_kits <= kits_available


def _forward_check(
    assignment: Dict[str, List[str]],
    domains: Dict[str, List[List[str]]],
) -> bool:
    """Return False if any unassigned variable has no valid value left."""
    assigned = set()
    for value in assignment.values():
        assigned.update(value)
    for var, domain in domains.items():
        if var in assignment:
            continue
        if not any(not (set(d) & assigned) for d in domain):
            return False
    return True


def _subsets(items: List[str], max_size: int) -> List[List[str]]:
    """Generate subsets up to max_size for CSP domains."""
    result: List[List[str]] = [[]]
    for item in items:
        result.extend([subset + [item] for subset in result if len(subset) < max_size])
    return result
