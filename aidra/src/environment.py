from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

GridPos = Tuple[int, int]


class CellType(str, Enum):
    SAFE = "SAFE"
    RISK = "RISK"
    BLOCKED = "BLOCKED"
    MED_CENTER = "MED_CENTER"
    VICTIM = "VICTIM"


@dataclass
class Victim:
    victim_id: str
    pos: GridPos
    severity: str
    kits_needed: int
    survival_prob: float = 0.5


@dataclass
class Agent:
    agent_id: str
    pos: GridPos


class Environment:
    """10x10 grid environment with dynamic events and resource state."""
    def __init__(self, scenario: str = "A") -> None:
        self.size = 10
        self.grid: List[List[CellType]] = [
            [CellType.SAFE for _ in range(self.size)] for _ in range(self.size)
        ]
        self.scenario = scenario.upper()
        self._replan_needed = False
        self.med_centers = [(0, 0), (9, 9)]
        self.ambulances = [Agent("A1", (0, 9)), Agent("A2", (9, 0))]
        self.rescue_team = Agent("T1", (5, 5))
        self.medical_kits = 10
        self.victims: List[Victim] = []
        self._init_grid()
        self._init_victims()
        self._init_risk_cells()
        self._event_schedule = self._build_event_schedule()

    def _init_grid(self) -> None:
        for pos in self.med_centers:
            self.grid[pos[0]][pos[1]] = CellType.MED_CENTER

    def _init_victims(self) -> None:
        baseline = [
            ("V1", (1, 2), "critical", 4),
            ("V2", (2, 7), "critical", 4),
            ("V3", (7, 3), "moderate", 2),
            ("V4", (8, 8), "moderate", 2),
            ("V5", (4, 4), "minor", 1),
        ]
        self.victims = [Victim(*v) for v in baseline]
        for victim in self.victims:
            self.grid[victim.pos[0]][victim.pos[1]] = CellType.VICTIM

    def _init_risk_cells(self) -> None:
        for pos in [(3, 3), (3, 4), (6, 6), (7, 6)]:
            if self.grid[pos[0]][pos[1]] == CellType.SAFE:
                self.grid[pos[0]][pos[1]] = CellType.RISK

    def _build_event_schedule(self) -> Dict[int, List[Tuple[str, Tuple]]]:
        if self.scenario == "B":
            return {3: [("block", (3, 5))]}
        if self.scenario == "C":
            return {
                2: [("block", (3, 5))],
                4: [("new_victim", ((6, 1), "moderate"))],
                5: [("block", (6, 5))],
            }
        return {}

    def get_state(self) -> Dict[str, object]:
        """Return a snapshot of the environment state."""
        return {
            "grid": self.grid,
            "victims": list(self.victims),
            "ambulances": list(self.ambulances),
            "rescue_team": self.rescue_team,
            "medical_kits": self.medical_kits,
            "scenario": self.scenario,
        }

    def update(self, step: int) -> List[Tuple[str, Tuple]]:
        """Apply scheduled events for the given step and return them."""
        events = self._event_schedule.get(step, [])
        for event in events:
            if event[0] == "block":
                self.trigger_road_block(event[1])
            elif event[0] == "new_victim":
                pos, severity = event[1]
                self.trigger_new_victim(pos, severity)
        return events

    def trigger_road_block(self, cell: GridPos) -> None:
        """Block a cell and flag replanning."""
        if self.grid[cell[0]][cell[1]] != CellType.MED_CENTER:
            self.grid[cell[0]][cell[1]] = CellType.BLOCKED
            self._replan_needed = True

    def trigger_new_victim(self, pos: GridPos, severity: str) -> None:
        """Add a new victim to the grid and flag replanning."""
        victim_id = f"V{len(self.victims) + 1}"
        kits_needed = self._kits_for_severity(severity)
        victim = Victim(victim_id, pos, severity, kits_needed)
        self.victims.append(victim)
        self.grid[pos[0]][pos[1]] = CellType.VICTIM
        self._replan_needed = True

    def change_risk_level(self, cell: GridPos, level: int) -> None:
        """Set a cell's risk level and flag replanning."""
        if level <= 0:
            self.grid[cell[0]][cell[1]] = CellType.SAFE
        else:
            self.grid[cell[0]][cell[1]] = CellType.RISK
        self._replan_needed = True

    def is_replan_needed(self) -> bool:
        """Return True if replanning has been triggered."""
        return self._replan_needed

    def clear_replan_flag(self) -> None:
        """Clear the replanning flag after handling."""
        self._replan_needed = False

    @staticmethod
    def _kits_for_severity(severity: str) -> int:
        mapping = {"critical": 4, "moderate": 2, "minor": 1}
        return mapping.get(severity, 1)
