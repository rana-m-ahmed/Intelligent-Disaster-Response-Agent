from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyRisk:
    def __init__(self) -> None:
        """Initialize fuzzy sets and rule base for risk weighting."""
        self.blockage = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "blockage")
        self.spread = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "spread")
        self.risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), "risk")

        self.blockage["low"] = fuzz.trimf(self.blockage.universe, [0, 0, 0.4])
        self.blockage["med"] = fuzz.trimf(self.blockage.universe, [0.2, 0.5, 0.8])
        self.blockage["high"] = fuzz.trimf(self.blockage.universe, [0.6, 1.0, 1.0])

        self.spread["low"] = fuzz.trimf(self.spread.universe, [0, 0, 0.4])
        self.spread["med"] = fuzz.trimf(self.spread.universe, [0.2, 0.5, 0.8])
        self.spread["high"] = fuzz.trimf(self.spread.universe, [0.6, 1.0, 1.0])

        self.risk["low"] = fuzz.trimf(self.risk.universe, [0, 0, 0.4])
        self.risk["med"] = fuzz.trimf(self.risk.universe, [0.2, 0.5, 0.8])
        self.risk["high"] = fuzz.trimf(self.risk.universe, [0.6, 1.0, 1.0])

        # Rule base combines blockage probability and hazard spread rate.
        rules = [
            ctrl.Rule(self.blockage["high"] & self.spread["high"], self.risk["high"]),
            ctrl.Rule(self.blockage["high"] & self.spread["med"], self.risk["high"]),
            ctrl.Rule(self.blockage["med"] & self.spread["high"], self.risk["high"]),
            ctrl.Rule(self.blockage["low"] & self.spread["low"], self.risk["low"]),
            ctrl.Rule(self.blockage["med"] & self.spread["med"], self.risk["med"]),
            ctrl.Rule(self.blockage["low"] & self.spread["med"], self.risk["med"]),
        ]
        system = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(system)

    def compute_risk_weight(self, block_prob: float, hazard_rate: float) -> float:
        """Return a crisp risk weight in [0, 1]."""
        self.simulator.input["blockage"] = block_prob
        self.simulator.input["spread"] = hazard_rate
        self.simulator.compute()
        return float(self.simulator.output["risk"])
