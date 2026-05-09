"""Compatibility shim for tests: re-export `run_scenario` from dev_tools.

This keeps offline scenario runner callable as `main.run_scenario` for tests
while the heavy plotting code lives in `dev_tools/main_offline.py`.
"""
from dev_tools.main_offline import run_scenario  # type: ignore

__all__ = ["run_scenario"]
