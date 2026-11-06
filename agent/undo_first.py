"""Framework adapter: registers the UndoAgent core as an ARC-AGI-3-Agents `Agent` subclass.

Drop this file next to the framework's agents/ package (the Kaggle rerun cell does that) and import it
before `AVAILABLE_AGENTS` is built, or register it explicitly: AVAILABLE_AGENTS["undofirst"] = UndoFirst.
"""
from __future__ import annotations

try:
    from agents.agent import Agent          # inside the framework package
except ImportError:                          # local testing with a copied framework module
    from fw_agents.agent import Agent        # type: ignore

from undo_agent import UndoAgent


class UndoFirst(Agent):
    """Reversibility-aware frontier exploration (mode 'undo'); MAX_ACTIONS bounds the per-game budget."""

    MAX_ACTIONS = 3000
    MODE = "undo"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.core = UndoAgent(mode=self.MODE, seed=0, max_actions=self.MAX_ACTIONS)

    def is_done(self, frames, latest_frame) -> bool:
        return self.core.is_done(frames, latest_frame)

    def choose_action(self, frames, latest_frame):
        return self.core.choose_action(frames, latest_frame)
