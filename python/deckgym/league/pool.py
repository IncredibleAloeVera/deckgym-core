#!/usr/bin/env python3
"""
Opponent Pool Management for League Training.

Handles the collection of checkpoints and baselines, including statistics
tracking and eviction logic.
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

from deckgym.config import PFSP_NEUTRAL_WINRATE


class OpponentPool:
    """
    Manages the pool of opponents (checkpoints and baselines).

    Responsibilities:
    - Maintaining the set of active opponents.
    - Tracking win/loss/draw statistics.
    - Handling eviction of weakest models.
    """

    def __init__(self, pool_size: int, checkpoint_dir: str):
        self.pool_size = pool_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # {name: {"path": str, "onnx_path": str, "wins": int, "losses": int, "draws": int, "added_at_step": int, "is_baseline": bool, "baseline_code": str}}
        self.opponents: Dict[str, Dict] = {}

    def add_opponent(self, name: str, data: Dict):
        """Add an opponent to the pool."""
        # Ensure total stats are present
        data.setdefault("total_wins", 0)
        data.setdefault("total_losses", 0)
        data.setdefault("total_draws", 0)
        self.opponents[name] = data

    def remove_opponent(self, name: str) -> Optional[Dict]:
        """Remove an opponent from the pool and return its data."""
        return self.opponents.pop(name, None)

    def contains(self, name: str) -> bool:
        """Check if an opponent is in the pool."""
        return name in self.opponents

    def get_data(self, name: str) -> Dict:
        """Get data for a specific opponent."""
        result = self.opponents.get(name)
        if result is None:
            raise KeyError(f"Opponent '{name}' not found")
        return result

    def reset_statistics(self):
        """Reset per-rollout win/loss/draw statistics for all opponents."""
        for data in self.opponents.values():
            data["wins"] = 0
            data["losses"] = 0
            data["draws"] = 0

    def reset_total_statistics(self):
        """Reset total (cumulative) statistics for all opponents.

        Call this when pool composition changes (eviction/addition).
        """
        for data in self.opponents.values():
            data["total_wins"] = 0
            data["total_losses"] = 0
            data["total_draws"] = 0

    def update_results(self, episode_results: List[Tuple[str, str]]):
        """Update win rates based on recent episode results."""
        for opp_name, result in episode_results:
            if opp_name in self.opponents:
                if result == "agent_win":
                    self.opponents[opp_name]["losses"] += 1
                    self.opponents[opp_name]["total_losses"] += 1
                elif result == "opp_win":
                    self.opponents[opp_name]["wins"] += 1
                    self.opponents[opp_name]["total_wins"] += 1
                elif result == "draw":
                    self.opponents[opp_name]["draws"] += 1
                    self.opponents[opp_name]["total_draws"] += 1


    def get_eviction_candidates(
        self, exclude_names: List[str] = []
    ) -> List[Tuple[str, float, int]]:
        """
        Get non-baseline opponents ranked for eviction.
        Returns: List of (name, priority, added_at_step)
        """
        exclude_names = exclude_names or []

        # Filter for models (non-baselines) that aren't excluded
        models = {
            n: d
            for n, d in self.opponents.items()
            if not d.get("is_baseline", False) and n not in exclude_names
        }

        candidates = []
        for name, data in models.items():
            total = data["wins"] + data["losses"] + data["draws"]
            # We use a simple winrate for eviction sorting (priorities are for selection)
            winrate = (data["wins"] + 1) / (total + 2)
            candidates.append((name, winrate, int(data.get("added_at_step", 0))))

        # Sort by winrate (lowest first), then age (oldest first)
        candidates.sort(key=lambda x: (x[1], x[2]))
        return candidates

    def cleanup_files(self, data: Dict):
        """Delete checkpoint and ONNX files associated with an opponent."""
        if data.get("path"):
            p = Path(data["path"])
            if p.exists():
                p.unlink()

        if data.get("onnx_path"):
            p = Path(data["onnx_path"])
            if p.exists():
                p.unlink()

    @property
    def total_count(self) -> int:
        return len(self.opponents)

    @property
    def model_count(self) -> int:
        return len([d for d in self.opponents.values() if not d.get("is_baseline")])

    @property
    def baseline_codes(self) -> List[str]:
        return [
            d["baseline_code"] for d in self.opponents.values() if d.get("is_baseline")
        ]
