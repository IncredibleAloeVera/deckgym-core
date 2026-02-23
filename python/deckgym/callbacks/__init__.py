#!/usr/bin/env python3
"""
Callback modules for RL training.

Provides SB3-compatible callbacks for:
- Episode metrics logging
- Frozen opponent updates
- PFSP (Prioritized Fictitious Self-Play)
"""

from deckgym.callbacks.episode_metrics import EpisodeMetricsCallback
from deckgym.callbacks.frozen_opponent import FrozenOpponentCallback
from deckgym.callbacks.pause_resume import PauseResumeCallback
from deckgym.callbacks.pfsp import PFSPCallback

__all__ = [
    "EpisodeMetricsCallback",
    "FrozenOpponentCallback",
    "PauseResumeCallback",
    "PFSPCallback",
]
