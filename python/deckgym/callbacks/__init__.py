#!/usr/bin/env python3
"""
Callback modules for RL training.

Provides SB3-compatible callbacks for:
- Episode metrics logging
- Frozen opponent updates
- PFSP (Prioritized Fictitious Self-Play)
- Interpretability metrics (entropy, action categories, value head stats)
- Memory monitoring (RSS tracking for leak detection)
"""

from deckgym.callbacks.episode_metrics import EpisodeMetricsCallback
from deckgym.callbacks.frozen_opponent import FrozenOpponentCallback
from deckgym.callbacks.interpretability import InterpretabilityCallback
from deckgym.callbacks.memory_monitor import MemoryMonitorCallback
from deckgym.callbacks.pause_resume import PauseResumeCallback
from deckgym.callbacks.pfsp import PFSPCallback

__all__ = [
    "EpisodeMetricsCallback",
    "FrozenOpponentCallback",
    "InterpretabilityCallback",
    "MemoryMonitorCallback",
    "PauseResumeCallback",
    "PFSPCallback",
]
