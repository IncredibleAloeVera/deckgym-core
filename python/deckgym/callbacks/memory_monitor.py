#!/usr/bin/env python3
"""
MemoryMonitorCallback - Logs process RAM usage to TensorBoard.

Tracks RSS (Resident Set Size) of the current process at each rollout end.
Useful to catch memory leaks during very long training runs.
"""

import os
import resource
from stable_baselines3.common.callbacks import BaseCallback


class MemoryMonitorCallback(BaseCallback):
    """
    Logs RAM usage (RSS) to TensorBoard at each rollout end.

    Metrics logged:
    - debug/rss_gb: Physical RAM used by the training process (GB)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> None:
        """Log RAM usage at end of each rollout."""
        if self.logger is None:
            return
        # ru_maxrss is in KB on Linux
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss_kb / 1024 / 1024
        self.logger.record("debug/rss_gb", rss_gb)

    def _on_step(self) -> bool:
        return True
