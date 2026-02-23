#!/usr/bin/env python3
"""
Pause/Resume Callback for Training.

Press 'p' during training to pause indefinitely, press 'p' again to resume.
Uses a background thread for non-blocking keyboard input.
"""

import sys
import time
import threading
from stable_baselines3.common.callbacks import BaseCallback


class PauseResumeCallback(BaseCallback):
    """
    Callback that listens for 'p' key to toggle pause/resume during training.
    Blocks in _on_step() while paused, keeping the process alive.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._paused = threading.Event()
        self._stop_listener = threading.Event()
        self._listener_thread = None

    def _on_training_start(self) -> None:
        self._listener_thread = threading.Thread(
            target=self._listen_for_key, daemon=True
        )
        self._listener_thread.start()
        if self.verbose > 0:
            print("[PauseResume] Press 'p' to pause/resume training.")

    def _listen_for_key(self) -> None:
        """Background thread: read raw keypresses from stdin."""
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop_listener.is_set():
                    # Use select to avoid blocking forever
                    import select

                    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if ready:
                        ch = sys.stdin.read(1)
                        if ch == "p":
                            if self._paused.is_set():
                                self._paused.clear()
                            else:
                                self._paused.set()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # If stdin is not a terminal (e.g. piped input), disable the feature
            if self.verbose > 0:
                print("[PauseResume] stdin is not a terminal, pause disabled.")

    def _on_step(self) -> bool:
        if self._paused.is_set():
            if self.verbose > 0:
                print(
                    f"\n[PAUSED] Training paused at step {self.num_timesteps}. "
                    "Press 'p' to resume."
                )
            while self._paused.is_set():
                time.sleep(0.5)
            if self.verbose > 0:
                print(f"[RESUMED] Training resumed at step {self.num_timesteps}.")
        return True

    def _on_training_end(self) -> None:
        self._stop_listener.set()
