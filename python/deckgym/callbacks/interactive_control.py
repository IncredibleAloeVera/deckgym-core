#!/usr/bin/env python3
"""
Interactive Control Callback for Training.

Controls during training:
  p → toggle pause/resume
  e → clean exit: finish current PPO optimisation phase, save checkpoint, quit
  q → brutal exit: immediate KeyboardInterrupt (equivalent to Ctrl+C)

Uses a background thread for non-blocking keyboard input.
"""

import os
import sys
import time
import threading
from stable_baselines3.common.callbacks import BaseCallback


class InteractiveControlCallback(BaseCallback):
    """
    Callback that listens for keypresses during training:
    - 'p': toggle pause/resume
    - 'e': clean exit — finish current rollout+optimisation cycle, save checkpoint, stop
    - 'q': brutal exit — immediate KeyboardInterrupt
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints", verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self._paused = threading.Event()
        self._clean_exit = (
            threading.Event()
        )  # set when 'e' pressed: finish rollout+PPO then save
        self._stop_now = (
            threading.Event()
        )  # set after saving: abort next rollout to exit loop
        self._stop_listener = threading.Event()
        self._listener_thread = None

    def _on_training_start(self) -> None:
        self._listener_thread = threading.Thread(
            target=self._listen_for_key, daemon=True
        )
        self._listener_thread.start()
        if self.verbose > 0:
            print(
                "[InteractiveControl] Press 'p' to pause/resume, 'e' for clean exit, 'q' to quit immediately."
            )

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
                    import select

                    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if ready:
                        ch = sys.stdin.read(1)
                        if ch == "p":
                            if self._paused.is_set():
                                self._paused.clear()
                            else:
                                self._paused.set()
                        elif ch == "e":
                            if self.verbose > 0:
                                print(
                                    f"\n[InteractiveControl] Clean exit requested at step {self.num_timesteps}. "
                                    "Will save after current optimisation phase..."
                                )
                            self._clean_exit.set()
                        elif ch == "q":
                            if self.verbose > 0:
                                print(
                                    f"\n[InteractiveControl] Brutal exit requested at step {self.num_timesteps}."
                                )
                            raise KeyboardInterrupt("User pressed 'q' for brutal exit")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except KeyboardInterrupt:
            raise
        except Exception:
            # If stdin is not a terminal (e.g. piped input), disable the feature
            if self.verbose > 0:
                print(
                    "[InteractiveControl] stdin is not a terminal, pause/exit disabled."
                )

    def _on_step(self) -> bool:
        # Handle pause
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

        # Phase 2 of clean exit: PPO has run, model saved — abort this rollout now.
        if self._stop_now.is_set():
            return False

        return True

    def _on_rollout_end(self) -> None:
        """Nothing to do here — PPO has NOT optimised yet at this point."""
        pass

    def _on_rollout_start(self) -> None:
        """Called at the start of each new rollout, i.e. AFTER the previous PPO optimisation.

        SB3 learn() loop:
            collect_rollouts()  ← _on_rollout_start / _on_stepxN / _on_rollout_end
            self.train()        ← PPO optimisation
            collect_rollouts()  ← _on_rollout_start fires HERE (post-optimisation)

        So if _clean_exit is set: save now (weights are fresh), then set _stop_now
        so that the very first _on_step of this rollout returns False, making
        collect_rollouts() return False and breaking the learn() loop cleanly.
        """
        if self._clean_exit.is_set():
            self._save_exit_checkpoint()
            self._stop_now.set()

    def _save_exit_checkpoint(self) -> None:
        """Save model with exact step count in the standard checkpoint format."""
        if self.model is None:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"rl_bot_{self.num_timesteps}_steps")
        self.model.save(path)
        if self.verbose > 0:
            print(f"\n[InteractiveControl] ✓ Model saved to {path}.zip")

    def _on_training_end(self) -> None:
        self._stop_listener.set()
