#!/usr/bin/env python3
"""Tests for InteractiveControlCallback."""
import threading
import time
import unittest
from unittest.mock import MagicMock

from deckgym.callbacks.interactive_control import InteractiveControlCallback


class TestInteractiveControlCallback(unittest.TestCase):
    def setUp(self):
        self.callback = InteractiveControlCallback(verbose=0)
        self.callback.model = MagicMock()
        self.callback.num_timesteps = 1000

    def test_on_step_returns_true_when_not_paused(self):
        """_on_step should return True when not paused."""
        self.assertTrue(self.callback._on_step())

    def test_pause_blocks_on_step(self):
        """When paused, _on_step should block until unpaused."""
        self.callback._paused.set()

        resumed = threading.Event()

        def run_step():
            self.callback._on_step()
            resumed.set()

        t = threading.Thread(target=run_step, daemon=True)
        t.start()

        # Give time for _on_step to enter the blocking loop
        time.sleep(0.3)
        self.assertFalse(resumed.is_set(), "_on_step should be blocking while paused")

        # Unpause
        self.callback._paused.clear()
        resumed.wait(timeout=2.0)
        self.assertTrue(resumed.is_set(), "_on_step should have returned after unpause")

    def test_on_step_returns_true_after_resume(self):
        """_on_step should return True after being unpaused."""
        self.callback._paused.set()

        result = [None]

        def run_step():
            result[0] = self.callback._on_step()

        t = threading.Thread(target=run_step, daemon=True)
        t.start()

        time.sleep(0.3)
        self.callback._paused.clear()
        t.join(timeout=2.0)
        self.assertTrue(result[0])

    def test_toggle_pause(self):
        """Toggling the pause event should work correctly."""
        self.assertFalse(self.callback._paused.is_set())

        self.callback._paused.set()
        self.assertTrue(self.callback._paused.is_set())

        self.callback._paused.clear()
        self.assertFalse(self.callback._paused.is_set())

    def test_on_training_end_stops_listener(self):
        """_on_training_end should signal the listener to stop."""
        self.callback._on_training_end()
        self.assertTrue(self.callback._stop_listener.is_set())

    def test_clean_exit_on_step_phase1(self):
        """_on_step returns True while only _clean_exit is set (rollout+PPO still running)."""
        self.callback._clean_exit.set()
        # Phase 1: current rollout must finish normally so PPO can run.
        self.assertTrue(self.callback._on_step())

    def test_clean_exit_on_step_phase2(self):
        """_on_step returns False once _stop_now is set (post-PPO abort)."""
        self.callback._clean_exit.set()
        self.callback._stop_now.set()
        self.assertFalse(self.callback._on_step())

    def test_clean_exit_saves_model_on_rollout_start(self):
        """_on_rollout_start saves the model and sets _stop_now when clean exit is requested.

        _on_rollout_start fires AFTER self.train() (PPO optimisation), so the
        saved weights are fully up-to-date.
        """
        import os

        # Not requested -> should not save
        self.callback._on_rollout_start()
        self.callback.model.save.assert_not_called()
        self.assertFalse(self.callback._stop_now.is_set())

        # Requested -> saves with precise step count and arms _stop_now
        self.callback._clean_exit.set()
        self.callback._on_rollout_start()

        expected_path = os.path.join(
            self.callback.checkpoint_dir, "rl_bot_1000_steps"
        )
        self.callback.model.save.assert_called_once_with(expected_path)
        self.assertTrue(self.callback._stop_now.is_set(), "_stop_now must be set to abort next rollout")

    def test_brutal_exit_raises_interrupt(self):
        """Pressing 'q' should immediately raise a KeyboardInterrupt in the listener thread."""
        import sys
        from unittest.mock import patch, MagicMock
        
        # Simulate 'q' being read from stdin
        with patch("sys.stdin.read", return_value="q"), \
             patch("select.select", return_value=([sys.stdin], [], [])), \
             patch("sys.stdin.fileno", return_value=0), \
             patch("termios.tcgetattr", return_value=MagicMock()), \
             patch("termios.tcsetattr"), \
             patch("tty.setcbreak"):
            
            # _listen_for_key should catch the inner KeyboardInterrupt and re-raise it
            with self.assertRaisesRegex(KeyboardInterrupt, "User pressed 'q'"):
                self.callback._listen_for_key()


if __name__ == "__main__":
    unittest.main()
