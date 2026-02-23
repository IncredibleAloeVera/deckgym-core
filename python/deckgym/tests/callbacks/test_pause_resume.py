#!/usr/bin/env python3
"""Tests for PauseResumeCallback."""
import threading
import time
import unittest
from unittest.mock import MagicMock

from deckgym.callbacks.pause_resume import PauseResumeCallback


class TestPauseResumeCallback(unittest.TestCase):
    def setUp(self):
        self.callback = PauseResumeCallback(verbose=0)
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


if __name__ == "__main__":
    unittest.main()
