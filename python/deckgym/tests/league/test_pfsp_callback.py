#!/usr/bin/env python3
"""Tests for PFSPCallback."""
import unittest
from unittest.mock import MagicMock, patch
from deckgym.pfsp_callback import PFSPCallback


class TestPFSPCallback(unittest.TestCase):
    def setUp(self):
        self.mock_env = MagicMock()
        self.mock_env.config = MagicMock()
        self.mock_env.config.pfsp_opponent_device = "cuda"
        self.mock_env.config.pfsp_baseline_curriculum = [(0, ["v"])]
        self.mock_env.vec_game = MagicMock()

        self.callback = PFSPCallback(
            env=self.mock_env,
            n_envs=4,
            pool_size=5,
            add_to_pool_every_n_rollouts=10,
            verbose=0,  # Suppress output
        )
        # Mock SB3 model
        self.callback.model = MagicMock()
        self.callback._logger = MagicMock()

        # Ensure config returns values, not mocks, for comparisons
        self.mock_env.config.pfsp_min_winrate_to_add = 0.50
        self.mock_env.config.draw_reward = -0.5

    def test_on_training_start(self):
        """Test training start initializes pool mode."""
        with patch.object(self.callback, "_add_to_pool"):
            self.callback._on_training_start()

            # Should have set pool mode
            self.assertTrue(self.callback.pool_mode)
            # Rust bridge should have been called for initial opponents
            self.assertEqual(self.mock_env.vec_game.set_env_opponent.call_count, 4)

    def test_on_step_recording(self):
        """Test that episode results are recorded correctly."""
        # Setup env state
        self.callback.env_opponent_names = ["opp1", "opp1", "opp1", "opp1"]
        self.callback.skip_next_episode = [False] * 4

        # Mock env info with episode results
        self.callback.locals = {
            "infos": [
                {"episode": {"r": 1.0}},   # Agent win
                {"episode": {"r": -1.0}},  # Agent loss (opp win)
                {},                         # No episode info
                {"episode": {"r": -0.5}},  # Draw
            ]
        }

        self.callback._on_step()

        # Results should be recorded for env 0, 1, 3
        expected = [("opp1", "agent_win"), ("opp1", "opp_win"), ("opp1", "draw")]
        self.assertEqual(self.callback.episode_results, expected)

    def test_rollout_end_increments_counter(self):
        """Test that rollout end increments rollout count."""
        # Mock components
        self.callback.pool = MagicMock()
        self.callback.selector = MagicMock()
        self.callback.league_logger = MagicMock()
        self.callback.bridge = MagicMock()

        self.callback.rollout_count = 0
        self.callback.add_to_pool_every_n_rollouts = 10
        self.callback.episode_results = []

        self.callback.selector.update_curriculum.return_value = ([], [])

        self.callback._on_rollout_end()

        # Rollout count should increment
        self.assertEqual(self.callback.rollout_count, 1)

    def test_rollout_end_curriculum_update(self):
        """Test curriculum update is called."""
        # Mock components
        self.callback.pool = MagicMock()
        self.callback.selector = MagicMock()
        self.callback.league_logger = MagicMock()
        self.callback.bridge = MagicMock()

        self.callback.rollout_count = 9
        self.callback.add_to_pool_every_n_rollouts = 10
        self.callback.episode_results = [("m1", "agent_win")] * 20

        self.callback.selector.update_curriculum.return_value = ([], [])
        self.callback.league_logger.get_pool_winrate.return_value = 0.8

        with patch.object(self.callback, "_add_to_pool"):
            self.callback._on_rollout_end()

        # Curriculum should be updated
        self.callback.selector.update_curriculum.assert_called()

    def test_rollout_end_stats_update(self):
        """Test pool results are updated on rollout end."""
        # Mock components
        self.callback.pool = MagicMock()
        self.callback.selector = MagicMock()
        self.callback.league_logger = MagicMock()
        self.callback.bridge = MagicMock()

        self.callback.rollout_count = 9
        self.callback.add_to_pool_every_n_rollouts = 10
        self.callback.episode_results = [("m1", "agent_win")] * 20

        self.callback.selector.update_curriculum.return_value = ([], [])
        self.callback.league_logger.get_pool_winrate.return_value = 0.8

        with patch.object(self.callback, "_add_to_pool"):
            self.callback._on_rollout_end()

        # Pool update should be called
        self.callback.pool.update_results.assert_called_once()
        # Logger should log metrics
        self.callback.league_logger.log_metrics.assert_called_once()

    def test_rollout_end_add_to_pool(self):
        """Test add_to_pool is called at correct interval."""
        # Mock components
        self.callback.pool = MagicMock()
        self.callback.pool.model_count = 1
        self.callback.pool.pool_size = 5
        self.callback.selector = MagicMock()
        self.callback.league_logger = MagicMock()
        self.callback.bridge = MagicMock()

        self.callback.rollout_count = 9
        self.callback.add_to_pool_every_n_rollouts = 10
        self.callback.episode_results = [("m1", "agent_win")] * 20

        self.callback.selector.update_curriculum.return_value = ([], [])
        self.callback.league_logger.get_pool_winrate.return_value = 0.8

        with patch.object(self.callback, "_add_to_pool") as mock_add:
            self.callback._on_rollout_end()

            # Should trigger pool addition (every 10)
            mock_add.assert_called_once()


if __name__ == "__main__":
    unittest.main()
