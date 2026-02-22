#!/usr/bin/env python3
"""Tests for LeagueLogger."""
import unittest
from unittest.mock import MagicMock
import numpy as np
from deckgym.league.logger import LeagueLogger
from deckgym.league.pool import OpponentPool


class TestLeagueLogger(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock(spec=OpponentPool)
        self.sb3_logger = MagicMock()
        self.league_logger = LeagueLogger(
            pool=self.pool, logger=self.sb3_logger, verbose=0  # 0 to suppress output
        )

    def test_is_omniscient(self):
        """Test omniscient baseline detection."""
        # Omniscient (expectiminimax with depth)
        self.assertTrue(self.league_logger._is_omniscient("e2"))
        self.assertTrue(self.league_logger._is_omniscient("e3"))
        self.assertTrue(self.league_logger._is_omniscient("e4"))
        self.assertTrue(self.league_logger._is_omniscient("m1"))
        
        # Non-omniscient (er=evolution rusher, et=?)
        self.assertFalse(self.league_logger._is_omniscient("er"))
        self.assertFalse(self.league_logger._is_omniscient("et"))
        self.assertFalse(self.league_logger._is_omniscient("v"))
        self.assertFalse(self.league_logger._is_omniscient("w"))
        self.assertFalse(self.league_logger._is_omniscient("aa"))

    def test_is_onnx_model(self):
        """Test ONNX model baseline detection."""
        self.assertTrue(self.league_logger._is_onnx_model("o1"))
        self.assertTrue(self.league_logger._is_onnx_model("o2t"))
        self.assertTrue(self.league_logger._is_onnx_model("o1c"))
        
        self.assertFalse(self.league_logger._is_onnx_model("e2"))
        self.assertFalse(self.league_logger._is_onnx_model("v"))

    def test_get_winrate(self):
        """Test winrate calculation from pool data."""
        self.pool.get_data.return_value = {"wins": 10, "losses": 30, "draws": 0}
        
        # Agent wins = opponent losses, so WR = 30/40 = 0.75
        wr = self.league_logger._get_winrate("test")
        self.assertEqual(wr, 0.75)

    def test_get_winrate_no_data(self):
        """Test winrate default when no data."""
        self.pool.get_data.return_value = None
        
        wr = self.league_logger._get_winrate("test")
        self.assertEqual(wr, 0.5)

    def test_get_self_play_winrate(self):
        """Test self-play winrate calculation (non-baseline only)."""
        self.pool.opponents = {
            "m1": {"wins": 10, "losses": 30, "draws": 0, "is_baseline": False},  # WR = 0.75
            "m2": {"wins": 30, "losses": 10, "draws": 0, "is_baseline": False},  # WR = 0.25
            "b1": {"wins": 10, "losses": 0, "draws": 0, "is_baseline": True},    # Excluded
        }
        self.pool.get_data.side_effect = lambda name: self.pool.opponents.get(name)

        wr = self.league_logger.get_self_play_winrate()
        # Average of 0.75 and 0.25 = 0.5
        self.assertEqual(wr, 0.5)

    def test_get_pool_winrate(self):
        """Test global winrate (excludes omniscient)."""
        self.pool.opponents = {
            "m1": {"wins": 10, "losses": 40, "draws": 0, "is_baseline": False},  # Fair
            "baseline_e2": {
                "wins": 40, "losses": 10, "draws": 0, "is_baseline": True, 
                "baseline_code": "e2"
            },  # Omniscient - excluded
            "baseline_er": {
                "wins": 20, "losses": 20, "draws": 0, "is_baseline": True,
                "baseline_code": "er"
            },  # Fair baseline
        }
        self.pool.get_data.side_effect = lambda name: self.pool.opponents.get(name)

        wr = self.league_logger.get_pool_winrate()
        # m1: 40 wins, 50 games; baseline_er: 20 wins, 40 games
        # Total: 60 wins, 90 games = 66.7%
        expected = 60 / 90
        self.assertAlmostEqual(wr, expected, places=2)

    def test_log_metrics(self):
        """Test that log_metrics records to SB3 logger."""
        self.pool.opponents = {
            "m1": {"wins": 10, "losses": 40, "draws": 0, "is_baseline": False},
        }
        self.pool.get_data.side_effect = lambda name: self.pool.opponents.get(name)

        self.league_logger.log_metrics(rollout_count=100)

        # Should record global winrate
        self.sb3_logger.record.assert_any_call("pfsp/winrate_global", 0.8)

    def test_log_metrics_with_rollout_results(self):
        """Test log_metrics with per-rollout results."""
        self.pool.opponents = {
            "m1": {"wins": 0, "losses": 0, "draws": 0, "is_baseline": False},
        }
        rollout_results = {
            "m1": {"wins": 5, "losses": 15, "draws": 0}  # Agent WR = 15/20 = 0.75
        }

        self.league_logger.log_metrics(rollout_count=100, rollout_results=rollout_results)

        self.sb3_logger.record.assert_any_call("pfsp/winrate_global", 0.75)


if __name__ == "__main__":
    unittest.main()
