#!/usr/bin/env python3
"""Tests for OpponentSelector."""

import unittest
from unittest.mock import MagicMock
import numpy as np
from deckgym.league.selector import OpponentSelector
from deckgym.league.pool import OpponentPool


class TestOpponentSelector(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock(spec=OpponentPool)
        self.curriculum = [(0, ["v"]), (100, ["v", "e2"]), (200, ["e2", "er"])]
        self.selector = OpponentSelector(
            pool=self.pool,
            priority_exponent=1.0,
            baseline_curriculum=self.curriculum,
            baseline_max_allocation=0.25,
            n_envs=8,
        )

    def test_baseline_envs_count_no_baselines(self):
        """No baselines = 0 baseline envs."""
        self.selector.current_baselines = []
        self.assertEqual(self.selector.baseline_envs_count, 0)

    def test_baseline_envs_count_with_baselines(self):
        """With baselines, should allocate envs."""
        self.selector.current_baselines = ["v"]
        # 8 envs * 0.25 / 1 baseline = 2 envs per baseline, capped at 8*0.25=2
        self.assertEqual(self.selector.baseline_envs_count, 2)

    def test_baseline_envs_count_multiple_baselines(self):
        """Multiple baselines share the allocation."""
        self.selector.current_baselines = ["v", "e2"]
        # 8 envs * 0.25 = 2 max, min(2*2, 2) = 2
        self.assertEqual(self.selector.baseline_envs_count, 2)

    def test_update_curriculum_stage_0(self):
        """At step 50, should be in stage 0."""
        added, removed = self.selector.update_curriculum(50)
        self.assertEqual(self.selector.current_baselines, ["v"])
        self.assertEqual(added, ["v"])
        self.assertEqual(removed, [])

    def test_update_curriculum_stage_1(self):
        """At step 150, should be in stage 1."""
        # First set stage 0
        self.selector.update_curriculum(50)

        # Now move to stage 1
        added, removed = self.selector.update_curriculum(150)
        self.assertEqual(set(self.selector.current_baselines), {"v", "e2"})
        self.assertEqual(added, ["e2"])
        self.assertEqual(removed, [])

    def test_update_curriculum_stage_2(self):
        """At step 250, should be in stage 2."""
        # Set up previous stages
        self.selector.update_curriculum(50)
        self.selector.update_curriculum(150)

        # Move to stage 2
        added, removed = self.selector.update_curriculum(250)
        self.assertEqual(set(self.selector.current_baselines), {"e2", "er"})
        self.assertEqual(added, ["er"])
        self.assertEqual(removed, ["v"])

    def test_get_priorities(self):
        """Test PFSP priority calculation."""
        self.pool.opponents = {
            "m1": {
                "wins": 10,
                "losses": 0,
                "draws": 0,
                "is_baseline": False,
            },  # High opp winrate -> high priority
            "m2": {
                "wins": 0,
                "losses": 10,
                "draws": 0,
                "is_baseline": False,
            },  # Low opp winrate -> low priority
            "b1": {"wins": 5, "losses": 5, "draws": 0, "is_baseline": True},
        }

        prios = self.selector.get_priorities(exclude_baselines=True)

        self.assertIn("m1", prios)
        self.assertIn("m2", prios)
        self.assertNotIn("b1", prios)

        # m1 has higher opp winrate so should have higher priority
        self.assertGreater(prios["m1"], prios["m2"])

    def test_get_priorities_with_exponent(self):
        """Test that priority exponent affects priorities."""
        self.pool.opponents = {
            # With Laplace smoothing, these become (11/12)^exp and (6/12)^exp
            "m1": {
                "wins": 10,
                "losses": 0,
                "draws": 0,
                "is_baseline": False,
            },  # Strong opp
            "m2": {
                "wins": 5,
                "losses": 5,
                "draws": 0,
                "is_baseline": False,
            },  # Medium opp
        }

        # With exp = 1.0
        self.selector.priority_exponent = 1.0
        prios_1 = self.selector.get_priorities(exclude_baselines=True)
        ratio_1 = prios_1["m1"] / prios_1["m2"]

        # With exp = 2.0 - should increase the ratio (focus more on hard opponents)
        self.selector.priority_exponent = 2.0
        prios_2 = self.selector.get_priorities(exclude_baselines=True)
        ratio_2 = prios_2["m1"] / prios_2["m2"]

        # Higher exponent should increase the ratio between priorities
        self.assertGreater(ratio_2, ratio_1)

    def test_select_opponent_pfsp(self):
        """Test PFSP selection returns valid opponent."""
        self.pool.opponents = {
            "m1": {"wins": 100, "losses": 0, "draws": 0, "is_baseline": False},
            "m2": {"wins": 0, "losses": 100, "draws": 0, "is_baseline": False},
        }

        # With high wins, m1 should be selected more often
        selections = [self.selector.select_opponent_pfsp() for _ in range(100)]
        self.assertGreater(selections.count("m1"), selections.count("m2"))

    def test_select_for_env_baseline(self):
        """Baseline envs should get baseline opponents."""
        self.selector.current_baselines = ["e2"]
        self.pool.opponents = {
            "baseline_e2": {"is_baseline": True},
            "m1": {"wins": 0, "losses": 0, "draws": 0, "is_baseline": False},
        }

        # Env 0 should get baseline
        selected = self.selector.select_for_env(0)
        self.assertEqual(selected, "baseline_e2")

    def test_select_for_env_pfsp(self):
        """Non-baseline envs should get PFSP selection."""
        self.selector.current_baselines = ["e2"]
        self.pool.opponents = {
            "baseline_e2": {"is_baseline": True},
            "m1": {"wins": 0, "losses": 0, "draws": 0, "is_baseline": False},
        }

        # Env 3 (beyond baseline count) should get PFSP
        selected = self.selector.select_for_env(3)
        # Should be m1 since it's the only non-baseline
        self.assertEqual(selected, "m1")


if __name__ == "__main__":
    unittest.main()
