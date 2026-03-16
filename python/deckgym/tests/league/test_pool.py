#!/usr/bin/env python3
import unittest
import tempfile
import shutil
from pathlib import Path
from deckgym.league.pool import OpponentPool


class TestOpponentPool(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.pool = OpponentPool(pool_size=3, checkpoint_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_add_remove_opponent(self):
        name = "test_opp"
        data = {"is_baseline": False, "added_at_step": 100}
        self.pool.add_opponent(name, data)

        self.assertTrue(self.pool.contains(name))
        self.assertEqual(self.pool.get_data(name), data)
        self.assertEqual(self.pool.total_count, 1)
        self.assertEqual(self.pool.model_count, 1)

        removed = self.pool.remove_opponent(name)
        self.assertEqual(removed, data)
        self.assertFalse(self.pool.contains(name))
        self.assertEqual(self.pool.total_count, 0)

    def test_statistics_reset(self):
        name = "opp1"
        self.pool.add_opponent(name, {"wins": 10, "losses": 5, "draws": 2})
        self.pool.reset_statistics()

        data = self.pool.get_data(name)
        self.assertEqual(data["wins"], 0)
        self.assertEqual(data["losses"], 0)
        self.assertEqual(data["draws"], 0)

    def test_update_results(self):
        name = "opp1"
        self.pool.add_opponent(name, {"wins": 0, "losses": 0, "draws": 0})

        results = [
            (name, "agent_win"),  # opp loss
            (name, "opp_win"),  # opp win
            (name, "draw"),
            (name, "agent_win"),  # opp loss
        ]
        self.pool.update_results(results)

        data = self.pool.get_data(name)
        self.assertEqual(data["wins"], 1)
        self.assertEqual(data["losses"], 2)
        self.assertEqual(data["draws"], 1)

    def test_eviction_candidates(self):
        # Pool size is 3
        o1 = {
            "wins": 10,
            "losses": 0,
            "draws": 0,
            "added_at_step": 100,
            "is_baseline": False,
        }  # Hardest (WR 11/12 ~ 91%)
        o2 = {
            "wins": 0,
            "losses": 10,
            "draws": 0,
            "added_at_step": 200,
            "is_baseline": False,
        }  # Easiest (WR 1/12 ~ 8%)
        o3 = {
            "wins": 5,
            "losses": 5,
            "draws": 0,
            "added_at_step": 300,
            "is_baseline": False,
        }  # Middle (WR 6/12 ~ 50%)
        b1 = {
            "wins": 10,
            "losses": 10,
            "draws": 0,
            "added_at_step": 50,
            "is_baseline": True,
        }  # Baseline

        self.pool.add_opponent("o1", o1)
        self.pool.add_opponent("o2", o2)
        self.pool.add_opponent("o3", o3)
        self.pool.add_opponent("b1", b1)

        candidates = self.pool.get_eviction_candidates()

        # Should only contain o1, o2, o3 (no b1)
        self.assertEqual(len(candidates), 3)
        # Should be sorted by winrate (lowest first). o2 is easiest.
        self.assertEqual(candidates[0][0], "o2")
        self.assertEqual(candidates[1][0], "o3")
        self.assertEqual(candidates[2][0], "o1")

    def test_eviction_tie_break_age(self):
        # Same winrate, different age
        o1 = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "added_at_step": 100,
            "is_baseline": False,
        }
        o2 = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "added_at_step": 50,
            "is_baseline": False,
        }

        self.pool.add_opponent("o1", o1)
        self.pool.add_opponent("o2", o2)

        candidates = self.pool.get_eviction_candidates()
        # Tie in WR (both 0.5), so oldest first (o2)
        self.assertEqual(candidates[0][0], "o2")
        self.assertEqual(candidates[1][0], "o1")


if __name__ == "__main__":
    unittest.main()
