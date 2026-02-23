#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock, patch
from deckgym.league.bridge import LeagueBridge


class TestLeagueBridge(unittest.TestCase):
    def setUp(self):
        # Mocking the Rust-side vec_game
        self.mock_game = MagicMock()
        self.mock_env = MagicMock()
        self.mock_env.vec_game = self.mock_game
        self.bridge = LeagueBridge(env=self.mock_env, device="cuda", verbose=1)

    def test_export_model(self):
        mock_model = MagicMock()
        with patch("deckgym.league.bridge.export_policy_to_onnx") as mock_export:
            path = self.bridge.export_model(mock_model, "test_name")
            self.assertIn("league_pool_test_name.onnx", path)
            mock_export.assert_called_once()

    def test_add_onnx_to_rust(self):
        self.bridge.add_onnx_to_rust("m1", "/path/to/m1.onnx")
        self.mock_game.add_onnx_to_pool.assert_called_with(
            "m1", "/path/to/m1.onnx", False, "cuda"
        )

    def test_add_baseline_to_rust(self):
        self.bridge.add_baseline_to_rust("b1", "e2")
        self.mock_game.add_baseline_to_pool.assert_called_with("b1", "e2")

    def test_remove_from_rust(self):
        self.bridge.remove_from_rust("m1")
        self.mock_game.remove_onnx_from_pool.assert_called_with("m1")

    def test_clear_rust_pool(self):
        self.bridge.clear_rust_pool()
        self.mock_game.clear_onnx_pool.assert_called_once()

    def test_assign_to_env(self):
        self.bridge.assign_to_env(0, "m1")
        self.mock_game.set_env_opponent.assert_called_with(0, "m1")


if __name__ == "__main__":
    unittest.main()
