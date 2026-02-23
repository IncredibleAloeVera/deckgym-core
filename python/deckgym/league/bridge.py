#!/usr/bin/env python3
"""
League Bridge for high-performance opponent execution.

Handles ONNX exports and interfaces with the Rust vec_game pool.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Any

from sb3_contrib import MaskablePPO
from deckgym.onnx_export import export_policy_to_onnx
from deckgym.batched_env import BatchedDeckGymEnv


class LeagueBridge:
    """
    Bridges the Python league state with the Rust-side execution engine.

    Responsibilities:
    - Exporting Stable-Baselines3 models to ONNX.
    - Synchronizing the opponent pool with the Rust vec_game.
    - Managing hardware devices (CUDA, CPU) for opponents.
    """

    def __init__(self, env: BatchedDeckGymEnv, device: str = "cuda", verbose: int = 1):
        self.env = env
        self.device = device
        self.verbose = verbose

        if not hasattr(self.env, "vec_game"):
            raise TypeError(
                "[LeagueBridge] Env must be a BatchedDeckGymEnv with vec_game support"
            )

    def export_model(self, model: MaskablePPO, name: str) -> str:
        """Export a model to ONNX and return the path."""
        onnx_path = os.path.join(tempfile.gettempdir(), f"league_pool_{name}.onnx")
        export_policy_to_onnx(model, onnx_path, validate=False)
        return onnx_path

    def add_onnx_to_rust(self, name: str, onnx_path: str):
        """Add an ONNX model to the Rust-side pool."""
        try:
            self.env.vec_game.add_onnx_to_pool(name, onnx_path, False, self.device)
            if self.verbose > 1:
                print(
                    f"[LeagueBridge] Added ONNX opponent '{name}' to Rust pool ({self.device})"
                )
        except Exception as e:
            if self.verbose > 0:
                print(f"[LeagueBridge WARNING] Failed to add ONNX {name}: {e}")

    def add_baseline_to_rust(self, name: str, code: str):
        """Add a built-in baseline bot to the Rust-side pool."""
        try:
            self.env.vec_game.add_baseline_to_pool(name, code)
            if self.verbose > 1:
                print(f"[LeagueBridge] Added baseline '{name}' ({code}) to Rust pool")
        except Exception as e:
            if self.verbose > 0:
                print(f"[LeagueBridge WARNING] Failed to add baseline {name}: {e}")

    def remove_from_rust(self, name: str):
        """Remove an opponent from the Rust-side pool."""
        try:
            self.env.vec_game.remove_onnx_from_pool(name)
        except Exception:
            pass

    def clear_rust_pool(self):
        """Clear all opponents from the Rust-side pool."""
        if hasattr(self.env.vec_game, "clear_onnx_pool"):
            self.env.vec_game.clear_onnx_pool()

    def assign_to_env(self, env_idx: int, opponent_name: str):
        """Set the opponent for a specific Rust-side environment."""
        try:
            self.env.vec_game.set_env_opponent(env_idx, opponent_name)
        except Exception as e:
            if self.verbose > 0:
                print(
                    f"[LeagueBridge WARNING] Failed to set opponent for env {env_idx}: {e}"
                )
