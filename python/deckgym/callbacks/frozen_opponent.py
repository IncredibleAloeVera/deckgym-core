#!/usr/bin/env python3
"""
FrozenOpponentCallback - Periodically updates the frozen opponent model.

Updates happen every N rollouts (not timesteps) for more predictable behavior.
Exports policy to ONNX for fast Rust-side inference when using BatchedDeckGymEnv.
"""

import os
import gc
import tempfile
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO


class FrozenOpponentCallback(BaseCallback):
    """
    Periodically updates the frozen opponent with current agent weights.

    Updates happen every N rollouts (not timesteps) for more predictable behavior.
    Exports policy to ONNX for fast Rust-side inference.
    """

    def __init__(
        self,
        env,
        n_envs: int = 1,
        update_every_n_rollouts: int = 8,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.n_envs = n_envs
        self.update_every_n_rollouts = update_every_n_rollouts
        self.rollout_count = 0
        self._frozen_model = None

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout collection."""
        self.rollout_count += 1

        if self.rollout_count % self.update_every_n_rollouts == 0:
            self._update_frozen_opponent()

    def _on_step(self) -> bool:
        return True

    def _update_frozen_opponent(self):
        """Export current model to ONNX and set as opponent."""
        # Import here to avoid circular imports
        from deckgym.envs.batched import BatchedDeckGymEnv

        # Clean up old frozen model
        if self._frozen_model is not None:
            del self._frozen_model
            gc.collect()

        if isinstance(self.env, BatchedDeckGymEnv):
            # BatchedDeckGymEnv: export to ONNX for Rust-side inference
            try:
                from deckgym.onnx_export import export_policy_to_onnx

                # Export current policy to ONNX file
                onnx_path = os.path.join(tempfile.gettempdir(), "frozen_opponent.onnx")
                export_policy_to_onnx(self.model, onnx_path, validate=False)

                # Set the ONNX model as opponent in VecGame
                device = getattr(self.env.config, "pfsp_opponent_device", "cuda")
                self.env.vec_game.set_onnx_opponent(
                    onnx_path, deterministic=False, device=device
                )

                if self.verbose > 0:
                    print(
                        f"[Rollout {self.rollout_count}] Updated frozen opponent (ONNX)"
                    )
            except AttributeError as e:
                print(f"[WARNING] ONNX opponent not available: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to set ONNX opponent: {e}")
        else:
            # CPU-based frozen model for non-batched envs
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "frozen_model")
                self.model.save(path)
                self._frozen_model = MaskablePPO.load(path, device="cpu")

            if self.n_envs == 1:
                inner_env = (
                    self.env.env.env if hasattr(self.env.env, "env") else self.env.env
                )
                inner_env.set_opponent_model(self._frozen_model)
            else:
                for i in range(self.n_envs):
                    self.env.envs[i].env.env.set_opponent_model(self._frozen_model)

            if self.verbose > 0:
                print(f"[Rollout {self.rollout_count}] Updated frozen opponent (CPU)")
