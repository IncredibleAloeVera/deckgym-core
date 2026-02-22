#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) Callback.
Modular implementation delegating to specialized components.
"""

import gc
from typing import Dict, List, Optional
from stable_baselines3.common.callbacks import BaseCallback

from deckgym.batched_env import BatchedDeckGymEnv
from deckgym.config import (
    DEFAULT_CONFIG,
    PFSP_MIN_EPISODES_FOR_UPDATE,
)
from deckgym.league.pool import OpponentPool
from deckgym.league.selector import OpponentSelector
from deckgym.league.bridge import LeagueBridge
from deckgym.league.logger import LeagueLogger


class PFSPCallback(BaseCallback):
    """
    Prioritized Fictitious Self-Play callback.
    Modular implementation delegating to specialized components.
    """

    def __init__(
        self,
        env,
        n_envs: int = DEFAULT_CONFIG.n_envs,
        pool_size: int = DEFAULT_CONFIG.pfsp_pool_size,
        add_to_pool_every_n_rollouts: int = DEFAULT_CONFIG.pfsp_add_every_n_rollouts,
        priority_exponent: float = DEFAULT_CONFIG.pfsp_priority_exponent,
        checkpoint_dir: str = DEFAULT_CONFIG.pfsp_checkpoint_dir,
        # Baseline curriculum parameters (baseline_slots is now dynamic)
        baseline_max_allocation: float = DEFAULT_CONFIG.pfsp_baseline_max_allocation,
        baseline_curriculum: list = None,
        brutal_resume: bool = DEFAULT_CONFIG.brutal_resume,
        verbose: int = 1,
        **kwargs,
    ):
        super().__init__(verbose)
        self.env = env
        self.n_envs = n_envs
        self.add_to_pool_every_n_rollouts = add_to_pool_every_n_rollouts

        # Initialize modular components
        self.pool = OpponentPool(pool_size, checkpoint_dir)
        self.selector = OpponentSelector(
            self.pool,
            priority_exponent,
            baseline_curriculum or DEFAULT_CONFIG.pfsp_baseline_curriculum,
            baseline_max_allocation,
            n_envs,
            brutal_resume,
        )
        self.bridge = LeagueBridge(
            env,
            device=getattr(env.config, "pfsp_opponent_device", "trt"),
            verbose=verbose,
        )
        self.league_logger = LeagueLogger(self.pool, verbose=verbose)

        # State tracking
        self.rollout_count = 0
        self.episode_results = []
        self.env_opponent_names = [None] * n_envs
        self.skip_next_episode = [False] * n_envs
        self.pool_mode = False

        if self.verbose > 0:
            print(
                f"[PFSP] Modular initialized (pool={pool_size}, exp={priority_exponent})"
            )

    def _on_training_start(self) -> None:
        """Initialize league state and Rust pool."""
        self.league_logger.logger = self.logger

        # Initial curriculum update
        added, _ = self.selector.update_curriculum(self.num_timesteps)
        for bl_code in added:
            self.pool.add_opponent(
                f"baseline_{bl_code}",
                {
                    "path": None,
                    "baseline_code": bl_code,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "added_at_step": self.num_timesteps,
                    "is_baseline": True,
                },
            )

        # Add initial untrained model
        self._add_to_pool()

        # Initialize Rust pool
        self.bridge.clear_rust_pool()
        for name, data in self.pool.opponents.items():
            if data.get("is_baseline"):
                bl_code = data["baseline_code"]
                if bl_code.startswith("o"):
                    self._add_onnx_baseline_to_rust(name, bl_code)
                else:
                    self.bridge.add_baseline_to_rust(name, bl_code)
            else:
                self.bridge.add_onnx_to_rust(name, data["onnx_path"])

        # Assign initial opponents
        for i in range(self.n_envs):
            self._assign_opponent_to_env(i)

        self.pool_mode = True

    def _assign_opponent_to_env(self, env_idx: int):
        """Select an opponent and assign it via bridge."""
        name = self.selector.select_for_env(env_idx)
        if name:
            self.bridge.assign_to_env(env_idx, name)
            self.env_opponent_names[env_idx] = name

    def _flag_reassigned_envs(self, opponent_name: str):
        """Flag environments to skip next result when opponent changes mid-game."""
        for i, name in enumerate(self.env_opponent_names):
            if name == opponent_name:
                self.skip_next_episode[i] = True

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout collection."""
        self.rollout_count += 1

        # Periodic curriculum update
        if self.rollout_count % 10 == 0:
            added, removed = self.selector.update_curriculum(self.num_timesteps)
            for bl_code in added:
                name = f"baseline_{bl_code}"
                self.pool.add_opponent(
                    name,
                    {
                        "path": None,
                        "baseline_code": bl_code,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "added_at_step": self.num_timesteps,
                        "is_baseline": True,
                    },
                )
                # Check if it's an ONNX baseline (optimized path)
                if bl_code.startswith("o"):
                    self._add_onnx_baseline_to_rust(name, bl_code)
                else:
                    self.bridge.add_baseline_to_rust(name, bl_code)
            for bl_code in removed:
                name = f"baseline_{bl_code}"
                self.pool.remove_opponent(name)
                self.bridge.remove_from_rust(name)

        # Update statistics and log
        if len(self.episode_results) >= PFSP_MIN_EPISODES_FOR_UPDATE:
            # Aggregate rollout stats for detailed logging
            rollout_stats = {}
            for opp_name, result in self.episode_results:
                if opp_name not in rollout_stats:
                    rollout_stats[opp_name] = {"wins": 0, "losses": 0, "draws": 0}
                if result == "agent_win":
                    rollout_stats[opp_name]["losses"] += 1
                elif result == "opp_win":
                    rollout_stats[opp_name]["wins"] += 1
                elif result == "draw":
                    rollout_stats[opp_name]["draws"] += 1

            self.pool.update_results(self.episode_results)
            self.league_logger.log_metrics(self.rollout_count, rollout_stats)
            self.league_logger.log_detailed_info(rollout_stats)
            self.episode_results.clear()

        # Add current agent to pool
        # "All-Time" means cumulative winrate reset at each refresh 
        if self.rollout_count % self.add_to_pool_every_n_rollouts == 0:
            # Use All-Time (cumulative) winrate for the addition decision
            all_time_wr = self.league_logger.get_pool_winrate(rollout_results=None)
            min_wr_to_add = getattr(self.env.config, "pfsp_min_winrate_to_add", 0.50)

            if self.verbose > 0:
                print(f"\n[PFSP] --- Pool Refresh Check (Rollout {self.rollout_count}) ---")
                print(f"[PFSP] All-Time Pool WR: {all_time_wr:.1%} (Required: {min_wr_to_add:.1%})")

            # Check strictly >= min_wr_to_add before entering pool (do not bypass for initial pool population)
            if all_time_wr >= min_wr_to_add:
                self._add_to_pool()
                # Reset ALL statistics only when pool composition changes

            elif self.verbose > 0:
                print(f"[PFSP] Agent rejected: Pool Winrate {all_time_wr:.1%} is below required {min_wr_to_add:.1%}")
            # Reset all stats at each refresh (even if agent not added)
            self.pool.reset_statistics()
            self.pool.reset_total_statistics()

    def _on_step(self) -> bool:
        """Called at each env step - track outcomes and reassign."""
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue

            reward = info["episode"].get("r", 0)
            draw_reward = getattr(self.env.config, "draw_reward", -0.5)
            result = (
                "agent_win"
                if reward > 0
                else ("draw" if abs(reward - draw_reward) < 1e-4 else "opp_win")
            )

            opp_name = self.env_opponent_names[env_idx]
            if opp_name:
                if self.skip_next_episode[env_idx]:
                    self.skip_next_episode[env_idx] = False
                else:
                    self.episode_results.append((opp_name, result))

            self._assign_opponent_to_env(env_idx)
        return True

    def _add_to_pool(self):
        """Orchestrate saving and adding a new model to the league."""
        name = f"pfsp_{self.num_timesteps // 1000}k"
        path = self.pool.checkpoint_dir / f"{name}.zip"

        # Save model
        self.model.save(str(path))

        # Add to local pool
        data = {
            "path": str(path),
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "added_at_step": self.num_timesteps,
            "is_baseline": False,
        }
        self.pool.add_opponent(name, data)

        # Add to Rust via bridge
        onnx_path = self.bridge.export_model(self.model, name)
        data["onnx_path"] = onnx_path
        self.bridge.add_onnx_to_rust(name, onnx_path)

        # Eviction if pool full
        while self.pool.model_count > self.pool.pool_size:
            candidates = self.pool.get_eviction_candidates(exclude_names=[name])
            if candidates:
                weakest_name, wr, _ = candidates[0]
                self._flag_reassigned_envs(weakest_name)
                self.bridge.remove_from_rust(weakest_name)
                weakest_data = self.pool.remove_opponent(weakest_name)
                self.pool.cleanup_files(weakest_data)

                # Immediate reassignment for envs playing against evicted opponent
                for i, curr_name in enumerate(self.env_opponent_names):
                    if curr_name == weakest_name:
                        self._assign_opponent_to_env(i)

                if self.verbose > 0:
                    print(
                        f"[PFSP] Evicted weakest model: {weakest_name} (WR: {wr:.1%})"
                    )
            else:
                break

        if self.verbose > 0:
            print(f"[PFSP] Saved and added model: {name}")
        gc.collect()

    def _add_onnx_baseline_to_rust(self, name: str, code: str):
        """Resolve an ONNX baseline code to a path and add it to Rust ONNX pool."""
        path = self._resolve_onnx_code_to_path(code)
        if path:
            if self.verbose > 0:
                print(f"[PFSP] Resolved baseline boss '{code}' to {path}")
            self.bridge.add_onnx_to_rust(name, path)
        else:
            if self.verbose > 0:
                print(
                    f"[WARNING] Could not resolve ONNX baseline '{code}', falling back to sequential."
                )
            self.bridge.add_baseline_to_rust(name, code)

    def _resolve_onnx_code_to_path(self, code: str) -> Optional[str]:
        """Resolve a code like 'o1' to a path in models/."""
        import glob
        import os
        import onnx
        from deckgym.config import OBSERVATION_SIZE

        if not code.startswith("o"):
            return None

        # Search for .onnx files in models/ and subdirectories
        model_paths = glob.glob("models/*.onnx") + glob.glob("models/**/*.onnx")
        if not model_paths:
            return None

        # Sort by modification time (newest first)
        model_paths.sort(key=os.path.getmtime, reverse=True)

        # Parse index from code (e.g., 'o1' -> index 0, 'o2' -> index 1)
        try:
            # Extract number from 'o1', 'o1t', 'o1c', etc.
            import re
            match = re.search(r'o(\d+)', code)
            idx = int(match.group(1)) - 1 if match else 0
        except (ValueError, IndexError):
            idx = 0

        # Filter models by observation size
        valid_models = []
        for path in model_paths:
            try:
                m = onnx.load(path)
                # Check input dimension (usually 'observation' at index 0)
                input_dim = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
                if input_dim == OBSERVATION_SIZE:
                    valid_models.append(path)
                elif self.verbose > 1:
                    print(f"[PFSP] Skipping {path}: expected dim {OBSERVATION_SIZE}, got {input_dim}")
            except Exception as e:
                if self.verbose > 1:
                    print(f"[PFSP] Could not head ONNX {path}: {e}")

        if not valid_models:
            if self.verbose > 0:
                print(f"[PFSP WARNING] No ONNX models found with dim {OBSERVATION_SIZE} in models/")
            return None

        # Return the requested index (clamped to available models)
        idx = min(idx, len(valid_models) - 1)
        return str(valid_models[idx])
