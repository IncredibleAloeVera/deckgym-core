#!/usr/bin/env python3
"""
InterpretabilityCallback — Logs interpretability metrics to TensorBoard.

Three metric groups are recorded at the end of every rollout:

1. Relative entropy (normalised)
   H(π) / log(|valid_actions|)  — comparable across steps regardless of
   how many actions are valid at that state.
   Key: interpretability/relative_entropy_mean

2. Per-action-category frequencies
   Fraction of agent steps that fell into each semantic category:
     end_turn   (action 0)
     attack     (1–2)
     retreat    (3–5)
     ability    (6–9)
     energy     (10–13)
     bench      (14–16)
     hand       (30–129) — play a card from hand
     resolution (130–168)
     other      (remaining indices)
   Keys: interpretability/freq_<category>

3. Value head statistics per episode outcome
   For each episode that ended during the rollout, the critic's value
   estimate at the *last* step is binned by outcome (win/loss/draw).
   Keys: interpretability/value_mean_{win,loss,draw}
         interpretability/value_std_{win,loss,draw}
         interpretability/value_n_{win,loss,draw}
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# ------------------------------------------------------------------
# Action-category boundaries (see RL_ARCHITECTURE.md, action_mask.rs)
# ------------------------------------------------------------------
ACTION_CATEGORIES: Dict[str, range] = {
    "end_turn":   range(0, 1),
    "attack":     range(1, 3),
    "retreat":    range(3, 6),
    "ability":    range(6, 10),
    "energy":     range(10, 14),
    "bench":      range(14, 17),
    # 17–29: DiscardFossil, Heal, AttachFromDiscard, FlipCoin → "other"
    "hand":       range(30, 130),
    "resolution": range(130, 169),
}

# Build a flat lookup array for speed: index → category name.
_ACTION_SPACE_SIZE = 179
_ACTION_TO_CATEGORY = ["other"] * _ACTION_SPACE_SIZE
for _cat, _rng in ACTION_CATEGORIES.items():
    for _idx in _rng:
        if _idx < _ACTION_SPACE_SIZE:
            _ACTION_TO_CATEGORY[_idx] = _cat
_ACTION_TO_CATEGORY = tuple(_ACTION_TO_CATEGORY)  # immutable for safety


class InterpretabilityCallback(BaseCallback):
    """
    Logs interpretability metrics every rollout without touching the model graph.

    Works with both BatchedDeckGymEnv (vectorised) and single-env setups.
    All metrics are averaged over the full rollout and logged via the SB3 logger
    so they appear in TensorBoard alongside the standard PPO keys.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        # Rollout buffers — reset at _on_rollout_start
        self._entropy_buffer: List[float] = []
        self._category_counts: Dict[str, int] = defaultdict(int)
        self._total_steps: int = 0

        # (value_estimate, final_reward) for each finished episode
        self._episode_value_reward: List[tuple] = []

        # Latest value estimates per env (updated each step)
        self._last_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    def _on_rollout_start(self) -> None:
        """Reset all per-rollout accumulators."""
        self._entropy_buffer.clear()
        self._category_counts = defaultdict(int)
        self._total_steps = 0
        self._episode_value_reward.clear()
        self._last_values = None

    def _on_rollout_end(self) -> None:
        """Flush all buffered metrics to the SB3 logger."""
        if not self.logger:
            return

        # 1 — Relative entropy
        if self._entropy_buffer:
            self.logger.record(
                "interpretability/relative_entropy_mean",
                float(np.mean(self._entropy_buffer)),
            )

        # 2 — Per-category action frequencies
        total = max(1, self._total_steps)
        all_cats = list(ACTION_CATEGORIES.keys()) + ["other"]
        for cat in all_cats:
            self.logger.record(
                f"interpretability/freq_{cat}",
                self._category_counts.get(cat, 0) / total,
            )

        # 3 — Value head stats per outcome
        if self._episode_value_reward:
            buckets: Dict[str, List[float]] = {"win": [], "loss": [], "draw": []}
            for val, rew in self._episode_value_reward:
                if rew > 0:
                    buckets["win"].append(val)
                elif rew < 0:
                    buckets["loss"].append(val)
                else:
                    buckets["draw"].append(val)

            for outcome, vals in buckets.items():
                if vals:
                    arr = np.array(vals, dtype=np.float32)
                    self.logger.record(
                        f"interpretability/value_mean_{outcome}", float(arr.mean())
                    )
                    self.logger.record(
                        f"interpretability/value_std_{outcome}",
                        float(arr.std()) if len(arr) > 1 else 0.0,
                    )
                    self.logger.record(
                        f"interpretability/value_n_{outcome}", len(arr)
                    )

    # ------------------------------------------------------------------
    # Per-step collection
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        """Collect entropy, action category, and episode value each step."""
        try:
            self._collect_entropy_and_categories()
            self._collect_episode_values()
        except Exception:
            # Never crash training due to a metrics callback
            pass
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_entropy_and_categories(self) -> None:
        """Compute relative entropy and tally action categories for this step."""
        locs = self.locals
        obs_tensor: Optional[torch.Tensor] = locs.get("obs_tensor")
        actions = locs.get("actions")
        # MaskablePPO stores the mask in the rollout buffer; retrieve from locs
        action_masks = locs.get("action_masks")

        if obs_tensor is None or actions is None:
            return

        with torch.no_grad():
            # evaluate_actions returns (values, log_probs, entropy)
            # For MaskablePPO, we need the distribution — use forward() instead.
            # policy.get_distribution gives us the masked distribution.
            dist = self.model.policy.get_distribution(
                obs_tensor, action_masks=action_masks
            )
            # probs shape: (n_envs, action_space_size)
            probs = dist.distribution.probs  # type: ignore[attr-defined]

            # --- Relative entropy ---
            n_valid = (probs > 0).float().sum(dim=-1).clamp(min=2)  # (n_envs,)
            max_entropy = torch.log(n_valid)  # log(|valid|)
            raw_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # H(π)
            rel_entropy = (raw_entropy / max_entropy).cpu().numpy()  # (n_envs,)
            self._entropy_buffer.extend(rel_entropy.tolist())

        # --- Action category frequencies ---
        if hasattr(actions, "cpu"):
            action_list = actions.cpu().numpy().astype(int).ravel().tolist()
        else:
            action_list = np.asarray(actions).astype(int).ravel().tolist()

        for a in action_list:
            cat = _ACTION_TO_CATEGORY[a] if 0 <= a < _ACTION_SPACE_SIZE else "other"
            self._category_counts[cat] += 1
            self._total_steps += 1

    def _collect_episode_values(self) -> None:
        """Record (value_estimate, final_reward) for each episode that just finished."""
        locs = self.locals
        dones = locs.get("dones")
        rewards = locs.get("rewards")
        values = locs.get("values")  # shape (n_envs, 1) tensor from PPO collect_rollouts

        if dones is None or rewards is None or values is None:
            return

        # Flatten
        if hasattr(dones, "cpu"):
            dones_arr = dones.cpu().numpy().ravel().astype(bool)
        else:
            dones_arr = np.asarray(dones).ravel().astype(bool)

        if hasattr(rewards, "cpu"):
            rews_arr = rewards.cpu().numpy().ravel().astype(float)
        else:
            rews_arr = np.asarray(rewards).ravel().astype(float)

        if hasattr(values, "cpu"):
            vals_arr = values.cpu().numpy().ravel().astype(float)
        else:
            vals_arr = np.asarray(values).ravel().astype(float)

        for i, done in enumerate(dones_arr):
            if done and i < len(vals_arr) and i < len(rews_arr):
                self._episode_value_reward.append((vals_arr[i], rews_arr[i]))
