#!/usr/bin/env python3
"""
Integration test: Mock training run.

Verifies that all components work together for a short training session.
"""

import pytest
import tempfile
from pathlib import Path


def get_action_masks(env):
    """Get action masks from wrapped environment."""
    # Navigate through wrappers to find action_masks
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "action_masks"):
            return current.action_masks()
        current = current.env
    if hasattr(current, "action_masks"):
        return current.action_masks()
    raise AttributeError("Could not find action_masks in environment chain")


class TestMockTraining:
    """Integration tests for short training runs."""

    @pytest.mark.slow
    def test_mock_training_mlp(self, sample_deck_pair, test_config):
        """Run a very short training session with MLP policy."""
        import numpy as np
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor

        from deckgym.envs import SelfPlayEnv
        from deckgym.callbacks import EpisodeMetricsCallback

        # Create mock deck loader
        class MockDeckLoader:
            def __init__(self, deck_pair):
                self._deck_pair = deck_pair

            def sample_pair(self):
                return self._deck_pair

        deck_loader = MockDeckLoader(sample_deck_pair)

        # Create minimal config for fast test
        from deckgym.config import TrainingConfig

        config = TrainingConfig(
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            total_timesteps=32,
            use_attention=False,  # MLP is faster
        )

        # Create environment
        def mask_fn(env):
            return env.action_masks()

        base_env = SelfPlayEnv(deck_loader, opponent_type="self", config=config)
        env = ActionMasker(base_env, mask_fn)
        env = Monitor(env)

        # Create model
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            verbose=0,
        )

        # Train for a few steps
        model.learn(total_timesteps=32, progress_bar=False)

        # Verify model can predict
        obs, _ = env.reset()
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)

        assert action is not None
        assert 0 <= action < base_env.action_space.n

    @pytest.mark.slow
    def test_mock_training_attention(self, sample_deck_pair, attention_config):
        """Run a very short training session with attention policy."""
        import numpy as np
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor

        from deckgym.envs import SelfPlayEnv
        from deckgym.models.extractors import create_attention_policy_kwargs

        # Create mock deck loader
        class MockDeckLoader:
            def __init__(self, deck_pair):
                self._deck_pair = deck_pair

            def sample_pair(self):
                return self._deck_pair

        deck_loader = MockDeckLoader(sample_deck_pair)

        # Create minimal config
        from deckgym.config import TrainingConfig

        config = TrainingConfig(
            n_envs=1,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            total_timesteps=16,
            use_attention=True,
            attention_embed_dim=64,  # Small for speed
            attention_num_heads=2,
            attention_num_layers=1,
        )

        # Create environment
        def mask_fn(env):
            return env.action_masks()

        base_env = SelfPlayEnv(deck_loader, opponent_type="self", config=config)
        env = ActionMasker(base_env, mask_fn)
        env = Monitor(env)

        # Create model with attention policy
        policy_kwargs = create_attention_policy_kwargs(config)

        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

        # Train for a few steps
        model.learn(total_timesteps=16, progress_bar=False)

        # Verify model can predict
        obs, _ = env.reset()
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)

        assert action is not None
        assert 0 <= action < base_env.action_space.n

    def test_callbacks_integration(self, sample_deck_pair, test_config):
        """Verify callbacks can be instantiated and attached."""
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor

        from deckgym.envs import SelfPlayEnv
        from deckgym.callbacks import EpisodeMetricsCallback, FrozenOpponentCallback

        # Create mock deck loader
        class MockDeckLoader:
            def __init__(self, deck_pair):
                self._deck_pair = deck_pair

            def sample_pair(self):
                return self._deck_pair

        deck_loader = MockDeckLoader(sample_deck_pair)

        # Create environment
        def mask_fn(env):
            return env.action_masks()

        base_env = SelfPlayEnv(deck_loader, opponent_type="self", config=test_config)
        env = ActionMasker(base_env, mask_fn)
        env = Monitor(env)

        # Create callbacks
        metrics_cb = EpisodeMetricsCallback(verbose=0)
        frozen_cb = FrozenOpponentCallback(
            env=env,
            n_envs=1,
            update_every_n_rollouts=1,
            verbose=0,
        )

        # Verify callbacks have expected methods
        assert hasattr(metrics_cb, "_on_step")
        assert hasattr(metrics_cb, "_on_rollout_end")
        assert hasattr(frozen_cb, "_on_step")
        assert hasattr(frozen_cb, "_update_frozen_opponent")

    def test_model_save_load(self, sample_deck_pair, test_config):
        """Verify model can be saved and loaded."""
        import tempfile
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor

        from deckgym.envs import SelfPlayEnv

        # Create mock deck loader
        class MockDeckLoader:
            def __init__(self, deck_pair):
                self._deck_pair = deck_pair

            def sample_pair(self):
                return self._deck_pair

        deck_loader = MockDeckLoader(sample_deck_pair)

        # Create environment
        def mask_fn(env):
            return env.action_masks()

        base_env = SelfPlayEnv(deck_loader, opponent_type="self", config=test_config)
        env = ActionMasker(base_env, mask_fn)
        env = Monitor(env)

        # Create and save model
        model = MaskablePPO("MlpPolicy", env, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            model.save(str(save_path))

            # Load model
            loaded_model = MaskablePPO.load(str(save_path), env=env)

            # Verify loaded model works
            obs, _ = env.reset()
            mask = get_action_masks(env)
            action, _ = loaded_model.predict(obs, action_masks=mask)

            assert action is not None
