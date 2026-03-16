#!/usr/bin/env python3
"""Tests for environment modules (base, batched, self_play)."""

import pytest
import numpy as np


class TestDeckGymEnv:
    """Tests for the base DeckGymEnv."""

    def test_env_creation(self, sample_deck_pair):
        """Environment should be creatable with deck strings."""
        from deckgym.envs import DeckGymEnv

        deck_a, deck_b = sample_deck_pair
        env = DeckGymEnv(deck_a, deck_b)

        assert env is not None
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_env_reset(self, sample_deck_pair):
        """Reset should return valid observation and info."""
        from deckgym.envs import DeckGymEnv

        deck_a, deck_b = sample_deck_pair
        env = DeckGymEnv(deck_a, deck_b)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == env.observation_space.shape

    def test_env_step_valid_action(self, sample_deck_pair):
        """Step with valid action should return expected tuple."""
        from deckgym.envs import DeckGymEnv

        deck_a, deck_b = sample_deck_pair
        env = DeckGymEnv(deck_a, deck_b)
        env.reset()

        # Get valid actions
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        assert len(valid_actions) > 0, "Should have at least one valid action"

        action = valid_actions[0]
        obs, reward, done, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_masks_shape(self, sample_deck_pair):
        """Action masks should have correct shape."""
        from deckgym.envs import DeckGymEnv
        from deckgym.config import ACTION_SPACE_SIZE

        deck_a, deck_b = sample_deck_pair
        env = DeckGymEnv(deck_a, deck_b)
        env.reset()

        mask = env.action_masks()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == bool

    def test_at_least_one_valid_action(self, sample_deck_pair):
        """Should always have at least one valid action (EndTurn)."""
        from deckgym.envs import DeckGymEnv

        deck_a, deck_b = sample_deck_pair
        env = DeckGymEnv(deck_a, deck_b)
        env.reset()

        mask = env.action_masks()
        assert mask.any(), "Should have at least one valid action"


class TestSelfPlayEnv:
    """Tests for SelfPlayEnv training wrapper."""

    def test_selfplay_env_creation(self, mock_deck_loader, test_config):
        """SelfPlayEnv should be creatable with loader and config."""
        from deckgym.envs import SelfPlayEnv

        env = SelfPlayEnv(
            deck_loader=mock_deck_loader,
            opponent_type="self",
            config=test_config,
        )

        assert env is not None
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_selfplay_reset(self, mock_deck_loader, test_config):
        """Reset should return valid observation."""
        from deckgym.envs import SelfPlayEnv

        env = SelfPlayEnv(
            deck_loader=mock_deck_loader,
            opponent_type="self",
            config=test_config,
        )

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_set_opponent_type(self, mock_deck_loader, test_config):
        """Should be able to change opponent type."""
        from deckgym.envs import SelfPlayEnv

        env = SelfPlayEnv(
            deck_loader=mock_deck_loader,
            opponent_type="self",
            config=test_config,
        )

        env.set_opponent_type("e2")
        assert env.opponent_type == "e2"

        env.set_opponent_type("self")
        assert env.opponent_type == "self"

    def test_action_masks_available(self, mock_deck_loader, test_config):
        """SelfPlayEnv should provide action masks."""
        from deckgym.envs import SelfPlayEnv

        env = SelfPlayEnv(
            deck_loader=mock_deck_loader,
            opponent_type="self",
            config=test_config,
        )
        env.reset()

        mask = env.action_masks()

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.any()  # At least one valid action


class TestBatchedDeckGymEnv:
    """Tests for BatchedDeckGymEnv vectorized environment."""

    def test_batched_env_attributes(self):
        """BatchedDeckGymEnv should be importable and have expected interface."""
        from deckgym.envs import BatchedDeckGymEnv

        # Just test the class exists and has key methods
        assert hasattr(BatchedDeckGymEnv, "step_async")
        assert hasattr(BatchedDeckGymEnv, "step_wait")
        assert hasattr(BatchedDeckGymEnv, "reset")
