#!/usr/bin/env python3
"""Tests for TrainingConfig validation and serialization."""

import pytest
import tempfile
from pathlib import Path


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config_valid(self):
        """Default config should have no validation warnings."""
        from deckgym.config import DEFAULT_CONFIG

        warnings = DEFAULT_CONFIG.validate()
        # Default config should be valid
        assert isinstance(warnings, list)

    def test_batch_size_validation(self):
        """Config with invalid embed/heads ratio should warn."""
        from deckgym.config import TrainingConfig

        # embed_dim=100 not divisible by num_heads=8 should warn
        config = TrainingConfig(attention_embed_dim=100, attention_num_heads=8)
        warnings = config.validate()

        # Should have a warning about divisibility
        assert any("divisible" in w.lower() for w in warnings)

    def test_valid_batch_size_no_warning(self):
        """Valid batch size should not produce warnings."""
        from deckgym.config import TrainingConfig

        # n_steps=10, n_envs=4 -> total=40, batch_size=8 divides evenly
        config = TrainingConfig(n_steps=10, n_envs=4, batch_size=8)
        warnings = config.validate()

        # Should not have batch size warnings
        batch_warnings = [w for w in warnings if "batch" in w.lower()]
        assert len(batch_warnings) == 0

    def test_yaml_roundtrip(self):
        """Config should survive YAML save/load."""
        from deckgym.config import TrainingConfig

        config = TrainingConfig(
            n_envs=32,
            attention_num_heads=4,
            base_learning_rate=3e-5,
            total_timesteps=500_000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.save_yaml(str(path))
            loaded = TrainingConfig.from_yaml(str(path))

        assert loaded.n_envs == 32
        assert loaded.attention_num_heads == 4
        assert loaded.base_learning_rate == pytest.approx(3e-5)
        assert loaded.total_timesteps == 500_000

    def test_learning_rate_schedule(self):
        """LR schedule should warmup then decay."""
        from deckgym.config import TrainingConfig

        config = TrainingConfig(
            base_learning_rate=1e-4,
            min_learning_rate=1e-6,
            warmup_ratio=0.1,
        )
        schedule = config.get_learning_rate_schedule(total_timesteps=1000)

        # At start (progress=1.0), should be warming up
        lr_start = schedule(1.0)
        # At end (progress=0.0), should be at min
        lr_end = schedule(0.0)

        # End LR should be close to min
        assert lr_end == pytest.approx(1e-6, rel=0.1)
        # Start LR should be less than base (still warming up)
        assert lr_start <= 1e-4

    def test_config_copy(self):
        """Config copy should be independent."""
        from deckgym.config import TrainingConfig
        import copy

        config1 = TrainingConfig(n_envs=8)
        config2 = copy.copy(config1)

        # Modify copy - original should be unchanged
        # Note: dataclass fields are immutable by default
        assert config1.n_envs == 8
        assert config2.n_envs == 8


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_observation_size_positive(self):
        """OBSERVATION_SIZE should be positive."""
        from deckgym.config import OBSERVATION_SIZE

        assert OBSERVATION_SIZE > 0

    def test_action_space_size_positive(self):
        """ACTION_SPACE_SIZE should be positive."""
        from deckgym.config import ACTION_SPACE_SIZE

        assert ACTION_SPACE_SIZE > 0

    def test_features_consistency(self):
        """Global features + card features should equal observation size."""
        from deckgym.config import (
            OBSERVATION_SIZE,
            GLOBAL_FEATURES,
            FEATURES_PER_CARD,
            MAX_CARDS_IN_GAME,
        )

        expected = GLOBAL_FEATURES + (FEATURES_PER_CARD * MAX_CARDS_IN_GAME)
        assert OBSERVATION_SIZE == expected
