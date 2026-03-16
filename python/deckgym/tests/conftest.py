#!/usr/bin/env python3
"""
Shared pytest fixtures for deckgym tests.

Provides common test utilities and mock objects for unit tests.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path


@pytest.fixture
def test_config():
    """Minimal TrainingConfig for testing."""
    from deckgym.config import TrainingConfig

    return TrainingConfig(
        n_envs=1,
        n_steps=8,
        batch_size=8,
        total_timesteps=100,
        use_attention=False,  # Faster tests
    )


@pytest.fixture
def attention_config():
    """TrainingConfig with attention enabled for testing models."""
    from deckgym.config import TrainingConfig

    return TrainingConfig(
        n_envs=1,
        n_steps=8,
        batch_size=8,
        total_timesteps=100,
        use_attention=True,
        attention_embed_dim=64,  # Smaller for faster tests
        attention_num_heads=2,
        attention_num_layers=1,
    )


@pytest.fixture
def example_deck_dir():
    """Path to example decks directory."""
    return Path(__file__).parent.parent.parent / "example_decks"


@pytest.fixture
def sample_deck_pair(example_deck_dir):
    """Sample deck strings for testing."""
    # Use real decks from example_decks
    mewtwo_path = example_deck_dir / "mewtwoex.txt"
    zerakoko_path = example_deck_dir / "zerakoko.txt"

    if mewtwo_path.exists() and zerakoko_path.exists():
        deck_a = mewtwo_path.read_text()
        deck_b = zerakoko_path.read_text()
    else:
        # Fallback using proper deck format
        deck_a = """Energy: Psychic
1 A1 128
2 A1 129
2 A1 130
2 A1 131
2 A1 132
1 A1 223
2 A1 225
2 A1a 065
2 P-A 002
2 P-A 005
2 P-A 007
"""
        deck_b = deck_a  # Use same deck for both sides as fallback

    return deck_a, deck_b


@pytest.fixture
def mock_deck_loader(sample_deck_pair):
    """Mock deck loader for testing environments."""

    class MockDeckLoader:
        def __init__(self, deck_pair):
            self._deck_pair = deck_pair

        def sample_pair(self):
            return self._deck_pair

        def sample_deck(self):
            return self._deck_pair[0]

    return MockDeckLoader(sample_deck_pair)


@pytest.fixture
def mock_observation():
    """Mock observation tensor for testing."""
    from deckgym.config import OBSERVATION_SIZE

    obs = np.random.rand(OBSERVATION_SIZE).astype(np.float32)
    # Clamp to valid range
    return np.clip(obs, 0.0, 1.0)


@pytest.fixture
def mock_action_mask():
    """Mock action mask for testing."""
    from deckgym.config import ACTION_SPACE_SIZE

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[0] = True  # At least EndTurn valid
    # Add some random valid actions
    random_actions = np.random.choice(
        range(1, ACTION_SPACE_SIZE), size=5, replace=False
    )
    mask[random_actions] = True
    return mask


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
