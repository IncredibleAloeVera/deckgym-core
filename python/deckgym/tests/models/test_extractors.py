#!/usr/bin/env python3
"""Tests for attention-based feature extractors."""

import pytest
import torch
import numpy as np


class TestCardAttentionExtractor:
    """Tests for CardAttentionExtractor."""

    def test_extractor_creation(self, attention_config):
        """Extractor should be creatable with config."""
        from deckgym.models.extractors import CardAttentionExtractor
        from deckgym.config import OBSERVATION_SIZE
        from gymnasium import spaces

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        extractor = CardAttentionExtractor(obs_space, config=attention_config)

        assert extractor is not None
        assert extractor.features_dim > 0

    def test_forward_pass(self, attention_config):
        """Forward pass should produce correct output shape."""
        from deckgym.models.extractors import CardAttentionExtractor
        from deckgym.config import OBSERVATION_SIZE
        from gymnasium import spaces

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        extractor = CardAttentionExtractor(obs_space, config=attention_config)

        # Create batch of dummy observations
        batch_size = 4
        obs = torch.rand(batch_size, OBSERVATION_SIZE)

        features = extractor(obs)

        assert features.shape == (batch_size, extractor.features_dim)

    def test_attention_stats_tracking(self, attention_config):
        """Should be able to track attention statistics."""
        from deckgym.models.extractors import CardAttentionExtractor
        from deckgym.config import OBSERVATION_SIZE
        from gymnasium import spaces

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        extractor = CardAttentionExtractor(obs_space, config=attention_config)
        obs = torch.rand(2, OBSERVATION_SIZE)

        # Forward with stats tracking
        _ = extractor(obs, track_attention_stats=True)

        stats = extractor.get_attention_stats()

        assert isinstance(stats, dict)
        # Should have stats for each layer
        assert len(stats) == attention_config.attention_num_layers

    def test_gradient_flow(self, attention_config):
        """Gradients should flow through extractor."""
        from deckgym.models.extractors import CardAttentionExtractor
        from deckgym.config import OBSERVATION_SIZE
        from gymnasium import spaces

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        extractor = CardAttentionExtractor(obs_space, config=attention_config)
        obs = torch.rand(2, OBSERVATION_SIZE, requires_grad=False)

        features = extractor(obs)
        loss = features.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        has_grads = False
        for param in extractor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "Should have non-zero gradients"


class TestOnnxSafeAttention:
    """Tests for OnnxSafeAttention layer."""

    def test_attention_creation(self):
        """OnnxSafeAttention should be creatable."""
        from deckgym.models.extractors.layers import OnnxSafeAttention

        attn = OnnxSafeAttention(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
        )

        assert attn is not None
        assert attn.embed_dim == 64
        assert attn.num_heads == 4

    def test_attention_forward(self):
        """Forward pass should work with and without mask."""
        from deckgym.models.extractors.layers import OnnxSafeAttention

        attn = OnnxSafeAttention(embed_dim=64, num_heads=4, num_layers=2)

        batch_size, seq_len = 2, 10
        x = torch.rand(batch_size, seq_len, 64)

        # Without mask
        out = attn(x)
        assert out.shape == x.shape

        # With mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True  # Mask last 2 positions

        out_masked = attn(x, key_padding_mask=mask)
        assert out_masked.shape == x.shape

    def test_attention_stats(self):
        """Should track attention statistics when requested."""
        from deckgym.models.extractors.layers import OnnxSafeAttention

        attn = OnnxSafeAttention(embed_dim=64, num_heads=4, num_layers=2)
        x = torch.rand(2, 10, 64)

        _ = attn(x, track_stats=True)
        stats = attn.get_attention_stats()

        assert "attn_logit_std" in stats
        assert "attn_entropy" in stats


class TestMultiHeadAttentionPooling:
    """Tests for MultiHeadAttentionPooling layer."""

    def test_pooling_creation(self):
        """MultiHeadAttentionPooling should be creatable."""
        from deckgym.models.extractors.layers import MultiHeadAttentionPooling

        pool = MultiHeadAttentionPooling(embed_dim=64, num_heads=4, num_queries=2)

        assert pool is not None
        assert pool.num_queries == 2

    def test_pooling_forward(self):
        """Pooling should aggregate sequence to fixed-size output."""
        from deckgym.models.extractors.layers import MultiHeadAttentionPooling

        pool = MultiHeadAttentionPooling(embed_dim=64, num_heads=4, num_queries=2)

        batch_size, seq_len = 2, 10
        x = torch.rand(batch_size, seq_len, 64)

        out = pool(x)

        # Output should be [batch, num_queries * embed_dim]
        assert out.shape == (batch_size, 2 * 64)

    def test_pooling_with_mask(self):
        """Pooling should handle padding mask correctly."""
        from deckgym.models.extractors.layers import MultiHeadAttentionPooling

        pool = MultiHeadAttentionPooling(embed_dim=64, num_heads=4, num_queries=2)

        batch_size, seq_len = 2, 10
        x = torch.rand(batch_size, seq_len, 64)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -3:] = True  # Mask last 3 positions

        out = pool(x, key_padding_mask=mask)

        assert out.shape == (batch_size, 2 * 64)


class TestPolicyKwargsFactory:
    """Tests for create_attention_policy_kwargs."""

    def test_creates_valid_kwargs(self, attention_config):
        """Should create valid policy kwargs dict."""
        from deckgym.models.extractors import create_attention_policy_kwargs

        kwargs = create_attention_policy_kwargs(attention_config)

        assert isinstance(kwargs, dict)
        assert "features_extractor_class" in kwargs
        assert "features_extractor_kwargs" in kwargs
        assert "net_arch" in kwargs

    def test_kwargs_contain_config(self, attention_config):
        """Policy kwargs should pass config to extractor."""
        from deckgym.models.extractors import create_attention_policy_kwargs

        kwargs = create_attention_policy_kwargs(attention_config)

        extractor_kwargs = kwargs["features_extractor_kwargs"]
        assert "config" in extractor_kwargs
        assert extractor_kwargs["config"] == attention_config
