"""
Attention-Based Policy for MaskablePPO - V2 with Gradient Stability Fixes.

Key improvements over V1:
    1. Final LayerNorm after attention layers (critical for gradient flow)
    2. Proper weight initialization (Xavier for projections, scaled residuals)
    3. Optional gradient scaling between attention and heads
    4. Attention logit monitoring for saturation detection
    5. Configurable temperature for attention softmax

Architecture:
    1. Split observation into (global_features, card_features)
    2. Pass cards through self-attention layers with pre-norm
    3. Apply final LayerNorm
    4. Pool attended cards via learned queries
    5. Concatenate with global features
    6. Pass through policy/value heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple, Dict, Any, Optional, List

# Enable TF32 for faster matmuls on Ampere+ GPUs (up to 2x speedup, minimal precision loss)
# This is a global setting that affects all subsequent operations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Import constants from centralized config
from deckgym.config import (
    GLOBAL_FEATURES,
    FEATURES_PER_CARD,
    MAX_CARDS_IN_GAME as MAX_CARDS,
    # Model architecture defaults
    MODEL_EMBED_DIM_DEFAULT,
    MODEL_NUM_HEADS_DEFAULT,
    MODEL_NUM_LAYERS_DEFAULT,
    MODEL_ATTENTION_DROPOUT,
    MODEL_ATTENTION_POOL_QUERIES,
    MODEL_ATTENTION_TEMPERATURE,
    MODEL_INIT_RESIDUAL_SCALE,
    MODEL_GLOBAL_PROJ_DIM,
    MODEL_FF_EXPANSION_FACTOR,
    MODEL_OUTPUT_PROJ_FACTOR,
    MODEL_MASK_THRESHOLD,
    MODEL_POLICY_LAYERS_DEFAULT,
    MODEL_VALUE_LAYERS_DEFAULT,
    # Diagnostic thresholds
    DIAG_ATTENTION_LOGIT_STD_WARNING,
    DIAG_ATTENTION_ENTROPY_WARNING,
    # Config class
    TrainingConfig,
    DEFAULT_CONFIG,
)


class OnnxSafeAttention(nn.Module):
    """
    ONNX-compatible multi-head self-attention with gradient stability improvements.

    Changes from V1:
        - Explicit Xavier initialization for QKV projections
        - Scaled initialization for output projection (residual scaling)
        - Optional temperature parameter for attention logits
        - Attention stats tracking for debugging
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = MODEL_ATTENTION_DROPOUT,
        attention_temperature: float = MODEL_ATTENTION_TEMPERATURE,
        init_residual_scale: bool = MODEL_INIT_RESIDUAL_SCALE,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = attention_temperature

        # Fused QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # For monitoring attention saturation
        self.register_buffer("_attn_logit_std", torch.tensor(0.0))
        self.register_buffer("_attn_entropy", torch.tensor(0.0))

        # Initialize weights properly
        self._init_weights(init_residual_scale)

    def _init_weights(self, init_residual_scale: bool):
        """
        Initialize weights for stable gradient flow.

        - QKV: Xavier uniform for balanced variance
        - Output: Scaled by 1/sqrt(2*num_layers) for residual streams (GPT-2 style)
        """
        nn.init.xavier_uniform_(self.qkv_proj.weight)

        if init_residual_scale:
            # Scale down output projection for stable residuals
            # This makes attention blocks start closer to identity
            nn.init.xavier_uniform_(
                self.out_proj.weight, gain=1.0 / math.sqrt(2 * self.num_layers)
            )
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        track_stats: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] where True = masked (ignored)
            track_stats: If True, update attention statistics for monitoring

        Returns:
            [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.embed_dim)
        q, k, v = qkv.unbind(dim=2)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Convert key_padding_mask to attention mask for SDPA
        # SDPA expects: True = attend, False = mask (opposite of our convention)
        attn_mask = None
        if key_padding_mask is not None:
            # Expand mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)

        # Track attention statistics for debugging (only when requested)
        if track_stats:
            with torch.no_grad():
                # Manual attention for stats (not used for output)
                attn_logits = (
                    torch.matmul(q, k.transpose(-2, -1)) * self.scale / self.temperature
                )
                if key_padding_mask is not None:
                    mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                    attn_logits = attn_logits.masked_fill(mask, float("-inf"))
                valid_logits = attn_logits[~attn_logits.isinf()]
                if valid_logits.numel() > 0:
                    self._attn_logit_std = valid_logits.std()
                attn_probs_for_stats = F.softmax(attn_logits, dim=-1)
                entropy = -(
                    attn_probs_for_stats * (attn_probs_for_stats + 1e-10).log()
                ).sum(dim=-1)
                self._attn_entropy = entropy.mean()

        # Use PyTorch SDPA (FlashAttention when available) - 2-4x faster
        # Apply temperature by scaling q (equivalent to dividing attention logits)
        q_scaled = q * (self.scale / self.temperature) ** 0.5
        k_scaled = k * (self.scale / self.temperature) ** 0.5

        attn_output = F.scaled_dot_product_attention(
            q_scaled,
            k_scaled,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)

    def get_attention_stats(self) -> Dict[str, float]:
        """Return attention statistics for monitoring."""
        return {
            "attn_logit_std": self._attn_logit_std.item(),
            "attn_entropy": self._attn_entropy.item(),
        }


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling with gradient stability improvements.

    Changes from V1:
        - Proper initialization matching the attention layers
        - LayerNorm on queries for stable cross-attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = MODEL_NUM_HEADS_DEFAULT,
        num_queries: int = MODEL_ATTENTION_POOL_QUERIES,
        attention_temperature: float = MODEL_ATTENTION_TEMPERATURE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = attention_temperature

        # Learned query vectors with proper initialization
        self.queries = nn.Parameter(torch.zeros(num_queries, embed_dim))

        # LayerNorm for queries (stabilizes cross-attention)
        self.query_norm = nn.LayerNorm(embed_dim)

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable gradients."""
        # Queries: small init so pooling starts near uniform
        nn.init.normal_(self.queries, std=0.02)

        # Projections: Xavier
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)  # Slightly scaled down

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] where True = masked

        Returns:
            [batch, num_queries * embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Expand and normalize queries
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_norm(queries)

        # Project queries
        q = self.q_proj(queries)

        # Fused KV projection
        kv = self.kv_proj(x)
        kv = kv.view(batch_size, seq_len, 2, self.embed_dim)
        k, v = kv.unbind(dim=2)

        # Reshape for multi-head
        q = q.view(
            batch_size, self.num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale / self.temperature
        )

        # Apply mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)

        # Weighted combination
        pooled = torch.matmul(attn_probs, v)

        # Reshape
        pooled = (
            pooled.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_queries, self.embed_dim)
        )

        # Output projection
        pooled = self.out_proj(pooled)

        # Flatten
        return pooled.view(batch_size, -1)


class CardAttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with gradient stability improvements.

    Key changes from V1:
        1. Final LayerNorm after attention layers
        2. Proper weight initialization throughout
        3. Optional gradient checkpointing for memory efficiency
        4. Attention stats tracking for debugging
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        config: TrainingConfig = DEFAULT_CONFIG,
        global_features: int = GLOBAL_FEATURES,
        features_per_card: int = FEATURES_PER_CARD,
        max_cards: int = MAX_CARDS,
    ):
        # Extract attention params from config
        embed_dim = config.attention_embed_dim
        num_heads = config.attention_num_heads
        num_layers = config.attention_num_layers

        # Calculate output features dimension
        features_dim = config.attention_global_proj_dim + (
            embed_dim * config.attention_output_proj_factor
        )
        super().__init__(observation_space, features_dim)

        self.config = config
        self.global_features = global_features
        self.features_per_card = features_per_card
        self.max_cards = max_cards
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Global features projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_features, config.attention_global_proj_dim),
            nn.GELU(),
            nn.LayerNorm(config.attention_global_proj_dim),
            nn.Linear(
                config.attention_global_proj_dim, config.attention_global_proj_dim
            ),
            nn.GELU(),
        )

        # Card embedding with LayerNorm
        self.card_embed = nn.Sequential(
            nn.Linear(features_per_card, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Attention layers
        self.attention_layers = nn.ModuleList(
            [
                OnnxSafeAttention(
                    embed_dim,
                    num_heads,
                    num_layers=num_layers,
                    dropout=config.attention_dropout,
                    attention_temperature=config.attention_temperature,
                    init_residual_scale=config.attention_init_residual_scale,
                )
                for _ in range(num_layers)
            ]
        )

        # Feed-forward layers with proper init
        self.ff_layers = nn.ModuleList(
            [self._make_ff_block(config) for _ in range(num_layers)]
        )

        # Layer norms (2 per layer: pre-attention, pre-FF)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_layers * 2)]
        )

        # CRITICAL: Final LayerNorm after all attention layers
        self.final_norm = nn.LayerNorm(embed_dim)

        # Attention pooling
        self.attention_pool = MultiHeadAttentionPooling(
            embed_dim,
            num_heads,
            num_queries=config.attention_pool_queries,
            attention_temperature=config.attention_temperature,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(
                embed_dim * config.attention_pool_queries,
                embed_dim * config.attention_output_proj_factor,
            ),
            nn.GELU(),
            nn.LayerNorm(embed_dim * config.attention_output_proj_factor),
        )

        # Initialize card embedding and global proj
        self._init_embeddings()

    def _make_ff_block(self, config: TrainingConfig) -> nn.Module:
        """Create a feed-forward block with proper initialization."""
        embed_dim = config.attention_embed_dim
        ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * config.attention_ff_expansion_factor),
            nn.GELU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(embed_dim * config.attention_ff_expansion_factor, embed_dim),
            nn.Dropout(config.attention_dropout),
        )

        # Scale down the output layer for stable residuals
        if config.attention_init_residual_scale:
            nn.init.xavier_uniform_(
                ff[3].weight, gain=1.0 / math.sqrt(2 * config.attention_num_layers)
            )
            nn.init.zeros_(ff[3].bias)

        return ff

    def _init_embeddings(self):
        """Initialize embedding layers."""
        for module in [self.global_proj, self.card_embed]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _attention_block(
        self,
        x: torch.Tensor,
        card_mask: torch.Tensor,
        layer_idx: int,
        track_stats: bool,
    ) -> torch.Tensor:
        """Single attention block (for gradient checkpointing)."""
        # Self-attention with pre-norm
        normed = self.layer_norms[layer_idx * 2](x)
        attn_out = self.attention_layers[layer_idx](
            normed,
            key_padding_mask=card_mask,
            track_stats=track_stats,
        )
        x = x + attn_out

        # Feed-forward with pre-norm
        normed = self.layer_norms[layer_idx * 2 + 1](x)
        ff_out = self.ff_layers[layer_idx](normed)
        x = x + ff_out

        return x

    def forward(
        self,
        observations: torch.Tensor,
        track_attention_stats: bool = False,
    ) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split into global and card features
        global_feats = observations[:, : self.global_features]
        card_feats_flat = observations[:, self.global_features :]

        # Reshape cards
        card_feats = card_feats_flat.view(
            batch_size, self.max_cards, self.features_per_card
        )

        # Create mask for empty slots
        card_sum = card_feats.abs().sum(dim=-1)
        card_mask = card_sum < self.config.attention_mask_threshold

        # Process global features
        global_out = self.global_proj(global_feats)

        # Embed cards
        x = self.card_embed(card_feats)

        # Apply attention layers with PRE-NORM
        for i in range(self.num_layers):
            if self.training and self.use_gradient_checkpointing:
                # Gradient checkpointing: trade compute for memory
                x = gradient_checkpoint(
                    self._attention_block,
                    x,
                    card_mask,
                    i,
                    track_attention_stats,
                    use_reentrant=False,
                )
            else:
                x = self._attention_block(x, card_mask, i, track_attention_stats)

        # CRITICAL: Final normalization before pooling
        x = self.final_norm(x)

        # Attention-based pooling
        pooled = self.attention_pool(x, key_padding_mask=card_mask)

        # Final projection
        cards_out = self.output_proj(pooled)

        # Concatenate global and card representations
        combined = torch.cat([global_out, cards_out], dim=1)

        return combined

    def get_attention_stats(self) -> Dict[str, Dict[str, float]]:
        """Get attention statistics from all layers for monitoring."""
        stats = {}
        for i, layer in enumerate(self.attention_layers):
            stats[f"layer_{i}"] = layer.get_attention_stats()
        return stats


class CardAttentionPolicy(ActorCriticPolicy):
    """
    Custom policy using CardAttentionExtractor.

    Compatible with MaskablePPO from sb3-contrib.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        config: TrainingConfig = DEFAULT_CONFIG,
        *args,
        **kwargs,
    ):
        self.training_config = config

        net_arch = dict(
            pi=list(config.policy_layers),
            vf=list(config.value_layers),
        )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=nn.SiLU if config.use_silu else nn.ReLU,
            features_extractor_class=CardAttentionExtractor,
            features_extractor_kwargs={"config": config},
            ortho_init=False,  # We handle init ourselves
            *args,
            **kwargs,
        )

        # Re-initialize action and value nets with proper scaling
        self._init_heads()

    def _init_heads(self):
        """Initialize policy and value heads for balanced gradients."""
        for net in [self.action_net, self.value_net]:
            if hasattr(net, "weight"):
                nn.init.xavier_uniform_(net.weight, gain=0.01)
                if hasattr(net, "bias") and net.bias is not None:
                    nn.init.zeros_(net.bias)


def create_attention_policy_kwargs(
    config: TrainingConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Create policy_kwargs dict for MaskablePPO.

    Usage:
        from attention_policy_v2 import create_attention_policy_kwargs
        from deckgym.config import TrainingConfig, CONSERVATIVE_CONFIG

        model = MaskablePPO(
            CardAttentionPolicy,
            env,
            policy_kwargs=create_attention_policy_kwargs(CONSERVATIVE_CONFIG),
            ...
        )
    """
    return {
        "features_extractor_class": CardAttentionExtractor,
        "features_extractor_kwargs": {"config": config},
        "net_arch": dict(
            pi=list(config.policy_layers),
            vf=list(config.value_layers),
        ),
        "activation_fn": nn.SiLU if config.use_silu else nn.ReLU,
        "ortho_init": False,
    }


# =============================================================================
# Diagnostic utilities
# =============================================================================


def diagnose_gradients(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms by module for debugging.

    Call after loss.backward() but before optimizer.step().

    Returns dict with gradient norms per module group.
    """
    grad_norms = {}

    groups = {
        "card_embed": [],
        "global_proj": [],
        "attention_qkv": [],
        "attention_out": [],
        "attention_ff": [],
        "layer_norms": [],
        "pool": [],
        "output_proj": [],
        "policy_net": [],
        "value_net": [],
        "action_net": [],
    }

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.norm().item()

        if "card_embed" in name:
            groups["card_embed"].append(grad_norm)
        elif "global_proj" in name:
            groups["global_proj"].append(grad_norm)
        elif "qkv_proj" in name:
            groups["attention_qkv"].append(grad_norm)
        elif "out_proj" in name and "attention" in name:
            groups["attention_out"].append(grad_norm)
        elif "ff_layers" in name:
            groups["attention_ff"].append(grad_norm)
        elif "layer_norm" in name or "final_norm" in name:
            groups["layer_norms"].append(grad_norm)
        elif "pool" in name:
            groups["pool"].append(grad_norm)
        elif "output_proj" in name:
            groups["output_proj"].append(grad_norm)
        elif "mlp_extractor" in name and "policy" in name:
            groups["policy_net"].append(grad_norm)
        elif "mlp_extractor" in name and "value" in name:
            groups["value_net"].append(grad_norm)
        elif "action_net" in name:
            groups["action_net"].append(grad_norm)

    for group_name, norms in groups.items():
        if norms:
            grad_norms[f"{group_name}_mean"] = sum(norms) / len(norms)
            grad_norms[f"{group_name}_max"] = max(norms)

    # Compute imbalance ratio
    if (
        grad_norms.get("attention_qkv_mean", 0) > 0
        and grad_norms.get("action_net_mean", 0) > 0
    ):
        grad_norms["qkv_action_ratio"] = (
            grad_norms["action_net_mean"] / grad_norms["attention_qkv_mean"]
        )

    return grad_norms


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    # Create config
    config = TrainingConfig(
        attention_embed_dim=256,
        attention_num_heads=8,
        attention_num_layers=3,
    )

    # Create dummy observation space
    obs_size = GLOBAL_FEATURES + MAX_CARDS * FEATURES_PER_CARD
    print(f"Observation size: {obs_size}")

    observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
    )

    # Create extractor
    extractor = CardAttentionExtractor(observation_space, config=config)
    print(f"Features dim: {extractor.features_dim}")

    # Count parameters
    n_params = sum(p.numel() for p in extractor.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    dummy_obs = torch.rand(4, obs_size)
    features = extractor(dummy_obs, track_attention_stats=True)
    print(f"Output shape: {features.shape}")

    # Check attention stats
    stats = extractor.get_attention_stats()
    for layer_name, layer_stats in stats.items():
        print(
            f"{layer_name}: logit_std={layer_stats['attn_logit_std']:.3f}, "
            f"entropy={layer_stats['attn_entropy']:.3f}"
        )

    print("✓ CardAttentionExtractor V2 works!")
