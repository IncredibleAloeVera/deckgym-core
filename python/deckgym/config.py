#!/usr/bin/env python3
"""
Centralized configuration for DeckGym RL training pipeline.

This module contains ALL hyperparameters and magic numbers used throughout
the training pipeline. Use this as the single source of truth for configuration.

Usage:
    from deckgym.config import TrainingConfig, DEFAULT_CONFIG
    config = TrainingConfig.from_yaml("configs/pfsp_conservative.yaml")

    # Or generate a template:
    python -m deckgym.config --generate-template configs/new_config.yaml
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union
import yaml
import math
import json

# =============================================================================
# OBSERVATION SPACE CONSTANTS (must match src/rl/observation.rs)
# =============================================================================

# Card features
GLOBAL_FEATURES = 171  # 1 turn + 3 points + 2 deck + 2 hand + 2 discard + 32 deck_energy + 1 has_stadium + 128 stadium_emb
FEATURES_PER_CARD = 731
MAX_CARDS_IN_GAME = 40
OBSERVATION_SIZE = GLOBAL_FEATURES + (MAX_CARDS_IN_GAME * FEATURES_PER_CARD)  # 24291

# Action space
ACTION_SPACE_SIZE = 179  # Must match src/rl/action_mask.rs

# Energy types
NUM_ENERGY_TYPES = 10  # Includes colorless
NUM_ENERGY_TYPES_DECK = 8  # Excludes colorless and "none"

# Effect categories (must match src/rl/*.rs)
NUM_ATTACK_EFFECT_CATEGORIES = 13
NUM_ABILITY_EFFECT_CATEGORIES = 17
NUM_SUPPORTER_EFFECT_CATEGORIES = 14
NUM_TOOL_TYPES = 8


# =============================================================================
# REWARD CONSTANTS (must match src/vec_game.rs calculate_reward)
# =============================================================================

REWARD_WIN_BASE = 1.0
REWARD_LOSS_BASE = -1.0
REWARD_DRAW_BASE = -0.5
REWARD_POINT_DIFF_SCALE = 6.0  # point_diff / 6 for scaling
REWARD_SPEED_REFERENCE_TURNS = 13  # speed_factor = 1 + (13 - turns) / 13
REWARD_SPEED_FACTOR_MIN = 1.0  # speed_factor is clamped to >= 1.0


# =============================================================================
# TRUESKILL EVALUATION CONSTANTS (evaluate.py)
# =============================================================================

TRUESKILL_MU = 1500  # Initial skill rating
TRUESKILL_SIGMA = 500  # Initial uncertainty (large for new agents)
TRUESKILL_BETA = 250  # Performance variance (sigma / 2)
TRUESKILL_TAU = 0  # Dynamic factor
TRUESKILL_DRAW_PROBABILITY = 0.01  # Probability of draw

# Default evaluation games
N_ROBIN_ROUNDS = 10  # Brute force for precision
EVAL_REPORTS_DIR = "eval_reports"
EVAL_DEFAULT_DEVICE = "cuda"


# =============================================================================
# DIAGNOSTIC THRESHOLDS (diagnose_model.py)
# =============================================================================

DIAG_GRADIENT_IMBALANCE_WARNING = 100  # Warn if ratio > 100
DIAG_GRADIENT_IMBALANCE_CRITICAL = 1000  # Critical if ratio > 1000
DIAG_LARGE_GRADIENT_THRESHOLD = 100  # Flag gradients > 100
DIAG_WEIGHT_STD_MIN = 0.001  # Unusual if std < 0.001
DIAG_WEIGHT_STD_MAX = 1.0  # Unusual if std > 1.0

# Attention-specific diagnostics
DIAG_ATTENTION_LOGIT_STD_WARNING = 10.0  # Softmax saturation if > 10
DIAG_ATTENTION_ENTROPY_WARNING = 0.5  # Attention too peaky if < 0.5


# =============================================================================
# MODEL ARCHITECTURE DEFAULTS (attention_policy.py)
# =============================================================================

# Attention model defaults
MODEL_EMBED_DIM_DEFAULT = 512  # Transformer embedding dimension
MODEL_NUM_HEADS_DEFAULT = 8  # Multi-head attention heads
MODEL_NUM_LAYERS_DEFAULT = 3  # Number of transformer layers
MODEL_ATTENTION_DROPOUT = 0.1  # Dropout in attention layers (reduced from 0.15)
MODEL_ATTENTION_POOL_QUERIES = 4  # Number of learnable queries for pooling

# Attention stability parameters (NEW)
MODEL_ATTENTION_TEMPERATURE = 1.0  # Softmax temperature (increase if saturating)
MODEL_INIT_RESIDUAL_SCALE = True  # Scale residual branch outputs for stability

# Policy/Value network architecture (MLP heads after attention/pooling)
MODEL_POLICY_LAYERS_DEFAULT = (512, 256, 128)
MODEL_VALUE_LAYERS_DEFAULT = (512, 256)

# Internal architecture scaling and thresholds
MODEL_GLOBAL_PROJ_DIM = 256  # Projection size for non-card (global) features
MODEL_FF_EXPANSION_FACTOR = 4  # Transformer FF block expansion ratio
MODEL_OUTPUT_PROJ_FACTOR = 2  # Output size = embed_dim * this factor
MODEL_MASK_THRESHOLD = 1e-6  # Threshold to detect all-zero padding


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# PFSP internal constants
PFSP_NEUTRAL_WINRATE = 0.5  # Winrate for untested opponents
PFSP_MIN_EPISODES_FOR_UPDATE = 10  # Minimum episodes before updating winrates


@dataclass
class TrainingConfig:
    """Complete training configuration with all hyperparameters."""

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    meta_decks_dir: str = "archetypes_by_era"
    save_path: str = "models/rl_bot"
    checkpoint_dir: str = "./checkpoints/"
    tensorboard_dir: str = "./logs/"
    resume_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # Training Duration
    # -------------------------------------------------------------------------
    total_timesteps: int = 30_000_000
    checkpoint_freq: int = 10_000
    brutal_resume: bool = False  # Skip curriculum and jump to final stage
    resume_subtract_steps: bool = (
        False  # Subtract already-done steps from total_timesteps on resume
    )

    # -------------------------------------------------------------------------
    # PPO Hyperparameters
    # -------------------------------------------------------------------------
    base_learning_rate: float = 1e-5  # Peak LR after warmup
    min_learning_rate: float = 5e-6  # Final LR after cosine decay
    warmup_ratio: float = 0.02  # 2% of total steps for warmup

    n_steps: int = 256  # Steps per env per rollout
    batch_size: int = 1024  # Minibatch size for PPO updates
    n_epochs: int = 8  # PPO epochs per update

    gamma: float = 0.98  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    ent_coef: float = 0.02  # Entropy coefficient (exploration)
    vf_coef: float = 0.5  # Value function coefficient
    clip_range: float = 0.2  # PPO clipping range
    draw_reward: float = REWARD_DRAW_BASE  # Reward for a tie
    max_grad_norm: float = 1.0  # Gradient clipping
    target_kl: float = 0.015  # Early stopping if KL > target

    # -------------------------------------------------------------------------
    # Network Architecture
    # -------------------------------------------------------------------------
    use_attention: bool = True
    attention_embed_dim: int = MODEL_EMBED_DIM_DEFAULT
    attention_num_heads: int = MODEL_NUM_HEADS_DEFAULT
    attention_num_layers: int = MODEL_NUM_LAYERS_DEFAULT
    use_silu: bool = True  # Use SiLU activation (vs ReLU)
    attention_global_proj_dim: int = MODEL_GLOBAL_PROJ_DIM
    attention_ff_expansion_factor: int = MODEL_FF_EXPANSION_FACTOR
    attention_output_proj_factor: int = MODEL_OUTPUT_PROJ_FACTOR
    attention_mask_threshold: float = MODEL_MASK_THRESHOLD

    # Attention stability parameters (NEW)
    attention_dropout: float = MODEL_ATTENTION_DROPOUT
    attention_pool_queries: int = MODEL_ATTENTION_POOL_QUERIES
    attention_temperature: float = MODEL_ATTENTION_TEMPERATURE
    attention_init_residual_scale: bool = MODEL_INIT_RESIDUAL_SCALE

    # Policy/Value head architecture
    policy_layers: Tuple[int, ...] = MODEL_POLICY_LAYERS_DEFAULT
    value_layers: Tuple[int, ...] = MODEL_VALUE_LAYERS_DEFAULT

    # -------------------------------------------------------------------------
    # Self-Play (Simple frozen opponent)
    # -------------------------------------------------------------------------
    frozen_opponent_update_rollouts: int = 8  # Update frozen opponent every N rollouts

    # -------------------------------------------------------------------------
    # PFSP (Prioritized Fictitious Self-Play)
    # -------------------------------------------------------------------------
    use_pfsp: bool = True  # Enable PFSP instead of simple self-play
    pfsp_pool_size: int = 10  # Maximum opponents in pool
    pfsp_add_every_n_rollouts: int = 8  # Add current agent to pool
    pfsp_select_every_n_rollouts: int = 2  # Select new opponent
    pfsp_priority_exponent: float = 2.0  # f(x) = x^p (higher = focus on hard opponents)
    pfsp_checkpoint_dir: str = "./checkpoints/pfsp_pool/"
    pfsp_min_winrate_to_add: float = 0.50  # Only add to pool if agent winrate >= this

    # PFSP Baseline Curriculum (permanent baseline slots with progressive difficulty)
    # These baselines are never evicted and provide diversity signal against non-self opponents
    # baseline_slots is now DYNAMIC based on stage size (see selector.py)
    pfsp_baseline_max_allocation: float = 0.20  # Max 20% of envs can play vs baselines
    # Curriculum stages: list of (step_threshold, [baseline_codes])
    # At each stage, the baselines are replaced. Codes:
    #   v=ValueFunction, w=WeightedRandom, aa=AttachAttack, er=EvolutionRusher
    #   e2/e3/e4=Expectiminimax(depth) - omniscient, excluded from global WR
    #   o[n][device]=ONNX model (n=index from newest, device=c/g for cpu/cuda)
    #     Examples: o1 = newest ONNX on CUDA, o2c = 2nd newest on CPU
    # Order: Learn coherence first (o2/attention), then resist exploits (o1/MLP)
    pfsp_baseline_curriculum: List[Tuple[int, List[str]]] = field(
        default_factory=lambda: [
            (0, ["v", "w"]),  # Stage 1: Easy warmup (0-500k)
            (500_000, ["aa", "er"]),  # Stage 2: Medium bots (500k-2M)
            (2_000_000, ["er", "o2"]),  # Stage 3: Attention anchor (2M-5M)
            (5_000_000, ["o2t", "o1"]),  # Stage 4: Both NN styles (5M-10M)
            (10_000_000, ["e2", "o1", "o2"]),  # Stage 5: Boss + both anchors (10M+)
        ]
    )

    # -------------------------------------------------------------------------
    # Game Environment Limits
    # -------------------------------------------------------------------------
    max_turns: int = 99  # Official Pokemon TCG Pocket limit
    max_actions_per_turn: int = 50  # Prevents single-turn infinite loops
    max_total_actions: int = 500  # Prevents runaway episodes

    # -------------------------------------------------------------------------
    # Parallel Training
    # -------------------------------------------------------------------------
    n_envs: int = 12  # Number of parallel environments
    use_batched_env: bool = True  # Use Rust-side batched environment

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    device: str = "auto"  # "auto", "cuda", "cpu"
    onnx_device: str = "cuda"  # "auto", "cuda", "cpu"
    pfsp_opponent_device: str = "cuda"  # "auto", "cuda", "cpu"
    use_gradient_checkpointing: bool = (
        False  # Trade compute for memory (~30% slower, ~60% less VRAM)
    )
    use_tf32: bool = True  # Enable TF32 for faster matmuls on Ampere+ GPUs

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    def __post_init__(self):
        """Perform type casting to ensure robustness after YAML loading or manual init."""
        # Numeric fields that MUST be float
        float_fields = [
            "base_learning_rate",
            "min_learning_rate",
            "warmup_ratio",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "clip_range",
            "draw_reward",
            "max_grad_norm",
            "target_kl",
            "attention_mask_threshold",
            "attention_dropout",
            "attention_temperature",
            "pfsp_priority_exponent",
            "pfsp_min_winrate_to_add",
        ]
        for field_name in float_fields:
            val = getattr(self, field_name)
            if val is not None:
                try:
                    setattr(self, field_name, float(val))
                except (ValueError, TypeError):
                    pass

        # Numeric fields that MUST be int
        int_fields = [
            "total_timesteps",
            "checkpoint_freq",
            "n_steps",
            "batch_size",
            "n_epochs",
            "attention_embed_dim",
            "attention_num_heads",
            "attention_num_layers",
            "attention_global_proj_dim",
            "attention_ff_expansion_factor",
            "attention_output_proj_factor",
            "attention_pool_queries",
            "frozen_opponent_update_rollouts",
            "pfsp_pool_size",
            "pfsp_add_every_n_rollouts",
            "pfsp_select_every_n_rollouts",
            "n_envs",
            "max_turns",
            "max_actions_per_turn",
            "max_total_actions",
        ]
        for field_name in int_fields:
            val = getattr(self, field_name)
            if val is not None:
                try:
                    setattr(self, field_name, int(val))
                except (ValueError, TypeError):
                    pass

        # Ensure tuples/lists
        if isinstance(self.policy_layers, list):
            self.policy_layers = tuple(self.policy_layers)
        if isinstance(self.value_layers, list):
            self.value_layers = tuple(self.value_layers)

    def get_learning_rate_schedule(self, total_timesteps: Optional[int] = None):
        """
        Learning rate schedule with warmup + cosine annealing.

        - Linear warmup: 0 → base_lr (first warmup_ratio % of training)
        - Cosine decay: base_lr → min_lr
        """
        base_lr = float(self.base_learning_rate)
        min_lr = float(self.min_learning_rate)

        if total_timesteps:
            total_updates = total_timesteps // self.n_steps
            warmup_updates = int(total_updates * self.warmup_ratio)

            def lr_schedule_cosine(progress_remaining: float) -> float:
                current_update = int((1.0 - progress_remaining) * total_updates)

                if current_update < warmup_updates:
                    # Warmup phase: linear 0 → base_lr
                    return (current_update / max(1, warmup_updates)) * base_lr
                else:
                    # Cosine annealing: base_lr → min_lr
                    decay_updates = total_updates - warmup_updates
                    if decay_updates <= 0:
                        return min_lr
                    progress = (current_update - warmup_updates) / decay_updates
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    return min_lr + (base_lr - min_lr) * cosine_decay

            return lr_schedule_cosine
        else:
            return lambda _: base_lr

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert tuples to lists for YAML
        d["policy_layers"] = list(self.policy_layers)
        d["value_layers"] = list(self.value_layers)
        return d

    def to_yaml(self) -> str:
        """Generate YAML string with organized sections."""
        return f"""# DeckGym Training Configuration
# Generated template - customize as needed

paths:
  meta_decks_dir: "{self.meta_decks_dir}"
  save_path: "{self.save_path}"
  checkpoint_dir: "{self.checkpoint_dir}"
  tensorboard_dir: "{self.tensorboard_dir}"
  resume_path: {f'"{self.resume_path}"' if self.resume_path else 'null'}

model:
  use_attention: {str(self.use_attention).lower()}
  attention_embed_dim: {self.attention_embed_dim}
  attention_num_heads: {self.attention_num_heads}
  attention_num_layers: {self.attention_num_layers}
  attention_global_proj_dim: {self.attention_global_proj_dim}
  attention_ff_expansion_factor: {self.attention_ff_expansion_factor}
  attention_output_proj_factor: {self.attention_output_proj_factor}
  attention_mask_threshold: {self.attention_mask_threshold}
  
  # Stability parameters
  attention_dropout: {self.attention_dropout}
  attention_pool_queries: {self.attention_pool_queries}
  attention_temperature: {self.attention_temperature}
  attention_init_residual_scale: {str(self.attention_init_residual_scale).lower()}
  
  use_silu: {str(self.use_silu).lower()}

  net_arch:
    pi: {list(self.policy_layers)}
    vf: {list(self.value_layers)}

ppo:
  learning_rate: {self.base_learning_rate}
  min_learning_rate: {self.min_learning_rate}
  warmup_ratio: {self.warmup_ratio}

  n_steps: {self.n_steps}
  batch_size: {self.batch_size}
  n_epochs: {self.n_epochs}

  gamma: {self.gamma}
  gae_lambda: {self.gae_lambda}
  ent_coef: {self.ent_coef}
  vf_coef: {self.vf_coef}
  clip_range: {self.clip_range}
  draw_reward: {self.draw_reward}
  max_grad_norm: {self.max_grad_norm}
  target_kl: {self.target_kl}

training:
  total_timesteps: {self.total_timesteps}
  checkpoint_freq: {self.checkpoint_freq}
  n_envs: {self.n_envs}
  use_batched_env: {str(self.use_batched_env).lower()}
  device: "{self.device}"
  onnx_device: "{self.onnx_device}"
  pfsp_opponent_device: "{self.pfsp_opponent_device}"
  frozen_opponent_update_rollouts: {self.frozen_opponent_update_rollouts}
  use_gradient_checkpointing: {str(self.use_gradient_checkpointing).lower()}
  brutal_resume: {str(self.brutal_resume).lower()}
  resume_subtract_steps: {str(self.resume_subtract_steps).lower()}

  # PFSP settings
  use_pfsp: {str(self.use_pfsp).lower()}
  pfsp_pool_size: {self.pfsp_pool_size}
  pfsp_add_every_n_rollouts: {self.pfsp_add_every_n_rollouts}
  pfsp_select_every_n_rollouts: {self.pfsp_select_every_n_rollouts}
  pfsp_priority_exponent: {self.pfsp_priority_exponent}
  pfsp_checkpoint_dir: "{self.pfsp_checkpoint_dir}"
  pfsp_baseline_curriculum: {json.dumps(self.pfsp_baseline_curriculum)}

environment:
  max_turns: {self.max_turns}
  max_actions_per_turn: {self.max_actions_per_turn}
  max_total_actions: {self.max_total_actions}
"""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure
        flat_config = {}

        # Paths
        if "paths" in config_dict:
            for key in [
                "meta_decks_dir",
                "save_path",
                "checkpoint_dir",
                "tensorboard_dir",
                "resume_path",
            ]:
                if key in config_dict["paths"]:
                    flat_config[key] = config_dict["paths"][key]

        # Model params
        if "model" in config_dict:
            model = config_dict["model"]
            for key in [
                "use_attention",
                "attention_embed_dim",
                "attention_num_heads",
                "attention_num_layers",
                "attention_global_proj_dim",
                "attention_ff_expansion_factor",
                "attention_output_proj_factor",
                "attention_mask_threshold",
                # New stability params
                "attention_dropout",
                "attention_pool_queries",
                "attention_temperature",
                "attention_init_residual_scale",
                "use_silu",
            ]:
                if key in model:
                    flat_config[key] = model[key]
            if "net_arch" in model:
                if "pi" in model["net_arch"]:
                    flat_config["policy_layers"] = tuple(model["net_arch"]["pi"])
                if "vf" in model["net_arch"]:
                    flat_config["value_layers"] = tuple(model["net_arch"]["vf"])

        # PPO params
        if "ppo" in config_dict:
            ppo = config_dict["ppo"]
            if "learning_rate" in ppo:
                flat_config["base_learning_rate"] = ppo["learning_rate"]
            for key in [
                "min_learning_rate",
                "warmup_ratio",
                "n_steps",
                "batch_size",
                "n_epochs",
                "gamma",
                "gae_lambda",
                "ent_coef",
                "vf_coef",
                "clip_range",
                "draw_reward",
                "max_grad_norm",
                "target_kl",
            ]:
                if key in ppo:
                    flat_config[key] = ppo[key]

        # Training params
        if "training" in config_dict:
            training = config_dict["training"]
            for key in [
                "total_timesteps",
                "checkpoint_freq",
                "n_envs",
                "use_batched_env",
                "device",
                "onnx_device",
                "pfsp_opponent_device",
                "frozen_opponent_update_rollouts",
                "use_gradient_checkpointing",
                "use_pfsp",
                "pfsp_pool_size",
                "pfsp_add_every_n_rollouts",
                "pfsp_select_every_n_rollouts",
                "pfsp_priority_exponent",
                "pfsp_checkpoint_dir",
                "brutal_resume",
                "resume_subtract_steps",
                "pfsp_baseline_curriculum",
            ]:
                if key in training:
                    flat_config[key] = training[key]

        # Environment params
        if "environment" in config_dict:
            env = config_dict["environment"]
            for key in ["max_turns", "max_actions_per_turn", "max_total_actions"]:
                if key in env:
                    flat_config[key] = env[key]

        # Use class initializer to trigger __post_init__ casting
        return cls(**flat_config)

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml())

    def validate(self) -> list:
        """
        Validate configuration and return list of warnings.

        Returns empty list if all good.
        """
        warnings = []

        # Architecture checks
        if self.attention_embed_dim % self.attention_num_heads != 0:
            warnings.append(
                f"attention_embed_dim ({self.attention_embed_dim}) must be divisible by "
                f"attention_num_heads ({self.attention_num_heads})"
            )

        if self.attention_embed_dim < 64:
            warnings.append(
                f"attention_embed_dim ({self.attention_embed_dim}) is very small, may underfit"
            )

        if self.attention_embed_dim > 1024:
            warnings.append(
                f"attention_embed_dim ({self.attention_embed_dim}) is very large, may be slow"
            )

        if self.attention_num_layers > 6:
            warnings.append(
                f"attention_num_layers ({self.attention_num_layers}) is high, "
                "consider gradient checkpointing"
            )

        # Stability checks
        if self.attention_temperature < 0.5:
            warnings.append(
                f"attention_temperature ({self.attention_temperature}) is low, "
                "may cause gradient issues"
            )

        if self.attention_dropout > 0.3:
            warnings.append(
                f"attention_dropout ({self.attention_dropout}) is high, "
                "may slow learning"
            )

        if not self.attention_init_residual_scale and self.attention_num_layers > 2:
            warnings.append(
                "attention_init_residual_scale=False with >2 layers may cause instability"
            )

        # Head checks
        if len(self.policy_layers) > 4:
            warnings.append(
                f"policy_layers has {len(self.policy_layers)} layers, "
                "may dilute gradients from attention"
            )

        return warnings


# Default configuration
DEFAULT_CONFIG = TrainingConfig()


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeckGym Training Configuration")
    parser.add_argument(
        "--generate-template",
        "-g",
        type=str,
        metavar="PATH",
        help="Generate a YAML template at the specified path",
    )
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        choices=["default"],
        default="default",
        help="Which preset to use for generation",
    )
    parser.add_argument(
        "--print-constants",
        "-c",
        action="store_true",
        help="Print all observation/action space constants",
    )
    parser.add_argument(
        "--validate",
        "-v",
        type=str,
        metavar="PATH",
        help="Validate a YAML config file",
    )
    args = parser.parse_args()

    presets = {
        "default": DEFAULT_CONFIG,
    }

    if args.generate_template:
        config = presets[args.preset]
        config.save_yaml(args.generate_template)
        print(f"Generated {args.preset} config at: {args.generate_template}")
    elif args.validate:
        config = TrainingConfig.from_yaml(args.validate)
        warnings = config.validate()
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
        else:
            print("✓ Configuration is valid")
    elif args.print_constants:
        print("=== Observation Space ===")
        print(f"GLOBAL_FEATURES: {GLOBAL_FEATURES}")
        print(f"FEATURES_PER_CARD: {FEATURES_PER_CARD}")
        print(f"MAX_CARDS_IN_GAME: {MAX_CARDS_IN_GAME}")
        print(f"OBSERVATION_SIZE: {OBSERVATION_SIZE}")
        print(f"\n=== Action Space ===")
        print(f"ACTION_SPACE_SIZE: {ACTION_SPACE_SIZE}")
        print(f"\n=== Effect Categories ===")
        print(f"NUM_ATTACK_EFFECT_CATEGORIES: {NUM_ATTACK_EFFECT_CATEGORIES}")
        print(f"NUM_ABILITY_EFFECT_CATEGORIES: {NUM_ABILITY_EFFECT_CATEGORIES}")
        print(f"NUM_SUPPORTER_EFFECT_CATEGORIES: {NUM_SUPPORTER_EFFECT_CATEGORIES}")
        print(f"\n=== Model Architecture ===")
        print(f"MODEL_EMBED_DIM_DEFAULT: {MODEL_EMBED_DIM_DEFAULT}")
        print(f"MODEL_NUM_HEADS_DEFAULT: {MODEL_NUM_HEADS_DEFAULT}")
        print(f"MODEL_NUM_LAYERS_DEFAULT: {MODEL_NUM_LAYERS_DEFAULT}")
        print(f"MODEL_ATTENTION_TEMPERATURE: {MODEL_ATTENTION_TEMPERATURE}")
        print(f"MODEL_INIT_RESIDUAL_SCALE: {MODEL_INIT_RESIDUAL_SCALE}")
        print(f"MODEL_POLICY_LAYERS_DEFAULT: {MODEL_POLICY_LAYERS_DEFAULT}")
        print(f"MODEL_VALUE_LAYERS_DEFAULT: {MODEL_VALUE_LAYERS_DEFAULT}")
        print(f"\n=== Diagnostic Thresholds ===")
        print(f"DIAG_GRADIENT_IMBALANCE_WARNING: {DIAG_GRADIENT_IMBALANCE_WARNING}")
        print(f"DIAG_ATTENTION_LOGIT_STD_WARNING: {DIAG_ATTENTION_LOGIT_STD_WARNING}")
        print(f"DIAG_ATTENTION_ENTROPY_WARNING: {DIAG_ATTENTION_ENTROPY_WARNING}")
        print(f"\n=== Reward ===")
        print(f"REWARD_WIN_BASE: {REWARD_WIN_BASE}")
        print(f"REWARD_LOSS_BASE: {REWARD_LOSS_BASE}")
        print(f"REWARD_SPEED_REFERENCE_TURNS: {REWARD_SPEED_REFERENCE_TURNS}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
