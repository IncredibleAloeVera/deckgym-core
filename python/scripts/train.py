#!/usr/bin/env python3
"""
Train a Pokemon TCG Pocket RL agent using MaskablePPO with self-play.

The agent learns by playing against a pool of frozen copies of itself following the Prioritized Fictitious Self-Play (PFSP) method, with different anchors ranging from random, heuristic to established baseline models, following a curriculum.

Starting player alternates randomly each game (controlled by game seed).
"""

# Suppress warnings BEFORE importing anything else
import warnings

warnings.filterwarnings("ignore", message=".*nested tensors.*")
warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
import torch
import typer
import yaml

# Enable TensorFloat32 for faster matmul on Ampere+ GPUs
torch.set_float32_matmul_precision("high")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from deckgym.envs import DeckGymEnv, BatchedDeckGymEnv, SelfPlayEnv
from deckgym.deck_loader import MetaDeckLoader
from deckgym.models.extractors import (
    CardAttentionExtractor,
    create_attention_policy_kwargs,
)
from deckgym.callbacks import PFSPCallback, EpisodeMetricsCallback, FrozenOpponentCallback, PauseResumeCallback, InterpretabilityCallback

# Import configuration and constants
from deckgym.config import TrainingConfig, DEFAULT_CONFIG, OBSERVATION_SIZE

# =============================================================================
# Configuration
# =============================================================================


# Configuration is now centralized in deckgym/config.py


# =============================================================================
# Self-Play Environment (Moved to deckgym/envs/self_play.py)
# =============================================================================

# SelfPlayEnv is now imported from deckgym.envs.SelfPlayEnv



# =============================================================================
# Training Callbacks (Moved to deckgym/callbacks/)
# =============================================================================

# EpisodeMetricsCallback, FrozenOpponentCallback, PFSPCallback are now imported
# from deckgym.callbacks



# =============================================================================
# Action Masking & Environment Factory
# =============================================================================


def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    """Provide action mask to MaskablePPO."""
    return env.action_masks()


def make_env(
    deck_loader: MetaDeckLoader,
    config: TrainingConfig,
) -> callable:
    """
    Factory function to create training environments.

    Args:
        deck_loader: Deck loader for sampling
        config: Training config

    Uses self-play (opponent_type=None) by default.
    """

    def _init() -> gym.Env:
        env = SelfPlayEnv(deck_loader, opponent_type=None, config=config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    return _init


# =============================================================================
# Training Loop
# =============================================================================


def train(config: TrainingConfig = DEFAULT_CONFIG):
    """
    Train the RL agent with MaskablePPO.

    Args:
        config: Training configuration object
    """
    print("=" * 60)
    print("Pokemon TCG Pocket RL Training")
    print("=" * 60)

    # Build config dict for logging/saving
    config_dict = {
        # Paths
        "meta_decks_dir": config.meta_decks_dir,
        "save_path": config.save_path,
        "checkpoint_dir": config.checkpoint_dir,
        "tensorboard_dir": config.tensorboard_dir,
        "resume_path": config.resume_path,
        # Training duration
        "total_timesteps": config.total_timesteps,
        "checkpoint_freq": config.checkpoint_freq,
        # PPO hyperparameters
        "base_learning_rate": config.base_learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "clip_range": config.clip_range,
        "max_grad_norm": config.max_grad_norm,
        "target_kl": config.target_kl,
        # Network architecture
        "use_attention": config.use_attention,
        "attention_embed_dim": config.attention_embed_dim,
        "attention_num_heads": config.attention_num_heads,
        "attention_num_layers": config.attention_num_layers,
        "policy_layers": list(config.policy_layers),
        "value_layers": list(config.value_layers),
        "use_silu": config.use_silu,
        # Self-play
        "frozen_opponent_update_rollouts": config.frozen_opponent_update_rollouts,
        # Game limits
        "max_turns": config.max_turns,
        "max_actions_per_turn": config.max_actions_per_turn,
        "max_total_actions": config.max_total_actions,
        # Environment
        "n_envs": config.n_envs,
        "use_batched_env": config.use_batched_env,
        "device": config.device,
        "brutal_resume": config.brutal_resume,
    }

    from deckgym.diagnostic_logger import get_logger

    diag_logger = get_logger()
    diag_logger.setup_excepthook()

    print(f"\n{'─' * 60}")
    print("  TRAINING CONFIGURATION (Self-Play)")
    print(f"{'─' * 60}")

    print("\n  [▸] Paths:")
    print(f"     Meta decks:        {config.meta_decks_dir}")
    print(f"     Checkpoint dir:    {config.checkpoint_dir}")
    if config.resume_path:
        print(f"     Resume from:       {config.resume_path}")
        print(f"     Brutal resume:     {config.brutal_resume}")

    print("\n  [◷] Training Duration:")
    print(f"     Total timesteps:   {config.total_timesteps:,}")
    print(f"     Checkpoint freq:   {config.checkpoint_freq:,}")

    print("\n  [⬢] PPO Hyperparameters:")
    print(
        f"     Learning rate:     0 → {config.base_learning_rate} → {config.min_learning_rate} (warmup {config.warmup_ratio:.0%})"
    )
    print(f"     Batch size:        {config.batch_size:,}")
    print(
        f"     Rollout buffer:    {config.n_steps * config.n_envs:,} ({config.n_steps} steps × {config.n_envs} envs)"
    )
    print(f"     N epochs:          {config.n_epochs}")
    print(f"     Gradient clip:     {config.max_grad_norm}")

    print("\n  [△] Network Architecture:")
    if config.use_attention:
        print(f"     Type:              Attention (CardAttentionExtractor)")
        print(f"     Embed dim:         {config.attention_embed_dim}")
        print(f"     Heads:             {config.attention_num_heads}")
        print(f"     Layers:            {config.attention_num_layers}")
        print(f"     Policy head:       {list(config.policy_layers)}")
        print(f"     Value head:        {list(config.value_layers)}")
    else:
        print(f"     Type:              MLP")
        print(f"     Policy layers:     {config.policy_layers}")
        print(f"     Value layers:      {config.value_layers}")

    print("\n  [↻] Self-Play:")
    print(
        f"     Opponent update:   every {config.frozen_opponent_update_rollouts} rollouts"
    )
    print(f"     Mode:              Pure self-play (ONNX frozen opponent)")

    print(f"{'─' * 60}\n")

    # Setup environment(s)
    print("[1/4] Loading meta decks...")
    deck_loader = MetaDeckLoader(config.meta_decks_dir)

    print(f"[2/4] Creating {config.n_envs} environment(s) for self-play...")
    if config.n_envs == 1:
        # Single environment (simpler)
        single_env = SelfPlayEnv(deck_loader, opponent_type="self", config=config)
        env = ActionMasker(single_env, mask_fn)
        env = Monitor(env)
    elif config.use_batched_env:
        # High-performance Rust-side batched environment
        print(f"      Using BatchedDeckGymEnv (Rust-side batching)")
        env = BatchedDeckGymEnv(
            n_envs=config.n_envs,
            deck_loader=deck_loader,
            opponent_type=None,  # Self-play via ONNX
            config=config,
        )
        single_env = None
    else:
        # Legacy: Multiple environments with DummyVecEnv
        env = DummyVecEnv([make_env(deck_loader, config) for _ in range(config.n_envs)])
        single_env = None

    # Verify observation space
    expected_obs_size = OBSERVATION_SIZE
    if env.observation_space.shape[0] != expected_obs_size:
        raise ValueError(
            f"Observation size mismatch! Expected {expected_obs_size}, "
            f"but environment returned {env.observation_space.shape[0]}. "
            "Check observation.rs and attention_policy.py constants."
        )
    print(f"      Observation space verified: {env.observation_space.shape}")

    # Setup model
    print("[3/4] Initializing model...")
    if torch.cuda.is_available():
        print(f"      PyTorch Training Device: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        print(f"      PyTorch Training Device: CPU")
    activation_fn = torch.nn.SiLU if config.use_silu else torch.nn.ReLU

    if config.use_attention:
        # Use attention-based policy with CardAttentionExtractor
        print(
            f"      Using attention policy (embed_dim={config.attention_embed_dim}, heads={config.attention_num_heads}, layers={config.attention_num_layers})"
        )
        policy_kwargs = create_attention_policy_kwargs(config)
    else:
        # Use standard MLP policy
        policy_kwargs = dict(
            net_arch=dict(
                pi=list(config.policy_layers),
                vf=list(config.value_layers),
            ),
            activation_fn=activation_fn,
        )

    if config.resume_path:
        # Resume from existing checkpoint
        print(f"      Resuming from: {config.resume_path}")
        model = MaskablePPO.load(
            config.resume_path,
            env=env,
            device=config.device,
            tensorboard_log=config.tensorboard_dir,
        )
        # Update hyperparameters for fine-tuning
        model.learning_rate = config.get_learning_rate_schedule(config.total_timesteps)
        model.n_steps = config.n_steps
        model.batch_size = config.batch_size
        model.n_epochs = config.n_epochs
        model.gamma = config.gamma
        model.gae_lambda = config.gae_lambda
        model.ent_coef = config.ent_coef
        model.vf_coef = config.vf_coef
        # clip_range must be a callable (schedule) for SB3
        model.clip_range = lambda _: config.clip_range
        model.target_kl = config.target_kl
        print(
            f"      Updated hyperparameters (LR={config.base_learning_rate}, target_kl={config.target_kl})"
        )
    else:
        # Create new model from scratch
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config.get_learning_rate_schedule(config.total_timesteps),
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,  # Gradient clipping
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range,  # Clip value function updates to reduce spikes
            target_kl=config.target_kl,  # Stop epoch early if KL too high
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config.device,
            tensorboard_log=config.tensorboard_dir,
        )

    # NOTE: torch.compile() is intentionally NOT used because it breaks model saving/loading.
    # The compiled model saves weights with "_orig_mod." prefix which SB3 cannot properly reload.
    # This results in models loading with random weights instead of trained weights.

    # Save training configuration metadata
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    config_path = os.path.join(config.checkpoint_dir, "training_config.json")
    with open(config_path, "w") as f:
        import json

        json.dump(config_dict, f, indent=2)
    print(f"      Config saved to {config_path}")

    # Also save to the TensorBoard run folder (MaskablePPO_*)
    if hasattr(model, "logger") and model.logger is not None:
        tb_log_dir = model.logger.dir
        if tb_log_dir:
            config_tb_path = os.path.join(tb_log_dir, "training_config.json")
            with open(config_tb_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            print(f"      Config saved to {config_tb_path}")

    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="rl_bot",
        ),
        EpisodeMetricsCallback(verbose=0),
        InterpretabilityCallback(verbose=0),
        PauseResumeCallback(verbose=1),
    ]

    # Add PFSP or simple frozen opponent callback
    if config.use_pfsp:
        print(f"      Using PFSP with pool_size={config.pfsp_pool_size}")
        callbacks.append(
            PFSPCallback(
                env,
                n_envs=config.n_envs,
                pool_size=config.pfsp_pool_size,
                add_to_pool_every_n_rollouts=config.pfsp_add_every_n_rollouts,
                priority_exponent=config.pfsp_priority_exponent,
                checkpoint_dir=config.pfsp_checkpoint_dir,
                baseline_curriculum=config.pfsp_baseline_curriculum,
                baseline_max_allocation=config.pfsp_baseline_max_allocation,
                brutal_resume=config.brutal_resume,
                verbose=1,
            )
        )
    else:
        callbacks.append(
            FrozenOpponentCallback(
                env,
                n_envs=config.n_envs,
                update_every_n_rollouts=config.frozen_opponent_update_rollouts,
                verbose=1,
            )
        )

    # Initialize frozen opponent with initial weights
    print("      Initializing frozen opponent...")
    if isinstance(env, BatchedDeckGymEnv):
        try:
            from deckgym.onnx_export import export_policy_to_onnx

            onnx_path = os.path.join(tempfile.gettempdir(), "frozen_opponent.onnx")
            export_policy_to_onnx(model, onnx_path, validate=False)
            env.vec_game.set_onnx_opponent(
                onnx_path, deterministic=False, device=config.onnx_device
            )
            print(f"      ONNX opponent initialized")
        except Exception as e:
            print(f"      [WARNING] Could not initialize ONNX opponent: {e}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "initial_frozen")
            model.save(path)
            initial_frozen = MaskablePPO.load(path, device="cpu")

        if config.n_envs == 1:
            env.env.env.set_opponent_model(initial_frozen)
        else:
            for i in range(config.n_envs):
                env.envs[i].env.env.set_opponent_model(initial_frozen)
        print(f"      CPU frozen opponent initialized")

    # Train
    print("[4/4] Starting training...")
    print(f"      Checkpoints: every {config.checkpoint_freq:,} steps")
    if config.use_pfsp:
        print(
            f"      PFSP: pool_size={config.pfsp_pool_size}, priority_exp={config.pfsp_priority_exponent}"
        )
        print(
            f"      PFSP: add every {config.pfsp_add_every_n_rollouts} rollouts, select every {config.pfsp_select_every_n_rollouts} rollouts"
        )
    else:
        print(
            f"      Opponent updates: every {config.frozen_opponent_update_rollouts} rollouts"
        )

    with diag_logger.capture_panic(env_provider=lambda: env):
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

    # Save final model
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    model.save(config.save_path)
    print(f"\nModel saved to {config.save_path}")
    print("Training complete!")


# =============================================================================
# Entry Point
# =============================================================================

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def cli(
    # Config file
    config: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config file"),
    # Paths
    meta: Optional[str] = typer.Option(None, "--meta", help="Meta decks directory"),
    save: Optional[str] = typer.Option(None, "--save", help="Model save path"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Path to checkpoint to resume from"),
    # Training
    steps: Optional[int] = typer.Option(None, "--steps", help="Total training steps"),
    checkpoint_freq: Optional[int] = typer.Option(None, "--checkpoint-freq", help="Checkpoint frequency"),
    # PPO
    lr: Optional[float] = typer.Option(None, "--lr", help="Peak learning rate (after warmup)"),
    min_lr: Optional[float] = typer.Option(None, "--min-lr", help="Final learning rate"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size"),
    n_steps: Optional[int] = typer.Option(None, "--n-steps", help="Steps per env per rollout"),
    n_epochs: Optional[int] = typer.Option(None, "--n-epochs", help="PPO epochs per update"),
    ent_coef: Optional[float] = typer.Option(None, "--ent-coef", help="Entropy coefficient"),
    # Self-play
    opponent_update_rollouts: Optional[int] = typer.Option(None, "--opponent-update-rollouts", help="Update frozen opponent every N rollouts"),
    # Parallelization
    n_envs: Optional[int] = typer.Option(None, "--n-envs", help="Number of parallel environments"),
    no_batched_env: bool = typer.Option(False, "--no-batched-env", help="Disable Rust-side batching"),
    # Attention
    no_attention: bool = typer.Option(False, "--no-attention", help="Disable attention, use MLP instead"),
    attention_dim: Optional[int] = typer.Option(None, "--attention-dim", help="Attention embedding dimension"),
    attention_heads: Optional[int] = typer.Option(None, "--attention-heads", help="Number of attention heads"),
    attention_layers: Optional[int] = typer.Option(None, "--attention-layers", help="Number of transformer layers"),
    # Device
    device: str = typer.Option("auto", "--device", help="Training device: cpu, cuda, auto"),
):
    """Train the Pokemon TCG Pocket RL agent."""
    _defaults = DEFAULT_CONFIG

    # Load config from YAML if provided, otherwise use defaults
    if config:
        print(f"Loading configuration from {config}...")
        cfg = TrainingConfig.from_yaml(str(config))
        print(f"  Loaded config: {config.stem}")
    else:
        cfg = TrainingConfig()

    # Override config with CLI arguments (only when explicitly provided)
    if meta is not None:
        cfg.meta_decks_dir = meta
    if save is not None:
        cfg.save_path = save
    if resume is not None:
        cfg.resume_path = resume
    if steps is not None:
        cfg.total_timesteps = steps
    if checkpoint_freq is not None:
        cfg.checkpoint_freq = checkpoint_freq
    if lr is not None:
        cfg.base_learning_rate = lr
    if min_lr is not None:
        cfg.min_learning_rate = min_lr
    if batch_size is not None:
        cfg.batch_size = batch_size
    if n_steps is not None:
        cfg.n_steps = n_steps
    if n_epochs is not None:
        cfg.n_epochs = n_epochs
    if ent_coef is not None:
        cfg.ent_coef = ent_coef
    if opponent_update_rollouts is not None:
        cfg.frozen_opponent_update_rollouts = opponent_update_rollouts
    if n_envs is not None:
        cfg.n_envs = n_envs
    if no_batched_env:
        cfg.use_batched_env = False
    if no_attention:
        cfg.use_attention = False
    if attention_dim is not None:
        cfg.attention_embed_dim = attention_dim
    if attention_heads is not None:
        cfg.attention_num_heads = attention_heads
    if attention_layers is not None:
        cfg.attention_num_layers = attention_layers
    if device != "auto":
        cfg.device = device

    train(cfg)


if __name__ == "__main__":
    app()
