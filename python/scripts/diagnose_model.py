#!/usr/bin/env python3
"""Diagnostic script for attention model.
Usage: python python/scripts/diagnose_model.py path/to/model.zip"""

import numpy as np
import torch
import typer
from sb3_contrib import MaskablePPO
from deckgym.config import (
    DIAG_GRADIENT_IMBALANCE_WARNING,
    DIAG_GRADIENT_IMBALANCE_CRITICAL,
    DIAG_LARGE_GRADIENT_THRESHOLD,
    DIAG_WEIGHT_STD_MIN,
    DIAG_WEIGHT_STD_MAX,
    DIAG_ATTENTION_LOGIT_STD_WARNING,
    DIAG_ATTENTION_ENTROPY_WARNING,
)
from deckgym.attention_policy import diagnose_gradients


def analyze_gradients(model):
    """Analyze gradient magnitudes."""
    print("\n" + "=" * 70)
    print("GRADIENT ANALYSIS")
    print("=" * 70)

    model.policy.train()

    # Create dummy input
    obs = torch.randn(1, model.observation_space.shape[0]).to(model.device)

    # Forward pass
    features = model.policy.features_extractor(obs)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    action_logits = model.policy.action_net(latent_pi)

    # Backward pass
    loss = action_logits.sum()
    loss.backward()

    # Collect gradients
    print("\n  Gradient Norms:")
    grad_stats = {}
    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats[name] = grad_norm

            status = "[OK]"
            if grad_norm < 1e-6:
                status = "[CRITICAL] VANISHING"
            elif grad_norm > DIAG_LARGE_GRADIENT_THRESHOLD:
                status = "[WARNING] LARGE"

            print(f"    {name:55s}: {grad_norm:10.6f} {status}")

    # Check imbalance
    if grad_stats:
        max_grad = max(grad_stats.values())
        min_grad = min([v for v in grad_stats.values() if v > 1e-9])
        ratio = max_grad / min_grad if min_grad > 0 else float("inf")

        print(f"\n  Gradient Imbalance Ratio: {ratio:.1f}")
        if ratio > DIAG_GRADIENT_IMBALANCE_CRITICAL:
            print(f"  [CRITICAL] Ratio >{DIAG_GRADIENT_IMBALANCE_CRITICAL}")
        elif ratio > DIAG_GRADIENT_IMBALANCE_WARNING:
            print(f"  [WARNING] Ratio >{DIAG_GRADIENT_IMBALANCE_WARNING}")
        else:
            print("  [OK] Balanced")

        # Use the internal group diagnostics
        print("\n  Gradients by Component Group:")
        group_stats = diagnose_gradients(model.policy)
        for key, val in group_stats.items():
            if "_max" in key:
                name = key.replace("_max", "")
                print(f"    {name:25s}: Max={val:10.6f}")

        return ratio
    return None


def analyze_weights(model):
    """Analyze weight distributions."""
    print("\n" + "=" * 70)
    print("WEIGHT ANALYSIS")
    print("=" * 70)

    for name, param in model.policy.named_parameters():
        if "weight" in name and param.numel() > 100:
            weights = param.data.cpu().numpy().flatten()
            mean = weights.mean()
            std = weights.std()

            status = "[OK]"
            if abs(mean) > 0.1:
                status = "[WARNING] High mean"
            if std < DIAG_WEIGHT_STD_MIN or std > DIAG_WEIGHT_STD_MAX:
                status = "[WARNING] Unusual std"

            print(f"  {name:55s}: μ={mean:7.4f}, σ={std:6.4f} {status}")


def analyze_attention_layers(model):
    """Analyze attention-specific layers."""
    print("\n" + "=" * 70)
    print("ATTENTION LAYER ANALYSIS")
    print("=" * 70)

    # Check if model has attention
    has_attention = hasattr(model.policy.features_extractor, "attention_layers")

    if not has_attention:
        print("  [WARNING] No attention layers found (MLP model)")
        return

    print("  [OK] Attention layers present")

    # Perform a forward pass with tracking ENABLED
    model.policy.eval()
    obs = torch.randn(1, model.observation_space.shape[0]).to(model.device)
    with torch.no_grad():
        model.policy.features_extractor(obs, track_attention_stats=True)

    stats = model.policy.features_extractor.get_attention_stats()

    print("\n  Forward Pass Statistics (Tracked):")
    for layer_name, layer_stats in stats.items():
        logit_std = layer_stats["attn_logit_std"]
        entropy = layer_stats["attn_entropy"]

        std_status = "[OK]"
        if logit_std > DIAG_ATTENTION_LOGIT_STD_WARNING:
            std_status = "[WARNING] SATURATING (High Logits)"

        entropy_status = "[OK]"
        if entropy < DIAG_ATTENTION_ENTROPY_WARNING:
            entropy_status = "[WARNING] PEAKY (Low Entropy)"

        print(
            f"    {layer_name:10s}: Logit Std={logit_std:7.4f} {std_status}, Entropy={entropy:7.4f} {entropy_status}"
        )

    # Analyze attention weights
    print("\n  Attention Weights Analysis:")
    for name, param in model.policy.named_parameters():
        if (
            "attention" in name.lower()
            or "q_proj" in name
            or "k_proj" in name
            or "v_proj" in name
        ):
            weights = param.data.cpu().numpy().flatten()
            print(f"    {name:55s}: μ={weights.mean():7.4f}, σ={weights.std():6.4f}")


def main(model: str = typer.Argument(..., help="Path to the model .zip file")):
    """Diagnose an RL model: gradient norms, weight distributions, attention stats."""
    print("=" * 70)
    print("MODEL DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {model}\n")

    # Load model
    print("Loading model...")
    loaded = MaskablePPO.load(model, device="cpu")
    print(f"  Device: {loaded.device}")
    print(f"  Observation space: {loaded.observation_space.shape}")
    print(f"  Action space: {loaded.action_space.n}")

    # Run diagnostics
    grad_ratio = analyze_gradients(loaded)
    analyze_weights(loaded)
    analyze_attention_layers(loaded)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if grad_ratio and grad_ratio > DIAG_GRADIENT_IMBALANCE_CRITICAL:
        issues.append(
            f"Severe gradient imbalance (>{DIAG_GRADIENT_IMBALANCE_CRITICAL})"
        )

    if issues:
        print("  [WARNING] Issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] No critical issues!")


if __name__ == "__main__":
    typer.run(main)
