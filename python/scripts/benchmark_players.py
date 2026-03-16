#!/usr/bin/env python3
"""
Benchmark script to compare games per second of various player baselines.
Usage: python python/scripts/benchmark_players.py
"""

import subprocess
import re
import sys
import typer

# Players to benchmark (excluding mcts and human)
PLAYERS = [
    ("aa", "AttachAttack"),
    ("et", "EndTurn"),
    ("r", "Random"),
    ("w", "WeightedRandom"),
    ("v", "ValueFunction"),
    ("er", "EvolutionRusher"),
    ("e2", "ExpectiMiniMax (depth 2)"),
    ("e3", "ExpectiMiniMax (depth 3)"),
    ("o1c", "ONNX CPU"),
    ("o1g", "ONNX GPU"),
]

NUM_GAMES = 1000
# Deck content should not matter
DECK_A = "example_decks/venusaur-exeggutor.txt"
DECK_B = "example_decks/altaria.txt"


def run_benchmark(player_code: str) -> float:
    """Run benchmark for a player and return total_time_sec."""
    cmd = ["cargo", "run", "--release"]

    # Add --features onnx for ONNX-based players
    if player_code.startswith("o"):
        cmd.extend(["--features", "onnx"])

    cmd.extend(
        [
            "simulate",
            DECK_A,
            DECK_B,
            "--num",
            str(NUM_GAMES),
            "--players",
            f"{player_code},{player_code}",
            "--parallel",
        ]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse total time - format: "Ran X simulations in Xm Xs Xms..." or "Xs Xms..."
    total_time_sec = 0.0

    # Try to parse "Xm Xs Xms" format
    time_match = re.search(r"in (\d+)m (\d+)s (\d+)ms", output)
    if time_match:
        total_time_sec = (
            int(time_match.group(1)) * 60
            + int(time_match.group(2))
            + int(time_match.group(3)) / 1000.0
        )
    else:
        # Try "Xs Xms" format
        time_match = re.search(r"in (\d+)s (\d+)ms", output)
        if time_match:
            total_time_sec = (
                int(time_match.group(1)) + int(time_match.group(2)) / 1000.0
            )
        else:
            # Try "Xms" only format
            time_match = re.search(r"in (\d+)ms", output)
            if time_match:
                total_time_sec = int(time_match.group(1)) / 1000.0

    return total_time_sec


def main():
    print("=" * 60)
    print("Player Baseline Benchmark (Games per Second)")
    print(f"Running {NUM_GAMES} games per player")
    print("=" * 60)
    print()

    results = []

    for code, name in PLAYERS:
        print(f"Benchmarking {name:.<30}", end=" ", flush=True)
        try:
            total_time = run_benchmark(code)
            gps = NUM_GAMES / total_time if total_time > 0 else 0.0
            results.append((name, code, gps, total_time))
            print(f"{gps:>10.1f} games/s ({total_time:.2f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, code, 0.0, 0.0))

    print()
    print("=" * 60)
    print("Results (sorted by games/second)")
    print("=" * 60)
    print(f"{'Player':<30} {'Code':<6} {'Games/s':>12} {'Time (s)':>10}")
    print("-" * 60)

    for name, code, gps, time in sorted(results, key=lambda x: -x[2]):
        print(f"{name:<30} {code:<6} {gps:>12.1f} {time:>10.2f}")


if __name__ == "__main__":
    typer.run(main)
