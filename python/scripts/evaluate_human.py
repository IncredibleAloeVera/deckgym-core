#!/usr/bin/env python3
"""
Human vs E2 Evaluation Script

Launches a series of TUI games for a human to play against e2 (Expectiminimax depth 2).
Uses random decks from the MetaDeckLoader for variety.
Tracks win/loss/draw statistics.

Usage:
    python scripts/evaluate_human.py                    # 50 games, meta decks
    python scripts/evaluate_human.py --games 20         # 20 games
    python scripts/evaluate_human.py --opponent e3      # vs e3 instead
    python scripts/evaluate_human.py --mirror           # Both players use same deck
"""

import argparse
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add python directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from deckgym.deck_loader import MetaDeckLoader


def create_temp_deck_file(deck_string: str) -> Path:
    """Create a temporary deck file from deck string."""
    temp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    temp.write(deck_string)
    temp.close()
    return Path(temp.name)


def run_game(
    deck_a_path: Path,
    deck_b_path: Path,
    human_position: int,
    opponent: str,
    seed: int = None,
) -> str | None:
    """
    Run a single TUI game and return result.

    Returns:
        "win", "loss", "draw", or None if game was quit early
    """
    # Build players string: human is 'h', opponent is the bot code (e2, e3, etc.)
    if human_position == 0:
        players = f"h,{opponent}"
    else:
        players = f"{opponent},h"

    cmd = [
        "cargo",
        "run",
        "--bin",
        "tui",
        "--features",
        "tui",
        "--release",
        "--",
        str(deck_a_path),
        str(deck_b_path),
        "--players",
        players,
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"\n{'='*60}")
    print(f"Starting game... (You are Player {human_position + 1})")
    print(f"Press 'q' or ESC to quit the game when finished")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        # Game completed - need to ask for result since TUI doesn't return it
        return None  # Will prompt user
    except subprocess.CalledProcessError:
        return None
    except KeyboardInterrupt:
        return "quit"


def prompt_result() -> str:
    """Prompt user for game result."""
    while True:
        result = (
            input("\nGame result? [w]in / [l]oss / [d]raw / [q]uit session: ")
            .strip()
            .lower()
        )
        if result in ("w", "win"):
            return "win"
        elif result in ("l", "loss"):
            return "loss"
        elif result in ("d", "draw"):
            return "draw"
        elif result in ("q", "quit"):
            return "quit"
        print("Invalid input. Please enter w/l/d/q.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate human skill against Expectiminimax bots"
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=50,
        help="Number of games to play (default: 50)",
    )
    parser.add_argument(
        "--opponent",
        "-o",
        type=str,
        default="e2",
        help="Opponent bot code: e1, e2, e3, e4, r, w, er, aa, v (default: e2)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Both players use the same deck (mirror match)",
    )
    parser.add_argument(
        "--deck-path",
        type=str,
        default=None,
        help="Path to deck JSON directory (default: archetypes_by_era)",
    )
    parser.add_argument(
        "--human-first",
        action="store_true",
        help="Human always plays first (default: alternates)",
    )

    args = parser.parse_args()

    # Load deck loader
    deck_json = args.deck_path or str(PROJECT_ROOT / "archetypes_by_era")
    if not Path(deck_json).exists():
        print(f"ERROR: Deck file not found: {deck_json}")
        sys.exit(1)

    loader = MetaDeckLoader(deck_json)
    print(f"Loaded {len(loader.decks)} decks from {len(loader.archetypes)} archetypes")

    # Statistics
    stats = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
    game_log = []

    print(f"\n{'='*60}")
    print(f"HUMAN VS {args.opponent.upper()} EVALUATION")
    print(f"{'='*60}")
    print(f"Games: {args.games}")
    print(f"Opponent: {args.opponent}")
    print(f"Mirror mode: {'ON' if args.mirror else 'OFF'}")
    print(f"{'='*60}\n")

    try:
        for game_num in range(1, args.games + 1):
            print(f"\n--- GAME {game_num}/{args.games} ---")
            print(
                f"Current stats: W:{stats['wins']} L:{stats['losses']} D:{stats['draws']}"
            )

            # Sample decks
            deck_a_info = loader.sample_deck_info(mode="hierarchical")
            if args.mirror:
                deck_b_info = deck_a_info
            else:
                deck_b_info = loader.sample_deck_info(mode="hierarchical")

            print(
                f"Deck A: {deck_a_info.archetype}"
            )
            print(
                f"Deck B: {deck_b_info.archetype}"
            )

            # Create temp deck files
            deck_a_path = create_temp_deck_file(deck_a_info.deck_string)
            deck_b_path = create_temp_deck_file(deck_b_info.deck_string)

            try:
                # Alternate who plays first (or always human first if requested)
                if args.human_first:
                    human_position = 0
                else:
                    human_position = (game_num - 1) % 2

                # Run the game
                run_game(deck_a_path, deck_b_path, human_position, args.opponent)

                # Get result from user
                result = prompt_result()

                if result == "quit":
                    print("\nSession ended early by user.")
                    break

                # Record stats
                stats["games"] += 1
                if result == "win":
                    stats["wins"] += 1
                elif result == "loss":
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1

                game_log.append(
                    {
                        "game": game_num,
                        "result": result,
                        "human_position": human_position,
                        "deck_a": deck_a_info.archetype,
                        "deck_b": deck_b_info.archetype,
                    }
                )

            finally:
                # Clean up temp files
                deck_a_path.unlink(missing_ok=True)
                deck_b_path.unlink(missing_ok=True)

    except KeyboardInterrupt:
        print("\n\nSession interrupted.")

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Games played: {stats['games']}")
    print(f"Wins:   {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"Draws:  {stats['draws']}")

    if stats["games"] > 0:
        total = stats["games"]
        win_rate = stats["wins"] / total * 100
        loss_rate = stats["losses"] / total * 100
        draw_rate = stats["draws"] / total * 100

        print(f"\nWin rate: {win_rate:.1f}%")
        print(f"Loss rate: {loss_rate:.1f}%")
        print(f"Draw rate: {draw_rate:.1f}%")

        # Interpretation
        print(f"\n--- INTERPRETATION vs {args.opponent.upper()} ---")
        if win_rate >= 60:
            print(f"🏆 ABOVE {args.opponent.upper()} level - You're stronger!")
        elif win_rate >= 45:
            print(f"⚖️  EQUIVALENT to {args.opponent.upper()} - Evenly matched")
        elif win_rate >= 30:
            print(f"📉 BELOW {args.opponent.upper()} level - Room for improvement")
        else:
            print(f"❌ SIGNIFICANTLY below {args.opponent.upper()} - Keep practicing!")

    # Save log
    log_path = (
        PROJECT_ROOT
        / f"human_eval_{args.opponent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(log_path, "w") as f:
        json.dump(
            {
                "opponent": args.opponent,
                "stats": stats,
                "games": game_log,
            },
            f,
            indent=2,
        )
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
