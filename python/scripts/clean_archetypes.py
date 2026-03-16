import json
import subprocess
import sys
from pathlib import Path


def get_incomplete_cards():
    """Run cargo bin card_status to get list of incomplete card IDs."""
    print("Running cargo run --bin card_status -- --incomplete-only...")
    try:
        result = subprocess.run(
            ["cargo", "run", "--bin", "card_status", "--", "--incomplete-only"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        incomplete_ids = set()
        for line in result.stdout.splitlines():
            # Lines look like: "A1 001   Card Name  Reason"
            # We need to extract the ID: "A1 001" -> "a1-001"
            parts = line.split()
            if len(parts) >= 2:
                set_code = parts[0].lower()
                number = parts[1]
                if number.isdigit():
                    # card_status outputs e.g. "P-A 064"; JSON uses "pa-064"
                    # Strip hyphens from the set code to match the JSON format.
                    set_code_normalized = set_code.replace("-", "")
                    incomplete_ids.add(f"{set_code_normalized}-{number}")

        # Manually exclude missing P-B promos (026-032)
        missing_pb_promos = {f"pb-0{i}" for i in range(26, 33)}
        incomplete_ids.update(missing_pb_promos)

        print(
            f"Found {len(incomplete_ids)} incomplete card IDs "
            f"(including {len(missing_pb_promos)} manual exclusions)."
        )
        return incomplete_ids
    except subprocess.CalledProcessError as e:
        print(f"Error running card_status: {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)


def clean_era_file(path: Path, incomplete_ids: set) -> tuple[int, int]:
    """
    Filter a single archetypes_by_era JSON file in-place.

    Each file is a flat list of archetypes:
        [ { "name": "...", "lists": [ { "cards": ["count:id", ...] } ] }, ... ]

    Returns (total_decks, removed_decks).
    """
    with open(path, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    total_decks = 0
    removed_decks = 0
    clean_archetypes = []

    for arch in archetypes:
        clean_lists = []
        for deck in arch.get("lists", []):
            total_decks += 1
            cards = deck.get("cards", [])

            # card_entry format: "count:id"  e.g. "2:pa-065"
            has_incomplete = any(
                card_entry.split(":", 1)[1] in incomplete_ids
                for card_entry in cards
                if ":" in card_entry
            )

            if has_incomplete:
                removed_decks += 1
            else:
                clean_lists.append(deck)

        if clean_lists:
            clean_archetypes.append({**arch, "lists": clean_lists})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_archetypes, f, indent=2)

    return total_decks, removed_decks


def main():
    """Filter all archetypes_by_era/*.json files in-place."""
    repo_root = Path(__file__).parent.parent.parent
    era_dir = repo_root / "archetypes_by_era"

    if not era_dir.is_dir():
        print(f"Directory not found: {era_dir}")
        sys.exit(1)

    incomplete_ids = get_incomplete_cards()

    json_files = sorted(era_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {era_dir}")
        sys.exit(1)

    total_all, removed_all = 0, 0
    for json_file in json_files:
        total, removed = clean_era_file(json_file, incomplete_ids)
        kept = total - removed
        print(
            f"  {json_file.name:20s}  total={total:6d}  removed={removed:5d}  kept={kept:6d}"
        )
        total_all += total
        removed_all += removed

    print(
        f"\nDone. Total: {total_all} decks, removed {removed_all}, "
        f"kept {total_all - removed_all}."
    )


if __name__ == "__main__":
    main()
