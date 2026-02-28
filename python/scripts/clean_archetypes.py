import json
import subprocess
import os
import sys
from pathlib import Path
import typer

def get_incomplete_cards():
    """Run cargo bin card_status to get list of incomplete cards."""
    print("Running cargo run --bin card_status -- --incomplete-only...")
    try:
        result = subprocess.run(
            ["cargo", "run", "--bin", "card_status", "--", "--incomplete-only"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).parent.parent.parent)
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
                    incomplete_ids.add(f"{set_code}-{number}")
        
        # Manually exclude missing P-B promos (026-032)
        missing_pb_promos = {f"pb-0{i}" for i in range(26, 33)}
        incomplete_ids.update(missing_pb_promos)

        print(f"Found {len(incomplete_ids)} incomplete card IDs (including {len(missing_pb_promos)} manual exclusions).")
        return incomplete_ids
    except subprocess.CalledProcessError as e:
        print(f"Error running card_status: {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

def clean_archetypes(input_path, output_path, incomplete_ids):
    """Filter archetypes to exclude decks with incomplete cards."""
    print(f"Loading archetypes from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clean_data = {}
    total_decks = 0
    removed_decks = 0
    
    for era, archetypes in data.items():
        clean_era_archetypes = []
        for arch in archetypes:
            clean_lists = []
            for deck in arch.get('lists', []):
                total_decks += 1
                cards = deck.get('cards', [])
                
                # Check if any card in the deck is incomplete
                is_incomplete = False
                for card_entry in cards:
                    # card_entry format is "count:id" (e.g. "2:a1-001")
                    if ":" in card_entry:
                        _, card_id = card_entry.split(":", 1)
                        if card_id in incomplete_ids:
                            is_incomplete = True
                            break
                
                if not is_incomplete:
                    clean_lists.append(deck)
                else:
                    removed_decks += 1
            
            if clean_lists:
                new_arch = arch.copy()
                new_arch['lists'] = clean_lists
                clean_era_archetypes.append(new_arch)
        
        if clean_era_archetypes:
            clean_data[era] = clean_era_archetypes

    print(f"Stats: Total decks: {total_decks}, Removed: {removed_decks}, Kept: {total_decks - removed_decks}")
    print(f"Writing clean data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2)

def main():
    """Filter archetype dataset to exclude decks containing incomplete cards."""
    repo_root = Path(__file__).parent.parent.parent
    input_file = repo_root / "ultimate-archetypes.json"
    output_file = repo_root / "ultimate-archetypes-clean.json"

    incomplete_cards = get_incomplete_cards()
    clean_archetypes(input_file, output_file, incomplete_cards)


if __name__ == "__main__":
    typer.run(main)
