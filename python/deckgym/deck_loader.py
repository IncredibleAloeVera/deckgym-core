"""
MetaDeckLoader - Load competitive decks from JSON for RL training.

Supports two JSON formats:
1. Simple format (simple_deck.json): flat list of decks with energy_type
2. Meta format (meta_deck.json): archetypes with score/strength per deck

Sampling strategy for generalization:
- Hierarchical: First pick archetype, then deck (ensures archetype diversity)
- Weighted: Sample weighted by deck strength

Meta deck data sourced from:
  https://github.com/chase-manning/pokemon-tcg-pocket-tier-list
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DeckInfo:
    """Information about a deck."""

    archetype: str
    deck_string: str


@dataclass
class ArchetypeInfo:
    """Aggregated statistics for an archetype."""

    name: str
    decks: list[DeckInfo] = field(default_factory=list)

    @property
    def deck_count(self) -> int:
        return len(self.decks)


class MetaDeckLoader:
    """
    Load and sample competitive decks with diversity-focused sampling.

    Sampling modes:
    - uniform: Random deck (may oversample popular archetypes)
    - hierarchical: Random archetype, then random deck (archetype diversity)
    """

    def __init__(
        self,
        json_dir: str,
        max_archetypes: int = None,
        max_decks_per_archetype: int = None,
    ):
        """
        Load decks from a directory containing JSON files (one per era).

        Args:
            json_dir: Path to the directory containing deck JSON files
            max_archetypes: Limit archetypes (None = load all)
            max_decks_per_archetype: Limit decks per archetype (None = load all)
        """
        self.decks: list[DeckInfo] = []
        self.archetypes: dict[str, ArchetypeInfo] = {}
        self.eras: dict[str, list[str]] = {}  # era_name -> list of archetype_names

        dir_path = Path(json_dir)
        if not dir_path.is_dir():
            raise ValueError(f"{json_dir} is not a valid directory.")

        for json_file in sorted(dir_path.glob("*.json")):
            era_name = json_file.stem
            with open(json_file, "r", encoding="utf-8") as f:
                archetypes_data = json.load(f)
            self._load_era_data(era_name, archetypes_data, max_archetypes, max_decks_per_archetype)

    def _load_era_data(
        self, era_name: str, archetypes_data: list, max_archetypes: int = None, max_decks_per_archetype: int = None
    ):
        """Load decks from a parsed JSON list for a specific era."""
        current_era_archetypes = []

        if max_archetypes:
            archetypes_data = archetypes_data[:max_archetypes]

        for arch_data in archetypes_data:
            archetype_name = arch_data.get("name", "Unknown")
            lists_data = arch_data.get("lists", [])

            if max_decks_per_archetype:
                lists_data = lists_data[:max_decks_per_archetype]

            # Only add archetype if it has at least one deck list
            if not lists_data:
                continue

            if archetype_name not in self.archetypes:
                self.archetypes[archetype_name] = ArchetypeInfo(name=archetype_name)
                current_era_archetypes.append(archetype_name)

            for deck_data in lists_data:
                cards = deck_data.get("cards", [])
                if not cards:
                    continue

                # Compact format: "count:set-number"
                deck_string = self._cards_to_string_compact(cards)
                if deck_string is None:
                    continue

                deck_info = DeckInfo(
                    archetype=archetype_name,
                    deck_string=deck_string,
                )

                self.decks.append(deck_info)
                self.archetypes[archetype_name].decks.append(deck_info)

        if current_era_archetypes:
            self.eras[era_name] = current_era_archetypes

    def _cards_to_string_compact(self, cards: list[str]) -> str:
        """Convert compact list ["2:a1-001", ...] to deck string."""
        lines = []
        for card_entry in cards:
            if ":" not in card_entry:
                continue
            count, card_id = card_entry.split(":", 1)
            if "-" not in card_id:
                continue
            set_code, number = card_id.split("-", 1)
            lines.append(f"{count} {set_code.upper()} {number}")
        return "\n".join(lines)



    # =========================================================================
    # Sampling Methods
    # =========================================================================

    def sample_deck(self, mode: str = "hierarchical") -> str:
        """
        Sample a deck with the specified strategy.

        Args:
            mode: Sampling mode
                - "uniform": Pure random (may oversample popular archetypes)
                - "hierarchical": Random archetype, then random deck
        """
        if mode == "uniform":
            return random.choice(self.decks).deck_string
        elif mode == "hierarchical":
            return self._sample_hierarchical()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _sample_hierarchical(self) -> str:
        """Sample: random era → random archetype in era → random deck."""
        era_name = random.choice(list(self.eras.keys()))
        archetype_name = random.choice(self.eras[era_name])
        archetype = self.archetypes[archetype_name]
        deck = random.choice(archetype.decks)
        return deck.deck_string


    def sample_deck_info(self, mode: str = "hierarchical") -> DeckInfo:
        """Sample a deck with full info."""
        if mode == "hierarchical":
            era_name = random.choice(list(self.eras.keys()))
            archetype_name = random.choice(self.eras[era_name])
            return random.choice(self.archetypes[archetype_name].decks)
        return random.choice(self.decks)

    def sample_n_deck_info(self, n: int, mode: str = "hierarchical") -> list[DeckInfo]:
        """
        Sample N decks with the specified strategy.
        
        Args:
            n: Number of decks to sample
            mode: Sampling mode ('uniform', 'hierarchical')
        """
        if n <= 0:
            return []
            
        if mode == "uniform":
            return random.choices(self.decks, k=n)
        elif mode == "hierarchical":
            # Era -> Archetype -> Deck diversity
            eras = list(self.eras.keys())
            sampled_decks = []
            for _ in range(n):
                era_name = random.choice(eras)
                arch_name = random.choice(self.eras[era_name])
                deck = random.choice(self.archetypes[arch_name].decks)
                sampled_decks.append(deck)
            return sampled_decks
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_archetypes(self) -> list[str]:
        """Get list of all archetypes."""
        return list(self.archetypes.keys())


    def sample_pair(self, mode: str = "hierarchical") -> tuple[str, str]:
        """Sample two decks for self-play."""
        return self.sample_deck(mode), self.sample_deck(mode)

    def __len__(self) -> int:
        return len(self.decks)


class CurriculumDeckLoader:
    """
    Curriculum learning across different deck sets.
    """

    def __init__(
        self,
        base_loader: MetaDeckLoader,
        target_loader: MetaDeckLoader,
    ):
        """
        Args:
            base_loader: Starting deck loader
            target_loader: Target deck loader
        """
        self.base_loader = base_loader
        self.target_loader = target_loader

        self.difficulty = 0.0  # 0 = all base, 1 = all target
        self.sampling_mode = "hierarchical"

    def set_difficulty(self, difficulty: float):
        """Set curriculum difficulty (0.0 = simple, 1.0 = meta)."""
        self.difficulty = max(0.0, min(1.0, difficulty))

    def set_sampling_mode(self, mode: str):
        """Set sampling mode: 'hierarchical' or 'uniform'."""
        self.sampling_mode = mode

    def sample_deck(self) -> str:
        """Sample a deck based on difficulty and mode."""
        if random.random() < self.difficulty:
            return self.target_loader.sample_deck(mode=self.sampling_mode)
        else:
            return self.base_loader.sample_deck(mode=self.sampling_mode)

    def sample_pair(self) -> tuple[str, str]:
        """Sample two decks for self-play."""
        return self.sample_deck(), self.sample_deck()

    def __len__(self) -> int:
        return len(self.base_loader) + len(self.target_loader)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deck_loader.py <meta_decks_dir>")
        sys.exit(1)

    loader = MetaDeckLoader(sys.argv[1])
    print(f"Loaded {len(loader.decks)} decks from {len(loader.archetypes)} archetypes")

    print("\n--- Archetype Statistics ---")
    archetypes = sorted(
        loader.archetypes.values(), key=lambda a: a.deck_count, reverse=True
    )
    for arch in archetypes[:10]:
        print(
            f"  {arch.name[:40]:40s} | decks={arch.deck_count:3d}"
        )

    print("\n--- Sample decks (hierarchical) ---")
    for _ in range(3):
        deck = loader.sample_deck_info(mode="hierarchical")
        print(f"  {deck.archetype[:30]}")
