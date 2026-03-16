"""
Data loader for the full TCG database.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .config import PATHS


class TCGDataLoader:
    """Loads and processes the full Pokemon TCG database."""

    def __init__(self, raw_data_path: Path = None):
        """
        Initialize loader with path to raw TCG data.

        Args:
            raw_data_path: Path to directory containing TCG JSON files
        """
        self.raw_data_path = raw_data_path or PATHS["raw_data"]
        self._cards: List[Dict] = []
        self._names: Set[str] = {"Pokémon", "Pokemon"}
        self._texts: Set[str] = set()

    def load(self) -> "TCGDataLoader":
        """
        Load all TCG data from JSON files.

        Returns:
            Self for chaining
        """
        self._cards = []

        for json_file in sorted(self.raw_data_path.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                cards = json.load(f)
                self._cards.extend(cards)

        # Extract names and texts
        self._extract_metadata()

        return self

    def _extract_metadata(self):
        """Extract Pokemon names and unique effect texts from loaded cards."""
        for card in self._cards:
            # Skip non-Pokemon/Trainer cards
            supertype = card.get("supertype", "")
            if supertype not in ("Pokémon", "Trainer"):
                continue

            # Collect names
            name = card.get("name", "")
            if name:
                self._names.add(name)
                # Handle variants like "Charizard ex"
                if " ex" in name:
                    self._names.add(name.replace(" ex", ""))
                if name.startswith("Mega "):
                    self._names.add(name.replace("Mega ", ""))

            # Collect ability texts
            for ability in card.get("abilities", []) or []:
                text = ability.get("text", "")
                if text:
                    self._texts.add(text)

            # Collect attack texts
            for attack in card.get("attacks", []) or []:
                text = attack.get("text", "")
                if text:
                    self._texts.add(text)

            # Collect trainer rule texts (for Supporters/Items)
            for rule in card.get("rules", []) or []:
                if rule:
                    self._texts.add(rule)

    @property
    def cards(self) -> List[Dict]:
        """All loaded cards."""
        return self._cards

    @property
    def pokemon_names(self) -> Set[str]:
        """Set of all Pokemon names for text normalization."""
        return self._names

    @property
    def unique_texts(self) -> List[str]:
        """Sorted list of unique effect texts."""
        return sorted(list(self._texts))

    def get_stats(self) -> Dict:
        """Get statistics about loaded data."""
        return {
            "total_cards": len(self._cards),
            "unique_names": len(self._names),
            "unique_texts": len(self._texts),
        }
