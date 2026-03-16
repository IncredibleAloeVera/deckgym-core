"""
Maps embeddings to Pocket database cards.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import PATHS
from .text_cleaner import TextCleaner


class PocketMapper:
    """Maps text embeddings to Pocket database cards."""

    def __init__(self, pocket_db_path: Path = None):
        """
        Initialize mapper.

        Args:
            pocket_db_path: Path to Pocket database.json
        """
        self.pocket_db_path = pocket_db_path or PATHS["pocket_db"]
        self._cards: List[Dict] = []
        self._names: set = set()

    def load(self) -> "PocketMapper":
        """Load Pocket database."""
        with open(self.pocket_db_path, "r", encoding="utf-8") as f:
            self._cards = json.load(f)

        # Extract names for cleaner
        for item in self._cards:
            card = item.get("Pokemon") or item.get("Trainer")
            if card:
                name = card.get("name", "")
                self._names.add(name)
                if " ex" in name:
                    self._names.add(name.replace(" ex", ""))
                if name.startswith("Mega "):
                    self._names.add(name.replace("Mega ", ""))

        self._names.add("Pokémon")
        self._names.add("Pokemon")

        return self

    @property
    def pokemon_names(self) -> set:
        """Pokemon names from Pocket database."""
        return self._names

    def generate_card_features(
        self,
        text_embeddings: Dict[str, Dict],
        cleaner: TextCleaner,
        embedding_dim: int = 128,
    ) -> Dict[str, Dict]:
        """
        Generate card_features.json content.

        Args:
            text_embeddings: Dict of cleaned_text -> {embedding, type_refs, mech_refs}
            cleaner: TextCleaner for text normalization
            embedding_dim: Expected embedding dimension

        Returns:
            Dict of card_id -> features
        """
        # Build evolution metadata
        evolves_to: Dict[str, List[str]] = {}
        evolves_from: Dict[str, str] = {}

        for item in self._cards:
            card = item.get("Pokemon")
            if not card:
                continue

            name = card.get("name", "")
            ef = card.get("evolves_from")

            if ef:
                evolves_from[name] = ef
                evolves_to.setdefault(ef, []).append(name)

        def get_line_size(name: str) -> int:
            """Get evolution line size for a Pokemon."""
            curr = name
            while curr in evolves_from:
                curr = evolves_from[curr]

            def max_dist(n: str) -> int:
                if n not in evolves_to:
                    return 1
                return 1 + max(max_dist(c) for c in evolves_to[n])

            return max_dist(curr)

        def get_text_data(text: str) -> Dict:
            """Get embedding data for a text, with fallback."""
            from .config import TYPE_MAP, MECH_MAP

            if not text:
                return {
                    "embedding": [0.0] * embedding_dim,
                    "type_refs": [0.0] * len(TYPE_MAP),
                    "mech_refs": [0.0] * len(MECH_MAP),
                }

            cleaned = cleaner.clean(text)
            data = text_embeddings.get(cleaned)

            if data:
                return data

            return {
                "embedding": [0.0] * embedding_dim,
                "type_refs": TextCleaner.get_type_references(text),
                "mech_refs": TextCleaner.get_mechanic_references(text),
            }

        # Generate features for each card
        card_features = {}

        for item in self._cards:
            card = item.get("Pokemon") or item.get("Trainer")
            if not card:
                continue

            card_id = card.get("id", "")

            # Ability
            ability_text = ""
            if card.get("ability"):
                ability_text = card["ability"].get("effect", "")

            # Attacks
            attacks = card.get("attacks", [])
            atk1_text = attacks[0].get("effect", "") if len(attacks) > 0 else ""
            atk2_text = attacks[1].get("effect", "") if len(attacks) > 1 else ""

            # Trainer effect
            trainer_text = card.get("effect", "")

            # Evolution metadata
            if "Pokemon" in item:
                line_size = get_line_size(card["name"])
                is_final = 2 if card["name"] not in evolves_to else 1
            else:
                line_size = 0
                is_final = 0

            card_features[card_id] = {
                "ability": get_text_data(ability_text),
                "atk1": get_text_data(atk1_text),
                "atk2": get_text_data(atk2_text),
                "supporter": get_text_data(trainer_text),
                "line_size": line_size,
                "is_final_stage": is_final,
            }

        return card_features

    def save_features(self, features: Dict, output_path: Path = None):
        """Save card features to JSON."""
        output_path = output_path or (PATHS["output_dir"] / "card_features.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)

        print(f"Saved card features to {output_path}")
