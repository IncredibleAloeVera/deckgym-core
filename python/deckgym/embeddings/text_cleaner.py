"""
Text cleaning utilities for card effect texts.
"""

import re
from typing import List, Set

from .config import TYPE_MAP, MECH_MAP


class TextCleaner:
    """Cleans and normalizes card effect texts for embedding generation."""

    def __init__(self, pokemon_names: Set[str]):
        """
        Initialize cleaner with Pokemon names for replacement.

        Args:
            pokemon_names: Set of all Pokemon names to normalize
        """
        # Sort by length descending to avoid partial matches
        sorted_names = sorted(list(pokemon_names), key=len, reverse=True)
        self._names_regex = re.compile(
            r"\b(" + "|".join(map(re.escape, sorted_names)) + r")\b", re.IGNORECASE
        )

    def clean(self, text: str) -> str:
        """
        Clean a card effect text for embedding.

        Args:
            text: Raw card effect text

        Returns:
            Cleaned, normalized text
        """
        if not text:
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Replace Pokemon names with generic "pokemon"
        text = self._names_regex.sub("pokemon", text)

        # 3. Catch remaining accented variants
        text = text.replace("pokémon", "pokemon")

        # 4. Normalize type symbols [G] -> grass, etc.
        for sym, (full, _) in TYPE_MAP.items():
            text = text.replace(f"[{sym.lower()}]", full.lower())

        return text

    @staticmethod
    def get_type_references(text: str) -> List[float]:
        """
        Extract type references from original (uncleaned) text.

        Args:
            text: Original card effect text

        Returns:
            9-dimensional binary vector of type references
        """
        if not text:
            return [0.0] * 9

        text_lower = text.lower()
        refs = [0.0] * 9

        for sym, (full, idx) in TYPE_MAP.items():
            if f"[{sym}]" in text or full.lower() in text_lower:
                refs[idx] = 1.0

        return refs

    @staticmethod
    def get_mechanic_references(text: str) -> List[float]:
        """
        Extract game mechanic references from original text.

        Args:
            text: Original card effect text

        Returns:
            Binary vector of mechanic references (dimension = len(MECH_MAP))
        """
        if not text:
            return [0.0] * len(MECH_MAP)

        text_lower = text.lower()
        keys = list(MECH_MAP.keys())
        refs = [0.0] * len(keys)

        for i, key in enumerate(keys):
            for keyword in MECH_MAP[key]:
                if keyword in text_lower:
                    refs[i] = 1.0
                    break

        return refs
