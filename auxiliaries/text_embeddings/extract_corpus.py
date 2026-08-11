"""Extract the English effect-text corpus from the sibling `cards-database` repo (tcgdex).

The corpus is the "super-set TCG" of RL_ARCHITECTURE §1.2.9: every English effect text
(attacks, abilities, trainer cards) across the whole TCG, used only to fit the PCA basis.
Card names, flavor texts and non-English languages are deliberately excluded.

Outputs:
  corpus.json — sorted, deduplicated list of effect strings;
  names.json  — every Pokémon card name (super-set + Pocket), used by common.normalize()
                to strip names (and their V / VMAX / Dark / ... markers) from encoder input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CARDS_DATABASE = Path(__file__).resolve().parents[3] / "cards-database" / "data"
DECKGYM_DATABASE = Path(__file__).resolve().parents[2] / "database.json"
OUTPUT = Path(__file__).resolve().parent / "corpus.json"
NAMES_OUTPUT = Path(__file__).resolve().parent / "names.json"

# An `effect: { ... en: "..." ... }` block; `[^{}]` keeps the scan inside one block.
EFFECT_EN = re.compile(r'\beffect:\s*\{[^{}]*?\ben:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
# The card's own name block (first `name:` in the file — ability/attack names come later).
NAME_EN = re.compile(r'\bname:\s*\{[^{}]*?\ben:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
POKEMON_CATEGORY = re.compile(r'\bcategory:\s*"Pokemon"')


def unescape(text: str) -> str:
    return (
        text.replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\t", " ")
        .replace("\\\\", "\\")
    )


def main() -> None:
    if not CARDS_DATABASE.is_dir():
        raise SystemExit(f"cards-database not found at {CARDS_DATABASE}")

    corpus: set[str] = set()
    names: set[str] = set()
    n_files = 0
    for path in CARDS_DATABASE.rglob("*.ts"):
        n_files += 1
        content = path.read_text(encoding="utf-8", errors="replace")
        for match in EFFECT_EN.finditer(content):
            text = unescape(match.group(1)).strip()
            if text:
                corpus.add(text)
        if POKEMON_CATEGORY.search(content):
            match = NAME_EN.search(content)
            if match:
                name = unescape(match.group(1)).strip()
                if name:
                    names.add(name)

    # Pocket's own Pokémon names (suffixed forms like "Venusaur ex" included).
    for entry in json.loads(DECKGYM_DATABASE.read_text(encoding="utf-8")):
        card = entry.get("Pokemon")
        if card:
            names.add(card["name"])

    texts = sorted(corpus)
    OUTPUT.write_text(
        json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    NAMES_OUTPUT.write_text(
        json.dumps(sorted(names), ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(
        f"{n_files} card files scanned, {len(texts)} unique effect texts -> {OUTPUT.name}, "
        f"{len(names)} Pokémon names -> {NAMES_OUTPUT.name}"
    )


if __name__ == "__main__":
    main()
