"""Compile the two deck DBs of RL_ARCHITECTURE 1.5.3 out of the auxiliary JSON archives.

    auxiliaries/meta_decks.json          --> decks/meta/<archetype>.txt      (O(100k))
    auxiliaries/preset_decks_fixed.json  --> decks/tutorial/<archetype>.txt  (O(1k))

Only `playable` decks survive: a deck whose `blockers` list is non-empty names a card the
engine cannot run, so sampling it would fail at game construction rather than at load.

Archetypes are directories' worth of decks, one file each, because the sampler's mirror
quota needs the grouping at draw time and walking 70k loose files is not something to do
per run. The grouping key differs per source:

    meta       the Limitless `archetype` id, already a label ("darkrai-ex-a2-giratina-ex-a2a")
    tutorial   the difficulty `tier` -- beginner / intermediate / advanced / expert, plus
               `rental` for the untiered rental bucket

The tutorial key is the tier and not the deck name (near-unique, so every archetype would
be a singleton) and not the source bucket, because the tier is the axis a run needs to
select on: drawing beginner and expert decks uniformly at the start of training is exactly
what the grouping exists to prevent.

Decks are deduplicated on the exact card multiset + energy set. `meta_decks.json` already
collapses identical lists, but two lists that differ only in a card the energy heuristic
ignores still land on the same block, so the check is kept here as well.

Output blocks are the ordinary deck text format of src/deck.rs, prefixed by a `# <hash>`
line that src/rl/train/deck_db.rs strips back off, separated by a blank line.

Run:  uv run --no-project auxiliaries/build_deck_dbs.py
"""

import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUX = ROOT / "auxiliaries"
OUT = ROOT / "decks"

# A deck declares one of these and nothing else. Colorless and Dragon are attack costs and
# Pokemon types; neither is an energy a deck can hold, and `Game::get_color` says so with a
# `todo!()`. A handful of preset decks list one anyway -- the upstream extractor read the
# headline Pokemon's type as the deck's energy -- and `repair_energy` puts them back.
LEGAL_ENERGY = {
    "Grass", "Fire", "Water", "Lightning",
    "Psychic", "Fighting", "Darkness", "Metal",
}


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "unnamed"


def repair_energy(energy):
    """Map a declared energy list back onto energies a deck can actually hold.

    A Dragon-type Pokemon attacks on Water + Lightning, so a deck the extractor labelled
    "Dragon" runs both. Colorless is a cost, never a source: alongside a real energy it is
    noise and drops out, alone it leaves nothing to attach and Water stands in.

    Returns None if the result still is not a legal set -- the caller drops those.
    """
    energy = set(energy or [])
    if "Dragon" in energy:
        energy = (energy - {"Dragon"}) | {"Water", "Lightning"}
    if "Colorless" in energy:
        energy = (energy - {"Colorless"}) or {"Water"}
    if not energy or not energy <= LEGAL_ENERGY:
        return None
    return sorted(energy)


def canonical(deck):
    """The identity of a deck: its card multiset plus its energy set, both order-free."""
    cards = sorted((c["id"], c["count"]) for c in deck["cards"])
    energy = sorted(deck.get("energy") or [])
    return json.dumps([cards, energy], separators=(",", ":"))


def block(deck, deck_id):
    lines = [f"# {deck_id}"]
    energy = sorted(deck.get("energy") or [])
    if energy:
        lines.append("Energy: " + ", ".join(energy))
    for card in sorted(deck["cards"], key=lambda c: c["id"]):
        lines.append(f"{card['count']} {card['id']}")
    return "\n".join(lines)


def emit(db_name, grouped):
    """Write one file per archetype, deck order fixed by id so reruns are byte-identical."""
    out_dir = OUT / db_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    total = 0
    for archetype, decks in sorted(grouped.items()):
        blocks = [block(deck, deck_id) for deck_id, deck in sorted(decks.items())]
        path = out_dir / f"{archetype}.txt"
        # newline="" so Windows does not translate to CRLF and reruns stay byte-identical.
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("\n\n".join(blocks) + "\n")
        total += len(blocks)
    print(f"{db_name}: {total} decks in {len(grouped)} archetypes -> {out_dir}")


def collect(entries, archetype_of):
    """Group playable decks by archetype, collapsing duplicate multisets to one entry."""
    grouped = defaultdict(dict)
    kept = unplayable = repaired = unusable_energy = 0
    for entry in entries:
        if not entry.get("playable"):
            unplayable += 1
            continue
        # An absent list counts as unusable, not as "derive it later": `Deck::from_string` would
        # fall back to reading the energy off the cards, the guess that produced the bad lists.
        energy = repair_energy(entry.get("energy"))
        if energy is None:
            unusable_energy += 1
            continue
        if energy != sorted(entry.get("energy") or []):
            repaired += 1
        entry = dict(entry, energy=energy)
        deck_id = hashlib.blake2b(canonical(entry).encode(), digest_size=4).hexdigest()
        grouped[archetype_of(entry)][deck_id] = entry
        kept += 1
    print(
        f"  {kept} kept ({repaired} energy repaired), "
        f"{unplayable} unplayable, {unusable_energy} unusable energy"
    )
    return grouped


def main():
    meta = json.loads((AUX / "meta_decks.json").read_text(encoding="utf-8"))
    print("meta:")
    emit("meta", collect(meta["decks"], lambda e: slugify(e["archetype"])))

    presets = json.loads((AUX / "preset_decks_fixed.json").read_text(encoding="utf-8"))
    print("tutorial:")
    # The rental bucket carries no tier; it is its own group rather than being assigned one.
    tiered = [
        dict(deck, _tier=deck.get("tier") or bucket)
        for bucket, decks in presets.items()
        for deck in decks
    ]
    emit("tutorial", collect(tiered, lambda e: slugify(e["_tier"])))


if __name__ == "__main__":
    main()
