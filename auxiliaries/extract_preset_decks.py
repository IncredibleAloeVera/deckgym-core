"""Extract the preset Pokemon TCG Pocket decks into a single JSON.

None of these decks are published in a machine readable form, so they come from
three sources, which become the three top level keys of the output:

    solo    Solo Battle ("Step Up" / "Expert") opponent decks, one Game8 guide
            page per difficulty tier, kept in a "tier" field.
    rental  Rental decks, one Game8 page, grouped by expansion.
    event   Event Solo Battle decks, transcribed by hand into
            raw_event_decks.txt. That file has no names: decks come in themed
            groups of four with increasing difficulty, so they are numbered by
            "theme" and get the same four tiers as the solo decks.

Deck sections on the Game8 pages look like this, and the two pages differ in
where they put the energy:

    <h3>Charmander Deck</h3>
    <table>                                                <- deck summary
      <img alt='Pokemon TCG Pocket - Charmander Deck (Crimson Blaze)'/>
      <img alt='Fire'/> Fire</a> Deck                      <- solo energy
    </table>
    <h4>Charmander Deck Card List</h4>
    <table>
      <th colspan=16>Charmander Deck (Crimson Blaze) Deck List</th>
      ... <img alt='Pokemon TCG Pocket- B1 021 Card'/> <a>Skiddo</a> x2 ...
      <th>Energy Used</th> <img alt='Fire Icon'/>          <- rental energy
    </table>

Names and expansions are read from the summary thumbnail rather than from the
card list header, because a few of those headers are copy-pasted from the
previous deck on the page. The card `alt` attribute carries the set code plus
number, which is exactly the id format used by `database.json`, so every card is
resolved against it and named from it (Game8 strips accents: "Poke Ball").

Each deck also gets a `playable` flag, meaning the simulator can run it: 20
cards, at most 2 copies of any id, and every card implemented according to
`cargo run --bin card_status -- --incomplete-only`. When it is false, `blockers`
says why.

Usage:
    python extract_solo_decks.py [--refresh] [--refresh-status]
    python extract_solo_decks.py --reannotate     # only re-judge preset_decks_fixed.json

Writes `preset_decks.json` next to this script. Stdlib only.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_DIR = HERE / ".game8_cache"
# Tracked, and outside CACHE_DIR: this one is our own output, not a copy of someone's page.
STATUS_CACHE = HERE / "card_status.txt"
RAW_EVENTS = HERE / "raw_event_decks.txt"
OUTPUT = HERE / "preset_decks.json"
FIXED = HERE / "preset_decks_fixed.json"
DATABASE = ROOT / "database.json"

TIERS = ("beginner", "intermediate", "advanced", "expert")

SOLO_SOURCES = dict(
    zip(TIERS, ("483809", "483633", "483770", "483771"))
)
RENTAL_SOURCE = "476889"
ARCHIVE = "https://game8.co/games/Pokemon-TCG-Pocket/archives/%s"

ENERGIES = (
    "Grass|Fire|Water|Lightning|Psychic|Fighting|Darkness|Metal|Dragon|Colorless"
)

# <th colspan=16>Charmander Deck (Crimson Blaze) Deck List</th> -- marks a deck
DECK_HEADER_RE = re.compile(r"colspan=16>.*?Deck List</th>", re.S)
# Deck thumbnail in the summary table. Note the space before the dash, which is
# what tells it apart from the card thumbnails ("Pokemon TCG Pocket- A1 001 Card").
DECK_LABEL_RE = re.compile(r"alt='Pokemon TCG Pocket - ([^']*)'")
RENTAL_LABEL_RE = re.compile(r"alt='([^']*) Rental Deck'")
# <h2>Crimson Blaze Rental Decks</h2>, the expansion the rental decks belong to
RENTAL_GROUP_RE = re.compile(r"<h2[^>]*>\s*([^<]*?)\s*Rental Decks\s*</h2>")
# Solo pages: <img alt='Fire' .../> Fire</a> Deck, in the deck summary table.
SOLO_ENERGY_RE = re.compile(r"alt='(%s)'" % ENERGIES)
# Rental page: an "Energy Used" row at the end of the card list, 1 to 3 icons.
RENTAL_ENERGY_RE = re.compile(
    r"Energy Used</th>(.*?)</tr>", re.S
)
ENERGY_ICON_RE = re.compile(r"alt='(%s) Icon'" % ENERGIES)
# One card cell: id from the image alt, name from the link, count from "x2".
CARD_RE = re.compile(
    r"alt='Pokemon TCG Pocket-\s*([A-Za-z0-9-]+)\s+(\d+)\s*Card'"  # set + number
    r".*?"
    r"<a class='a-link'[^>]*>(.*?)</a>\s*(?:&times;|×|x)\s*(\d+)",
    re.S,
)
# raw_event_decks.txt: an id line, then a "<name> x<count>" line.
RAW_CARD_RE = re.compile(
    r"Pokemon TCG Pocket-\s*([A-Za-z0-9-]+)\s+(\d+)\s*Card\s*\n\s*(.*?)\s*(?:×|x)\s*(\d+)"
)
# "Charmander Deck (Crimson Blaze)" -> name + expansion
LABEL_PARTS_RE = re.compile(r"^(.*?)(?:\s*Deck)?\s*(?:\(([^()]*)\))?$")
# card_status rows: "A1 067   Cloyster   Ability not implemented"
STATUS_ROW_RE = re.compile(r"^([A-Za-z0-9-]+ \d+)\s\s+(.*?)\s\s+(\S.*?)\s*$")

# Misspelled deck names on Game8, caught by checking that a deck contains the
# Pokemon it is named after.
NAME_FIXES = {"Alolan Nintetales": "Alolan Ninetales"}

MAX_COPIES = 2
DECK_SIZE = 20


def fetch(archive_id: str, refresh: bool) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cached = CACHE_DIR / f"{archive_id}.html"
    if cached.exists() and not refresh:
        return cached.read_text(encoding="utf-8")
    req = urllib.request.Request(ARCHIVE % archive_id, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", "replace")
    cached.write_text(body, encoding="utf-8")
    return body


def text(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def normalize_name(name: str) -> str:
    # Deck names mix "X and Y" with "X & Y" across pages.
    return NAME_FIXES.get(name, name).replace(" and ", " & ")


def read_cards(table: str, names: dict[str, str]) -> list[dict]:
    cards = []
    for set_code, number, card_name, count in CARD_RE.findall(table):
        card_id = f"{set_code} {number}"
        # Game8 strips accents, so prefer the database spelling.
        cards.append(
            {"id": card_id, "name": names.get(card_id, text(card_name)), "count": int(count)}
        )
    return cards


def parse_solo_page(page: str, tier: str, names: dict[str, str]) -> list[dict]:
    """Return the decks of one difficulty tier, in page order."""
    decks = []
    headers = list(DECK_HEADER_RE.finditer(page))
    for i, header in enumerate(headers):
        # The card list table runs from its header to the closing </table>.
        table = page[header.end() : page.find("</table>", header.end())]
        # Everything between the previous card list and this one is the summary
        # of the current deck; its last thumbnail and type icon describe it.
        summary = page[headers[i - 1].end() if i else 0 : header.start()]

        labels = DECK_LABEL_RE.findall(summary)
        name, expansion = LABEL_PARTS_RE.match(text(labels[-1]) if labels else "").groups()
        energies = SOLO_ENERGY_RE.findall(summary)

        decks.append(
            {
                "name": normalize_name(name),
                "expansion": expansion,
                "tier": tier,
                "energy": energies[-1:],
                "cards": read_cards(table, names),
            }
        )
    return decks


def parse_rental_page(page: str, names: dict[str, str]) -> list[dict]:
    """Return the rental decks, in page order. Energy follows the card list."""
    decks = []
    headers = list(DECK_HEADER_RE.finditer(page))
    for i, header in enumerate(headers):
        summary = page[headers[i - 1].end() if i else 0 : header.start()]
        # The "Energy Used" row sits inside the card list table, after the cards.
        section_end = headers[i + 1].start() if i + 1 < len(headers) else len(page)
        section = page[header.end() : section_end]
        table, _, tail = section.partition("Energy Used</th>")

        labels = RENTAL_LABEL_RE.findall(summary)
        groups = RENTAL_GROUP_RE.findall(page[: header.start()])
        energy_row = tail[: tail.find("</tr>")] if tail else ""

        decks.append(
            {
                "name": normalize_name(text(labels[-1]) if labels else ""),
                "expansion": text(groups[-1]) if groups else None,
                "energy": ENERGY_ICON_RE.findall(energy_row),
                "cards": read_cards(table, names),
            }
        )
    return decks


def parse_raw_events(raw: str, names: dict[str, str]) -> list[dict]:
    """Return the event decks from the hand transcribed dump.

    Blank lines separate decks, decks come in themed groups of four, and the
    last line of a deck is its energy ("Water", or "Fighting/Psychic" for two).
    """
    energy_names = set(ENERGIES.split("|"))
    decks = []
    for index, block in enumerate(b for b in re.split(r"\n\s*\n", raw) if b.strip()):
        last = block.splitlines()[-1].strip()
        parts = last.split("/")
        energy = parts if parts and all(p in energy_names for p in parts) else []

        cards: list[dict] = []
        for set_code, number, card_name, count in RAW_CARD_RE.findall(block):
            card_id = f"{set_code} {number}"
            # The transcription repeats a line here and there; a card can only
            # appear once in a deck, so an identical repeat is a copy artifact.
            if cards and cards[-1]["id"] == card_id and cards[-1]["count"] == int(count):
                continue
            cards.append(
                {"id": card_id, "name": names.get(card_id, card_name), "count": int(count)}
            )

        decks.append(
            {
                "theme": index // len(TIERS) + 1,
                "tier": TIERS[index % len(TIERS)],
                "energy": energy,
                "cards": cards,
            }
        )
    return decks


def load_card_names() -> dict[str, str]:
    names = {}
    for entry in json.loads(DATABASE.read_text(encoding="utf-8")):
        (_, card), = entry.items()
        names[card["id"]] = card["name"]
    return names


def load_unimplemented(refresh: bool) -> dict[str, str]:
    """Map card id -> why the engine cannot run it, from the card_status bin."""
    if refresh or not STATUS_CACHE.exists():
        report = subprocess.run(
            ["cargo", "run", "--quiet", "--bin", "card_status", "--", "--incomplete-only"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        STATUS_CACHE.write_text(report, encoding="utf-8")
    else:
        report = STATUS_CACHE.read_text(encoding="utf-8")

    reasons = {}
    for line in report.splitlines():
        row = STATUS_ROW_RE.match(line)
        if row and "not implemented" in row.group(3):
            reasons[row.group(1)] = row.group(3)
    return reasons


def annotate(deck: dict, names: dict[str, str], unimplemented: dict[str, str]) -> None:
    """Add `playable` and `blockers` to a deck, in place."""
    blockers = []

    total = sum(card["count"] for card in deck["cards"])
    if total != DECK_SIZE:
        blockers.append(f"{total} cards instead of {DECK_SIZE}")
    if not deck["energy"]:
        blockers.append("no energy listed")

    copies: dict[str, int] = {}
    for card in deck["cards"]:
        copies[card["id"]] = copies.get(card["id"], 0) + card["count"]
    for card_id, count in copies.items():
        if count > MAX_COPIES:
            blockers.append(f"{card_id} in {count} copies, max is {MAX_COPIES}")
        if card_id not in names:
            blockers.append(f"{card_id} is not in database.json")
        elif card_id in unimplemented:
            blockers.append(f"{card_id} {names[card_id]}: {unimplemented[card_id].lower()}")

    # Reinserted after popping `cards` so that a deck reads header first.
    cards = deck.pop("cards")
    deck["playable"] = not blockers
    deck["blockers"] = blockers
    deck["cards"] = cards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download the Game8 pages")
    parser.add_argument(
        "--refresh-status", action="store_true", help="re-run the card_status binary"
    )
    parser.add_argument(
        "--reannotate",
        action="store_true",
        help="only re-run playable/blockers over preset_decks_fixed.json, in place",
    )
    args = parser.parse_args()

    names = load_card_names()
    unimplemented = load_unimplemented(args.refresh_status)

    if args.reannotate:
        return reannotate(names, unimplemented)

    data = {
        "solo": [
            deck
            for tier, archive_id in SOLO_SOURCES.items()
            for deck in parse_solo_page(fetch(archive_id, args.refresh), tier, names)
        ],
        "rental": parse_rental_page(fetch(RENTAL_SOURCE, args.refresh), names),
        "event": parse_raw_events(RAW_EVENTS.read_text(encoding="utf-8"), names),
    }
    for decks in data.values():
        for deck in decks:
            annotate(deck, names, unimplemented)

    OUTPUT.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for source, decks in data.items():
        playable = sum(deck["playable"] for deck in decks)
        print(f"{source:7} {len(decks):4} decks, {playable:4} playable")
    for source, decks in data.items():
        for deck in decks:
            for blocker in deck["blockers"]:
                if "not implemented" not in blocker:
                    print(f"  warn: {source}/{describe(deck)}: {blocker}", file=sys.stderr)
    print(f"wrote {OUTPUT.name}")
    return 0


def reannotate(names: dict[str, str], unimplemented: dict[str, str]) -> int:
    """Re-run `annotate` over the hand-repaired copy the deck DBs are built from.

    A newly implemented card clears blockers on decks that are otherwise unchanged, and the
    fixes in `preset_decks_fixed.json` are not reproducible from the Game8 pages, so those
    decks are re-judged in place rather than re-extracted.
    """
    if not FIXED.exists():
        print(f"{FIXED.name} does not exist", file=sys.stderr)
        return 1

    data = json.loads(FIXED.read_text(encoding="utf-8"))
    for source, decks in data.items():
        before = sum(deck["playable"] for deck in decks)
        for deck in decks:
            annotate(deck, names, unimplemented)
        after = sum(deck["playable"] for deck in decks)
        print(f"{source:7} {len(decks):4} decks, {before:4} -> {after:4} playable")

    FIXED.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXED.name}")
    return 0


def describe(deck: dict) -> str:
    if "theme" in deck:
        return f"theme {deck['theme']} {deck['tier']}"
    return f"{deck['name']} ({deck['expansion']})"


if __name__ == "__main__":
    raise SystemExit(main())
