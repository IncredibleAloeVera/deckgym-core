"""Derive the energy types a decklist generates, plus the engine's playability check.

Pocket decklists almost never state their energy: the Energy Zone is configured in the
client, not written down, so no public source carries it. `decide_energy` recovers it from
what the deck asks for, and `blockers_for` says whether the simulator can run the list at
all. Both are shared by every deck importer in `auxiliaries/`.

`sweep` scores the algorithm's two knobs against the 30 `example_decks/` that do declare an
energy. Read that 100% as what it is -- 30 decks, chosen as examples. Against the tournament
lists that state their energy, which are thousands and were chosen by nobody,
`build_meta_decks.py check` puts the same defaults at 89.5%. Prefer that number, and re-run
it after touching anything here: the example decks are too few and too kind to catch a
regression.

    exclude free energy   exceptions   matches
    yes                   no            26/30   87%
    yes                   yes           30/30  100%   <- defaults
    no                    no            25/30   83%
    no                    yes           30/30  100%

  * `TYPE_NEUTRAL` is the hand maintained list of cards a deck runs for their ability or
    their Colorless attack, never for their type: one Indeedee ex among ten Water cards is
    not a second energy. Their types are held aside and only used when nothing else asks
    for one. `--no-exceptions` drops it.
  * The free energy exclusion is fenced in two ways, because in Pocket energy only ever
    comes out of the Energy Zone, which only makes the types the deck declares. It never
    touches a mono type basket, and it never fires on an attack with a typed cost: Moltres
    ex paying [R] to hand out [R] proves the deck runs Fire. Abilities are taken at face
    value, and 24 attacks in the pool cost nothing or Colorless and do qualify.

Usage:
    uv run --no-project --python 3.14 auxiliaries/deck_energy.py sweep
    uv run --no-project --python 3.14 auxiliaries/deck_energy.py validate

    ... validate --no-exclude-free --no-exceptions   # the bare algorithm, no fences

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATABASE = ROOT / "database.json"
STATUS_CACHE = HERE / "card_status.txt"
EXAMPLE_DECKS = ROOT / "example_decks"

DECK_SIZE = 20
MAX_COPIES = 2

# Defaults of the two knobs of `decide_energy`, scored by `sweep` against the
# 30 example decks that declare an energy (see the module docstring).
EXCLUDE_FREE = True
USE_EXCEPTIONS = True

# Cards a deck runs for their ability or their Colorless attack, never for
# their type: their type is a red herring in the basket, and only counts when
# it is the only type the deck asks for. A rule matches a card by name, plus
# the attack or ability title when only one printing is meant; `with` makes the
# rule conditional on another card being in the deck. Pre-evolutions of a
# matched card are pulled in with it.
TYPE_NEUTRAL = (
    {"name": "Greninja", "ability": "Water Shuriken"},
    {"name": "Giratina ex"},
    {"name": "Indeedee"},
    {"name": "Indeedee ex"},
    {"name": "Sylveon ex"},
    {"name": "Druddigon"},
    {"name": "Mew ex", "attack": "Genome Hacking"},
    {"name": "Chandelure", "ability": "Slow Sear"},
    {"name": "Decidueye ex"},
    {"name": "Aegislash", "ability": "Cursed Metal"},
    {"name": "Crobat", "ability": "Cunning Link", "with": "Arceus"},
    # A Dragon basic asking Water+Psychic to attack, played in Grass, Fire and Metal decks
    # alike for its body. Its cost was the single largest error in the archive.
    {"name": "Goomy", "attack": "Ram"},
    # Bench sitters: an ability worth the slot, and an attack with no effect at all, which
    # is the attack nobody ever pays for. Shaymin alone accounted for 1399 archive errors.
    # Tempting to generalise to "has an ability, all attacks vanilla", but that was measured
    # at +0.16% against +2.5% for these three: it also neutralises real attackers.
    {"name": "Shaymin", "attack": "Flop"},
    {"name": "Oricorio", "attack": "Zzzap"},
    {"name": "Giratina", "attack": "Spooky Shot"},
)
# "If this Pokémon has 2 or more different types of Energy attached, this
# attack does 60 more damage": a deck running it wants its two types.
DUAL_TYPE = {"name": "Altaria", "attack": "Dragon Arcana"}

# Energy types the Energy Zone can generate. Colorless and Dragon can never be
# a deck energy, so they never make it into a basket.
GENERABLE = (
    "Grass",
    "Fire",
    "Water",
    "Lightning",
    "Psychic",
    "Fighting",
    "Darkness",
    "Metal",
)
ORDER = {energy: index for index, energy in enumerate(GENERABLE)}

# [R] style energy symbols, as they appear inside effect text.
SYMBOLS = {
    "G": "Grass",
    "R": "Fire",
    "W": "Water",
    "L": "Lightning",
    "P": "Psychic",
    "F": "Fighting",
    "D": "Darkness",
    "M": "Metal",
    "N": "Dragon",
    "C": "Colorless",
}
SYMBOL_RE = re.compile(r"\[([A-Z])\]")
# A card that hands out energy says "take a/an/2/3 ... from your Energy Zone".
TAKES_RE = re.compile(r"\btake (?:a|an|\d+) ", re.I)
# card_status rows: "A1 067   Cloyster   Ability not implemented"
STATUS_ROW_RE = re.compile(r"^([A-Za-z0-9-]+ \d+)\s\s+(.*?)\s\s+(\S.*?)\s*$")
# "2 Bulbasaur A1 1" in a DeckGym deck file. The name is optional: some of the
# example decks are written as bare "1 A1 128".
DECK_LINE_RE = re.compile(r"^(\d+)\s+(?:.*?\s+)?([A-Za-z0-9-]+)\s+(\d+)$")


# ---------------------------------------------------------------- card database


def card_id(set_code: str, number: str) -> str:
    """Normalize to the `<SET> <NNN>` ids of database.json.

    The pieces reach us in every casing: `b1a-024`, `pa-007`, or a
    `{"set": "P-A", "number": "7"}` pair. Promo sets lose their dash in some
    sources, so `pb` has to become `P-B` rather than `Pb`.
    """
    promo = re.fullmatch(r"p-?([a-z])", set_code, re.I)
    if promo:
        set_code = f"P-{promo.group(1).upper()}"
    else:
        set_code = set_code[:1].upper() + set_code[1:].lower()
    return f"{set_code} {int(number):03d}"


def load_database() -> dict[str, dict]:
    """Card id -> {name, energy_type, evolves_from, attacks, ability, trainer_effect}.

    `energy_type` is None for Trainer cards, which is what tells the two apart.
    Attack and ability titles are kept: the exception list names cards by the
    printing that carries a given attack or ability, and most of those cards
    have several printings under the same name.
    """
    cards = {}
    for entry in json.loads(DATABASE.read_text(encoding="utf-8")):
        (kind, card), = entry.items()
        if kind == "Pokemon":
            ability = card.get("ability") or {}
            cards[card["id"]] = {
                "name": card["name"],
                "energy_type": card.get("energy_type"),
                "evolves_from": card.get("evolves_from"),
                "attacks": [
                    {
                        "title": attack.get("title") or "",
                        "cost": attack.get("energy_required") or [],
                        "effect": attack.get("effect") or "",
                    }
                    for attack in card.get("attacks") or []
                ],
                "ability_title": ability.get("title") or "",
                "ability": ability.get("effect") or "",
                "trainer_effect": None,
            }
        else:
            cards[card["id"]] = {
                "name": card["name"],
                "energy_type": None,
                "evolves_from": None,
                "attacks": [],
                "ability_title": "",
                "ability": "",
                "trainer_effect": card.get("effect") or "",
            }
    return cards


# ------------------------------------------------------------ energy detection


def mentioned_energies(text: str) -> list[str]:
    """Generable energy types named by [X] symbols in a card's text, in order."""
    found = []
    for symbol in SYMBOL_RE.findall(text or ""):
        energy = SYMBOLS.get(symbol)
        if energy in ORDER and energy not in found:
            found.append(energy)
    return found


def generated_energies(text: str) -> list[str]:
    """Energy types a card hands out for free.

    The text has to pull energy out of *your* Energy Zone, rather than merely
    react to an attachment ("whenever you attach a [D] Energy...") or talk
    about the opponent's zone.
    """
    lowered = (text or "").lower()
    if "your energy zone" not in lowered or "their energy zone" in lowered:
        return []
    if not TAKES_RE.search(lowered):
        return []
    return mentioned_energies(text)


def take_top(basket: Counter) -> list[str]:
    """Keep at most three types -- "3 parmi k" -- ranked by how much the deck
    asks for each, ties broken by the canonical energy order."""
    ranked = sorted(basket.items(), key=lambda item: (-item[1], ORDER[item[0]]))
    return sorted((energy for energy, _ in ranked[:3]), key=ORDER.__getitem__)


def matches(card: dict, rule: dict, names: set[str]) -> bool:
    """Does a card match an exception rule, in a deck holding `names`?"""
    if card["name"] != rule["name"]:
        return False
    if "ability" in rule and card["ability_title"] != rule["ability"]:
        return False
    if "attack" in rule and not any(a["title"] == rule["attack"] for a in card["attacks"]):
        return False
    if "with" in rule and not any(name.startswith(rule["with"]) for name in names):
        return False
    return True


def colorless_line(pokemon: list[tuple[str, dict]]) -> set[str]:
    """Pre-evolutions of a line whose last card attacks for Colorless only.

    Nobody attacks with the Frogadier of a line that ends on a Colorless
    attacker, so what its own attacks cost says nothing about the deck.
    """
    evolutions: dict[str, list[tuple[str, dict]]] = {}
    for cid, card in pokemon:
        if card["evolves_from"]:
            evolutions.setdefault(card["evolves_from"], []).append((cid, card))

    neutral = set()
    for cid, card in pokemon:
        # Walk down to the cards this one evolves into that evolve no further.
        finals, pending = [], list(evolutions.get(card["name"], []))
        while pending:
            _, into = pending.pop()
            following = evolutions.get(into["name"], [])
            pending.extend(following) if following else finals.append(into)
        if finals and all(
            energy == "Colorless"
            for final in finals
            for attack in final["attacks"]
            for energy in attack["cost"]
        ):
            neutral.add(cid)
    return neutral


def type_neutral(pokemon: list[tuple[str, dict]]) -> set[str]:
    """Card ids of the deck whose type must not weigh in on its own."""
    names = {card["name"] for _, card in pokemon}
    neutral = {
        cid
        for cid, card in pokemon
        for rule in TYPE_NEUTRAL
        if matches(card, rule, names)
    }
    neutral |= colorless_line(pokemon)

    # An exception covers the whole line: a Froakie is only in the deck to
    # become the Greninja that earned the exception.
    by_name: dict[str, list[str]] = {}
    for cid, card in pokemon:
        by_name.setdefault(card["name"], []).append(cid)
    cards = dict(pokemon)
    pending = list(neutral)
    while pending:
        parent = cards[pending.pop()]["evolves_from"]
        for cid in by_name.get(parent, []):
            if cid not in neutral:
                neutral.add(cid)
                pending.append(cid)
    return neutral


def decide_energy(
    cards: list[tuple[str, int]],
    db: dict[str, dict],
    exclude_free: bool = EXCLUDE_FREE,
    use_exceptions: bool = USE_EXCEPTIONS,
) -> tuple[list[str], str]:
    """Derive the energy types a deck generates, for decks that do not say.

    1. Pot the non-Colorless energy every attack in the deck asks for, minus
       the cards on the `TYPE_NEUTRAL` list, which are in the deck for
       something other than their type. Those go in a side pot, and are poured
       back in when they turn out to be the only types the deck asks for.
    2. Drop the types the deck gets for free: a Pokemon with an ability, or
       with an attack costing nothing but Colorless, that takes energy from the
       Energy Zone.
    3. Stop as soon as the pot holds 1 to 3 types.
    4. Otherwise add the types named by the Trainer cards and stop there,
       keeping the 3 most asked for if the pot holds more.
    5. If the pot is still empty, fall back to the non-Colorless types of the
       Pokemon themselves, and finally to a single arbitrary type.

    `exclude_free` runs step 2 and `use_exceptions` runs the exception list of
    step 1; both are knobs the `sweep` command scores against the example decks.

    Returns the energy list and which step decided it.
    """
    pokemon = [(cid, count) for cid, count in cards if (db.get(cid) or {}).get("energy_type")]
    trainers = [
        (cid, count) for cid, count in cards if (db.get(cid) or {}).get("trainer_effect") is not None
    ]
    neutral = type_neutral([(cid, db[cid]) for cid, _ in pokemon]) if use_exceptions else set()

    basket: Counter = Counter()
    aside: Counter = Counter()
    for cid, count in pokemon:
        pot = aside if cid in neutral else basket
        for attack in db[cid]["attacks"]:
            for energy in attack["cost"]:
                if energy in ORDER:
                    pot[energy] += count
    if not basket:
        # The excepted cards were the only types present after all.
        basket, aside = aside, Counter()

    if use_exceptions and len(set(basket) | set(aside)) == 2:
        # Dragon Arcana wants two types attached, so keep the deck's two.
        names = {db[cid]["name"] for cid, _ in pokemon}
        if any(matches(db[cid], DUAL_TYPE, names) for cid, _ in pokemon):
            return sorted(set(basket) | set(aside), key=ORDER.__getitem__), "attacks (dual type)"

    if exclude_free and len(basket) > 1:
        # A mono type deck is never touched: its one type is what the free
        # generator draws from the Energy Zone, not something it replaces.
        free: set[str] = set()
        for cid, _ in pokemon:
            card = db[cid]
            free.update(generated_energies(card["ability"]))
            for attack in card["attacks"]:
                # "attaque colorless pure, sans énergie": an empty cost qualifies.
                if all(energy == "Colorless" for energy in attack["cost"]):
                    free.update(generated_energies(attack["effect"]))
        kept = Counter({e: w for e, w in basket.items() if e not in free})
        # Excluding every type would say the deck runs none, so keep them then.
        basket = kept or basket

    if 1 <= len(basket) <= 3:
        return sorted(basket, key=ORDER.__getitem__), "attacks"

    from_attacks = bool(basket)
    for cid, count in trainers:
        for energy in mentioned_energies(db[cid]["trainer_effect"]):
            basket[energy] += count
    if basket:
        source = "attacks+trainers" if from_attacks else "trainers"
        if len(basket) > 3:
            source += " (top 3)"
        return take_top(basket), source

    for cid, count in pokemon:
        energy = db[cid]["energy_type"]
        if energy in ORDER and cid not in neutral:
            basket[energy] += count
    if not basket:
        for cid, count in pokemon:
            if db[cid]["energy_type"] in ORDER:
                basket[db[cid]["energy_type"]] += count
    if basket:
        return take_top(basket), "pokemon types" + (" (top 3)" if len(basket) > 3 else "")

    # Nothing in the deck points at an energy type, so pick one. Seeded by the
    # deck itself, so that rebuilding the archive gives the same answer.
    seed = hashlib.sha256(repr(sorted(cards)).encode()).digest()
    return [GENERABLE[seed[0] % len(GENERABLE)]], "arbitrary"


# -------------------------------------------------------------- playable check


def load_unimplemented() -> dict[str, str]:
    """Card id -> why the engine cannot run it, from the card_status cache."""
    if not STATUS_CACHE.exists():
        print(f"warn: {STATUS_CACHE.name} missing, blockers ignore implementation status",
              file=sys.stderr)
        return {}
    reasons = {}
    for line in STATUS_CACHE.read_text(encoding="utf-8-sig").splitlines():
        row = STATUS_ROW_RE.match(line)
        if row and "not implemented" in row.group(3):
            reasons[row.group(1)] = row.group(3)
    return reasons


def blockers_for(
    cards: list[tuple[str, int]], db: dict[str, dict], unimplemented: dict[str, str]
) -> list[str]:
    """Why the simulator cannot run this deck, same rules as the preset decks."""
    blockers = []

    total = sum(count for _, count in cards)
    if total != DECK_SIZE:
        blockers.append(f"{total} cards instead of {DECK_SIZE}")
    for card_ref, count in cards:
        if count > MAX_COPIES:
            blockers.append(f"{card_ref} in {count} copies, max is {MAX_COPIES}")
        if card_ref not in db:
            blockers.append(f"{card_ref} is not in database.json")
        elif card_ref in unimplemented:
            blockers.append(f"{card_ref} {db[card_ref]['name']}: {unimplemented[card_ref].lower()}")
    return blockers


def newest_expansion(cards: list[tuple[str, int]], db: dict[str, dict]) -> str | None:
    """The newest set a deck draws from: it dates the list inside the meta.
    Promo sets (P-A, P-B) date nothing."""
    sets = {cid.split(" ")[0] for cid, _ in cards if cid in db}
    sets = {code for code in sets if not code.startswith("P-")}
    if not sets:
        return None
    # "A1" < "A1a" < "A2" < ... < "B1" < "B1a"
    return max(sets, key=lambda code: (code[0], int(code[1:2] or 0), code[2:]))


# ------------------------------------------------------------------ validation


def parse_deck_file(raw: str) -> tuple[list[str], list[tuple[str, int]]]:
    """Read a DeckGym deck file: an `Energy:` line plus `<count> <name> <set> <number>`."""
    energy: list[str] = []
    cards: list[tuple[str, int]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("Pokémon:", "Trainer:")):
            continue
        if line.startswith("Energy:"):
            energy = [e.strip() for e in line.removeprefix("Energy:").split(",") if e.strip()]
            continue
        row = DECK_LINE_RE.match(line)
        if row:
            cards.append((card_id(row.group(2), row.group(3)), int(row.group(1))))
    return energy, cards


def load_examples() -> list[tuple[str, list[str], list[tuple[str, int]]]]:
    """The example decks that declare an energy, i.e. the ground truth."""
    decks = []
    for path in sorted(EXAMPLE_DECKS.glob("*.txt")):
        declared, cards = parse_deck_file(path.read_text(encoding="utf-8"))
        decks.append((path.stem, declared, cards))
    return decks


def sweep() -> int:
    """Score the algorithm's two knobs against the example decks."""
    db = load_database()
    truth = [(name, declared, cards) for name, declared, cards in load_examples() if declared]

    print(f"{'exclude free energy':<21} {'exceptions':<11} {'matches'}")
    for exclude_free in (True, False):
        for use_exceptions in (False, True):
            hits = sum(
                set(declared) == set(decide_energy(cards, db, exclude_free, use_exceptions)[0])
                for _, declared, cards in truth
            )
            print(
                f"{'yes (as specified)' if exclude_free else 'no':<21} "
                f"{'yes' if use_exceptions else 'no':<11} "
                f"{hits:>4}/{len(truth):<3} {100 * hits / len(truth):>3.0f}%"
            )
    return 0


def validate(exclude_free: bool = EXCLUDE_FREE, use_exceptions: bool = USE_EXCEPTIONS) -> int:
    """Run the energy algorithm on the example decks, which declare their energy."""
    db = load_database()
    hits = total = skipped = 0
    mismatches = []

    print(f"{'deck':<32} {'declared':<18} {'derived':<18} {'source':<24} ")
    for name, declared, cards in load_examples():
        if not declared:
            skipped += 1
            print(f"{name:<32} {'(not declared)':<18} {'-':<18} {'-':<24} skipped")
            continue
        energy, source = decide_energy(cards, db, exclude_free, use_exceptions)
        ok = set(declared) == set(energy)
        total += 1
        hits += ok
        if not ok:
            mismatches.append((name, declared, energy, source))
        print(
            f"{name:<32} {','.join(declared):<18} {','.join(energy):<18} "
            f"{source:<24} {'ok' if ok else 'MISMATCH'}"
        )

    print(f"\n{hits}/{total} decks match ({100 * hits / max(total, 1):.0f}%), "
          f"{skipped} skipped for lack of a declared energy")
    for name, declared, energy, source in mismatches:
        print(f"  {name}: declared {declared}, derived {energy} from {source}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "sweep"), nargs="?", default="validate")
    parser.add_argument(
        "--no-exclude-free",
        action="store_true",
        help="skip step 2, which drops the energy types the deck gets for free",
    )
    parser.add_argument(
        "--no-exceptions",
        action="store_true",
        help="ignore the TYPE_NEUTRAL card list of step 1",
    )
    args = parser.parse_args()

    if args.command == "sweep":
        return sweep()
    return validate(not args.no_exclude_free, not args.no_exceptions)


if __name__ == "__main__":
    raise SystemExit(main())
