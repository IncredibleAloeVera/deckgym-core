"""Build the meta deck archive out of the Limitless tournament API.

    play.limitlesstcg.com/api  --> auxiliaries/.limitless_cache/  --> meta_decks.json

Replaces the previous archive, which walked the git history of a third party tier list
repo: that file was overwritten on every meta refresh, changed schema mid-life, and stopped
publishing lists in bulk. Limitless serves decklists that were actually played, each with
the tournament, the placing and the match record attached, back to 2024-10-23.

Three commands, only the first of which touches the network:

    fetch   walk the tournament index, then GET the standings of every tournament not
            already cached. Ctrl-C is the pause button: one cached file per tournament,
            written atomically, so a rerun resumes at the tournament it stopped on.
    build   read the cache and write `meta_decks.json`. Offline, so the energy heuristic
            and the playability rules can be re-run at will.
    check   score `decide_energy` against the lists that state their energy.

About a fifth of the archive declares its energy outright, in an `energy` section holding
type names rather than cards. Those lists are taken at their word, and they double as a
ground truth for `deck_energy.py`, which until now had only 30 example decks to answer to.

Decks are keyed by their exact card multiset, so one entry is one distinct list, however
many players brought it. The `entries` and `record` fields say how popular and how good it
was, which the tier list archive could not.

Only tournaments held before B4 are kept: `database.json` stops at B3b, so a B4 list would
be dropped by `blockers_for` anyway, one card at a time. `--until` moves the cutoff.

Rate limit: the API allows 50 requests per 5 minutes per IP, announced in the `RateLimit`
headers. This paces itself at `--budget` of that (60% by default) and parks until the
window resets when the remaining count runs low, rather than racing the limit and eating
429s. A full run is ~3.9k requests, so about 11 hours at that budget; the cache makes every
later run cost only the tournaments played since.

Usage:
    uv run --no-project --python 3.14 auxiliaries/build_meta_decks.py fetch
    uv run --no-project --python 3.14 auxiliaries/build_meta_decks.py build

    ... fetch --max 100          # a taste of the archive without the full run
    ... build --exclude-format NOEX

Stdlib only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from deck_energy import (
    EXCLUDE_FREE,
    ORDER,
    USE_EXCEPTIONS,
    blockers_for,
    card_id,
    decide_energy,
    load_database,
    load_unimplemented,
    newest_expansion,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".limitless_cache"
STANDINGS = CACHE / "standings"
INDEX = CACHE / "index.json"
OUTPUT = HERE / "meta_decks.json"

API = "https://play.limitlesstcg.com/api"
GAME = "POCKET"
PAGE_SIZE = 1000
# Limitless documents no User-Agent requirement; this is ordinary etiquette. It buys two
# things: the default `Python-urllib/3.x` is the signature blanket bans are written against,
# and a named client with a URL can be contacted about a problem instead of just cut off.
USER_AGENT = "deckgym-core/1.0 (+https://github.com/IncredibleAloeVera)"

# B4 "Ruler of the Skies" went live 2026-07-29 18:00 PDT. Tournaments from then on can play
# cards `database.json` does not have.
B4_RELEASE = "2026-07-30T01:00:00Z"

# `RateLimit: "50-in-5min"; r=30; t=58` and `RateLimit-Policy: "50-in-5min"; q=50; w=300`.
LIMIT_RE = re.compile(r"\br=(\d+)")
RESET_RE = re.compile(r"\bt=(\d+)")
QUOTA_RE = re.compile(r"\bq=(\d+)")
WINDOW_RE = re.compile(r"\bw=(\d+)")

DEFAULT_BUDGET = 0.6
MAX_RETRIES = 5


def parse_time(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


class Client:
    """A GET that keeps itself inside the server's announced rate limit.

    The pacing is derived from the headers rather than hardcoded, so a change of policy on
    the server side slows the run down instead of breaking it. Two mechanisms, because
    either alone is not enough: a minimum interval between requests keeps the average rate
    under budget, and parking when `r` runs low covers the case where the window was
    already partly spent by something else on this IP.
    """

    def __init__(self, budget: float = DEFAULT_BUDGET, verbose: bool = True):
        self.budget = budget
        self.verbose = verbose
        self.quota, self.window = 50, 300
        self.interval = self.window / (self.quota * self.budget)
        self.last = 0.0
        self.count = 0

    def _observe(self, headers) -> None:
        policy = headers.get("RateLimit-Policy", "")
        quota = QUOTA_RE.search(policy)
        window = WINDOW_RE.search(policy)
        if quota and window:
            self.quota, self.window = int(quota.group(1)), int(window.group(1))
            self.interval = self.window / (self.quota * self.budget)

        limit = headers.get("RateLimit", "")
        remaining = LIMIT_RE.search(limit)
        reset = RESET_RE.search(limit)
        if remaining and reset:
            # The reserve is the slice of the quota the budget deliberately leaves unspent.
            reserve = max(1, round(self.quota * (1 - self.budget)))
            if int(remaining.group(1)) <= reserve:
                pause = int(reset.group(1)) + 1
                if self.verbose:
                    print(f"  rate limit reserve reached, parking {pause}s", flush=True)
                time.sleep(pause)

    def get(self, path: str) -> object:
        for attempt in range(MAX_RETRIES):
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

            request = urllib.request.Request(
                f"{API}{path}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.count += 1
                    self._observe(response.headers)
                    return payload
            except urllib.error.HTTPError as error:
                if error.code == 429:
                    pause = int(error.headers.get("Retry-After") or self.window)
                    print(f"  429, waiting {pause}s", file=sys.stderr, flush=True)
                    time.sleep(pause + 1)
                    continue
                if error.code >= 500:
                    pause = 5 * 2**attempt
                    print(f"  {error.code}, retrying in {pause}s", file=sys.stderr, flush=True)
                    time.sleep(pause)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as error:
                pause = 5 * 2**attempt
                print(f"  {error}, retrying in {pause}s", file=sys.stderr, flush=True)
                time.sleep(pause)
        raise RuntimeError(f"{path} failed after {MAX_RETRIES} attempts")


# ----------------------------------------------------------------------- fetch


def fetch_index(client: Client) -> list[dict]:
    """Every POCKET tournament, newest first. Four pages at the time of writing."""
    tournaments: list[dict] = []
    page = 1
    while True:
        batch = client.get(
            f"/tournaments?game={GAME}&limit={PAGE_SIZE}&page={page}"
        )
        tournaments.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    return tournaments


def standings_path(tournament_id: str) -> Path:
    return STANDINGS / f"{tournament_id}.json.gz"


def fetch(until: str, budget: float, maximum: int | None) -> int:
    STANDINGS.mkdir(parents=True, exist_ok=True)
    client = Client(budget)
    cutoff = parse_time(until)

    print(f"pacing at {client.interval:.1f}s between requests ({budget:.0%} of the limit)")
    index = fetch_index(client)
    INDEX.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"index: {len(index)} tournaments, oldest {min(t['date'] for t in index)[:10]}")

    wanted = [t for t in index if parse_time(t["date"]) < cutoff]
    missing = [t for t in wanted if not standings_path(t["id"]).exists()]
    cached = len(wanted) - len(missing)
    if maximum is not None:
        # Newest first, so a capped run samples the most recent meta rather than 2024.
        missing = missing[:maximum]

    eta = len(missing) * client.interval / 3600
    print(f"{len(wanted)} before {until}, {cached} cached, {len(missing)} to fetch (~{eta:.1f}h)")

    done = 0
    try:
        for done, tournament in enumerate(missing, 1):
            standings = client.get(f"/tournaments/{tournament['id']}/standings")
            # Written aside and moved into place, so that a Ctrl-C mid-write cannot leave a
            # truncated archive behind -- which the next run would count as cached and the
            # build would then choke on. `os.replace` is atomic on Windows and POSIX alike.
            final = standings_path(tournament["id"])
            staging = final.with_suffix(".part")
            with gzip.open(staging, "wt", encoding="utf-8") as handle:
                json.dump(standings, handle, ensure_ascii=False)
            os.replace(staging, final)
            if done % 25 == 0 or done == len(missing):
                print(
                    f"  {done}/{len(missing)}  {tournament['date'][:10]}  "
                    f"{tournament['name'][:48]}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print(f"\ninterrupted after {done} tournaments; rerun `fetch` to carry on", flush=True)
        return 130
    print(f"fetched {client.count} requests, cache holds {len(list(STANDINGS.glob('*.json.gz')))}")
    return 0


# ----------------------------------------------------------------------- build


def read_cache() -> tuple[dict[str, dict], list[Path]]:
    if not INDEX.exists():
        print(f"error: {INDEX} missing, run `fetch` first", file=sys.stderr)
        raise SystemExit(1)
    index = {t["id"]: t for t in json.loads(INDEX.read_text(encoding="utf-8"))}
    return index, sorted(STANDINGS.glob("*.json.gz"))


def parse_decklist(decklist: dict) -> tuple[list[tuple[str, int]], list[str]] | None:
    """A Limitless decklist as (card multiset, declared energy), or None if unusable.

    `energy` is a list of plain type names, not of cards -- the Energy Zone holds types, not
    printings. About a fifth of the archive carries it, which is the only place a Pocket
    decklist ever states its energy outright; the rest goes through `decide_energy`.

    Card sections beyond pokemon/trainer are read too: Pocket has none today, but a list that
    grew one would silently lose cards and then fail the 20 card check for the wrong reason.
    """
    cards: Counter = Counter()
    declared: list[str] = []
    for name, section in decklist.items():
        if not isinstance(section, list):
            continue
        if name == "energy":
            declared = [energy for energy in section if isinstance(energy, str)]
            continue
        for entry in section:
            try:
                cards[card_id(entry["set"], entry["number"])] += int(entry["count"])
            except (KeyError, TypeError, ValueError):
                return None
    if not cards:
        return None
    return sorted(cards.items()), declared


def train_and_predict(decks, db, use_model: bool) -> dict[tuple, list[str]]:
    """Energy for the lists that do not declare one, learned from the lists that do.

    Trained here rather than shipped as an artefact: the training set is the cache sitting
    next to this script, the seed is fixed, so a rerun on the same cache gives the same
    answer, and the model follows the meta as the archive grows. Returns an empty mapping
    when the model is off or scikit-learn is missing -- the caller falls back to
    `decide_energy`, which needs nothing but the stdlib.
    """
    if not use_model:
        return {}
    try:
        from energy_model import EnergyModel
    except ImportError as error:
        print(f"warn: no model ({error}), falling back to the heuristic", file=sys.stderr)
        return {}

    labelled = [(key, deck) for key, deck in decks_with_keys(decks) if deck["declared"]]
    unlabelled = [(key, deck) for key, deck in decks_with_keys(decks) if not deck["declared"]]
    if not labelled or not unlabelled:
        return {}

    model = EnergyModel.train(
        [deck["cards"] for _, deck in labelled], [deck["declared"] for _, deck in labelled], db
    )
    guesses = model.predict([deck["cards"] for _, deck in unlabelled], db)
    print(f"model trained on {len(labelled)} declared lists, predicting {len(unlabelled)}")
    return {key: energy for (key, _), energy in zip(unlabelled, guesses)}


def decks_with_keys(decks):
    """`lists` is passed as its values; rebuild the key each entry was stored under."""
    for deck in decks:
        yield (tuple(deck["cards"]), tuple(deck["declared"])), deck


def build(
    until: str,
    exclude_formats: set[str],
    use_model: bool = True,
    exclude_free: bool = EXCLUDE_FREE,
    use_exceptions: bool = USE_EXCEPTIONS,
) -> int:
    db = load_database()
    unimplemented = load_unimplemented()
    index, files = read_cache()
    cutoff = parse_time(until)

    lists: dict[tuple, dict] = {}
    seen = players = listed = malformed = 0
    for path in files:
        tournament = index.get(path.stem.removesuffix(".json"))
        if tournament is None:
            continue
        if parse_time(tournament["date"]) >= cutoff:
            continue
        if (tournament.get("format") or "") in exclude_formats:
            continue
        seen += 1

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            standings = json.load(handle)
        date = tournament["date"]
        for entry in standings:
            players += 1
            decklist = entry.get("decklist")
            if not decklist:
                continue
            parsed = parse_decklist(decklist)
            if parsed is None:
                malformed += 1
                continue
            cards, declared = parsed
            declared = sorted((e for e in declared if e in ORDER), key=ORDER.__getitem__)
            listed += 1

            # A list that declares its energy and the same list that does not are separate
            # entries: the declaration is the fact, and merging would let the heuristic
            # overwrite it. Identical results are collapsed again by build_deck_dbs.py.
            deck = lists.setdefault(
                (tuple(cards), tuple(declared)),
                {
                    "cards": cards,
                    "declared": declared,
                    "archetypes": Counter(),
                    "entries": 0,
                    "tournaments": set(),
                    "record": Counter(),
                    "best_placing": None,
                    "first_seen": date,
                    "last_seen": date,
                },
            )
            deck["entries"] += 1
            deck["tournaments"].add(tournament["id"])
            deck["first_seen"] = min(deck["first_seen"], date)
            deck["last_seen"] = max(deck["last_seen"], date)
            if entry.get("deck"):
                deck["archetypes"][(entry["deck"].get("id"), entry["deck"].get("name"))] += 1
            deck["record"].update(entry.get("record") or {})
            placing = entry.get("placing")
            if placing and (deck["best_placing"] is None or placing < deck["best_placing"]):
                deck["best_placing"] = placing

    predicted = train_and_predict(lists.values(), db, use_model)

    decks = []
    sources: Counter = Counter()
    for key, deck in lists.items():
        cards = deck["cards"]
        if deck["declared"]:
            energy, source = deck["declared"], "declared"
        elif key in predicted:
            energy, source = predicted[key], "model"
        else:
            energy, source = decide_energy(cards, db, exclude_free, use_exceptions)
        blockers = blockers_for(cards, db, unimplemented)
        sources[source] += 1
        # The archetype the list was classified as most often. A list can drift between
        # labels as Limitless updates its rules, and the majority is the honest answer.
        (archetype, name), _ = (
            deck["archetypes"].most_common(1)[0] if deck["archetypes"] else ((None, None), 0)
        )
        decks.append(
            {
                "archetype": archetype or "unclassified",
                "name": name or "Unclassified",
                "expansion": newest_expansion(cards, db),
                "first_seen": deck["first_seen"],
                "last_seen": deck["last_seen"],
                "entries": deck["entries"],
                "tournaments": len(deck["tournaments"]),
                "best_placing": deck["best_placing"],
                "record": {key: deck["record"].get(key, 0) for key in ("wins", "losses", "ties")},
                "energy": energy,
                "energy_source": source,
                "playable": not blockers,
                "blockers": blockers,
                "cards": [
                    {"id": cid, "name": (db.get(cid) or {}).get("name", ""), "count": count}
                    for cid, count in cards
                ],
            }
        )
    decks.sort(key=lambda deck: (-deck["entries"], deck["archetype"]))

    OUTPUT.write_text(
        json.dumps({"decks": decks}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    playable = sum(deck["playable"] for deck in decks)
    archetypes = {deck["archetype"] for deck in decks}
    print(f"{seen} tournaments, {players} entries, {listed} with a decklist")
    print(f"{len(decks)} distinct lists in {len(archetypes)} archetypes, {playable} playable")
    if malformed:
        print(f"  {malformed} decklists could not be parsed")
    for source, count in sources.most_common():
        print(f"  energy from {source:<24} {count:6}")
    print(f"wrote {OUTPUT.name}")
    return 0


def check(
    until: str, exclude_free: bool = EXCLUDE_FREE, use_exceptions: bool = USE_EXCEPTIONS
) -> int:
    """Score `decide_energy` against the decklists that declare their energy.

    `deck_energy.py sweep` has 30 example decks to work with. The archive has thousands, so
    this is the same measurement on a ground truth wide enough to trust -- and it is free,
    the declarations are already cached.
    """
    db = load_database()
    index, files = read_cache()
    cutoff = parse_time(until)

    truth: dict[tuple, list[str]] = {}
    dates: dict[tuple, str] = {}
    for path in files:
        tournament = index.get(path.stem.removesuffix(".json"))
        if tournament is None or parse_time(tournament["date"]) >= cutoff:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for entry in json.load(handle):
                parsed = parse_decklist(entry.get("decklist") or {})
                if parsed is None:
                    continue
                cards, declared = parsed
                declared = sorted((e for e in declared if e in ORDER), key=ORDER.__getitem__)
                if declared:
                    key = tuple(cards)
                    truth[key] = declared
                    # First sighting, so a list is never trained on before it was played.
                    date = tournament["date"]
                    dates[key] = min(dates.get(key, date), date)

    hits = 0
    mismatches: Counter = Counter()
    for cards, declared in truth.items():
        energy, _ = decide_energy(list(cards), db, exclude_free, use_exceptions)
        if set(energy) == set(declared):
            hits += 1
        else:
            mismatches[(",".join(declared), ",".join(energy))] += 1

    total = len(truth)
    print(f"{hits}/{total} distinct declared lists match ({100 * hits / max(total, 1):.1f}%)")
    for (declared, derived), count in mismatches.most_common(20):
        print(f"  {count:5}  declared {declared:<24} derived {derived}")

    score_model(truth, dates, db, exclude_free, use_exceptions)
    return 0


def score_model(truth, dates, db, exclude_free, use_exceptions, folds: int = 5) -> None:
    """Both predictors, cross-validated over a rolling origin.

    Not k-fold: shuffling the folds would train the model on a meta later than the one it is
    tested on, and the same list recurs across dozens of tournaments, so a random split
    leaks and flatters. `TimeSeriesSplit` grows the training window forward instead, which
    is the question actually being asked -- does this survive a meta it has not seen.

    The heuristic is scored on the very same folds. It learns nothing, so its spread across
    folds is a free measurement of how much the metagame itself moves.
    """
    try:
        import numpy as np
        from sklearn.model_selection import TimeSeriesSplit

        from energy_model import EnergyModel
    except ImportError as error:
        print(f"\nno model to score ({error})")
        return

    order = sorted(truth, key=lambda cards: dates[cards])
    if len(order) < folds * 2:
        return

    print(f"\nrolling-origin cross-validation, {folds} folds:")
    print(f"  {'train':>7} {'test':>7}  {'from':>10}   heuristic    model")
    scores = {"heuristic": [], "model": []}
    for train_idx, test_idx in TimeSeriesSplit(n_splits=folds).split(order):
        train = [order[i] for i in train_idx]
        test = [order[i] for i in test_idx]

        model = EnergyModel.train([list(c) for c in train], [truth[c] for c in train], db)
        guesses = model.predict([list(c) for c in test], db)

        base = sum(
            set(decide_energy(list(c), db, exclude_free, use_exceptions)[0]) == set(truth[c])
            for c in test
        ) / len(test)
        learned = sum(set(g) == set(truth[c]) for c, g in zip(test, guesses)) / len(test)
        scores["heuristic"].append(base)
        scores["model"].append(learned)
        print(f"  {len(train):7} {len(test):7}  {dates[test[0]][:10]}     "
              f"{100 * base:6.2f}%  {100 * learned:6.2f}%")

    for label in ("heuristic", "model"):
        values = np.array(scores[label])
        print(f"  {label:>9}  mean {100 * values.mean():6.2f}%  "
              f"std {100 * values.std():.2f}  worst fold {100 * values.min():6.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("fetch", "build", "check"), nargs="?", default="build"
    )
    parser.add_argument(
        "--until", default=B4_RELEASE, help="drop tournaments held on or after this ISO instant"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help="fraction of the announced rate limit to use (fetch)",
    )
    parser.add_argument("--max", type=int, help="fetch at most this many tournaments")
    parser.add_argument(
        "--exclude-format",
        action="append",
        default=[],
        metavar="ID",
        help="skip tournaments run in this format, e.g. NOEX (build, repeatable)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="derive every undeclared energy with the heuristic, never the model (build)",
    )
    parser.add_argument("--no-exclude-free", action="store_true")
    parser.add_argument("--no-exceptions", action="store_true")
    args = parser.parse_args()

    if args.command == "fetch":
        return fetch(args.until, args.budget, args.max)
    if args.command == "check":
        return check(args.until, not args.no_exclude_free, not args.no_exceptions)
    return build(
        args.until,
        set(args.exclude_format),
        not args.no_model,
        not args.no_exclude_free,
        not args.no_exceptions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
