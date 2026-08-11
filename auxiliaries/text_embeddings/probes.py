"""Sanity probes for the frozen encoder (the falsifiable gate before freezing).

Each probe is a triplet (anchor, near, far): the anchor must be closer to `near` than to
`far` (cosine). Probes are checked in the raw bge space AND after PCA projection at 128
(effect table) and 48 (ability table) dims — a probe that passes raw but fails projected
means the compression, not the encoder, destroyed the distinction.

Exit code 1 if any probe fails UNEXPECTEDLY in any space (so this can gate CI). Probes in
EXPECTED_FAILURES are known encoder limitations, reported but non-fatal.
"""

from __future__ import annotations

import sys

import numpy as np

import common

# (label, anchor, near, far) — phrased in Pocket's rules language, mechanics the spec
# calls load-bearing: coin flips, bench snipe, energy accel vs denial, status, heal,
# search/draw, self-damage, attack/retreat locks, target side.
PROBES = [
    (
        "coin flip damage bonus",
        "Flip a coin. If heads, this attack does 40 more damage.",
        "Flip a coin. If heads, this attack does 60 more damage.",
        "Heal 30 damage from this Pokémon.",
    ),
    (
        "multi coin flip scaling",
        "Flip 2 coins. This attack does 50 damage for each heads.",
        "Flip 4 coins. This attack does 40 damage for each heads.",
        "Draw 2 cards.",
    ),
    (
        "bench snipe",
        "This attack does 20 damage to 1 of your opponent's Benched Pokémon.",
        "This attack does 30 damage to 1 of your opponent's Benched Pokémon.",
        "Your opponent's Active Pokémon is now Asleep.",
    ),
    (
        "status conditions cluster",
        "Your opponent's Active Pokémon is now Poisoned.",
        "Your opponent's Active Pokémon is now Burned.",
        "Draw 2 cards.",
    ),
    (
        "energy acceleration",
        "Take a Grass Energy from your Energy Zone and attach it to this Pokémon.",
        "Take a Water Energy from your Energy Zone and attach it to 1 of your Benched Pokémon.",
        "Discard an Energy from your opponent's Active Pokémon.",
    ),
    (
        "energy discard target side",
        "Discard an Energy from this Pokémon.",
        "Discard 2 Energy from this Pokémon.",
        "Discard an Energy from your opponent's Active Pokémon.",
    ),
    (
        "deck search",
        "Put 1 random Basic Pokémon from your deck into your hand.",
        "Put 1 random Pokémon from your deck into your hand.",
        "This attack does 20 damage to itself.",
    ),
    (
        "heal",
        "Heal 30 damage from this Pokémon.",
        "Heal 50 damage from each of your Pokémon.",
        "This attack does 50 damage to 1 of your opponent's Pokémon.",
    ),
    (
        "attack lock vs retreat lock",
        "During your opponent's next turn, the Defending Pokémon can't attack.",
        "During your opponent's next turn, the Defending Pokémon can't retreat.",
        "Take a Fire Energy from your Energy Zone and attach it to this Pokémon.",
    ),
    (
        "damage reduction",
        "During your opponent's next turn, this Pokémon takes −30 damage from attacks.",
        "During your opponent's next turn, this Pokémon takes −20 damage from attacks.",
        "During your opponent's next turn, the Defending Pokémon can't attack.",
    ),
    (
        "self switch vs gust",
        "Switch this Pokémon with 1 of your Benched Pokémon.",
        "Switch out your opponent's Active Pokémon to the Bench.",
        "Heal 20 damage from this Pokémon.",
    ),
    (
        "self damage recoil",
        "This attack also does 20 damage to itself.",
        "This Pokémon also does 50 damage to itself.",
        "This attack does 20 more damage for each of your Benched Pokémon.",
    ),
    (
        "bench scaling",
        "This attack does 20 more damage for each of your Benched Pokémon.",
        "This attack does 30 damage for each of your Benched Lightning Pokémon.",
        "Discard a random card from your opponent's hand.",
    ),
    (
        "hand disruption",
        "Discard a random card from your opponent's hand.",
        "Your opponent reveals their hand.",
        "Heal 30 damage from this Pokémon.",
    ),
    (
        "energy symbol normalization",
        "Take a [G] Energy from your Energy Zone and attach it to this Pokémon.",
        "Take a Grass Energy from your Energy Zone and attach it to this Pokémon.",
        "Take a Fire Energy from your Energy Zone and attach it to this Pokémon.",
    ),
    (
        "pokemon name normalization",
        "This attack does 30 more damage for each of your Benched Nidoking.",
        "This attack does 30 more damage for each of your Benched Pokémon.",
        "This attack does 30 more damage for each of your opponent's Benched Pokémon.",
    ),
    (
        "mechanic marker normalization",
        "This attack does 30 more damage to Pokémon VMAX.",
        "This attack does 30 more damage to Pokémon ex.",
        "This attack does 30 more damage to Basic Pokémon.",
    ),
]


# Known limitation of a generic sentence encoder (fails already in the raw 384 space, so
# compression is not at fault): surface overlap ("This attack ... 20 ... damage") outweighs
# the logical role ("to itself" vs "per Benched Pokémon"). This is exactly the numeric/logical
# blind spot §1.2.9 assigns to the future structured attack schema, not to a better encoder.
EXPECTED_FAILURES = {"self damage recoil"}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    model = common.load_encoder()
    components, mean, effect_scale, ability_scale = common.load_pca()

    texts = [t for probe in PROBES for t in probe[1:]]
    raw = common.embed(model, [common.normalize(t) for t in texts])
    # Each projected space carries its own block's scale: the whitening changes cosines, so
    # slicing the effect space down to 48 dims would gate a geometry no table is written in.
    spaces = {
        "raw-384": raw,
        "pca-128": common.project(raw, components, mean, effect_scale),
        "pca-48": common.project(raw, components, mean, ability_scale),
    }

    unexpected = 0
    for space, emb in spaces.items():
        print(f"\n=== {space} ===")
        for i, (label, *_rest) in enumerate(PROBES):
            anchor, near, far = emb[3 * i], emb[3 * i + 1], emb[3 * i + 2]
            sim_near, sim_far = cosine(anchor, near), cosine(anchor, far)
            ok = sim_near > sim_far
            if ok:
                status = "ok   "
            elif label in EXPECTED_FAILURES:
                status = "xfail"
            else:
                status = "FAIL "
                unexpected += 1
            print(f"  {status} {label:32s} near={sim_near:+.3f} far={sim_far:+.3f}")

    print(
        f"\n{unexpected} unexpected failure(s) across {len(spaces)} spaces x "
        f"{len(PROBES)} probes ({len(EXPECTED_FAILURES)} known xfail)"
    )
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
