"""Learn a deck's energy from the lists that declare theirs.

`deck_energy.decide_energy` is a hand-written decision tree. A third of the Limitless
archive states its energy outright, which turns the same problem into a supervised one:
train on those, predict the rest.

Scored by `build_meta_decks.py check`, five rolling-origin folds, exact set match:

    fold train size   test from     heuristic   model
       1      10 308   2025-02-28      93.19%   92.40%
       2      20 612   2025-04-06      92.67%   94.10%
       3      30 916   2025-08-01      93.59%   93.81%
       4      41 220   2025-12-15      93.91%   94.91%
       5      51 524   2026-03-27      90.52%   94.24%
                                mean   92.78%   93.89%
                                 std     1.20     0.83

Read the folds, not the mean. The model loses fold 1 outright: under ~20k labelled lists
the hand-written rules are better, so this only earns its place on a full archive. And the
mean understates what matters -- on the newest meta the heuristic drops to its worst score
while the model holds, because hand-written rules age as mechanics arrive and a retrained
model does not. That gap, +3.7, is the one that applies to the decks being added from here.

A single 80/20 split reported 95.3% for this module against 91.4%; that was one lucky cut,
which is why the folds are what this docstring quotes.

On how far there is left to go: 577 of the 61 828 declared multisets (0.9%) were declared
two or more different ways by different players -- one of them seven different ways. Since
both predictors read nothing but the cards, those are identical inputs with contradictory
labels. That caps the *per player entry* metric at 99.24%. It does not cap the numbers
above, which grade one row per multiset against its majority label and so have already
collapsed the contradictions away. Either way it is a generous bound: it only counts
ambiguity that happened to be observed twice, and a Colorless-attacker deck free to run any
energy is just as unguessable when only one player ever published it.

Other models on the same features, single split, for the record: one depth-12 tree 94.4%,
random forest 94.6%, plain gradient boosting 95.0%. The 1-3 energies repair is worth +0.3
on top of any of them.

The features are not the card ids. Those were measured at 86.4% -- 1902 sparse columns
force the model to memorise decks rather than generalise, and adding them on top of the
features below buys 0.1 point. What carries the signal is the four aggregates the
heuristic already reasons over, one weight per energy type: what the attacks cost, what
the Pokemon are, what the Trainers name, and what comes free out of the Energy Zone.
Attack cost dominates; the Trainer and Pokemon-type blocks weigh under 0.01 between them,
which says the heuristic's step 4 and 5 fallbacks are near-irrelevant.

Hyperparameters are left at their defaults on purpose: lr=0.05 with 500 iterations and 63
leaves was measured at the same 94.99% as the defaults, so there is nothing to tune here.

Needs scikit-learn and numpy, which the callers pull with `uv run --with`. Import it
lazily: `fetch` and the heuristic itself must keep working without them.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from deck_energy import GENERABLE, ORDER, generated_energies, mentioned_energies

BLOCKS = ("attack cost", "pokemon type", "trainer names", "free generator")
FEATURE_NAMES = [f"{block} {energy}" for block in BLOCKS for energy in GENERABLE]
SEED = 0


def featurize(decks: list[list[tuple[str, int]]], db: dict[str, dict]) -> np.ndarray:
    """One row per deck: four blocks of eight weights, one per generable energy type."""
    matrix = np.zeros((len(decks), len(FEATURE_NAMES)), dtype=np.float32)
    for row, cards in enumerate(decks):
        for cid, count in cards:
            card = db.get(cid)
            if not card:
                continue
            for attack in card["attacks"]:
                for energy in attack["cost"]:
                    if energy in ORDER:
                        matrix[row, ORDER[energy]] += count
            if card["energy_type"] in ORDER:
                matrix[row, 8 + ORDER[card["energy_type"]]] += count
            if card["trainer_effect"] is not None:
                for energy in mentioned_energies(card["trainer_effect"]):
                    matrix[row, 16 + ORDER[energy]] += count
            for energy in generated_energies(card["ability"]):
                matrix[row, 24 + ORDER[energy]] += count
    return matrix


class EnergyModel:
    """One binary gradient booster per energy type, plus the deck-legality repair."""

    def __init__(self, boosters):
        self.boosters = boosters

    @classmethod
    def train(cls, decks: list[list[tuple[str, int]]], energies: list[list[str]], db) -> "EnergyModel":
        features = featurize(decks, db)
        labels = np.array(
            [[energy in declared for energy in GENERABLE] for declared in energies], dtype=np.int8
        )
        boosters = []
        for index in range(len(GENERABLE)):
            booster = HistGradientBoostingClassifier(random_state=SEED)
            booster.fit(features, labels[:, index])
            boosters.append(booster)
        return cls(boosters)

    def predict(self, decks: list[list[tuple[str, int]]], db) -> list[list[str]]:
        """Energy lists in canonical order, always 1 to 3 long.

        Nothing in a per-type booster knows a Pocket deck holds between one and three
        energies, so the raw thresholds have to be repaired: an empty prediction keeps its
        likeliest type, an over-full one keeps its best three. Worth +0.3 points, and it is
        the difference between a prediction and one `Deck::from_string` would reject.
        """
        if not decks:
            return []
        features = featurize(decks, db)
        proba = np.column_stack(
            [booster.predict_proba(features)[:, 1] for booster in self.boosters]
        )

        predictions = []
        for row in proba:
            picked = [index for index, p in enumerate(row) if p >= 0.5]
            if not picked:
                picked = [int(row.argmax())]
            elif len(picked) > 3:
                picked = sorted(np.argsort(row)[::-1][:3])
            predictions.append([GENERABLE[index] for index in sorted(picked)])
        return predictions
