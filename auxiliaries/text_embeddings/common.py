"""Shared pieces of the frozen text-encoder pipeline."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DECKGYM_DATABASE = HERE.parents[1] / "database.json"
CORPUS = HERE / "corpus.json"
NAMES = HERE / "names.json"
OUT_DIR = HERE / "out"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EFFECT_DIM = 128  # src/rl/text_embedding.rs EFFECT_TEXT_DIM
ABILITY_DIM = 48  # src/rl/text_embedding.rs ABILITY_TEXT_DIM

# Energy symbol notation -> plain words. Pocket uses [X]; the super-set corpus (tcgdex) uses
# {X}, with two extra letters Pocket lacks ("R" is Fire, "N" is Dragon in TCG notation).
_ENERGY_LETTERS = {
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
ENERGY_SYMBOLS = {
    f"{open}{letter}{close}": word
    for open, close in (("[", "]"), ("{", "}"))
    for letter, word in _ENERGY_LETTERS.items()
}
# Corpus-only oddballs: {ex} is the literal "ex" marker, {*} is the Prism Star symbol.
ENERGY_SYMBOLS["{ex}"] = "ex"
ENERGY_SYMBOLS["{*}"] = ""


# Special-rule markers that do not exist in Pocket (only "ex" and "Mega ... ex" do). After
# name stripping, any leftover generic marker is mapped onto the one role Pocket knows: "ex".
# Ordered — longer / more specific first.
MECHANIC_MARKERS = [
    (re.compile(r"\bV-UNION\b"), "ex"),
    (re.compile(r"\bVMAX\b"), "ex"),
    (re.compile(r"\bVSTAR\b"), "ex"),
    (re.compile(r"\bLV\.\s?X\b"), "ex"),
    (re.compile(r"-(?:GX|EX)\b"), " ex"),
    (re.compile(r"\bGX\b"), "ex"),
    (re.compile(r"\bEX\b"), "ex"),
    (re.compile(r"\bPokémon V\b"), "Pokémon ex"),
]


def _trie_pattern(words: list[str]) -> str:
    """Compact alternation regex matching any of `words` (longest match wins by trie shape)."""
    trie: dict = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = {}

    def build(node: dict) -> str:
        if list(node) == [""]:
            return ""
        alternatives = []
        terminal = False
        for char, child in sorted(node.items()):
            if char == "":
                terminal = True
            else:
                alternatives.append(re.escape(char) + build(child))
        if len(alternatives) == 1:
            pattern = alternatives[0]
        else:
            pattern = "(?:" + "|".join(alternatives) + ")"
        if terminal:
            pattern = "(?:" + pattern + ")?"
        return pattern

    return build(trie)


@lru_cache(maxsize=1)
def _names_regex() -> re.Pattern:
    names = json.loads(NAMES.read_text(encoding="utf-8"))
    return re.compile(r"(?<!\w)(?:" + _trie_pattern(names) + r")(?!\w)")


# Damage counters are Pocket-absent bookkeeping for tens of damage: make them explicit
# (1 damage counter = 10 damage), so "remove 3 damage counters" reads as healing 30 and
# "put 5 damage counters" as doing 50. Ordered — numbered forms first, generic leftover last.
DAMAGE_COUNTERS = [
    (re.compile(r"\b(\d+) damage counters?\b"), lambda m: f"{int(m.group(1)) * 10} damage"),
    (re.compile(r"\ba damage counter\b"), "10 damage"),
    (re.compile(r"\bdamage counters?\b"), "damage"),
]
# Degenerate-board fallback clauses like "(1 if he or she has only 1)" carry no mechanic.
FALLBACK_CLAUSE = re.compile(r"\s*\(\d+ if [^)]*\)")
# Prize cards do not transfer to Pocket (points, not prizes): the whole sentence goes.
PRIZE_SENTENCE = re.compile(r"(?:^|(?<=[.!?]))\s*[^.!?]*[Pp]rize[^.!?]*[.!?]?")
# GX-attack / VSTAR-Power once-per-game reminders, equally Pocket-absent.
ONCE_PER_GAME_REMINDER = re.compile(r"\s*\(You can[’']t use more than 1[^)]*\)")
# Weakness/Resistance application reminders ("(Don't apply Weakness and Resistance for
# Benched Pokémon.)", "(after applying Weakness and Resistance)"): pure rulebook glue.
WEAKNESS_REMINDER = re.compile(r"\s*\([^)]*(?:Weakness|Resistance)[^)]*\)")
# Resistance itself does not exist in Pocket (only Weakness does): remaining sentences
# about it are dropped like Prize sentences.
RESISTANCE_SENTENCE = re.compile(r"(?:^|(?<=[.!?]))\s*[^.!?]*Resistance[^.!?]*[.!?]?")
# Corpus metadata artifacts: "{title}: " ability prefixes and "[2DD] Lost Crisis (100) "
# LEGEND-half prefixes.
TITLE_PREFIX = re.compile(r"^\{title\}:\s*")
LEGEND_PREFIX = re.compile(r"^\[\w+\]\s*[^()]*\(\d+\)\s*")
# Non-Latin scripts (Japanese duplicates inside English texts, delta symbols).
NON_LATIN = re.compile(r"[Ͱ-Ͽ　-ヿ一-鿿＀-￯]+")
# Era wording -> Pocket wording (Poké-Power/Poké-Body/Pokémon Power are today's Abilities;
# "his or her" is today's "their").
ERA_TERMS = [
    (re.compile(r"\bPoké-(?:Power|Body|BODY|POWER)s?\b"), "Ability"),
    (re.compile(r"\bPokémon Powers?\b"), "Ability"),
    (re.compile(r"\bhis or her\b"), "their"),
    (re.compile(r"\bhim or her\b"), "them"),
    (re.compile(r"\bhe or she\b"), "they"),
]


def normalize(text: str) -> str:
    """Normalize encoder input: energy symbols to words, Pokémon names to "Pokémon",
    non-Pocket special-rule markers to "ex", damage counters to explicit tens of damage,
    Prize-card sentences dropped.

    Lookup keys stay the *original* strings; only the encoder input is normalized. May return
    an empty string (text with no Pocket-transferable content) — callers must filter those.
    """
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    text = TITLE_PREFIX.sub("", text)
    text = LEGEND_PREFIX.sub("", text)
    text = NON_LATIN.sub(" ", text)
    for symbol, word in ENERGY_SYMBOLS.items():
        text = text.replace(symbol, word)
    text = WEAKNESS_REMINDER.sub("", text)
    text = PRIZE_SENTENCE.sub(" ", text)
    text = RESISTANCE_SENTENCE.sub(" ", text)
    text = ONCE_PER_GAME_REMINDER.sub("", text)
    text = FALLBACK_CLAUSE.sub("", text)
    for pattern, replacement in DAMAGE_COUNTERS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in ERA_TERMS:
        text = pattern.sub(replacement, text)
    text = _names_regex().sub("Pokémon", text)
    for pattern, replacement in MECHANIC_MARKERS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def load_pocket_texts() -> tuple[list[str], list[str]]:
    """(effect_texts, ability_texts) from deckgym's database.json, original spelling."""
    db = json.loads(DECKGYM_DATABASE.read_text(encoding="utf-8"))
    effects: set[str] = set()
    abilities: set[str] = set()
    for entry in db:
        for kind, card in entry.items():
            if kind == "Pokemon":
                if card.get("ability"):
                    abilities.add(card["ability"]["effect"])
                for attack in card.get("attacks", []):
                    if attack.get("effect"):
                        effects.add(attack["effect"])
            elif card.get("effect"):
                effects.add(card["effect"])
    return sorted(effects), sorted(abilities)


def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(model, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# A corpus text whose share of words unknown to the Pocket vocabulary exceeds this is
# rules language that does not transfer (Lost Zone, LEGEND, metadata garbage) and is
# excluded from the PCA fit. Chosen from vocab_report.py's distribution: above 0.3 sit
# only artifacts and mechanics Pocket cannot express.
OOV_MAX_RATIO = 0.3

_WORD = re.compile(r"[a-zA-Zé']+")


def vocabulary_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


@lru_cache(maxsize=1)
def pocket_vocabulary() -> frozenset[str]:
    """All words of the normalized Pocket texts (the transferable rules language)."""
    effect_texts, ability_texts = load_pocket_texts()
    vocab: set[str] = set()
    for text in effect_texts + ability_texts:
        vocab.update(vocabulary_words(normalize(text)))
    return frozenset(vocab)


def oov_ratio(normalized_text: str) -> float:
    """Share of words unknown to the Pocket vocabulary; 1.0 for wordless texts."""
    words = vocabulary_words(normalized_text)
    if not words:
        return 1.0
    vocab = pocket_vocabulary()
    return sum(1 for w in words if w not in vocab) / len(words)


def load_pca() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(components [128 x 384], mean [384], effect_scale [128], ability_scale [48])."""
    data = np.load(OUT_DIR / "pca.npz")
    return data["components"], data["mean"], data["effect_scale"], data["ability_scale"]


def whitening_scale(explained_variance: np.ndarray, dim: int) -> np.ndarray:
    """Per-component divisor for a `dim`-wide block: whitening, then one global constant.

    Unwhitened, a PCA block is unusable as a *linear* model's input even though the variance is
    there: PC0's standard deviation is ~15x PC127's, so at Kaiming init the leading components
    decide the projection's output and the tail is below the gradient noise floor. Dividing by
    each component's corpus standard deviation puts every rules-language direction on the same
    footing, which is the only form in which `src/rl/model/encoder.rs`'s single linear can weigh
    them on their merits rather than on their spectrum position.

    The global constant then sets the block's energy to ~1 per token, commensurate with one set
    bit of the one-hot blocks it is concatenated with. Raw, the block carried a squared norm of
    0.29 against ~7 for an Attack descriptor's damage thermometer — a channel that starts 25x
    below its neighbours has to climb through weights shared by every card before it can say
    anything.
    """
    scale = np.sqrt(explained_variance[:dim])
    # Measured on the corpus the basis was fitted on, so the constant is a property of the frozen
    # artifact rather than of whichever texts a caller happens to project.
    return (scale * np.sqrt(dim)).astype(np.float32)


def project(
    embeddings: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Project into the frozen basis and rescale. `scale` fixes the output width."""
    dim = len(scale)
    return ((embeddings - mean) @ components[:dim].T) / scale
