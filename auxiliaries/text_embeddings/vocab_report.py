"""Vocabulary diagnostic between the Pocket sub-set and the super-set corpus.

Two directions:
  1. sub-set -> corpus: every (normalized) Pocket word should appear in the normalized
     corpus vocabulary — a miss means the PCA basis never saw a word that matters;
  2. corpus -> sub-set: per-text ratio of words unknown to the Pocket vocabulary — a
     high ratio flags rules language that does not transfer (Lost Zone, LEGEND, ...).
     Prints the distribution and the worst offenders so the filter threshold in
     build_embeddings.py stays an informed choice, not a guess.
"""

from __future__ import annotations

import json
from collections import Counter

import common
from common import vocabulary_words as words


def main() -> None:
    corpus = json.loads(common.CORPUS.read_text(encoding="utf-8"))
    corpus_norm = sorted({t for t in (common.normalize(t) for t in corpus) if t})

    pocket_vocab = common.pocket_vocabulary()
    corpus_vocab: Counter = Counter()
    for text in corpus_norm:
        corpus_vocab.update(words(text))

    print(f"pocket vocab: {len(pocket_vocab)} words | corpus vocab: {len(corpus_vocab)} words")

    missing = sorted(pocket_vocab - set(corpus_vocab))
    print(f"\n[1] pocket words ABSENT from corpus ({len(missing)}):")
    print("   ", ", ".join(missing) if missing else "(none)")

    print("\n[2] corpus OOV-ratio distribution (share of words unknown to Pocket):")
    ratios = []
    for text in corpus_norm:
        toks = words(text)
        oov = [w for w in toks if w not in pocket_vocab]
        ratios.append((len(oov) / len(toks) if toks else 1.0, text, oov))
    ratios.sort(key=lambda r: r[0])
    for bound in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        n = sum(1 for r, _, _ in ratios if r > bound)
        print(f"    ratio > {bound:.1f}: {n:5d} texts ({100 * n / len(ratios):.1f}%)")

    print("\n    worst offenders:")
    for ratio, text, oov in ratios[-8:]:
        print(f"    {ratio:.2f} {text[:100]!r} oov={oov[:8]}")

    print("\n    most frequent OOV words (candidate normalization gaps or ban list):")
    oov_freq: Counter = Counter()
    for _, _, oov in ratios:
        oov_freq.update(oov)
    for word, count in oov_freq.most_common(40):
        print(f"    {count:5d} {word}")


if __name__ == "__main__":
    main()
