"""Build the frozen text-embedding table consumed by src/rl/text_embedding.rs.

Pipeline (RL_ARCHITECTURE §1.2.9):
  1. embed the super-set corpus (corpus.json, from extract_corpus.py) with bge-small-en-v1.5;
  2. fit a 128-component PCA on those embeddings — the basis captures the variance of the
     whole TCG rules language, not just the Pocket subset;
  3. embed the Pocket texts (deckgym database.json), project them:
       - effect texts (attacks + trainers) -> all 128 components,
       - ability texts -> the first 48 components (PCA components are nested);
  4. write out/text_embeddings.json keyed by the ORIGINAL Pocket strings, plus the frozen
     PCA basis (out/pca.npz) and out/meta.json.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.decomposition import PCA

import common


def main() -> None:
    corpus = json.loads(common.CORPUS.read_text(encoding="utf-8"))
    corpus_norm = [common.normalize(t) for t in corpus]
    # A normalization that leaves nothing (e.g. a pure Prize-card effect) has no
    # Pocket-transferable content and must not weigh on the PCA basis; nor should texts
    # dominated by words Pocket's rules language does not know (see vocab_report.py).
    corpus_norm = sorted(
        {t for t in corpus_norm if t and common.oov_ratio(t) <= common.OOV_MAX_RATIO}
    )
    effect_texts, ability_texts = common.load_pocket_texts()
    print(
        f"corpus: {len(corpus)} texts ({len(corpus_norm)} after normalization) | "
        f"pocket: {len(effect_texts)} effects, {len(ability_texts)} abilities"
    )

    model = common.load_encoder()
    corpus_emb = common.embed(model, corpus_norm)

    pca = PCA(n_components=common.EFFECT_DIM, random_state=0)
    pca.fit(corpus_emb)
    evr = pca.explained_variance_ratio_
    print(
        f"PCA explained variance: {evr[:common.ABILITY_DIM].sum():.3f} @48, "
        f"{evr.sum():.3f} @128"
    )

    # The two blocks are scaled independently: they are concatenated into different descriptors
    # and each has to weigh ~1 against the bits beside it, which a shared constant cannot give
    # two widths at once. The components stay nested — the scale is per-component.
    effect_scale = common.whitening_scale(pca.explained_variance_, common.EFFECT_DIM)
    ability_scale = common.whitening_scale(pca.explained_variance_, common.ABILITY_DIM)

    effect_emb = common.embed(model, [common.normalize(t) for t in effect_texts])
    ability_emb = common.embed(model, [common.normalize(t) for t in ability_texts])
    effect_proj = common.project(effect_emb, pca.components_, pca.mean_, effect_scale)
    ability_proj = common.project(ability_emb, pca.components_, pca.mean_, ability_scale)
    print(
        f"block energy (mean squared norm): effect {np.mean((effect_proj**2).sum(1)):.3f}, "
        f"ability {np.mean((ability_proj**2).sum(1)):.3f}"
    )

    common.OUT_DIR.mkdir(exist_ok=True)
    table = {
        "effect": {
            text: [round(float(v), 6) for v in row]
            for text, row in zip(effect_texts, effect_proj)
        },
        "ability": {
            text: [round(float(v), 6) for v in row]
            for text, row in zip(ability_texts, ability_proj)
        },
    }
    (common.OUT_DIR / "text_embeddings.json").write_text(
        json.dumps(table, ensure_ascii=False), encoding="utf-8"
    )
    np.savez(
        common.OUT_DIR / "pca.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        # Part of the frozen basis, not a preprocessing choice: a text embedded later without
        # these lands in a different space than every text in the table.
        effect_scale=effect_scale,
        ability_scale=ability_scale,
    )

    import sentence_transformers, sklearn

    meta = {
        "model": common.MODEL_NAME,
        "sentence_transformers": sentence_transformers.__version__,
        "sklearn": sklearn.__version__,
        "corpus_size": len(corpus),
        "corpus_size_normalized": len(corpus_norm),
        "pocket_effect_texts": len(effect_texts),
        "pocket_ability_texts": len(ability_texts),
        "effect_dim": common.EFFECT_DIM,
        "ability_dim": common.ABILITY_DIM,
        "explained_variance_at_48": float(evr[: common.ABILITY_DIM].sum()),
        "explained_variance_at_128": float(evr.sum()),
        "whitened": True,
        "effect_block_energy": float(np.mean((effect_proj**2).sum(1))),
        "ability_block_energy": float(np.mean((ability_proj**2).sum(1))),
    }
    (common.OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"wrote {common.OUT_DIR / 'text_embeddings.json'}")


if __name__ == "__main__":
    main()
