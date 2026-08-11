# text_embeddings — the frozen "super-set TCG" text encoder

Implements the v1 text encoder of `RL_ARCHITECTURE.md` §1.2.9: a frozen, meta-neutral
embedding of every Pocket effect/ability text, consumed by `src/rl/text_embedding.rs`
(`TextEmbeddings::from_json_file`). Meta-neutral: the pipeline sees card texts only —
never winrates, never co-occurrence.

## Design

- **Encoder**: `BAAI/bge-small-en-v1.5` (frozen sentence-transformer, 384-dim, L2-normalized).
- **Super-set corpus**: all English effect texts (attacks, abilities, trainers) of the whole
  TCG, extracted from the sibling [`cards-database`](https://github.com/tcgdex/cards-database)
  repo (expected at `../../../cards-database`) — ~12k unique texts.
- **Compression**: PCA fitted on the super-set corpus (the basis captures the variance of the
  full TCG rules language, not just the Pocket subset). Effect texts keep all 128 components
  (`EFFECT_TEXT_DIM`); ability texts keep the first 48 (`ABILITY_TEXT_DIM` — PCA components
  are nested). Explained variance: ~0.95 @128, ~0.82 @48.
- **Whitening** (`common.whitening_scale`, part of the frozen basis): each component is divided by
  its corpus standard deviation, then the block by `sqrt(dim)`. Unwhitened the table was measured
  at a participation-ratio effective rank of 24 out of 128 — PC0's std was 15x PC127's, the 64
  weakest dims held 7 % of the variance, and the whole block carried a squared norm of 0.29 against
  ~7 for the damage thermometer it is concatenated with in an Attack descriptor. Both properties
  are fatal to the single `width -> d_model` linear that consumes it: the leading components decide
  the output and the tail sits under the gradient noise floor. After whitening the effective rank
  is 124/128 and the block's energy ~1, i.e. worth about one set bit. Whitening changes cosines, so
  it changes the geometry `probes.py` gates — that is the point, and the two blocks carry separate
  scales because they are concatenated into descriptors of different widths.
- **Input normalization** (encoder input only — lookup keys stay the original strings):
  - energy symbols, Pocket's `[X]` and the corpus's `{X}` (plus `{N}` Dragon, `{ex}`, `{*}`)
    → plain words;
  - every Pokémon card name (super-set + Pocket, longest match first) → `Pokémon`, so no
    variance is spent on name lexemes;
  - special-rule markers that don't exist in Pocket (`V`, `VMAX`, `VSTAR`, `V-UNION`, `GX`,
    `EX`, `LV.X`) → `ex`, the one special-rule role Pocket has (`Dark`/`Light`/… vanish with
    the full names);
  - damage counters made explicit as tens of damage (`3 damage counters` → `30 damage`, so
    removing reads as healing and placing as inflicting);
  - era wording mapped to Pocket wording: `Poké-Power`/`Poké-Body`/`Pokémon Power` →
    `Ability`, `his or her` → `their`; curly quotes straightened;
  - Pocket-untransferable material dropped: any sentence mentioning Prize cards or
    Resistance, Weakness/Resistance application reminders, once-per-game GX/VSTAR
    reminders, degenerate-board fallbacks like `(1 if there is only 1)`; `{title}:` and
    LEGEND-half metadata prefixes stripped, non-Latin scripts (embedded Japanese, δ)
    removed; whitespace (embedded newlines included) collapsed.
- **Corpus filter**: after normalization, texts that are empty or whose share of words
  unknown to the Pocket vocabulary exceeds `OOV_MAX_RATIO = 0.3` are excluded from the PCA
  fit (~12k → ~10.2k texts) — above 0.3 sit only metadata garbage and mechanics Pocket
  cannot express (Lost Zone, LEGEND, Rule Boxes). `vocab_report.py` prints both directions
  (Pocket words missing from the corpus; corpus OOV distribution + worst offenders) to keep
  the threshold and the normalization rules honest.

## Pipeline

```
uv sync
uv run python extract_corpus.py    # cards-database + database.json -> corpus.json, names.json
uv run python build_embeddings.py  # embed, fit PCA, export out/
uv run python probes.py            # sanity gate (exit 1 on unexpected failure)
uv run python vocab_report.py      # two-way vocabulary diagnostic (informs OOV_MAX_RATIO)
```

## Outputs (`out/`, committed — this is the freeze)

- `text_embeddings.json` — `{"effect": {text: [f32;128]}, "ability": {text: [f32;48]}}`, keyed
  by the exact strings of `database.json`. Loaded by `TextEmbeddings::from_json_file`; the Rust
  test `frozen_artifact_covers_every_pool_text` asserts every pool text resolves non-zero.
- `pca.npz` — frozen PCA basis (components, mean, explained variance, `effect_scale`,
  `ability_scale`), needed to embed any new text into the same space. The scales are not optional:
  a text projected without them lands in a different space than every text in the table.
- `meta.json` — model + library versions, corpus/table sizes, explained variance.

## Probes (the falsifiable gate)

`probes.py` checks ~17 (anchor, near, far) triplets on load-bearing mechanics (coin flips,
bench snipe, energy accel vs denial, status, locks, normalization identities) in the raw-384,
pca-128 and pca-48 spaces. A probe passing raw but failing projected indicts the compression.
Known `xfail` — "self damage recoil": surface overlap outweighs the logical role of "to itself";
that numeric/logical blind spot is assigned by §1.2.9 to the future structured attack schema, not
to a better sentence encoder.

**Whitening moved two of these and the gate is currently red on one.** "self damage recoil" now
*passes* at pca-128 (+0.424 near / +0.194 far, against +0.757 / +0.847 raw) — dropping the leading
components' dominance is what let the logical role outweigh the surface overlap. "hand disruption"
fails at pca-128 (+0.054 near / +0.070 far) while passing raw and pca-48. Its "near" pair —
*discard a random card from your opponent's hand* against *your opponent reveals their hand* —
shares a noun and not a mechanic, and whitening is precisely what stops a shared noun from carrying
a cosine. Both readings are within 0.02 of orthogonal, so the ordering is noise rather than an
inversion. Left failing rather than moved to `EXPECTED_FAILURES`: the honest fix is a probe whose
"near" is a real mechanical neighbour, and suppressing the gate to pass a change is how a gate
stops meaning anything.
