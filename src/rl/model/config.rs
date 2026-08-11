//! Model hyperparameters (§1.4.3) — v1 defaults, `.toml`-tunable.
//!
//! The struct derives `serde::Deserialize` so a Part-5 run configuration can carry a
//! `[model]` table; every field has the §1.4.3 default. It derives `Serialize` too, because a
//! §1.5.2 baked model has to *record* the sizes it was trained at — a stored model whose shape is
//! guessed at load time is a stored model that loads into the wrong network.

use serde::{Deserialize, Serialize};

/// Sizes of the Part 4 model. §1.4.3 default: `d_model = 192`, 2 blocks, 6 heads (32/head),
/// FFN = 384, `d_id = 64`. Sizing derivation: NOTES.md, "Redimensionner v1 à la baisse".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Encoder width. Must be ≥ 3·`d_id`: the Pokémon token carries `3 × d_id` of identity —
    /// at `d_id = 64` the default 192 sits exactly on that floor.
    pub d_model: usize,
    /// Pre-LN transformer blocks.
    pub num_blocks: usize,
    /// Attention heads per block.
    pub num_heads: usize,
    /// FFN inner width.
    pub d_ff: usize,
    /// Identity-embedding width, shared by the three ID spaces (§1.2.2).
    pub d_id: usize,
    /// History `head_id` embedding width (§1.2.7).
    pub d_head_emb: usize,
    /// Hidden width of the pointer-head / value MLPs.
    pub d_head_hidden: usize,
    /// Seed of the deterministic meta-neutral init projections (§1.2.2). Changing it changes
    /// the frozen embedding init — it is part of the model identity, not a run seed.
    pub init_seed: u64,
    /// Compute cap on scored `CANDIDATE_PTR` positions. The wire block stays
    /// [`super::super::action_mask::MAX_CANDIDATE_PTR`] wide, but real frames stay two orders of
    /// magnitude below it (§1.3.8 — widest observed: 20), so only this prefix is ever scored.
    /// Input assembly asserts no candidate lands beyond it.
    pub max_scored_candidates: usize,
    /// Cap on pooled entity references per candidate (§1.3.5 `pool(referenced-entity
    /// embeddings)`). Extra references are truncated — an encoding approximation, never a
    /// legality one.
    pub max_candidate_refs: usize,
    /// Let each `CANDIDATE_PTR` candidate attend over the whole encoded sequence, beside the mean
    /// over its own references (§1.3.5).
    ///
    /// Off by default, and that is what keeps it a one-variable change: it adds parameters, so a
    /// model built with it on cannot read a checkpoint written with it off. `long_v5` measured the
    /// reason it exists — the encoder spends 5.8–6.4× chance on the Attack family while the scorer
    /// that decides between actions sees only an unweighted mean of the rows each one references.
    #[serde(default)]
    pub candidate_cross_attention: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 192,
            num_blocks: 2,
            num_heads: 6,
            d_ff: 384,
            d_id: 64,
            d_head_emb: 16,
            d_head_hidden: 64,
            init_seed: 0x00DE_C4C4_0000_0001,
            max_scored_candidates: 64,
            max_candidate_refs: 8,
            candidate_cross_attention: false,
        }
    }
}

impl ModelConfig {
    /// Head width of one attention head.
    pub fn d_head(&self) -> usize {
        debug_assert_eq!(self.d_model % self.num_heads, 0);
        self.d_model / self.num_heads
    }
}
