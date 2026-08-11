//! The factorized actor heads (§1.4.2) and the value head.
//!
//! Every head reads rows of the encoder output `H`:
//!
//! - **Pointer heads** — `logit_i = MLP(H[token_i])` over the head's own row map (the self-scoped
//!   slices of §1.3.8), emitted block-by-block in the exact [`HEADS`] wire order so the flat logit
//!   vector is bit-aligned with the Part-3 mask.
//! - **Nullary heads** (`action_type`, `END_TURN`, `USE_STADIUM`, `STATUS_CAT`) — `MLP(H[global])`.
//! - **`REVEALED_HAND_PTR`** — scored from `H[global]` for now: §1.3.6.2's belief-backed tokens
//!   are a Part-2 change that has not landed, so the encoder cannot yet see the revealed set.
//!   The block stays mask-aligned either way.
//! - **`CANDIDATE_PTR`** — §1.3.5: each candidate is `type_emb ⊕ pool(referenced-entity rows) ⊕
//!   H[global]`, scored by one shared MLP. Under `[model] candidate_cross_attention` a fourth term
//!   joins it, [`CandidateAttention`], letting the candidate weigh the board instead of reading it
//!   only through the mean over its own arguments.
//! - **Value** — the only pooling in the model: `MLP(AttnPool₁(H) ⊕ H[global]) ∈ [−1, 1]`.

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};

use crate::rl::action_mask::{
    ActionFamily, Head, ACTION_MASK_DIM, ACTION_TYPE_DIM, MAX_CANDIDATE_PTR, MAX_REVEALED_HAND_PTR,
    STATUS_CAT_DIM,
};
use crate::rl::damage::BOARD_SLOTS;
use crate::rl::history::HEAD_TABLE_SIZE;

use super::config::ModelConfig;
use super::embedding::LearnedEmbedding;
use super::input::ModelInput;

/// A two-layer scorer: `out(gelu(hidden(x)))`.
#[derive(Module, Debug)]
pub struct Scorer<B: Backend> {
    hidden: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> Scorer<B> {
    fn new(d_in: usize, d_hidden: usize, d_out: usize, device: &B::Device) -> Self {
        Self {
            hidden: LinearConfig::new(d_in, d_hidden).init(device),
            out: LinearConfig::new(d_hidden, d_out).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.out.forward(gelu(self.hidden.forward(x)))
    }
}

/// One candidate's look at the whole board, beside the mean over its own references.
///
/// **Why the mean is kept and not replaced.** A candidate *is* its references — they are the
/// action's arguments, and which entities an action touches is not recoverable from a query that
/// does not already carry them. So the mean identifies the candidate and the attention says what
/// else on the board bears on it. Replacing one with the other would drop the identity.
///
/// Single-headed, and `d_model` wide throughout: this scores at most `max_scored_candidates` rows
/// against a ~45-token sequence, so the expressiveness that matters is having a content-based
/// weighting at all, not having six of them.
#[derive(Module, Debug)]
pub struct CandidateAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
}

impl<B: Backend> CandidateAttention<B> {
    fn new(d_query: usize, d_model: usize, device: &B::Device) -> Self {
        Self {
            query: LinearConfig::new(d_query, d_model).init(device),
            key: LinearConfig::new(d_model, d_model).init(device),
            value: LinearConfig::new(d_model, d_model).init(device),
        }
    }

    /// `descriptor: [batch × scored × d_query]`, `h: [batch × n × d]`, `mask: [batch × n]` with
    /// `1.0` on real tokens. Returns `[batch × scored × d]`.
    fn forward(
        &self,
        descriptor: Tensor<B, 3>,
        h: &Tensor<B, 3>,
        mask: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch, slots, d_model] = h.dims();
        let scores = self
            .query
            .forward(descriptor)
            .matmul(self.key.forward(h.clone()).swap_dims(1, 2))
            .div_scalar((d_model as f64).sqrt());

        // Padded keys are excluded here rather than trusted to score low. Their rows are zeros, so
        // an unmasked softmax would spend real mass on them — and how much would depend on how full
        // the board happens to be, which is a property of the frame and not of the candidate.
        let padding = mask
            .clone()
            .equal_elem(0.0)
            .reshape([batch, 1, slots])
            .expand([batch, scores.dims()[1], slots]);
        softmax(scores.mask_fill(padding, f32::MIN), 2).matmul(self.value.forward(h.clone()))
    }
}

/// Gather `rows: [batch × k]` sequence positions out of `h: [batch × n × d]`.
fn gather_rows<B: Backend>(h: &Tensor<B, 3>, rows: &Tensor<B, 2, Int>) -> Tensor<B, 3> {
    let [batch, _, d_model] = h.dims();
    let [_, k] = rows.dims();
    let index = rows
        .clone()
        .unsqueeze_dim::<3>(2)
        .expand([batch, k, d_model]);
    h.clone().gather(1, index)
}

#[derive(Module, Debug)]
pub struct PolicyHeads<B: Backend> {
    action_type: Scorer<B>,
    place: Scorer<B>,
    evolve: Scorer<B>,
    attach_energy: Scorer<B>,
    retreat: Scorer<B>,
    attack: Scorer<B>,
    use_ability: Scorer<B>,
    play_trainer: Scorer<B>,
    use_stadium: Scorer<B>,
    end_turn: Scorer<B>,
    discard_fossil: Scorer<B>,
    slot_ptr_self: Scorer<B>,
    slot_ptr_opp: Scorer<B>,
    slot_pair: Scorer<B>,
    hand_ptr: Scorer<B>,
    status_cat: Scorer<B>,
    revealed_hand: Scorer<B>,
    candidate_type: LearnedEmbedding<B>,
    candidate: Scorer<B>,
    /// `None` unless `[model] candidate_cross_attention`. Absent rather than present-and-unused, so
    /// a model built without it has the parameter count and the record shape it had before.
    candidate_attention: Option<CandidateAttention<B>>,
}

impl<B: Backend> PolicyHeads<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let d = config.d_model;
        let hidden = config.d_head_hidden;
        let scorer = |d_in: usize, d_out: usize| Scorer::new(d_in, hidden, d_out, device);
        Self {
            action_type: scorer(d, ACTION_TYPE_DIM),
            place: scorer(d, BOARD_SLOTS),
            evolve: scorer(d, BOARD_SLOTS),
            attach_energy: scorer(d, 1),
            retreat: scorer(d, 1),
            attack: scorer(d, 1),
            use_ability: scorer(d, 1),
            play_trainer: scorer(d, 1),
            use_stadium: scorer(d, 1),
            end_turn: scorer(d, 1),
            discard_fossil: scorer(d, 1),
            slot_ptr_self: scorer(d, 1),
            slot_ptr_opp: scorer(d, 1),
            slot_pair: scorer(2 * d, 1),
            hand_ptr: scorer(d, 1),
            status_cat: scorer(d, STATUS_CAT_DIM),
            revealed_hand: scorer(d, MAX_REVEALED_HAND_PTR),
            candidate_type: LearnedEmbedding::new(HEAD_TABLE_SIZE, config.d_head_emb, device),
            // The attended context is a fourth term in the scorer's input when it is on, so the
            // width is read off the same flag rather than restated.
            candidate: scorer(
                config.d_head_emb
                    + if config.candidate_cross_attention {
                        3
                    } else {
                        2
                    } * d,
                1,
            ),
            candidate_attention: config
                .candidate_cross_attention
                // The query is what the candidate already knows about itself: its head type and
                // the mean over its references. Anything less and it cannot ask a question that
                // depends on which action it is.
                .then(|| CandidateAttention::new(config.d_head_emb + d, d, device)),
        }
    }

    /// The flat, unmasked logit vector `[batch × ACTION_MASK_DIM]`, blocks in [`HEADS`] order —
    /// bit-aligned with [`crate::rl::action_mask::ActionMaskWire`].
    pub fn forward(&self, h: &Tensor<B, 3>, input: &ModelInput<B>) -> Tensor<B, 2> {
        let [batch, _, d_model] = h.dims();
        let global = h.clone().slice([0..batch, 0..1]); // [batch × 1 × d]
        let over_global = |scorer: &Scorer<B>, width: usize| -> Tensor<B, 2> {
            scorer.forward(global.clone()).reshape([batch, width])
        };
        let over_rows =
            |scorer: &Scorer<B>, rows: &Tensor<B, 2, Int>, width: usize| -> Tensor<B, 2> {
                scorer.forward(gather_rows(h, rows)).reshape([batch, width])
            };

        let board_self = &input.board_self_rows;
        let bench_rows = board_self.clone().slice([0..batch, 1..BOARD_SLOTS]);

        // SLOT_PAIR: all 16 (from, to) pairs of the self board.
        let board = gather_rows(h, board_self); // [batch × 4 × d]
        let from = board
            .clone()
            .unsqueeze_dim::<4>(2)
            .expand([batch, BOARD_SLOTS, BOARD_SLOTS, d_model])
            .reshape([batch, BOARD_SLOTS * BOARD_SLOTS, d_model]);
        let to = board
            .unsqueeze_dim::<4>(1)
            .expand([batch, BOARD_SLOTS, BOARD_SLOTS, d_model])
            .reshape([batch, BOARD_SLOTS * BOARD_SLOTS, d_model]);
        let pair_logits = self
            .slot_pair
            .forward(Tensor::cat(vec![from, to], 2))
            .reshape([batch, BOARD_SLOTS * BOARD_SLOTS]);

        // CANDIDATE_PTR: type_emb ⊕ pool(referenced rows) ⊕ H[global], shared scorer.
        let [_, scored, refs] = input.candidate_ref_rows.dims();
        let type_emb = self.candidate_type.embed(input.candidate_type_ids.clone());
        let ref_rows = gather_rows(
            h,
            &input
                .candidate_ref_rows
                .clone()
                .reshape([batch, scored * refs]),
        )
        .reshape([batch, scored, refs, d_model]);
        let ref_mask = input.candidate_ref_mask.clone().unsqueeze_dim::<4>(3);
        let pooled = (ref_rows * ref_mask.clone()).sum_dim(2).squeeze_dims(&[2])
            / input.candidate_ref_mask.clone().sum_dim(2).clamp_min(1.0);
        let global_broadcast = global.clone().expand([batch, scored, d_model]);
        let mut terms = vec![type_emb, pooled];
        if let Some(attention) = &self.candidate_attention {
            let descriptor = Tensor::cat(terms.clone(), 2);
            terms.push(attention.forward(descriptor, h, &input.seq_mask));
        }
        terms.push(global_broadcast);
        let candidate_logits = self
            .candidate
            .forward(Tensor::cat(terms, 2))
            .reshape([batch, scored]);
        let candidate_block = if scored < MAX_CANDIDATE_PTR {
            Tensor::cat(
                vec![
                    candidate_logits,
                    Tensor::zeros([batch, MAX_CANDIDATE_PTR - scored], &h.device()),
                ],
                1,
            )
        } else {
            candidate_logits
        };

        // Emit in the exact wire order of `HEADS`.
        let blocks: Vec<Tensor<B, 2>> = vec![
            over_global(&self.action_type, ACTION_TYPE_DIM),
            over_rows(&self.place, &input.self_pokemon_rows, Head::Place.dim()),
            over_rows(&self.evolve, &input.self_pokemon_rows, Head::Evolve.dim()),
            over_rows(&self.attach_energy, board_self, BOARD_SLOTS),
            over_rows(&self.retreat, &bench_rows, BOARD_SLOTS - 1),
            over_rows(&self.attack, &input.self_attack_rows, Head::Attack.dim()),
            over_rows(&self.use_ability, board_self, BOARD_SLOTS),
            over_rows(
                &self.play_trainer,
                &input.self_trainer_rows,
                Head::PlayTrainer.dim(),
            ),
            over_global(&self.use_stadium, 1),
            over_global(&self.end_turn, 1),
            over_rows(&self.discard_fossil, board_self, BOARD_SLOTS),
            over_rows(&self.slot_ptr_self, board_self, BOARD_SLOTS),
            over_rows(&self.slot_ptr_opp, &input.board_opp_rows, BOARD_SLOTS),
            pair_logits,
            over_rows(
                &self.hand_ptr,
                &input.self_pokemon_rows,
                Head::HandPtr.dim(),
            ),
            over_global(&self.status_cat, STATUS_CAT_DIM),
            over_global(&self.revealed_hand, MAX_REVEALED_HAND_PTR),
            candidate_block,
        ];
        let flat = Tensor::cat(blocks, 1);
        debug_assert_eq!(flat.dims(), [batch, ACTION_MASK_DIM]);
        flat
    }
}

/// The wire-layout constants that let [`masked_policy`] run as whole-tensor ops instead of a
/// per-family loop of `slice` / `slice_assign` — the loops cost ~70 kernel launches on a
/// `[batch × 804]` tensor, which dominates small-batch GPU latency. Both are pure functions of the
/// Part-3 head layout, built once per model, and hold as constants (never parameters).
#[derive(Module, Debug)]
pub struct MaskLayout<B: Backend> {
    /// `[1 × ACTION_MASK_DIM]`: the family index each bit belongs to, or the sentinel
    /// `ACTION_TYPE_DIM` for bits under no family (the `ACTION_TYPE` block, stack-only heads).
    family_of_bit: Tensor<B, 2, Int>,
    /// `[ACTION_MASK_DIM × ACTION_TYPE_DIM]`, 1.0 where a bit belongs to a family's argument
    /// block — one matmul then replaces the marginals loop.
    family_onehot: Tensor<B, 2>,
}

impl<B: Backend> MaskLayout<B> {
    pub fn new(device: &B::Device) -> Self {
        let mut of_bit = vec![ACTION_TYPE_DIM as i64; ACTION_MASK_DIM];
        let mut onehot = vec![0.0f32; ACTION_MASK_DIM * ACTION_TYPE_DIM];
        for family in ActionFamily::ALL {
            let head = family.head();
            for bit in head.offset()..head.offset() + head.dim() {
                // Each family owns a distinct argument head, so no bit is claimed twice.
                debug_assert_eq!(of_bit[bit], ACTION_TYPE_DIM as i64);
                of_bit[bit] = family.index() as i64;
                onehot[bit * ACTION_TYPE_DIM + family.index()] = 1.0;
            }
        }
        Self {
            family_of_bit: Tensor::from_data(
                burn::tensor::TensorData::new(of_bit, [1, ACTION_MASK_DIM]),
                device,
            ),
            family_onehot: Tensor::from_data(
                burn::tensor::TensorData::new(onehot, [ACTION_MASK_DIM, ACTION_TYPE_DIM]),
                device,
            ),
        }
    }
}

/// The masked policy over the flat logit vector.
///
/// Joint logit of an argument bit = its head logit, **plus its family's `action_type` logit** for
/// the ten free-play families (§1.3.4: family choice then argument choice; stack-only heads have
/// no family term). Probabilities are an exact masked softmax over the set argument bits: unset
/// bits come out *exactly* 0, set bits sum to 1. The `ACTION_TYPE` block of the returned vector
/// carries the induced family marginals rather than raw probabilities of its own.
///
/// The family term is broadcast with one `gather` through [`MaskLayout::family_of_bit`] (the
/// sentinel row is a zero column, so bits under no family are left untouched), and the marginals
/// come back with one `matmul` against [`MaskLayout::family_onehot`].
pub fn masked_policy<B: Backend>(
    layout: &MaskLayout<B>,
    logits: Tensor<B, 2>,
    mask_bits: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch, dim] = logits.dims();
    debug_assert_eq!(dim, ACTION_MASK_DIM);
    let type_offset = Head::ActionType.offset();
    let device = logits.device();

    // Add each family's action_type logit onto its argument block; the sentinel indexes a zero
    // column, so the ACTION_TYPE block and the stack-only heads keep their raw logits.
    let family_logits = logits
        .clone()
        .slice([0..batch, type_offset..type_offset + ACTION_TYPE_DIM]);
    let family_padded = Tensor::cat(vec![family_logits, Tensor::zeros([batch, 1], &device)], 1);
    let joint = logits
        + family_padded.gather(
            1,
            layout
                .family_of_bit
                .clone()
                .expand([batch, ACTION_MASK_DIM]),
        );

    // Softmax over the set *argument* bits only.
    let argument_mask = mask_bits.slice_assign(
        [0..batch, type_offset..type_offset + ACTION_TYPE_DIM],
        Tensor::zeros([batch, ACTION_TYPE_DIM], &device),
    );
    let shifted = joint.clone() + (argument_mask.clone() - 1.0) * 1.0e9;
    let max = shifted.max_dim(1);
    let exp = (joint - max).exp() * argument_mask;
    let probs = exp.clone() / exp.sum_dim(1).clamp_min(1.0e-30);

    // Family marginals into the ACTION_TYPE block.
    let marginals = probs.clone().matmul(layout.family_onehot.clone());
    probs.slice_assign(
        [0..batch, type_offset..type_offset + ACTION_TYPE_DIM],
        marginals,
    )
}

/// The value head (§1.4.2): `v = MLP(AttnPool₁(H) ⊕ H[global]) ∈ [−1, 1]`, the model's only
/// pooling, value-only.
#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    /// The single learned attention query.
    query: Param<Tensor<B, 1>>,
    hidden: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            query: Param::from_tensor(Tensor::random(
                [config.d_model],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            )),
            hidden: LinearConfig::new(2 * config.d_model, config.d_head_hidden).init(device),
            out: LinearConfig::new(config.d_head_hidden, 1).init(device),
        }
    }

    /// `h: [batch × n × d]`, `seq_mask: [batch × n]` → `[batch]` in `[−1, 1]`.
    pub fn forward(&self, h: &Tensor<B, 3>, seq_mask: &Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch, n, d_model] = h.dims();
        let query = self.query.val().reshape([1, 1, d_model]);
        let scores = (h.clone() * query).sum_dim(2).reshape([batch, n]) / (d_model as f64).sqrt()
            + (seq_mask.clone() - 1.0) * 1.0e9;
        let weights = softmax(scores, 1).reshape([batch, n, 1]);
        let pooled = (h.clone() * weights).sum_dim(1).reshape([batch, d_model]);
        let global = h.clone().slice([0..batch, 0..1]).reshape([batch, d_model]);
        let joined = Tensor::cat(vec![pooled, global], 1).unsqueeze_dim::<3>(1);
        self.out
            .forward(gelu(self.hidden.forward(joined)))
            .reshape([batch])
            .tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Module;

    /// A padded key must not enter the candidate's context. Its row is zeros, so an unmasked
    /// softmax would spend real mass on it and the amount would track how full the board is — a
    /// property of the frame, not of the candidate being scored. Checked by making the padded row
    /// enormous: if it were read at all, nothing else would survive the average.
    #[test]
    fn a_padded_token_does_not_enter_the_candidate_context() {
        let device = Default::default();
        let attention = CandidateAttention::<NdArray>::new(4, 2, &device);
        let descriptor = Tensor::<NdArray, 3>::from_floats([[[1.0f32, -1.0, 0.5, 0.0]]], &device);
        let mask = Tensor::<NdArray, 2>::from_floats([[1.0, 1.0, 0.0]], &device);

        let real =
            Tensor::<NdArray, 3>::from_floats([[[1.0f32, 0.0], [0.0, 1.0], [0.0, 0.0]]], &device);
        let shouting = Tensor::<NdArray, 3>::from_floats(
            [[[1.0f32, 0.0], [0.0, 1.0], [900.0, -900.0]]],
            &device,
        );

        let quiet = attention.forward(descriptor.clone(), &real, &mask);
        let loud = attention.forward(descriptor, &shouting, &mask);
        let gap = (quiet - loud).abs().max().into_scalar();
        assert!(
            gap < 1.0e-4,
            "the padded row moved the context by {gap}, so it is being attended to"
        );
    }

    /// The flag is what keeps this a one-variable change, so what it costs has to be exactly the
    /// cross-attention and the scorer input it widens — not a parameter more, which is what a
    /// stray always-on tensor would show up as.
    #[test]
    fn the_flag_adds_exactly_the_cross_attention_parameters() {
        let device = Default::default();
        let off = ModelConfig::default();
        let mut on = off.clone();
        on.candidate_cross_attention = true;

        let (d, hidden) = (off.d_model, off.d_head_hidden);
        let projections = (off.d_head_emb + d) * d + d + 2 * (d * d + d);
        let widened_scorer = d * hidden;

        let count =
            |config: &ModelConfig| PolicyHeads::<NdArray>::new(config, &device).num_params();
        assert_eq!(
            count(&on) - count(&off),
            projections + widened_scorer,
            "the flag moved something other than the candidate path"
        );
    }
}

#[cfg(test)]
mod flag_record_tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    /// **Characterization test: this asserts a hole, not a guarantee.** A record written without
    /// the flag loads into a model built with it and reports success, even though the candidate
    /// scorer's input width differs (400 against 592) and the cross-attention is not in the record
    /// at all. Burn's recorder validates the length of a `Vec` of layers — which is what catches a
    /// `num_blocks` mismatch — and not the shape of a tensor or the presence of an `Option` module.
    ///
    /// So the loaded model keeps its freshly initialized weights wherever the record has nothing to
    /// say, silently. The exposure is a resume across a config edit; baked models are safe because
    /// `BakedMeta` carries the `[model]` table the weights were trained at and rebuilds from it.
    ///
    /// When this test starts failing, someone has added the shape check. Delete it and assert the
    /// error instead. See TODO.md.
    #[test]
    fn a_record_without_the_flag_loads_into_a_model_with_it_silently() {
        let device = Default::default();
        let off = ModelConfig::default();
        let mut on = off.clone();
        on.candidate_cross_attention = true;

        let dir = std::env::temp_dir().join("deckgym_flag_record");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("heads");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        PolicyHeads::<NdArray>::new(&off, &device)
            .save_file(path.clone(), &recorder)
            .expect("save");

        let loaded = PolicyHeads::<NdArray>::new(&on, &device).load_file(path, &recorder, &device);
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            loaded.is_ok(),
            "the shape mismatch is now caught — update this test to assert the error"
        );
    }
}
