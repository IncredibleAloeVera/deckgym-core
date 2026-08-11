//! Attention read-out: what each head of each block looks at.
//!
//! Two numbers per head, both folded over the real query rows of a batch:
//!
//! - **entropy** of the key distribution, in nats. A head pinned on one token reads near `0`;
//!   `ln(SEQ_LEN) ≈ 4.89` is uniform. Falling entropy across a run is a head specializing, and a
//!   head that reaches `0` and stays there is one that has stopped discriminating.
//! - **mass by target family** — the share of attention landing on Global / Pokémon / Attack /
//!   Trainer / History, beside [`AttentionStats::family_share`], the share of the batch's real
//!   tokens each family holds. This is the reading the token layout buys us: the families are
//!   named, so "block 1, head 3 spends 80 % of its attention on Attack tokens" is available
//!   directly, where an interpretability pass over text would have to earn the same statement.
//!   The two travel together because a mass alone cannot distinguish a preference from an abundant
//!   family — their ratio can.
//!
//! **Why there is a pairwise number too.** Both readings above are per head, and per head they
//! cannot separate two heads reading the same tokens from two splitting a family between them:
//! either way the series show two equal masses on one family. Redundancy is a property of a
//! *pair*, so it takes a pair's measurement — [`HeadPair::divergence`]. Only within a block: a
//! block's heads are concatenated into one projection and are permutation-symmetric, which is what
//! makes "these two are the same head" a statement about wasted capacity. Across blocks the two
//! distributions are read at different depths, and their divergence measures nothing.
//!
//! **Why the Pokémon and Trainer families are also split by zone.** A family's share is the
//! baseline its mass is read against, and for those two the share counts tokens that carry no
//! information at most frames: Trainer tokens are emitted for the hand *and* the deck *and* both
//! discards, so ~11 of a batch's ~14 Trainer tokens are cards nobody can play. Measured on
//! `long_v3` and `long_v4`, all 48 head readings sat below chance on the Trainer family in both —
//! including the run that had no text features at all, which falsifies the reading that the text
//! channel was the cause. A head spreading its mass over the *relevant* tokens alone lands near
//! `0.5` on that baseline, so the aggregate cannot separate "ignores Trainers" from "ignores
//! Trainers in the deck". The zone split makes the two distinguishable; the aggregate families stay
//! beside it, both because they are what earlier runs logged and because the zoned buckets refine
//! them rather than replace them.
//!
//! **Why the size of each block's write is measured beside its attention.** An attention pattern
//! only matters in proportion to what the block does with it, and a near-uniform head has two
//! readings the entropy cannot separate: a block whose write is small against the residual stream is
//! near-identity and its pattern is irrelevant, while a block writing hard through a near-uniform
//! pattern is pooling the sequence into every token — which is what lets the *next* block's queries
//! carry context and be selective. The first is wasted capacity; the second is the work that makes
//! the rest possible. [`BlockWrite`] is what tells them apart.
//!
//! **A caveat that decides how the mass series may be read.** The blocks run a plain softmax
//! (burn's `quiet_softmax` default), so every query row spends a total mass of exactly 1 whether or
//! not the head has anything to say about that row. A head with nothing to contribute therefore
//! *cannot* abstain; it deposits its mass somewhere, and in practice that somewhere is a fixed
//! token — the attention-sink effect. Mass concentrated on the Global token is the shape this
//! takes here, since row 0 is the one token that is never padded. Read such a concentration as a
//! head with no signal, not as a head that has learned to consult the global features.

use burn::prelude::*;

use super::encoder::{AttentionAblation, BlockDrift, TOKEN_TYPES};
use super::input::{
    ModelInput, ATTACK_OFFSET, GLOBAL_ROW, HISTORY_OFFSET, POKEMON_OFFSET, SEQ_LEN, TRAINER_OFFSET,
};
use super::RlModel;
use crate::rl::observation::{TokenZone, ZONE_FEATURE_OFFSET};

/// Series names of the five families, in sequence order.
pub const FAMILY_NAMES: [&str; TOKEN_TYPES] = ["global", "pokemon", "attack", "trainer", "history"];

/// Where each family sits in the sequence. Derived from the [`super::input`] offsets rather than
/// restated, so a layout change moves the buckets with it instead of silently mislabelling them.
const FAMILY_RANGES: [(usize, usize); TOKEN_TYPES] = [
    (GLOBAL_ROW, POKEMON_OFFSET),
    (POKEMON_OFFSET, ATTACK_OFFSET),
    (ATTACK_OFFSET, TRAINER_OFFSET),
    (TRAINER_OFFSET, HISTORY_OFFSET),
    (HISTORY_OFFSET, SEQ_LEN),
];

/// The two families whose tokens span [`TokenZone`]s, in the order [`ZONED_BUCKETS`] enumerates.
///
/// Attack tokens are board-only, History is not a card entity, and Global is one row — those three
/// have no zone to split by. The names are the family names, so a zoned series key reads as a
/// refinement of the family it came from.
pub const ZONED_FAMILIES: [&str; 2] = [FAMILY_NAMES[1], FAMILY_NAMES[3]];

/// `(family, zone)` buckets, `ZONED_FAMILIES × TokenZone::DIM`.
pub const ZONED_BUCKETS: usize = ZONED_FAMILIES.len() * TokenZone::DIM;

/// Family index into [`FAMILY_RANGES`] for each entry of [`ZONED_FAMILIES`].
const ZONED_FAMILY_INDEX: [usize; ZONED_FAMILIES.len()] = [1, 3];

/// `(family, zone)` name of bucket `bucket`, as it appears in the series key.
pub fn zoned_bucket_name(bucket: usize) -> String {
    let (family, zone) = (bucket / TokenZone::DIM, bucket % TokenZone::DIM);
    format!("{}.{}", ZONED_FAMILIES[family], TokenZone::NAMES[zone])
}

/// One head's read-out, averaged over the real query rows of the probe batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadAttention {
    pub block: usize,
    pub head: usize,
    /// Shannon entropy of the key distribution, in nats.
    pub entropy: f64,
    /// Attention mass by target family, in [`FAMILY_NAMES`] order. Sums to 1.
    pub family_mass: [f64; TOKEN_TYPES],
    /// Attention mass by `(family, zone)`, in [`zoned_bucket_name`] order. Sums to the mass of the
    /// two zoned families, not to 1 — this is a refinement of part of the partition above.
    pub zoned_mass: [f64; ZONED_BUCKETS],
}

/// Two heads of one block, read against each other on the query rows they share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPair {
    pub block: usize,
    /// The lower-indexed head. The divergence is symmetric, so a pair is reported once and this
    /// ordering is what makes "once" checkable.
    pub low: usize,
    pub high: usize,
    /// Jensen-Shannon divergence between the two key distributions, in nats. `0` is two copies of
    /// one head; `ln 2` is two heads whose supports never meet. Bounded by construction, which is
    /// the half of the reading that can be falsified.
    pub divergence: f64,
}

/// What one block wrote into the residual stream, per real token and averaged over them.
///
/// Ratios rather than norms because the question is one of proportion: `0.05` is a block that
/// barely perturbs what it was given, whatever the absolute scale of the stream at that depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockWrite {
    pub block: usize,
    /// `‖Attn(LN(x))‖ / ‖x‖` — the attention sublayer's write alone.
    pub attention: f64,
    /// `‖block(x) − x‖ / ‖x‖` — attention and FFN together, read off the block's own output so it
    /// is the write that happened rather than one this module reconstructed.
    pub total: f64,
    /// `‖x‖` itself. Logged because the two ratios above do not share a denominator across blocks:
    /// pre-LN accumulates into the stream with depth, so a deeper block writing the same amount in
    /// absolute terms reads as a smaller ratio, and comparing blocks without this is comparing two
    /// different questions.
    pub residual: f64,
}

/// Every head of every block, in `(block, head)` order, against the batch they were read on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AttentionStats {
    pub heads: Vec<HeadAttention>,
    /// Every unordered within-block pair, in `(block, low, high)` order.
    ///
    /// The reading [`HeadAttention`] cannot give on its own: a family mass says where a head looks,
    /// never whether the head beside it is already there.
    pub pairs: Vec<HeadPair>,
    /// What each block wrote into the residual stream, in block order — the scale factor every
    /// per-head reading above is implicitly multiplied by.
    pub writes: Vec<BlockWrite>,
    /// Share of the batch's *real* tokens held by each family, in [`FAMILY_NAMES`] order.
    ///
    /// The baseline [`HeadAttention::family_mass`] has to be read against, and without which it
    /// cannot be read at all: a family's mass rises with how many tokens that family has in play,
    /// so an abundant family looks like a preference. History fills 20 slots and stays full,
    /// Pokémon has 40 mostly padded — a head spending 30 % on History may be at chance while one
    /// spending 30 % on Attack is not.
    ///
    /// This is exactly the mass a head attending uniformly over the unmasked keys would spend.
    /// Pooling the counts over the batch is what makes that true: each query row is normalized, so
    /// the per-sample expectation `nᶠ/n` averages over rows into `Σ nᶠ / Σ n`.
    pub family_share: [f64; TOKEN_TYPES],
    /// The same baseline for the `(family, zone)` buckets, over the same denominator (the batch's
    /// real tokens) so a zoned focus and a family focus are the same kind of number.
    pub zoned_share: [f64; ZONED_BUCKETS],
}

impl<B: Backend> RlModel<B> {
    /// Disable part of one encoder block's attention in place. See [`Encoder::ablate`].
    pub fn ablate(&mut self, block: usize, ablation: AttentionAblation) {
        self.encoder.ablate(block, ablation);
    }

    /// Trainable parameters per component, `total` last.
    ///
    /// The frozen tables are absent because they are gathered, never trained — the split §1.4.3
    /// reports, and the reason a parameter count says so little about this model's cost.
    pub fn parameter_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut rows = vec![(
            "id embeddings",
            self.card_ids.num_params()
                + self.species_ids.num_params()
                + self.line_ids.num_params()
                + self.head_ids.num_params(),
        )];
        rows.extend(self.encoder.parameter_breakdown());
        rows.push(("policy heads", self.heads.num_params()));
        rows.push(("value head", self.value.num_params()));
        rows.push(("total", self.num_params()));
        rows
    }

    /// How far each encoder block has moved from `reference`'s.
    ///
    /// The reading the write measurement cannot give: a block can write hard through parameters it
    /// never learned — a fixed random pooling that the next block learned to read — and the norm of
    /// what it writes is the same in both cases.
    pub fn block_drift(&self, reference: &Self) -> Vec<BlockDrift> {
        self.encoder.drift(&reference.encoder)
    }

    /// Fold one batch's attention into [`AttentionStats`].
    ///
    /// Costs a forward of the embeddings and the blocks, plus one extra attention per block
    /// ([`super::encoder::Encoder::attention`]) — so this belongs on a cadence, over a single
    /// micro-batch, never on the training path. The pairwise fold adds no forward but is quadratic
    /// in `num_heads`, which is the term to watch if that config ever grows.
    pub fn attention_stats(&self, input: &ModelInput<B>) -> AttentionStats {
        let blocks = self
            .encoder
            .trace(self.assemble(input), input.seq_mask.clone());
        let zones = zone_indicators(input);
        let mut heads = Vec::new();
        let mut pairs = Vec::new();
        let mut writes = Vec::new();
        for (block, trace) in blocks.into_iter().enumerate() {
            heads.extend(fold_block(
                block,
                trace.weights.clone(),
                &input.seq_mask,
                &zones,
            ));
            pairs.extend(fold_pairs(block, trace.weights, &input.seq_mask));
            writes.push(fold_write(
                block,
                trace.attention,
                trace.input,
                trace.output,
                &input.seq_mask,
            ));
        }
        AttentionStats {
            heads,
            pairs,
            writes,
            family_share: family_share(&input.seq_mask),
            zoned_share: zoned_share(&zones, &input.seq_mask),
        }
    }
}

/// The zone one-hot column of each `(family, zone)` bucket, `[batch × slots]`.
///
/// Read back out of the dynamic feature block rather than carried beside it: the zone reaches the
/// model only as those four floats, and adding an index for a diagnostic would put a tensor on
/// every training forward that nothing forwards. Padding needs no separate mask — a padded slot's
/// features are zeros, so its one-hot is zero in all four zones.
fn zone_indicators<B: Backend>(input: &ModelInput<B>) -> [Tensor<B, 2>; ZONED_BUCKETS] {
    let banks = [&input.pokemon_features, &input.trainer_features];
    std::array::from_fn(|bucket| {
        let bank = banks[bucket / TokenZone::DIM].clone();
        let [batch, slots, _] = bank.dims();
        bank.narrow(2, ZONE_FEATURE_OFFSET + bucket % TokenZone::DIM, 1)
            .reshape([batch, slots])
    })
}

/// Each bucket's unmasked tokens as a share of the batch's unmasked tokens.
fn zoned_share<B: Backend>(
    zones: &[Tensor<B, 2>; ZONED_BUCKETS],
    mask: &Tensor<B, 2>,
) -> [f64; ZONED_BUCKETS] {
    let counts = zones.iter().map(|zone| zone.clone().sum()).collect();
    let shares = Tensor::cat(counts, 0)
        .div(mask.clone().sum())
        .to_data()
        .to_vec::<f32>()
        .expect("zoned shares are f32");
    std::array::from_fn(|bucket| shares[bucket] as f64)
}

/// The unmasked tokens of each family as a share of the batch's unmasked tokens.
fn family_share<B: Backend>(mask: &Tensor<B, 2>) -> [f64; TOKEN_TYPES] {
    let counts = FAMILY_RANGES
        .iter()
        .map(|(start, end)| mask.clone().narrow(1, *start, end - start).sum())
        .collect();
    let shares = Tensor::cat(counts, 0)
        .div(mask.clone().sum())
        .to_data()
        .to_vec::<f32>()
        .expect("family shares are f32");
    std::array::from_fn(|family| shares[family] as f64)
}

/// `weights: [batch × heads × query × key]`, `mask: [batch × query]`, `1.0` on real slots.
fn fold_block<B: Backend>(
    block: usize,
    weights: Tensor<B, 4>,
    mask: &Tensor<B, 2>,
    zones: &[Tensor<B, 2>; ZONED_BUCKETS],
) -> Vec<HeadAttention> {
    let [batch, heads, queries, _] = weights.dims();

    // Padded *queries* carry a full, normalized distribution like any other row — the softmax
    // does not know they are padding — so they have to be dropped here rather than trusted to be
    // small. Padded *keys* need no such care: burn masks their scores to `min_float` first.
    let real = mask.clone().reshape([batch, 1, queries, 1]);
    let rows = real.clone().sum();

    let mean_per_head = |quantity: Tensor<B, 4>| -> Vec<f32> {
        (quantity * real.clone())
            .sum_dims_squeeze::<1, _>(&[0usize, 2, 3])
            .div(rows.clone())
            .to_data()
            .to_vec::<f32>()
            .expect("attention fold is f32")
    };

    // `clamp_min` before the log, not after: a masked key's probability underflows to exactly zero,
    // and `0 · ln 0` is the term the entropy sum needs to contribute nothing rather than `NaN`.
    let entropy = mean_per_head(
        -(weights.clone() * weights.clone().clamp_min(f32::MIN_POSITIVE).log()).sum_dim(3),
    );
    let mass: Vec<Vec<f32>> = FAMILY_RANGES
        .iter()
        .map(|(start, end)| {
            mean_per_head(weights.clone().narrow(3, *start, end - start).sum_dim(3))
        })
        .collect();
    let zoned: Vec<Vec<f32>> = zones
        .iter()
        .enumerate()
        .map(|(bucket, zone)| {
            let (start, end) = FAMILY_RANGES[ZONED_FAMILY_INDEX[bucket / TokenZone::DIM]];
            let slots = end - start;
            let indicator = zone.clone().reshape([batch, 1, 1, slots]);
            mean_per_head((weights.clone().narrow(3, start, slots) * indicator).sum_dim(3))
        })
        .collect();

    (0..heads)
        .map(|head| HeadAttention {
            block,
            head,
            entropy: entropy[head] as f64,
            family_mass: std::array::from_fn(|family| mass[family][head] as f64),
            zoned_mass: std::array::from_fn(|bucket| zoned[bucket][head] as f64),
        })
        .collect()
}

/// One block's residual write. `[batch × n × d_model]` throughout, `mask: [batch × n]`.
fn fold_write<B: Backend>(
    block: usize,
    attention: Tensor<B, 3>,
    input: Tensor<B, 3>,
    output: Tensor<B, 3>,
    mask: &Tensor<B, 2>,
) -> BlockWrite {
    let [batch, slots, _] = input.dims();
    let real = mask.clone().reshape([batch, slots, 1]);
    let rows = real.clone().sum();

    let norm = |value: Tensor<B, 3>| value.powf_scalar(2.0).sum_dim(2).sqrt();
    let mean = |value: Tensor<B, 3>| (value * real.clone()).sum();
    // A padded token's stream is exactly zero at the first block, so the ratio there is `0/0`. The
    // mask drops those rows anyway; the clamp is what keeps the division from producing the `NaN`
    // that would poison the sum before the mask can.
    let stream = norm(input.clone()).clamp_min(f32::MIN_POSITIVE);
    let ratio = |write: Tensor<B, 3>| mean(norm(write).div(stream.clone()));

    let folded = Tensor::cat(
        vec![
            ratio(attention),
            ratio(output - input.clone()),
            mean(norm(input)),
        ],
        0,
    )
    .div(rows)
    .to_data()
    .to_vec::<f32>()
    .expect("residual writes are f32");

    BlockWrite {
        block,
        attention: folded[0] as f64,
        total: folded[1] as f64,
        residual: folded[2] as f64,
    }
}

/// One block's within-block pairs. Same tensor layout and same row convention as [`fold_block`].
fn fold_pairs<B: Backend>(
    block: usize,
    weights: Tensor<B, 4>,
    mask: &Tensor<B, 2>,
) -> Vec<HeadPair> {
    let [batch, heads, queries, _] = weights.dims();

    // Padded queries are dropped for the reason [`fold_block`] drops them, and here it bites
    // harder: two heads' padded rows are near-identical by construction, so keeping them would pull
    // every pair toward `0` — toward "redundant" — with no head having agreed about anything.
    let real = mask.clone().reshape([batch, 1, queries, 1]);
    let rows = real.clone().sum();

    // `clamp_min` before the log, as in [`fold_block`]: a masked key underflows to exactly zero and
    // its term has to contribute nothing rather than `NaN`.
    let entropy = |p: Tensor<B, 4>| -> Tensor<B, 4> {
        -(p.clone() * p.clamp_min(f32::MIN_POSITIVE).log()).sum_dim(3)
    };

    let pairs: Vec<(usize, usize)> = (0..heads)
        .flat_map(|low| (low + 1..heads).map(move |high| (low, high)))
        .collect();
    let totals: Vec<Tensor<B, 1>> = pairs
        .iter()
        .map(|(low, high)| {
            let p = weights.clone().narrow(1, *low, 1);
            let q = weights.clone().narrow(1, *high, 1);
            let mixture = (p.clone() + q.clone()).div_scalar(2.0);
            let divergence = entropy(mixture) - (entropy(p) + entropy(q)).div_scalar(2.0);
            (divergence * real.clone()).sum()
        })
        .collect();
    let divergences = Tensor::cat(totals, 0)
        .div(rows)
        .to_data()
        .to_vec::<f32>()
        .expect("pairwise divergences are f32");

    pairs
        .into_iter()
        .zip(divergences)
        .map(|((low, high), divergence)| HeadPair {
            block,
            low,
            high,
            divergence: divergence as f64,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    use crate::rl::action_mask::project;
    use crate::rl::model::config::ModelConfig;
    use crate::rl::model::input::DecisionPoint;
    use crate::rl::observation::get_observation;
    use crate::rl::text_embedding::TextEmbeddings;
    use crate::test_support::init_random_players;
    use crate::Game;

    /// A decision point from a real game, so the probe meets the padding pattern it will actually
    /// see rather than a full sequence.
    fn probe_batch() -> (RlModel<NdArray>, ModelInput<NdArray>) {
        let game = Game::new(init_random_players(), 5);
        let state = game.get_state_clone();
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(&state, actor, &actions, None, None);
        let mask = project(&state, &actions, &observation);
        let model = RlModel::new(
            &ModelConfig::default(),
            &TextEmbeddings::zeros(),
            &Default::default(),
        );
        let input = ModelInput::from_points(
            &[DecisionPoint {
                observation: &observation,
                mask: &mask,
            }],
            &ModelConfig::default(),
            &Default::default(),
        );
        (model, input)
    }

    /// A breakdown that does not add up is a table with a component silently missing from it,
    /// which is exactly what a new module on `RlModel` would produce.
    #[test]
    fn the_parameter_breakdown_accounts_for_every_parameter() {
        let (model, _) = probe_batch();
        let rows = model.parameter_breakdown();
        let (last, components) = rows.split_last().expect("a breakdown has rows");
        assert_eq!(last.0, "total");
        let summed: usize = components.iter().map(|(_, params)| params).sum();
        assert_eq!(
            summed, last.1,
            "components sum to {summed}, total {}",
            last.1
        );
    }

    /// The baseline has to be a distribution over the same partition the masses are, or the ratio
    /// of the two is not a "times chance" reading of anything.
    #[test]
    fn the_family_shares_partition_the_real_tokens() {
        let (model, input) = probe_batch();
        let stats = model.attention_stats(&input);
        let total: f64 = stats.family_share.iter().sum();
        assert!((total - 1.0).abs() < 1.0e-4, "shares sum to {total}, not 1");
        // The global token is never padded and is its family's only member, so its share is the
        // reciprocal of the real token count — the one entry with a closed form to check against.
        let real = 1.0 / stats.family_share[0];
        assert!(
            (1.0..=SEQ_LEN as f64).contains(&real),
            "the global share implies {real} real tokens, outside 1..={SEQ_LEN}"
        );
    }

    #[test]
    fn every_head_of_every_block_is_reported() {
        let config = ModelConfig::default();
        let (model, input) = probe_batch();
        let stats = model.attention_stats(&input);
        assert_eq!(stats.heads.len(), config.num_blocks * config.num_heads);
        for (index, head) in stats.heads.iter().enumerate() {
            assert_eq!(head.block, index / config.num_heads);
            assert_eq!(head.head, index % config.num_heads);
        }
    }

    /// The families partition the sequence: nothing the softmax spends may fall outside them, and
    /// nothing may be counted twice. This is what would break if [`super::input`]'s offsets moved
    /// without [`FAMILY_RANGES`] following.
    #[test]
    fn the_family_masses_partition_the_attention() {
        let (model, input) = probe_batch();
        for head in model.attention_stats(&input).heads {
            let total: f64 = head.family_mass.iter().sum();
            assert!(
                (total - 1.0).abs() < 1.0e-4,
                "block {} head {} spends {total}, not 1",
                head.block,
                head.head
            );
        }
    }

    /// The zoned buckets have to refine the families they came from, not merely correlate with
    /// them: a token counted in `trainer` but in none of the four `trainer.*` zones (or in two of
    /// them) would make the zone split a second, disagreeing measurement of the same thing. Checked
    /// on both the baseline and every head's mass, which is where a wrong key range would show.
    #[test]
    fn the_zoned_buckets_refine_their_families() {
        let (model, input) = probe_batch();
        let stats = model.attention_stats(&input);
        for (family, name) in ZONED_FAMILIES.iter().enumerate() {
            let index = FAMILY_NAMES.iter().position(|f| f == name).expect("family");
            let zones = family * TokenZone::DIM..(family + 1) * TokenZone::DIM;

            let share: f64 = stats.zoned_share[zones.clone()].iter().sum();
            assert!(
                (share - stats.family_share[index]).abs() < 1.0e-4,
                "{name}: zones hold {share} of the tokens, the family {}",
                stats.family_share[index]
            );
            for head in &stats.heads {
                let mass: f64 = head.zoned_mass[zones.clone()].iter().sum();
                assert!(
                    (mass - head.family_mass[index]).abs() < 1.0e-4,
                    "{name}: block {} head {} spends {mass} across zones, {} on the family",
                    head.block,
                    head.head,
                    head.family_mass[index]
                );
            }
        }
    }

    /// The uniform ablation has to *be* uniform, and the read-out is what says so: every family at
    /// exactly chance, every block-0 pair at zero divergence, and an entropy equal to the log of the
    /// real token count. An ablation that only approximated this would confound the winrate it is
    /// measured by with some other pattern nobody chose.
    #[test]
    fn the_uniform_ablation_leaves_block_zero_at_chance_on_every_family() {
        let (mut model, input) = probe_batch();
        model.ablate(0, AttentionAblation::UniformPattern);
        let stats = model.attention_stats(&input);

        let real = 1.0 / stats.family_share[0];
        for head in stats.heads.iter().filter(|head| head.block == 0) {
            assert!(
                (head.entropy - real.ln()).abs() < 1.0e-4,
                "{head:?} reads {} nats against ln({real})",
                head.entropy
            );
            // Families absent from the batch are skipped for the reason `diagnostics::attention`
            // skips them: a bucket with no token has no chance level to divide by, and a probe
            // frame from the first decision of a game carries neither attacks nor history.
            for (mass, share) in head.family_mass.iter().zip(&stats.family_share) {
                if *share > 0.0 {
                    assert!(
                        (mass / share - 1.0).abs() < 1.0e-4,
                        "{head:?} is not at chance on every family that is present"
                    );
                }
            }
        }
        for pair in stats.pairs.iter().filter(|pair| pair.block == 0) {
            assert!(
                pair.divergence < 1.0e-6,
                "{pair:?} — two uniform heads must read as one head"
            );
        }
    }

    /// The silent ablation must take the attention write and leave the feed-forward, which is the
    /// write measurement's business to confirm.
    #[test]
    fn the_silent_ablation_stops_block_zero_attention_writing() {
        let (mut model, input) = probe_batch();
        let before = model.attention_stats(&input).writes[0];
        model.ablate(0, AttentionAblation::Silent);
        let after = model.attention_stats(&input).writes[0];

        assert!(before.attention > 0.0, "{before:?} wrote nothing to start");
        assert_eq!(after.attention, 0.0, "{after:?} still writes");
        assert!(
            after.total > 0.0,
            "{after:?} lost its feed-forward too, which this ablation does not touch"
        );
    }

    /// A model has not moved from itself, and two initializations of it have. The first is the
    /// reading the drift series exists to make, so it must be exactly `0` and not merely small;
    /// the second is what proves the comparison is sensitive to anything at all.
    #[test]
    fn a_model_has_not_drifted_from_itself_but_has_from_another_init() {
        let config = ModelConfig::default();
        let device = Default::default();
        let embeddings = TextEmbeddings::zeros();

        NdArray::<f32>::seed(&device, 0x51D);
        let model = RlModel::<NdArray>::new(&config, &embeddings, &device);
        for block in model.block_drift(&model) {
            assert_eq!(block.pattern, 0.0, "{block:?} moved from itself");
            assert_eq!(block.value, 0.0, "{block:?} moved from itself");
            assert_eq!(block.feed_forward, 0.0, "{block:?} moved from itself");
        }

        NdArray::<f32>::seed(&device, 0x0A7E);
        let other = RlModel::<NdArray>::new(&config, &embeddings, &device);
        for block in model.block_drift(&other) {
            for (part, drift) in [
                ("pattern", block.pattern),
                ("value", block.value),
                ("feed_forward", block.feed_forward),
            ] {
                assert!(
                    drift > 0.1,
                    "{block:?} reads two inits as the same on {part}"
                );
            }
        }
    }

    /// The two readings the whole point of this series rests on: a block that wrote nothing reads
    /// `0`, and one that wrote as much as it was given reads `1`. Synthetic, because a real block
    /// lands between them and so cannot pin either end.
    #[test]
    fn a_block_that_writes_nothing_reads_zero_and_one_that_doubles_reads_one() {
        let device = Default::default();
        let stream = Tensor::<NdArray, 3>::from_floats([[[3.0f32, 4.0], [1.0, 0.0]]], &device);
        let mask = Tensor::<NdArray, 2>::from_floats([[1.0, 1.0]], &device);

        let idle = fold_write(
            0,
            stream.clone().zeros_like(),
            stream.clone(),
            stream.clone(),
            &mask,
        );
        assert!(
            idle.attention.abs() < 1.0e-6,
            "idle wrote {}",
            idle.attention
        );
        assert!(idle.total.abs() < 1.0e-6, "idle wrote {}", idle.total);
        // ‖(3,4)‖ = 5 and ‖(1,0)‖ = 1, so the mean stream norm is 3 — a value that would not come
        // out of an accidental sum over the wrong axis.
        assert!((idle.residual - 3.0).abs() < 1.0e-6, "{}", idle.residual);

        let doubling = fold_write(
            0,
            stream.clone(),
            stream.clone(),
            stream.clone() * 2.0,
            &mask,
        );
        assert!((doubling.attention - 1.0).abs() < 1.0e-6, "{doubling:?}");
        assert!((doubling.total - 1.0).abs() < 1.0e-6, "{doubling:?}");
    }

    /// A padded token's stream is exactly zero at the first block, so its ratio is `0/0`. The
    /// reading has to be the one that row does not exist in — and must not be `NaN`, which a sum
    /// taken before the mask would produce however carefully the mask is applied afterwards.
    #[test]
    fn a_padded_token_neither_enters_the_write_nor_poisons_it() {
        let device = Default::default();
        let stream = Tensor::<NdArray, 3>::from_floats([[[3.0f32, 4.0], [0.0, 0.0]]], &device);
        let write = Tensor::<NdArray, 3>::from_floats([[[3.0f32, 4.0], [0.0, 0.0]]], &device);
        let mask = Tensor::<NdArray, 2>::from_floats([[1.0, 0.0]], &device);

        let folded = fold_write(0, write, stream.clone(), stream.clone(), &mask);
        assert!(folded.attention.is_finite(), "{folded:?} is not finite");
        assert!((folded.attention - 1.0).abs() < 1.0e-6, "{folded:?}");
        assert!((folded.residual - 5.0).abs() < 1.0e-6, "{folded:?}");
    }

    /// Every block reports a write, and a stream norm that is a norm — the falsifiable half, since
    /// a `0` here would mean the probe read a sequence that carries nothing.
    #[test]
    fn every_block_reports_what_it_wrote() {
        let config = ModelConfig::default();
        let (model, input) = probe_batch();
        let stats = model.attention_stats(&input);

        assert_eq!(stats.writes.len(), config.num_blocks);
        for (block, write) in stats.writes.iter().enumerate() {
            assert_eq!(write.block, block);
            assert!(write.residual > 0.0, "{write:?} reads an empty stream");
            assert!(
                write.attention >= 0.0 && write.attention.is_finite(),
                "{write:?}"
            );
            assert!(write.total >= 0.0 && write.total.is_finite(), "{write:?}");
        }
    }

    /// Both ends of the scale the pairwise reading rests on: a run will call two heads redundant on
    /// a number near `0` and complementary on one near `ln 2`, so those two values have to be the
    /// ones the fold actually produces. Synthetic, because a real block lands nowhere near either
    /// end and so cannot pin the scale.
    #[test]
    fn twin_heads_diverge_by_zero_and_disjoint_heads_by_ln_two() {
        let device = Default::default();
        // Heads 0 and 1 are one head twice; head 2 is one-hot on a key neither of them touches.
        let rows = [
            [[1.0f32, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ];
        let weights = Tensor::<NdArray, 3>::from_floats(rows, &device).reshape([1, 3, 2, 4]);
        let mask = Tensor::<NdArray, 2>::from_floats([[1.0, 1.0]], &device);

        let pairs = fold_pairs(0, weights, &mask);
        assert_eq!(
            pairs.iter().map(|p| (p.low, p.high)).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (1, 2)]
        );
        assert!(
            pairs[0].divergence.abs() < 1.0e-6,
            "two copies of one head read {}, not 0",
            pairs[0].divergence
        );
        for pair in &pairs[1..] {
            assert!(
                (pair.divergence - 2.0f64.ln()).abs() < 1.0e-6,
                "disjoint heads read {}, not ln 2",
                pair.divergence
            );
        }
    }

    /// A padded query row carries a normalized distribution the head never meant, and two heads'
    /// padded rows agree with each other far more than their real ones do — so counting them would
    /// bias every pair toward "redundant". Same weights, one row masked off: the reading must be
    /// the one that row does not exist in.
    #[test]
    fn a_padded_query_row_does_not_enter_the_divergence() {
        let device = Default::default();
        let rows = [
            [[1.0f32, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
            [[0.0, 1.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
        ];
        let weights = Tensor::<NdArray, 3>::from_floats(rows, &device).reshape([1, 2, 2, 4]);

        let masked = fold_pairs(
            0,
            weights.clone(),
            &Tensor::<NdArray, 2>::from_floats([[1.0, 0.0]], &device),
        );
        let counted = fold_pairs(
            0,
            weights,
            &Tensor::<NdArray, 2>::from_floats([[1.0, 1.0]], &device),
        );
        assert!(
            (masked[0].divergence - 2.0f64.ln()).abs() < 1.0e-6,
            "the real row alone reads {}, not ln 2",
            masked[0].divergence
        );
        assert!(
            counted[0].divergence < masked[0].divergence,
            "the agreeing padded row did not move the reading, so the mask proves nothing here"
        );
    }

    /// Every unordered within-block pair, exactly once, inside the divergence's own bounds — which
    /// is what a fold dividing by rows it did not sum over would break.
    #[test]
    fn every_pair_of_every_block_is_reported_once_and_in_bounds() {
        let config = ModelConfig::default();
        let (model, input) = probe_batch();
        let stats = model.attention_stats(&input);

        let per_block = config.num_heads * (config.num_heads - 1) / 2;
        assert_eq!(stats.pairs.len(), config.num_blocks * per_block);

        let mut seen: Vec<_> = stats
            .pairs
            .iter()
            .map(|pair| {
                assert!(pair.low < pair.high, "{pair:?} is not an unordered pair");
                assert!(
                    pair.divergence >= 0.0 && pair.divergence <= 2.0f64.ln() + 1.0e-6,
                    "{pair:?} is outside the bounds of a Jensen-Shannon divergence"
                );
                (pair.block, pair.low, pair.high)
            })
            .collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "a pair is reported twice");
    }

    /// An untrained head is near-uniform over the *unmasked* keys, so its entropy sits below the
    /// `ln(SEQ_LEN)` ceiling and well above zero. The ceiling is the falsifiable half: an entropy
    /// above it would mean the fold is summing something that is not a distribution.
    #[test]
    fn entropy_stays_under_the_uniform_ceiling() {
        let (model, input) = probe_batch();
        let ceiling = (SEQ_LEN as f64).ln();
        for head in model.attention_stats(&input).heads {
            assert!(
                head.entropy > 0.0 && head.entropy <= ceiling,
                "block {} head {} reads {} nats against a {ceiling} ceiling",
                head.block,
                head.head,
                head.entropy
            );
        }
    }
}
