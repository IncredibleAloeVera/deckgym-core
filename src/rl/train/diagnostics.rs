//! §1.5.6's diagnostic block — pathology detection, read off the collected batch.
//!
//! These are not the training signal; they are what tells you *why* the training signal is doing
//! what it does. The standard line can only say the agent is not improving. These say whether the
//! critic is flat, whether the policy has collapsed onto one head, whether the games are ending at
//! turn 4 or turn 90, and whether the mask is leaving the model any choice at all.
//!
//! Everything here folds over frames the collector already produced, so a run pays nothing for it
//! beyond the fold — no second forward, no second rollout.

use crate::rl::action_mask::{Head, HEADS};
use crate::rl::model::introspect::{
    zoned_bucket_name, AttentionStats, FAMILY_NAMES, ZONED_BUCKETS,
};

use crate::rl::model::input::DecisionPoint;

use super::magnet::MagnetMetrics;
use super::rollout::{Episode, Frame, HeadEntropy, RolloutStats};
use super::update::{StepMetrics, VALUE_BUCKET_LABELS};

/// A named scalar, in the form [`super::logger::MetricLog`] writes and TensorBoard reads.
pub type Scalar = (String, f64);

/// §1.5.6's standard line, restricted to the terms whose systems exist.
///
/// Self-play elo (§1.5.2) and the curriculum stage (§1.5.4, [`curriculum`]) are their own series,
/// pushed by the loop only when their systems are in play — a flat zero curve reads as a
/// measurement, so absent is the honest reading for a run without them. The KL to the magnet
/// follows the same rule *within* this function: it is emitted only by a run that has a magnet,
/// since zero is the reading that says the best-response has stopped moving away from the average
/// policy.
pub fn standard(stats: &RolloutStats, metrics: &StepMetrics, episodes: &[Episode]) -> Vec<Scalar> {
    let games = episodes.len().max(1) as f64;
    let wins = episodes.iter().filter(|e| e.reward > 0.0).count() as f64;
    let ties = episodes.iter().filter(|e| e.reward == 0.0).count() as f64;

    let mut out = vec![
        // Against the §1.5.2 panel, which §1.5.2 warns is a *saturation* signal: the panel is in
        // the training mix, so this is not held-out generalization. At ~20 games a batch it also
        // carries a ±22 % interval, which is why the games denominator is logged beside it.
        ("panel/winrate".to_string(), wins / games),
        ("panel/tierate".to_string(), ties / games),
        ("panel/games".to_string(), stats.games as f64),
        ("loss/policy".to_string(), metrics.policy_loss as f64),
        ("loss/value".to_string(), metrics.value_loss as f64),
        ("policy/entropy".to_string(), metrics.entropy as f64),
        // §1.5.6 puts value calibration in from the start, early for a diagnostic, because it is
        // what separates "the agent is not learning" from "the critic is flat".
        ("value/abs_error".to_string(), metrics.value_error as f64),
        // The three that have a scale. `value/abs_error` and `loss/value` are errors against the
        // λ-return, which is built from the critic's own predictions, so neither can be compared
        // against anything — see [`super::update::ValueDiagnostics`]. `value/explained` is the
        // same measurement with the scale divided out, and the `mc_` pair drops the bootstrap and
        // asks against the actual result of the game.
        (
            "value/explained".to_string(),
            metrics.value.explained as f64,
        ),
        (
            "value/mc_explained".to_string(),
            metrics.value.mc_explained as f64,
        ),
        (
            "value/mc_abs_error".to_string(),
            metrics.value.mc_abs_error as f64,
        ),
        (
            "value/calibration_error".to_string(),
            metrics.value.calibration_error as f64,
        ),
        ("optim/grad_norm".to_string(), metrics.grad_norm as f64),
        ("rollout/frames".to_string(), metrics.frames as f64),
        // Games §1.5.5 threw away because the engine panicked. A curve rather than a footnote:
        // recovery makes a crash survivable, not free — the dropped games are missing from every
        // other series on this line, and a rate that climbs is a regression in the simulator that
        // nothing else here would show.
        (
            "rollout/engine_panics".to_string(),
            stats.crashes.len() as f64,
        ),
        // Mean inference batch size. §1.5.5's ×37 depends on it staying near the env count; it
        // decays as envs finish out of step, and that decay is invisible in games/s alone.
        (
            "rollout/mean_forward_batch".to_string(),
            stats.frames as f64 / stats.forwards.max(1) as f64,
        ),
        // What the §1.5.5 schedules actually produced. A curve read off the `.toml` is a plan; a
        // schedule whose phases resolved against the wrong batch count looks identical on paper
        // and different here.
        (
            "sched/learning_rate".to_string(),
            metrics.coefficients.learning_rate,
        ),
        (
            "sched/entropy_coeff".to_string(),
            metrics.coefficients.entropy_coeff as f64,
        ),
        (
            "sched/value_coeff".to_string(),
            metrics.coefficients.value_coeff as f64,
        ),
        (
            "sched/residual_decay".to_string(),
            metrics.coefficients.residual_decay as f64,
        ),
    ];

    // The calibration table, one triple per bucket. `share` travels with the other two because a
    // gap read off six frames is not a gap, and an empty bucket writes zeros that would otherwise
    // read as a perfectly calibrated one.
    for (label, bucket) in VALUE_BUCKET_LABELS.iter().zip(&metrics.value.calibration) {
        out.push((
            format!("value/calibration/{label}/share"),
            bucket.share as f64,
        ));
        out.push((
            format!("value/calibration/{label}/predicted"),
            bucket.predicted as f64,
        ));
        out.push((
            format!("value/calibration/{label}/observed"),
            bucket.observed as f64,
        ));
    }

    // What each term does to the trunk, on the batches the probe ran — absent on the others rather
    // than carried forward, since a stale reading beside a live `optim/grad_norm` would date the
    // two differently. These are the series that decide whether `value_coeff` is well set: the loss
    // magnitudes cannot, because a normalized advantage puts `loss/policy` near zero whatever its
    // gradient does.
    if let Some(terms) = metrics.grad_terms {
        out.push(("optim/grad_trunk/policy".to_string(), terms.policy as f64));
        out.push(("optim/grad_trunk/value".to_string(), terms.value as f64));
        out.push(("optim/grad_trunk/entropy".to_string(), terms.entropy as f64));
        out.push((
            "optim/grad_trunk/residual".to_string(),
            terms.residual as f64,
        ));
        if let Some(kl) = terms.kl_magnet {
            out.push(("optim/grad_trunk/kl_magnet".to_string(), kl as f64));
        }
    }

    // §1.5.1's magnetic term, as the two numbers it is made of. `loss/kl_magnet` is the divergence
    // itself and `sched/eta` what the loss paid for it — the product is what a single series would
    // show, and it cannot distinguish a KL that collapsed from an `η` that was annealed.
    if let Some(kl) = metrics.kl_magnet {
        out.push(("loss/kl_magnet".to_string(), kl as f64));
    }
    if let Some(eta) = metrics.coefficients.eta {
        out.push(("sched/eta".to_string(), eta as f64));
    }

    out
}

/// The magnet's own line (§1.5.1b), emitted on the batches its SL step ran.
///
/// Kept out of [`standard`] rather than folded in: the cloning step is skipped while the reservoir
/// is below its fill floor, so these series have gaps by design, and a zero written into one of
/// those gaps would read as "the magnet learned nothing" instead of "the magnet did not step".
pub fn magnet(metrics: &MagnetMetrics) -> Vec<Scalar> {
    vec![
        ("magnet/loss".to_string(), metrics.loss as f64),
        ("magnet/grad_norm".to_string(), metrics.grad_norm as f64),
        ("magnet/frames".to_string(), metrics.frames as f64),
        // Fill against seen: the first says whether the SL step has data, the second how much of the
        // run's history that data is a sample *of*. A fill pinned at capacity with `seen` climbing
        // is the reservoir working; a fill that stops growing while `seen` does too is a rollout
        // that stopped producing frames.
        ("magnet/reservoir_fill".to_string(), metrics.fill as f64),
        ("magnet/reservoir_seen".to_string(), metrics.seen as f64),
        ("magnet/accepted".to_string(), metrics.accepted as f64),
        ("sched/magnet_lr".to_string(), metrics.learning_rate),
    ]
}

/// The encoder's attention read-out, on the batches `[step] attn_probe_every` asked for one.
///
/// Its own function rather than a branch of [`standard`], same rule as [`magnet`]: the probe has
/// gaps by design, and a zero carried into one of them would read as a head attending to nothing
/// — which is a real pathology, and so exactly the reading that must not be manufactured.
///
/// Per head, keyed `b<block>h<head>`: `attn_entropy/*` in nats, and `attn_focus/*/<family>` — the
/// attention mass a head spends on a family **divided by** that family's share of the batch's real
/// tokens. `1.0` is chance. The raw mass is not logged beside it because it is the product of the
/// two series that are: `focus × attn_share`.
///
/// The ratio rather than the mass because the mass alone is not a reading. A family's mass rises
/// with how many of its tokens are in play, and the families are not the same size: History fills
/// 20 slots and stays full where Pokémon has 40 mostly padded, so 30 % on History can be chance
/// while 30 % on Attack is a threefold preference. `attn_share/*` is logged for itself too — it
/// says how the board fills over a run, which nothing else here measures.
///
/// The Pokémon and Trainer families are logged twice: once whole, and once split by zone
/// (`attn_focus/*/trainer.hand`, …). See [`crate::rl::model::introspect`] for why the aggregate
/// baseline is unreadable for those two — a family whose share is dominated by deck and discard
/// tokens puts chance where no head should be.
///
/// Per unordered pair of heads *within* a block, keyed `b<block>h<low>h<high>`: `attn_js/*`, the
/// Jensen-Shannon divergence between the two key distributions in nats — `0` for two copies of one
/// head, `ln 2 ≈ 0.693` for two that never look at the same token. `attn_focus/*` cannot stand in
/// for it: two heads sitting on one family read as two equal masses whether they duplicate each
/// other or split the family between them, and only the pair's own number separates those.
///
/// Per block, keyed `b<block>`: `attn_write/*` and `block_write/*`, the attention sublayer's write
/// and the whole block's, each as a fraction of the residual stream it wrote into, and
/// `stream_norm/*`, that stream's own norm. What every series above is implicitly multiplied by — a
/// head's pattern only matters in proportion to what its block does with it, and a near-uniform
/// pattern in a block that barely writes is a different finding from the same pattern in a block
/// that writes hard. The stream norm is logged because pre-LN accumulates with depth, so the two
/// ratios are not read against the same denominator at both blocks.
///
/// The head index is the series' whole identity: heads are not named by anything, and the one at
/// position 3 of block 0 is only the same head as last run's because neither the block count nor
/// the head count moved. A run that changes `[model] num_heads` starts a new set of curves and
/// must not be read against the old ones.
pub fn attention(stats: &AttentionStats) -> Vec<Scalar> {
    // A probe that did not run has no token mix either — the shares of a batch nobody read are
    // zeros, and five of them would draw a sequence that holds no tokens at all.
    if stats.heads.is_empty() {
        return Vec::new();
    }

    let buckets: Vec<String> = FAMILY_NAMES
        .iter()
        .map(|family| (*family).to_string())
        .chain((0..ZONED_BUCKETS).map(zoned_bucket_name))
        .collect();

    let mut out = Vec::with_capacity(stats.heads.len() * (1 + buckets.len()));
    for (bucket, share) in buckets
        .iter()
        .zip(stats.family_share.iter().chain(&stats.zoned_share))
    {
        out.push((format!("attn_share/{bucket}"), *share));
    }
    for head in &stats.heads {
        let tag = format!("b{}h{}", head.block, head.head);
        out.push((format!("attn_entropy/{tag}"), head.entropy));
        for ((bucket, mass), share) in buckets
            .iter()
            .zip(head.family_mass.iter().chain(&head.zoned_mass))
            .zip(stats.family_share.iter().chain(&stats.zoned_share))
        {
            // A bucket with no unmasked token in the batch has no chance level to divide by, and
            // its mass is zero for the same reason. Absent rather than a manufactured `0.0`, which
            // would read as a head avoiding tokens that were never there. This is the common case
            // for a zone, not an edge case: `pokemon.discard` is empty for the first turns of every
            // game, and a zero there would read as an aversion.
            if *share > 0.0 {
                out.push((format!("attn_focus/{tag}/{bucket}"), mass / share));
            }
        }
    }
    for pair in &stats.pairs {
        out.push((
            format!("attn_js/b{}h{}h{}", pair.block, pair.low, pair.high),
            pair.divergence,
        ));
    }
    for write in &stats.writes {
        let tag = format!("b{}", write.block);
        out.push((format!("attn_write/{tag}"), write.attention));
        out.push((format!("block_write/{tag}"), write.total));
        out.push((format!("stream_norm/{tag}"), write.residual));
    }
    out
}

/// At most `count` frames of a batch, spread evenly over its episodes.
///
/// A stride, not the first `count`. At ~25 decisions an episode, the head of the flattened list is
/// two or three games out of the hundred-odd a batch collects — and the §1.5.2 panel gives those
/// games different opponents while the sampler gives them different decks. How many tokens of each
/// family a frame carries follows both: an opening frame against the random player and a turn-12
/// frame against a bake have different boards, different hands, different history fills. A
/// head-of-list probe would report whichever games sorted first and call it a property of the
/// weights, and the deck lottery would move it from batch to batch with nothing having changed.
pub fn probe_points(episodes: &[Episode], count: usize) -> Vec<DecisionPoint<'_>> {
    let frames: Vec<&Frame> = episodes
        .iter()
        .flat_map(|episode| episode.frames.iter())
        .collect();
    probe_indices(frames.len(), count)
        .map(|index| DecisionPoint {
            observation: &frames[index].observation,
            mask: &frames[index].mask,
        })
        .collect()
}

/// Positions in the flattened frame list. Split out because this is the half with a property worth
/// testing, and testing it here costs no fabricated observations.
fn probe_indices(total: usize, count: usize) -> impl Iterator<Item = usize> {
    (0..total)
        .step_by((total / count.max(1)).max(1))
        .take(count)
}

/// The curriculum's own line (§1.5.4), emitted only by a run with `[[curriculum.stages]]` —
/// absent rather than a flat `0` for a run without one, same rule as [`magnet`].
pub fn curriculum(stage_index: usize, stage_count: usize) -> Vec<Scalar> {
    vec![
        ("curriculum/stage".to_string(), stage_index as f64),
        ("curriculum/stage_count".to_string(), stage_count as f64),
    ]
}

/// Folds one batch of episodes into §1.5.6's diagnostics.
pub fn diagnostics(episodes: &[Episode], head_entropy: &HeadEntropy) -> Vec<Scalar> {
    let mut out = Vec::new();
    let frames: Vec<_> = episodes
        .iter()
        .flat_map(|episode| episode.frames.iter())
        .collect();
    if frames.is_empty() {
        return out;
    }

    let games = episodes.len() as f64;
    out.push((
        "episode/turns".to_string(),
        episodes.iter().map(|e| e.turns as f64).sum::<f64>() / games,
    ));
    out.push(("episode/decisions".to_string(), frames.len() as f64 / games));

    let mut sizes: Vec<usize> = frames
        .iter()
        .map(|frame| frame.mask.entries.len())
        .collect();
    sizes.sort_unstable();
    out.push((
        "mask/size_mean".to_string(),
        sizes.iter().sum::<usize>() as f64 / sizes.len() as f64,
    ));
    out.push(("mask/size_p50".to_string(), quantile(&sizes, 0.50)));
    out.push(("mask/size_p90".to_string(), quantile(&sizes, 0.90)));
    out.push((
        "mask/size_max".to_string(),
        *sizes.last().expect("non-empty") as f64,
    ));

    // Logged as one series per head rather than a histogram: TensorBoard draws scalars without
    // being told anything about bucketing, and 18 named curves is what one actually reads.
    let mut chosen = [0u64; HEADS.len()];
    for frame in &frames {
        if let Some(entry) = frame
            .mask
            .entries
            .iter()
            .find(|entry| entry.head.offset() + entry.index == frame.chosen_bit)
        {
            if let Some(index) = HEADS.iter().position(|head| *head == entry.head) {
                chosen[index] += 1;
            }
        }
    }
    for (index, head) in HEADS.iter().enumerate() {
        // `ActionType` carries the induced family marginals, never a chosen bit — the collector
        // asserts as much. Emitting it would be a series pinned to zero for the life of the run,
        // and the family distribution §1.5.6 asks for is what the other seventeen already are.
        if *head == Head::ActionType {
            continue;
        }
        out.push((
            format!("head_share/{}", name(*head)),
            chosen[index] as f64 / frames.len() as f64,
        ));
        if let Some(entropy) = head_entropy.mean(*head) {
            out.push((format!("head_entropy/{}", name(*head)), entropy));
        }
        // The companion to the line above, on its own denominator: of the frames that offered this
        // head at all, the share where it offered a single bit and the entropy fold therefore
        // skipped it. Entropy that falls while this climbs is the mask narrowing, not the policy
        // collapsing — and the two are indistinguishable from `head_entropy/*` alone.
        //
        // Restricted to heads that could carry two bits in the first place. `EndTurn` and
        // `UseStadium` have a one-slot domain, so their rate is 1 by arithmetic on every batch of
        // every run — the same reason the whole-frame forced rate is an invariant (§1.5.6) and not
        // a series, and these are the two heads that never get an entropy curve either.
        if head.dim() >= 2 {
            if let Some(rate) = head_entropy.forced_rate(*head) {
                out.push((format!("head_forced/{}", name(*head)), rate));
            }
        }
    }

    out
}

/// Nearest-rank, on a sorted slice.
fn quantile(sorted: &[usize], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1] as f64
}

/// Stable series names. Spelled out rather than derived from `Debug` so a rename in the mask
/// cannot silently break the continuity of a run's curves.
fn name(head: Head) -> &'static str {
    match head {
        Head::ActionType => "action_type",
        Head::Place => "place",
        Head::Evolve => "evolve",
        Head::AttachEnergy => "attach_energy",
        Head::Retreat => "retreat",
        Head::Attack => "attack",
        Head::UseAbility => "use_ability",
        Head::PlayTrainer => "play_trainer",
        Head::UseStadium => "use_stadium",
        Head::EndTurn => "end_turn",
        Head::DiscardFossil => "discard_fossil",
        Head::SlotPtrSelf => "slot_ptr_self",
        Head::SlotPtrOpp => "slot_ptr_opp",
        Head::SlotPair => "slot_pair",
        Head::HandPtr => "hand_ptr",
        Head::StatusCat => "status_cat",
        Head::RevealedHandPtr => "revealed_hand_ptr",
        Head::CandidatePtr => "candidate_ptr",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rl::model::introspect::{BlockWrite, HeadAttention, HeadPair};

    /// The property the stride exists for. A §1.5.5 batch is ~100 games of ~25 decisions, and the
    /// probe reads 64 frames: taking the head of the list would confine them to three games, whose
    /// decks and panel opponent the reading would then be about. Spread, it reaches nearly all of
    /// them.
    #[test]
    fn the_probe_samples_across_the_episodes() {
        let (games, per_game, count) = (100usize, 25usize, 64usize);
        let touched: std::collections::HashSet<usize> = probe_indices(games * per_game, count)
            .map(|index| index / per_game)
            .collect();

        assert_eq!(probe_indices(games * per_game, count).count(), count);
        assert!(
            touched.len() >= 60,
            "{count} frames landed in only {} of {games} games",
            touched.len()
        );
    }

    /// A batch smaller than the probe asks for yields what it has, in order, and never panics on
    /// the division that sizes the stride.
    #[test]
    fn the_probe_never_asks_for_more_frames_than_the_batch_has() {
        assert_eq!(
            probe_indices(7, 64).collect::<Vec<_>>(),
            (0..7).collect::<Vec<_>>()
        );
        assert_eq!(probe_indices(0, 64).count(), 0);
        assert_eq!(probe_indices(25, 0).count(), 0);
    }

    /// A probe that did not run writes nothing, same contract as the magnet's gaps.
    #[test]
    fn an_attention_probe_with_no_heads_produces_no_scalars() {
        assert!(attention(&AttentionStats::default()).is_empty());
    }

    /// Two heads must never collide on a series name: the curves are keyed by position, and a
    /// collision would silently interleave two heads into one curve. The pair keys share that
    /// alphabet, so `b1h3h5` must not be reachable as a head key either.
    #[test]
    fn every_head_gets_its_own_attention_series() {
        let stats = AttentionStats {
            heads: (0..2)
                .flat_map(|block| {
                    (0..6).map(move |head| HeadAttention {
                        block,
                        head,
                        entropy: 1.0,
                        family_mass: [0.2; FAMILY_NAMES.len()],
                        zoned_mass: [0.05; ZONED_BUCKETS],
                    })
                })
                .collect(),
            pairs: (0..2)
                .flat_map(|block| {
                    (0..6).flat_map(move |low| {
                        (low + 1..6).map(move |high| HeadPair {
                            block,
                            low,
                            high,
                            divergence: 0.4,
                        })
                    })
                })
                .collect(),
            writes: (0..2)
                .map(|block| BlockWrite {
                    block,
                    attention: 0.1,
                    total: 0.2,
                    residual: 8.0,
                })
                .collect(),
            family_share: [0.2; FAMILY_NAMES.len()],
            zoned_share: [0.05; ZONED_BUCKETS],
        };
        let scalars = attention(&stats);
        let buckets = FAMILY_NAMES.len() + ZONED_BUCKETS;
        assert_eq!(scalars.len(), buckets + 12 * (1 + buckets) + 2 * 15 + 2 * 3);
        let mut names: Vec<_> = scalars.iter().map(|(name, _)| name.clone()).collect();
        names.sort();
        let total = names.len();
        names.dedup();
        assert_eq!(names.len(), total, "two heads share a series name");
    }

    #[test]
    fn an_empty_batch_produces_no_scalars() {
        assert!(diagnostics(&[], &HeadEntropy::default()).is_empty());
    }

    #[test]
    fn quantiles_are_nearest_rank() {
        let sorted = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(quantile(&sorted, 0.5), 5.0);
        assert_eq!(quantile(&sorted, 0.9), 9.0);
        assert_eq!(quantile(&sorted, 1.0), 10.0);
        assert_eq!(quantile(&[], 0.5), 0.0);
    }

    /// The series names are a run's identity over months of curves; a rename in [`Head`] must not
    /// reach them silently.
    #[test]
    fn every_head_has_a_stable_name() {
        let mut names: Vec<_> = HEADS.iter().map(|head| name(*head)).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two heads share a series name");
    }
}
