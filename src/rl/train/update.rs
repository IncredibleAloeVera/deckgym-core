//! The learning step — `RL_ARCHITECTURE.md` §1.5.1, best-response half.
//!
//! One mirror-descent proximal step per on-policy batch: **no PPO clip, no multi-epoch**. The
//! batch is consumed once, the step is taken, the batch is thrown away. Per-step objective:
//!
//! ```text
//! −E[ log π(a|s) · Â ]  +  c_v · (V(s) − R̂)²  −  τ · H(π)
//!                       +  η · KL(π_BR ‖ magnet)  +  wd · ‖embedding residuals‖²
//! ```
//!
//! **The magnetic term is what makes this mirror descent rather than policy gradient.** Its target
//! is [`super::magnet::Magnet`]'s network, passed in as an *inference* model: the KL pulls the
//! best-response toward the average policy, never the average policy toward the best-response, so
//! no gradient may reach it. Passing `None` drops the term from the loss entirely and leaves
//! [`StepMetrics::kl_magnet`] absent — §1.5.6's rule that an unmeasured series must not be logged
//! as a flat zero. That is the §1.1.6 stage 1 ablation (*validate that the agent learns at all*),
//! not the algorithm §1.5.1 specifies.
//!
//! The entropy and the KL are both taken over the **argument bits only**. The `ACTION_TYPE` block of
//! a masked policy row carries the induced family marginals (§1.3.4), which are functions of those
//! same bits — summing it in counts part of the distribution twice: as a divergence that is partly
//! between marginals, and as an entropy that charges `τ` for family-level spread twice over.
//!
//! **Weight decay is a loss term, not the optimizer's.** §1.5.5 puts decay on *the player
//! embedding residuals* (§1.2.2), not on every parameter, so AdamW runs with `weight_decay = 0`
//! and [`RlModel::embedding_residual_l2`] carries it explicitly.
//!
//! **The batch is re-forwarded in micro-batches and the gradients accumulated.** §1.4.3 measures
//! the training knee at batch 256 on a 4 GB card, with a cliff — not a slope — past the peak VRAM
//! reservation. A rollout batch is larger than that, so it is split for the *forward* while
//! staying one step for the *optimizer*: accumulation is what keeps "one step per batch" true
//! without asking the card for a forward it cannot hold.

use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;

use super::checkpoint::AdamRecord;
use super::gae::{batch_targets, Target};
use super::rollout::{Episode, Frame};
use super::schedule::Schedule;
use crate::rl::action_mask::ACTION_TYPE_DIM;
use crate::rl::model::config::ModelConfig;
use crate::rl::model::input::{DecisionPoint, ModelInput};
use crate::rl::model::RlModel;

/// §1.5.1 / §1.5.5 step hyperparameters. Every value is the spec's v1 default.
///
/// The four loss and optimizer coefficients are [`Schedule`]s rather than numbers — an unscheduled
/// one is a constant schedule, so the step has one kind of thing to evaluate instead of two.
/// `grad_clip` is deliberately *not* among them: Burn takes it when the optimizer is built, so
/// scheduling it would mean rebuilding AdamW mid-run, and a clip is a stability bound rather than
/// a term one anneals.
#[derive(Debug, Clone)]
pub struct StepConfig {
    pub learning_rate: Schedule,
    /// Weight of the value regression against the policy gradient.
    pub value_coeff: Schedule,
    /// `τ` — entropy bonus.
    pub entropy_coeff: Schedule,
    /// `η` — weight of the magnetic term. `None` is the BR-only ablation: no `sched/eta` series,
    /// and no KL in the loss even when a magnet is passed, which is what lets the divergence be
    /// *measured* without being charged for. It is an `Option` rather than a `0.0` default because
    /// the term is only meaningful when [`super::magnet::Magnet`] exists, and a zero coefficient
    /// beside a live magnet is a different (and silently useless) run than no magnet at all.
    pub eta: Option<Schedule>,
    /// Decay on the embedding residuals only (§1.5.5).
    pub residual_decay: Schedule,
    /// Grad-norm clip (§1.5.5).
    pub grad_clip: f32,
    /// Frames per forward. Bounded by VRAM, not by the algorithm — the step is one step whatever
    /// this is (§1.4.3: knee at 256, cliff past the peak reservation).
    pub micro_batch: usize,
    /// Batches between two runs of [`Learner::probe_grad_norms`]; `0` turns it off.
    ///
    /// A cadence rather than a flag because the probe is not free: one forward and one backward
    /// per term, over a single micro-batch. At the §1.5.5 sizes that is ~1.5 % of a batch's frames
    /// paid five times, once every `n` batches — negligible at 50, and real at 1.
    pub grad_probe_every: u64,
    /// Batches between two attention read-outs ([`crate::rl::model::introspect`]); `0` turns it
    /// off.
    ///
    /// Read by the loop rather than by [`Learner::step`], unlike its neighbour above: the read-out
    /// is a property of the *weights*, not of a loss term, so it needs no backward and nothing in
    /// the step has to know about it. It lives here anyway because the two are the same kind of
    /// decision — how much of a batch a diagnostic may cost — and a cadence configured somewhere
    /// else would be tuned against a different budget.
    ///
    /// Priced rather than guessed. On the §1.5.5 sizes a batch is ~21 s: ~195 forward-equivalents
    /// of optimization over its 4160 frames plus ~65 of rollout, against the probe's ~1.5. Running
    /// it every batch would cost ~0.6 %, so the cadence is not a budget decision — what it buys is
    /// resolution, and what it has to out-resolve is the probe's own sampling noise, which one
    /// point every few hundred batches cannot even measure.
    pub attn_probe_every: u64,
}

impl Default for StepConfig {
    fn default() -> Self {
        StepConfig {
            learning_rate: Schedule::constant(3.0e-4),
            value_coeff: Schedule::constant(0.5),
            entropy_coeff: Schedule::constant(0.01),
            eta: None,
            residual_decay: Schedule::constant(1.0e-4),
            grad_clip: 0.5,
            micro_batch: 128,
            grad_probe_every: 50,
            attn_probe_every: 25,
        }
    }
}

/// The coefficients at one batch, evaluated once per step so every micro-batch of that step sees
/// the same values — a coefficient that moved between micro-batches would make the accumulated
/// gradient the gradient of no single loss.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Coefficients {
    pub learning_rate: f64,
    pub value_coeff: f32,
    pub entropy_coeff: f32,
    /// `None` when the run has no magnet — see [`StepConfig::eta`].
    pub eta: Option<f32>,
    pub residual_decay: f32,
}

impl StepConfig {
    pub fn at(&self, batch: u64) -> Coefficients {
        Coefficients {
            learning_rate: self.learning_rate.at(batch),
            value_coeff: self.value_coeff.at(batch) as f32,
            entropy_coeff: self.entropy_coeff.at(batch) as f32,
            eta: self.eta.as_ref().map(|eta| eta.at(batch) as f32),
            residual_decay: self.residual_decay.at(batch) as f32,
        }
    }
}

/// What one step reports — the §1.5.6 standard log, minus the terms whose systems do not exist
/// (elo needs §1.5.2's checkpoint pool).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepMetrics {
    pub frames: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    /// `KL(π_BR ‖ magnet)`, in nats, over the argument bits. `None` in a run with no magnet.
    ///
    /// It is the diagnostic of the magnetic term rather than a loss component: what enters the loss
    /// is `η ·` this, and reading the product would confuse a KL that collapsed with an `η` that
    /// was annealed. Zero is the meaningful reading — the BR has stopped moving away from the
    /// average policy — which is exactly why it must not also be what an absent magnet logs.
    pub kl_magnet: Option<f32>,
    /// Mean `|V(s) − R̂|`. §1.5.6 calls this value calibration, and it is the diagnostic that
    /// separates "the agent is not learning" from "the critic is flat", which a winrate curve
    /// alone cannot.
    pub value_error: f32,
    /// The scale-free readings of the same critic — see [`ValueDiagnostics`]. Zero on the
    /// micro-batch metrics, which have no view of the whole batch; filled in by [`Learner::step`].
    pub value: ValueDiagnostics,
    /// Per-term gradient norms at the shared trunk, on the batches [`StepConfig::grad_probe_every`]
    /// asked for them. `None` on every other batch — a probe that did not run has no reading, and
    /// §1.5.6 forbids writing that as a zero.
    pub grad_terms: Option<GradNorms>,
    pub grad_norm: f32,
    /// What the schedules said this batch. Reported rather than inferred: a coefficient curve read
    /// off the `.toml` is a plan, and the one that ran is the measurement.
    pub coefficients: Coefficients,
}

/// What each loss term contributes to the gradient of the parameters the policy and the value head
/// share — see [`Learner::probe_grad_norms`], which is also where the caveats are.
///
/// Comparable across terms, and that is the whole point: each is `|c| · ‖∇L‖` at the same
/// parameters, so `value` against `policy` is the answer to whether the critic's regression or the
/// policy gradient is shaping the encoder.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GradNorms {
    pub policy: f32,
    pub value: f32,
    pub entropy: f32,
    pub residual: f32,
    /// `None` in a run with no magnet, or one that measures the KL without paying for it.
    pub kl_magnet: Option<f32>,
}

/// How many buckets [`ValueDiagnostics::calibration`] cuts `[−1, 1]` into.
pub const VALUE_BUCKETS: usize = 4;

/// The buckets' ranges, for the series names. Beside the constant so the two cannot drift.
pub const VALUE_BUCKET_LABELS: [&str; VALUE_BUCKETS] =
    ["-1.0..-0.5", "-0.5..0.0", "0.0..0.5", "0.5..1.0"];

/// One calibration bucket: what the critic claimed against what happened.
///
/// A `share` of zero means no frame landed here and the other two are not measurements.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CalibrationBucket {
    pub share: f32,
    /// Mean `V(s)` over the bucket.
    pub predicted: f32,
    /// Mean terminal outcome over the bucket, on `V`'s own `[−1, 1]` scale rather than as a win
    /// frequency — a tie is a `0` here, and a win rate would have to decide where to put it.
    pub observed: f32,
}

/// What `loss/value` cannot say.
///
/// That loss is an MSE against the λ-return, and the λ-return is built from the critic's own
/// predictions (§1.5.1): the residual it measures is the unnormalized advantage, not the critic's
/// error against what happened, and it shrinks with `λ` whatever the critic knows. It also has no
/// scale — an MSE of 0.29 means nothing except beside the variance of what was being predicted,
/// and in self-play that variance is itself moving, since the pool tracks the learner and keeps the
/// games near even. So a flat `loss/value` curve cannot tell a critic sitting near the Bayes floor
/// of an imperfect-information game from one that has given up and predicts the batch mean.
///
/// These do. The two explained variances divide the scale out; the buckets ask whether the critic's
/// numbers mean what they claim, which no aggregate error can.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ValueDiagnostics {
    /// `1 − Var(R̂ − V) / Var(R̂)` against the λ-return — the target the regression is actually
    /// fit to, so this is the honest reading of `loss/value` and moves with it.
    pub explained: f32,
    /// The same against the terminal outcome. This is the one with a meaning outside the run:
    /// there is no bootstrap in it, so it cannot be improved by a critic agreeing with itself.
    pub mc_explained: f32,
    /// Mean `|V(s) − R|` against the terminal outcome. `1.0` is what a critic stuck at zero scores.
    pub mc_abs_error: f32,
    /// `Σ share · |predicted − observed|` over the buckets. Zero says the critic's numbers can be
    /// read as win probabilities; it says nothing about whether they are *sharp*, which is what
    /// [`Self::mc_explained`] is for — a critic that always predicts the base rate is perfectly
    /// calibrated and worthless.
    pub calibration_error: f32,
    pub calibration: [CalibrationBucket; VALUE_BUCKETS],
}

/// Reads the critic off the batch that was just collected.
///
/// `episodes` is the flattening [`batch_targets`] consumed — the collector's own `V(s)`, produced
/// by the weights this step begins from, which is why nothing here costs a forward.
pub fn value_diagnostics<'a>(
    episodes: impl IntoIterator<Item = (&'a [f32], f32)>,
    targets: &[Target],
) -> ValueDiagnostics {
    let mut values: Vec<f32> = Vec::with_capacity(targets.len());
    let mut outcomes: Vec<f32> = Vec::with_capacity(targets.len());
    for (episode_values, reward) in episodes {
        values.extend_from_slice(episode_values);
        outcomes.resize(values.len(), reward);
    }
    // The caller has to flatten in the order `batch_targets` did. A mismatch would pair a frame
    // with another frame's target and still produce entirely plausible numbers.
    if values.is_empty() || values.len() != targets.len() {
        return ValueDiagnostics::default();
    }

    // `Target::advantage` is not this residual: `batch_targets` normalizes it across the batch,
    // and `value_target − V` is what the regression is left with.
    let lambda_targets: Vec<f32> = targets.iter().map(|target| target.value_target).collect();
    let lambda_residuals: Vec<f32> = lambda_targets
        .iter()
        .zip(&values)
        .map(|(target, value)| target - value)
        .collect();
    let mc_residuals: Vec<f32> = outcomes
        .iter()
        .zip(&values)
        .map(|(outcome, value)| outcome - value)
        .collect();

    let mut counts = [0usize; VALUE_BUCKETS];
    let mut predicted = [0.0f32; VALUE_BUCKETS];
    let mut observed = [0.0f32; VALUE_BUCKETS];
    for (value, outcome) in values.iter().zip(&outcomes) {
        // The head is `tanh`, so the clamp only ever catches the closed `+1` edge.
        let index = (((value + 1.0) * 0.5 * VALUE_BUCKETS as f32) as usize).min(VALUE_BUCKETS - 1);
        counts[index] += 1;
        predicted[index] += value;
        observed[index] += outcome;
    }

    let total = values.len() as f32;
    let mut calibration = [CalibrationBucket::default(); VALUE_BUCKETS];
    let mut calibration_error = 0.0;
    for index in 0..VALUE_BUCKETS {
        if counts[index] == 0 {
            continue;
        }
        let count = counts[index] as f32;
        let bucket = CalibrationBucket {
            share: count / total,
            predicted: predicted[index] / count,
            observed: observed[index] / count,
        };
        calibration_error += bucket.share * (bucket.predicted - bucket.observed).abs();
        calibration[index] = bucket;
    }

    ValueDiagnostics {
        explained: explained_variance(&lambda_residuals, &lambda_targets),
        mc_explained: explained_variance(&mc_residuals, &outcomes),
        mc_abs_error: mean(&mc_residuals.iter().map(|r| r.abs()).collect::<Vec<_>>()),
        calibration_error,
        calibration,
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn variance(values: &[f32]) -> f32 {
    let mean = mean(values);
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32
}

fn explained_variance(residuals: &[f32], targets: &[f32]) -> f32 {
    let total = variance(targets);
    // A batch whose games all ended the same way has nothing to explain, and 0 is both the honest
    // reading and what the mean predictor it is measured against would score.
    if total <= f32::EPSILON {
        return 0.0;
    }
    1.0 - variance(residuals) / total
}

type Adam<B> = burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, RlModel<B>, B>;

/// Holds the optimizer state across steps.
pub struct Learner<B: AutodiffBackend> {
    optimizer: Adam<B>,
    config: StepConfig,
}

impl<B: AutodiffBackend> Learner<B> {
    pub fn new(config: StepConfig) -> Self {
        Learner {
            optimizer: Self::optimizer(&config),
            config,
        }
    }

    /// Decay is applied in the loss, on the residuals alone — see the module docs.
    fn optimizer(config: &StepConfig) -> Adam<B> {
        AdamWConfig::new()
            .with_weight_decay(0.0)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.grad_clip)))
            .init()
    }

    /// AdamW's moments, for the hot checkpoint (§1.5.5).
    pub fn optimizer_record(&self) -> AdamRecord<B> {
        self.optimizer.to_record()
    }

    /// Restores the moments from a hot checkpoint, rebuilding the optimizer from *this* run's
    /// config so a resume cannot inherit a grad clip the current `.toml` no longer sets.
    pub fn load_optimizer(&mut self, record: AdamRecord<B>) {
        self.optimizer = Self::optimizer(&self.config).load_record(record);
    }

    /// Consume one on-policy batch and take the single MMD step.
    ///
    /// `magnet` is the KL target (§1.5.1b), as an inference model so the term cannot leak a
    /// gradient into the average policy. `None` drops the term — see the module docs.
    ///
    /// The model is moved in and out because that is how Burn's optimizer applies an update.
    pub fn step(
        &mut self,
        mut model: RlModel<B>,
        episodes: &[Episode],
        magnet: Option<&RlModel<B::InnerBackend>>,
        model_config: &ModelConfig,
        device: &B::Device,
        batch: u64,
    ) -> (RlModel<B>, StepMetrics) {
        let coefficients = self.config.at(batch);
        let frames: Vec<&Frame> = episodes
            .iter()
            .flat_map(|episode| episode.frames.iter())
            .collect();
        if frames.is_empty() {
            return (model, StepMetrics::default());
        }

        let values_by_episode: Vec<(Vec<f32>, f32)> = episodes
            .iter()
            .map(|episode| {
                (
                    episode.frames.iter().map(|frame| frame.value).collect(),
                    episode.reward,
                )
            })
            .collect();
        let targets = batch_targets(
            values_by_episode
                .iter()
                .map(|(values, reward)| (values.as_slice(), *reward)),
        );

        let mut accumulator = GradientsAccumulator::<RlModel<B>>::new();
        let mut metrics = StepMetrics {
            frames: frames.len(),
            coefficients,
            value: value_diagnostics(
                values_by_episode
                    .iter()
                    .map(|(values, reward)| (values.as_slice(), *reward)),
                &targets,
            ),
            ..StepMetrics::default()
        };
        // Every micro-batch contributes its share of the batch mean, so the accumulated gradient
        // is the gradient of the whole-batch loss and not of the last chunk.
        let total = frames.len() as f32;

        // Before the accumulation loop and on its own forward: the probe backwards each term
        // separately, and reusing the step's graph for that would interleave five backward passes
        // with the one whose gradients are kept.
        let probe = self.config.grad_probe_every;
        if probe > 0 && batch.is_multiple_of(probe) {
            let chunk = &frames[..frames.len().min(self.config.micro_batch.max(1))];
            metrics.grad_terms = Some(self.probe_grad_norms(
                &model,
                chunk,
                &targets[..chunk.len()],
                magnet,
                model_config,
                device,
                coefficients,
            ));
        }

        let mut start = 0;
        for chunk in frames.chunks(self.config.micro_batch.max(1)) {
            let chunk_targets = &targets[start..start + chunk.len()];
            start += chunk.len();
            let share = chunk.len() as f32 / total;

            let (loss, chunk_metrics) = self.micro_batch_loss(
                &model,
                chunk,
                chunk_targets,
                magnet,
                model_config,
                device,
                share,
                coefficients,
            );
            metrics.policy_loss += chunk_metrics.policy_loss;
            metrics.value_loss += chunk_metrics.value_loss;
            metrics.entropy += chunk_metrics.entropy;
            metrics.value_error += chunk_metrics.value_error;
            if let Some(kl) = chunk_metrics.kl_magnet {
                metrics.kl_magnet = Some(metrics.kl_magnet.unwrap_or(0.0) + kl);
            }

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        }

        let grads = accumulator.grads();
        metrics.grad_norm = grad_norm(&grads, &model);
        model = self
            .optimizer
            .step(coefficients.learning_rate, model, grads);
        (model, metrics)
    }

    /// The loss of one micro-batch, already scaled by its share of the batch.
    // Eight distinct inputs with nothing to group: bundling them into a context struct would move
    // the argument list rather than shorten it.
    #[allow(clippy::too_many_arguments)]
    fn micro_batch_loss(
        &self,
        model: &RlModel<B>,
        frames: &[&Frame],
        targets: &[Target],
        magnet: Option<&RlModel<B::InnerBackend>>,
        model_config: &ModelConfig,
        device: &B::Device,
        share: f32,
        coefficients: Coefficients,
    ) -> (Tensor<B, 1>, StepMetrics) {
        let terms = self.loss_terms(model, frames, targets, magnet, model_config, device);
        let metrics = StepMetrics {
            frames: frames.len(),
            coefficients,
            policy_loss: scalar(&terms.policy) * share,
            value_loss: scalar(&terms.value) * share,
            entropy: scalar(&terms.entropy) * share,
            kl_magnet: terms.kl.as_ref().map(|kl| scalar(kl) * share),
            value_error: terms.value_error * share,
            value: ValueDiagnostics::default(),
            grad_terms: None,
            grad_norm: 0.0,
        };
        (terms.total(coefficients) * share, metrics)
    }

    /// `‖∇(c · L)‖` at the shared trunk, one loss term at a time.
    ///
    /// What `optim/grad_norm` cannot say: which term is shaping the encoder. Nor can the terms'
    /// *loss* magnitudes, and reading them that way is a trap — the advantages are normalized to
    /// zero mean ([`batch_targets`]), so `loss/policy` is near zero by construction however large
    /// its gradient is, while a positive-definite MSE never is. The comparison is only meaningful
    /// between gradients.
    ///
    /// Restricted to the parameters the policy and the value head share, because that is the only
    /// place they compete: a head's own weights take gradient from one term by construction.
    ///
    /// The coefficient multiplies the norm rather than the loss — `‖∇(c·L)‖ = |c|·‖∇L‖`, and this
    /// way one backward per term suffices. What is reported is therefore what each term actually
    /// contributes to the accumulated gradient, not its shape before the coefficient.
    // Same seven inputs as the loss it differentiates, for the same reason.
    #[allow(clippy::too_many_arguments)]
    fn probe_grad_norms(
        &self,
        model: &RlModel<B>,
        frames: &[&Frame],
        targets: &[Target],
        magnet: Option<&RlModel<B::InnerBackend>>,
        model_config: &ModelConfig,
        device: &B::Device,
        coefficients: Coefficients,
    ) -> GradNorms {
        // Burn frees the graph on `backward`, so the terms cannot share one forward: each is built
        // again for the backward that consumes it. That is the probe's whole cost, and the reason
        // it runs on a cadence instead of every step.
        let fresh = || self.loss_terms(model, frames, targets, magnet, model_config, device);
        let norm = |loss: Tensor<B, 1>, coefficient: f32| {
            let grads = GradientsParams::from_grads(loss.backward(), model);
            coefficient.abs() * shared_grad_norm(&grads, model)
        };

        GradNorms {
            // The policy gradient carries no coefficient of its own: it is the term the other
            // three are weighted *against*, so its scale is the unit here.
            policy: norm(fresh().policy, 1.0),
            value: norm(fresh().value, coefficients.value_coeff),
            entropy: norm(fresh().entropy, coefficients.entropy_coeff),
            residual: norm(fresh().residual, coefficients.residual_decay),
            // A magnet whose KL is measured but not charged (`eta = None`) contributes no gradient
            // at all, and a zero would read as a term that pulls and fails to move anything.
            kl_magnet: coefficients
                .eta
                .and_then(|eta| fresh().kl.map(|kl| norm(kl, eta))),
        }
    }

    /// The loss terms of one micro-batch, before the coefficients fold them into one number.
    ///
    /// Split out so [`Self::probe_grad_norms`] differentiates the very expressions the step
    /// optimizes: a probe that rebuilt them would drift from the loss the first time either moved.
    fn loss_terms(
        &self,
        model: &RlModel<B>,
        frames: &[&Frame],
        targets: &[Target],
        magnet: Option<&RlModel<B::InnerBackend>>,
        model_config: &ModelConfig,
        device: &B::Device,
    ) -> LossTerms<B> {
        let points: Vec<DecisionPoint<'_>> = frames
            .iter()
            .map(|frame| DecisionPoint {
                observation: &frame.observation,
                mask: &frame.mask,
            })
            .collect();
        let input = ModelInput::<B>::from_points(&points, model_config, device);
        let output = model.forward(&input);

        // Exact zeros off-mask make `ln` produce −inf, and the chosen bit is on-mask by
        // construction, so the floor only ever touches bits the gather does not read.
        let log_policy = output.policy.clone().clamp_min(1.0e-9).log();

        let chosen_logprob = chosen_logprob(
            log_policy.clone(),
            frames.iter().map(|frame| frame.chosen_bit),
            device,
        );

        let advantage = tensor_1d(targets.iter().map(|t| t.advantage), device);
        let value_target = tensor_1d(targets.iter().map(|t| t.value_target), device);

        let policy_loss = -(chosen_logprob * advantage).mean();
        let value_error = (output.value.clone() - value_target).abs();
        let value_loss = value_error.clone().powf_scalar(2.0).mean();
        // Over the legal argument bits: summed across them, averaged over the batch. The
        // `ACTION_TYPE` block is dropped for the same reason the KL below drops it — it holds the
        // induced family marginals, so leaving it in would pay `τ` for spread across families once
        // through the arguments and again through their sums. Off-mask bits are exactly zero in
        // `policy` and contribute nothing whatever the clamped log says.
        let entropy = -argument_bits(output.policy.clone() * log_policy.clone())
            .sum_dim(1)
            .mean();

        // `η · KL(π_BR ‖ magnet)` (§1.5.1). The magnet's forward runs on the inner backend over
        // this very micro-batch's input, so it costs one forward and no activations.
        let kl = magnet.map(|magnet| {
            let target = Tensor::<B, 2>::from_inner(magnet.forward(&input.to_inner()).policy);
            // Same floor as `log_policy`, and it is never read where it bites: a bit off the mask
            // is exactly zero in `output.policy`, so the product below is zero there regardless of
            // what the clamped logarithm says the magnet thought.
            let log_target = target.clamp_min(1.0e-9).log();
            let per_bit = output.policy * (log_policy - log_target);
            argument_bits(per_bit).sum_dim(1).mean()
        });

        LossTerms {
            policy: policy_loss,
            value: value_loss,
            entropy,
            residual: model.embedding_residual_l2(),
            kl,
            value_error: scalar(&value_error.mean()),
        }
    }
}

/// One micro-batch's loss terms, unweighted and unsummed — see [`Learner::loss_terms`].
struct LossTerms<B: AutodiffBackend> {
    policy: Tensor<B, 1>,
    value: Tensor<B, 1>,
    entropy: Tensor<B, 1>,
    /// The §1.5.5 decay, which is a loss term here rather than the optimizer's.
    residual: Tensor<B, 1>,
    /// `KL(π_BR ‖ magnet)`, `None` in a run with no magnet.
    kl: Option<Tensor<B, 1>>,
    /// Mean `|V(s) − R̂|`, already collapsed: a metric, not a term.
    value_error: f32,
}

impl<B: AutodiffBackend> LossTerms<B> {
    /// §1.5.1's objective, exactly as the module header writes it.
    fn total(&self, coefficients: Coefficients) -> Tensor<B, 1> {
        let mut loss = self.policy.clone() + self.value.clone() * coefficients.value_coeff
            - self.entropy.clone() * coefficients.entropy_coeff
            + self.residual.clone() * coefficients.residual_decay;
        if let (Some(kl), Some(eta)) = (&self.kl, coefficients.eta) {
            loss = loss + kl.clone() * eta;
        }
        loss
    }
}

/// Zero the `ACTION_TYPE` block of a per-bit quantity, leaving the argument bits alone.
///
/// The masked policy puts the induced family marginals in that block rather than probabilities of
/// its own (§1.3.4), so any per-bit sum meant to be over *the distribution* has to drop it: the
/// marginals are sums of the argument bits, and adding them back in counts every probability twice.
pub(crate) fn argument_bits<B: Backend>(per_bit: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch, _] = per_bit.dims();
    let offset = crate::rl::action_mask::Head::ActionType.offset();
    let device = per_bit.device();
    per_bit.slice_assign(
        [0..batch, offset..offset + ACTION_TYPE_DIM],
        Tensor::zeros([batch, ACTION_TYPE_DIM], &device),
    )
}

/// `log π(a|s)` at the taken action, one row per frame.
///
/// Shared with the magnet's cloning step ([`super::magnet`]): the two differ in what they weight
/// this by — an advantage there, nothing here — never in which number they read, and two gathers
/// that disagreed on the bit layout would be two silently different objectives.
pub(crate) fn chosen_logprob<B: Backend>(
    log_policy: Tensor<B, 2>,
    bits: impl Iterator<Item = usize>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let bits: Vec<i64> = bits.map(|bit| bit as i64).collect();
    let batch = bits.len();
    let index =
        Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(bits, [batch]), device)
            .reshape([batch, 1]);
    log_policy.gather(1, index).reshape([batch])
}

fn tensor_1d<B: Backend>(values: impl Iterator<Item = f32>, device: &B::Device) -> Tensor<B, 1> {
    let values: Vec<f32> = values.collect();
    let len = values.len();
    Tensor::from_data(burn::tensor::TensorData::new(values, [len]), device)
}

fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor.clone().into_scalar().to_f64() as f32
}

/// Global L2 norm of the accumulated gradients — §1.5.6's grad-norm, read *before* the clip so it
/// reports what the step actually produced rather than what survived clipping.
pub(crate) fn grad_norm<B: AutodiffBackend>(grads: &GradientsParams, model: &RlModel<B>) -> f32 {
    let mut visitor = NormVisitor::<B> {
        grads,
        total: 0.0,
        marker: std::marker::PhantomData,
    };
    model.visit(&mut visitor);
    visitor.total.sqrt()
}

/// The same norm over the trunk alone — the parameters both heads read from, and the only ones the
/// loss terms compete over ([`Learner::probe_grad_norms`]).
fn shared_grad_norm<B: AutodiffBackend>(grads: &GradientsParams, model: &RlModel<B>) -> f32 {
    let mut visitor = NormVisitor::<B> {
        grads,
        total: 0.0,
        marker: std::marker::PhantomData,
    };
    model.visit_shared(&mut visitor);
    visitor.total.sqrt()
}

struct NormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    total: f32,
    marker: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for NormVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            self.total += grad.powf_scalar(2.0).sum().into_scalar().to_f64() as f32;
        }
    }
}

/// The inference-side twin of a training model, for the collector: §1.5.1 is strictly on-policy,
/// so rollouts must run the *current* parameters, not a stale copy.
pub fn inference_model<B: AutodiffBackend>(model: &RlModel<B>) -> RlModel<B::InnerBackend> {
    model.valid()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::PlayerCode;
    use crate::rl::action_mask::Head;
    use crate::rl::text_embedding::TextEmbeddings;
    use crate::rl::train::deck_db::DeckDb;
    use crate::rl::train::opponent::OpponentModels;
    use crate::rl::train::rollout::{Collector, RolloutConfig};
    use crate::rl::train::sampler::{DeckSampler, SamplerConfig};
    use burn::backend::{Autodiff, NdArray};
    use std::path::Path;

    type Ad = Autodiff<NdArray>;
    type Device = burn::backend::ndarray::NdArrayDevice;

    /// Small enough that a debug build can take real gradient steps. Nothing asserted below is a
    /// claim about the §1.4.3 sizes — these tests are about the step, not the encoder.
    fn small_config() -> ModelConfig {
        ModelConfig {
            d_model: 24,
            num_blocks: 1,
            num_heads: 2,
            d_ff: 32,
            d_id: 8,
            d_head_hidden: 16,
            max_scored_candidates: 24,
            ..ModelConfig::default()
        }
    }

    /// One on-policy batch, collected the way §1.5.1 would collect it.
    fn batch(frames: usize, seed: u64) -> (Vec<Episode>, RlModel<Ad>, ModelConfig, Device) {
        let config = small_config();
        let device = Device::default();
        let model = RlModel::<Ad>::new(&config, &TextEmbeddings::zeros(), &device);

        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let sampler = DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.05,
                mirror: 0.10,
                archetypes: vec!["beginner".to_string()],
            },
        )
        .expect("sampler");
        let mut collector = Collector::new(
            sampler,
            RolloutConfig {
                envs: 4,
                opponents: vec![PlayerCode::R],
                max_crashes_per_batch: 8,
            },
            seed,
            None,
        )
        .expect("collector");

        let (episodes, _) = collector
            .collect(
                &inference_model(&model),
                &OpponentModels::new(),
                &config,
                &device,
                frames,
                0,
            )
            .expect("rollout");
        (episodes, model, config, device)
    }

    /// The go/no-go for the whole step: repeated steps on a **fixed** batch must drive the policy
    /// loss down. Nothing else in §1.5.1a is meaningful if this does not hold — the batch is on
    /// policy, the advantages are fixed, so a working policy gradient can only push the chosen
    /// actions' log-probabilities the way their advantages point.
    ///
    /// Multi-epoch is exactly what §1.5.1 forbids in a real run; it is used here because it turns
    /// a one-step property into a monotone curve a test can read.
    #[test]
    fn steps_on_a_fixed_batch_reduce_the_policy_loss() {
        let (episodes, mut model, config, device) = batch(60, 11);
        let mut learner = Learner::<Ad>::new(StepConfig {
            // Larger than the §1.5.5 default: this asks whether the gradient points the right way
            // in a handful of steps, not what a well-tuned run does.
            learning_rate: Schedule::constant(1.0e-2),
            ..StepConfig::default()
        });

        let mut losses = Vec::new();
        for batch in 0..6 {
            let (next, metrics) = learner.step(model, &episodes, None, &config, &device, batch);
            model = next;
            assert!(metrics.policy_loss.is_finite(), "{metrics:?}");
            assert!(metrics.value_loss.is_finite(), "{metrics:?}");
            assert!(
                metrics.grad_norm.is_finite() && metrics.grad_norm > 0.0,
                "{metrics:?}"
            );
            losses.push(metrics.policy_loss);
        }

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "policy loss did not fall: {losses:?}"
        );
    }

    /// Micro-batching is a VRAM accommodation, not an algorithm change: splitting the batch must
    /// produce the *same* step. Each chunk is scaled by its share of the batch, and if that scaling
    /// were wrong the accumulated gradient would be the last chunk's, or the batch's times the
    /// chunk count — either way this diverges immediately.
    #[test]
    fn micro_batching_does_not_change_the_step() {
        let (episodes, model, config, device) = batch(60, 12);

        let norm = |micro_batch| {
            let mut learner = Learner::<Ad>::new(StepConfig {
                micro_batch,
                // The clip would mask a scaling error by flattening both norms onto it.
                grad_clip: 1.0e9,
                ..StepConfig::default()
            });
            let (_, metrics) = learner.step(model.clone(), &episodes, None, &config, &device, 0);
            metrics
        };

        let whole = norm(4096);
        let split = norm(8);
        assert_eq!(whole.frames, split.frames);
        let relative = (whole.grad_norm - split.grad_norm).abs() / whole.grad_norm.max(1e-12);
        assert!(
            relative < 1e-3,
            "grad norm {} whole vs {} split",
            whole.grad_norm,
            split.grad_norm
        );
        assert!((whole.policy_loss - split.policy_loss).abs() < 1e-4);
    }

    /// An empty batch must be a no-op rather than a panic: a collection that produced nothing is a
    /// throughput problem, and the loop should log it and go round again.
    #[test]
    fn an_empty_batch_is_a_no_op() {
        let config = small_config();
        let device = Device::default();
        let model = RlModel::<Ad>::new(&config, &TextEmbeddings::zeros(), &device);
        let mut learner = Learner::<Ad>::new(StepConfig::default());
        let (_, metrics) = learner.step(model, &[], None, &config, &device, 0);
        assert_eq!(metrics, StepMetrics::default());
    }

    /// `KL(π ‖ π) = 0`, at the one point where the identity is checkable end to end: a magnet that
    /// *is* the best-response. Anything that made the term a divergence between something other
    /// than the two masked policies — the `ACTION_TYPE` marginals summed in, the log floor reaching
    /// an off-mask bit, a misaligned gather — shows up here as a non-zero, because those errors do
    /// not cancel when the two rows are identical.
    #[test]
    fn the_magnetic_term_is_zero_against_a_copy_of_the_policy() {
        let (episodes, model, config, device) = batch(60, 41);
        let mut learner = Learner::<Ad>::new(StepConfig {
            eta: Some(Schedule::constant(1.0)),
            ..StepConfig::default()
        });

        let magnet = inference_model(&model);
        let (_, metrics) =
            learner.step(model.clone(), &episodes, Some(&magnet), &config, &device, 0);

        let kl = metrics.kl_magnet.expect("a magnet was passed");
        assert!(kl.abs() < 1.0e-5, "KL to an identical policy is {kl}");

        // And the same batch against a *different* magnet is not zero, so the assertion above is
        // reading the term rather than a term that is always zero.
        let other = RlModel::<Ad>::new(&small_config(), &TextEmbeddings::zeros(), &device);
        let (_, metrics) = learner.step(
            model,
            &episodes,
            Some(&inference_model(&other)),
            &config,
            &device,
            0,
        );
        assert!(
            metrics.kl_magnet.expect("a magnet was passed") > 1.0e-4,
            "two independently initialized models produced no divergence"
        );
    }

    /// The KL enters the loss and reaches the gradient — a term computed, reported and then dropped
    /// would leave every curve in this file looking exactly the same.
    ///
    /// Asserted as a *difference between two runs* rather than as a falling curve, because the
    /// magnetic term is one of four and does not have to win on any given batch: from the same
    /// weights, over the same batch, against the same magnet, the run that pays `η` has to end
    /// closer to the magnet than the run that does not. Passing the magnet with `eta: None` is what
    /// makes the control possible at all — the divergence is then measured and not charged.
    #[test]
    fn the_magnetic_term_pulls_the_policy_toward_the_magnet() {
        let (episodes, model, config, device) = batch(60, 42);
        let magnet = inference_model(&RlModel::<Ad>::new(
            &small_config(),
            &TextEmbeddings::zeros(),
            &device,
        ));

        let divergence_after = |eta: Option<Schedule>| {
            let mut learner = Learner::<Ad>::new(StepConfig {
                learning_rate: Schedule::constant(1.0e-2),
                eta,
                ..StepConfig::default()
            });
            let mut model = model.clone();
            let mut kl = 0.0;
            for batch in 0..4 {
                let (next, metrics) =
                    learner.step(model, &episodes, Some(&magnet), &config, &device, batch);
                model = next;
                kl = metrics.kl_magnet.expect("a magnet was passed");
            }
            kl
        };

        let charged = divergence_after(Some(Schedule::constant(10.0)));
        let free = divergence_after(None);
        assert!(
            charged < free,
            "the magnetic term left the policy further from the magnet ({charged}) than no term at \
             all ({free})"
        );
    }

    /// `H(π) ≤ ln n` over the legal argument bits, the one bound an entropy over *a distribution*
    /// cannot break. It is the entropy's counterpart to the KL identity above, and it fails the same
    /// way: summing the `ACTION_TYPE` block back in adds the family marginals' own entropy on top of
    /// the arguments', and a near-uniform policy is already at the bound before that is added.
    ///
    /// `n` is counted off the wire mask rather than the entries, so the support the test bounds is
    /// the one the loss actually sums over.
    #[test]
    fn the_entropy_is_bounded_by_the_legal_argument_bits() {
        let (episodes, model, config, device) = batch(60, 44);
        let mut learner = Learner::<Ad>::new(StepConfig::default());
        let (_, metrics) = learner.step(model, &episodes, None, &config, &device, 0);

        let type_offset = Head::ActionType.offset();
        let legal: Vec<f32> = episodes
            .iter()
            .flat_map(|episode| episode.frames.iter())
            .map(|frame| {
                frame
                    .mask
                    .to_wire()
                    .bits
                    .iter()
                    .enumerate()
                    .filter(|(bit, set)| {
                        **set && !(type_offset..type_offset + ACTION_TYPE_DIM).contains(bit)
                    })
                    .count() as f32
            })
            .collect();
        assert!(!legal.is_empty());
        // The batch mean of `ln nᵢ`: `metrics.entropy` is the batch mean of the per-frame entropy,
        // so the per-frame bound averages into the bound on what is reported.
        let bound = legal.iter().map(|n| n.ln()).sum::<f32>() / legal.len() as f32;

        assert!(
            metrics.entropy <= bound + 1.0e-4,
            "entropy {} exceeds ln(legal argument bits) = {bound}",
            metrics.entropy
        );
        // And the bound is not vacuous: a freshly initialized policy is near-uniform over the legal
        // bits, so the reported entropy has to be close to it rather than merely under it.
        assert!(
            metrics.entropy > bound * 0.5,
            "entropy {} is far below the bound {bound} — the term is not being measured",
            metrics.entropy
        );
    }

    /// No magnet, no term, and no series either (§1.5.6): a `Some(0.0)` would be logged as a
    /// measurement of a divergence that was never computed.
    #[test]
    fn a_run_without_a_magnet_reports_no_divergence() {
        let (episodes, model, config, device) = batch(40, 43);
        let mut learner = Learner::<Ad>::new(StepConfig::default());
        let (_, metrics) = learner.step(model, &episodes, None, &config, &device, 0);

        assert_eq!(metrics.kl_magnet, None);
        assert_eq!(metrics.coefficients.eta, None);
    }

    /// What a change between two attention read-outs has to beat before it says anything.
    ///
    /// Same weights, two disjoint samples of the same batch: whatever separates the two readings is
    /// sampling noise, since nothing else differs. A run reading a drift smaller than this is
    /// reading which frames it drew.
    ///
    /// The bound is loose on purpose and the measurement is the printed number — this is a guard
    /// against a sampler that stops spreading, not a claim about the floor of a *trained* model,
    /// which an untrained one at this size cannot stand in for. Its heads are near-uniform, so the
    /// gap here is a lower bound on the one a real run pays.
    #[test]
    fn disjoint_samples_of_one_batch_agree_on_the_attention() {
        let (episodes, model, config, device) = batch(600, 71);
        let model = inference_model(&model);

        // Disjoint *games*, not disjoint frames: frames of one episode share a board and would
        // agree for reasons that have nothing to do with the sampler.
        let cut = episodes.len() / 2;
        assert!(cut > 0, "the batch collected too few episodes to split");
        let halves: Vec<_> = [&episodes[..cut], &episodes[cut..]]
            .iter()
            .map(|side| {
                let points = crate::rl::train::diagnostics::probe_points(side, 64);
                assert!(!points.is_empty(), "a half collected no frames");
                model.attention_stats(&ModelInput::from_points(&points, &config, &device))
            })
            .collect();

        let gap = |a: f64, b: f64| (a - b).abs();
        let entropy = halves[0]
            .heads
            .iter()
            .zip(&halves[1].heads)
            .map(|(a, b)| gap(a.entropy, b.entropy))
            .fold(0.0f64, f64::max);
        let share = halves[0]
            .family_share
            .iter()
            .zip(&halves[1].family_share)
            .map(|(a, b)| gap(*a, *b))
            .fold(0.0f64, f64::max);
        // The zoned buckets are the series the Trainer reading is now taken on, and they are the
        // ones whose noise could swamp it: a zone holds a fraction of its family's tokens and moves
        // with the game phase, so a bucket's share is a smaller count on a more volatile quantity.
        let zoned = halves[0]
            .zoned_share
            .iter()
            .zip(&halves[1].zoned_share)
            .map(|(a, b)| gap(*a, *b))
            .fold(0.0f64, f64::max);
        // The pair reading is the one a redundancy claim rests on, and it is bounded by `ln 2` —
        // so a gap here is a much larger fraction of its range than the same gap in nats of
        // entropy, and it needs its own floor rather than the entropy's standing in for it.
        let divergence = halves[0]
            .pairs
            .iter()
            .zip(&halves[1].pairs)
            .map(|(a, b)| gap(a.divergence, b.divergence))
            .fold(0.0f64, f64::max);
        // The write is a ratio of norms and so has no natural scale to borrow from the readings
        // above; it gets measured here for the same reason they do, and judged against itself.
        let write = halves[0]
            .writes
            .iter()
            .zip(&halves[1].writes)
            .map(|(a, b)| gap(a.attention, b.attention).max(gap(a.total, b.total)))
            .fold(0.0f64, f64::max);
        println!(
            "sampling noise — entropy {entropy:.4} nats, family share {share:.4}, \
             zoned share {zoned:.4}, pair divergence {divergence:.4} nats, \
             residual write {write:.4}"
        );

        // Loose against the ~0.07 an untrained model reads here, and deliberately so: this bound is
        // the guard, the printed number is the measurement. What it has to stay under is the gap
        // the series is used to read — the two blocks of `long_v5` sit 0.3 apart on the attention
        // write and 3.0 apart on the whole block's.
        assert!(
            write < 0.25,
            "two samples of one batch disagree by {write} on a block's write — the depth reading \
             is reading its sample"
        );

        assert!(
            divergence < 0.2,
            "two samples of one batch disagree by {divergence} nats on a pair — \
             the redundancy reading is reading its sample"
        );

        assert!(
            entropy < 0.5,
            "two samples of one batch disagree by {entropy} nats — the probe is reading its sample"
        );
        assert!(
            share < 0.15,
            "the family mix moved {share} between two samples of one batch"
        );
        assert!(
            zoned < 0.15,
            "a zone's share of the token mix moved {zoned} between two samples of one batch"
        );
    }

    /// The flattening a probe needs, in the order [`batch_targets`] used.
    fn probe_inputs(episodes: &[Episode]) -> (Vec<&Frame>, Vec<Target>) {
        let frames: Vec<&Frame> = episodes
            .iter()
            .flat_map(|episode| episode.frames.iter())
            .collect();
        let values: Vec<(Vec<f32>, f32)> = episodes
            .iter()
            .map(|episode| {
                (
                    episode.frames.iter().map(|f| f.value).collect(),
                    episode.reward,
                )
            })
            .collect();
        let targets = batch_targets(
            values
                .iter()
                .map(|(values, reward)| (values.as_slice(), *reward)),
        );
        (frames, targets)
    }

    /// The reading has to be comparable *across terms*, which means the coefficient has to be in
    /// it. Doubling `value_coeff` doubles what the value term contributes to the trunk and touches
    /// nothing else — the property that lets `value` be read against `policy` at all.
    #[test]
    fn a_term_scales_with_its_coefficient_and_only_its_own() {
        let (episodes, model, config, device) = batch(40, 51);
        let learner = Learner::<Ad>::new(StepConfig::default());
        let (frames, targets) = probe_inputs(&episodes);
        let (frames, targets) = (&frames[..16], &targets[..16]);

        let probe = |value_coeff: f32| {
            learner.probe_grad_norms(
                &model,
                frames,
                targets,
                None,
                &config,
                &device,
                Coefficients {
                    value_coeff,
                    entropy_coeff: 0.01,
                    residual_decay: 1.0e-4,
                    ..Coefficients::default()
                },
            )
        };
        let half = probe(0.5);
        let double = probe(1.0);

        assert!(half.value > 0.0 && half.policy > 0.0, "{half:?}");
        assert!(
            (double.value - 2.0 * half.value).abs() < 1.0e-4 * half.value.max(1.0),
            "{half:?} / {double:?}"
        );
        assert_eq!(half.policy, double.policy, "{half:?} / {double:?}");
        assert_eq!(half.entropy, double.entropy, "{half:?} / {double:?}");
    }

    /// The trunk restriction is the measurement, not a detail: over the whole model the value term
    /// also owns the value head's weights, which no other term can touch, so a full-model norm
    /// would credit it for gradient it never competes for.
    #[test]
    fn the_probe_reads_the_trunk_and_not_the_heads() {
        let (episodes, model, config, device) = batch(40, 53);
        let learner = Learner::<Ad>::new(StepConfig::default());
        let (frames, targets) = probe_inputs(&episodes);
        let terms = learner.loss_terms(
            &model,
            &frames[..16],
            &targets[..16],
            None,
            &config,
            &device,
        );

        let grads = GradientsParams::from_grads(terms.value.backward(), &model);
        let trunk = shared_grad_norm(&grads, &model);
        let whole = grad_norm(&grads, &model);

        assert!(trunk > 0.0, "the value term does reach the trunk");
        assert!(
            whole > trunk * 1.000_1,
            "trunk {trunk} is not a strict part of {whole} — the head was counted in"
        );
    }

    /// A cadence, not a flag: the probe costs a forward and a backward per term, and the series has
    /// gaps by design. `None` on the batches it did not run, so no zero is written where nothing
    /// was measured.
    #[test]
    fn the_probe_runs_on_its_cadence_and_reports_nothing_between() {
        let (episodes, mut model, config, device) = batch(40, 57);
        let mut learner = Learner::<Ad>::new(StepConfig {
            grad_probe_every: 2,
            ..StepConfig::default()
        });

        for batch in 0..4u64 {
            let (next, metrics) = learner.step(model, &episodes, None, &config, &device, batch);
            model = next;
            match batch % 2 {
                0 => {
                    let terms = metrics.grad_terms.expect("probed batch");
                    assert!(
                        terms.policy > 0.0 && terms.value > 0.0 && terms.entropy > 0.0,
                        "{terms:?}"
                    );
                    // No magnet, so no term and no series — same rule as `loss/kl_magnet`.
                    assert_eq!(terms.kl_magnet, None, "{terms:?}");
                }
                _ => assert_eq!(metrics.grad_terms, None),
            }
        }
    }

    /// The wiring, which the hand-built cases below cannot check: `step` flattens the episodes
    /// itself, and a flattening that disagreed with `batch_targets` would pair each frame with
    /// another frame's target and still report numbers in range.
    #[test]
    fn a_real_batch_reports_diagnostics_over_all_of_its_frames() {
        let (episodes, model, config, device) = batch(40, 47);
        let mut learner = Learner::<Ad>::new(StepConfig::default());
        let (_, metrics) = learner.step(model, &episodes, None, &config, &device, 0);

        let shares: f32 = metrics.value.calibration.iter().map(|b| b.share).sum();
        assert!((shares - 1.0).abs() < 1.0e-4, "{:?}", metrics.value);
        assert!(metrics.value.explained.is_finite(), "{:?}", metrics.value);
        assert!(
            metrics.value.mc_explained.is_finite(),
            "{:?}",
            metrics.value
        );
        // An untrained critic is wrong by roughly a full unit, and cannot be wrong by more than
        // two — the head is a `tanh` and the outcome is in `[−1, 1]`.
        assert!(
            metrics.value.mc_abs_error > 0.0 && metrics.value.mc_abs_error <= 2.0,
            "{:?}",
            metrics.value
        );
    }

    /// Builds the diagnostics the way [`Learner::step`] does, from episodes given as
    /// `(per-frame V, outcome)`.
    fn diagnose(episodes: &[(Vec<f32>, f32)]) -> ValueDiagnostics {
        let flat = || {
            episodes
                .iter()
                .map(|(values, reward)| (values.as_slice(), *reward))
        };
        value_diagnostics(flat(), &batch_targets(flat()))
    }

    /// The upper end of both scales. A critic that already knows the result makes every GAE
    /// residual zero, so it explains all of the λ-return *and* all of the outcome — the two
    /// readings only come apart when it is wrong.
    #[test]
    fn a_perfect_critic_explains_both_targets() {
        let diagnostics = diagnose(&[
            (vec![1.0, 1.0, 1.0], 1.0),
            (vec![-1.0, -1.0], -1.0),
            (vec![1.0, 1.0], 1.0),
        ]);

        assert!(
            (diagnostics.explained - 1.0).abs() < 1.0e-5,
            "{diagnostics:?}"
        );
        assert!(
            (diagnostics.mc_explained - 1.0).abs() < 1.0e-5,
            "{diagnostics:?}"
        );
        assert!(diagnostics.mc_abs_error < 1.0e-5, "{diagnostics:?}");
        assert!(diagnostics.calibration_error < 1.0e-5, "{diagnostics:?}");
    }

    /// The lower end, and the reason the series exists: a critic pinned at zero is the null model,
    /// and both explained variances have to read it as zero however low its MSE looks.
    #[test]
    fn a_critic_stuck_at_zero_explains_nothing() {
        let diagnostics = diagnose(&[
            (vec![0.0, 0.0, 0.0], 1.0),
            (vec![0.0, 0.0], -1.0),
            (vec![0.0, 0.0], 1.0),
        ]);

        assert!(diagnostics.explained.abs() < 1.0e-5, "{diagnostics:?}");
        assert!(diagnostics.mc_explained.abs() < 1.0e-5, "{diagnostics:?}");
        // Every frame is a full unit away from a `±1` outcome.
        assert!(
            (diagnostics.mc_abs_error - 1.0).abs() < 1.0e-5,
            "{diagnostics:?}"
        );
    }

    /// Why the buckets are not redundant with the aggregates. Both critics here explain nothing;
    /// only the calibration series says that one of them is also lying about its confidence.
    #[test]
    fn calibration_separates_confidence_from_accuracy() {
        let overconfident = diagnose(&[(vec![0.9, 0.9], 1.0), (vec![0.9, 0.9], -1.0)]);
        let base_rate = diagnose(&[(vec![0.0, 0.0], 1.0), (vec![0.0, 0.0], -1.0)]);

        assert!(
            overconfident.mc_explained.abs() < 1.0e-5 && base_rate.mc_explained.abs() < 1.0e-5,
            "{overconfident:?} / {base_rate:?}"
        );
        // Everything landed in `0.5..1.0`, claiming a near-certain win against an even split.
        let bucket = overconfident.calibration[VALUE_BUCKETS - 1];
        assert!((bucket.share - 1.0).abs() < 1.0e-5, "{overconfident:?}");
        assert!((bucket.predicted - 0.9).abs() < 1.0e-5, "{overconfident:?}");
        assert!(bucket.observed.abs() < 1.0e-5, "{overconfident:?}");
        assert!(
            (overconfident.calibration_error - 0.9).abs() < 1.0e-5,
            "{overconfident:?}"
        );
        // The base-rate critic is useless and honest, and the two series say so separately.
        assert!(base_rate.calibration_error.abs() < 1.0e-5, "{base_rate:?}");
    }

    /// The trap the implementation is written around: `batch_targets` normalizes `advantage` in
    /// place, so the residual has to be recomputed from `value_target`. Reading `advantage`
    /// instead would divide the residuals by their own standard deviation and report an explained
    /// variance that moves with the batch's spread rather than with the critic.
    #[test]
    fn the_lambda_residual_is_not_the_normalized_advantage() {
        let episodes = [(vec![0.2, -0.4, 0.1], 1.0), (vec![-0.3, 0.5], -1.0)];
        let flat = || {
            episodes
                .iter()
                .map(|(values, reward)| (values.as_slice(), *reward))
        };
        let targets = batch_targets(flat());

        let normalized: Vec<f32> = targets.iter().map(|t| t.advantage).collect();
        let residuals: Vec<f32> = targets
            .iter()
            .zip(episodes.iter().flat_map(|(values, _)| values))
            .map(|(target, value)| target.value_target - value)
            .collect();

        assert!(
            (variance(&normalized) - 1.0).abs() < 1.0e-4,
            "advantages are normalized: {normalized:?}"
        );
        assert!(
            (variance(&residuals) - 1.0).abs() > 1.0e-2,
            "residuals must not be the normalized advantages: {residuals:?}"
        );
    }
}
