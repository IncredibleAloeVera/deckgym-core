//! The magnet — `RL_ARCHITECTURE.md` §1.5.1b.
//!
//! §1.5.1 is two networks, and this is the second one: **a separate off-policy network trained by
//! supervised behavioral cloning on a reservoir of the best-response's past `(state, action)`
//! pairs**. It approximates the NFSP time-average policy, and it is the target of the
//! `η·KL(π_BR ‖ magnet)` term in [`super::update`]'s loss.
//!
//! The division of labour §1.5.1 states — *PFSP picks the opponent, MMD does the update, the
//! average clone carries the equilibrium* — is what fixes every choice here:
//!
//! - **It is a full [`RlModel`], not a head on the BR.** Sharing an encoder would make the average
//!   policy a function of the current best-response's representation, and the KL would then measure
//!   the distance between two readings of the same trunk rather than between two policies. The cost
//!   is the second resident model §1.4.3 sized the run for (`micro_batch = 64`, ≈ 1 GB of 4).
//! - **It is trained off-policy, and only off-policy.** Its gradient comes from the reservoir; it
//!   never sees an advantage, a return, or the value head's target. The value head it carries is
//!   inherited from the shared architecture and is simply not trained — cloning has no value
//!   target, and the magnet is never asked to evaluate a position.
//! - **The KL flows one way.** The BR's step receives the magnet as an *inference* model, so the
//!   proximal term pulls the best-response toward the average and never the reverse.
//!
//! **What a resume restores.** The weights and the optimizer moments, both in §1.5.5's hot
//! checkpoint. Not the reservoir — see [`super::reservoir`] — so the SL step is held until the
//! buffer is refilled past [`MagnetConfig::min_fill`]. The magnet stands still for those batches
//! and the KL term keeps pulling toward the policy it had at the interrupt, which is the correct
//! behaviour rather than a degradation: a magnet fitted on a few hundred correlated frames would
//! move the target somewhere neither the old nor the new BR has been.

use burn::module::AutodiffModule;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;

use super::checkpoint::AdamRecord;
use super::reservoir::{Reservoir, Sample};
use super::rollout::Episode;
use super::schedule::Schedule;
use super::update::{chosen_logprob, grad_norm};
use crate::rl::env::{env_rng, split_seed};
use crate::rl::model::config::ModelConfig;
use crate::rl::model::input::{DecisionPoint, ModelInput};
use crate::rl::model::RlModel;

/// Stream tags, in the same scheme as [`super::rollout`]'s: keyed by batch index rather than
/// advanced, so a resume reconstructs the position from the counter it checkpointed instead of from
/// how many draws the run had made.
const STREAM_MAGNET_FILL: u64 = 0x4D41_474E_0000_0001;
const STREAM_MAGNET_DRAW: u64 = 0x4D41_474E_0000_0002;
const STREAM_MAGNET_SEED: u64 = 0x4D41_474E_0000_0003;

/// §1.5.1b's hyperparameters. All v1 defaults, none of them measured — see `config/default.toml`.
#[derive(Debug, Clone)]
pub struct MagnetConfig {
    /// Frames the reservoir holds. Host RAM, not VRAM: it stores observations and masks, which are
    /// the same objects an on-policy batch holds ~2 000 of.
    pub capacity: usize,
    /// Frames the reservoir must hold before the SL step runs at all. Below it the buffer is one
    /// batch of one policy, and cloning it would teach the magnet the *current* BR rather than the
    /// average of every BR — the exact failure the reservoir exists to prevent.
    pub min_fill: usize,
    /// Frames per SL step.
    pub batch: usize,
    /// Frames per forward inside that step. A VRAM bound, like [`super::update::StepConfig`]'s.
    pub micro_batch: usize,
    pub learning_rate: Schedule,
    /// Decay on the embedding residuals (§1.2.2). The magnet fine-tunes its own copy of the ID
    /// tables like the BR does, so it carries the same regularizer — but its own coefficient: it
    /// sees a different objective and a different number of gradient steps.
    pub residual_decay: Schedule,
    pub grad_clip: f32,
}

impl Default for MagnetConfig {
    fn default() -> Self {
        MagnetConfig {
            capacity: 20_000,
            min_fill: 4_000,
            batch: 256,
            micro_batch: 64,
            learning_rate: Schedule::constant(1.0e-3),
            residual_decay: Schedule::constant(1.0e-4),
            grad_clip: 0.5,
        }
    }
}

/// What one SL step reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnetMetrics {
    pub frames: usize,
    /// Mean `−log π_magnet(a*|s)` over the drawn minibatch — the cloning loss, in nats.
    pub loss: f32,
    pub grad_norm: f32,
    pub learning_rate: f64,
    /// Frames resident, and frames ever offered. The ratio is the acceptance rate the reservoir is
    /// currently running at, which is what says how much of the run's history the magnet averages.
    pub fill: usize,
    pub seen: u64,
    /// Frames this batch's rollout put *into* the buffer. Falls through the run as `capacity / seen`
    /// does, and a zero here on a full buffer is normal rather than a stall.
    pub accepted: usize,
}

type Adam<B> = burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, RlModel<B>, B>;

pub struct Magnet<B: AutodiffBackend> {
    /// `Option` only because Burn's optimizer consumes the module it updates; it is `Some` between
    /// calls, always.
    model: Option<RlModel<B>>,
    optimizer: Adam<B>,
    config: MagnetConfig,
    reservoir: Reservoir<Sample>,
    master_seed: u64,
}

impl<B: AutodiffBackend> Magnet<B> {
    /// A magnet at its own random init. Seeding it on the heuristic anchor (§1.1.3's "heuristic
    /// anchor as the initial magnet") is [`super::anchor`]'s job, and it happens through
    /// [`Magnet::reservoir_mut`] before the loop starts.
    pub fn new(model: RlModel<B>, config: MagnetConfig, master_seed: u64) -> Self {
        Magnet {
            model: Some(model),
            optimizer: Self::optimizer(&config),
            reservoir: Reservoir::new(config.capacity),
            config,
            master_seed,
        }
    }

    fn optimizer(config: &MagnetConfig) -> Adam<B> {
        AdamWConfig::new()
            .with_weight_decay(0.0)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.grad_clip)))
            .init()
    }

    pub fn config(&self) -> &MagnetConfig {
        &self.config
    }

    fn model(&self) -> &RlModel<B> {
        self.model.as_ref().expect("the magnet is between steps")
    }

    /// The KL target for one BR step — the inference twin, so no gradient can reach the average
    /// policy. Rebuilt per batch for the same reason [`super::update::inference_model`] is: the
    /// magnet moved on the previous batch, and a stale target is a KL to a policy that no longer
    /// exists.
    pub fn target(&self) -> RlModel<B::InnerBackend> {
        self.model().valid()
    }

    /// The weights, for §1.5.5's hot checkpoint.
    pub fn weights(&self) -> &RlModel<B> {
        self.model()
    }

    pub fn optimizer_record(&self) -> AdamRecord<B> {
        self.optimizer.to_record()
    }

    /// Restores a checkpointed magnet. The optimizer is rebuilt from *this* run's config before the
    /// moments are loaded into it, so a resume cannot inherit a grad clip the `.toml` dropped —
    /// same contract as [`super::update::Learner::load_optimizer`].
    pub fn restore(&mut self, model: RlModel<B>, optimizer: AdamRecord<B>) {
        self.model = Some(model);
        self.optimizer = Self::optimizer(&self.config).load_record(optimizer);
    }

    /// The buffer, for the §1.5.1 heuristic seed.
    pub fn reservoir_mut(&mut self) -> &mut Reservoir<Sample> {
        &mut self.reservoir
    }

    /// The buffer encoded for §1.5.5's hot checkpoint, or `None` while it is still empty — a
    /// checkpoint written before the first batch has nothing to carry, and an empty record would
    /// make the resume path look like it restored something.
    pub fn encoded_reservoir(&self) -> Option<Result<Vec<u8>, String>> {
        (!self.reservoir.is_empty()).then(|| self.reservoir.encode())
    }

    /// Restore a checkpointed buffer. Separate from [`Magnet::restore`] because the two halves come
    /// from different files and only the weights are always present: a checkpoint written by an
    /// older build, or by a crash, carries weights without a buffer and must still resume.
    pub fn restore_reservoir(&mut self, encoded: &[u8]) -> Result<(), String> {
        self.reservoir.restore(encoded)
    }

    pub fn reservoir(&self) -> &Reservoir<Sample> {
        &self.reservoir
    }

    /// Offer one collected batch's decision frames to the reservoir.
    ///
    /// Every frame of every episode, including the ones the BR played badly: the target is the
    /// time-average of what the best-response *did*, and filtering it by outcome would make the
    /// magnet an average of the winning policies, which is not a policy any process converges to.
    pub fn observe(&mut self, episodes: &[Episode], batch: u64) -> usize {
        let mut rng = env_rng(self.master_seed, split_seed(STREAM_MAGNET_FILL, batch));
        let mut accepted = 0;
        for frame in episodes.iter().flat_map(|episode| episode.frames.iter()) {
            let kept = self.reservoir.offer_with(&mut rng, || Sample {
                observation: frame.observation.clone(),
                mask: frame.mask.clone(),
                chosen_bit: frame.chosen_bit,
            });
            accepted += usize::from(kept);
        }
        accepted
    }

    /// One behavioral-cloning step, or `None` while the buffer is below [`MagnetConfig::min_fill`].
    ///
    /// The objective is plain cross-entropy on the taken bit: `−log π_magnet(a*|s)`, averaged over
    /// the draw. There is no advantage weighting and no baseline — cloning asks what the BR *did*,
    /// not whether it worked.
    pub fn step(
        &mut self,
        model_config: &ModelConfig,
        device: &B::Device,
        batch: u64,
        accepted: usize,
    ) -> Option<MagnetMetrics> {
        let learning_rate = self.config.learning_rate.at(batch);
        let residual_decay = self.config.residual_decay.at(batch) as f32;
        self.sl_step(
            model_config,
            device,
            split_seed(STREAM_MAGNET_DRAW, batch),
            learning_rate,
            residual_decay,
            accepted,
        )
    }

    /// The §1.1.3 heuristic seed's cloning steps, run against the anchor-filled buffer before batch
    /// 0. Returns the last step's metrics, or `None` if the buffer never reached the fill floor.
    ///
    /// Every step reads the schedules at batch 0 — the run has not started, so there is no batch to
    /// read them at — but each draws its *own* minibatch. Reusing the batch-0 draw `steps` times
    /// would fit the magnet to one sample of the anchor rather than to the anchor.
    pub fn pretrain(
        &mut self,
        model_config: &ModelConfig,
        device: &B::Device,
        steps: usize,
    ) -> Option<MagnetMetrics> {
        let learning_rate = self.config.learning_rate.at(0);
        let residual_decay = self.config.residual_decay.at(0) as f32;
        let mut last = None;
        for step in 0..steps as u64 {
            last = self.sl_step(
                model_config,
                device,
                split_seed(STREAM_MAGNET_SEED, step),
                learning_rate,
                residual_decay,
                0,
            );
        }
        last
    }

    fn sl_step(
        &mut self,
        model_config: &ModelConfig,
        device: &B::Device,
        draw_key: u64,
        learning_rate: f64,
        residual_decay: f32,
        accepted: usize,
    ) -> Option<MagnetMetrics> {
        if self.reservoir.len() < self.config.min_fill {
            return None;
        }
        let mut rng = env_rng(self.master_seed, draw_key);
        let model = self.model.take().expect("the magnet is between steps");
        let samples = self.reservoir.draw(self.config.batch, &mut rng);
        let total = samples.len() as f32;

        let mut accumulator = GradientsAccumulator::<RlModel<B>>::new();
        let mut loss_total = 0.0;
        for chunk in samples.chunks(self.config.micro_batch.max(1)) {
            let share = chunk.len() as f32 / total;
            let (loss, value) =
                clone_loss(&model, chunk, model_config, device, residual_decay, share);
            loss_total += value * share;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        }

        let grads = accumulator.grads();
        let norm = grad_norm(&grads, &model);
        self.model = Some(self.optimizer.step(learning_rate, model, grads));

        Some(MagnetMetrics {
            frames: total as usize,
            loss: loss_total,
            grad_norm: norm,
            learning_rate,
            fill: self.reservoir.len(),
            seen: self.reservoir.seen(),
            accepted,
        })
    }
}

/// One micro-batch of the cloning loss, already scaled by its share of the SL batch.
///
/// Returns the tensor to differentiate and the unscaled cross-entropy, so the reported loss is a
/// per-frame mean whatever `micro_batch` is set to.
fn clone_loss<B: AutodiffBackend>(
    model: &RlModel<B>,
    samples: &[&Sample],
    model_config: &ModelConfig,
    device: &B::Device,
    residual_decay: f32,
    share: f32,
) -> (Tensor<B, 1>, f32) {
    let points: Vec<DecisionPoint<'_>> = samples
        .iter()
        .map(|sample| DecisionPoint {
            observation: &sample.observation,
            mask: &sample.mask,
        })
        .collect();
    let input = ModelInput::<B>::from_points(&points, model_config, device);
    let policy = model.forward(&input).policy;

    // The same floor and the same gather as the BR step: the chosen bit is on-mask by construction,
    // so the clamp only ever touches bits nothing reads.
    let log_policy = policy.clamp_min(1.0e-9).log();
    let taken = chosen_logprob(
        log_policy,
        samples.iter().map(|sample| sample.chosen_bit),
        device,
    );
    let cross_entropy = -taken.mean();
    let value = cross_entropy.clone().into_scalar().to_f64() as f32;
    let loss = (cross_entropy + model.embedding_residual_l2() * residual_decay) * share;
    (loss, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::PlayerCode;
    use crate::rl::text_embedding::TextEmbeddings;
    use crate::rl::train::deck_db::DeckDb;
    use crate::rl::train::opponent::OpponentModels;
    use crate::rl::train::rollout::{Collector, RolloutConfig};
    use crate::rl::train::sampler::{DeckSampler, SamplerConfig};
    use crate::rl::train::update::inference_model;
    use burn::backend::{Autodiff, NdArray};
    use std::path::Path;

    type Ad = Autodiff<NdArray>;
    type Device = burn::backend::ndarray::NdArrayDevice;

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

    /// One on-policy batch, the way §1.5.1 collects it.
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

    fn magnet(
        config: &ModelConfig,
        device: &Device,
        tune: impl Fn(&mut MagnetConfig),
    ) -> Magnet<Ad> {
        let mut magnet_config = MagnetConfig {
            capacity: 400,
            min_fill: 20,
            // Larger than anything these tests put in the buffer, so a draw is always the *whole*
            // buffer: a cloning curve read across steps that each drew a different minibatch is
            // measuring the draw, not the fit.
            batch: 4096,
            micro_batch: 16,
            learning_rate: Schedule::constant(1.0e-2),
            ..MagnetConfig::default()
        };
        tune(&mut magnet_config);
        Magnet::new(
            RlModel::<Ad>::new(config, &TextEmbeddings::zeros(), device),
            magnet_config,
            17,
        )
    }

    /// The go/no-go for the cloning step: repeated steps on a fixed buffer must drive the
    /// cross-entropy down. A magnet whose loss does not fall is not an average policy, it is a
    /// random one, and the KL term in [`super::super::update`] would then be pulling the
    /// best-response toward noise.
    #[test]
    fn cloning_steps_reduce_the_cross_entropy() {
        let (episodes, _, config, device) = batch(80, 21);
        let mut magnet = magnet(&config, &device, |_| {});
        magnet.observe(&episodes, 0);

        let mut losses = Vec::new();
        for batch in 0..6 {
            let metrics = magnet
                .step(&config, &device, batch, 0)
                .expect("the buffer is past min_fill");
            assert!(metrics.loss.is_finite(), "{metrics:?}");
            assert!(metrics.grad_norm > 0.0, "{metrics:?}");
            losses.push(metrics.loss);
        }

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "cloning loss did not fall: {losses:?}"
        );
    }

    /// Below the fill floor the magnet does not move at all — the case a resume lands in, where a
    /// step would fit the whole average policy to whatever the first batch back happened to play.
    #[test]
    fn it_holds_its_step_until_the_buffer_is_filled() {
        let (episodes, _, config, device) = batch(40, 22);
        let mut magnet = magnet(&config, &device, |config| config.min_fill = 100_000);
        let accepted = magnet.observe(&episodes, 0);

        assert!(accepted > 0, "the rollout produced no frames");
        assert!(magnet.step(&config, &device, 0, accepted).is_none());
    }

    /// Every frame reaches the buffer, and the frames are the *decision* frames — not the games.
    #[test]
    fn it_observes_every_frame_of_the_batch() {
        let (episodes, _, config, device) = batch(60, 23);
        let mut magnet = magnet(&config, &device, |config| config.capacity = 100_000);
        let frames: usize = episodes.iter().map(|episode| episode.frames.len()).sum();

        assert_eq!(magnet.observe(&episodes, 0), frames);
        assert_eq!(magnet.reservoir().len(), frames);
        assert_eq!(magnet.reservoir().seen(), frames as u64);
    }
}
