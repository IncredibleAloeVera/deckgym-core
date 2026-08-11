//! Part 4 model (v1) — `RL_ARCHITECTURE.md` §1.4.
//!
//! One shared encoder over the Part-2 observation; heads read rows of its output
//! `H : [N × d_model]`, `N ≤ 133`; the Part-3 mask gates every probability. **No centralized
//! critic** — value and policy share the same imperfect-information observation. Feature-gated
//! (`rl-model`) so the pure simulator build never pays for the deep-learning stack; the backend is
//! Burn (CPU `NdArray` by default, per §1.4.3 "CPU-viable in Burn").
//!
//! Module map:
//!
//! | Module        | Owns                                                                        |
//! | ------------- | --------------------------------------------------------------------------- |
//! | [`config`]    | §1.4.3 sizes, `.toml`-deserializable                                        |
//! | [`tables`]    | frozen static tables + meta-neutral embedding inits (seeded, deterministic) |
//! | [`embedding`] | `frozen init ⊕ regularized learned residual` ID tables (§1.2.2)             |
//! | [`encoder`]   | five input projections, token-type tags, Pre-LN MHA blocks, padding mask    |
//! | [`input`]     | batching, self-scoped pointer row maps, candidate encodings                 |
//! | [`heads`]     | pointer / nullary heads bit-aligned with the 804-bit mask, value head       |
//! | [`introspect`]| attention read-out: what each head of each block looks at                   |
//!
//! The falsifiable properties this module ships tests for, before any training:
//! masked policy exactly zero outside the legal set on random rollouts; egocentrism (a
//! player-swapped state observed from the other seat is the *same* input, hence the same output);
//! deterministic forward; parameter count ≈ 4.4 M (§1.4.3); forward latency measured on `N = 133`.

pub mod config;
pub mod embedding;
pub mod encoder;
pub mod heads;
pub mod input;
pub mod introspect;
pub mod tables;

use burn::prelude::*;

use crate::rl::text_embedding::TextEmbeddings;

use config::ModelConfig;
use embedding::{IdEmbedding, LearnedEmbedding};
use encoder::Encoder;
use heads::{masked_policy, MaskLayout, PolicyHeads, ValueHead};
use input::ModelInput;
use tables::FrozenTables;

pub use config::ModelConfig as RlModelConfig;
pub use input::{DecisionPoint, SEQ_LEN};

/// Token-family tags, in sequence order.
const TYPE_GLOBAL: usize = 0;
const TYPE_POKEMON: usize = 1;
const TYPE_ATTACK: usize = 2;
const TYPE_TRAINER: usize = 3;
const TYPE_HISTORY: usize = 4;

/// The player model: encoder + factorized heads + value.
#[derive(Module, Debug)]
pub struct RlModel<B: Backend> {
    tables: FrozenTables<B>,
    card_ids: IdEmbedding<B>,
    species_ids: IdEmbedding<B>,
    line_ids: IdEmbedding<B>,
    head_ids: LearnedEmbedding<B>,
    encoder: Encoder<B>,
    heads: PolicyHeads<B>,
    mask_layout: MaskLayout<B>,
    value: ValueHead<B>,
}

/// One forward's outputs.
pub struct ModelOutput<B: Backend> {
    /// Unmasked flat logits, `[batch × ACTION_MASK_DIM]`, bit-aligned with the Part-3 wire.
    pub logits: Tensor<B, 2>,
    /// Masked policy: exact zeros off-mask, sums to 1 over the legal argument bits;
    /// the `ACTION_TYPE` block carries the induced family marginals.
    pub policy: Tensor<B, 2>,
    /// `[batch]`, in `[−1, 1]`.
    pub value: Tensor<B, 1>,
}

impl<B: Backend> RlModel<B> {
    /// Build the model. `embeddings` is the frozen text-encoder artifact
    /// ([`TextEmbeddings::zeros`] until one is plugged in — §1.2.9); everything frozen is
    /// derived deterministically from the pool and `config.init_seed`.
    pub fn new(config: &ModelConfig, embeddings: &TextEmbeddings, device: &B::Device) -> Self {
        let tables = FrozenTables::new(embeddings, config.d_id, config.init_seed, device);
        Self {
            card_ids: IdEmbedding::new(tables.card_init.clone()),
            species_ids: IdEmbedding::new(tables.species_init.clone()),
            line_ids: IdEmbedding::new(tables.line_init.clone()),
            head_ids: LearnedEmbedding::new(
                crate::rl::history::HEAD_TABLE_SIZE,
                config.d_head_emb,
                device,
            ),
            encoder: Encoder::new(config, device),
            heads: PolicyHeads::new(config, device),
            mask_layout: MaskLayout::new(device),
            value: ValueHead::new(config, device),
            tables,
        }
    }

    /// Full forward: assemble the five token families, encode, score every head, mask.
    pub fn forward(&self, input: &ModelInput<B>) -> ModelOutput<B> {
        let h = self
            .encoder
            .forward(self.assemble(input), input.seq_mask.clone());

        let logits = self.heads.forward(&h, input);
        let policy = masked_policy(&self.mask_layout, logits.clone(), input.mask_bits.clone());
        let value = self.value.forward(&h, &input.seq_mask);

        ModelOutput {
            logits,
            policy,
            value,
        }
    }

    /// The encoder's input sequence: the five families projected, tagged, and concatenated in the
    /// [`input`] layout's order.
    ///
    /// Split out of [`Self::forward`] for [`introspect`], which reads attention over the sequence a
    /// step actually ran on. A second assembly written beside it would agree today and drift the
    /// first time either changed, and the drift would show up as a diagnostic quietly describing a
    /// model that was never forwarded.
    fn assemble(&self, input: &ModelInput<B>) -> Tensor<B, 3> {
        let batch = input.batch;

        // Static gathers (identity is an index — §1.2.1 principle 1).
        let gather_static = |table: &Tensor<B, 2>, ids: &Tensor<B, 2, Int>| -> Tensor<B, 3> {
            let [_, width] = table.dims();
            let [_, slots] = ids.dims();
            table
                .clone()
                .select(0, ids.clone().reshape([batch * slots]))
                .reshape([batch, slots, width])
        };

        // Global token: floats ⊕ stadium embedding (shared card table).
        let stadium = self.card_ids.embed(input.stadium_ids.clone());
        let global_token = self.encoder.project_global.forward(Tensor::cat(
            vec![input.global.clone().unsqueeze_dim(1), stadium],
            2,
        )) + self.encoder.type_tag(TYPE_GLOBAL);

        // Pokémon: 3 × id ⊕ static ⊕ dynamic ⊕ tool embedding.
        let pokemon_token = self.encoder.project_pokemon.forward(Tensor::cat(
            vec![
                self.card_ids.embed(input.pokemon_card_ids.clone()),
                self.species_ids.embed(input.pokemon_species_ids.clone()),
                self.line_ids.embed(input.pokemon_line_ids.clone()),
                gather_static(&self.tables.pokemon_static, &input.pokemon_card_ids),
                input.pokemon_features.clone(),
                self.card_ids.embed(input.pokemon_tool_ids.clone()),
            ],
            2,
        )) + self.encoder.type_tag(TYPE_POKEMON);

        // Attack: static (by `(src_card, slot)` row) ⊕ dynamic.
        let attack_token = self.encoder.project_attack.forward(Tensor::cat(
            vec![
                gather_static(&self.tables.attack_static, &input.attack_rows),
                input.attack_features.clone(),
            ],
            2,
        )) + self.encoder.type_tag(TYPE_ATTACK);

        // Trainer: id ⊕ static ⊕ dynamic ⊕ target-set bag (live-gathered on the *trainable*
        // species/line tables, summed — §1.2.6).
        let [_, target_slots] = input.trainer_target_species.dims();
        let bag_width = target_slots / crate::rl::observation::MAX_TRAINER_TOKENS;
        let d_id = self.species_ids.d_id();
        let target_bag = (self.species_ids.embed(input.trainer_target_species.clone())
            + self.line_ids.embed(input.trainer_target_lines.clone()))
        .reshape([
            batch,
            crate::rl::observation::MAX_TRAINER_TOKENS,
            bag_width,
            d_id,
        ])
        .sum_dim(2)
        .reshape([batch, crate::rl::observation::MAX_TRAINER_TOKENS, d_id]);
        let trainer_token = self.encoder.project_trainer.forward(Tensor::cat(
            vec![
                self.card_ids.embed(input.trainer_card_ids.clone()),
                gather_static(&self.tables.trainer_static, &input.trainer_card_ids),
                input.trainer_features.clone(),
                target_bag,
            ],
            2,
        )) + self.encoder.type_tag(TYPE_TRAINER);

        // History: card id ⊕ head embedding ⊕ recency (order lives in recency, not position).
        let history_token = self.encoder.project_history.forward(Tensor::cat(
            vec![
                self.card_ids.embed(input.history_card_ids.clone()),
                self.head_ids.embed(input.history_head_ids.clone()),
                input.history_features.clone(),
            ],
            2,
        )) + self.encoder.type_tag(TYPE_HISTORY);

        Tensor::cat(
            vec![
                global_token,
                pokemon_token,
                attack_token,
                trainer_token,
                history_token,
            ],
            1,
        )
    }

    /// Squared L2 norm of the three learned ID residuals — the §1.2.2 / §1.5.5 regularization
    /// target ("weight-decay on the player embedding residuals").
    pub fn embedding_residual_l2(&self) -> Tensor<B, 1> {
        self.card_ids.residual_l2() + self.species_ids.residual_l2() + self.line_ids.residual_l2()
    }

    /// Visits the trunk: the embeddings and the encoder, everything the policy heads and the value
    /// head both read from.
    ///
    /// The two heads' own weights are excluded because they take gradient from one loss term by
    /// construction — the trunk is where the terms compete, and the only place a per-term gradient
    /// norm compares like with like (`super::train::update::GradNorms`).
    pub fn visit_shared<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.card_ids.visit(visitor);
        self.species_ids.visit(visitor);
        self.line_ids.visit(visitor);
        self.head_ids.visit(visitor);
        self.encoder.visit(visitor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    use crate::rl::action_mask::{project, ActionMask, Regime, ACTION_MASK_DIM};
    use crate::rl::observation::{get_observation, Observation};
    use crate::test_support::init_random_players;
    use crate::{Game, State};

    fn model() -> RlModel<NdArray> {
        RlModel::new(
            &ModelConfig::default(),
            &TextEmbeddings::zeros(),
            &Default::default(),
        )
    }

    fn decision_point(state: &State) -> (Observation, ActionMask) {
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(state, actor, &actions, None, None);
        let mask = project(state, &actions, &observation);
        (observation, mask)
    }

    fn forward_one(model: &RlModel<NdArray>, state: &State) -> (ModelOutput<NdArray>, ActionMask) {
        let (observation, mask) = decision_point(state);
        let input = ModelInput::from_points(
            &[DecisionPoint {
                observation: &observation,
                mask: &mask,
            }],
            &ModelConfig::default(),
            &Default::default(),
        );
        (model.forward(&input), mask)
    }

    /// §1.4.1: the five input widths are the spec's 170 / 854 / 285 / 195 / 82 — modulo the
    /// ability vocabulary, which tracks the engine (§1.2.4).
    #[test]
    fn input_widths_match_the_spec() {
        use crate::rl::static_tables::ABILITY_MECHANIC_DIM;
        let widths = encoder::InputWidths::of(&ModelConfig::default());
        assert_eq!(widths.global, 170);
        assert_eq!(widths.pokemon, 774 + ABILITY_MECHANIC_DIM); // 854 at |mechanics| = 80
        assert_eq!(widths.trainer, 285);
        assert_eq!(widths.attack, 195);
        assert_eq!(widths.history, 82);
    }

    /// §1.4.3 trainable-parameter budget. The frozen tables are constants and must not count.
    #[test]
    fn parameter_count_is_in_the_spec_ballpark() {
        let model = model();
        let params = model.num_params();
        println!("trainable parameters: {params}");
        assert!(
            (1_250_000..=1_650_000).contains(&params),
            "got {params} parameters, spec says ≈ 1.43 M"
        );
    }

    /// The masked policy is a genuine distribution over the legal set and **exactly** zero
    /// elsewhere, at every decision point of random rollouts; every supported bit round-trips
    /// through `ActionMask::select` into an engine action.
    #[test]
    fn masked_policy_is_zero_outside_the_legal_set_on_random_rollouts() {
        let model = model();
        let mut checked = 0;
        for seed in 0..2u64 {
            let mut game = Game::new(init_random_players(), seed);
            while !game.is_game_over() && checked < 12 {
                let state = game.get_state_clone();
                let (_, actions) = state.generate_possible_actions();
                if Regime::of(&state, &actions).needs_policy() {
                    let (output, mask) = forward_one(&model, &state);
                    let wire = mask.to_wire();
                    let probs = output.policy.to_data().to_vec::<f32>().unwrap();
                    assert_eq!(probs.len(), ACTION_MASK_DIM);

                    let type_block = crate::rl::action_mask::Head::ActionType.offset()
                        ..crate::rl::action_mask::Head::ActionType.offset() + 10;
                    let mut total = 0.0;
                    for (bit, probability) in probs.iter().enumerate() {
                        if type_block.contains(&bit) {
                            continue; // family marginals, not probabilities of their own
                        }
                        if wire.bits[bit] {
                            total += probability;
                        } else {
                            assert_eq!(*probability, 0.0, "bit {bit} is not legal");
                        }
                    }
                    assert!((total - 1.0).abs() < 1.0e-4, "sums to {total}");

                    // The argmax legal bit selects an engine action (§1.3.7 round-trip).
                    let (best, _) = probs
                        .iter()
                        .enumerate()
                        .filter(|(bit, _)| wire.bits[*bit] && !type_block.contains(bit))
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .expect("a legal bit exists");
                    let head = crate::rl::action_mask::HEADS
                        .into_iter()
                        .find(|head| (head.offset()..head.offset() + head.dim()).contains(&best))
                        .unwrap();
                    assert!(
                        mask.select(head, best - head.offset()).is_some(),
                        "the selected bit resolves to an action"
                    );
                    checked += 1;
                }
                game.play_tick();
            }
        }
        assert!(checked >= 12, "the rollouts must reach real decisions");
    }

    /// Egocentrism: swapping the two players and observing from the other seat is the *same*
    /// input — same wires, same forward, bit for bit. No player index reaches the model.
    #[test]
    fn a_player_swapped_state_is_the_same_input_and_output() {
        let mut game = Game::new(init_random_players(), 3);
        // Play into the mid-game so boards, discards and energies are populated.
        for _ in 0..40 {
            if game.is_game_over() {
                break;
            }
            game.play_tick();
        }
        let state = game.get_state_clone();
        let (actor, _) = state.generate_possible_actions();

        let mut swapped = state.clone();
        swapped.points.swap(0, 1);
        swapped.energy_zone.swap(0, 1);
        swapped.hands.swap(0, 1);
        swapped.decks.swap(0, 1);
        swapped.discard_piles.swap(0, 1);
        swapped.discard_energies.swap(0, 1);
        swapped.in_play_pokemon.swap(0, 1);
        swapped.has_used_stadium.swap(0, 1);
        swapped.current_player = 1 - swapped.current_player;
        swapped.active_stadium_owner = swapped.active_stadium_owner.map(|owner| 1 - owner);
        swapped.move_generation_stack = state
            .move_generation_stack
            .iter()
            .map(|(frame_actor, actions)| (1 - frame_actor, actions.clone()))
            .collect();

        let (swapped_actor, _) = swapped.generate_possible_actions();
        assert_eq!(swapped_actor, 1 - actor, "the frame follows its actor");

        let (observation, mask) = decision_point(&state);
        let (observation_swapped, mask_swapped) = decision_point(&swapped);

        assert_eq!(observation.to_wire(), observation_swapped.to_wire());
        assert_eq!(mask.to_wire().bits, mask_swapped.to_wire().bits);
        assert_eq!(mask.regime, mask_swapped.regime);

        let model = model();
        let (out_a, _) = forward_one(&model, &state);
        let (out_b, _) = forward_one(&model, &swapped);
        assert_eq!(
            out_a.logits.to_data().to_vec::<f32>().unwrap(),
            out_b.logits.to_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            out_a.value.to_data().to_vec::<f32>().unwrap(),
            out_b.value.to_data().to_vec::<f32>().unwrap()
        );
    }

    /// The forward is a pure function: same input, bit-identical output.
    #[test]
    fn the_forward_is_deterministic() {
        let model = model();
        let game = Game::new(init_random_players(), 11);
        let state = game.get_state_clone();
        let (first, _) = forward_one(&model, &state);
        let (second, _) = forward_one(&model, &state);
        assert_eq!(
            first.logits.to_data().to_vec::<f32>().unwrap(),
            second.logits.to_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            first.policy.to_data().to_vec::<f32>().unwrap(),
            second.policy.to_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            first.value.to_data().to_vec::<f32>().unwrap(),
            second.value.to_data().to_vec::<f32>().unwrap()
        );
        let value = first.value.into_scalar();
        assert!((-1.0..=1.0).contains(&value));
    }

    /// The residual regularizer starts at exactly zero: the model *is* its meta-neutral init.
    #[test]
    fn the_embedding_residual_starts_at_zero() {
        assert_eq!(model().embedding_residual_l2().into_scalar(), 0.0);
    }

    /// One decision point cloned `batch` times, timed over enough runs to stabilize.
    ///
    /// **Both** `policy` and `value` are read back: they are the two outputs the sampler consumes,
    /// and on the lazily-fused CubeCL backends (wgpu / CUDA) an output nobody reads is never
    /// materialized. Syncing on `value` alone would elide [`PolicyHeads::forward`] and
    /// [`masked_policy`] — `value` does not depend on the logits — and report an encoder-only
    /// figure as if it were the model's.
    fn sweep_forward<B: Backend>(label: &str, device: &B::Device) -> std::time::Duration {
        let model = RlModel::<B>::new(&ModelConfig::default(), &TextEmbeddings::zeros(), device);
        let game = Game::new(init_random_players(), 5);
        let state = game.get_state_clone();
        let (observation, mask) = decision_point(&state);
        let config = ModelConfig::default();

        let mut single = std::time::Duration::ZERO;
        for batch in [1usize, 4, 16, 64, 128, 256] {
            let points: Vec<DecisionPoint> = (0..batch)
                .map(|_| DecisionPoint {
                    observation: &observation,
                    mask: &mask,
                })
                .collect();
            let input = ModelInput::from_points(&points, &config, device);
            let run = || {
                let output = model.forward(&input);
                let _ = output.policy.into_data();
                let _ = output.value.into_data();
            };
            for _ in 0..5 {
                run();
            }
            let runs = (256 / batch).clamp(3, 30) as u32;
            let start = std::time::Instant::now();
            for _ in 0..runs {
                run();
            }
            let per_sample = start.elapsed() / runs / batch as u32;
            if batch == 1 {
                single = per_sample;
            }
            println!(
                "{label} forward  batch {batch:>3}: {per_sample:>12?}/sample  \
                 ({:>8.0} samples/s)",
                1.0 / per_sample.as_secs_f64()
            );
        }
        single
    }

    /// A training-step proxy: forward + backward through every head, the encoder and the
    /// embedding residuals. No optimizer step — this isolates the autodiff overhead Part 5 pays.
    #[cfg(any(feature = "rl-model-wgpu", feature = "rl-model-cuda"))]
    fn sweep_train<B: burn::tensor::backend::AutodiffBackend>(label: &str, device: &B::Device) {
        let model = RlModel::<B>::new(&ModelConfig::default(), &TextEmbeddings::zeros(), device);
        let game = Game::new(init_random_players(), 5);
        let state = game.get_state_clone();
        let (observation, mask) = decision_point(&state);
        let config = ModelConfig::default();

        // Autodiff keeps every intermediate alive, so this is where a 4 GB card runs out first —
        // the sweep goes past the forward sweep's range to locate the knee rather than assume it.
        for batch in [8usize, 32, 64, 128, 256, 512] {
            let points: Vec<DecisionPoint> = (0..batch)
                .map(|_| DecisionPoint {
                    observation: &observation,
                    mask: &mask,
                })
                .collect();
            let input = ModelInput::from_points(&points, &config, device);
            let step = || {
                let output = model.forward(&input);
                // `policy` is in the loss so the masked softmax is part of the measured graph —
                // the real MMD objective differentiates through it, and leaving it out would
                // shrink the backward the same way syncing on `value` alone shrinks the forward.
                let loss = output.logits.sum()
                    + output.policy.sum()
                    + output.value.sum()
                    + model.embedding_residual_l2();
                let _gradients = loss.backward();
            };
            for _ in 0..3 {
                step();
            }
            let runs = (64 / batch).clamp(2, 8) as u32;
            let start = std::time::Instant::now();
            for _ in 0..runs {
                step();
            }
            let per_sample = start.elapsed() / runs / batch as u32;
            println!(
                "{label} fwd+bwd  batch {batch:>3}: {per_sample:>12?}/sample  \
                 ({:>8.0} samples/s)",
                1.0 / per_sample.as_secs_f64()
            );
        }
    }

    /// Every state of `games` random rollouts at which a policy is actually consulted, plus the
    /// decision count per game — the figure §1.4.3's games/s budget rests on and that nothing
    /// measured until now.
    fn collect_decision_states(games: u64, cap: usize) -> (Vec<State>, f64) {
        let mut states = Vec::new();
        let mut decisions = 0usize;
        for seed in 0..games {
            let mut game = Game::new(init_random_players(), seed);
            while !game.is_game_over() {
                let state = game.get_state_clone();
                let (_, actions) = state.generate_possible_actions();
                if Regime::of(&state, &actions).needs_policy() {
                    decisions += 1;
                    if states.len() < cap {
                        states.push(state);
                    }
                }
                game.play_tick();
            }
        }
        (states, decisions as f64 / games as f64)
    }

    /// The **end-to-end** per-decision budget of the self-play loop (§1.5.5), which the forward
    /// sweeps do not cover: they assemble `ModelInput` once, outside the timed loop, from a single
    /// observation cloned `batch` times. Production pays, per decision, the observation build (with
    /// the §1.2.5 threat matrix — the spec's own "heaviest computation"), the mask projection, the
    /// wire flattening and the host→device copy, on top of the forward.
    ///
    /// `generate_possible_actions` is timed but reported separately: the engine calls it during
    /// `play_tick` regardless, so it is shared cost, not RL overhead.
    fn end_to_end<B: Backend>(label: &str, device: &B::Device, batch: usize) {
        let (states, decisions_per_game) = collect_decision_states(8, 256);
        assert!(!states.is_empty(), "the rollouts must reach real decisions");
        let n = states.len() as u32;

        let start = std::time::Instant::now();
        let enumerations: Vec<_> = states
            .iter()
            .map(|state| state.generate_possible_actions())
            .collect();
        let enumerate = start.elapsed() / n;

        let start = std::time::Instant::now();
        let observations: Vec<Observation> = states
            .iter()
            .zip(&enumerations)
            .map(|(state, (actor, actions))| get_observation(state, *actor, actions, None, None))
            .collect();
        let observe = start.elapsed() / n;

        let start = std::time::Instant::now();
        let masks: Vec<ActionMask> = states
            .iter()
            .zip(&enumerations)
            .zip(&observations)
            .map(|((state, (_, actions)), observation)| project(state, actions, observation))
            .collect();
        let project_mask = start.elapsed() / n;

        let config = ModelConfig::default();
        let model = RlModel::<B>::new(&config, &TextEmbeddings::zeros(), device);
        let points: Vec<DecisionPoint> = observations
            .iter()
            .zip(&masks)
            .map(|(observation, mask)| DecisionPoint { observation, mask })
            .collect();

        // Real batches: `batch` distinct decision points, not one cloned.
        let chunks: Vec<&[DecisionPoint]> = points.chunks(batch).collect();
        let samples = points.len() as u32;

        for _ in 0..3 {
            let input = ModelInput::from_points(chunks[0], &config, device);
            let output = model.forward(&input);
            let _ = output.policy.into_data();
            let _ = output.value.into_data();
        }

        let start = std::time::Instant::now();
        let inputs: Vec<ModelInput<B>> = chunks
            .iter()
            .map(|chunk| ModelInput::from_points(chunk, &config, device))
            .collect();
        let assemble = start.elapsed() / samples;

        // Per-chunk, then a second pass over the same inputs: on a GPU backend the two differ if
        // the first pass is paying a one-time cost (autotune, memory-pool growth) that the warmup
        // did not cover, rather than a genuine per-decision cost.
        let mut pass = |label: &str| -> std::time::Duration {
            let start = std::time::Instant::now();
            let mut per_chunk = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let chunk_start = std::time::Instant::now();
                let output = model.forward(input);
                let _ = output.policy.into_data();
                let _ = output.value.into_data();
                per_chunk.push(chunk_start.elapsed());
            }
            let total = start.elapsed();
            println!("  {label} per chunk: {per_chunk:?}");
            total / samples
        };
        let forward_cold = pass("forward pass 1");
        let forward = pass("forward pass 2");
        println!("  forward pass 1 {forward_cold:?}/decision vs pass 2 {forward:?}/decision");

        let rl_only = observe + project_mask + assemble + forward;
        let total = enumerate + rl_only;
        println!(
            "\n{label} end-to-end, batch {batch}, {} decision points, \
             {decisions_per_game:.1} policy decisions/game\n  \
             generate_possible_actions {enumerate:>12?}/decision  (shared with the engine)\n  \
             get_observation           {observe:>12?}/decision\n  \
             project (mask)            {project_mask:>12?}/decision\n  \
             ModelInput::from_points   {assemble:>12?}/decision\n  \
             forward + readback        {forward:>12?}/decision\n  \
             RL cost                   {rl_only:>12?}/decision  \
             ({:.0} decisions/s → {:.1} games/s)\n  \
             incl. enumeration         {total:>12?}/decision  \
             ({:.0} decisions/s → {:.1} games/s)",
            states.len(),
            1.0 / rl_only.as_secs_f64(),
            1.0 / rl_only.as_secs_f64() / decisions_per_game,
            1.0 / total.as_secs_f64(),
            1.0 / total.as_secs_f64() / decisions_per_game,
        );
    }

    /// End-to-end budget, CPU. Release only:
    /// `cargo test --release --features rl-model -- --ignored end_to_end --nocapture`.
    #[test]
    #[ignore = "budget is only meaningful in release; run with --release -- --ignored"]
    fn end_to_end_decision_budget() {
        end_to_end::<NdArray>("ndarray", &Default::default(), 64);
    }

    /// End-to-end budget on CUDA — the configuration §1.4.3 draws its games/s conclusion from.
    #[cfg(feature = "rl-model-cuda")]
    #[test]
    #[ignore = "GPU budget; run with --release --features rl-model-cuda -- --ignored"]
    fn end_to_end_decision_budget_cuda() {
        end_to_end::<burn::backend::Cuda>("cuda   ", &Default::default(), 64);
    }

    /// Inference throughput at batch 64, on a model and input built fresh at call time — so the
    /// figure reflects whatever state the *process* is in, which is the whole point of calling it
    /// before and after something else.
    #[cfg(any(feature = "rl-model-wgpu", feature = "rl-model-cuda"))]
    fn measure_inference<B: Backend>(label: &str, tag: &str, device: &B::Device) {
        let config = ModelConfig::default();
        let game = Game::new(init_random_players(), 5);
        let state = game.get_state_clone();
        let (observation, mask) = decision_point(&state);
        let model = RlModel::<B>::new(&config, &TextEmbeddings::zeros(), device);
        let points: Vec<DecisionPoint> = (0..64)
            .map(|_| DecisionPoint {
                observation: &observation,
                mask: &mask,
            })
            .collect();
        let input = ModelInput::<B>::from_points(&points, &config, device);
        let once = || {
            let output = model.forward(&input);
            let _ = output.policy.into_data();
            let _ = output.value.into_data();
        };
        for _ in 0..3 {
            once();
        }
        let start = std::time::Instant::now();
        for _ in 0..5 {
            once();
        }
        let per_sample = start.elapsed() / 5 / 64;
        println!(
            "{label} inference b64 {tag:>26}: {per_sample:>12?}/sample ({:>7.0}/s)",
            1.0 / per_sample.as_secs_f64()
        );
    }

    /// Isolates what actually degrades inference in a long-lived process on a 4 GB card: the
    /// large-batch **forward** sweep, or the autodiff steps? [`train_then_infer`] answers the
    /// second, this one the first.
    #[cfg(feature = "rl-model-cuda")]
    #[test]
    #[ignore = "GPU interaction; run with --release --features rl-model-cuda -- --ignored"]
    fn cuda_inference_after_a_large_forward_sweep() {
        type Gpu = burn::backend::Cuda;
        let device = Default::default();
        measure_inference::<Gpu>("cuda   ", "baseline", &device);
        sweep_forward::<Gpu>("cuda   ", &device);
        measure_inference::<Gpu>("cuda   ", "after forward sweep to b256", &device);
    }

    /// Does a training step poison the inference that follows it **in the same process**?
    ///
    /// §1.5.5 runs a synchronous single learner: collect (inference) → one MMD step (fwd + bwd) →
    /// repeat, all in one process on one device. Measuring the two sweeps separately hides any
    /// interaction between them; running `cuda_latency_on_a_full_board` before the end-to-end
    /// budget in one process made inference ≈ 25× slower and it never recovered. This isolates the
    /// effect and asks the question that actually matters: is the spec's "training batch ≤ 64"
    /// rule enough to keep the collect phase fast?
    #[cfg(any(feature = "rl-model-wgpu", feature = "rl-model-cuda"))]
    fn train_then_infer<B: Backend>(label: &str, device: &B::Device) {
        use burn::backend::Autodiff;

        let config = ModelConfig::default();
        let game = Game::new(init_random_players(), 5);
        let state = game.get_state_clone();
        let (observation, mask) = decision_point(&state);

        let points = |count: usize| -> Vec<DecisionPoint> {
            (0..count)
                .map(|_| DecisionPoint {
                    observation: &observation,
                    mask: &mask,
                })
                .collect()
        };

        let measure = |tag: &str| measure_inference::<B>(label, tag, device);

        measure("baseline");
        for batch in [8usize, 64, 128] {
            let train_model =
                RlModel::<Autodiff<B>>::new(&config, &TextEmbeddings::zeros(), device);
            let train_input =
                ModelInput::<Autodiff<B>>::from_points(&points(batch), &config, device);
            for _ in 0..3 {
                let output = train_model.forward(&train_input);
                let loss = output.logits.sum()
                    + output.policy.sum()
                    + output.value.sum()
                    + train_model.embedding_residual_l2();
                let _gradients = loss.backward();
            }
            measure(&format!("after autodiff b{batch}"));
        }
    }

    /// Inference-after-training on CUDA. Release only:
    /// `cargo test --release --features rl-model-cuda -- --ignored survives_a_training_step --nocapture`.
    #[cfg(feature = "rl-model-cuda")]
    #[test]
    #[ignore = "GPU interaction; run with --release --features rl-model-cuda -- --ignored"]
    fn cuda_inference_survives_a_training_step() {
        train_then_infer::<burn::backend::Cuda>("cuda   ", &Default::default());
    }

    /// §1.4.3 latency, CPU reference point. The original "sub-ms forward, CPU-viable" claim was
    /// falsified by this measurement (≈ 42 ms/forward — a v1 forward is ≈ 0.9 GFLOP and the
    /// NdArray GEMM stream sustains ≈ 20 GFLOPS); the spec now states measured budgets instead.
    /// Run explicitly, in release:
    /// `cargo test --release --features rl-model -- --ignored forward_latency --nocapture`.
    #[test]
    #[ignore = "latency is only meaningful in release; run with --release -- --ignored"]
    fn forward_latency_on_a_full_board() {
        let single = sweep_forward::<NdArray>("ndarray", &Default::default());
        assert!(
            single < std::time::Duration::from_millis(150),
            "single forward took {single:?} — an order of magnitude beyond the measured baseline \
             (≈ 42 ms), CPU viability regressed"
        );
    }

    /// GPU sweep, wgpu backend (Vulkan/DX12 — no CUDA toolkit needed). Release only:
    /// `cargo test --release --features rl-model-wgpu -- --ignored wgpu_latency --nocapture`.
    #[cfg(feature = "rl-model-wgpu")]
    #[test]
    #[ignore = "GPU latency; run with --release --features rl-model-wgpu -- --ignored"]
    fn wgpu_latency_on_a_full_board() {
        type Gpu = burn::backend::Wgpu;
        let device = Default::default();
        sweep_forward::<Gpu>("wgpu   ", &device);
        sweep_train::<burn::backend::Autodiff<Gpu>>("wgpu   ", &device);
    }

    /// GPU sweep, CUDA backend (requires the CUDA toolkit for NVRTC). Release only:
    /// `cargo test --release --features rl-model-cuda -- --ignored cuda_latency --nocapture`.
    #[cfg(feature = "rl-model-cuda")]
    #[test]
    #[ignore = "GPU latency; run with --release --features rl-model-cuda -- --ignored"]
    fn cuda_latency_on_a_full_board() {
        type Gpu = burn::backend::Cuda;
        let device = Default::default();
        sweep_forward::<Gpu>("cuda   ", &device);
        sweep_train::<burn::backend::Autodiff<Gpu>>("cuda   ", &device);
    }
}
