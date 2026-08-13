//! Batched model input: Part-2 observations + Part-3 masks, flattened to tensors.
//!
//! Besides the wire banks themselves, the assembly precomputes what the heads need to stay
//! aligned with the mask's index spaces (§1.3.8):
//!
//! - **Self-scoped pointer rows** — for each self head, the *sequence positions* of the allied
//!   subsequence of its bank, in bank order. A `PLACE` index `row × 4 + slot` then names encoder
//!   row `self_pokemon_rows[row]` directly, exactly as the mask promised.
//! - **Board slot rows** — sequence positions of the 8 board tokens by `(role, slot)`, for the
//!   slot-indexed heads.
//! - **Candidate encodings** (§1.3.5) — per `CANDIDATE_PTR` entry: the action-family id (the same
//!   `discriminant(SimpleAction)` table the History token uses) and the sequence rows of the
//!   entities the candidate references, pooled in-model. Reference extraction is best-effort by
//!   design: it shapes the *encoding* of a candidate, never its legality.
//!
//! Padded positions in every row map point at row 0 (the global token); they are only ever read
//! behind a mask bit that is false, so the gathered garbage never reaches a probability.

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::actions::SimpleAction;
use crate::models::Card;

use crate::rl::action_mask::{
    ActionMask, ActionMaskWire, Head, Regime, ATTACK_SELF, POKEMON_SELF, TRAINER_SELF,
};
use crate::rl::damage::BOARD_SLOTS;
use crate::rl::history::{head_id, HISTORY_LEN};
use crate::rl::ids::card_index;
use crate::rl::observation::{
    Observation, TokenZone, MAX_ATTACK_TOKENS, MAX_POKEMON_TOKENS, MAX_TRAINER_TARGET_IDS,
    MAX_TRAINER_TOKENS,
};
use crate::rl::recover::{catch, EnginePanic};
use crate::rl::static_tables::attack_table_row;

use super::config::ModelConfig;

/// Sequence layout: `[global, Pokémon×40, Attack×32, Trainer×40, History×20]`.
pub const GLOBAL_ROW: usize = 0;
pub const POKEMON_OFFSET: usize = 1;
pub const ATTACK_OFFSET: usize = POKEMON_OFFSET + MAX_POKEMON_TOKENS;
pub const TRAINER_OFFSET: usize = ATTACK_OFFSET + MAX_ATTACK_TOKENS;
pub const HISTORY_OFFSET: usize = TRAINER_OFFSET + MAX_TRAINER_TOKENS;
/// `N ≤ 133` (§1.4).
pub const SEQ_LEN: usize = HISTORY_OFFSET + HISTORY_LEN;

/// One decision point: the observation and the mask of the same frame, same actor.
pub struct DecisionPoint<'a> {
    pub observation: &'a Observation,
    pub mask: &'a ActionMask,
}

/// The batched tensors one forward consumes.
pub struct ModelInput<B: Backend> {
    pub batch: usize,
    // Family banks.
    pub global: Tensor<B, 2>,
    pub stadium_ids: Tensor<B, 2, Int>,
    pub pokemon_features: Tensor<B, 3>,
    pub pokemon_card_ids: Tensor<B, 2, Int>,
    pub pokemon_species_ids: Tensor<B, 2, Int>,
    pub pokemon_line_ids: Tensor<B, 2, Int>,
    pub pokemon_tool_ids: Tensor<B, 2, Int>,
    pub attack_features: Tensor<B, 3>,
    /// Row into the frozen attack table: `attack_table_row(src_card_id, attack_slot)`.
    pub attack_rows: Tensor<B, 2, Int>,
    pub trainer_features: Tensor<B, 3>,
    pub trainer_card_ids: Tensor<B, 2, Int>,
    /// `[batch × (40 · MAX_TRAINER_TARGET_IDS)]` species / line halves of the target-set bags.
    pub trainer_target_species: Tensor<B, 2, Int>,
    pub trainer_target_lines: Tensor<B, 2, Int>,
    pub history_features: Tensor<B, 3>,
    pub history_card_ids: Tensor<B, 2, Int>,
    pub history_head_ids: Tensor<B, 2, Int>,
    /// `[batch × SEQ_LEN]`, 1.0 for real tokens (global always real).
    pub seq_mask: Tensor<B, 2>,
    // Pointer row maps (sequence positions).
    pub self_pokemon_rows: Tensor<B, 2, Int>,
    pub self_attack_rows: Tensor<B, 2, Int>,
    pub self_trainer_rows: Tensor<B, 2, Int>,
    pub board_self_rows: Tensor<B, 2, Int>,
    pub board_opp_rows: Tensor<B, 2, Int>,
    // Candidate encodings (§1.3.5), capped at `max_scored_candidates`.
    pub candidate_type_ids: Tensor<B, 2, Int>,
    pub candidate_ref_rows: Tensor<B, 3, Int>,
    pub candidate_ref_mask: Tensor<B, 3>,
    /// The flat Part-3 mask, 1.0 on set bits.
    pub mask_bits: Tensor<B, 2>,
    /// The per-sample wire masks, kept for selection / round-trips on the CPU side.
    pub wires: Vec<ActionMaskWire>,
    pub regimes: Vec<Regime>,
}

impl<B: Backend> ModelInput<B> {
    /// Flatten a batch of decision points. Panics if an observation and its mask disagree on the
    /// actor, or if a candidate lands beyond `max_scored_candidates` (§1.3.8: real frames stay two
    /// orders of magnitude below the wire cap).
    pub fn from_points(
        points: &[DecisionPoint<'_>],
        config: &ModelConfig,
        device: &B::Device,
    ) -> Self {
        let batch = points.len();
        assert!(batch > 0, "an empty batch has no shapes");
        let scored = config.max_scored_candidates;
        let refs = config.max_candidate_refs;

        let mut builder = Builder::with_capacity(batch, scored, refs);
        for point in points {
            assert_eq!(
                point.observation.perspective, point.mask.actor,
                "a decision point is observed by its actor (§1.3.7 invariant 5)"
            );
            builder.push(point, scored, refs);
        }
        builder.into_tensors(batch, scored, refs, device)
    }

    /// [`ModelInput::from_points`] with the panic attributed to the point that raised it.
    ///
    /// The encoder asserts the wire caps (§1.3.8), and a frame that overflows one is the same kind
    /// of event as an engine panic: one game found a position nothing anticipated, out of the
    /// millions a run plays. But a rollout batches 128 of them into a single call, so an
    /// unattributed panic costs the whole batch and — worse — says nothing about *which* frame to
    /// go and read. So the batch is encoded under one guard, and only a failure pays for the
    /// row-by-row re-encoding that names the offender.
    ///
    /// Re-encoding is exact rather than merely indicative: [`Builder::push`] reads one point and
    /// appends to the builder, and no assertion it makes looks at what a previous point pushed. A
    /// panic that then reproduces nowhere alone came from [`Builder::into_tensors`] — a shape bug,
    /// which is the run's problem and not one frame's.
    pub fn try_from_points(
        points: &[DecisionPoint<'_>],
        config: &ModelConfig,
        device: &B::Device,
    ) -> Result<Self, EncodeFault> {
        let batch = match catch(|| Self::from_points(points, config, device)) {
            Ok(input) => return Ok(input),
            Err(panic) => panic,
        };
        for (row, point) in points.iter().enumerate() {
            let alone = std::slice::from_ref(point);
            if let Err(panic) = catch(|| Self::from_points(alone, config, device)) {
                return Err(EncodeFault::Row { row, panic });
            }
        }
        Err(EncodeFault::Batch(batch))
    }
}

/// An encoding that panicked, and how much of the batch it condemns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeFault {
    /// The point at `row` panics on its own — drop that frame and encode the rest.
    Row { row: usize, panic: EnginePanic },
    /// No single point reproduces it, so nothing here is worth retrying without it.
    Batch(EnginePanic),
}

impl std::fmt::Display for EncodeFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeFault::Row { row, panic } => write!(f, "row {row}: {panic}"),
            EncodeFault::Batch(panic) => write!(f, "the batch as a whole: {panic}"),
        }
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> ModelInput<B> {
    /// The same batch on the inner backend, so a *second* model can be forwarded over it without
    /// building an autodiff graph.
    ///
    /// This exists for §1.5.1's magnet: the KL term needs the magnet's policy at the BR's own
    /// frames, as a constant. The two alternatives are both worse on the 4 GB card §1.4.3 sizes the
    /// run for. Re-assembling the batch at [`ModelInput::from_points`] pays the flattening and the
    /// host→device copy twice per micro-batch; forwarding the magnet on the autodiff backend and
    /// detaching afterwards builds a graph that is retained for the length of the forward and then
    /// thrown away, which is the *activation* memory `micro_batch` exists to bound. Reading the
    /// tensors' inner view costs neither: it is the same device buffers, minus the autodiff nodes.
    pub fn to_inner(&self) -> ModelInput<B::InnerBackend> {
        ModelInput {
            batch: self.batch,
            global: self.global.clone().inner(),
            stadium_ids: self.stadium_ids.clone().inner(),
            pokemon_features: self.pokemon_features.clone().inner(),
            pokemon_card_ids: self.pokemon_card_ids.clone().inner(),
            pokemon_species_ids: self.pokemon_species_ids.clone().inner(),
            pokemon_line_ids: self.pokemon_line_ids.clone().inner(),
            pokemon_tool_ids: self.pokemon_tool_ids.clone().inner(),
            attack_features: self.attack_features.clone().inner(),
            attack_rows: self.attack_rows.clone().inner(),
            trainer_features: self.trainer_features.clone().inner(),
            trainer_card_ids: self.trainer_card_ids.clone().inner(),
            trainer_target_species: self.trainer_target_species.clone().inner(),
            trainer_target_lines: self.trainer_target_lines.clone().inner(),
            history_features: self.history_features.clone().inner(),
            history_card_ids: self.history_card_ids.clone().inner(),
            history_head_ids: self.history_head_ids.clone().inner(),
            seq_mask: self.seq_mask.clone().inner(),
            self_pokemon_rows: self.self_pokemon_rows.clone().inner(),
            self_attack_rows: self.self_attack_rows.clone().inner(),
            self_trainer_rows: self.self_trainer_rows.clone().inner(),
            board_self_rows: self.board_self_rows.clone().inner(),
            board_opp_rows: self.board_opp_rows.clone().inner(),
            candidate_type_ids: self.candidate_type_ids.clone().inner(),
            candidate_ref_rows: self.candidate_ref_rows.clone().inner(),
            candidate_ref_mask: self.candidate_ref_mask.clone().inner(),
            mask_bits: self.mask_bits.clone().inner(),
            wires: self.wires.clone(),
            regimes: self.regimes.clone(),
        }
    }
}

#[derive(Default)]
struct Builder {
    global: Vec<f32>,
    stadium_ids: Vec<i64>,
    pokemon_features: Vec<f32>,
    pokemon_card_ids: Vec<i64>,
    pokemon_species_ids: Vec<i64>,
    pokemon_line_ids: Vec<i64>,
    pokemon_tool_ids: Vec<i64>,
    attack_features: Vec<f32>,
    attack_rows: Vec<i64>,
    trainer_features: Vec<f32>,
    trainer_card_ids: Vec<i64>,
    trainer_target_species: Vec<i64>,
    trainer_target_lines: Vec<i64>,
    history_features: Vec<f32>,
    history_card_ids: Vec<i64>,
    history_head_ids: Vec<i64>,
    seq_mask: Vec<f32>,
    self_pokemon_rows: Vec<i64>,
    self_attack_rows: Vec<i64>,
    self_trainer_rows: Vec<i64>,
    board_self_rows: Vec<i64>,
    board_opp_rows: Vec<i64>,
    candidate_type_ids: Vec<i64>,
    candidate_ref_rows: Vec<i64>,
    candidate_ref_mask: Vec<f32>,
    mask_bits: Vec<f32>,
    wires: Vec<ActionMaskWire>,
    regimes: Vec<Regime>,
}

impl Builder {
    fn with_capacity(batch: usize, scored: usize, refs: usize) -> Self {
        let mut builder = Self::default();
        builder
            .global
            .reserve(batch * crate::rl::observation::GLOBAL_DIM);
        builder.candidate_ref_rows.reserve(batch * scored * refs);
        builder
    }

    fn push(&mut self, point: &DecisionPoint<'_>, scored: usize, refs: usize) {
        let observation = point.observation;
        let wire = observation.to_wire();

        self.global.extend(&wire.global);
        self.stadium_ids.push(wire.stadium_id as i64);
        self.pokemon_features.extend(&wire.pokemon);
        for token in wire.pokemon_indices.chunks_exact(4) {
            self.pokemon_card_ids.push(token[0] as i64);
            self.pokemon_species_ids.push(token[1] as i64);
            self.pokemon_line_ids.push(token[2] as i64);
            self.pokemon_tool_ids.push(token[3] as i64);
        }
        self.attack_features.extend(&wire.attack);
        for slot in 0..MAX_ATTACK_TOKENS {
            let row = observation
                .attacks
                .get(slot)
                .map(|token| attack_table_row(token.src_card_id, token.attack_slot))
                .unwrap_or(0);
            self.attack_rows.push(row as i64);
        }
        self.trainer_features.extend(&wire.trainer);
        self.trainer_card_ids
            .extend(wire.trainer_indices.iter().map(|id| *id as i64));
        for pair in wire.trainer_target_indices.chunks_exact(2) {
            self.trainer_target_species.push(pair[0] as i64);
            self.trainer_target_lines.push(pair[1] as i64);
        }
        self.history_features.extend(&wire.history);
        for pair in wire.history_indices.chunks_exact(2) {
            self.history_card_ids.push(pair[0] as i64);
            self.history_head_ids.push(pair[1] as i64);
        }

        self.seq_mask.push(1.0); // global
        for mask in [
            &wire.pokemon_mask[..],
            &wire.attack_mask[..],
            &wire.trainer_mask[..],
            &wire.history_mask[..],
        ] {
            self.seq_mask
                .extend(mask.iter().map(|set| if *set { 1.0 } else { 0.0 }));
        }

        // Self-scoped pointer rows (§1.3.8): the allied subsequences, in bank order.
        let allied_rows = |allied: Vec<usize>, cap: usize, bank: &str| -> Vec<i64> {
            assert!(
                allied.len() <= cap,
                "self {bank} slice overflows its {cap} rows"
            );
            let mut rows: Vec<i64> = allied.into_iter().map(|row| row as i64).collect();
            rows.resize(cap, GLOBAL_ROW as i64);
            rows
        };
        self.self_pokemon_rows.extend(allied_rows(
            observation
                .pokemon
                .iter()
                .enumerate()
                .filter(|(_, token)| token.allied)
                .map(|(row, _)| POKEMON_OFFSET + row)
                .collect(),
            POKEMON_SELF,
            "Pokémon",
        ));
        self.self_attack_rows.extend(allied_rows(
            observation
                .attacks
                .iter()
                .enumerate()
                .filter(|(_, token)| token.allied)
                .map(|(row, _)| ATTACK_OFFSET + row)
                .collect(),
            ATTACK_SELF,
            "Attack",
        ));
        self.self_trainer_rows.extend(allied_rows(
            observation
                .trainers
                .iter()
                .enumerate()
                .filter(|(_, token)| token.allied)
                .map(|(row, _)| TRAINER_OFFSET + row)
                .collect(),
            TRAINER_SELF,
            "Trainer",
        ));
        for role_allied in [true, false] {
            let mut rows = [GLOBAL_ROW as i64; BOARD_SLOTS];
            for (row, token) in observation.pokemon.iter().enumerate() {
                if token.allied == role_allied && token.zone == TokenZone::Board {
                    if let Some(slot) = token.slot {
                        rows[slot] = (POKEMON_OFFSET + row) as i64;
                    }
                }
            }
            if role_allied {
                self.board_self_rows.extend(rows);
            } else {
                self.board_opp_rows.extend(rows);
            }
        }

        // Candidate encodings.
        let mut candidate_types = vec![0i64; scored];
        let mut candidate_refs = vec![GLOBAL_ROW as i64; scored * refs];
        let mut candidate_masks = vec![0.0f32; scored * refs];
        for entry in point
            .mask
            .entries
            .iter()
            .filter(|entry| entry.head == Head::CandidatePtr)
        {
            assert!(
                entry.index < scored,
                "candidate {} beyond the {scored} scored positions — widen \
                 max_scored_candidates (§1.3.8: widest observed frame is 20)",
                entry.index
            );
            candidate_types[entry.index] = head_id(&entry.action) as i64;
            let rows = candidate_reference_rows(&entry.action, observation);
            for (position, row) in rows.into_iter().take(refs).enumerate() {
                candidate_refs[entry.index * refs + position] = row as i64;
                candidate_masks[entry.index * refs + position] = 1.0;
            }
        }
        self.candidate_type_ids.extend(candidate_types);
        self.candidate_ref_rows.extend(candidate_refs);
        self.candidate_ref_mask.extend(candidate_masks);

        let wire_mask = point.mask.to_wire();
        self.mask_bits.extend(
            wire_mask
                .bits
                .iter()
                .map(|set| if *set { 1.0 } else { 0.0 }),
        );
        self.regimes.push(wire_mask.regime);
        self.wires.push(wire_mask);
    }

    fn into_tensors<B: Backend>(
        self,
        batch: usize,
        scored: usize,
        refs: usize,
        device: &B::Device,
    ) -> ModelInput<B> {
        let floats = |data: Vec<f32>, shape: [usize; 2]| {
            Tensor::<B, 2>::from_data(TensorData::new(data, shape), device)
        };
        let floats3 = |data: Vec<f32>, shape: [usize; 3]| {
            Tensor::<B, 3>::from_data(TensorData::new(data, shape), device)
        };
        let ints = |data: Vec<i64>, shape: [usize; 2]| {
            Tensor::<B, 2, Int>::from_data(TensorData::new(data, shape), device)
        };

        use crate::rl::action_mask::ACTION_MASK_DIM;
        use crate::rl::history::HISTORY_DYNAMIC_DIM;
        use crate::rl::observation::{
            ATTACK_DYNAMIC_DIM, GLOBAL_DIM, POKEMON_DYNAMIC_DIM, TRAINER_DYNAMIC_DIM,
        };

        ModelInput {
            batch,
            global: floats(self.global, [batch, GLOBAL_DIM]),
            stadium_ids: ints(self.stadium_ids, [batch, 1]),
            pokemon_features: floats3(
                self.pokemon_features,
                [batch, MAX_POKEMON_TOKENS, POKEMON_DYNAMIC_DIM],
            ),
            pokemon_card_ids: ints(self.pokemon_card_ids, [batch, MAX_POKEMON_TOKENS]),
            pokemon_species_ids: ints(self.pokemon_species_ids, [batch, MAX_POKEMON_TOKENS]),
            pokemon_line_ids: ints(self.pokemon_line_ids, [batch, MAX_POKEMON_TOKENS]),
            pokemon_tool_ids: ints(self.pokemon_tool_ids, [batch, MAX_POKEMON_TOKENS]),
            attack_features: floats3(
                self.attack_features,
                [batch, MAX_ATTACK_TOKENS, ATTACK_DYNAMIC_DIM],
            ),
            attack_rows: ints(self.attack_rows, [batch, MAX_ATTACK_TOKENS]),
            trainer_features: floats3(
                self.trainer_features,
                [batch, MAX_TRAINER_TOKENS, TRAINER_DYNAMIC_DIM],
            ),
            trainer_card_ids: ints(self.trainer_card_ids, [batch, MAX_TRAINER_TOKENS]),
            trainer_target_species: ints(
                self.trainer_target_species,
                [batch, MAX_TRAINER_TOKENS * MAX_TRAINER_TARGET_IDS],
            ),
            trainer_target_lines: ints(
                self.trainer_target_lines,
                [batch, MAX_TRAINER_TOKENS * MAX_TRAINER_TARGET_IDS],
            ),
            history_features: floats3(
                self.history_features,
                [batch, HISTORY_LEN, HISTORY_DYNAMIC_DIM],
            ),
            history_card_ids: ints(self.history_card_ids, [batch, HISTORY_LEN]),
            history_head_ids: ints(self.history_head_ids, [batch, HISTORY_LEN]),
            seq_mask: floats(self.seq_mask, [batch, SEQ_LEN]),
            self_pokemon_rows: ints(self.self_pokemon_rows, [batch, POKEMON_SELF]),
            self_attack_rows: ints(self.self_attack_rows, [batch, ATTACK_SELF]),
            self_trainer_rows: ints(self.self_trainer_rows, [batch, TRAINER_SELF]),
            board_self_rows: ints(self.board_self_rows, [batch, BOARD_SLOTS]),
            board_opp_rows: ints(self.board_opp_rows, [batch, BOARD_SLOTS]),
            candidate_type_ids: ints(self.candidate_type_ids, [batch, scored]),
            candidate_ref_rows: Tensor::from_data(
                TensorData::new(self.candidate_ref_rows, [batch, scored, refs]),
                device,
            ),
            candidate_ref_mask: floats3(self.candidate_ref_mask, [batch, scored, refs]),
            mask_bits: floats(self.mask_bits, [batch, ACTION_MASK_DIM]),
            wires: self.wires,
            regimes: self.regimes,
        }
    }
}

/// The sequence rows of the entities a candidate references (§1.3.5's
/// `pool(referenced-entity embeddings)`). Best-effort: an unmatched reference contributes
/// nothing, which degrades the candidate's *encoding*, never its legality.
fn candidate_reference_rows(action: &SimpleAction, observation: &Observation) -> Vec<usize> {
    let board_row = |player_is_actor: bool, slot: usize| -> Option<usize> {
        observation
            .pokemon
            .iter()
            .enumerate()
            .find(|(_, token)| {
                token.allied == player_is_actor
                    && token.zone == TokenZone::Board
                    && token.slot == Some(slot)
            })
            .map(|(row, _)| POKEMON_OFFSET + row)
    };
    let self_board = |slot: usize| board_row(true, slot);
    // Rows of an allied off-board card, wherever it sits (hand for most candidates; the deck /
    // discard variants reuse it unchanged). Pokémon bank first, then Trainer.
    let card_rows = |card: &Card, zone: TokenZone| -> Vec<usize> {
        let id = card_index(card.get_card_id());
        let pokemon = observation
            .pokemon
            .iter()
            .enumerate()
            .filter(|(_, token)| token.allied && token.zone == zone && token.card_id == id)
            .map(|(row, _)| POKEMON_OFFSET + row);
        let trainer = observation
            .trainers
            .iter()
            .enumerate()
            .filter(|(_, token)| token.allied && token.zone == zone && token.card_id == id)
            .map(|(row, _)| TRAINER_OFFSET + row);
        pokemon.chain(trainer).collect()
    };
    let first_hand_row = |card: &Card| card_rows(card, TokenZone::Hand).into_iter().next();
    // The opponent's hand emits tokens only for what a reveal effect exposed, so this resolves iff
    // the belief overlay is on — otherwise the candidate keeps the empty encoding it had before
    // (§1.3.6.2). Only the Trainer bank is searched: the engine offers Supporters here and nothing
    // else.
    let revealed_opp_hand_row = |card: &Card| -> Option<usize> {
        let id = card_index(card.get_card_id());
        observation
            .trainers
            .iter()
            .enumerate()
            .find(|(_, token)| {
                !token.allied && token.zone == TokenZone::Hand && token.card_id == id
            })
            .map(|(row, _)| TRAINER_OFFSET + row)
    };

    match action {
        SimpleAction::Attach { attachments, .. } => attachments
            .iter()
            .filter_map(|(_, _, slot)| self_board(*slot))
            .collect(),
        SimpleAction::SadaAttach { assignments } => assignments
            .iter()
            .filter_map(|(_, slot)| self_board(*slot))
            .collect(),
        SimpleAction::Heal { in_play_idx, .. }
        | SimpleAction::HealAndDiscardEnergy { in_play_idx, .. }
        | SimpleAction::AttachFromDiscard { in_play_idx, .. }
        | SimpleAction::AttachTypedFromDiscard { in_play_idx, .. }
        | SimpleAction::ReturnPokemonToHand { in_play_idx }
        | SimpleAction::ShuffleInPlayPokemonIntoDeck { in_play_idx }
        | SimpleAction::UseAbility { in_play_idx }
        | SimpleAction::DiscardFossil { in_play_idx } => {
            self_board(*in_play_idx).into_iter().collect()
        }
        SimpleAction::AttachTool {
            in_play_idx,
            tool_card,
        } => self_board(*in_play_idx)
            .into_iter()
            .chain(first_hand_row(tool_card))
            .collect(),
        SimpleAction::Activate {
            player,
            in_play_idx,
        }
        | SimpleAction::DiscardToolFromPokemon {
            player,
            in_play_idx,
        } => {
            // The observation is the actor's, so "allied" means `player == actor`.
            board_row(observation.perspective == *player, *in_play_idx)
                .into_iter()
                .collect()
        }
        SimpleAction::ApplyDamage { targets, .. } => targets
            .iter()
            .filter_map(|(_, player, slot)| board_row(observation.perspective == *player, *slot))
            .collect(),
        SimpleAction::ScheduleDelayedSpotDamage {
            target_player,
            target_in_play_idx,
            ..
        } => board_row(
            observation.perspective == *target_player,
            *target_in_play_idx,
        )
        .into_iter()
        .collect(),
        SimpleAction::MoveEnergy {
            from_in_play_idx,
            to_in_play_idx,
            ..
        } => [*from_in_play_idx, *to_in_play_idx]
            .iter()
            .filter_map(|slot| self_board(*slot))
            .collect(),
        SimpleAction::MoveAllDamage { from, to } => [*from, *to]
            .iter()
            .filter_map(|slot| self_board(*slot))
            .collect(),
        SimpleAction::Retreat(in_play_idx) => [0, *in_play_idx]
            .iter()
            .filter_map(|slot| self_board(*slot))
            .collect(),
        SimpleAction::Attack(_) => self_board(0).into_iter().collect(),
        SimpleAction::Place(card, _) | SimpleAction::CommunicatePokemon { hand_pokemon: card } => {
            first_hand_row(card).into_iter().collect()
        }
        SimpleAction::Evolve {
            evolution,
            in_play_idx,
            from_deck,
        } => {
            let zone = if *from_deck {
                TokenZone::Deck
            } else {
                TokenZone::Hand
            };
            card_rows(evolution, zone)
                .into_iter()
                .take(1)
                .chain(self_board(*in_play_idx))
                .collect()
        }
        SimpleAction::Play { trainer_card } => {
            crate::card_ids::CardId::from_card_id(&trainer_card.id)
                .map(|card_id| {
                    let id = card_index(card_id);
                    observation
                        .trainers
                        .iter()
                        .enumerate()
                        .filter(|(_, token)| {
                            token.allied && token.zone == TokenZone::Hand && token.card_id == id
                        })
                        .map(|(row, _)| TRAINER_OFFSET + row)
                        .take(1)
                        .collect()
                })
                .unwrap_or_default()
        }
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => {
            hand_pokemon.iter().filter_map(first_hand_row).collect()
        }
        SimpleAction::ShuffleOwnCardsIntoDeck { cards }
        | SimpleAction::DiscardOwnCards { cards } => {
            cards.iter().filter_map(first_hand_row).collect()
        }
        SimpleAction::SwitchHandCardForRandomTool { hand_card } => {
            first_hand_row(hand_card).into_iter().collect()
        }
        SimpleAction::ShuffleOpponentSupporter { supporter_card }
        | SimpleAction::DiscardOpponentSupporter { supporter_card } => {
            revealed_opp_hand_row(supporter_card).into_iter().collect()
        }

        // Nullary choices and engine-internal frames: nothing to point at.
        SimpleAction::ApplyStatusToOpponentActive { .. }
        | SimpleAction::ApplyEeveeBagDamageBoost
        | SimpleAction::HealAllEeveeEvolutions
        | SimpleAction::DiscardActiveStadium
        | SimpleAction::DiscardRandomOpponentActiveEnergy
        | SimpleAction::UseStadium
        | SimpleAction::EndTurn
        | SimpleAction::DrawCard { .. }
        | SimpleAction::Noop => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    use crate::actions::Action;
    use crate::card_ids::CardId;
    use crate::database::get_card_by_enum;
    use crate::models::PlayedCard;
    use crate::rl::action_mask::{project, MaskEntry};
    use crate::rl::observation::get_observation;
    use crate::test_support::get_test_game_with_board;

    /// The §1.5.5 panic, raised by the *encoder* rather than by the engine: a frame whose candidate
    /// list overruns the wire cap. Batched with clean frames — which is the only way a rollout ever
    /// encodes one — it has to come back named, since a caller can only drop a game it can
    /// identify, and the 127 frames sharing the forward did nothing wrong.
    #[test]
    fn an_overflowing_candidate_is_attributed_to_its_own_row() {
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1005Caterpie)],
        );
        let state = game.get_state_clone();
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(&state, actor, &actions, None, None);
        let mask = project(&state, &actions, &observation);
        let config = ModelConfig::default();

        let mut overflowing = mask.clone();
        overflowing.entries.push(MaskEntry {
            head: Head::CandidatePtr,
            index: config.max_scored_candidates,
            action: SimpleAction::EndTurn,
            is_stack: false,
        });
        let point = |mask| DecisionPoint {
            observation: &observation,
            mask,
        };
        let points = vec![point(&mask), point(&overflowing), point(&mask)];

        let fault = ModelInput::<NdArray>::try_from_points(&points, &config, &Default::default())
            .err()
            .expect("a candidate past the cap is not encodable");
        let EncodeFault::Row { row, panic } = fault else {
            panic!("one frame overflows, so the batch is not to blame: {fault}");
        };
        assert_eq!(row, 1);
        assert!(
            panic.message.contains("max_scored_candidates"),
            "unhelpful: {panic}"
        );
        // The location is the assertion's, not the guard's — the point of re-encoding row by row
        // is to keep the panic exactly as the encoder raised it.
        assert!(panic
            .location
            .as_deref()
            .expect("a location")
            .contains("input.rs"));
    }

    /// The happy path pays nothing: the guarded call returns the same input the unguarded one does.
    #[test]
    fn a_clean_batch_passes_through_the_guard() {
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1005Caterpie)],
        );
        let state = game.get_state_clone();
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(&state, actor, &actions, None, None);
        let mask = project(&state, &actions, &observation);
        let points = vec![DecisionPoint {
            observation: &observation,
            mask: &mask,
        }];

        let config = ModelConfig::default();
        let guarded = ModelInput::<NdArray>::try_from_points(&points, &config, &Default::default())
            .expect("a legal frame encodes");
        let plain = ModelInput::<NdArray>::from_points(&points, &config, &Default::default());
        assert_eq!(guarded.batch, plain.batch);
        assert_eq!(guarded.wires, plain.wires);
        assert_eq!(
            guarded.mask_bits.to_data().to_vec::<f32>().expect("bits"),
            plain.mask_bits.to_data().to_vec::<f32>().expect("bits")
        );
    }

    /// §1.3.6.2: the two reveal candidates used to reference nothing, because the card they point
    /// at was in a zone the encoder could not see. With the belief render they resolve to their
    /// own row — which is the whole point of the head being learned rather than random.
    #[test]
    fn a_reveal_candidate_references_the_card_it_points_at() {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1005Caterpie)],
        );
        game.enable_belief();

        let mut state = game.get_state_clone();
        state.hands[0] = vec![get_card_by_enum(CardId::A4a032Misdreavus)];
        state.hands[1] = vec![get_card_by_enum(CardId::PA001Potion)];
        game.set_state(state);
        game.apply_action(&Action {
            actor: 0,
            action: SimpleAction::Place(get_card_by_enum(CardId::A4a032Misdreavus), 1),
            is_stack: false,
        });

        let observation = game.get_observation(0);
        let candidate = SimpleAction::DiscardOpponentSupporter {
            supporter_card: get_card_by_enum(CardId::PA001Potion),
        };
        let rows = candidate_reference_rows(&candidate, &observation);
        assert_eq!(rows.len(), 1, "the revealed Potion is one row: {rows:?}");

        let row = rows[0] - TRAINER_OFFSET;
        let token = &observation.trainers[row];
        assert!(!token.allied);
        assert_eq!(token.zone, TokenZone::Hand);
        assert!(
            !token.playable_now,
            "an opponent's card is never ours to play"
        );
    }

    /// The same candidate against a game in spectator mode: no token, no row, and the encoding
    /// degrades exactly as `candidate_reference_rows` documents — never the legality.
    #[test]
    fn a_reveal_candidate_references_nothing_without_the_overlay() {
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1005Caterpie)],
        );
        let observation = game.get_observation(0);
        let candidate = SimpleAction::DiscardOpponentSupporter {
            supporter_card: get_card_by_enum(CardId::PA001Potion),
        };
        assert!(candidate_reference_rows(&candidate, &observation).is_empty());
    }
}
