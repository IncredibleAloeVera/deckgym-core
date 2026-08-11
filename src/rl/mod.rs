//! Observation (v1) — `RL_ARCHITECTURE.md` §1.2.
//!
//! The frozen specification of what the RL agents see. Five families of features:
//!
//! | Family  | Where it lives                                                                  |
//! | ------- | ------------------------------------------------------------------------------- |
//! | Global  | [`observation::GlobalFeatures`] — 106 floats + 1 index                           |
//! | Pokémon | [`observation::PokemonToken`] — 32 floats + 4 indices, static gathered in-model  |
//! | Attack  | [`observation::AttackToken`] — action-affordance satellite, 14 floats + 2 idx    |
//! | Trainer | [`observation::TrainerToken`] — 7 floats + 1 idx + a target-set bag              |
//! | History | [`history::HistoryToken`] — the opponent's last 20 observable *choices*, ordered |
//!
//! and, on top of them, the Part 3 [`action_mask`]: the same `generate_possible_actions`
//! enumeration bucketed onto the factorized actor heads, its pointer indices naming rows of the
//! very token banks above.
//!
//! The two load-bearing design decisions:
//!
//! 1. **Identity is an index, not a payload.** The wire carries `card_id` / `species_id` /
//!    `line_id` / `tool_id`; the heavy descriptor lives in [`static_tables`] and is gathered
//!    in-model. See [`ids`] for the three granularities.
//! 2. **Imperfect information is respected.** [`observation::get_observation`] is egocentric: it
//!    emits the perspective player's own hand and deck, both boards, both discard piles, and the
//!    opponent's hand/deck *sizes* only. Nothing else about the opponent's hidden zones reaches the
//!    wire — see the module docs of [`observation`] for the zone-by-zone contract.
//!
//! The observation is a *sibling projection* of `generate_possible_actions`, never a
//! reimplementation of legality: the same enumeration feeds `get_observation` and the (future)
//! Part 3 action mask.

pub mod action_mask;
pub mod damage;
pub mod encoding;
pub mod env;
pub mod history;
pub mod ids;
#[cfg(feature = "rl-model")]
pub mod model;
pub mod observation;
#[cfg(feature = "rl-model")]
pub mod play;
pub mod recover;
#[cfg(feature = "rl-model")]
pub mod seat;
pub mod static_tables;
pub mod text_embedding;
pub mod train;

pub use action_mask::{
    canonical_action, project as project_action_mask, ActionFamily, ActionMask, ActionMaskWire,
    Head, MaskEntry, Regime, ACTION_MASK_DIM, ATTACK_SELF, POKEMON_SELF, TRAINER_SELF,
};
pub use damage::{
    estimate_attack_affordance, estimate_attack_threat, estimate_damage, AttackAffordance,
    DamageEstimate, ProjectionScratch,
};
pub use env::{
    AgentId, Crashed, DecisionRequest, Env, EnvOutcome, EnvStep, SeatPolicy, SubmitError,
    SubmitFault, VecEnv,
};
pub use history::{ActionTrace, HistoryToken, HISTORY_LEN};
pub use observation::{
    get_observation, AttackToken, GlobalFeatures, Observation, ObservationWire, PokemonToken,
    TokenZone, TrainerToken, MAX_ATTACK_TOKENS, MAX_POKEMON_TOKENS, MAX_TRAINER_TOKENS,
};
pub use recover::{catch as catch_engine_panic, EnginePanic};
pub use text_embedding::TextEmbeddings;
pub use train::{
    DeckDb, DeckSampler, DeckSource, SampledDeck, SamplerConfig, SourceSpec, TrainConfig,
};

/// Finite horizon (Part 1): games are capped at 99 turns, and every turn-derived feature is
/// normalized by it. The engine enforces the same cap in `State::advance_turn`.
pub const HORIZON: f32 = 99.0;

/// What §1.5.2's opponent pool checks before it will play a stored model.
///
/// Bump this by hand when the *meaning* of the wire changes — a feature that starts counting
/// something else, an index space that is renumbered, a mask bit that moves family. Those are
/// invisible to [`schema_fingerprint`] below, which only sees widths, and they are exactly the
/// changes that turn an old checkpoint into a policy reading noise. Version history and the
/// reasoning behind each bump: NOTES.md, "Schéma d'observation — historique des versions".
pub const OBS_SCHEMA_VERSION: u32 = 4;

/// FNV-1a, so [`schema_fingerprint`] can be a `const fn`.
const fn mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01B3)
}

/// A digest of every width that decides whether a stored model can read this build's observations.
/// Covers the wire shape, not [`model::config::ModelConfig`] — two models may differ freely in
/// `d_model`, block count, etc. and still be interchangeable opponents, since each runs its own
/// instance. Derived rather than hand-maintained so the mechanical half cannot be forgotten;
/// [`OBS_SCHEMA_VERSION`] is mixed in for the semantic half, so bumping either invalidates.
pub const fn schema_fingerprint() -> u64 {
    let widths: [u64; 22] = [
        OBS_SCHEMA_VERSION as u64,
        observation::GLOBAL_DIM as u64,
        observation::POKEMON_DYNAMIC_DIM as u64,
        observation::ATTACK_DYNAMIC_DIM as u64,
        observation::TRAINER_DYNAMIC_DIM as u64,
        observation::MAX_POKEMON_TOKENS as u64,
        observation::MAX_TRAINER_TOKENS as u64,
        observation::MAX_ATTACK_TOKENS as u64,
        observation::MAX_TRAINER_TARGET_IDS as u64,
        history::HISTORY_LEN as u64,
        action_mask::ACTION_MASK_DIM as u64,
        action_mask::ACTION_TYPE_DIM as u64,
        action_mask::STATUS_CAT_DIM as u64,
        action_mask::MAX_CANDIDATE_PTR as u64,
        action_mask::MAX_REVEALED_HAND_PTR as u64,
        action_mask::POKEMON_SELF as u64,
        action_mask::TRAINER_SELF as u64,
        action_mask::ATTACK_SELF as u64,
        encoding::ENERGY_DIM as u64,
        encoding::HP_DIM as u64,
        encoding::DAMAGE_DIM as u64,
        encoding::RETREAT_COST_DIM as u64,
    ];
    let mut hash = 0xcbf2_9ce4_8422_2325;
    let mut index = 0;
    while index < widths.len() {
        hash = mix(hash, widths[index]);
        index += 1;
    }
    hash
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    /// A canary, not a specification: it fails whenever a width above changes, which is the moment
    /// to decide whether stored models are still readable — and, if they are not, to bump
    /// [`OBS_SCHEMA_VERSION`] so the fingerprint moves for the semantic reason too.
    #[test]
    fn the_schema_fingerprint_is_stable() {
        assert_eq!(schema_fingerprint(), 0x4cae_a8b7_fd0b_4e2c);
    }

    #[test]
    fn the_fingerprint_moves_with_the_version() {
        assert_ne!(mix(schema_fingerprint(), 0), schema_fingerprint());
    }
}
