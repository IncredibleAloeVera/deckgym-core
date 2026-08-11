//! The observation itself (§1.2.3 – §1.2.8): what one player sees at one decision point.
//!
//! # Imperfect information
//!
//! [`get_observation`] is egocentric — everything is encoded by role (self / opponent), never by
//! absolute player index — and it is the enforcement point of §1.2.1's information table:
//!
//! | Zone                                    | Self                          | Opponent                          |
//! | --------------------------------------- | ----------------------------- | --------------------------------- |
//! | Board (Pokémon, tools, attached energy) | full                          | full (public)                     |
//! | Discard pile                            | full                          | full (public)                     |
//! | Hand                                    | full contents                 | **size**, plus what a reveal showed |
//! | Deck / draw pile                        | full contents (unordered)     | **size + energy types seen so far** |
//!
//! Concretely: a card of the opponent's hand or deck emits no token unless a reveal effect exposed
//! it. Their size rides in the global vector as-is. Their energy types do **not** — the deck's full
//! declared set (≤ 3, fixed at deck construction) is not public in TCG Pocket, only whatever has
//! actually rolled through their energy zone so far (TODO.md, "Opponent deck energy"), so
//! `deck_energy_types[opponent]` reads `belief.seen_opponent_energy` instead of
//! `Deck::energy_types` — monotone, and empty when `belief` is `None` (spectator mode), same as the
//! rest of this table's belief-gated row. Both energy zones (`current` + `next`) are themselves
//! public in TCG Pocket the instant they roll, so both ride in the global vector unconditionally —
//! it is only the *cumulative* memory of past rolls that is belief-gated.
//!
//! The reveal effects that punch holes in this table (Silver, Mega Absol ex) are the one exception,
//! and they reach the wire only through the `belief` argument: the `state` is the spectator view and
//! records nothing about who has seen what, so a hidden card of the opponent's becomes a token iff
//! the belief overlay ([`crate::belief`]) says this observer is entitled to it — `zone = Hand` while
//! it is located, `zone = Unknown` once only the memory of it survives. Passing `None` is the
//! pre-reveal table above, exactly.
//!
//! # Legality features are a sibling projection
//!
//! `legal_actions = generate_possible_actions(state)` feeds *both* this function and the (future)
//! Part 3 action mask. The legality bits below (`can_evolve_this_turn`, `ability_activatable_now`,
//! `playable_now`) are that same enumeration projected onto tokens — never a second implementation
//! of legality, and never derived from the mask. They are defined for the acting player's own
//! entities and are 0 elsewhere.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::actions::abilities::{AbilityMechanic, AbilityMechanicDiscriminants};
use crate::actions::{get_ability_mechanic, Action, SimpleAction};
use crate::belief::BeliefTracker;
use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::hooks::get_retreat_cost;
use crate::models::{Attack, Card, EnergyType, TrainerType};
use crate::state::PlayedCard;
use crate::State;

use super::damage::{ProjectionScratch, BOARD_SLOTS};
use super::encoding::*;
use super::history::{ActionTrace, HistoryToken, HISTORY_DYNAMIC_DIM, HISTORY_LEN};
use super::ids::{card_index, identity_indices, PAD_INDEX};
use super::static_tables::{trainer_targeting, MAX_ATTACKS_PER_CARD};
use super::HORIZON;

/// Global vector width (§1.2.3), excluding the stadium index.
pub const GLOBAL_DIM: usize = 106;
/// Pokémon token, dynamic block (§1.2.4).
pub const POKEMON_DYNAMIC_DIM: usize = 33;
/// Attack token, dynamic block (§1.2.5).
pub const ATTACK_DYNAMIC_DIM: usize = 14;
/// Trainer token, dynamic block (§1.2.6).
pub const TRAINER_DYNAMIC_DIM: usize = 8;

/// Padded bank sizes (§1.2.8). A 20-card deck bounds a player's own tokens, so "everything of
/// mine + everything public of theirs" fits: 20 self + ≤ 20 opponent-public per entity type.
pub const MAX_POKEMON_TOKENS: usize = 40;
pub const MAX_TRAINER_TOKENS: usize = 40;
pub const MAX_ATTACK_TOKENS: usize = 32;
/// Padded width of a trainer token's target-set bag.
pub const MAX_TRAINER_TARGET_IDS: usize = 8;

/// Where a card sits. The only spatial signal is the board `slot`, and it is a feature — the token
/// sets are permutation-invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenZone {
    Board,
    Hand,
    Deck,
    Discard,
    /// A card the observer knows the opponent holds in *some* hidden zone, without a live position
    /// marker saying which — the belief overlay's monotone `presence`, netted against everything
    /// currently visible.
    Unknown,
}

impl TokenZone {
    pub const DIM: usize = 5;

    /// Series names, in [`TokenZone::index`] order.
    pub const NAMES: [&'static str; Self::DIM] = ["board", "hand", "deck", "discard", "unknown"];

    pub fn index(self) -> usize {
        match self {
            TokenZone::Board => 0,
            TokenZone::Hand => 1,
            TokenZone::Deck => 2,
            TokenZone::Discard => 3,
            TokenZone::Unknown => 4,
        }
    }
}

/// Where the zone one-hot starts inside a Pokémon / Trainer dynamic block.
///
/// Public because §1.5.6's attention read-out splits those two families by zone, and the zone is
/// nowhere else on the wire: it has to read the one-hot back out of the feature block. Asserted
/// against both encoders by `the_zone_one_hot_leads_both_dynamic_blocks` rather than trusted — a
/// field inserted ahead of it would otherwise relabel every zone series in silence.
pub const ZONE_FEATURE_OFFSET: usize = 0;

// ---------------------------------------------------------------------------------------------
// Global
// ---------------------------------------------------------------------------------------------

/// The state summary (§1.2.3). Every `[2]` array is `[self, opponent]` — role, not player index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalFeatures {
    pub turn_count: u8,
    /// The perspective player takes (or took) turn 1. **0 for both players during the setup
    /// phase**: in the real game both players place their boards simultaneously, so the engine's
    /// placement alternation is an artifact that must not leak into the observation.
    pub on_the_play: bool,
    /// The perspective player owns the current turn (a decision frame may still be theirs when
    /// it is not — forced promotion after a KO, Part 3 §4.1).
    pub is_my_turn: bool,
    /// `turn_count <= 2` — the opening window in which evolutions are not yet allowed.
    pub is_setup_phase: bool,
    pub points: [u8; 2],
    pub draw_pile: [usize; 2],
    pub hand_size: [usize; 2],
    pub discard_size: [usize; 2],
    /// `[self, opponent]`. Self: the deck's full declared energy pool (≤ 3 types), known outright.
    /// Opponent: only the types actually seen so far in their energy zone — the declared pool
    /// itself is not public in TCG Pocket. Multi-hot over the canonical `Energy` order.
    pub deck_energy_types: [[bool; ENERGY_DIM]; 2],
    /// `[[current, next]; 2]`, both public in TCG Pocket.
    pub energy_zone: [[Option<EnergyType>; 2]; 2],
    /// This turn's generated energy has already been placed. Only ever set for the turn player
    /// once generation has happened (the player going first gets no energy on turn 1) — a bare
    /// `current.is_none()` would conflate "placed" with "never generated".
    pub energy_already_attached: [bool; 2],
    pub discard_energy: [[u32; ENERGY_DIM]; 2],
    pub has_stadium: bool,
    /// Index into the shared card-embedding table; `0` when there is no stadium.
    pub stadium_id: u32,
    pub has_played_support: [bool; 2],
    pub has_retreated: [bool; 2],
    pub has_used_stadium: [bool; 2],
    /// `[this turn, last turn]` — the engine's watchlist flags for a KO by an opponent's attack.
    pub knocked_out_by_opponent_attack: [bool; 2],
}

impl GlobalFeatures {
    /// The [`GLOBAL_DIM`] floats this vector puts on the wire, in spec-table order.
    pub fn values(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(GLOBAL_DIM);
        let turn = self.turn_count as f32;

        out.push(turn / HORIZON);
        out.push((1.0 + turn).ln() / (1.0 + HORIZON).ln());
        out.push(((HORIZON - turn) / HORIZON).clamp(0.0, 1.0));

        push_bit(&mut out, self.on_the_play);
        push_bit(&mut out, self.is_my_turn);
        push_bit(&mut out, self.is_setup_phase);

        push_ratio(&mut out, self.points[0] as f32, 2.0);
        push_ratio(&mut out, self.points[1] as f32, 2.0);
        push_signed_ratio(&mut out, self.points[0] as f32 - self.points[1] as f32, 2.0);

        for size in self.draw_pile {
            push_ratio(&mut out, size as f32, 17.0);
        }
        for size in self.hand_size {
            push_ratio(&mut out, size as f32, 10.0);
        }
        for size in self.discard_size {
            push_ratio(&mut out, size as f32, 19.0);
        }

        for mask in &self.deck_energy_types {
            for declared in mask {
                push_bit(&mut out, *declared);
            }
        }

        for zone in &self.energy_zone {
            for slot in zone {
                push_energy_one_hot(&mut out, *slot);
            }
        }
        for attached in self.energy_already_attached {
            push_bit(&mut out, attached);
        }
        for counts in &self.discard_energy {
            push_energy_counts(&mut out, counts, DISCARD_ENERGY_DENOM);
        }

        push_bit(&mut out, self.has_stadium);
        for played in self.has_played_support {
            push_bit(&mut out, played);
        }
        for retreated in self.has_retreated {
            push_bit(&mut out, retreated);
        }
        for used in self.has_used_stadium {
            push_bit(&mut out, used);
        }
        for knocked_out in self.knocked_out_by_opponent_attack {
            push_bit(&mut out, knocked_out);
        }

        debug_assert_eq!(out.len(), GLOBAL_DIM);
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------------------------

/// A Pokémon entity (§1.2.4). Emitted for every board Pokémon on both sides, every Pokémon and
/// Fossil in the perspective player's hand / deck / discard, and every one in the opponent's
/// (public) discard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PokemonToken {
    pub card_id: u32,
    pub species_id: u32,
    pub line_id: u32,
    /// Attached tool, `0` when none — attached tools ride on their host rather than as tokens.
    pub tool_id: u32,
    pub zone: TokenZone,
    /// Belongs to the perspective player.
    pub allied: bool,
    /// Board slot (`0` = active), `None` off the board.
    pub slot: Option<usize>,
    pub remaining_hp_ratio: f32,
    /// Jungle-Totem-aware effective attachment (Serperior doubles Grass).
    pub attached_energy: [u32; ENERGY_DIM],
    /// Signed additional retreat cost from tools/abilities, relative to the printed cost.
    pub retreat_cost_delta: i32,
    /// `[poison, paralyze, sleep, burn, confuse]`.
    pub status: [bool; 5],
    /// Legality bit: this token participates in a legal `Evolve` right now.
    pub can_evolve_this_turn: bool,
    pub ability_used: bool,
    /// Legality bit: a legal `UseAbility` points at this slot.
    pub ability_activatable_now: bool,
    /// A typed start/end-of-turn ability whose condition is currently met.
    pub ability_will_proc: bool,
    pub has_tool: bool,
}

impl PokemonToken {
    /// The 33 floats this token puts on the wire.
    pub fn features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(POKEMON_DYNAMIC_DIM);
        push_one_hot(&mut out, Some(self.zone.index()), TokenZone::DIM);
        push_bit(&mut out, self.allied);
        push_one_hot(&mut out, self.slot, BOARD_SLOTS);
        push_bit(&mut out, self.slot == Some(0));
        push_ratio(&mut out, self.remaining_hp_ratio, 1.0);
        push_energy_counts(&mut out, &self.attached_energy, ATTACHED_ENERGY_DENOM);
        push_signed_ratio(
            &mut out,
            self.retreat_cost_delta as f32,
            RETREAT_DELTA_DENOM,
        );
        for condition in self.status {
            push_bit(&mut out, condition);
        }
        push_bit(&mut out, self.can_evolve_this_turn);
        push_bit(&mut out, self.ability_used);
        push_bit(&mut out, self.ability_activatable_now);
        push_bit(&mut out, self.ability_will_proc);
        push_bit(&mut out, self.has_tool);
        debug_assert_eq!(out.len(), POKEMON_DYNAMIC_DIM);
        out
    }

    /// The four indices this token puts on the wire.
    pub fn indices(&self) -> [u32; 4] {
        [self.card_id, self.species_id, self.line_id, self.tool_id]
    }
}

/// An action-affordance satellite (§1.2.5), aligned with the factorized `Attack(Attack)` head.
/// One token per usable attack of each board Pokémon on either side — including benched
/// attackers, which is what makes the threat matrix a full our-attacks × their-Pokémon picture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttackToken {
    /// Row of the parent Pokémon in the Pokémon token bank.
    pub parent_pokemon_ref: u32,
    /// The card the attack's descriptor comes from — the Pokémon itself, or an earlier stage for
    /// an attack borrowed through `cards_behind`.
    pub src_card_id: u32,
    /// Which attack on the source card.
    pub attack_slot: usize,
    pub allied: bool,
    pub can_pay: bool,
    /// Missing energies, given the parent's current effective attachment.
    pub deficit: u32,
    /// Energies beyond the cost.
    pub surplus: u32,
    /// Expected damage ÷ that defender's remaining HP, per opposing board slot.
    pub threat: [f32; BOARD_SLOTS],
    /// Guaranteed-KO floor, per opposing board slot.
    pub is_lethal: [bool; BOARD_SLOTS],
}

impl AttackToken {
    /// The 14 floats this token puts on the wire.
    pub fn features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(ATTACK_DYNAMIC_DIM);
        push_one_hot(&mut out, Some(self.attack_slot), MAX_ATTACKS_PER_CARD);
        push_bit(&mut out, self.allied);
        push_bit(&mut out, self.can_pay);
        push_ratio(&mut out, self.deficit as f32, ATTACK_COST_DENOM);
        push_ratio(&mut out, self.surplus as f32, ATTACK_COST_DENOM);
        for ratio in self.threat {
            push_ratio(&mut out, ratio, 1.0);
        }
        for lethal in self.is_lethal {
            push_bit(&mut out, lethal);
        }
        debug_assert_eq!(out.len(), ATTACK_DYNAMIC_DIM);
        out
    }

    /// The two indices this token puts on the wire.
    pub fn indices(&self) -> [u32; 2] {
        [self.parent_pokemon_ref, self.src_card_id]
    }
}

/// An Item / Supporter / Tool / Stadium card (§1.2.6). Fossils are Pokémon tokens instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainerToken {
    pub card_id: u32,
    /// `(species_id, line_id)` of every Pokémon the card names — gathered live and summed into a
    /// `d_id` bag in-model, so it indexes the *trainable* embeddings rather than a frozen copy.
    pub target_ids: Vec<(u32, u32)>,
    pub zone: TokenZone,
    pub allied: bool,
    /// Legality bit: a legal `Play` points at this card.
    pub playable_now: bool,
    /// The card's own condition holds, independently of whether it can be played right now
    /// (a Supporter after the once-per-turn Supporter has been used, for instance).
    pub activation_condition_met: bool,
}

impl TrainerToken {
    /// The 8 floats this token puts on the wire.
    pub fn features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(TRAINER_DYNAMIC_DIM);
        push_one_hot(&mut out, Some(self.zone.index()), TokenZone::DIM);
        push_bit(&mut out, self.allied);
        push_bit(&mut out, self.playable_now);
        push_bit(&mut out, self.activation_condition_met);
        debug_assert_eq!(out.len(), TRAINER_DYNAMIC_DIM);
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------------------------

/// One decision point, seen by one player.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Whose view this is. Every `allied` flag and `[self, opponent]` pair is relative to it.
    pub perspective: usize,
    pub global: GlobalFeatures,
    pub pokemon: Vec<PokemonToken>,
    pub attacks: Vec<AttackToken>,
    pub trainers: Vec<TrainerToken>,
    /// Ordered, oldest first.
    pub history: Vec<HistoryToken>,
}

/// The padded, masked, flat form the model consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationWire {
    pub global: Vec<f32>,
    pub stadium_id: u32,
    pub pokemon: Vec<f32>,
    pub pokemon_indices: Vec<u32>,
    pub pokemon_mask: Vec<bool>,
    pub attack: Vec<f32>,
    pub attack_indices: Vec<u32>,
    pub attack_mask: Vec<bool>,
    pub trainer: Vec<f32>,
    pub trainer_indices: Vec<u32>,
    pub trainer_target_indices: Vec<u32>,
    pub trainer_mask: Vec<bool>,
    pub history: Vec<f32>,
    pub history_indices: Vec<u32>,
    pub history_mask: Vec<bool>,
}

impl Observation {
    /// Flatten to the wire: fixed-size banks, zero padding, and a mask per bank.
    ///
    /// Panics if a bank overflows its cap — an assert, per §1.2.8, rather than a silent
    /// truncation: dropping entities would corrupt the observation instead of degrading it.
    pub fn to_wire(&self) -> ObservationWire {
        assert!(
            self.pokemon.len() <= MAX_POKEMON_TOKENS,
            "Pokémon token overflow: {} > {MAX_POKEMON_TOKENS}",
            self.pokemon.len()
        );
        assert!(
            self.attacks.len() <= MAX_ATTACK_TOKENS,
            "Attack token overflow: {} > {MAX_ATTACK_TOKENS}",
            self.attacks.len()
        );
        assert!(
            self.trainers.len() <= MAX_TRAINER_TOKENS,
            "Trainer token overflow: {} > {MAX_TRAINER_TOKENS}",
            self.trainers.len()
        );

        let mut wire = ObservationWire {
            global: self.global.values(),
            stadium_id: self.global.stadium_id,
            pokemon: Vec::with_capacity(MAX_POKEMON_TOKENS * POKEMON_DYNAMIC_DIM),
            pokemon_indices: Vec::with_capacity(MAX_POKEMON_TOKENS * 4),
            pokemon_mask: vec![false; MAX_POKEMON_TOKENS],
            attack: Vec::with_capacity(MAX_ATTACK_TOKENS * ATTACK_DYNAMIC_DIM),
            attack_indices: Vec::with_capacity(MAX_ATTACK_TOKENS * 2),
            attack_mask: vec![false; MAX_ATTACK_TOKENS],
            trainer: Vec::with_capacity(MAX_TRAINER_TOKENS * TRAINER_DYNAMIC_DIM),
            trainer_indices: Vec::with_capacity(MAX_TRAINER_TOKENS),
            trainer_target_indices: Vec::with_capacity(
                MAX_TRAINER_TOKENS * MAX_TRAINER_TARGET_IDS * 2,
            ),
            trainer_mask: vec![false; MAX_TRAINER_TOKENS],
            history: Vec::with_capacity(HISTORY_LEN * HISTORY_DYNAMIC_DIM),
            history_indices: Vec::with_capacity(HISTORY_LEN * 2),
            history_mask: vec![false; HISTORY_LEN],
        };

        for slot in 0..MAX_POKEMON_TOKENS {
            match self.pokemon.get(slot) {
                Some(token) => {
                    wire.pokemon.extend(token.features());
                    wire.pokemon_indices.extend(token.indices());
                    wire.pokemon_mask[slot] = true;
                }
                None => {
                    wire.pokemon
                        .extend(std::iter::repeat_n(0.0, POKEMON_DYNAMIC_DIM));
                    wire.pokemon_indices.extend([PAD_INDEX; 4]);
                }
            }
        }

        for slot in 0..MAX_ATTACK_TOKENS {
            match self.attacks.get(slot) {
                Some(token) => {
                    wire.attack.extend(token.features());
                    wire.attack_indices.extend(token.indices());
                    wire.attack_mask[slot] = true;
                }
                None => {
                    wire.attack
                        .extend(std::iter::repeat_n(0.0, ATTACK_DYNAMIC_DIM));
                    wire.attack_indices.extend([PAD_INDEX; 2]);
                }
            }
        }

        for slot in 0..MAX_TRAINER_TOKENS {
            match self.trainers.get(slot) {
                Some(token) => {
                    wire.trainer.extend(token.features());
                    wire.trainer_indices.push(token.card_id);
                    for target in 0..MAX_TRAINER_TARGET_IDS {
                        let (species, line) = token
                            .target_ids
                            .get(target)
                            .copied()
                            .unwrap_or((PAD_INDEX, PAD_INDEX));
                        wire.trainer_target_indices.extend([species, line]);
                    }
                    wire.trainer_mask[slot] = true;
                }
                None => {
                    wire.trainer
                        .extend(std::iter::repeat_n(0.0, TRAINER_DYNAMIC_DIM));
                    wire.trainer_indices.push(PAD_INDEX);
                    wire.trainer_target_indices
                        .extend(std::iter::repeat_n(PAD_INDEX, MAX_TRAINER_TARGET_IDS * 2));
                }
            }
        }

        for slot in 0..HISTORY_LEN {
            match self.history.get(slot) {
                Some(token) => {
                    wire.history.extend(token.features());
                    wire.history_indices.extend([token.card_id, token.head_id]);
                    wire.history_mask[slot] = true;
                }
                None => {
                    wire.history
                        .extend(std::iter::repeat_n(0.0, HISTORY_DYNAMIC_DIM));
                    wire.history_indices.extend([PAD_INDEX; 2]);
                }
            }
        }

        wire
    }
}

/// Build the observation of `state` as seen by `perspective`.
///
/// `legal_actions` is the output of `generate_possible_actions(state)` — the *same* enumeration the
/// action mask projects, passed in rather than recomputed so the two projections cannot drift.
/// Only actions whose actor is `perspective` contribute legality bits.
///
/// `trace` is the (optional) action trace maintained alongside the game; without it the History
/// bank is simply empty.
///
/// `belief` is the (optional) player-mode overlay ([`crate::belief`]). The `state` is the spectator
/// view and knows nothing of who has seen what, so reveal effects can only reach the observation
/// through it: without it the opponent's hand stays a count, which is the pre-reveal default of
/// §1.2.1 rather than a degraded mode.
pub fn get_observation(
    state: &State,
    perspective: usize,
    legal_actions: &[Action],
    trace: Option<&ActionTrace>,
    belief: Option<&BeliefTracker>,
) -> Observation {
    let opponent = (perspective + 1) % 2;
    let legality = Legality::project(legal_actions, perspective);
    let revealed_hand = revealed_opponent_hand(belief, perspective);
    let hidden_elsewhere = hidden_elsewhere(state, belief, perspective, opponent);

    let mut pokemon = Vec::new();
    let mut board_token_row = [[None; BOARD_SLOTS]; 2];

    for (role, player) in [(true, perspective), (false, opponent)] {
        for (slot, occupant) in state.in_play_pokemon[player].iter().enumerate() {
            let Some(played) = occupant.as_ref() else {
                continue;
            };
            board_token_row[player][slot] = Some(pokemon.len() as u32);
            pokemon.push(board_token(state, player, slot, played, role, &legality));
        }
    }

    for (zone, cards) in [
        (TokenZone::Hand, &state.hands[perspective]),
        (TokenZone::Deck, &state.decks[perspective].cards),
        (TokenZone::Discard, &state.discard_piles[perspective]),
    ] {
        for card in cards.iter().filter(|card| is_pokemon_token(card)) {
            pokemon.push(off_board_pokemon_token(card, zone, true, &legality));
        }
    }
    // The opponent's discard pile is public; their hand and deck are not, and emit a token only for
    // the cards a reveal effect put into `revealed_hand` (§1.3.6.2).
    for card in state.discard_piles[opponent]
        .iter()
        .filter(|card| is_pokemon_token(card))
    {
        pokemon.push(off_board_pokemon_token(
            card,
            TokenZone::Discard,
            false,
            &legality,
        ));
    }
    for (zone, cards) in [
        (TokenZone::Hand, &revealed_hand),
        (TokenZone::Unknown, &hidden_elsewhere),
    ] {
        for card in cards.iter().filter(|card| is_pokemon_token(card)) {
            pokemon.push(off_board_pokemon_token(card, zone, false, &legality));
        }
    }

    // One scratch for the whole threat matrix (§1.2.5): every attacker of both boards is projected
    // into it in place and restored, instead of each one cloning the state for itself.
    let mut attacks = Vec::new();
    let mut scratch = ProjectionScratch::new(state);
    for (role, player) in [(true, perspective), (false, opponent)] {
        for (slot, occupant) in state.in_play_pokemon[player].iter().enumerate() {
            let Some(played) = occupant.as_ref() else {
                continue;
            };
            let Some(parent_row) = board_token_row[player][slot] else {
                continue;
            };
            for (source, attack_slot, attack) in available_attacks(state, player, played) {
                attacks.push(attack_token(
                    state,
                    &mut scratch,
                    (player, slot),
                    parent_row,
                    source,
                    attack_slot,
                    &attack,
                    role,
                ));
            }
        }
    }

    let mut trainers = Vec::new();
    for (zone, cards) in [
        (TokenZone::Hand, &state.hands[perspective]),
        (TokenZone::Deck, &state.decks[perspective].cards),
        (TokenZone::Discard, &state.discard_piles[perspective]),
    ] {
        for card in cards.iter().filter(|card| is_trainer_token(card)) {
            trainers.push(trainer_token(
                state,
                card,
                zone,
                perspective,
                true,
                &legality,
            ));
        }
    }
    for card in state.discard_piles[opponent]
        .iter()
        .filter(|card| is_trainer_token(card))
    {
        trainers.push(trainer_token(
            state,
            card,
            TokenZone::Discard,
            opponent,
            false,
            &legality,
        ));
    }
    for (zone, cards) in [
        (TokenZone::Hand, &revealed_hand),
        (TokenZone::Unknown, &hidden_elsewhere),
    ] {
        for card in cards.iter().filter(|card| is_trainer_token(card)) {
            trainers.push(trainer_token(state, card, zone, opponent, false, &legality));
        }
    }

    let history = trace
        .map(|trace| trace.tokens_for(perspective, state))
        .unwrap_or_default();

    Observation {
        perspective,
        global: global_features(state, perspective, belief),
        pokemon,
        attacks,
        trainers,
        history,
    }
}

// ---------------------------------------------------------------------------------------------
// Global assembly
// ---------------------------------------------------------------------------------------------

fn global_features(
    state: &State,
    perspective: usize,
    belief: Option<&BeliefTracker>,
) -> GlobalFeatures {
    let opponent = (perspective + 1) % 2;
    let by_role = |values: [bool; 2]| [values[perspective], values[opponent]];

    // Turn ownership alternates every `advance_turn`, so from turn 1 the parity of the turn count
    // recovers who is on the play. During setup (`turn_count == 0`) no one is: placement is
    // simultaneous in the real game and the engine's alternation is an artifact, so the bit stays
    // 0 for both players rather than leaking that artifact.
    let player_on_the_play = if state.turn_count % 2 == 1 {
        state.current_player
    } else {
        (state.current_player + 1) % 2
    };
    let on_the_play = state.turn_count >= 1 && player_on_the_play == perspective;

    // The engine keeps `has_played_support` / `has_retreated` as single turn flags: they describe
    // the turn player. Attribute them to the matching role and leave the other at 0.
    let turn_player_flag = |flag: bool| {
        let mut roles = [false, false];
        if flag {
            roles[usize::from(state.current_player != perspective)] = true;
        }
        roles
    };

    // Set only for the turn player, and only once generation has happened: turn 1 grants no
    // energy, and the off-turn player's `current` slot may still hold last turn's leftover.
    let energy_already_attached = |player: usize| {
        state.current_player == player
            && state.turn_count >= 2
            && state.energy_zone[player].current.is_none()
    };

    // Own deck: fully known. Opponent: only the types actually seen in their energy zone so
    // far — the deck's full declared set is not public in TCG Pocket (TODO.md, "Opponent deck
    // energy"), and `belief` is the observer bookkeeping that earns it back one reveal at a time.
    let declared_energy = |player: usize| {
        let mut mask = [false; ENERGY_DIM];
        for energy in &state.decks[player].energy_types {
            mask[energy_index(*energy)] = true;
        }
        mask
    };
    let seen_opponent_energy = || {
        let mut mask = [false; ENERGY_DIM];
        if let Some(belief) = belief {
            for energy in belief.seen_opponent_energy(perspective) {
                mask[energy_index(*energy)] = true;
            }
        }
        mask
    };

    GlobalFeatures {
        turn_count: state.turn_count,
        on_the_play,
        is_my_turn: state.current_player == perspective,
        is_setup_phase: state.turn_count <= 2,
        points: [state.points[perspective], state.points[opponent]],
        draw_pile: [
            state.decks[perspective].cards.len(),
            state.decks[opponent].cards.len(),
        ],
        hand_size: [state.hands[perspective].len(), state.hands[opponent].len()],
        discard_size: [
            state.discard_piles[perspective].len(),
            state.discard_piles[opponent].len(),
        ],
        deck_energy_types: [declared_energy(perspective), seen_opponent_energy()],
        energy_zone: [
            [
                state.energy_zone[perspective].current,
                state.energy_zone[perspective].next,
            ],
            [
                state.energy_zone[opponent].current,
                state.energy_zone[opponent].next,
            ],
        ],
        energy_already_attached: [
            energy_already_attached(perspective),
            energy_already_attached(opponent),
        ],
        discard_energy: [
            energy_counts(state.discard_energies[perspective].iter()),
            energy_counts(state.discard_energies[opponent].iter()),
        ],
        has_stadium: state.active_stadium.is_some(),
        stadium_id: state
            .active_stadium
            .as_ref()
            .map(|stadium| card_index(stadium.get_card_id()))
            .unwrap_or(PAD_INDEX),
        has_played_support: turn_player_flag(state.has_played_support),
        has_retreated: turn_player_flag(state.has_retreated),
        has_used_stadium: by_role(state.has_used_stadium),
        knocked_out_by_opponent_attack: [
            state.knocked_out_by_opponent_attack_this_turn,
            state.knocked_out_by_opponent_attack_last_turn,
        ],
    }
}

// ---------------------------------------------------------------------------------------------
// Legality projection
// ---------------------------------------------------------------------------------------------

/// The legality bits of the token features, read off `generate_possible_actions` once.
#[derive(Debug, Default)]
struct Legality {
    evolve_slots: HashSet<usize>,
    evolve_cards: HashSet<String>,
    ability_slots: HashSet<usize>,
    playable_trainers: HashSet<String>,
}

impl Legality {
    fn project(legal_actions: &[Action], perspective: usize) -> Self {
        let mut legality = Self::default();
        for action in legal_actions.iter().filter(|a| a.actor == perspective) {
            match &action.action {
                SimpleAction::Evolve {
                    evolution,
                    in_play_idx,
                    ..
                } => {
                    legality.evolve_slots.insert(*in_play_idx);
                    legality.evolve_cards.insert(evolution.get_id());
                }
                SimpleAction::UseAbility { in_play_idx } => {
                    legality.ability_slots.insert(*in_play_idx);
                }
                SimpleAction::Play { trainer_card } => {
                    legality.playable_trainers.insert(trainer_card.id.clone());
                }
                _ => {}
            }
        }
        legality
    }
}

// ---------------------------------------------------------------------------------------------
// Pokémon tokens
// ---------------------------------------------------------------------------------------------

fn is_pokemon_token(card: &Card) -> bool {
    matches!(card, Card::Pokemon(_)) || card.is_fossil()
}

fn is_trainer_token(card: &Card) -> bool {
    matches!(card, Card::Trainer(_)) && !card.is_fossil()
}

/// The opponent's hand as `perspective` is entitled to see it (§1.3.6.2): the cards with a live
/// `Hand` position marker.
fn revealed_opponent_hand(belief: Option<&BeliefTracker>, perspective: usize) -> Vec<Card> {
    let Some(belief) = belief else {
        return Vec::new();
    };
    expand(belief.known_opponent_hand(perspective))
}

/// Cards the observer knows the opponent holds without knowing where — the monotone `presence`
/// half of the overlay, netted against what is visible right now.
///
/// The netting is the whole difficulty (§1.3.6.2). `presence` never decreases, so a revealed card
/// the opponent then played still counts; without subtracting the public copies, the same card
/// would be rendered twice — once as the discard token everyone can see, once as a claim that it
/// is still hidden.
fn hidden_elsewhere(
    state: &State,
    belief: Option<&BeliefTracker>,
    perspective: usize,
    opponent: usize,
) -> Vec<Card> {
    let Some(belief) = belief else {
        return Vec::new();
    };
    let mut public: HashMap<CardId, u32> = HashMap::new();
    let board = state.in_play_pokemon[opponent]
        .iter()
        .flatten()
        .flat_map(|played| {
            std::iter::once(&played.card)
                .chain(played.cards_behind.iter())
                .chain(played.attached_tool.iter())
        });
    for card in board.chain(state.discard_piles[opponent].iter()) {
        *public.entry(card.get_card_id()).or_insert(0) += 1;
    }
    expand(belief.opponent_hidden_elsewhere(perspective, &public))
}

/// One card per known copy, ordered by [`card_index`].
///
/// The ordering is load-bearing. The overlay answers with a `HashMap`, so leaving its iteration
/// order alone would let two observations of one state present the same cards as different banks,
/// and the encoder reads a bank, not a set.
fn expand(known: HashMap<CardId, u32>) -> Vec<Card> {
    let mut known: Vec<(CardId, u32)> = known.into_iter().collect();
    known.sort_unstable_by_key(|(card, _)| card_index(*card));
    known
        .into_iter()
        .flat_map(|(card, count)| std::iter::repeat_n(get_card_by_enum(card), count as usize))
        .collect()
}

fn board_token(
    state: &State,
    player: usize,
    slot: usize,
    played: &PlayedCard,
    allied: bool,
    legality: &Legality,
) -> PokemonToken {
    let card_id = played.card.get_card_id();
    let (card, species, line) = identity_indices(card_id);
    let total_hp = played.get_effective_total_hp().max(1);

    let printed_retreat = played
        .card
        .get_retreat_cost()
        .map(|cost| cost.len() as i32)
        .unwrap_or(0);
    let effective_retreat = get_retreat_cost(state, played).len() as i32;

    PokemonToken {
        card_id: card,
        species_id: species,
        line_id: line,
        tool_id: played
            .attached_tool
            .as_ref()
            .map(|tool| card_index(tool.get_card_id()))
            .unwrap_or(PAD_INDEX),
        zone: TokenZone::Board,
        allied,
        slot: Some(slot),
        remaining_hp_ratio: played.get_remaining_hp() as f32 / total_hp as f32,
        attached_energy: energy_counts(played.get_effective_attached_energy(state, player).iter()),
        retreat_cost_delta: effective_retreat - printed_retreat,
        status: [
            played.is_poisoned(),
            played.is_paralyzed(),
            played.is_asleep(),
            played.is_burned(),
            played.is_confused(),
        ],
        can_evolve_this_turn: allied && legality.evolve_slots.contains(&slot),
        ability_used: played.ability_used,
        ability_activatable_now: allied && legality.ability_slots.contains(&slot),
        ability_will_proc: ability_will_proc(state, player, slot, played),
        has_tool: played.attached_tool.is_some(),
    }
}

fn off_board_pokemon_token(
    card: &Card,
    zone: TokenZone,
    allied: bool,
    legality: &Legality,
) -> PokemonToken {
    let (card_id, species, line) = identity_indices(card.get_card_id());
    PokemonToken {
        card_id,
        species_id: species,
        line_id: line,
        tool_id: PAD_INDEX,
        zone,
        allied,
        slot: None,
        // Off the board a card carries no damage; the ratio is of its *own* HP, so it is full.
        remaining_hp_ratio: 1.0,
        attached_energy: [0; ENERGY_DIM],
        retreat_cost_delta: 0,
        status: [false; 5],
        can_evolve_this_turn: allied
            && zone == TokenZone::Hand
            && legality.evolve_cards.contains(&card.get_id()),
        ability_used: false,
        ability_activatable_now: false,
        ability_will_proc: false,
        has_tool: false,
    }
}

/// The typed start/end-of-turn abilities whose condition currently holds. Limited to the typed
/// `AbilityMechanic` vocabulary — text-only passive triggers stay at 0 (§1.2.9).
fn ability_will_proc(state: &State, player: usize, slot: usize, played: &PlayedCard) -> bool {
    let Some(mechanic) = get_ability_mechanic(&played.card) else {
        return false;
    };
    let opponent = (player + 1) % 2;
    match AbilityMechanicDiscriminants::from(mechanic) {
        // Unconditional while in play.
        AbilityMechanicDiscriminants::CheckupDamageToOpponentActive
        | AbilityMechanicDiscriminants::CheckupDamageToAllOpponentPokemon
        | AbilityMechanicDiscriminants::StartTurnRandomPokemonToHand => true,
        // Active-only.
        AbilityMechanicDiscriminants::EndTurnDrawCardIfActive
        | AbilityMechanicDiscriminants::EndTurnHealSelfIfActive => slot == 0,
        // Darkrai's Bad Dreams only fires on a sleeping defender.
        AbilityMechanicDiscriminants::BadDreamsEndOfTurn => state.in_play_pokemon[opponent][0]
            .as_ref()
            .is_some_and(|active| active.is_asleep()),
        AbilityMechanicDiscriminants::EndFirstTurnAttachEnergyToSelf => state.turn_count <= 2,
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------
// Attack tokens
// ---------------------------------------------------------------------------------------------

/// Every attack a board Pokémon could use, as `(source card, slot on that card, attack)`.
///
/// This is the *affordance* enumeration, not the legality one: the engine only ever generates
/// attack actions for the turn player's active, while the threat matrix needs both sides and the
/// bench. It mirrors the engine's rules — own attacks, plus the earlier stages recorded in
/// `cards_behind` while Celebi's Time Recall is in play, deduplicated on the whole `Attack`.
pub(crate) fn available_attacks(
    state: &State,
    player: usize,
    played: &PlayedCard,
) -> Vec<(Card, usize, Attack)> {
    if played.is_fossil() {
        return Vec::new();
    }
    let mut sources = vec![played.card.clone()];
    let time_recall_active = state.enumerate_in_play_pokemon(player).any(|(_, pokemon)| {
        matches!(
            get_ability_mechanic(&pokemon.card),
            Some(AbilityMechanic::TimeRecall)
        )
    });
    if time_recall_active {
        sources.extend(played.cards_behind.iter().cloned());
    }

    let mut offered: Vec<Attack> = Vec::new();
    let mut available = Vec::new();
    for source in sources {
        for (slot, attack) in source
            .get_attacks()
            .into_iter()
            .take(MAX_ATTACKS_PER_CARD)
            .enumerate()
        {
            if offered.contains(&attack) {
                continue;
            }
            offered.push(attack.clone());
            available.push((source.clone(), slot, attack));
        }
    }
    available
}

// The parent Pokémon's identity reaches the token through four separate arguments (board ref, row,
// source card, attack slot) because an attack can be borrowed from an earlier stage (§1.2.5) —
// bundling them would only move the widening one level up.
#[allow(clippy::too_many_arguments)]
fn attack_token(
    state: &State,
    scratch: &mut ProjectionScratch,
    attacker: (usize, usize),
    parent_pokemon_ref: u32,
    source: Card,
    attack_slot: usize,
    attack: &Attack,
    allied: bool,
) -> AttackToken {
    let (player, _) = attacker;
    let opponent = (player + 1) % 2;

    // Payability and threat come from one call, evaluated on one projection (§1.2.5) — computing
    // them on different states is what allowed `can_pay = 0` with a non-zero threat.
    let affordance = scratch.attack_affordance(attacker, attack);
    let mut threat = [0.0; BOARD_SLOTS];
    let mut is_lethal = [false; BOARD_SLOTS];
    for (defender_slot, estimate) in affordance.threat.iter().enumerate() {
        let Some(defender) = state.in_play_pokemon[opponent][defender_slot].as_ref() else {
            continue;
        };
        let remaining = defender.get_remaining_hp().max(1);
        threat[defender_slot] = estimate.expected / remaining as f32;
        is_lethal[defender_slot] = estimate.is_lethal_against(defender.get_remaining_hp());
    }

    AttackToken {
        parent_pokemon_ref,
        src_card_id: card_index(source.get_card_id()),
        attack_slot,
        allied,
        can_pay: affordance.can_pay,
        deficit: affordance.deficit,
        surplus: affordance.surplus,
        threat,
        is_lethal,
    }
}

// ---------------------------------------------------------------------------------------------
// Trainer tokens
// ---------------------------------------------------------------------------------------------

fn trainer_token(
    state: &State,
    card: &Card,
    zone: TokenZone,
    owner: usize,
    allied: bool,
    legality: &Legality,
) -> TrainerToken {
    let trainer = card.as_trainer();
    let playable_now =
        allied && zone == TokenZone::Hand && legality.playable_trainers.contains(&trainer.id);

    TrainerToken {
        card_id: card_index(card.get_card_id()),
        target_ids: trainer_targeting(&trainer).target_ids.clone(),
        zone,
        allied,
        playable_now,
        activation_condition_met: allied && activation_condition_met(state, owner, &trainer),
    }
}

/// Whether the card's *own* condition holds **for its owner**, with the once-per-turn Supporter /
/// no-Item gates lifted. `trainer_move_generation_implementation` is the per-card predicate
/// underneath `generate_possible_trainer_actions`; querying it with the owner rather than the turn
/// player separates "this card would do something" from "I am allowed to play a card right now" —
/// crucially, an off-turn perspective's cards are evaluated against *their* board.
fn activation_condition_met(
    state: &State,
    owner: usize,
    trainer: &crate::models::TrainerCard,
) -> bool {
    if trainer.trainer_card_type == TrainerType::Tool
        || trainer.trainer_card_type == TrainerType::Stadium
    {
        // Tools and Stadiums have no separate activation condition beyond playability.
        return true;
    }
    crate::move_generation::trainer_move_generation_implementation(state, owner, trainer)
        .is_some_and(|actions| !actions.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_ids::CardId;
    use crate::models::PlayedCard;
    use crate::test_support::get_test_game_with_board;

    #[test]
    fn banks_have_the_spec_widths() {
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1033Charmander)],
        );
        let state = game.get_state_clone();
        let (_, legal) = state.generate_possible_actions();
        let wire = get_observation(&state, 0, &legal, None, None).to_wire();

        assert_eq!(wire.global.len(), GLOBAL_DIM);
        assert_eq!(wire.pokemon.len(), MAX_POKEMON_TOKENS * POKEMON_DYNAMIC_DIM);
        assert_eq!(wire.pokemon_indices.len(), MAX_POKEMON_TOKENS * 4);
        assert_eq!(wire.attack.len(), MAX_ATTACK_TOKENS * ATTACK_DYNAMIC_DIM);
        assert_eq!(wire.attack_indices.len(), MAX_ATTACK_TOKENS * 2);
        assert_eq!(wire.trainer.len(), MAX_TRAINER_TOKENS * TRAINER_DYNAMIC_DIM);
        assert_eq!(wire.trainer_indices.len(), MAX_TRAINER_TOKENS);
        assert_eq!(
            wire.trainer_target_indices.len(),
            MAX_TRAINER_TOKENS * MAX_TRAINER_TARGET_IDS * 2
        );
        assert_eq!(wire.history.len(), HISTORY_LEN * HISTORY_DYNAMIC_DIM);
        assert_eq!(wire.history_indices.len(), HISTORY_LEN * 2);
    }

    /// [`ZONE_FEATURE_OFFSET`] is a claim about two encoders, and §1.5.6's zone-split attention
    /// read-out is the only consumer that can be silently wrong if it drifts: a field pushed ahead
    /// of the one-hot would relabel every `attn_focus/*/<family>.<zone>` series without failing
    /// anything else. Both families are checked in every zone, including the padding row: a padded
    /// slot must carry no zone at all, which is what makes the five indicators a partition of the
    /// *real* tokens rather than of the bank.
    #[test]
    fn the_zone_one_hot_leads_both_dynamic_blocks() {
        for zone in [
            TokenZone::Board,
            TokenZone::Hand,
            TokenZone::Deck,
            TokenZone::Discard,
            TokenZone::Unknown,
        ] {
            let pokemon = PokemonToken {
                card_id: 1,
                species_id: 1,
                line_id: 1,
                tool_id: 0,
                zone,
                allied: true,
                slot: None,
                remaining_hp_ratio: 1.0,
                attached_energy: [0; ENERGY_DIM],
                retreat_cost_delta: 0,
                status: [false; 5],
                can_evolve_this_turn: false,
                ability_used: false,
                ability_activatable_now: false,
                ability_will_proc: false,
                has_tool: false,
            };
            let trainer = TrainerToken {
                card_id: 1,
                target_ids: Vec::new(),
                zone,
                allied: true,
                playable_now: false,
                activation_condition_met: false,
            };
            let expected: Vec<f32> = (0..TokenZone::DIM)
                .map(|index| f32::from(index == zone.index()))
                .collect();
            for features in [pokemon.features(), trainer.features()] {
                assert_eq!(
                    features[ZONE_FEATURE_OFFSET..ZONE_FEATURE_OFFSET + TokenZone::DIM],
                    expected[..],
                    "{zone:?} is not the leading one-hot"
                );
            }
        }
    }
}
