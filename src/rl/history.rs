//! History tokens — the opponent's action trace (§1.2.7).
//!
//! The fourth token family, and the only one that is **not** permutation-invariant: an ordered
//! trace of the opponent's last [`HISTORY_LEN`] observable action *choices*. Because there is no
//! centralized critic (hidden information stays part of the game dynamics, never privileged to the
//! value head), this trace is the model's only belief-bearing signal. It therefore encodes **what
//! the opponent chose**, never an outcome, and stays as lean as possible.
//!
//! The scope rules, verbatim from the spec and enforced here:
//!
//! - **Opponent only** — [`ActionTrace::tokens_for`] never returns the observer's own actions.
//! - **Choices only** — engine-internal frames ([`is_traceable`]) and single-candidate
//!   auto-resolutions never enter.
//! - **No deltas / outcomes** — only the action identity.
//! - **Public-index rule** — the `card_id` rides along only if the referenced card is public
//!   (board / played / discard). A choice referencing a hidden card enters with `card_id = 0`.
//!   This one rule is what keeps the trace a leak-free proto-belief.
//! - **Crosses turns** — FIFO over the 20 most recent qualifying decisions of *that* player.
//!
//! `head_id` reuses the engine enumeration (`discriminant(SimpleAction)`), so it auto-tracks any
//! change to the action space — symmetric to how the policy emits its own actions.

use std::collections::{HashMap, VecDeque};
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use strum::{EnumCount, IntoEnumIterator};

use crate::actions::{Action, SimpleAction, SimpleActionDiscriminants};
use crate::card_ids::CardId;
use crate::State;

use super::ids::{card_index, PAD_INDEX};
use super::HORIZON;

/// Length of the per-player FIFO.
pub const HISTORY_LEN: usize = 20;

/// Width of the head-embedding table, PAD row included.
pub const HEAD_TABLE_SIZE: usize = SimpleActionDiscriminants::COUNT + 1;

/// The `recency` block: step offset and turn offset.
pub const HISTORY_DYNAMIC_DIM: usize = 2;

/// One recorded decision, before it is turned into a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEntry {
    /// Who made the choice.
    pub actor: usize,
    /// `discriminant(SimpleAction)`, 1-based (0 is PAD).
    pub head_id: u32,
    /// Public referenced card, or [`PAD_INDEX`] for none / hidden.
    pub card_id: u32,
    /// Engine tick at which the choice was made.
    pub step: u32,
    /// Turn at which the choice was made.
    pub turn: u8,
}

/// A History token on the wire: two indices resolved in-model to embeddings, two recency floats.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HistoryToken {
    /// Public referenced card; `0` = none / hidden.
    pub card_id: u32,
    /// Action-family index, resolved in-model to a learned head embedding.
    pub head_id: u32,
    /// `(t − t_a) / H`.
    pub step_recency: f32,
    /// `(turn − turn_a) / H`.
    pub turn_recency: f32,
}

impl HistoryToken {
    /// The two floats this token puts on the wire.
    pub fn features(&self) -> [f32; HISTORY_DYNAMIC_DIM] {
        [
            self.step_recency.clamp(0.0, 1.0),
            self.turn_recency.clamp(0.0, 1.0),
        ]
    }
}

/// The per-player decision trace. Maintained alongside the engine state (like the belief overlay),
/// never inside it: it is observer bookkeeping, not part of the game's identity.
#[derive(Debug, Clone, Default)]
pub struct ActionTrace {
    entries: [VecDeque<TraceEntry>; 2],
    step: u32,
}

impl ActionTrace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Engine ticks seen so far — the `t` of the recency encoding.
    pub fn step(&self) -> u32 {
        self.step
    }

    /// Offer a resolved decision frame to the trace, then advance the tick.
    ///
    /// `candidate_count` is the size of the frame the actor faced: a frame with a single candidate
    /// was auto-resolved by the engine and is not a choice. Call this with the state as it was
    /// **before** the action is applied, so board references still resolve.
    pub fn record(&mut self, state: &State, action: &Action, candidate_count: usize) {
        if candidate_count > 1 && is_traceable(&action.action) {
            let entry = TraceEntry {
                actor: action.actor,
                head_id: head_id(&action.action),
                card_id: public_card_index(state, action.actor, &action.action),
                step: self.step,
                turn: state.turn_count,
            };
            let queue = &mut self.entries[action.actor];
            if queue.len() == HISTORY_LEN {
                queue.pop_front();
            }
            queue.push_back(entry);
        }
        self.step += 1;
    }

    /// The tokens `observer` may see: the *other* player's trace, oldest first, with recency
    /// measured against the current tick and turn.
    pub fn tokens_for(&self, observer: usize, state: &State) -> Vec<HistoryToken> {
        let opponent = (observer + 1) % 2;
        self.entries[opponent]
            .iter()
            .map(|entry| HistoryToken {
                card_id: entry.card_id,
                head_id: entry.head_id,
                step_recency: self.step.saturating_sub(entry.step) as f32 / HORIZON,
                turn_recency: state.turn_count.saturating_sub(entry.turn) as f32 / HORIZON,
            })
            .collect()
    }

    /// Raw entries of one actor, oldest first (introspection and tests).
    pub fn entries_of(&self, actor: usize) -> &VecDeque<TraceEntry> {
        &self.entries[actor]
    }
}

/// Whether an action is a genuine decision rather than an engine-internal frame. `DrawCard`,
/// `ApplyDamage` and `ScheduleDelayedSpotDamage` are resolved by the engine and never chosen.
pub fn is_traceable(action: &SimpleAction) -> bool {
    !matches!(
        action,
        SimpleAction::DrawCard { .. }
            | SimpleAction::ApplyDamage { .. }
            | SimpleAction::ScheduleDelayedSpotDamage { .. }
    )
}

static HEAD_INDEX: LazyLock<HashMap<SimpleActionDiscriminants, u32>> = LazyLock::new(|| {
    SimpleActionDiscriminants::iter()
        .enumerate()
        .map(|(index, discriminant)| (discriminant, index as u32 + 1))
        .collect()
});

/// `discriminant(SimpleAction)` as a 1-based index into the head-embedding table (0 is PAD).
pub fn head_id(action: &SimpleAction) -> u32 {
    HEAD_INDEX[&SimpleActionDiscriminants::from(action)]
}

/// The card an action references, **only if that card is public** (board, played, or discarded).
/// Anything touching a hidden zone resolves to [`PAD_INDEX`] — this is the whole leak-freedom
/// argument of the History token.
pub fn public_card_index(state: &State, actor: usize, action: &SimpleAction) -> u32 {
    let board_card = |player: usize, idx: usize| {
        state.in_play_pokemon[player][idx]
            .as_ref()
            .map(|played| card_index(played.card.get_card_id()))
            .unwrap_or(PAD_INDEX)
    };

    match action {
        // The card becomes public by being played onto the board or into the discard.
        SimpleAction::Place(card, _) => card_index(card.get_card_id()),
        SimpleAction::Evolve { evolution, .. } => card_index(evolution.get_card_id()),
        SimpleAction::Play { trainer_card } => CardId::from_card_id(&trainer_card.id)
            .map(card_index)
            .unwrap_or(PAD_INDEX),
        SimpleAction::AttachTool { tool_card, .. } => card_index(tool_card.get_card_id()),
        SimpleAction::DiscardOwnCards { cards } => cards
            .first()
            .map(|card| card_index(card.get_card_id()))
            .unwrap_or(PAD_INDEX),
        // Pointing at a card in *our* hand: the observer of this entry is its owner, and the card
        // ends up shuffled or discarded either way.
        SimpleAction::ShuffleOpponentSupporter { supporter_card }
        | SimpleAction::DiscardOpponentSupporter { supporter_card } => {
            card_index(supporter_card.get_card_id())
        }

        // Choices that point at a slot on a public board.
        SimpleAction::Attack(_) => board_card(actor, 0),
        SimpleAction::UseAbility { in_play_idx }
        | SimpleAction::Heal { in_play_idx, .. }
        | SimpleAction::HealAndDiscardEnergy { in_play_idx, .. }
        | SimpleAction::AttachFromDiscard { in_play_idx, .. }
        | SimpleAction::AttachTypedFromDiscard { in_play_idx, .. }
        | SimpleAction::DiscardFossil { in_play_idx }
        | SimpleAction::ReturnPokemonToHand { in_play_idx }
        | SimpleAction::ShuffleInPlayPokemonIntoDeck { in_play_idx } => {
            board_card(actor, *in_play_idx)
        }
        SimpleAction::Retreat(in_play_idx) => board_card(actor, *in_play_idx),
        SimpleAction::Activate {
            player,
            in_play_idx,
        }
        | SimpleAction::DiscardToolFromPokemon {
            player,
            in_play_idx,
        } => board_card(*player, *in_play_idx),

        // Everything else is nullary, energy-only, or reaches into a hidden zone.
        _ => PAD_INDEX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::get_card_by_enum;

    fn action(actor: usize, action: SimpleAction) -> Action {
        Action {
            actor,
            action,
            is_stack: false,
        }
    }

    #[test]
    fn head_ids_are_distinct_and_leave_room_for_pad() {
        let mut seen = std::collections::HashSet::new();
        for discriminant in SimpleActionDiscriminants::iter() {
            let index = HEAD_INDEX[&discriminant];
            assert!(index >= 1 && (index as usize) < HEAD_TABLE_SIZE);
            assert!(seen.insert(index));
        }
        assert_eq!(seen.len(), SimpleActionDiscriminants::COUNT);
        assert_eq!(
            head_id(&SimpleAction::EndTurn),
            HEAD_INDEX[&SimpleActionDiscriminants::EndTurn]
        );
    }

    #[test]
    fn internal_frames_and_forced_frames_are_not_traced() {
        let state = State::default();
        let mut trace = ActionTrace::new();

        trace.record(&state, &action(1, SimpleAction::DrawCard { amount: 1 }), 3);
        assert!(trace.entries_of(1).is_empty(), "internal frame");

        trace.record(&state, &action(1, SimpleAction::EndTurn), 1);
        assert!(trace.entries_of(1).is_empty(), "single-candidate frame");

        trace.record(&state, &action(1, SimpleAction::EndTurn), 2);
        assert_eq!(trace.entries_of(1).len(), 1);
    }

    #[test]
    fn the_trace_is_opponent_only_and_ordered() {
        let mut state = State::default();
        state.turn_count = 5;
        let mut trace = ActionTrace::new();

        let bulbasaur = get_card_by_enum(CardId::A1001Bulbasaur);
        trace.record(&state, &action(1, SimpleAction::Place(bulbasaur, 0)), 4);
        trace.record(&state, &action(0, SimpleAction::EndTurn), 4);
        state.turn_count = 7;
        trace.record(&state, &action(1, SimpleAction::EndTurn), 4);

        let tokens = trace.tokens_for(0, &state);
        assert_eq!(tokens.len(), 2, "only player 1's choices");
        assert_eq!(tokens[0].card_id, card_index(CardId::A1001Bulbasaur));
        assert_eq!(tokens[1].card_id, PAD_INDEX, "EndTurn is nullary");
        assert!(
            tokens[0].step_recency > tokens[1].step_recency,
            "oldest first"
        );
        assert_eq!(tokens[1].turn_recency, 0.0);

        assert_eq!(trace.tokens_for(1, &state).len(), 1, "player 0's EndTurn");
    }

    #[test]
    fn a_choice_on_a_hidden_card_carries_no_index() {
        let state = State::default();
        let mut trace = ActionTrace::new();
        let bulbasaur = get_card_by_enum(CardId::A1001Bulbasaur);

        // Pokémon Communication puts a hand Pokémon into the deck — the opponent never sees it.
        trace.record(
            &state,
            &action(
                1,
                SimpleAction::CommunicatePokemon {
                    hand_pokemon: bulbasaur,
                },
            ),
            3,
        );
        assert_eq!(trace.entries_of(1)[0].card_id, PAD_INDEX);
    }

    #[test]
    fn the_fifo_keeps_the_most_recent_twenty() {
        let state = State::default();
        let mut trace = ActionTrace::new();
        for _ in 0..(HISTORY_LEN + 5) {
            trace.record(&state, &action(1, SimpleAction::EndTurn), 2);
        }
        assert_eq!(trace.entries_of(1).len(), HISTORY_LEN);
        assert_eq!(trace.entries_of(1)[0].step, 5, "oldest five dropped");
    }
}
