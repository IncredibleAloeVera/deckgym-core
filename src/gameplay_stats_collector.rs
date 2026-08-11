//! Gameplay statistics harvested from simulations, shaped for the deckbuilder's label tables
//! (RL_ARCHITECTURE §1.5.7).
//!
//! Two rules govern everything here:
//!
//! 1. **Counts, never rates.** Every quantity is stored as a raw count next to the denominator it
//!    would be divided by. A rate is a normalization decision taken too early and irreversibly;
//!    counts let the offline aggregation pick the conditioning it wants (per copy, per deck slot,
//!    or conditional on the card having been drawn).
//! 2. **`card_id@copies` is the identity, not `card_id`.** A card run in 1 copy and the same card
//!    run in 2 copies have different draw dynamics and different marginal value, so they are
//!    different rows (see [`CardSlotKey`] and RL_ARCHITECTURE §1.6.1). Statistics are attributed
//!    to the printed card, never to a physical copy — the engine gives `PlayedCard` no instance
//!    identity, and the deckbuilder's atom is the `card_id` anyway.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use log::warn;
use uuid::Uuid;

use crate::{
    actions::{Action, SimpleAction},
    state::GameOutcome,
    Deck, State,
};

/// Stable content hash of a `Deck` (cards + normalized energy set).
pub type DeckId = u64;

/// Who piloted a deck, in the vocabulary the run already names its opponents with (`w`, `e2`,
/// `pool:b000001234`, `baked:proto`, `learner`) — see `rl::train::rating::OpponentId`.
pub type PilotId = String;

/// What a pilot slot says when nobody set one: a mass-labeling pass that does not care, or a
/// caller that forgot. Kept as a value rather than an `Option` so the column is never absent from
/// a row, only uninformative.
pub const UNKNOWN_PILOT: &str = "unknown";

/// The key of the per-deck table: the decklist, plus who played it and who they faced.
///
/// The pilots are part of the identity rather than a note attached to it, because every counter
/// under them is a *behavioural* measurement: `times_played` under a `w` pilot measures the
/// weighted-random draw, not the card. Merging pilots into one row averages a policy's judgement
/// with a coin flip and the split can never be recovered (RL_ARCHITECTURE §1.5.7). The opponent's
/// pilot is in the key for the same reason on the other side — a winrate is a property of the
/// matchup, and §1.6.2 fits it as one.
///
/// `opponent_deck` is in the key for the same reason: without it, every opposing decklist a pilot
/// ever ran collapses into one row the moment two curriculum archetypes coexist, and the winrate
/// against any one of them is gone for good — no lower table records it. Its `DeckId` alone is
/// enough; resolving the archetype it belongs to is an offline join against `deck_dictionary`, the
/// same way `deck` itself is resolved.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeckSlotKey {
    pub deck: DeckId,
    pub pilot: PilotId,
    pub opponent_deck: DeckId,
    pub opponent_pilot: PilotId,
}

/// Sorts a deck's cards by id so its hash is a function of the *decklist*, not of the shuffle.
/// Without this the same deck would get a fresh [`DeckId`] every game.
fn canonicalize(deck: &mut Deck) {
    deck.cards.sort_by_key(|card| card.get_id());
}

fn deck_id(deck: &Deck) -> DeckId {
    let mut hasher = DefaultHasher::new();
    deck.hash(&mut hasher);
    hasher.finish()
}

/// The deckbuilder-facing identity of a deck slot: a printed card **plus the number of copies its
/// owner's deck runs**.
///
/// Keeping `copies_in_deck` in the key is what lets the offline fit answer "is the 2nd copy worth
/// its slot" — and it is what keeps a `card_id × card_id` pair from ever standing in for a
/// 2-copy card in the Coherence head.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CardSlotKey {
    pub card_id: String,
    pub copies_in_deck: u8,
}

/// Per-`(deck, card_id@copies)` counters. All fields are counts; every ratio worth forming has its
/// denominator present in this same struct.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CardStats {
    /// Games in which the owning deck was played (the outermost denominator).
    pub games: u32,
    /// Printed HP; 0 for Trainers. Denominator for the damage-absorption ratios.
    pub base_hp: u32,

    // --- availability -----------------------------------------------------------------------
    /// Copies that left the deck at least once, summed over games (≤ `copies_in_deck × games`).
    pub copies_drawn: u32,
    /// Times the card was put into play or played, summed over games. This counts *plays*, not
    /// copies, and can legitimately exceed `copies_drawn`: an effect that returns a Pokémon to
    /// hand (Koga) or a Trainer to the deck lets the same copy be played again.
    pub times_played: u32,
    /// Games where no copy was ever drawn.
    pub games_never_drawn: u32,
    /// Games where a copy was drawn and none was ever played — the "dead card" signal.
    pub games_drawn_never_played: u32,
    /// Copies still sitting in hand at game end, summed over games.
    pub ended_in_hand: u32,
    /// Turn of first play, summed over the `first_play_games` games where it happened.
    pub first_play_turn_sum: u32,
    pub first_play_games: u32,

    // --- ability ----------------------------------------------------------------------------
    /// `UseAbility` actions taken on this card (counts re-use across turns).
    pub ability_activations: u32,
    /// Turns on which the engine offered `UseAbility` on this card — the denominator that turns
    /// `ability_activations` into a "dead ability" signal. Read off `playable_actions`, so the
    /// engine stays the authority (RL_ARCHITECTURE §1.3.1).
    pub turns_ability_available: u32,

    // --- offense ----------------------------------------------------------------------------
    pub attacks_used: u32,
    /// Damage this card caused on the opposing board, absolute. Never stored as a share of the
    /// deck's total: shares are compositional and would manufacture anti-synergy between two
    /// attackers in the same deck. The share stays derivable from `DeckStats::damage_dealt_total`.
    pub damage_dealt: u32,
    /// Knock-outs this card caused. The actual currency of the game; damage is the low-variance
    /// proxy for it.
    pub kos_dealt: u32,

    // --- defense ----------------------------------------------------------------------------
    /// Turns spent in the active slot — the exposure denominator. A wall that is never attacked
    /// absorbs nothing, and that is the opponent's decision, not the card's.
    pub turns_active: u32,
    pub turns_benched: u32,
    /// Damage absorbed while active (chosen exposure) vs on the bench (imposed) — kept apart.
    pub damage_taken_active: u32,
    pub damage_taken_bench: u32,
    pub healing_received: u32,
    /// Times knocked out. Absorbing 1.0×HP and dying is not absorbing 0.9×HP and surviving.
    pub times_koed: u32,
}

impl CardStats {
    fn merge_from(&mut self, other: &CardStats) {
        self.games += other.games;
        self.base_hp = self.base_hp.max(other.base_hp);
        self.copies_drawn += other.copies_drawn;
        self.times_played += other.times_played;
        self.games_never_drawn += other.games_never_drawn;
        self.games_drawn_never_played += other.games_drawn_never_played;
        self.ended_in_hand += other.ended_in_hand;
        self.first_play_turn_sum += other.first_play_turn_sum;
        self.first_play_games += other.first_play_games;
        self.ability_activations += other.ability_activations;
        self.turns_ability_available += other.turns_ability_available;
        self.attacks_used += other.attacks_used;
        self.damage_dealt += other.damage_dealt;
        self.kos_dealt += other.kos_dealt;
        self.turns_active += other.turns_active;
        self.turns_benched += other.turns_benched;
        self.damage_taken_active += other.damage_taken_active;
        self.damage_taken_bench += other.damage_taken_bench;
        self.healing_received += other.healing_received;
        self.times_koed += other.times_koed;
    }
}

/// Per-deck counters. Outcomes are kept as raw win/loss/tie counts (plus point margins, which are
/// strictly more informative than the binary outcome at no extra cost).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeckStats {
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub ties: u32,
    pub games_on_the_play: u32,
    pub points_scored: u32,
    pub points_conceded: u32,
    pub turns_sum: u32,
    /// Total damage the deck put on the opposing board — the denominator for a per-card damage
    /// share, kept separately so the absolute is never lost.
    pub damage_dealt_total: u32,
    pub deck_out_games: u32,
    pub deck_out_turn_sum: u32,
    pub hand_size_sum: u32,
    pub hand_size_samples: u32,
    pub cards: HashMap<CardSlotKey, CardStats>,
}

impl DeckStats {
    fn merge_from(&mut self, other: &DeckStats) {
        self.games += other.games;
        self.wins += other.wins;
        self.losses += other.losses;
        self.ties += other.ties;
        self.games_on_the_play += other.games_on_the_play;
        self.points_scored += other.points_scored;
        self.points_conceded += other.points_conceded;
        self.turns_sum += other.turns_sum;
        self.damage_dealt_total += other.damage_dealt_total;
        self.deck_out_games += other.deck_out_games;
        self.deck_out_turn_sum += other.deck_out_turn_sum;
        self.hand_size_sum += other.hand_size_sum;
        self.hand_size_samples += other.hand_size_samples;
        for (key, stats) in &other.cards {
            self.cards.entry(key.clone()).or_default().merge_from(stats);
        }
    }
}

/// Everything accumulated for one player over one game, keyed by `card_id`. Promoted to
/// `CardSlotKey` at flush time, when the deck's copy counts are known.
#[derive(Default)]
struct GameAccumulator {
    deck: Option<Deck>,
    /// `card_id -> copies the deck runs`.
    copies: HashMap<String, u8>,
    /// Minimum count still in the draw pile over the game; `copies - min` = copies drawn.
    min_in_deck: HashMap<String, u32>,
    played: HashMap<String, u32>,
    first_play_turn: HashMap<String, u8>,
    /// `(card_id, turn)` pairs on which the engine offered `UseAbility`.
    ability_available: HashSet<(String, u8)>,
    per_card: HashMap<String, CardStats>,
    damage_dealt_total: u32,
    deck_out_turn: Option<u8>,
    hand_size_sum: u32,
    hand_size_samples: u32,
}

impl GameAccumulator {
    fn card(&mut self, card_id: &str) -> &mut CardStats {
        self.per_card.entry(card_id.to_string()).or_default()
    }
}

/// Snapshot of one board slot, used to diff damage between consecutive observations.
#[derive(Clone)]
struct SlotSnapshot {
    card_id: String,
    damage: u32,
    remaining_hp: u32,
    /// Board position, 0 = active. Only used to split absorbed damage active vs bench; the diff
    /// itself is keyed on `card_id` so that switches are not read as damage.
    slot: usize,
}

/// Collects deckbuilder-facing gameplay statistics during simulations.
///
/// Damage is measured by diffing consecutive board observations: `on_action` hands us the state
/// *before* each action, so the damage caused by action *k* is visible at action *k+1*. It is
/// credited to the card that was the source of action *k* (attacker, ability holder, or played
/// Trainer), which covers attack damage, ability pings and Trainer burn uniformly.
pub struct GameplayStatsCollector {
    decks: HashMap<DeckSlotKey, DeckStats>,
    /// Resolves a [`DeckId`] back to its deck; without it the hash is not interpretable.
    deck_dictionary: HashMap<DeckId, Deck>,
    num_games: u32,
    /// Who is playing each seat, in seat order. Set by the caller — the engine has no name for a
    /// policy, and the collector cannot infer one from the play.
    pilots: [PilotId; 2],

    // --- per-game state -----------------------------------------------------------------------
    current_game_id: Option<Uuid>,
    current_turn: u8,
    acc: [GameAccumulator; 2],
    prev_board: [[Option<SlotSnapshot>; 4]; 2],
    prev_discard: [HashMap<String, u32>; 2],
    /// Actor and source card of the last observed action, for damage attribution.
    last_source: Option<(usize, String)>,
    captured: bool,
}

impl Default for GameplayStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl GameplayStatsCollector {
    pub fn new() -> Self {
        Self {
            decks: HashMap::new(),
            deck_dictionary: HashMap::new(),
            num_games: 0,
            pilots: [UNKNOWN_PILOT.to_string(), UNKNOWN_PILOT.to_string()],
            current_game_id: None,
            current_turn: 0,
            acc: Default::default(),
            prev_board: Default::default(),
            prev_discard: Default::default(),
            last_source: None,
            captured: false,
        }
    }

    pub fn total_games(&self) -> u32 {
        self.num_games
    }

    /// Names the seats, in seat order. Call before the game starts; a collector that is never told
    /// files its rows under [`UNKNOWN_PILOT`].
    pub fn set_pilots(&mut self, pilots: [PilotId; 2]) {
        self.pilots = pilots;
    }

    /// The harvested tables, keyed by deck content hash and the two pilots.
    pub fn decks(&self) -> &HashMap<DeckSlotKey, DeckStats> {
        &self.decks
    }

    /// Resolves a [`DeckId`] back to the deck it was computed from.
    pub fn deck_dictionary(&self) -> &HashMap<DeckId, Deck> {
        &self.deck_dictionary
    }

    /// Captures each player's decklist from the first observed state.
    ///
    /// At that point nothing has been played or discarded, so `draw pile ∪ hand` *is* the deck —
    /// which keeps the collector independent of how the decks were sampled (file, meta DB, or an
    /// RL deck sampler), at the cost of one consistency warning if the union is not 20 cards.
    fn capture_decks(&mut self, state: &State) {
        for player in 0..2 {
            let mut cards = state.decks[player].cards.clone();
            cards.extend(state.hands[player].iter().cloned());
            if cards.len() != 20 {
                warn!(
                    "player {player}: reconstructed decklist has {} cards, expected 20 — copy \
                     counts and drawn/played denominators will be off for this game",
                    cards.len()
                );
            }

            let mut copies: HashMap<String, u8> = HashMap::new();
            for card in &cards {
                *copies.entry(card.get_id()).or_insert(0) += 1;
            }

            let mut deck = state.decks[player].clone();
            deck.cards = cards;
            canonicalize(&mut deck);
            self.acc[player].deck = Some(deck);
            self.acc[player].copies = copies;
        }
        self.captured = true;
    }

    /// Records the minimum draw-pile count seen per card, from which `copies_drawn` follows.
    fn track_draw_pile(&mut self, state: &State) {
        for player in 0..2 {
            let mut in_deck: HashMap<String, u32> = HashMap::new();
            for card in &state.decks[player].cards {
                *in_deck.entry(card.get_id()).or_insert(0) += 1;
            }
            for card_id in self.acc[player].copies.keys().cloned().collect::<Vec<_>>() {
                let count = in_deck.get(&card_id).copied().unwrap_or(0);
                let entry = self.acc[player]
                    .min_in_deck
                    .entry(card_id)
                    .or_insert(u32::MAX);
                *entry = (*entry).min(count);
            }

            if state.decks[player].cards.is_empty() && self.acc[player].deck_out_turn.is_none() {
                self.acc[player].deck_out_turn = Some(self.current_turn);
            }
        }
    }

    /// Groups a board side by `card_id`, most damaged first.
    fn group_by_card(board: &[Option<SlotSnapshot>; 4]) -> HashMap<String, Vec<SlotSnapshot>> {
        let mut grouped: HashMap<String, Vec<SlotSnapshot>> = HashMap::new();
        for snapshot in board.iter().flatten() {
            grouped
                .entry(snapshot.card_id.clone())
                .or_default()
                .push(snapshot.clone());
        }
        for entries in grouped.values_mut() {
            entries.sort_by_key(|entry| std::cmp::Reverse(entry.damage));
        }
        grouped
    }

    /// Diffs the board against the previous observation and attributes the damage moved.
    ///
    /// The diff is keyed on **`card_id`, not on the board slot**: a switch (Sabrina, gust, retreat,
    /// post-KO promotion) permutes the slots, and a slot-wise diff would read that permutation as a
    /// hit landing on the incoming Pokémon. Copies of the same card on board are paired
    /// most-damaged-first, which is exact whenever damage only accumulates.
    ///
    /// Evolution keeps the damage counters and changes the `card_id`, so the outgoing stage is
    /// counted as having left the board and the new stage as having arrived. An `Evolve` moves no
    /// damage, so nothing is lost — only the pre-evolution's absorbed damage stays on its own row,
    /// which is what attribution to the printed card means.
    fn diff_board(&mut self, state: &State) {
        let mut discard: [HashMap<String, u32>; 2] = Default::default();
        for (player, counts) in discard.iter_mut().enumerate() {
            for card in &state.discard_piles[player] {
                *counts.entry(card.get_id()).or_insert(0) += 1;
            }
        }

        // The seat indexes five parallel arrays here — `state.in_play_pokemon`, `self.prev_board`,
        // `self.prev_discard`, `discard` and `self.acc` — so none of them is the natural subject of
        // an iterator and the range stays the honest form.
        #[allow(clippy::needless_range_loop)]
        for player in 0..2 {
            let mut after_board: [Option<SlotSnapshot>; 4] = Default::default();
            for (slot, entry) in after_board.iter_mut().enumerate() {
                *entry = state.in_play_pokemon[player][slot]
                    .as_ref()
                    .map(|played| SlotSnapshot {
                        card_id: played.card.get_id(),
                        damage: played.get_damage_counters(),
                        remaining_hp: played.get_remaining_hp(),
                        slot,
                    });
            }

            let before = Self::group_by_card(&self.prev_board[player]);
            let after = Self::group_by_card(&after_board);

            for (card_id, before_entries) in &before {
                let empty = Vec::new();
                let after_entries = after.get(card_id).unwrap_or(&empty);

                let mut damage_taken_active = 0;
                let mut damage_taken_bench = 0;
                let mut healed = 0;
                let mut koed = 0;

                let paired = before_entries.len().min(after_entries.len());
                for i in 0..paired {
                    let (was, now) = (&before_entries[i], &after_entries[i]);
                    let delta = now.damage.saturating_sub(was.damage);
                    // Damage lands where the Pokémon stood when it was hit, so the split follows
                    // the *previous* slot — active exposure is chosen, bench damage is imposed.
                    if was.slot == 0 {
                        damage_taken_active += delta;
                    } else {
                        damage_taken_bench += delta;
                    }
                    healed += was.damage.saturating_sub(now.damage);
                }

                // Copies that left the board. It is a knock-out only if the card reached the
                // discard pile — `ReturnPokemonToHand` and `ShuffleInPlayPokemonIntoDeck` empty a
                // slot too, and must not be counted as deaths.
                let departed = before_entries.len().saturating_sub(after_entries.len());
                if departed > 0 {
                    let was_discarded =
                        self.prev_discard[player].get(card_id).copied().unwrap_or(0);
                    let now_discarded = discard[player].get(card_id).copied().unwrap_or(0);
                    let ko_budget =
                        (now_discarded.saturating_sub(was_discarded) as usize).min(departed);
                    for entry in before_entries.iter().take(ko_budget) {
                        if entry.slot == 0 {
                            damage_taken_active += entry.remaining_hp;
                        } else {
                            damage_taken_bench += entry.remaining_hp;
                        }
                    }
                    koed = ko_budget as u32;
                }

                let damage_taken = damage_taken_active + damage_taken_bench;
                if damage_taken == 0 && healed == 0 && koed == 0 {
                    continue;
                }

                {
                    let stats = self.acc[player].card(card_id);
                    stats.damage_taken_active += damage_taken_active;
                    stats.damage_taken_bench += damage_taken_bench;
                    stats.healing_received += healed;
                    stats.times_koed += koed;
                }

                // Credit the other side's acting card, when the damage crossed the board.
                if damage_taken > 0 || koed > 0 {
                    if let Some((actor, source_id)) = self.last_source.clone() {
                        if actor != player {
                            let stats = self.acc[actor].card(&source_id);
                            stats.damage_dealt += damage_taken;
                            stats.kos_dealt += koed;
                            self.acc[actor].damage_dealt_total += damage_taken;
                        }
                    }
                }
            }
        }

        for player in 0..2 {
            for slot in 0..4 {
                self.prev_board[player][slot] =
                    state.in_play_pokemon[player][slot]
                        .as_ref()
                        .map(|played| SlotSnapshot {
                            card_id: played.card.get_id(),
                            damage: played.get_damage_counters(),
                            remaining_hp: played.get_remaining_hp(),
                            slot,
                        });
            }
        }
        self.prev_discard = discard;
    }

    /// Records base HP and which abilities the engine is currently offering.
    fn track_board_affordances(&mut self, state: &State, actor: usize, actions: &[Action]) {
        for player in 0..2 {
            for (_, played) in state.enumerate_in_play_pokemon(player) {
                let card_id = played.card.get_id();
                let base_hp = played.get_base_hp();
                self.acc[player].card(&card_id).base_hp = base_hp;
            }
        }

        let turn = self.current_turn;
        for action in actions {
            if let SimpleAction::UseAbility { in_play_idx } = &action.action {
                if let Some(played) = state.in_play_pokemon[actor][*in_play_idx].as_ref() {
                    self.acc[actor]
                        .ability_available
                        .insert((played.card.get_id(), turn));
                }
            }
        }
    }

    /// Identifies the card responsible for the action about to be applied, and folds in the
    /// per-action counters (plays, ability activations, attacks).
    fn track_action(&mut self, state: &State, actor: usize, action: &Action) {
        let turn = self.current_turn;
        let source = match &action.action {
            SimpleAction::Place(card, _) => {
                let card_id = card.get_id();
                *self.acc[actor].played.entry(card_id.clone()).or_insert(0) += 1;
                self.acc[actor]
                    .first_play_turn
                    .entry(card_id.clone())
                    .or_insert(turn);
                Some(card_id)
            }
            SimpleAction::Evolve { evolution, .. } => {
                let card_id = evolution.get_id();
                *self.acc[actor].played.entry(card_id.clone()).or_insert(0) += 1;
                self.acc[actor]
                    .first_play_turn
                    .entry(card_id.clone())
                    .or_insert(turn);
                Some(card_id)
            }
            SimpleAction::Play { trainer_card } => {
                let card_id = trainer_card.id.clone();
                *self.acc[actor].played.entry(card_id.clone()).or_insert(0) += 1;
                self.acc[actor]
                    .first_play_turn
                    .entry(card_id.clone())
                    .or_insert(turn);
                Some(card_id)
            }
            SimpleAction::UseAbility { in_play_idx } => state.in_play_pokemon[actor][*in_play_idx]
                .as_ref()
                .map(|played| played.card.get_id())
                .inspect(|card_id| {
                    self.acc[actor].card(card_id).ability_activations += 1;
                }),
            SimpleAction::Attack(_) => state.in_play_pokemon[actor][0]
                .as_ref()
                .map(|played| played.card.get_id())
                .inspect(|card_id| {
                    self.acc[actor].card(card_id).attacks_used += 1;
                }),
            // End-of-turn damage (poison, burn) is caused by a status, not by the last card that
            // happened to act. Dropping the attribution at the turn boundary keeps it off the
            // Supporter that ended the turn. Damage that resolves over several frames *within* a
            // turn is still credited to the attack or Trainer that pushed those frames.
            SimpleAction::EndTurn => {
                self.last_source = None;
                None
            }
            _ => None,
        };

        if let Some(card_id) = source {
            self.last_source = Some((actor, card_id));
        }
    }

    fn track_hand_sizes(&mut self, state: &State) {
        for player in 0..2 {
            self.acc[player].hand_size_sum += state.hands[player].len() as u32;
            self.acc[player].hand_size_samples += 1;
        }
    }

    /// Counts one turn of exposure for every Pokémon on the board, split active vs bench.
    fn track_turn_exposure(&mut self, state: &State) {
        for player in 0..2 {
            for slot in 0..4 {
                let Some(card_id) = state.in_play_pokemon[player][slot]
                    .as_ref()
                    .map(|played| played.card.get_id())
                else {
                    continue;
                };
                let stats = self.acc[player].card(&card_id);
                if slot == 0 {
                    stats.turns_active += 1;
                } else {
                    stats.turns_benched += 1;
                }
            }
        }
    }

    /// Folds one game's accumulators into the persistent per-deck tables.
    fn flush_game(&mut self, state: &State, outcome: Option<GameOutcome>) {
        // Both ids are needed before either row is built — seat 0's key wants seat 1's id and
        // vice versa — so this is a pass of its own rather than folded into the loop below.
        let mut ids: [Option<DeckId>; 2] = [None, None];
        for (player, slot) in ids.iter_mut().enumerate() {
            if let Some(deck) = self.acc[player].deck.clone() {
                let id = deck_id(&deck);
                self.deck_dictionary.entry(id).or_insert(deck);
                *slot = Some(id);
            }
        }

        for player in 0..2 {
            let Some(id) = ids[player] else {
                continue;
            };
            let opponent = 1 - player;
            let Some(opponent_id) = ids[opponent] else {
                continue;
            };
            let key = DeckSlotKey {
                deck: id,
                pilot: self.pilots[player].clone(),
                opponent_deck: opponent_id,
                opponent_pilot: self.pilots[opponent].clone(),
            };

            let mut ended_in_hand: HashMap<String, u32> = HashMap::new();
            for card in &state.hands[player] {
                *ended_in_hand.entry(card.get_id()).or_insert(0) += 1;
            }

            let mut ability_turns: HashMap<String, u32> = HashMap::new();
            for (card_id, _) in &self.acc[player].ability_available {
                *ability_turns.entry(card_id.clone()).or_insert(0) += 1;
            }

            let acc = std::mem::take(&mut self.acc[player]);
            let entry = self.decks.entry(key).or_default();

            entry.games += 1;
            match outcome {
                Some(GameOutcome::Win(winner)) if winner == player => entry.wins += 1,
                Some(GameOutcome::Win(_)) => entry.losses += 1,
                Some(GameOutcome::Tie) | None => entry.ties += 1,
            }
            // `on_the_play` is the player that took the first turn; setup is simultaneous.
            if player == 0 {
                entry.games_on_the_play += 1;
            }
            entry.points_scored += state.points[player] as u32;
            entry.points_conceded += state.points[opponent] as u32;
            entry.turns_sum += state.turn_count as u32;
            entry.damage_dealt_total += acc.damage_dealt_total;
            if let Some(turn) = acc.deck_out_turn {
                entry.deck_out_games += 1;
                entry.deck_out_turn_sum += turn as u32;
            }
            entry.hand_size_sum += acc.hand_size_sum;
            entry.hand_size_samples += acc.hand_size_samples;

            // Every card the deck runs gets a row every game, including the ones that were never
            // drawn — that absence is the signal, and dropping the row would silently condition
            // all downstream ratios on "was drawn".
            for (card_id, copies_in_deck) in &acc.copies {
                let mut stats = acc.per_card.get(card_id).cloned().unwrap_or_default();
                stats.games = 1;

                let min_in_deck = acc
                    .min_in_deck
                    .get(card_id)
                    .copied()
                    .unwrap_or(*copies_in_deck as u32);
                let drawn = (*copies_in_deck as u32).saturating_sub(min_in_deck);
                let played = acc.played.get(card_id).copied().unwrap_or(0);
                stats.copies_drawn = drawn;
                stats.times_played = played;
                if drawn == 0 {
                    stats.games_never_drawn = 1;
                } else if played == 0 {
                    stats.games_drawn_never_played = 1;
                }
                stats.ended_in_hand = ended_in_hand.get(card_id).copied().unwrap_or(0);
                if let Some(turn) = acc.first_play_turn.get(card_id) {
                    stats.first_play_turn_sum = *turn as u32;
                    stats.first_play_games = 1;
                }
                stats.turns_ability_available = ability_turns.get(card_id).copied().unwrap_or(0);

                let key = CardSlotKey {
                    card_id: card_id.clone(),
                    copies_in_deck: *copies_in_deck,
                };
                entry.cards.entry(key).or_default().merge_from(&stats);
            }
        }

        self.num_games += 1;
    }
}

impl crate::simulation_event_handler::SimulationEventHandler for GameplayStatsCollector {
    fn on_game_start(&mut self, game_id: Uuid) {
        self.current_game_id = Some(game_id);
        self.current_turn = 0;
        self.acc = Default::default();
        self.prev_board = Default::default();
        self.prev_discard = Default::default();
        self.last_source = None;
        self.captured = false;
    }

    fn on_action(
        &mut self,
        _game_id: Uuid,
        state_before_action: &State,
        actor: usize,
        playable_actions: &[Action],
        action: &Action,
    ) {
        if !self.captured {
            self.capture_decks(state_before_action);
        }
        self.current_turn = state_before_action.turn_count;

        // Attribute the damage caused by the *previous* action, which is only visible now.
        self.diff_board(state_before_action);
        self.track_draw_pile(state_before_action);
        self.track_board_affordances(state_before_action, actor, playable_actions);

        if matches!(action.action, SimpleAction::EndTurn) {
            self.track_turn_exposure(state_before_action);
            self.track_hand_sizes(state_before_action);
        }

        self.track_action(state_before_action, actor, action);
    }

    fn on_game_end(&mut self, _game_id: Uuid, state: State, result: Option<GameOutcome>) {
        if !self.captured {
            // The game produced no decision point; nothing meaningful to harvest.
            self.current_game_id = None;
            return;
        }
        // The final state carries the last action's damage, including the lethal one.
        self.diff_board(&state);
        self.track_draw_pile(&state);
        self.flush_game(&state, result);
        self.current_game_id = None;
    }

    fn merge(&mut self, other: &dyn crate::simulation_event_handler::SimulationEventHandler) {
        let Some(other) = (other as &dyn std::any::Any).downcast_ref::<GameplayStatsCollector>()
        else {
            panic!("Attempted to merge GameplayStatsCollector with incompatible type");
        };

        self.num_games += other.num_games;
        for (id, deck) in &other.deck_dictionary {
            self.deck_dictionary
                .entry(*id)
                .or_insert_with(|| deck.clone());
        }
        for (key, stats) in &other.decks {
            self.decks.entry(key.clone()).or_default().merge_from(stats);
        }
    }
}
