//! The *player mode* belief overlay.
//!
//! The engine is fully observable (*spectator mode* — the identity, sees everything). The belief
//! layer is a per-player **oracle helper** intercalated between the engine state and any future
//! `get_observation`: it maintains, for each observer, what that observer knows about the
//! opponent's hidden cards. It is **not** an observation vector — it is the state a renderer would
//! later mask the spectator view through.
//!
//! Two knowledge markers per card, directional ("revealed to the non-owner"):
//! - **presence** — *monotone*: how many copies of a card are known to exist in the opponent's
//!   hidden zones. Once seen it never un-sees (`NOTES.md`: "flip 1 → jamais reflip 0").
//! - **position** — *volatile*: which hidden zone (`Hand` / `Deck`) the card is currently known to
//!   be in. Maintained by analysing pure movements between zones (see [`reveal::RevealEvent`]).
//!   A deck shuffle randomises order only, never zone membership, so it does not touch position.

mod reveal;

pub use reveal::{card_category, reveal_taxonomy, CardCategory, RevealEvent, RevealPattern, Zone};

use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::models::EnergyType;
use std::collections::{HashMap, HashSet};

/// How many copies of a card are currently known to sit in each hidden zone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZoneCounts {
    pub hand: u32,
    pub deck: u32,
}

impl ZoneCounts {
    fn is_empty(&self) -> bool {
        self.hand == 0 && self.deck == 0
    }
}

/// One observer's knowledge about the *opponent's* hidden cards.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlayerBelief {
    /// Monotone counter: max copies of each card ever seen to exist in the opponent's hidden zones.
    presence: HashMap<CardId, u32>,
    /// Volatile per-zone position overlay (which hidden zone each known card sits in).
    position: HashMap<CardId, ZoneCounts>,
    /// Monotone: energy types seen so far in the opponent's energy zone (`current`/`next`).
    energy_seen: HashSet<EnergyType>,
}

impl PlayerBelief {
    /// Presence counter, keyed by exact printed card. Monotone over the whole game.
    pub fn presence(&self) -> &HashMap<CardId, u32> {
        &self.presence
    }

    /// Energy types seen so far in the opponent's energy zone. Monotone over the whole game.
    pub fn energy_seen(&self) -> &HashSet<EnergyType> {
        &self.energy_seen
    }

    /// Leak-free render of the opponent's hand: cards with a live `Hand` position marker.
    pub fn known_hand(&self) -> HashMap<CardId, u32> {
        self.position
            .iter()
            .filter(|(_, z)| z.hand > 0)
            .map(|(&c, z)| (c, z.hand))
            .collect()
    }

    /// Cards currently known to sit in the opponent's deck (e.g. an in-play Pokémon shuffled back).
    pub fn known_deck(&self) -> HashMap<CardId, u32> {
        self.position
            .iter()
            .filter(|(_, z)| z.deck > 0)
            .map(|(&c, z)| (c, z.deck))
            .collect()
    }

    /// Copies known to exist in a hidden zone without a live position marker saying which — what
    /// the observer remembers seeing and can no longer locate.
    ///
    /// `public` is the observer's count of the opponent's *currently visible* copies (board,
    /// discard), which has to be subtracted: `presence` is monotone and stays put when a card is
    /// played, so `presence − position` alone would claim a card is hidden somewhere while it sits
    /// face-up in the discard. Subtracting a public copy that presence never counted only hides a
    /// residual that was real, and the error is deliberately taken in that direction: a card the
    /// observer is not told about costs a signal, one they are told about wrongly costs the truth.
    pub fn hidden_elsewhere(&self, public: &HashMap<CardId, u32>) -> HashMap<CardId, u32> {
        self.presence
            .iter()
            .filter_map(|(&card, &seen)| {
                let located = self
                    .position
                    .get(&card)
                    .map_or(0, |zone| zone.hand + zone.deck);
                let accounted = located + public.get(&card).copied().unwrap_or(0);
                seen.checked_sub(accounted)
                    .filter(|residual| *residual > 0)
                    .map(|residual| (card, residual))
            })
            .collect()
    }

    /// Raise presence so at least `count` copies of `card` are recorded (never lowers it).
    fn bump_presence(&mut self, card: CardId, count: u32) {
        let entry = self.presence.entry(card).or_insert(0);
        *entry = (*entry).max(count);
    }

    fn add_position(&mut self, card: CardId, zone: Zone, count: u32) {
        let entry = self.position.entry(card).or_default();
        match zone {
            Zone::Hand => entry.hand += count,
            Zone::Deck => entry.deck += count,
            Zone::Public => {}
        }
        let total = entry.hand + entry.deck;
        self.bump_presence(card, total);
    }

    fn set_hand_position_at_least(&mut self, card: CardId, count: u32) {
        let entry = self.position.entry(card).or_default();
        entry.hand = entry.hand.max(count);
        if entry.is_empty() {
            self.position.remove(&card);
        }
    }

    fn remove_position(&mut self, card: CardId, zone: Zone) {
        if let Some(entry) = self.position.get_mut(&card) {
            match zone {
                Zone::Hand => entry.hand = entry.hand.saturating_sub(1),
                Zone::Deck => entry.deck = entry.deck.saturating_sub(1),
                Zone::Public => {}
            }
            if entry.is_empty() {
                self.position.remove(&card);
            }
        }
    }

    /// Reset (zero) the given zone for every tracked card whose category matches.
    fn reset_category(&mut self, zone: Zone, category: CardCategory) {
        let mut to_remove = Vec::new();
        for (&card, entry) in self.position.iter_mut() {
            if card_category(&get_card_by_enum(card)) != category {
                continue;
            }
            match zone {
                Zone::Hand => entry.hand = 0,
                Zone::Deck => entry.deck = 0,
                Zone::Public => {}
            }
            if entry.is_empty() {
                to_remove.push(card);
            }
        }
        for card in to_remove {
            self.position.remove(&card);
        }
    }

    /// Clear a whole zone's position overlay.
    fn clear_zone(&mut self, zone: Zone) {
        for entry in self.position.values_mut() {
            match zone {
                Zone::Hand => entry.hand = 0,
                Zone::Deck => entry.deck = 0,
                Zone::Public => {}
            }
        }
        self.position.retain(|_, z| !z.is_empty());
    }
}

/// Per-player belief. `beliefs[observer]` is what `observer` knows about `1 - observer`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BeliefTracker {
    beliefs: [PlayerBelief; 2],
}

impl BeliefTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// What `observer` knows about their opponent.
    pub fn belief(&self, observer: usize) -> &PlayerBelief {
        &self.beliefs[observer]
    }

    /// The opponent's hand as known to `observer` (leak-free position overlay).
    pub fn known_opponent_hand(&self, observer: usize) -> HashMap<CardId, u32> {
        self.beliefs[observer].known_hand()
    }

    /// Cards known to `observer` to be in the opponent's deck.
    pub fn known_opponent_deck(&self, observer: usize) -> HashMap<CardId, u32> {
        self.beliefs[observer].known_deck()
    }

    /// The presence counters `observer` has accumulated about their opponent.
    pub fn presence(&self, observer: usize) -> &HashMap<CardId, u32> {
        self.beliefs[observer].presence()
    }

    /// Energy types `observer` has seen appear in the opponent's energy zone. Monotone over the
    /// whole game — the honest substitute for the opponent's full declared energy set.
    pub fn seen_opponent_energy(&self, observer: usize) -> &HashSet<EnergyType> {
        self.beliefs[observer].energy_seen()
    }

    /// See [`PlayerBelief::hidden_elsewhere`].
    pub fn opponent_hidden_elsewhere(
        &self,
        observer: usize,
        public: &HashMap<CardId, u32>,
    ) -> HashMap<CardId, u32> {
        self.beliefs[observer].hidden_elsewhere(public)
    }

    /// Apply a batch of reveal events. Each event about `owner` informs the non-owner
    /// (`1 - owner`) only — revealing to the owner themselves is a no-op.
    pub fn observe(&mut self, events: &[RevealEvent]) {
        for event in events {
            let observer = 1 - event.owner();
            let belief = &mut self.beliefs[observer];
            match event {
                RevealEvent::HandRevealed { cards, .. } => {
                    let mut counts: HashMap<CardId, u32> = HashMap::new();
                    for &card in cards {
                        *counts.entry(card).or_insert(0) += 1;
                    }
                    for (&card, &count) in &counts {
                        belief.bump_presence(card, count);
                        belief.set_hand_position_at_least(card, count);
                    }
                }
                RevealEvent::KnownCardMoved { card, from, to, .. } => {
                    belief.remove_position(*card, *from);
                    belief.add_position(*card, *to, 1);
                }
                RevealEvent::TypedZoneReset { zone, category, .. } => {
                    belief.reset_category(*zone, *category);
                }
                RevealEvent::ZoneCleared { zone, .. } => {
                    belief.clear_zone(*zone);
                }
                RevealEvent::EnergyRevealed { energy, .. } => {
                    belief.energy_seen.insert(*energy);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_ids::CardId;

    // A1001Bulbasaur = Basic; A1002Ivysaur = Stage 1; A1003Venusaur = Stage 2.
    const BASIC: CardId = CardId::A1001Bulbasaur;
    const STAGE1: CardId = CardId::A1002Ivysaur;

    fn count(map: &HashMap<CardId, u32>, card: CardId) -> u32 {
        map.get(&card).copied().unwrap_or(0)
    }

    #[test]
    fn hand_revealed_sets_presence_and_hand_position() {
        let mut t = BeliefTracker::new();
        t.observe(&[RevealEvent::HandRevealed {
            owner: 1,
            cards: vec![BASIC, BASIC, STAGE1],
        }]);
        assert_eq!(count(&t.known_opponent_hand(0), BASIC), 2);
        assert_eq!(count(&t.known_opponent_hand(0), STAGE1), 1);
        assert_eq!(count(t.presence(0), BASIC), 2);
        assert!(t.known_opponent_hand(1).is_empty(), "directional");
    }

    #[test]
    fn played_card_leaves_hand_and_does_not_leak() {
        let mut t = BeliefTracker::new();
        t.observe(&[RevealEvent::HandRevealed {
            owner: 1,
            cards: vec![BASIC, STAGE1],
        }]);
        // Opponent plays the Basic (visible → public).
        t.observe(&[RevealEvent::KnownCardMoved {
            owner: 1,
            card: BASIC,
            from: Zone::Hand,
            to: Zone::Public,
        }]);
        assert_eq!(count(&t.known_opponent_hand(0), BASIC), 0, "no leak");
        assert_eq!(count(&t.known_opponent_hand(0), STAGE1), 1);
        assert_eq!(count(t.presence(0), BASIC), 1, "presence monotone");
    }

    #[test]
    fn in_play_card_shuffled_into_deck_is_known_in_deck() {
        // Aerodactyl: opponent's Active (public) shuffled into their deck.
        let mut t = BeliefTracker::new();
        t.observe(&[RevealEvent::KnownCardMoved {
            owner: 1,
            card: BASIC,
            from: Zone::Public,
            to: Zone::Deck,
        }]);
        assert_eq!(count(&t.known_opponent_deck(0), BASIC), 1, "known in deck");
        assert_eq!(count(t.presence(0), BASIC), 1);
        assert!(t.known_opponent_hand(0).is_empty());
    }

    #[test]
    fn deck_shuffle_membership_survives_but_draw_clears_it() {
        let mut t = BeliefTracker::new();
        t.observe(&[RevealEvent::KnownCardMoved {
            owner: 1,
            card: BASIC,
            from: Zone::Public,
            to: Zone::Deck,
        }]);
        // A fresh draw (unknown card) destroys deck-position certainty.
        t.observe(&[RevealEvent::ZoneCleared {
            owner: 1,
            zone: Zone::Deck,
        }]);
        assert!(t.known_opponent_deck(0).is_empty());
        assert_eq!(count(t.presence(0), BASIC), 1, "presence survives");
    }

    #[test]
    fn typed_reset_only_clears_the_moved_category() {
        let mut t = BeliefTracker::new();
        // Two cards known in the deck: a Basic and a Stage 1.
        t.observe(&[
            RevealEvent::KnownCardMoved {
                owner: 1,
                card: BASIC,
                from: Zone::Public,
                to: Zone::Deck,
            },
            RevealEvent::KnownCardMoved {
                owner: 1,
                card: STAGE1,
                from: Zone::Public,
                to: Zone::Deck,
            },
        ]);
        // A random Basic is drawn from the deck (Poké Ball): only Basic deck-positions reset.
        t.observe(&[RevealEvent::TypedZoneReset {
            owner: 1,
            zone: Zone::Deck,
            category: CardCategory::BasicPokemon,
        }]);
        assert_eq!(count(&t.known_opponent_deck(0), BASIC), 0, "basic reset");
        assert_eq!(count(&t.known_opponent_deck(0), STAGE1), 1, "stage1 kept");
    }

    #[test]
    fn zone_cleared_wipes_hand_positions_but_keeps_presence() {
        let mut t = BeliefTracker::new();
        t.observe(&[RevealEvent::HandRevealed {
            owner: 1,
            cards: vec![BASIC, STAGE1],
        }]);
        t.observe(&[RevealEvent::ZoneCleared {
            owner: 1,
            zone: Zone::Hand,
        }]);
        assert!(t.known_opponent_hand(0).is_empty());
        assert_eq!(count(t.presence(0), BASIC), 1);
        assert_eq!(count(t.presence(0), STAGE1), 1);
    }

    #[test]
    fn energy_revealed_is_monotone_and_directional() {
        use crate::models::EnergyType;

        let mut t = BeliefTracker::new();
        assert!(t.seen_opponent_energy(0).is_empty());

        t.observe(&[RevealEvent::EnergyRevealed {
            owner: 1,
            energy: EnergyType::Fire,
        }]);
        assert!(t.seen_opponent_energy(0).contains(&EnergyType::Fire));
        assert!(
            t.seen_opponent_energy(1).is_empty(),
            "directional: does not inform the owner about themselves"
        );

        // A later roll of a type already seen does not shrink the set (nor duplicate it).
        t.observe(&[RevealEvent::EnergyRevealed {
            owner: 1,
            energy: EnergyType::Fire,
        }]);
        assert_eq!(t.seen_opponent_energy(0).len(), 1);

        t.observe(&[RevealEvent::EnergyRevealed {
            owner: 1,
            energy: EnergyType::Water,
        }]);
        assert_eq!(t.seen_opponent_energy(0).len(), 2);
        assert!(t.seen_opponent_energy(0).contains(&EnergyType::Water));
    }
}
