//! Typed reveal / movement events emitted from effect resolution.
//!
//! Today card effects mutate the state silently. To support the *mode joueur* belief overlay (see
//! `NOTES.md`), effects now push a [`RevealEvent`] onto the transient reveal log of
//! [`crate::State`], and the belief maintainer ([`crate::belief::BeliefTracker`]) drains and
//! interprets them.
//!
//! The model tracks two knowledge markers about the owner's hidden cards, directional ("revealed
//! to the non-owner"): **presence** (monotone existence in a hidden zone) and **position** (which
//! hidden zone — `Hand` or `Deck` — the card is currently known to be in). Position is *volatile*
//! and is maintained by analysing pure movements between zones:
//!
//! - a card whose identity the observer **sees** move ([`RevealEvent::KnownCardMoved`]) mutates its
//!   position to the destination zone (or drops it when the destination is public);
//! - a card of some category leaving a zone with its identity **unknown** to the observer
//!   ([`RevealEvent::TypedZoneReset`]) — a random deck search, a secret selective hand→deck shuffle
//!   — resets that category's positions in the source zone;
//! - a bulk shuffle-and-redraw ([`RevealEvent::ZoneCleared`]) destroys a whole zone's localization.
//!
//! Shuffling a deck only randomises order, never zone membership, so it carries **no** belief event
//! on its own (a card known to be in the deck stays known to be in the deck).

use crate::card_ids::CardId;
use crate::models::{Card, EnergyType, TrainerType};

/// A zone a card can move to/from. Only `Hand` and `Deck` carry hidden position; `Public`
/// (board / discard pile) is visible and not position-tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Hand,
    Deck,
    Public,
}

/// Card category at **stage + subtype** granularity — the grain at which a typed search / shuffle
/// invalidates position knowledge. Poké Ball moves a `BasicPokemon`; a Supporter search moves a
/// `Supporter`; so resetting must be scoped to the moved category, not to "any card".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardCategory {
    BasicPokemon,
    Stage1Pokemon,
    Stage2Pokemon,
    Supporter,
    Item,
    Tool,
    Fossil,
    Stadium,
}

/// The stage-and-subtype category of a card.
pub fn card_category(card: &Card) -> CardCategory {
    match card {
        Card::Pokemon(p) => match p.stage {
            0 => CardCategory::BasicPokemon,
            1 => CardCategory::Stage1Pokemon,
            _ => CardCategory::Stage2Pokemon,
        },
        Card::Trainer(t) => match t.trainer_card_type {
            TrainerType::Supporter => CardCategory::Supporter,
            TrainerType::Item => CardCategory::Item,
            TrainerType::Tool => CardCategory::Tool,
            TrainerType::Fossil => CardCategory::Fossil,
            TrainerType::Stadium => CardCategory::Stadium,
        },
    }
}

/// A single reveal / movement event, emitted from effect resolution. `owner` is the player whose
/// card / zone is affected; every event informs the **non-owner** (`1 - owner`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevealEvent {
    /// The owner's whole hand was seen: presence **and** `Hand` position over `cards`.
    HandRevealed { owner: usize, cards: Vec<CardId> },
    /// A card whose identity the observer knows moved `from` → `to` (played / discarded, revealed-
    /// then-shuffled, or an in-play card shuffled into the deck). Mutates its position: dropped from
    /// `from`, added to `to` (presence bumped when `to` is a hidden zone).
    KnownCardMoved {
        owner: usize,
        card: CardId,
        from: Zone,
        to: Zone,
    },
    /// A card of `category` left `zone` with its identity **unknown** to the observer (random deck
    /// search, secret selective hand→deck shuffle) → reset that category's positions in `zone`.
    /// `zone` is `Hand` or `Deck` (`Public` is a no-op).
    TypedZoneReset {
        owner: usize,
        zone: Zone,
        category: CardCategory,
    },
    /// A whole zone's localization is destroyed (bulk hand shuffle + redraw: Iono / Mars / Red Card;
    /// or a fresh draw invalidating deck-position certainty). Clears all positions in `zone`.
    ZoneCleared { owner: usize, zone: Zone },
    /// A new energy type was rolled into the owner's energy zone (`current`/`next`), which is
    /// public the instant it appears. Bumps the non-owner's monotone memory of which of the
    /// owner's declared energy types have actually been seen — the honest substitute for reading
    /// the deck's full declared set off `Deck::energy_types` (TODO.md, "Opponent deck energy").
    EnergyRevealed { owner: usize, energy: EnergyType },
}

impl RevealEvent {
    /// The player whose card/zone this event is about.
    pub fn owner(&self) -> usize {
        match self {
            RevealEvent::HandRevealed { owner, .. }
            | RevealEvent::KnownCardMoved { owner, .. }
            | RevealEvent::TypedZoneReset { owner, .. }
            | RevealEvent::ZoneCleared { owner, .. }
            | RevealEvent::EnergyRevealed { owner, .. } => *owner,
        }
    }
}

/// The knowledge a hand-reveal informs the non-owner about, per `NOTES.md` §"Taxonomie des reveals".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevealPattern {
    /// Presence + position over the revealed cards (e.g. "reveals their hand").
    PresenceAndPosition,
    /// Presence only — position destroyed by an accompanying shuffle (reveal-then-shuffle).
    PresenceOnly,
}

/// Offline classification of the reveal-bearing effect texts in the frozen pool.
///
/// This is **documentation / a coverage guard**, not the runtime path: emission is direct-typed at
/// the resolution site (which knows the concrete cards). The map lets a test assert that every
/// known opponent-hand-reveal text is accounted for, so a newly added reveal card can't silently
/// drift from this classification.
pub fn reveal_taxonomy() -> Vec<(&'static str, RevealPattern)> {
    vec![
        // reveals their hand → presence + position on the whole hand
        ("Your opponent reveals their hand.", RevealPattern::PresenceAndPosition),
        (
            "Your opponent reveals their hand. Choose a Supporter card you find there and discard it.",
            RevealPattern::PresenceAndPosition,
        ),
        (
            "Flip a coin. If heads, your opponent reveals their hand. Choose a Supporter card you find there and discard it.",
            RevealPattern::PresenceAndPosition,
        ),
        (
            "Your opponent reveals their hand. Choose a card you find there and shuffle it into your opponent's deck.",
            RevealPattern::PresenceAndPosition,
        ),
        // Misdreavus: "when you put this Pokémon from your hand onto your Bench, you may have your
        // opponent reveal their hand."
        (
            "Once during your turn, when you put this Pokémon from your hand onto your Bench, you may have your opponent reveal their hand.",
            RevealPattern::PresenceAndPosition,
        ),
        // reveal-then-shuffle → presence only (location kept only at zone grain: now in the deck)
        (
            "Flip a coin. If heads, your opponent reveals a random card from their hand and shuffles it into their deck.",
            RevealPattern::PresenceOnly,
        ),
        (
            "Flip 3 coins. For each heads, a card is chosen at random from your opponent's hand. Your opponent reveals that card and shuffles it into their deck.",
            RevealPattern::PresenceOnly,
        ),
        (
            "Your opponent reveals a random card from their hand and shuffles it into their deck.",
            RevealPattern::PresenceOnly,
        ),
        (
            "Your opponent reveals a random card from their hand and shuffles it into their deck. Shuffle this Pokémon into your deck.",
            RevealPattern::PresenceOnly,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::get_card_by_enum;
    use std::collections::HashSet;
    use strum::IntoEnumIterator;

    /// Guard against drift: every opponent-hand-reveal text in the frozen pool must be classified
    /// in [`reveal_taxonomy`], so a newly added reveal card forces a conscious classification.
    #[test]
    fn taxonomy_covers_all_opponent_reveal_texts() {
        let classified: HashSet<&str> = reveal_taxonomy().into_iter().map(|(t, _)| t).collect();
        let mut uncovered: Vec<String> = Vec::new();
        for id in CardId::iter() {
            let card = get_card_by_enum(id);
            let mut texts: Vec<String> = card
                .get_attacks()
                .into_iter()
                .filter_map(|a| a.effect)
                .collect();
            if let Some(ability) = card.get_ability() {
                texts.push(ability.effect);
            }
            for text in texts {
                // "opponent reveal(s)" covers both the attack wording and Misdreavus's ability.
                if text.contains("opponent reveal") && !classified.contains(text.as_str()) {
                    uncovered.push(text);
                }
            }
        }
        uncovered.sort();
        uncovered.dedup();
        assert!(
            uncovered.is_empty(),
            "Unclassified opponent-reveal texts (add to reveal_taxonomy): {uncovered:#?}"
        );
    }

    /// A presence-only classification must correspond to a reveal-then-shuffle text.
    #[test]
    fn presence_only_implies_shuffle() {
        for (text, pattern) in reveal_taxonomy() {
            if pattern == RevealPattern::PresenceOnly {
                assert!(
                    text.contains("shuffle"),
                    "presence-only pattern should be a reveal-then-shuffle: {text}"
                );
            }
        }
    }
}
