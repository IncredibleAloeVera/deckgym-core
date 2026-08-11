//! Falsifiable tests for the deckbuilder label harvest (RL_ARCHITECTURE §1.5.7).
//!
//! Driven at the `Simulation` public API level (per the repo's testing guideline): run real games
//! with the collector registered, then assert the invariants that make the harvest trustworthy —
//! the `card_id@copies` identity, the presence of never-drawn rows, count/denominator ordering,
//! and damage attribution conservation.

use std::collections::HashMap;

use deckgym::{
    gameplay_stats_collector::GameplayStatsCollector, players::PlayerCode,
    test_support::load_test_decks, Deck, Simulation,
};

const NUM_GAMES: u32 = 24;

fn run_harvest() -> Simulation {
    let (deck_a, deck_b) = load_test_decks();
    let mut simulation = Simulation::new_with_decks(
        deck_a,
        deck_b,
        // Two *different* pilots, so the pilot columns of §1.5.7 have something to separate.
        vec![PlayerCode::R, PlayerCode::W],
        NUM_GAMES,
        // Unseeded on purpose: the simulation reuses a fixed seed for every game, so a seed here
        // would replay one identical game 24 times. These are invariants, so they must hold over
        // varied random rollouts.
        None,
        false,
        None,
    )
    .expect("simulation builds")
    .register_with_closure(|| {
        let mut collector = GameplayStatsCollector::new();
        collector.set_pilots(["r".to_string(), "w".to_string()]);
        Box::new(collector)
    });

    simulation.run();
    simulation
}

fn collector_of(simulation: &Simulation) -> &GameplayStatsCollector {
    simulation
        .get_event_handler::<GameplayStatsCollector>()
        .expect("collector was registered")
}

fn copy_counts(deck: &Deck) -> HashMap<String, u8> {
    let mut counts: HashMap<String, u8> = HashMap::new();
    for card in &deck.cards {
        *counts.entry(card.get_id()).or_insert(0) += 1;
    }
    counts
}

/// Finds the harvested entry whose decklist matches, comparing copy counts rather than card order:
/// the collector stores decks canonically, the source deck file does not.
fn stats_for<'a>(
    collector: &'a GameplayStatsCollector,
    deck: &Deck,
) -> &'a deckgym::gameplay_stats_collector::DeckStats {
    let expected = copy_counts(deck);
    collector
        .decks()
        .iter()
        .find(|(key, _)| {
            collector
                .deck_dictionary()
                .get(&key.deck)
                .is_some_and(|harvested| copy_counts(harvested) == expected)
        })
        .map(|(_, stats)| stats)
        .expect("deck present in the harvest")
}

/// Every card the deck runs gets a row every game — including the ones that were never drawn.
/// Dropping those rows would silently condition every downstream ratio on "was drawn", which is
/// exactly the bias the harvest exists to avoid.
#[test]
fn test_harvest_emits_a_row_for_every_deck_card_including_never_drawn() {
    let simulation = run_harvest();
    let collector = collector_of(&simulation);
    let (deck_a, deck_b) = load_test_decks();

    assert_eq!(collector.total_games(), NUM_GAMES);
    assert_eq!(
        collector.decks().len(),
        2,
        "both decks appear in the harvest"
    );

    for deck in [&deck_a, &deck_b] {
        let expected = copy_counts(deck);
        let stats = stats_for(collector, deck);

        assert_eq!(stats.games, NUM_GAMES);
        assert_eq!(
            stats.wins + stats.losses + stats.ties,
            stats.games,
            "outcomes partition the games"
        );
        assert_eq!(
            stats.cards.len(),
            expected.len(),
            "one row per distinct card, drawn or not"
        );

        for (key, card) in &stats.cards {
            assert_eq!(
                card.games, NUM_GAMES,
                "{} is counted every game",
                key.card_id
            );
            assert!(
                card.games_never_drawn > 0 || card.copies_drawn > 0,
                "{} is either drawn or accounted as never drawn",
                key.card_id
            );
        }
    }
}

/// `card_id@1` and `card_id@2` are different entities. Within one deck a card resolves to exactly
/// one copy count, and that count is the deck's actual multiplicity.
#[test]
fn test_card_identity_carries_the_deck_copy_count() {
    let simulation = run_harvest();
    let collector = collector_of(&simulation);
    let (deck_a, deck_b) = load_test_decks();

    for deck in [&deck_a, &deck_b] {
        let expected = copy_counts(deck);
        let stats = stats_for(collector, deck);

        let mut seen: HashMap<&str, u8> = HashMap::new();
        for key in stats.cards.keys() {
            assert_eq!(
                Some(&key.copies_in_deck),
                expected.get(&key.card_id),
                "{} keyed with the deck's real multiplicity",
                key.card_id
            );
            assert!(
                seen.insert(key.card_id.as_str(), key.copies_in_deck)
                    .is_none(),
                "{} appears under a single copy count within one deck",
                key.card_id
            );
        }

        assert!(
            stats.cards.keys().any(|key| key.copies_in_deck == 2),
            "the test decks run at least one card in 2 copies"
        );
    }
}

/// Labels are only worth what the pilot is, so every row says who played the deck and who they
/// faced. Without the pair in the key, a `w` seat's plays and a trained policy's would merge into
/// one unrecoverable average.
#[test]
fn test_harvest_attributes_every_row_to_the_pilot_that_produced_it() {
    let simulation = run_harvest();
    let collector = collector_of(&simulation);
    let (deck_a, _) = load_test_decks();

    let mut pairs: Vec<_> = collector
        .decks()
        .keys()
        .map(|key| (key.pilot.clone(), key.opponent_pilot.clone()))
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("r".to_string(), "w".to_string()),
            ("w".to_string(), "r".to_string()),
        ],
        "each seat is labelled from its own side"
    );

    // Seat order is the order the codes were given in, so deck A's rows are the `r` ones.
    let expected = copy_counts(&deck_a);
    let key = collector
        .decks()
        .keys()
        .find(|key| {
            collector
                .deck_dictionary()
                .get(&key.deck)
                .is_some_and(|harvested| copy_counts(harvested) == expected)
        })
        .expect("deck A present in the harvest");
    assert_eq!(key.pilot, "r");
    assert_eq!(key.opponent_pilot, "w");
}

/// `opponent_deck` is what lets a matchup be resolved after the fact — without it, every opposing
/// decklist a pilot ever faced would collapse into the same row.
#[test]
fn test_harvest_cross_references_the_opposing_decklist() {
    let simulation = run_harvest();
    let collector = collector_of(&simulation);
    let (deck_a, deck_b) = load_test_decks();

    let a_key = collector
        .decks()
        .keys()
        .find(|key| {
            collector
                .deck_dictionary()
                .get(&key.deck)
                .is_some_and(|harvested| copy_counts(harvested) == copy_counts(&deck_a))
        })
        .expect("deck A present in the harvest");
    let b_key = collector
        .decks()
        .keys()
        .find(|key| {
            collector
                .deck_dictionary()
                .get(&key.deck)
                .is_some_and(|harvested| copy_counts(harvested) == copy_counts(&deck_b))
        })
        .expect("deck B present in the harvest");

    assert_eq!(a_key.opponent_deck, b_key.deck, "A's opponent_deck is B");
    assert_eq!(b_key.opponent_deck, a_key.deck, "B's opponent_deck is A");
}

/// Counts never outrun the denominator they are meant to be divided by.
#[test]
fn test_counts_respect_their_denominators() {
    let simulation = run_harvest();
    let collector = collector_of(&simulation);

    for stats in collector.decks().values() {
        for (key, card) in &stats.cards {
            let max_copies = key.copies_in_deck as u32 * card.games;
            assert!(
                card.copies_drawn <= max_copies,
                "{}: drawn {} > {} available",
                key.card_id,
                card.copies_drawn,
                max_copies
            );
            // `times_played` counts plays, not copies, so it is *not* bounded by `copies_drawn`:
            // Koga returns a Weezing to hand to be evolved again. It is still bounded by having
            // been drawn at all.
            assert!(
                card.times_played == 0 || card.copies_drawn > 0,
                "{}: played without ever being drawn",
                key.card_id
            );
            assert!(
                card.ended_in_hand <= card.copies_drawn,
                "{}: a card can only end in hand if it was drawn",
                key.card_id
            );
            assert!(
                card.games_never_drawn + card.games_drawn_never_played <= card.games,
                "{}: the two dead-card cases are disjoint subsets of the games",
                key.card_id
            );
            assert!(
                card.first_play_games <= card.games,
                "{}: first play happens at most once per game",
                key.card_id
            );
            assert!(
                card.times_koed <= max_copies,
                "{}: cannot be knocked out more often than it was in play",
                key.card_id
            );
        }

        // Points are 1 per knock-out and 2 for an ex, so they bound the KOs from above.
        let kos: u32 = stats.cards.values().map(|card| card.kos_dealt).sum();
        assert!(
            kos <= stats.points_scored,
            "{kos} knock-outs vs {} points scored",
            stats.points_scored
        );
    }
}

/// Per-card damage attribution sums exactly to the deck total. The total is kept separately so the
/// absolute is never lost to a share — shares are compositional and would fabricate anti-synergy
/// between two attackers in the same deck.
#[test]
fn test_damage_attribution_is_conservative() {
    let simulation = run_harvest();
    let collector = collector_of(&simulation);

    let mut any_damage = false;
    for stats in collector.decks().values() {
        let attributed: u32 = stats.cards.values().map(|card| card.damage_dealt).sum();
        assert_eq!(
            attributed, stats.damage_dealt_total,
            "every point of damage is credited to exactly one card"
        );
        any_damage |= stats.damage_dealt_total > 0;
    }
    assert!(any_damage, "the games produced some damage to attribute");
}
