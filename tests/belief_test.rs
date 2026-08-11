//! Falsifiable tests for the *player mode* belief overlay.
//!
//! Driven at the `Game` public API level (per the repo's testing guideline): enable the belief in
//! player mode, drive reveal-bearing actions, and assert the presence/position dichotomy holds —
//! notably the leak-free render (hand and deck) and monotone presence.

use std::collections::HashMap;

use deckgym::{
    actions::{Action, SimpleAction},
    card_ids::CardId,
    database::get_card_by_enum,
    models::{Card, PlayedCard},
    test_support::{get_test_game_with_board, init_random_players},
    Game,
};

fn multiset(cards: &[Card]) -> HashMap<CardId, u32> {
    let mut m = HashMap::new();
    for c in cards {
        *m.entry(c.get_card_id()).or_insert(0) += 1;
    }
    m
}

fn place(actor: usize, card: CardId, index: usize) -> Action {
    Action {
        actor,
        action: SimpleAction::Place(get_card_by_enum(card), index),
        is_stack: false,
    }
}

/// Misdreavus's Infiltrating Inspection reveals the opponent's whole hand → presence + Hand position.
#[test]
fn misdreavus_reveals_opponent_hand() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1005Caterpie)],
    );
    game.enable_belief();

    let mut state = game.get_state_clone();
    state.hands[0] = vec![get_card_by_enum(CardId::A4a032Misdreavus)];
    state.hands[1] = vec![
        get_card_by_enum(CardId::A1001Bulbasaur),
        get_card_by_enum(CardId::A1001Bulbasaur),
        get_card_by_enum(CardId::A1005Caterpie),
    ];
    game.set_state(state);

    game.apply_action(&place(0, CardId::A4a032Misdreavus, 1));

    let belief = game.belief().expect("belief enabled");
    let known = belief.known_opponent_hand(0);
    assert_eq!(known.get(&CardId::A1001Bulbasaur), Some(&2));
    assert_eq!(known.get(&CardId::A1005Caterpie), Some(&1));
    assert_eq!(belief.presence(0).get(&CardId::A1001Bulbasaur), Some(&2));
}

/// After a reveal, a revealed card the opponent then plays leaves the rendered hand (no leak),
/// while presence stays monotone. Exercises the central Hand → Public move.
#[test]
fn played_revealed_card_leaves_known_hand_but_presence_survives() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1005Caterpie)],
    );
    game.enable_belief();

    let mut state = game.get_state_clone();
    state.hands[0] = vec![get_card_by_enum(CardId::A4a032Misdreavus)];
    state.hands[1] = vec![
        get_card_by_enum(CardId::A1001Bulbasaur),
        get_card_by_enum(CardId::A1005Caterpie),
    ];
    game.set_state(state);

    game.apply_action(&place(0, CardId::A4a032Misdreavus, 1));
    assert_eq!(
        game.belief()
            .unwrap()
            .known_opponent_hand(0)
            .get(&CardId::A1001Bulbasaur),
        Some(&1)
    );

    // Player 1 benches Bulbasaur from their (revealed) hand — the opponent sees it become public.
    game.apply_action(&place(1, CardId::A1001Bulbasaur, 1));

    let belief = game.belief().unwrap();
    assert_eq!(
        belief.known_opponent_hand(0).get(&CardId::A1001Bulbasaur),
        None,
        "played card must not leak into the rendered hand"
    );
    assert_eq!(
        belief.known_opponent_hand(0).get(&CardId::A1005Caterpie),
        Some(&1),
        "the still-held card remains known"
    );
    assert_eq!(
        belief.presence(0).get(&CardId::A1001Bulbasaur),
        Some(&1),
        "presence is monotone — it survives the card going public"
    );
}

/// Spectator mode (the default) never allocates a belief — the engine identity is bypassed.
#[test]
fn spectator_mode_has_no_belief() {
    let game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1005Caterpie)],
    );
    assert!(game.belief().is_none());
}

/// Over random games: the render never leaks — known hand ⊆ actual opponent hand and known deck ⊆
/// actual opponent deck — and presence is monotone at every step.
#[test]
fn no_leak_and_presence_monotone_over_random_games() {
    for seed in 0..8u64 {
        let mut game = Game::new(init_random_players(), seed);
        game.enable_belief();
        let mut prev_presence: [HashMap<CardId, u32>; 2] = [HashMap::new(), HashMap::new()];

        while !game.is_game_over() {
            game.play_tick();
            let state = game.get_state_clone();
            let belief = game.belief().unwrap();

            for observer in [0usize, 1] {
                let opp = 1 - observer;
                let actual_hand = multiset(&state.hands[opp]);
                for (card, &known) in &belief.known_opponent_hand(observer) {
                    let have = actual_hand.get(card).copied().unwrap_or(0);
                    assert!(
                        known <= have,
                        "seed {seed}: hand leak for {card:?} — rendered {known} > actual {have}"
                    );
                }
                let actual_deck = multiset(&state.decks[opp].cards);
                for (card, &known) in &belief.known_opponent_deck(observer) {
                    let have = actual_deck.get(card).copied().unwrap_or(0);
                    assert!(
                        known <= have,
                        "seed {seed}: deck leak for {card:?} — rendered {known} > actual {have}"
                    );
                }
                for (card, &p) in belief.presence(observer) {
                    let before = prev_presence[observer].get(card).copied().unwrap_or(0);
                    assert!(
                        p >= before,
                        "seed {seed}: presence for {card:?} decreased {before} -> {p}"
                    );
                }
                prev_presence[observer] = belief.presence(observer).clone();
            }
        }
    }
}
