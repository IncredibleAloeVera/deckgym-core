use deckgym::{
    actions::{Action, SimpleAction},
    card_ids::CardId,
    database::get_card_by_enum,
    models::{Card, PlayedCard, TrainerCard},
    test_support::get_initialized_game,
};

fn kiawe_trainer_card() -> TrainerCard {
    match get_card_by_enum(CardId::A3150Kiawe) {
        Card::Trainer(tc) => tc,
        _ => panic!("Expected trainer card"),
    }
}

/// Kiawe's "Your turn ends" must go through `end_turn_pending`, never onto the move generation
/// stack. Promotions are inserted at the *bottom* of that stack, so a stacked `EndTurn` outranks
/// a promotion queued after it and the turn ends with an empty Active Spot.
#[test]
fn test_kiawe_does_not_queue_end_turn_on_the_move_generation_stack() {
    let mut game = get_initialized_game(0);
    let mut state = game.get_state_clone();
    state.current_player = 0;
    state.turn_count = 3;

    state.set_board(
        vec![
            PlayedCard::from_id(CardId::A3037Turtonator),
            PlayedCard::from_id(CardId::A3027AlolanMarowak),
        ],
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
    );
    let kiawe = kiawe_trainer_card();
    state.hands[0] = vec![Card::Trainer(kiawe.clone())];
    game.set_state(state);

    game.apply_action(&Action {
        actor: 0,
        action: SimpleAction::Play {
            trainer_card: kiawe,
        },
        is_stack: false,
    });

    let state = game.get_state_clone();
    assert!(
        !state
            .move_generation_stack
            .iter()
            .any(|(_, choices)| choices.contains(&SimpleAction::EndTurn)),
        "Kiawe must not put an EndTurn on the stack, where it would outrank a later promotion"
    );

    // The energy choice is still what the player is asked for next.
    let (actor, choices) = state.generate_possible_actions();
    assert_eq!(actor, 0);
    assert!(
        choices
            .iter()
            .all(|c| matches!(c.action, SimpleAction::Attach { .. })),
        "the only pending decision should be where Kiawe's Energy goes, got {choices:?}"
    );
}

/// The flag still ends the turn once the Energy choice is resolved.
#[test]
fn test_kiawe_ends_the_turn_after_the_energy_choice_resolves() {
    let mut game = get_initialized_game(0);
    let mut state = game.get_state_clone();
    state.current_player = 0;
    state.turn_count = 3;

    state.set_board(
        vec![PlayedCard::from_id(CardId::A3037Turtonator)],
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
    );
    let kiawe = kiawe_trainer_card();
    state.hands[0] = vec![Card::Trainer(kiawe.clone())];
    game.set_state(state);

    game.apply_action(&Action {
        actor: 0,
        action: SimpleAction::Play {
            trainer_card: kiawe,
        },
        is_stack: false,
    });

    let (_, choices) = game.get_state_clone().generate_possible_actions();
    game.apply_action(&choices[0]);

    let (_, choices) = game.get_state_clone().generate_possible_actions();
    assert!(
        choices
            .iter()
            .all(|c| matches!(c.action, SimpleAction::EndTurn)),
        "after the Energy is attached, ending the turn should be the only option, got {choices:?}"
    );

    game.apply_action(&choices[0]);
    assert_eq!(
        game.get_state_clone().current_player,
        1,
        "Kiawe should have ended player 0's turn"
    );
}
