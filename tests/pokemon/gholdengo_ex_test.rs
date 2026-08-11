//! Gholdengo ex's Spending Rush: one of the opponent's Pokémon is chosen at random per [M] Energy
//! attached, for 40 damage each.
//!
//! The energy count is unbounded, and the forecast used to enumerate one branch per *targeting
//! sequence* — `targets^energy`. Fourteen Energy against four Pokémon is 268 million of them, which
//! is what wedged `runs/long_v3` for 40 minutes inside a single frame.

use std::time::{Duration, Instant};

use deckgym::{
    actions::Action,
    card_ids::CardId,
    database::get_card_by_enum,
    models::{EnergyType, PlayedCard},
    test_support::{attack_action, get_test_game_with_board},
};

/// HP high enough that nothing is knocked out: the invariant under test is where the 40s land, and
/// a KO would take its Pokémon (and its damage counters) off the board before it could be read.
const TANK_HP: u32 = 2000;

fn tank(card_id: CardId) -> PlayedCard {
    PlayedCard::new(get_card_by_enum(card_id), 0, TANK_HP, vec![], false, vec![])
}

fn gholdengo_with_metal(count: usize) -> PlayedCard {
    PlayedCard::from_id(CardId::B2a078GholdengoEx).with_energy(vec![EnergyType::Metal; count])
}

/// Every hit lands somewhere on the opponent's board, whatever the seed sends it at.
#[test]
fn spending_rush_deals_forty_per_metal_energy_across_the_opponents_board() {
    let energies = 6;
    let mut game = get_test_game_with_board(
        vec![gholdengo_with_metal(energies)],
        vec![
            tank(CardId::A1001Bulbasaur),
            tank(CardId::A1033Charmander),
            tank(CardId::A1053Squirtle),
            tank(CardId::A1094Pikachu),
        ],
    );

    game.apply_action(&Action {
        actor: 0,
        action: attack_action(CardId::B2a078GholdengoEx, 0),
        is_stack: false,
    });
    game.play_until_stable();

    let state = game.get_state_clone();
    let dealt: Vec<u32> = state
        .enumerate_in_play_pokemon(1)
        .map(|(_, pokemon)| TANK_HP - pokemon.get_remaining_hp())
        .collect();

    assert_eq!(dealt.iter().sum::<u32>(), 40 * energies as u32);
    assert!(
        dealt.iter().all(|damage| damage % 40 == 0),
        "each hit is a whole 40: {dealt:?}"
    );
}

/// The regression. Fourteen Energy is reachable in a real game and was unreachable in practice: the
/// forecast alone took ~40 minutes, and the RL threat matrix pays it on *every* frame this board is
/// alive. A bound rather than an exact time, since the point is the difference between milliseconds
/// and tens of minutes.
#[test]
fn fourteen_metal_energy_resolves_without_enumerating_every_sequence() {
    let mut game = get_test_game_with_board(
        vec![gholdengo_with_metal(14)],
        vec![
            tank(CardId::A1001Bulbasaur),
            tank(CardId::A1033Charmander),
            tank(CardId::A1053Squirtle),
            tank(CardId::A1094Pikachu),
        ],
    );

    let started = Instant::now();
    game.apply_action(&Action {
        actor: 0,
        action: attack_action(CardId::B2a078GholdengoEx, 0),
        is_stack: false,
    });
    game.play_until_stable();
    let elapsed = started.elapsed();

    let state = game.get_state_clone();
    let dealt: u32 = state
        .enumerate_in_play_pokemon(1)
        .map(|(_, pokemon)| TANK_HP - pokemon.get_remaining_hp())
        .sum();
    assert_eq!(dealt, 40 * 14);
    assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
}
