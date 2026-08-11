//! Replays a `runs/*/crashes/*.json` dump: rebuild the game from the dumped `State` and drive it
//! with random seats until it panics again. Ad-hoc triage tool for the crash families in that
//! directory, not part of the training pipeline.

use std::fs;

use deckgym::players::{create_players, PlayerCode};
use deckgym::{Deck, Game, State};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: replay_crash <dump.json> [ticks]");
    let ticks: usize = args.next().map_or(200, |a| a.parse().expect("ticks"));

    let dump: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
    let state: State = serde_json::from_value(dump["state"].clone()).expect("state");
    let seed = dump["seed"].as_u64().expect("seed");

    println!(
        "turn {} player {} stack {} | {}",
        state.turn_count,
        state.current_player,
        state.move_generation_stack.len(),
        dump["panic_message"].as_str().unwrap_or("")
    );

    // The dumped `decks` are the *remaining* draw piles, so they cannot stand in for the original
    // decklists; the state already carries the piles, and the players only need to pick actions.
    let players = create_players(
        Deck::default(),
        Deck::default(),
        vec![PlayerCode::R, PlayerCode::R],
    );
    let mut game = Game::from_state(state, players, seed);
    game.set_debug(false);

    for i in 0..ticks {
        if game.state().winner.is_some() {
            println!("finished cleanly after {i} ticks");
            return;
        }
        let action = game.play_tick();
        println!("{i:3}: {action:?}");
    }
    println!("still running after {ticks} ticks");
}
