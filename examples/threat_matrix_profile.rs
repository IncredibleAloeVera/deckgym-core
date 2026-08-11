//! Profile of the §1.2.5 threat matrix inside `get_observation`.
//!
//! The matrix is the observation's heaviest computation, and its cost used to be dominated by one
//! `State::clone` per attack inside the attacker projection. This measures the two paths side by
//! side: the shared [`ProjectionScratch`] the observation now uses (one copy per observation) and
//! the standalone [`estimate_attack_affordance`] entry point (one copy per attack), which is what
//! the observation did before.
//!
//! Run with: cargo run --release --example threat_matrix_profile

use std::time::Instant;

use deckgym::players::{create_players, PlayerCode};
use deckgym::rl::{estimate_attack_affordance, ProjectionScratch};
use deckgym::{Deck, Game};

fn main() {
    let deck_a = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
    let deck_b = Deck::from_file("example_decks/weezing-arbok.txt").unwrap();

    let mut n_points = 0usize;
    let mut t_obs = 0u128;
    let mut t_scratch = 0u128;
    let mut t_per_attack_clone = 0u128;
    let mut t_clone = 0u128;
    let mut n_affordance = 0usize;
    let mut n_projected = 0usize;

    for seed in 0..40u64 {
        let players = create_players(
            deck_a.clone(),
            deck_b.clone(),
            vec![PlayerCode::R, PlayerCode::R],
        );
        let mut game = Game::new(players, seed);
        game.enable_action_trace();

        while !game.is_game_over() {
            let state = game.get_state_clone();

            // 1. full observation
            let t = Instant::now();
            let _ = game.get_observation(state.current_player);
            t_obs += t.elapsed().as_nanos();

            // The attackers the threat matrix walks, resolved once so both paths see the same set.
            let mut targets = Vec::new();
            let mut projected_here = 0usize;
            for player in 0..2usize {
                for idx in 0..4usize {
                    let Some(pokemon) = state.in_play_pokemon[player][idx].as_ref() else {
                        continue;
                    };
                    let deckgym::models::Card::Pokemon(card) = &pokemon.card else {
                        continue;
                    };
                    for attack in &card.attacks {
                        targets.push((player, idx, attack.clone()));
                        n_affordance += 1;
                        if idx != 0 || state.current_player != player {
                            n_projected += 1;
                            projected_here += 1;
                        }
                    }
                }
            }

            // 2a. the shared-scratch path (what the observation does): one copy per observation
            let t = Instant::now();
            let mut scratch = ProjectionScratch::new(&state);
            for (player, idx, attack) in &targets {
                std::hint::black_box(scratch.attack_affordance((*player, *idx), attack));
            }
            t_scratch += t.elapsed().as_nanos();

            // 2b. the per-attack cloning path (the standalone entry point, and the old behaviour)
            let t = Instant::now();
            for (player, idx, attack) in &targets {
                std::hint::black_box(estimate_attack_affordance(&state, (*player, *idx), attack));
            }
            t_per_attack_clone += t.elapsed().as_nanos();

            // 3. cost of the State clones the per-attack path performs
            let t = Instant::now();
            for _ in 0..projected_here {
                let c = state.clone();
                std::hint::black_box(&c);
            }
            t_clone += t.elapsed().as_nanos();

            n_points += 1;
            game.play_tick();
        }
    }

    let per = |total: u128| total as f64 / n_points as f64 / 1000.0;
    println!("decision points        : {n_points}");
    println!("get_observation          : {:.1} µs/point", per(t_obs));
    println!("threat matrix, scratch   : {:.1} µs/point", per(t_scratch));
    println!(
        "threat matrix, per-attack: {:.1} µs/point",
        per(t_per_attack_clone)
    );
    println!("  State::clone in it     : {:.1} µs/point", per(t_clone));
    println!(
        "affordance calls         : {:.2}/point, of which projected: {:.2}",
        n_affordance as f64 / n_points as f64,
        n_projected as f64 / n_points as f64
    );
}
