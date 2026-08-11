//! The §1.5.3 deck DBs against the engine, at the `Game` level.
//!
//! The unit tests in `src/rl/train/` check that the compiled DBs *parse*. That is the weaker
//! claim: `playable` is a property the JSON archive asserted, computed from `card_status` by a
//! script outside this repo. The claim that matters is that a sampled pair actually plays — that
//! nothing in the 70k meta decks hits an unimplemented effect, an illegal composition, or a
//! setup with no basic Pokémon to promote.
//!
//! Tutorial is swept in full. Meta is sampled: a full sweep is 70k games, and the failure mode
//! here is a whole card being unimplemented, which a sample across archetypes finds.

use std::path::Path;

use deckgym::players::{create_players, PlayerCode};
use deckgym::rl::env::env_rng;
use deckgym::rl::train::{DeckDb, DeckSampler, SamplerConfig};
use deckgym::{Deck, Game};

fn sampler(db: &str) -> DeckSampler {
    let db = DeckDb::load(Path::new("decks").join(db).as_path()).expect("deck db");
    DeckSampler::new(
        db,
        SamplerConfig {
            pure_mirror: 0.05,
            mirror: 0.10,
            archetypes: Vec::new(),
        },
    )
    .expect("sampler")
}

fn play(deck_a: Deck, deck_b: Deck, seed: u64, label: &str) {
    let players = create_players(deck_a, deck_b, vec![PlayerCode::R, PlayerCode::R]);
    let mut game = Game::new(players, seed);
    assert!(game.play().is_some(), "{label} did not terminate");
}

/// Every compiled tutorial deck, paired with its successor so each one is seen from both seats.
#[test]
fn every_tutorial_deck_plays() {
    let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
    let decks: Vec<_> = db
        .archetypes
        .iter()
        .flat_map(|archetype| {
            archetype
                .decks
                .iter()
                .map(move |entry| (archetype.name.clone(), entry.id.clone(), entry.build()))
        })
        .collect();
    assert_eq!(decks.len(), db.deck_count());

    for (index, (archetype, id, deck)) in decks.iter().enumerate() {
        let deck = deck.as_ref().expect("playable");
        let (_, opponent_id, opponent) = &decks[(index + 1) % decks.len()];
        play(
            deck.clone(),
            opponent.as_ref().expect("playable").clone(),
            index as u64,
            &format!("tutorial {archetype}/{id} vs {opponent_id}"),
        );
    }
}

#[test]
fn sampled_meta_decks_play() {
    let sampler = sampler("meta");
    let mut rng = env_rng(5, 0);
    for index in 0..120u64 {
        let [a, b] = sampler.sample(&mut rng).expect("draw");
        play(
            a.deck,
            b.deck,
            index,
            &format!("meta {}/{} vs {}/{}", a.archetype, a.id, b.archetype, b.id),
        );
    }
}
