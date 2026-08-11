//! A baked model on the seat of a game somebody is *watching*, one decision at a time.
//!
//! [`crate::rl::play`] exists because a thousand-game run cannot afford batch-1 forwards. The TUI
//! is the opposite trade: one game, advanced at a human's pace, with the engine's own `Game` in the
//! caller's hands (the replay buffer and the interactive log are built from it). Batching has
//! nothing to widen there, and a ≈ 40 ms forward is under a tick, so this module hands the model a
//! single decision point and keeps the control flow the TUI already has.
//!
//! The model still does not go behind `Player::decision_fn` — it cannot, that call sees neither the
//! action trace the History tokens are built from nor a way to fail. [`play_tick_with`] is the
//! substitute: the same frame-by-frame advance `play_tick` gives, with the policy consulted on the
//! frames a model seat owns.

use std::path::Path;

use burn::tensor::backend::Backend;
use rand::rngs::StdRng;

use crate::actions::Action;
use crate::rl::action_mask::{ActionMask, Regime};
use crate::rl::env::env_rng;
use crate::rl::model::config::ModelConfig;
use crate::rl::model::input::{DecisionPoint, ModelInput};
use crate::rl::model::RlModel;
use crate::rl::observation::Observation;
use crate::rl::text_embedding::TextEmbeddings;
use crate::rl::train::baked::{load_model, Baked};
use crate::rl::train::rollout::sample_entry;
use crate::rl::{get_observation, project_action_mask};
use crate::Game;

/// Stream tag, so a TUI game's action sampling cannot alias the seed the game itself was dealt
/// from.
const STREAM_TUI_ACTION: u64 = 0x5455_4900_0000_0001;

/// A policy that can answer one decision point.
///
/// Object-safe and free of the backend parameter: the TUI picks CPU or CUDA at startup and should
/// not become generic over that choice everywhere it holds a seat.
pub trait ModelSeat {
    /// The `rl:<name>` this seat was loaded from, for the log and the header.
    fn name(&self) -> &str;

    /// Its rating, as the bake recorded it.
    fn rating(&self) -> f64;

    /// Draw a legal action from the masked policy. `mask` must be the projection of the same
    /// enumeration `observation` was built from.
    fn choose(&mut self, observation: &Observation, mask: &ActionMask) -> Result<Action, String>;
}

struct BakedSeat<B: Backend> {
    name: String,
    rating: f64,
    model: RlModel<B>,
    config: ModelConfig,
    device: B::Device,
    rng: StdRng,
}

impl<B: Backend> ModelSeat for BakedSeat<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn rating(&self) -> f64 {
        self.rating
    }

    fn choose(&mut self, observation: &Observation, mask: &ActionMask) -> Result<Action, String> {
        let points = [DecisionPoint { observation, mask }];
        let policy = self
            .model
            .forward(&ModelInput::<B>::from_points(
                &points,
                &self.config,
                &self.device,
            ))
            .policy
            .to_data()
            .to_vec::<f32>()
            .map_err(|err| format!("policy readback failed: {err:?}"))?;
        let (entry, _) = sample_entry(mask, &policy, &mut self.rng);
        mask.select(entry.head, entry.index).ok_or_else(|| {
            format!(
                "rl:{}: sampled {:?}[{}], which the mask does not resolve",
                self.name, entry.head, entry.index
            )
        })
    }
}

/// Load `root/<name>/` onto the CPU, or onto CUDA when `cuda` and the build has that backend.
pub fn load_seat(
    root: &Path,
    name: &str,
    cuda: bool,
    seed: u64,
) -> Result<Box<dyn ModelSeat>, String> {
    if cuda {
        #[cfg(feature = "rl-model-cuda")]
        {
            return load_on::<burn::backend::Cuda>(root, name, seed);
        }
        #[cfg(not(feature = "rl-model-cuda"))]
        return Err(
            "--cuda needs a build with --features rl-model-cuda (this one has none)".to_string(),
        );
    }
    load_on::<burn::backend::NdArray>(root, name, seed)
}

fn load_on<B: Backend>(root: &Path, name: &str, seed: u64) -> Result<Box<dyn ModelSeat>, String> {
    let baked = Baked::load(root, name)?;
    let device = B::Device::default();
    let model = load_model::<B>(&baked, &TextEmbeddings::zeros(), &device)?;
    Ok(Box::new(BakedSeat {
        name: name.to_string(),
        rating: baked.meta.rating.rating,
        config: baked.meta.model.clone(),
        model,
        device,
        rng: env_rng(seed, STREAM_TUI_ACTION),
    }))
}

/// Advance `game` by one frame, asking `seats[actor]` on the frames a model owns, and return the
/// action that was applied.
///
/// Drop-in for `Game::play_tick`, which stays the resolver for every other frame — forced ones
/// included, on a model seat too (§1.3.6.3): those have one candidate and never reach a policy.
///
/// The caller must have enabled the action trace, or the model plays without History (§1.2.7);
/// [`crate::rl::env::Env`] does that for the batched path and the TUI does it in `App::new`.
pub fn play_tick_with(
    game: &mut Game<'_>,
    seats: &mut [Option<Box<dyn ModelSeat>>; 2],
) -> Result<Action, String> {
    let (actor, actions) = game.state().generate_possible_actions();
    let Some(seat) = seats[actor]
        .as_mut()
        .filter(|_| Regime::of(game.state(), &actions).needs_policy())
    else {
        return Ok(game.play_tick());
    };

    let observation = get_observation(
        game.state(),
        actor,
        &actions,
        game.action_trace(),
        game.belief(),
    );
    let mask = project_action_mask(game.state(), &actions, &observation);
    let action = seat.choose(&observation, &mask)?;
    Ok(game.resolve_decision(actor, &actions, action))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::{create_players, PlayerCode};
    use crate::Deck;

    /// The scripted path through `play_tick_with` is the one every non-model seat takes, and it
    /// must stay exactly `play_tick` — a TUI built against this should not play a different game
    /// than the CLI does from the same seed.
    #[test]
    fn a_game_with_no_model_seat_plays_as_play_tick_does() {
        let deck_a = Deck::from_file("example_decks/venusaur-exeggutor.txt").expect("deck A");
        let deck_b = Deck::from_file("example_decks/weezing-arbok.txt").expect("deck B");
        let codes = vec![PlayerCode::R, PlayerCode::R];

        let mut driven = Game::new(
            create_players(deck_a.clone(), deck_b.clone(), codes.clone()),
            42,
        );
        let mut plain = Game::new(create_players(deck_a, deck_b, codes), 42);
        let mut seats: [Option<Box<dyn ModelSeat>>; 2] = [None, None];

        while !plain.is_game_over() {
            let expected = plain.play_tick();
            let applied = play_tick_with(&mut driven, &mut seats).expect("no model to fail");
            assert_eq!(format!("{applied:?}"), format!("{expected:?}"));
        }
        assert!(driven.is_game_over());
    }
}
