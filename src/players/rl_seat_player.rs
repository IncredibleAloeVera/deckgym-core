use std::fmt::{Debug, Formatter, Result as FmtResult};

use rand::rngs::StdRng;

use super::Player;
use crate::{actions::Action, Deck, State};

/// The seat a baked model occupies, as far as the engine is concerned.
///
/// A model is not a [`Player`] and cannot become one: `decision_fn` is blocking and per-seat, so
/// behind it every game is a batch-1 forward — the reason `crate::rl::env` inverts the control flow
/// in the first place. What the engine still needs from an agent seat is its *deck* (to deal the
/// game) and its *name* (the `-v` log prints `{player:?}` on every frame), and that is all this
/// carries.
///
/// `decision_fn` therefore panics rather than falling back to a heuristic: a silent fallback would
/// report a winrate for a model that never played, which is worse than a crash by exactly the
/// amount nobody would notice.
pub struct RlSeatPlayer {
    pub deck: Deck,
    pub name: String,
}

impl Debug for RlSeatPlayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "rl:{}", self.name)
    }
}

impl Player for RlSeatPlayer {
    fn get_deck(&self) -> Deck {
        self.deck.clone()
    }

    fn decision_fn(&mut self, _: &mut StdRng, _: &State, _: &[Action]) -> Action {
        panic!(
            "rl:{} was asked for a decision through the Player trait: a model seat is only driven \
             by the batched runner (`deckgym simulate`, built with --features rl-model)",
            self.name
        )
    }
}
