mod attach_attack_player;
mod end_turn_player;
mod evolution_rusher_player;
mod expectiminimax_player;
mod human_player;
mod mcts_player;
mod random_player;
mod rl_seat_player;
mod value_function_player;
pub mod value_functions;
mod weighted_random_player;

pub use attach_attack_player::AttachAttackPlayer;
pub use end_turn_player::EndTurnPlayer;
pub use evolution_rusher_player::EvolutionRusherPlayer;
pub use expectiminimax_player::{ExpectiMiniMaxPlayer, ValueFunction};
pub use human_player::HumanPlayer;
pub use mcts_player::MctsPlayer;
pub use random_player::RandomPlayer;
pub use rl_seat_player::RlSeatPlayer;
pub use value_function_player::ValueFunctionPlayer;
pub use value_functions::*;
pub use weighted_random_player::WeightedRandomPlayer;

use crate::{actions::Action, Deck, State};
use rand::rngs::StdRng;
use std::fmt::Debug;

pub trait Player: Debug {
    fn get_deck(&self) -> Deck;
    fn decision_fn(
        &mut self,
        rng: &mut StdRng,
        state: &State,
        possible_actions: &[Action],
    ) -> Action;
}

/// Enum for allowed player strategies
///
/// `Eq + Hash` so a code can key §1.5.2's rating table, and `Ord` so a table of opponents has one
/// stable print order — a log whose rows permute between runs cannot be diffed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PlayerCode {
    AA,
    ET,
    R,
    H,
    W,
    M,
    V,
    E {
        max_depth: usize,
    },
    ER, // Evolution Rusher
    /// A baked model (`rl:<name>`), named by its directory under the models root — nested paths
    /// included, so a family of prototypes can live in one subfolder.
    ///
    /// Unlike every other code this one does not resolve to a working [`Player`]: it marks a seat
    /// the batched runner has to drive (`crate::rl::play`). A command that cannot do that must
    /// reject it rather than play it.
    RL {
        name: String,
    },
}
/// The inverse of [`parse_player_code`], so a code can name itself.
///
/// The RL eval harness labels a metric series per opponent, and those names are a run's identity
/// across months of curves — deriving them from `Debug` would let a rename in the enum break the
/// continuity of every past run silently.
impl std::fmt::Display for PlayerCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlayerCode::AA => write!(f, "aa"),
            PlayerCode::ET => write!(f, "et"),
            PlayerCode::R => write!(f, "r"),
            PlayerCode::H => write!(f, "h"),
            PlayerCode::W => write!(f, "w"),
            PlayerCode::M => write!(f, "m"),
            PlayerCode::V => write!(f, "v"),
            PlayerCode::E { max_depth } => write!(f, "e{max_depth}"),
            PlayerCode::ER => write!(f, "er"),
            PlayerCode::RL { name } => write!(f, "rl:{name}"),
        }
    }
}

/// Custom parser function enforcing case-insensitivity
pub fn parse_player_code(s: &str) -> Result<PlayerCode, String> {
    // Split before lowercasing: the rest of a model code is a path on disk, and folder names are
    // case-sensitive everywhere but here.
    if let Some((prefix, name)) = s.split_once(':') {
        if !prefix.eq_ignore_ascii_case("rl") {
            return Err(format!(
                "Invalid player code: {s}. The only prefixed form is 'rl:<model>'"
            ));
        }
        if name.is_empty() {
            return Err(
                "Invalid player code: 'rl:' needs a model name, e.g. 'rl:my_model'".to_string(),
            );
        }
        return Ok(PlayerCode::RL {
            name: name.to_string(),
        });
    }

    let lower = s.to_ascii_lowercase();

    // `e` followed by digits is the depth-parameterized ExpectiMiniMax. Anything else starting
    // with `e` falls through to the table below, which owns `er` and `et` — rejecting it here is
    // how `et` spent its life unreachable from every command line.
    if let Some(rest) = lower.strip_prefix('e') {
        if let Ok(max_depth) = rest.parse::<usize>() {
            return Ok(PlayerCode::E { max_depth });
        }
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return Err(format!("Invalid player code: {s}. Use 'e<number>' for ExpectiMiniMax with depth, e.g., 'e2', 'e5'"));
        }
    }

    match lower.as_str() {
        "aa" => Ok(PlayerCode::AA),
        "et" => Ok(PlayerCode::ET),
        "r" => Ok(PlayerCode::R),
        "h" => Ok(PlayerCode::H),
        "w" => Ok(PlayerCode::W),
        "m" => Ok(PlayerCode::M),
        "v" => Ok(PlayerCode::V),
        "e" => Ok(PlayerCode::E { max_depth: 3 }), // Default depth
        "er" => Ok(PlayerCode::ER),
        _ => Err(format!("Invalid player code: {s}")),
    }
}

pub fn parse_player_code_generic(s: String) -> Result<PlayerCode, String> {
    parse_player_code(s.as_ref())
}

pub fn fill_code_array(maybe_players: Option<Vec<PlayerCode>>) -> Vec<PlayerCode> {
    match maybe_players {
        Some(mut player_codes) => {
            if player_codes.is_empty() || player_codes.len() > 2 {
                panic!("Invalid number of players");
            } else if player_codes.len() == 1 {
                player_codes.push(PlayerCode::R);
            }
            player_codes
        }
        None => vec![PlayerCode::R, PlayerCode::R],
    }
}

pub fn create_players(
    deck_a: Deck,
    deck_b: Deck,
    players: Vec<PlayerCode>,
) -> Vec<Box<dyn Player>> {
    let player_a: Box<dyn Player> = get_player(deck_a.clone(), &players[0]);
    let player_b: Box<dyn Player> = get_player(deck_b.clone(), &players[1]);
    vec![player_a, player_b]
}

fn get_player(deck: Deck, player: &PlayerCode) -> Box<dyn Player> {
    match player {
        PlayerCode::AA => Box::new(AttachAttackPlayer { deck }),
        PlayerCode::ET => Box::new(EndTurnPlayer { deck }),
        PlayerCode::R => Box::new(RandomPlayer { deck }),
        PlayerCode::H => Box::new(HumanPlayer { deck }),
        PlayerCode::W => Box::new(WeightedRandomPlayer { deck }),
        PlayerCode::M => Box::new(MctsPlayer::new(deck, 100)),
        PlayerCode::V => Box::new(ValueFunctionPlayer { deck }),
        PlayerCode::E { max_depth } => Box::new(ExpectiMiniMaxPlayer {
            deck,
            max_depth: *max_depth,
            write_debug_trees: false,
            value_function: Box::new(value_functions::baseline_value_function),
        }),
        PlayerCode::ER => Box::new(EvolutionRusherPlayer { deck }),
        PlayerCode::RL { name } => Box::new(RlSeatPlayer {
            deck,
            name: name.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every scripted code the enum can print is a code the parser takes back — `et` was not, and
    /// no command line could reach `EndTurnPlayer` because of it.
    #[test]
    fn scripted_codes_round_trip() {
        for code in [
            PlayerCode::AA,
            PlayerCode::ET,
            PlayerCode::R,
            PlayerCode::H,
            PlayerCode::W,
            PlayerCode::M,
            PlayerCode::V,
            PlayerCode::E { max_depth: 3 },
            PlayerCode::ER,
        ] {
            assert_eq!(parse_player_code(&code.to_string()), Ok(code));
        }
        assert!(parse_player_code("e2.5").is_err());
        assert!(parse_player_code("ex").is_err());
    }

    /// `Display` is what a `.toml` panel and an eval log key on, so it has to be the exact string
    /// the parser accepts back.
    #[test]
    fn model_codes_round_trip() {
        for code in [
            PlayerCode::RL {
                name: "long_v2_b972".to_string(),
            },
            PlayerCode::RL {
                name: "proto/mmd_v3".to_string(),
            },
        ] {
            assert_eq!(parse_player_code(&code.to_string()), Ok(code));
        }
    }

    /// A model name is a path on disk; lowercasing it the way the scripted codes are lowercased
    /// would silently look up the wrong directory on a case-sensitive filesystem.
    #[test]
    fn model_names_keep_their_case() {
        assert_eq!(
            parse_player_code("RL:Long_V2"),
            Ok(PlayerCode::RL {
                name: "Long_V2".to_string()
            })
        );
    }

    #[test]
    fn colon_forms_other_than_rl_are_rejected() {
        assert!(parse_player_code("mcts:100").is_err());
        assert!(parse_player_code("rl:").is_err());
    }
}
