use crate::{
    actions::{Action, SimpleAction},
    players::{create_players, Player, PlayerCode},
    Deck, Game, State,
};
use rand::{thread_rng, Rng};
use std::{error::Error, path::PathBuf};

/// The seats the TUI drives itself, in seat order. Empty on a build without `rl-model`, where an
/// `rl:` code is refused at construction instead.
#[cfg(feature = "rl-model")]
type ModelSeats = [Option<Box<dyn crate::rl::seat::ModelSeat>>; 2];
#[cfg(not(feature = "rl-model"))]
type ModelSeats = [Option<()>; 2];

pub enum AppMode {
    Replay {
        states: Vec<State>,
        actions: Vec<Action>,
        current_index: usize,
    },
    Interactive {
        game: Box<Game<'static>>,
        current_actor: usize,
        possible_actions: Vec<Action>,
        action_history: Vec<Action>, // Track actions as they happen
        turn_history: Vec<u8>,       // Track turn number when each action was taken
        /// Which seats the human owns; every other one is played for them on tick.
        human: [bool; 2],
        models: ModelSeats,
    },
}

/// What the TUI was asked to show. A struct because the model seats added two arguments that only
/// matter to one kind of run, and `App::new` was already at four.
pub struct AppConfig {
    pub deck_a_path: String,
    pub deck_b_path: String,
    pub players: Vec<PlayerCode>,
    pub seed: Option<u64>,
    /// Folder an `rl:<name>` seat is resolved against.
    pub models_root: PathBuf,
    /// Run those seats on CUDA. Needs a `rl-model-cuda` build; at one game and one forward per
    /// frame it buys nothing but is here so a GPU-only machine is not a special case.
    pub cuda: bool,
}

impl AppConfig {
    pub fn new(deck_a_path: &str, deck_b_path: &str, players: Vec<PlayerCode>) -> Self {
        AppConfig {
            deck_a_path: deck_a_path.to_string(),
            deck_b_path: deck_b_path.to_string(),
            players,
            seed: None,
            models_root: PathBuf::from("models"),
            cuda: false,
        }
    }
}

pub enum SelectionState {
    AwaitingActionSelection,
    ActionSelected { action_index: usize },
}

pub struct App {
    pub mode: AppMode,
    /// How to name each seat in the UI. `None` for an ordinary player code, which the footer
    /// already describes; `Some` for a model, whose name and rating are the whole reason somebody
    /// is watching this game.
    pub seat_labels: [Option<String>; 2],
    pub selection_state: SelectionState,
    pub scroll_offset: u16,
    pub player_hand_scroll: usize,
    pub opponent_hand_scroll: usize,
    pub lock_actions_center: bool,
}

fn action_priority_for_tui(action: &SimpleAction) -> u8 {
    match action {
        SimpleAction::Place(_, _) => 0,
        SimpleAction::Evolve { .. } => 1,
        SimpleAction::Play { .. } => 2,
        SimpleAction::Attach { .. }
        | SimpleAction::AttachFromDiscard { .. }
        | SimpleAction::AttachTool { .. } => 3,
        SimpleAction::Attack(_) => 4,
        SimpleAction::Retreat(_) => 5,
        SimpleAction::EndTurn => 255,
        _ => 6,
    }
}

/// Load the baked model behind every `rl:` seat, once, before the game starts.
#[cfg(feature = "rl-model")]
fn load_models(config: &AppConfig, seed: u64) -> Result<ModelSeats, Box<dyn Error>> {
    let mut seats: ModelSeats = [None, None];
    for (seat, code) in config.players.iter().enumerate().take(2) {
        if let PlayerCode::RL { name } = code {
            seats[seat] = Some(crate::rl::seat::load_seat(
                &config.models_root,
                name,
                config.cuda,
                seed,
            )?);
        }
    }
    Ok(seats)
}

/// The same entry point in a build without the deep-learning stack: `rl:` parses there, so the
/// error can name what to rebuild with instead of failing as an unknown seat.
#[cfg(not(feature = "rl-model"))]
fn load_models(config: &AppConfig, _seed: u64) -> Result<ModelSeats, Box<dyn Error>> {
    if let Some(code) = config
        .players
        .iter()
        .find(|code| matches!(code, PlayerCode::RL { .. }))
    {
        return Err(format!(
            "`{code}` needs a build with the model stack: rebuild with --features \"tui,rl-model\""
        )
        .into());
    }
    Ok([None, None])
}

#[cfg(feature = "rl-model")]
fn seat_labels(models: &ModelSeats) -> [Option<String>; 2] {
    let label = |seat: &Option<Box<dyn crate::rl::seat::ModelSeat>>| {
        seat.as_ref()
            .map(|model| format!("rl:{} ({:.0})", model.name(), model.rating()))
    };
    [label(&models[0]), label(&models[1])]
}

#[cfg(not(feature = "rl-model"))]
fn seat_labels(_models: &ModelSeats) -> [Option<String>; 2] {
    [None, None]
}

/// The TUI installs no logger and draws its own battle log, so the engine's per-frame one has
/// nowhere to go — and its colouring has a `todo!()` on Colorless decks, which a frame resolved
/// with it on walks straight into.
fn silence_engine_log(game: &mut Game<'_>) {
    game.set_debug(false);
}

/// One frame, from whichever seat owes it.
#[cfg(feature = "rl-model")]
fn advance(game: &mut Game<'_>, models: &mut ModelSeats) -> Result<Action, String> {
    crate::rl::seat::play_tick_with(game, models)
}

#[cfg(not(feature = "rl-model"))]
fn advance(game: &mut Game<'_>, _models: &mut ModelSeats) -> Result<Action, String> {
    Ok(game.play_tick())
}

fn sort_actions_for_tui(actions: &mut Vec<Action>) {
    let mut indexed_actions: Vec<(usize, Action)> = actions.drain(..).enumerate().collect();
    indexed_actions.sort_by_key(|(idx, action)| (action_priority_for_tui(&action.action), *idx));
    *actions = indexed_actions
        .into_iter()
        .map(|(_, action)| action)
        .collect();
}

impl App {
    pub fn new(config: &AppConfig) -> Result<App, Box<dyn Error>> {
        // Load decks from files
        let deck_a = Deck::from_file(&config.deck_a_path)?;
        let deck_b = Deck::from_file(&config.deck_b_path)?;
        let player_codes = config.players.clone();

        // Detect if any player is human
        let has_human = player_codes.contains(&PlayerCode::H);

        // Use provided seed or generate a random one
        let seed = config.seed.unwrap_or_else(|| {
            let mut rng = thread_rng();
            rng.gen::<u64>()
        });

        let mut models = load_models(config, seed)?;
        let seat_labels = seat_labels(&models);

        let mode = if has_human {
            // Interactive mode - create live game
            let players: Vec<Box<dyn Player>> =
                create_players(deck_a, deck_b, player_codes.clone());
            let mut game = Box::new(Game::new(players, seed));
            silence_engine_log(&mut game);
            game.enable_action_trace();
            // Same player mode the training env runs in: a model seat that observed revealed cards
            // while it learned must observe them here too, or it plays a game it never saw.
            game.enable_belief();

            // Get initial state and possible actions
            let (current_actor, mut possible_actions) =
                game.get_state_clone().generate_possible_actions();
            sort_actions_for_tui(&mut possible_actions);

            let mut human = [false; 2];
            for (seat, code) in player_codes.iter().enumerate().take(2) {
                human[seat] = *code == PlayerCode::H;
            }

            AppMode::Interactive {
                game,
                current_actor,
                possible_actions,
                action_history: vec![],
                turn_history: vec![],
                human,
                models,
            }
        } else {
            // Replay mode - pre-compute entire game
            let players: Vec<Box<dyn Player>> = create_players(deck_a, deck_b, player_codes);
            let mut game = Game::new(players, seed);
            silence_engine_log(&mut game);
            game.enable_action_trace();
            game.enable_belief();

            let mut states = Vec::new();
            let mut actions = Vec::new();
            states.push(game.get_state_clone());

            while !game.is_game_over() {
                let action = advance(&mut game, &mut models)?;
                actions.push(action);
                states.push(game.get_state_clone());
            }

            AppMode::Replay {
                states,
                actions,
                current_index: 0,
            }
        };

        Ok(App {
            mode,
            seat_labels,
            selection_state: SelectionState::AwaitingActionSelection,
            scroll_offset: 0,
            player_hand_scroll: 0,
            opponent_hand_scroll: 0,
            lock_actions_center: true,
        })
    }

    pub fn get_state(&self) -> State {
        match &self.mode {
            AppMode::Replay {
                states,
                current_index,
                ..
            } => states[*current_index].clone(),
            AppMode::Interactive { game, .. } => game.get_state_clone(),
        }
    }

    // Helper method to calculate turn boundaries in the battle log
    // Returns the scroll offset (line number) where each turn header appears
    fn calculate_turn_boundaries(&self) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let mut line_count = 0;

        match &self.mode {
            AppMode::Interactive {
                action_history,
                turn_history,
                ..
            } => {
                // Even if there are no recorded actions yet, we should at least
                // expose the initial turn header so "jump" can move the battle
                // log to the start of a turn in interactive mode.
                let mut current_turn: u8 = if !turn_history.is_empty() {
                    turn_history[0]
                } else {
                    // No actions yet - use the game's current turn number as the initial header
                    self.get_state().turn_count
                };

                // Initial turn header
                boundaries.push(line_count);
                line_count += 1;

                // For each recorded action add its line and detect turn changes
                for i in 0..action_history.len() {
                    // Each action occupies a single line
                    line_count += 1;

                    // If next action has different turn, add header boundary
                    if i + 1 < turn_history.len() {
                        let next_turn = turn_history[i + 1];
                        if next_turn != current_turn {
                            line_count += 1; // empty line before header
                            boundaries.push(line_count);
                            line_count += 1; // header line
                            current_turn = next_turn;
                        }
                    }
                }
            }
            AppMode::Replay {
                states,
                actions,
                current_index,
                ..
            } => {
                if states.is_empty() {
                    return boundaries;
                }

                let mut current_turn = states[0].turn_count;
                boundaries.push(line_count); // Initial turn header
                line_count += 1;

                for i in 0..actions.len() {
                    // Add cursor marker line if this is the current action
                    if i == *current_index && i < actions.len() {
                        line_count += 1; // Cursor marker ">>> CURRENT <<<"
                    }

                    // Each action takes exactly 1 line
                    line_count += 1;

                    // Check if turn changed after this action
                    if i + 1 < states.len() {
                        let next_turn = states[i + 1].turn_count;
                        if next_turn != current_turn && i + 1 < actions.len() {
                            line_count += 1; // Empty line
                            boundaries.push(line_count);
                            line_count += 1; // Turn header
                            current_turn = next_turn;
                        }
                    }
                }
            }
        }

        boundaries
    }

    pub fn next_state(&mut self) {
        if let AppMode::Replay {
            current_index,
            states,
            ..
        } = &mut self.mode
        {
            if *current_index < states.len() - 1 {
                *current_index += 1;
            }
        }
    }

    pub fn prev_state(&mut self) {
        if let AppMode::Replay { current_index, .. } = &mut self.mode {
            if *current_index > 0 {
                *current_index -= 1;
            }
        }
    }

    pub fn toggle_lock_actions_center(&mut self) {
        self.lock_actions_center = !self.lock_actions_center;
    }

    fn jump_turn(&mut self, forward: bool) {
        if self.lock_actions_center {
            // Center lock on: jump state to beginning of next/previous turn
            match &mut self.mode {
                AppMode::Replay {
                    states,
                    current_index,
                    ..
                } => {
                    let valid_range = if forward {
                        *current_index < states.len()
                    } else {
                        *current_index > 0
                    };

                    if valid_range {
                        let current_turn = states[*current_index].turn_count;

                        // Find a state with different turn number
                        let mut target_turn = None;
                        if forward {
                            for state in states.iter().skip(*current_index + 1) {
                                if state.turn_count != current_turn {
                                    target_turn = Some(state.turn_count);
                                    break;
                                }
                            }
                        } else {
                            for state in states.iter().take(*current_index).rev() {
                                if state.turn_count != current_turn {
                                    target_turn = Some(state.turn_count);
                                    break;
                                }
                            }
                        }

                        // If we found a different turn, find the FIRST state of that turn
                        if let Some(turn) = target_turn {
                            for (i, state) in states.iter().enumerate() {
                                if state.turn_count == turn {
                                    *current_index = i;
                                    return;
                                }
                            }
                        }
                    }
                }
                AppMode::Interactive { .. } => {
                    // In interactive mode we don't have a precomputed states vector,
                    // but we can still move the battle log view to the next/previous
                    // turn header. Compute turn boundaries and adjust the scroll
                    // offset similarly to the non-center-lock branch.
                    let boundaries = self.calculate_turn_boundaries();
                    if boundaries.is_empty() {
                        return;
                    }

                    let current_line = self.scroll_offset as usize;
                    if forward {
                        if let Some(&next_line) =
                            boundaries.iter().find(|&&line| line > current_line)
                        {
                            self.scroll_offset = next_line as u16;
                        }
                    } else if let Some(&prev_line) =
                        boundaries.iter().rev().find(|&&line| line < current_line)
                    {
                        self.scroll_offset = prev_line as u16;
                    }
                }
            }
        } else {
            // Center lock off: just scroll the battle log to next/previous turn header
            let boundaries = self.calculate_turn_boundaries();
            let current_line = self.scroll_offset as usize;

            if forward {
                if let Some(&next_line) = boundaries.iter().find(|&&line| line > current_line) {
                    self.scroll_offset = next_line as u16;
                }
            } else if let Some(&prev_line) =
                boundaries.iter().rev().find(|&&line| line < current_line)
            {
                self.scroll_offset = prev_line as u16;
            }
        }
    }

    pub fn jump_to_next_turn(&mut self) {
        self.jump_turn(true);
    }

    pub fn jump_to_prev_turn(&mut self) {
        self.jump_turn(false);
    }

    pub fn scroll_page_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(10);
    }

    pub fn scroll_page_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(10);
    }

    pub fn scroll_player_hand_left(&mut self) {
        self.player_hand_scroll = self.player_hand_scroll.saturating_sub(1);
    }

    pub fn scroll_player_hand_right(&mut self) {
        let player_hand_size = self.get_state().hands[self.perspective()].len();
        if self.player_hand_scroll < player_hand_size.saturating_sub(5) {
            self.player_hand_scroll += 1;
        }
    }

    pub fn scroll_opponent_hand_left(&mut self) {
        self.opponent_hand_scroll = self.opponent_hand_scroll.saturating_sub(1);
    }

    pub fn scroll_opponent_hand_right(&mut self) {
        let opponent_hand_size = self.get_state().hands[1 - self.perspective()].len();
        if self.opponent_hand_scroll < opponent_hand_size.saturating_sub(5) {
            self.opponent_hand_scroll += 1;
        }
    }

    // Interactive mode methods
    pub fn handle_action_selection(&mut self, index: usize) {
        if let AppMode::Interactive {
            possible_actions, ..
        } = &self.mode
        {
            if index < possible_actions.len() {
                self.selection_state = SelectionState::ActionSelected {
                    action_index: index,
                };
            }
        }
    }

    pub fn tick_game(&mut self) -> Result<(), String> {
        let AppMode::Interactive {
            game,
            current_actor,
            possible_actions,
            action_history,
            turn_history,
            human,
            models,
        } = &mut self.mode
        else {
            return Ok(());
        };

        match &self.selection_state {
            SelectionState::ActionSelected { action_index } => {
                // Record current turn before applying action
                let current_turn = game.get_state_clone().turn_count;

                // Apply the selected action
                let action = possible_actions[*action_index].clone();
                action_history.push(action.clone());
                turn_history.push(current_turn);
                // Through `resolve_decision`, not `apply_action`: the History tokens a model seat
                // reads are built from the trace, and a human move that skipped it would leave the
                // opponent modelling a game where nobody moved.
                let (actor, enumeration) = game.state().generate_possible_actions();
                game.resolve_decision(actor, &enumeration, action);

                // Reset selection state
                self.selection_state = SelectionState::AwaitingActionSelection;
            }
            SelectionState::AwaitingActionSelection => {
                // Anything that is not a human seat plays itself; a model seat goes through the
                // policy, every other one through its `Player`.
                if human[*current_actor] {
                    return Ok(());
                }
                let current_turn = game.get_state_clone().turn_count;
                let action = advance(game, models)?;
                action_history.push(action);
                turn_history.push(current_turn);
            }
        }

        // Refresh game state and possible actions for the next frame
        let (new_actor, mut new_actions) = game.get_state_clone().generate_possible_actions();
        sort_actions_for_tui(&mut new_actions);
        *current_actor = new_actor;
        *possible_actions = new_actions;
        Ok(())
    }

    pub fn is_game_over(&self) -> bool {
        match &self.mode {
            AppMode::Replay { .. } => false, // Replay never "ends" automatically
            AppMode::Interactive { game, .. } => game.is_game_over(),
        }
    }

    pub fn get_possible_actions(&self) -> Vec<Action> {
        match &self.mode {
            AppMode::Replay {
                states,
                current_index,
                ..
            } => {
                let mut actions = states[*current_index].generate_possible_actions().1;
                sort_actions_for_tui(&mut actions);
                actions
            }
            AppMode::Interactive {
                possible_actions, ..
            } => possible_actions.clone(),
        }
    }

    pub fn get_current_actor(&self) -> usize {
        match &self.mode {
            AppMode::Replay {
                states,
                current_index,
                ..
            } => states[*current_index].generate_possible_actions().0,
            AppMode::Interactive { current_actor, .. } => *current_actor,
        }
    }

    /// The seat the board is drawn from: yours when you are playing, P2 otherwise.
    ///
    /// Not a cosmetic choice. The opponent's hand is rendered face down, so a mat drawn from the
    /// wrong seat would show the person at the keyboard the cards they are playing against.
    pub fn perspective(&self) -> usize {
        match &self.mode {
            AppMode::Interactive { human, .. } => human.iter().position(|seat| *seat).unwrap_or(1),
            AppMode::Replay { .. } => 1,
        }
    }

    /// Whether the frame on screen is one the person at the keyboard has to answer.
    pub fn is_human_turn(&self) -> bool {
        match &self.mode {
            AppMode::Replay { .. } => false,
            AppMode::Interactive {
                human,
                current_actor,
                ..
            } => human[*current_actor],
        }
    }

    pub fn get_current_state_index(&self) -> usize {
        match &self.mode {
            AppMode::Replay { current_index, .. } => *current_index,
            AppMode::Interactive { .. } => 0, // Not really meaningful in interactive mode
        }
    }

    pub fn get_states_len(&self) -> usize {
        match &self.mode {
            AppMode::Replay { states, .. } => states.len(),
            AppMode::Interactive { .. } => 1, // Only current state
        }
    }

    pub fn get_actions(&self) -> Vec<Action> {
        match &self.mode {
            AppMode::Replay { actions, .. } => actions.clone(),
            AppMode::Interactive { action_history, .. } => action_history.clone(),
        }
    }

    pub fn get_turn_history(&self) -> Option<Vec<u8>> {
        match &self.mode {
            AppMode::Interactive { turn_history, .. } => Some(turn_history.clone()),
            _ => None,
        }
    }
}

/// A model seat in the TUI, on the two paths that reach it: the replay is played out in full
/// before anything is drawn, and the interactive game is advanced a frame at a time.
///
/// Both need a baked model on disk; `models/default_mmd_prot` is the one this repository tracks.
///
/// **Ignored: nothing under `models/` loads.** All five tracked directories were baked at
/// `schema_version = 1` and have been rejected by `check_schema` since the bump to 2 — version 4's
/// belief render only widens the gap. That is the check working, not a bug to route around, and no
/// re-bake rescues the old weights: their input layer is the wrong shape. These come back with the
/// first model trained on the current schema.
#[cfg(all(test, feature = "rl-model"))]
mod model_seat_tests {
    use super::*;

    const MODEL: &str = "default_mmd_prot";

    fn config(players: Vec<PlayerCode>) -> AppConfig {
        AppConfig {
            seed: Some(7),
            ..AppConfig::new(
                "example_decks/venusaur-exeggutor.txt",
                "example_decks/weezing-arbok.txt",
                players,
            )
        }
    }

    #[test]
    #[ignore = "no model under models/ is baked against the current schema"]
    fn a_replay_against_a_model_plays_to_the_end() {
        let app = App::new(&config(vec![
            PlayerCode::RL {
                name: MODEL.to_string(),
            },
            PlayerCode::R,
        ]))
        .expect("the tracked model loads");

        let actions = app.get_actions();
        assert!(!actions.is_empty(), "a replay records what was played");
        assert_eq!(app.get_states_len(), actions.len() + 1);
        assert!(
            matches!(&app.mode, AppMode::Replay { states, .. }
                if states.last().expect("states").winner.is_some()),
            "the game was played to a winner"
        );
    }

    /// The mat is drawn from the human's seat, so a model on P2 must not put the person at the
    /// keyboard behind P1's face-down hand.
    #[test]
    #[ignore = "no model under models/ is baked against the current schema"]
    fn the_human_keeps_the_near_side_of_the_mat_on_either_seat() {
        let model = PlayerCode::RL {
            name: MODEL.to_string(),
        };

        let far = App::new(&config(vec![model.clone(), PlayerCode::H])).expect("model on P1");
        assert_eq!(far.perspective(), 1);

        let near = App::new(&config(vec![PlayerCode::H, model])).expect("model on P2");
        assert_eq!(near.perspective(), 0);
    }

    #[test]
    #[ignore = "no model under models/ is baked against the current schema"]
    fn an_interactive_game_plays_the_model_and_then_waits_for_the_human() {
        let mut app = App::new(&config(vec![
            PlayerCode::RL {
                name: MODEL.to_string(),
            },
            PlayerCode::H,
        ]))
        .expect("the tracked model loads");

        // The model owns seat 0 and opens the game; it has to move by itself for the human's first
        // prompt to ever appear.
        for _ in 0..64 {
            app.tick_game().expect("the model answers");
            if app.get_current_actor() == 1 {
                break;
            }
        }
        assert_eq!(app.get_current_actor(), 1, "the human is asked to play");
        assert!(!app.get_actions().is_empty(), "the model played first");

        let played = app.get_actions().len();
        app.tick_game().expect("no model frame to answer");
        assert_eq!(
            app.get_actions().len(),
            played,
            "a human seat is never played for"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::sort_actions_for_tui;
    use crate::{
        actions::{Action, SimpleAction},
        models::{Attack, Card, EnergyType, PokemonCard, TrainerCard, TrainerType},
    };

    fn action(action: SimpleAction) -> Action {
        Action {
            actor: 1,
            action,
            is_stack: false,
        }
    }

    fn test_pokemon(name: &str) -> Card {
        Card::Pokemon(PokemonCard {
            id: format!("test-{name}"),
            name: name.to_string(),
            stage: 0,
            evolves_from: None,
            hp: 60,
            energy_type: EnergyType::Colorless,
            ability: None,
            attacks: vec![],
            weakness: None,
            retreat_cost: vec![],
            rarity: String::new(),
            booster_pack: String::new(),
        })
    }

    #[test]
    fn sorts_actions_for_tui_in_expected_priority_order() {
        let mut actions = vec![
            action(SimpleAction::EndTurn),
            action(SimpleAction::Retreat(1)),
            action(SimpleAction::Attack(Attack {
                energy_required: vec![],
                title: "Test Attack".to_string(),
                fixed_damage: 0,
                effect: None,
            })),
            action(SimpleAction::Attach {
                attachments: vec![],
                is_turn_energy: true,
            }),
            action(SimpleAction::Play {
                trainer_card: TrainerCard {
                    id: "potion".to_string(),
                    trainer_card_type: TrainerType::Item,
                    name: "Potion".to_string(),
                    effect: String::new(),
                    rarity: String::new(),
                    booster_pack: String::new(),
                },
            }),
            action(SimpleAction::Evolve {
                evolution: test_pokemon("Ivysaur"),
                in_play_idx: 0,
                from_deck: false,
            }),
            action(SimpleAction::Place(test_pokemon("Bulbasaur"), 1)),
        ];

        sort_actions_for_tui(&mut actions);

        assert!(matches!(actions[0].action, SimpleAction::Place(_, _)));
        assert!(matches!(actions[1].action, SimpleAction::Evolve { .. }));
        assert!(matches!(actions[2].action, SimpleAction::Play { .. }));
        assert!(matches!(actions[3].action, SimpleAction::Attach { .. }));
        assert!(matches!(actions[4].action, SimpleAction::Attack(_)));
        assert!(matches!(actions[5].action, SimpleAction::Retreat(_)));
        assert!(matches!(actions[6].action, SimpleAction::EndTurn));
    }
}
