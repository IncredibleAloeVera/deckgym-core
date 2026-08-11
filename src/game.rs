use colored::Colorize;
use log::{debug, info, trace};
use rand::{rngs::StdRng, SeedableRng};
use uuid::Uuid;

use crate::{
    actions::{apply_action, Action},
    belief::BeliefTracker,
    models::EnergyType,
    players::Player,
    rl::{self, ActionMask, ActionTrace, Observation},
    simulation_event_handler::{CompositeSimulationEventHandler, SimulationEventHandler},
    state::GameOutcome,
    State,
};

// It has a lifetime to allow it to borrow the event handler mutably for the duration of the game
pub struct Game<'a> {
    seed: u64,
    rng: StdRng,
    id: Uuid,
    players: Vec<Box<dyn Player>>,

    state: State,

    debug: bool,
    event_handler: Option<HandlerSlot<'a>>,
    // Player-mode belief overlay (see `NOTES.md`). `None` = spectator mode (engine identity,
    // fully observable — the belief is bypassed). `Some` = maintain per-player belief from the
    // reveal events emitted during effect resolution.
    belief: Option<BeliefTracker>,
    // Per-player trace of observable action *choices*, feeding the RL observation's History tokens
    // (`RL_ARCHITECTURE.md` §1.2.7). Like the belief overlay it is observer bookkeeping kept
    // outside `State`, and it is off by default.
    action_trace: Option<ActionTrace>,
    // The action currently being applied, recorded before `apply_action` so that a panic raised
    // inside effect resolution can still be attributed to what caused it. `Game` unwinds past
    // every local that knew, and a crash dump of a broken state that cannot name the action that
    // broke it is a puzzle rather than a report (`RL_ARCHITECTURE.md` §1.5.5).
    last_action: Option<Action>,
}

/// How a [`Game`] holds its event handler.
///
/// Borrowed is the batch-simulation shape: the caller keeps the handler, lends it for the game's
/// scope and reads it afterwards. Owned exists for the RL loop, whose envs are `'static` and
/// outlive any stack frame that could lend one — see [`crate::rl::env::Env`].
enum HandlerSlot<'a> {
    Borrowed(&'a mut CompositeSimulationEventHandler),
    Owned(Box<CompositeSimulationEventHandler>),
}

impl HandlerSlot<'_> {
    fn get(&mut self) -> &mut CompositeSimulationEventHandler {
        match self {
            HandlerSlot::Borrowed(handler) => handler,
            HandlerSlot::Owned(handler) => handler,
        }
    }
}

impl<'a> Game<'a> {
    pub fn from_state(state: State, players: Vec<Box<dyn Player>>, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Game {
            seed,
            rng,
            id: Uuid::new_v4(),
            players,
            state,
            debug: false,
            event_handler: None,
            belief: None,
            action_trace: None,
            last_action: None,
        }
    }

    pub fn new(players: Vec<Box<dyn Player>>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let deck_a = players[0].get_deck();
        let deck_b = players[1].get_deck();
        let state = State::initialize(&deck_a, &deck_b, &mut rng);
        Game {
            seed,
            rng,
            id: Uuid::new_v4(),
            players,
            state,
            debug: true,
            event_handler: None,
            belief: None,
            action_trace: None,
            last_action: None,
        }
    }

    pub fn new_with_event_handlers(
        game_id: Uuid,
        players: Vec<Box<dyn Player>>,
        seed: u64,
        event_handler: &'a mut CompositeSimulationEventHandler,
    ) -> Self {
        let mut game = Game::new(players, seed);
        game.event_handler = Some(HandlerSlot::Borrowed(event_handler));
        game.id = game_id;
        game
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    /// The seed this game was built from — with the two decks, the whole of what it takes to
    /// replay it (`RL_ARCHITECTURE.md` §1.5.5).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The last action handed to `apply_action`. After a panic it is the action that raised it,
    /// because it is recorded before the apply rather than after.
    pub fn last_action(&self) -> Option<&Action> {
        self.last_action.as_ref()
    }

    /// Installs a handler the game owns, for callers that cannot lend one.
    pub fn own_event_handler(&mut self, handler: CompositeSimulationEventHandler) {
        self.event_handler = Some(HandlerSlot::Owned(Box::new(handler)));
    }

    /// Takes back an owned handler. A borrowed one stays where it is — its owner already has it.
    pub fn take_event_handler(&mut self) -> Option<CompositeSimulationEventHandler> {
        match self.event_handler.take() {
            Some(HandlerSlot::Owned(handler)) => Some(*handler),
            other => {
                self.event_handler = other;
                None
            }
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    // Returns None if the game times out
    pub fn play(&mut self) -> Option<GameOutcome> {
        if self.debug {
            info!("Playing game with seed: {}", self.seed);
        }
        while !self.state.is_game_over() {
            self.play_tick();
        }
        self.state.winner
    }

    pub fn play_until_stable(&mut self) {
        while self.state.turn_count == 0 || !self.state.move_generation_stack.is_empty() {
            self.play_tick();
        }
    }

    pub fn play_tick(&mut self) -> Action {
        let (actor, actions) = self.state.generate_possible_actions();

        if self.debug {
            let color = self.get_color(actor);
            self.print_turn_header(actor, self.players[actor].as_ref(), &color);
        }
        let action = if actions.len() == 1 {
            debug!("Only one possible action, selecting it.");
            actions[0].clone()
        } else {
            let player = self.players[actor].as_mut();
            trace!(
                "Possible Actions: {:?}",
                actions.iter().map(|x| x.action.clone()).collect::<Vec<_>>()
            );
            player.decision_fn(&mut self.rng, &self.state, &actions)
        };

        self.resolve(actor, &actions, action)
    }

    /// Apply a decision that was made *outside* `play_tick` — by an RL policy driving the game
    /// through [`Self::get_decision_point`] rather than through a [`Player`] impl
    /// (`RL_ARCHITECTURE.md` §1.5.5: the learner batches inference across envs, so it cannot sit
    /// behind the blocking `decision_fn`).
    ///
    /// `actions` must be the enumeration `action` was chosen from: the event handler receives it,
    /// and the History trace needs its length to tell a genuine choice from a forced frame. This is
    /// the same path `play_tick` takes once its player has answered — the two must not diverge, or
    /// externally driven games would silently stop feeding the trace and the stats collector.
    pub fn resolve_decision(&mut self, actor: usize, actions: &[Action], action: Action) -> Action {
        self.resolve(actor, actions, action)
    }

    fn resolve(&mut self, actor: usize, actions: &[Action], action: Action) -> Action {
        // `get_color` reads the deck's declared energy and has a `todo!()` on Colorless, so it is
        // only ever reached when logging is actually on — the RL loop resolves thousands of these.
        if self.debug {
            let color = self.get_color(actor);
            self.print_action(&action, actor, self.players[actor].as_ref(), &color);
        }

        if let Some(slot) = &mut self.event_handler {
            slot.get()
                .on_action(self.id, &self.state, actor, actions, &action);
        }
        // Offer the resolved frame to the History trace *before* applying it, so board references
        // still resolve. The trace itself drops forced frames and engine-internal actions.
        if let Some(trace) = &mut self.action_trace {
            trace.record(&self.state, &action, actions.len());
        }
        self.last_action = Some(action.clone());
        self.apply_action(&action);
        self.print_state();
        action
    }

    pub fn get_state_clone(&self) -> State {
        self.state.clone()
    }

    /// Borrow the state. The RL env drives the game from outside and needs to enumerate the next
    /// frame before deciding who owns it — cloning the state for that would cost more than the
    /// decision itself (`State::clone` is ≈ 14 µs, see `NOTES.md`).
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Turn the per-action logging on or off. Off is the right setting for self-play: the log
    /// formatting reads the deck's declared energy, which is neither free nor total.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    // TODO: Maybe make these only available for testing?
    pub fn apply_action(&mut self, action: &Action) {
        apply_action(&mut self.rng, &mut self.state, action);
        // Drain the reveal events this action produced so the transient log can't grow, feeding
        // them to the belief overlay when in player mode (a no-op in spectator mode).
        let events = self.state.take_reveals();
        if let Some(belief) = &mut self.belief {
            belief.observe(&events);
        }
    }

    /// Switch this game into *player mode*: start maintaining the per-player belief overlay from
    /// reveal events. Spectator mode (the default) leaves it disabled and sees the full state.
    pub fn enable_belief(&mut self) {
        if self.belief.is_none() {
            self.belief = Some(BeliefTracker::new());
        }
    }

    /// The belief overlay, if player mode is enabled (`None` in spectator mode).
    pub fn belief(&self) -> Option<&BeliefTracker> {
        self.belief.as_ref()
    }

    /// Start recording the per-player action trace that feeds the RL observation's History tokens.
    /// Off by default — a game that never asks for an observation pays nothing for it.
    pub fn enable_action_trace(&mut self) {
        if self.action_trace.is_none() {
            self.action_trace = Some(ActionTrace::new());
        }
    }

    /// The action trace, if recording is enabled.
    pub fn action_trace(&self) -> Option<&ActionTrace> {
        self.action_trace.as_ref()
    }

    /// The RL observation of the current state, as seen by `perspective`
    /// (`RL_ARCHITECTURE.md` §1.2). The legality features are projected from the *same*
    /// `generate_possible_actions` enumeration the engine would offer right now, and the History
    /// bank is empty unless [`Self::enable_action_trace`] was called.
    pub fn get_observation(&self, perspective: usize) -> Observation {
        let (_, actions) = self.state.generate_possible_actions();
        rl::get_observation(
            &self.state,
            perspective,
            &actions,
            self.action_trace.as_ref(),
            self.belief.as_ref(),
        )
    }

    /// The current decision point as the agents consume it (`RL_ARCHITECTURE.md` §1.2 + §1.3):
    /// the frame's actor, their observation, and the action mask over the *same*
    /// `generate_possible_actions` enumeration — projected once, so the two cannot drift.
    ///
    /// The actor is not necessarily the turn player: a forced promotion after a KO or a Sabrina
    /// switch-in prompts the other player mid-turn (§1.3.6.1).
    pub fn get_decision_point(&self) -> (usize, Observation, ActionMask) {
        let (actor, actions) = self.state.generate_possible_actions();
        let observation = rl::get_observation(
            &self.state,
            actor,
            &actions,
            self.action_trace.as_ref(),
            self.belief.as_ref(),
        );
        let mask = rl::project_action_mask(&self.state, &actions, &observation);
        (actor, observation, mask)
    }

    /// The action mask of the current decision point, for the frame's actor. Prefer
    /// [`Self::get_decision_point`] when the observation is wanted too — this builds one anyway.
    pub fn get_action_mask(&self) -> ActionMask {
        self.get_decision_point().2
    }

    pub fn set_state(&mut self, state: State) {
        self.state = state;
    }

    fn print_turn_header(&self, actor: usize, player: &dyn Player, color: &str) {
        if self.debug {
            debug!(
                "{}{}",
                format!("===== {}|{:?}|", self.state.turn_count, self.state.points).color(color),
                format!("{actor}:{player:?}").color(color),
            );
        }
    }

    fn print_action(&self, action: &Action, _: usize, player: &dyn Player, color: &str) {
        if self.debug {
            info!(
                "{} chose {}",
                format!("{}:{:?}", self.state.turn_count, player).color(color),
                format!("{:?}", action.action).bold()
            );
        }
    }

    fn print_state(&self) {
        if self.debug {
            trace!("{}", self.state.debug_string());
        }
    }

    /// see https://github.com/colored-rs/colored?tab=readme-ov-file#colors
    fn get_color(&self, actor: usize) -> String {
        let energy = self.state.decks[actor].energy_types[0];
        let color = match energy {
            EnergyType::Fighting => "red",
            EnergyType::Fire => "red",
            EnergyType::Grass => "green",
            EnergyType::Lightning => "yellow",
            EnergyType::Psychic => "magenta",
            EnergyType::Water => "blue",
            EnergyType::Darkness => "bright_black",
            EnergyType::Metal => "bright_black",
            // The Energy Zone cannot generate these, so a deck built around one is
            // not playable. `Deck::is_valid` rejects them, so getting here means an
            // unvalidated deck reached `Game`.
            EnergyType::Colorless | EnergyType::Dragon => panic!(
                "Player {actor}'s deck declares a {} energy zone, which the Energy Zone \
                 cannot generate. Check Deck::is_valid before starting a Game.",
                energy.as_str()
            ),
        };
        color.to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        models::StatusCondition,
        players::{AttachAttackPlayer, EndTurnPlayer, Player},
        state::GameOutcome,
        test_support::load_test_decks,
        Game,
    };

    #[test]
    fn test_poison() {
        let (deck_a, deck_b) = load_test_decks();
        let player_a = Box::new(AttachAttackPlayer { deck: deck_a });
        let player_b = Box::new(EndTurnPlayer { deck: deck_b });
        let players: Vec<Box<dyn Player>> = vec![player_a, player_b];
        let mut game = Game::new(players, 3);

        // Play initial setup phase
        while game.get_state_clone().turn_count == 0 {
            game.play_tick();
        }

        // Manually poison the opponent's Koffing
        let mut state = game.get_state_clone();
        state.apply_status_condition(1, 0, StatusCondition::Poisoned);
        game.set_state(state);

        // The game starts with AA playing. After each turn 10 damage should be subtracted.
        // So ending 1 Koffing should have 60HP, 2 => 50HP, 3 => 40HP, 4 => 30HP, 5 => 20HP
        while game.get_state_clone().turn_count == 1 {
            game.play_tick();
        }
        // Koffing should have 60 HP starting turn 2
        assert_eq!(game.get_state_clone().get_remaining_hp(1, 0), 60);
        while game.get_state_clone().turn_count == 2 {
            game.play_tick();
        }
        // Koffing should have 50 HP starting turn 3
        assert_eq!(game.get_state_clone().get_remaining_hp(1, 0), 50);
        while game.get_state_clone().turn_count == 3 {
            game.play_tick();
        }

        // Now play the rest. AA should win b.c. ET has no bench pokemon
        let winner = game.play();
        assert_eq!(game.get_state_clone().turn_count, 5);
        assert_eq!(winner, Some(GameOutcome::Win(0)));
    }

    #[test]
    fn test_ko_by_posion() {
        let (deck_a, deck_b) = load_test_decks();
        let player_a = Box::new(EndTurnPlayer { deck: deck_a });
        let player_b = Box::new(AttachAttackPlayer { deck: deck_b });
        let players: Vec<Box<dyn Player>> = vec![player_a, player_b];
        let mut game = Game::new(players, 4); // EndTurnPlayer starts

        // Turn 1, EE ends. Turn 2, AA attaches and attacks. Exeggcute should have 30 HP.
        // Turn 3, ET ends. We artificially poision, so that after playing out turn 4
        // (AA attacks) Exeggcute has 10 HP and KO from poison.
        while game.state.turn_count < 4 {
            game.play_tick();
        }
        assert_eq!(game.get_state_clone().get_remaining_hp(0, 0), 30);

        // Artificially poison Exeggcute
        let mut state = game.get_state_clone();
        state.apply_status_condition(0, 0, StatusCondition::Poisoned);
        game.set_state(state);

        // Turn 45, AA attacks. After ending, AA should win since no bench.
        while game.state.turn_count == 4 {
            game.play_tick();
        }
        assert_eq!(game.get_state_clone().points[0], 0);
        assert_eq!(game.get_state_clone().points[1], 1);
        game.play();
        assert_eq!(game.get_state_clone().turn_count, 5);
    }

    // TODO: Look for a game that has bench, and pokemon can die from attack + poison
    //   to launche the complicated sequence of Poison K.O. then user having
    //   to select one pokemon to promote to active.

    // TODO: Multiple bench KO
}
