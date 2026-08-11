//! Self-play environment (v1) — `RL_ARCHITECTURE.md` §1.5.5.
//!
//! The layer that turns "observation + mask + model" into trajectories. Its one load-bearing
//! decision is that **the learner is not a [`Player`]**.
//!
//! `Player::decision_fn` is blocking and per-seat: behind it, one game means one forward at
//! batch 1. §1.4.3 measures a lone CUDA forward at ≈ 16 ms against ≈ 386 µs/sample saturated — a
//! factor ≈ 40. So control is inverted: [`Env::step`] advances a game until it reaches a decision
//! the learner owns, then *yields* it. [`VecEnv::poll`] does that across N games and hands back
//! one batch of pending decisions; the caller runs a single batched forward and submits the
//! answers. Nothing here knows about the model, the backend, or the loss.
//!
//! What the env resolves on its own, without ever asking:
//!
//! - **FORCED frames** (§1.3.6.3, `len(candidates) == 1`, `end_turn_pending`, `DrawCard`) — the
//!   engine auto-resolves them inside `play_tick`. They cost no forward and are not decisions.
//! - **Scripted seats** — a [`SeatPolicy::Scripted`] seat is an ordinary engine `Player` (the §1.5.2
//!   frozen panel: random, weighted-random, expectiminimax). It answers in-process, so batching it
//!   would buy nothing.
//!
//! What it yields is exactly the set §1.5.1 calls the trajectory: *the agent's own decision frames*,
//! **off-turn reactive frames included** (forced promotion after a KO, a Sabrina switch-in — §1.3.6.1
//! decouples decision points from turn ownership). The env carries no buffer and no reward shaping:
//! the terminal outcome is all it reports, and how that becomes a return is §1.5.1's business.

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use uuid::Uuid;

use crate::actions::Action;
use crate::players::Player;
use crate::simulation_event_handler::{CompositeSimulationEventHandler, SimulationEventHandler};
use crate::state::GameOutcome;
use crate::{Deck, Game, State};

use super::action_mask::{project as project_action_mask, ActionMask, Head, Regime};
use super::observation::{get_observation, Observation};
use super::recover::{catch, EnginePanic};

/// A run of frames this long inside one [`Env::step`] is a cycle, not a game — the engine's only
/// bound on an endless game is `turn_count > 99` in `State::advance_turn`, unreachable by a cycle
/// that never ends its turn, so neither [`catch`] nor the §1.5.5 crash budget sees a spin.
///
/// So the env makes it a panic, and the game takes the path every other broken game takes — dumped
/// with its seed and decks ([`crate::rl::train::crash`]), discarded, replaced. Counts every frame
/// `step` resolves without yielding, scripted-seat decisions included, since [`Env::run_scripted`]
/// legitimately plays a whole game in one call. Set at ≈ 6.8× the longest complete game measured,
/// deliberately lopsided (firing late costs microseconds, firing early costs a game silently) —
/// measurement and margin: NOTES.md, "Guards de l'environnement".
const FORCED_FRAME_LIMIT: usize = 4096;

/// How long one game may spend inside the engine before it is a hang, not a game.
///
/// [`FORCED_FRAME_LIMIT`] bounds how many frames a game resolves; this bounds how long they take —
/// the two miss opposite things, a single pathologically slow frame among them, and the run only
/// survives with both. Charged across the whole game rather than per [`Env::step`] call, because
/// the failure it has to catch is total: a game paying a second a frame is as dead as one paying an
/// hour once, and only the sum sees both. Incident and margin: NOTES.md, "Guards de
/// l'environnement".
///
/// It cannot interrupt a call in progress: the budget is read between frames and where `step`
/// returns, so a single engine call that never returns at all still hangs the run. Bounding *that*
/// would mean a watchdog thread and an engine that can be unwound from outside, which no part of
/// this design supports today.
const ENGINE_TIME_LIMIT: Duration = Duration::from_secs(10);

/// How many of the cycling actions the panic quotes.
///
/// The dump's `last_action` and terminal state say where the game stopped, not what repeats — and
/// the repeating tail is what names the card. Recording starts only once [`FORCED_FRAME_LIMIT`] is
/// past, so the frames a healthy game resolves pay one increment and no allocation.
const FORCED_FRAME_TRACE: usize = 32;

/// Which model answers a seat. The learner and the frozen checkpoints PFSP samples (§1.5.2) are all
/// "agents"; the id is what lets [`VecEnv::poll`] group pending decisions into one batch per model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AgentId(pub u16);

impl AgentId {
    /// The seat being trained. Conventional, not enforced — nothing here privileges it.
    pub const LEARNER: AgentId = AgentId(0);
}

/// How a seat answers its decision frames.
pub enum SeatPolicy {
    /// An engine `Player`, resolved in-process by `play_tick`.
    Scripted,
    /// A model outside the env; its frames are yielded as [`DecisionRequest`]s.
    Agent(AgentId),
}

impl SeatPolicy {
    fn agent(&self) -> Option<AgentId> {
        match self {
            SeatPolicy::Scripted => None,
            SeatPolicy::Agent(id) => Some(*id),
        }
    }
}

/// One decision point the env cannot resolve by itself.
///
/// `observation` and `mask` are the two sibling projections of the *same*
/// `generate_possible_actions` enumeration (§1.3.1), built once here so they cannot drift.
pub struct DecisionRequest {
    /// The model expected to answer.
    pub agent: AgentId,
    /// `frame.actor` — the seat this frame belongs to, which is **not** necessarily the turn
    /// player (§1.3.6.1). The observation is egocentric to it.
    pub actor: usize,
    pub regime: Regime,
    pub observation: Observation,
    pub mask: ActionMask,
}

/// What a finished game reports. Terminal reward only (§1.5.1) — the env forms no other signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvOutcome {
    /// `None` when the game hit the 99-turn horizon without a winner.
    pub winner: Option<GameOutcome>,
    /// Decision frames actually yielded, per seat. The denominator the §1.4.3 budget is read
    /// against, and §1.5.6's "decisions per game".
    pub decisions: [u32; 2],
    /// Game turns at the end. Kept next to `decisions` because the two answer different §1.5.6
    /// questions: decisions counts what the *model* was asked, turns counts how long the *game*
    /// ran, and a policy that stalls moves one without the other.
    pub turns: u8,
}

impl EnvOutcome {
    /// Win `+1` / loss `−1` / tie `0`, from `seat`'s point of view (§1.5.1, `γ = 1`).
    pub fn reward_for(&self, seat: usize) -> f32 {
        match self.winner {
            Some(GameOutcome::Win(winner)) if winner == seat => 1.0,
            Some(GameOutcome::Win(_)) => -1.0,
            Some(GameOutcome::Tie) | None => 0.0,
        }
    }
}

/// What one [`Env::step`] produced.
///
/// The variants are lopsided — a request carries a whole observation, an outcome is a handful of
/// bytes — but boxing the big one would trade a struct move for a heap allocation on the one path
/// that runs thousands of times a second. The observation's own payload is already behind its
/// `Vec`s; only the header moves.
#[allow(clippy::large_enum_variant)]
pub enum EnvStep {
    /// The env is blocked on a model. Answer with [`Env::submit`].
    Pending(DecisionRequest),
    /// The game is over; this env yields nothing more until it is replaced.
    Done(EnvOutcome),
}

/// One self-play game, driven from outside.
pub struct Env<'a> {
    game: Game<'a>,
    seats: [SeatPolicy; 2],
    decisions: [u32; 2],
    /// The outstanding request's `(actor, enumeration, mask)`.
    ///
    /// All three are kept rather than recomputed on submission. `resolve_decision` needs the exact
    /// enumeration the choice was made against (the event handler and the History trace both read
    /// it), and the mask is what maps the chosen bit back to an action — rebuilding it would mean
    /// rebuilding the observation too, at ≈ 130 µs a decision for an answer already in hand.
    pending: Option<(usize, Vec<Action>, ActionMask)>,
    /// [`FORCED_FRAME_LIMIT`], as a field so a test can lower it to a length a healthy game reaches
    /// — the guard's own path is then exercised on a real game rather than on a hand-built cycle
    /// nobody has an example of yet.
    forced_frame_limit: usize,
    /// [`ENGINE_TIME_LIMIT`] and what this game has spent against it, as fields for the same reason
    /// `forced_frame_limit` is one.
    engine_time_limit: Duration,
    engine_time_spent: Duration,
}

impl<'a> Env<'a> {
    /// Wrap a game. `players` is the engine's own seat vector: a [`SeatPolicy::Scripted`] seat uses
    /// its `Player` normally, an [`SeatPolicy::Agent`] seat only ever uses its *deck*.
    ///
    /// The placeholder `Player` an agent seat carries is never consulted: `step` hands a frame to
    /// `play_tick` only when it is FORCED (which `play_tick` resolves from the candidate list
    /// itself, without calling `decision_fn`) or when the seat is scripted. `never_consults_the_
    /// placeholder_player_of_an_agent_seat` pins that down.
    pub fn new(mut game: Game<'a>, seats: [SeatPolicy; 2]) -> Self {
        game.set_debug(false);
        // The History bank (§1.2.7) is empty unless the trace is recording — an env that forgets
        // this trains on a blind observation.
        game.enable_action_trace();
        // Player mode: without it the state stays the spectator view, no reveal ever reaches the
        // observation, and `REVEALED_HAND_PTR` points into a set the encoder cannot see (§1.3.6.2).
        game.enable_belief();
        Env {
            game,
            seats,
            decisions: [0, 0],
            pending: None,
            forced_frame_limit: FORCED_FRAME_LIMIT,
            engine_time_limit: ENGINE_TIME_LIMIT,
            engine_time_spent: Duration::ZERO,
        }
    }

    #[cfg(test)]
    fn set_forced_frame_limit(&mut self, limit: usize) {
        self.forced_frame_limit = limit;
    }

    #[cfg(test)]
    fn set_engine_time_limit(&mut self, limit: Duration) {
        self.engine_time_limit = limit;
    }

    #[cfg(test)]
    fn engine_time_spent(&self) -> Duration {
        self.engine_time_spent
    }

    /// Adds `elapsed` to this game's engine time and panics past [`ENGINE_TIME_LIMIT`], so a game
    /// that stopped advancing takes the same dumped-and-replaced path a panicking one takes.
    fn charge_engine_time(&mut self, elapsed: Duration) {
        self.engine_time_spent += elapsed;
        if self.engine_time_spent <= self.engine_time_limit {
            return;
        }
        let state = self.game.state();
        panic!(
            "engine time budget: {:.1?} spent on one game (turn {}, current player {}, stack depth \
             {}), last action {:?}",
            self.engine_time_spent,
            state.turn_count,
            state.current_player,
            state.move_generation_stack.len(),
            self.game.last_action(),
        );
    }

    /// Build a game from two decks and a seed, with `players` supplying both the decks and the
    /// scripted seats' behaviour.
    pub fn from_players(players: Vec<Box<dyn Player>>, seats: [SeatPolicy; 2], seed: u64) -> Self {
        Env::new(Game::new(players, seed), seats)
    }

    /// Turns the engine's per-frame log back on.
    ///
    /// Off by default here because a training run resolves millions of frames and `get_color` has a
    /// `todo!()` on Colorless decks, so logging is a cost *and* a hazard on that path. The CLI
    /// runner is the opposite case: a handful of games a person is watching (`-vv` and up).
    pub fn set_debug(&mut self, debug: bool) {
        self.game.set_debug(debug);
    }

    /// Attaches the §1.5.7 harvest to this game, and opens it.
    ///
    /// Owned rather than lent because the collector's envs are `'static` and outlive any frame
    /// that could lend a handler. The pair to this is [`Env::close_handler`], which the env calls
    /// itself: `on_game_end` needs the terminal state, and the caller sees the env only after it
    /// has moved past it.
    pub fn open_handler(&mut self, handler: CompositeSimulationEventHandler) {
        let mut handler = handler;
        handler.on_game_start(self.game.id());
        self.game.own_event_handler(handler);
    }

    /// Closes and returns the handler installed by [`Env::open_handler`], if any.
    pub fn close_handler(&mut self) -> Option<CompositeSimulationEventHandler> {
        let mut handler = self.game.take_event_handler()?;
        handler.on_game_end(
            self.game.id(),
            self.game.get_state_clone(),
            self.game.state().winner,
        );
        Some(handler)
    }

    /// The decks in play, in seat order — what the §1.5.7 harvest keys its labels on.
    pub fn decks(&self) -> [&Deck; 2] {
        let state = self.game.state();
        [&state.decks[0], &state.decks[1]]
    }

    /// Which agent, if any, owns `seat`.
    pub fn agent_of(&self, seat: usize) -> Option<AgentId> {
        self.seats[seat].agent()
    }

    /// The game state. Read-only, and the only reason it is exposed: a crashed env is dumped
    /// before it is thrown away (§1.5.5, [`crate::rl::train::crash`]).
    pub fn state(&self) -> &State {
        self.game.state()
    }

    /// The seed the game was built from — half of what makes a crashed game reproducible, the
    /// other half being [`Env::decks`].
    pub fn seed(&self) -> u64 {
        self.game.seed()
    }

    pub fn game_id(&self) -> Uuid {
        self.game.id()
    }

    /// The action being applied when the game last did anything. After a panic, the one that
    /// raised it.
    pub fn last_action(&self) -> Option<&Action> {
        self.game.last_action()
    }

    /// The player whose turn it is. Not the same as the actor of the next frame: a reactive frame
    /// prompts the *other* player mid-turn (§1.3.6.1).
    pub fn turn_player(&self) -> usize {
        self.game.state().current_player
    }

    /// Advance until a model is needed or the game ends.
    ///
    /// Every frame the env can resolve — forced, scripted, engine-internal — is consumed here, so
    /// the caller only ever sees genuine learned decisions. Calling `step` while a request is
    /// outstanding re-derives an identical one (nothing has been applied to advance the game), so
    /// it is idempotent — but it pays for the observation twice, so callers should submit first.
    ///
    /// Panics past [`FORCED_FRAME_LIMIT`] frames resolved without yielding, and past
    /// [`ENGINE_TIME_LIMIT`] of engine time on this game — see there for why that is this layer's
    /// job and not the engine's.
    pub fn step(&mut self) -> EnvStep {
        let mut resolved = 0usize;
        let mut cycle: Vec<String> = Vec::new();
        // Restarted after every charge, so each stretch of engine work is counted once.
        let mut since = Instant::now();

        loop {
            if self.game.is_game_over() {
                self.charge_engine_time(since.elapsed());
                return EnvStep::Done(EnvOutcome {
                    winner: self.game.state().winner,
                    decisions: self.decisions,
                    turns: self.game.state().turn_count,
                });
            }

            let (actor, actions) = self.game.state().generate_possible_actions();
            let regime = Regime::of(self.game.state(), &actions);

            // FORCED wins over everything (§1.3.2): a one-candidate frame is auto-resolved even on
            // an agent seat, and `play_tick` does exactly that without touching `decision_fn`.
            let Some(agent) = self.seats[actor].agent().filter(|_| regime.needs_policy()) else {
                let applied = self.game.play_tick();
                self.charge_engine_time(since.elapsed());
                since = Instant::now();
                resolved += 1;
                if resolved > self.forced_frame_limit {
                    cycle.push(format!("{}:{:?}", applied.actor, applied.action));
                    if cycle.len() >= FORCED_FRAME_TRACE {
                        let state = self.game.state();
                        panic!(
                            "frame cycle: {resolved} frames resolved with no decision and no \
                             winner, on turn {} (current player {}, stack depth {}); repeating \
                             tail: {}",
                            state.turn_count,
                            state.current_player,
                            state.move_generation_stack.len(),
                            cycle.join(" -> "),
                        );
                    }
                }
                continue;
            };

            // Both projections come off the enumeration already in hand — `get_decision_point`
            // would re-run `generate_possible_actions` for the same answer.
            let observation = get_observation(
                self.game.state(),
                actor,
                &actions,
                self.game.action_trace(),
                self.game.belief(),
            );
            let mask = project_action_mask(self.game.state(), &actions, &observation);
            // Charged before the frame is handed out, so a §1.2.5 threat matrix that takes minutes
            // to build is caught on the game that built it rather than on whatever it yields to.
            self.charge_engine_time(since.elapsed());
            self.pending = Some((actor, actions, mask.clone()));
            return EnvStep::Pending(DecisionRequest {
                agent,
                actor,
                regime,
                observation,
                mask,
            });
        }
    }

    /// Answer the outstanding request with a chosen mask bit, and apply it.
    ///
    /// `(head, index)` must be a bit the request's mask actually set; the round-trip back to a
    /// `SimpleAction` the engine accepts is §1.3.7 invariant 3, and a bit that does not resolve is a
    /// masking bug, not a recoverable input — hence [`SubmitError`] rather than a silent fallback.
    pub fn submit(&mut self, head: Head, index: usize) -> Result<(), SubmitError> {
        let Some((actor, actions, mask)) = self.pending.take() else {
            return Err(SubmitError::NothingPending);
        };
        let Some(action) = mask.select(head, index) else {
            return Err(SubmitError::IllegalBit { head, index });
        };
        let applying = Instant::now();
        self.game.resolve_decision(actor, &actions, action);
        // Applying the effect is engine work like any other frame's, and §1.5.5 already treats this
        // as the likeliest place for one to break.
        self.charge_engine_time(applying.elapsed());
        self.decisions[actor] += 1;
        Ok(())
    }

    /// Play the game out with no model in the loop. Only valid when both seats are scripted —
    /// the calibration path the §1.5.3 deck sampler and the §1.5.7 harvest use.
    pub fn run_scripted(&mut self) -> EnvOutcome {
        // `step` consumes every scripted and forced frame itself, so with no agent seat it can only
        // come back `Done` — one call plays the whole game.
        match self.step() {
            EnvStep::Done(outcome) => outcome,
            EnvStep::Pending(request) => panic!(
                "run_scripted on an env with an agent seat ({:?} owns seat {})",
                request.agent, request.actor
            ),
        }
    }
}

/// Why a submission was rejected. Both variants are contract violations by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// No request is outstanding — `step` was never called, or was answered already.
    NothingPending,
    /// The chosen bit is not set in the mask.
    IllegalBit { head: Head, index: usize },
}

/// A pending decision, tagged with the env it came from.
pub struct Pending {
    pub env: usize,
    pub request: DecisionRequest,
}

/// A finished game, tagged with the env it came from.
pub struct Finished {
    pub env: usize,
    pub outcome: EnvOutcome,
}

/// A game the engine could not carry on with — an `expect` in the simulator, caught rather than
/// fatal (§1.5.5, [`super::recover`]).
///
/// The env is **left in its slot**, still holding the state that panicked, so the caller can dump
/// it before calling [`VecEnv::replace`]. It will not progress again: every later `poll` panics in
/// the same place.
pub struct Crashed {
    pub env: usize,
    pub panic: EnginePanic,
}

/// Why a [`VecEnv::submit`] did not go through.
///
/// The two are not the same kind of event and must not be handled the same way. `Rejected` is a
/// **contract violation by the caller** — the bit was never legal — and stays fatal; `Panicked` is
/// the engine giving up on a legal action, which costs one game (§1.5.5).
#[derive(Debug)]
pub enum SubmitFault {
    Rejected(SubmitError),
    Panicked(EnginePanic),
}

/// N envs advanced together, so one batched forward serves all of them (§1.5.5).
///
/// It deliberately does **not** reset finished envs: what deck the next game draws is the §1.5.3
/// sampler's decision (meta / tutorial DB, uniform quota, mirror quota), not this layer's. Finished
/// envs are reported and left idle until [`VecEnv::replace`].
pub struct VecEnv<'a> {
    envs: Vec<Option<Env<'a>>>,
}

impl<'a> VecEnv<'a> {
    pub fn new(envs: Vec<Env<'a>>) -> Self {
        VecEnv {
            envs: envs.into_iter().map(Some).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Advance every env to its next learned decision.
    ///
    /// Returns the pending decisions — group them by [`DecisionRequest::agent`], one batched
    /// forward per group — the games that ended on the way, and the ones the engine panicked on.
    /// An env that is already waiting re-reports its outstanding request, so `poll` is idempotent
    /// between submissions.
    ///
    /// The panic guard is **per env**: one broken game is one lost game, not a lost batch, and the
    /// envs that polled cleanly before and after it keep their pending decisions.
    pub fn poll(&mut self) -> (Vec<Pending>, Vec<Finished>, Vec<Crashed>) {
        let mut pending = Vec::new();
        let mut finished = Vec::new();
        let mut crashed = Vec::new();
        for (index, slot) in self.envs.iter_mut().enumerate() {
            let Some(env) = slot.as_mut() else {
                continue;
            };
            match catch(|| env.step()) {
                Ok(EnvStep::Pending(request)) => pending.push(Pending {
                    env: index,
                    request,
                }),
                Ok(EnvStep::Done(outcome)) => finished.push(Finished {
                    env: index,
                    outcome,
                }),
                Err(panic) => crashed.push(Crashed { env: index, panic }),
            }
        }
        (pending, finished, crashed)
    }

    /// Answer one env's outstanding request.
    ///
    /// Guarded like [`VecEnv::poll`], and for the same reason — applying the chosen action is
    /// where an engine invariant is most likely to break, since that is where the effect actually
    /// resolves.
    pub fn submit(&mut self, env: usize, head: Head, index: usize) -> Result<(), SubmitFault> {
        match self.envs[env].as_mut() {
            Some(env) => catch(|| env.submit(head, index))
                .map_err(SubmitFault::Panicked)?
                .map_err(SubmitFault::Rejected),
            None => Err(SubmitFault::Rejected(SubmitError::NothingPending)),
        }
    }

    /// Mutable access to one env — what [`Env::close_handler`] needs on a finished slot, before
    /// [`VecEnv::replace`] drops the game the handler was watching.
    pub fn get_mut(&mut self, env: usize) -> Option<&mut Env<'a>> {
        self.envs.get_mut(env)?.as_mut()
    }

    pub fn get(&self, env: usize) -> Option<&Env<'a>> {
        self.envs[env].as_ref()
    }

    /// Put a fresh game in a finished slot. The caller owns deck sampling (§1.5.3).
    pub fn replace(&mut self, index: usize, env: Env<'a>) {
        self.envs[index] = Some(env);
    }

    /// Drop a finished env without replacing it.
    pub fn clear(&mut self, index: usize) {
        self.envs[index] = None;
    }
}

/// Per-env seed from a master seed (§1.5.5: "master seed → per-env child seeds, fully
/// reproducible"). SplitMix64 — the standard finalizer, chosen because it is stateless: the seed of
/// env `i` of run `s` is a pure function of `(s, i)`, so a single env can be replayed in isolation
/// without stepping a shared generator up to its position.
pub fn split_seed(master: u64, index: u64) -> u64 {
    let mut z = master.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// An RNG seeded from [`split_seed`], for the caller's own sampling (deck draws, action sampling).
pub fn env_rng(master: u64, index: u64) -> StdRng {
    StdRng::seed_from_u64(split_seed(master, index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::{create_players, PlayerCode};
    use rand::Rng;

    fn decks() -> (Deck, Deck) {
        (
            Deck::from_file("example_decks/venusaur-exeggutor.txt").expect("deck a"),
            Deck::from_file("example_decks/weezing-arbok.txt").expect("deck b"),
        )
    }

    fn scripted_env<'a>(seed: u64) -> Env<'a> {
        let (deck_a, deck_b) = decks();
        let players = create_players(deck_a, deck_b, vec![PlayerCode::R, PlayerCode::R]);
        Env::from_players(players, [SeatPolicy::Scripted, SeatPolicy::Scripted], seed)
    }

    /// Seat 0 is agent-driven: every frame it owns must come back as a request, and the placeholder
    /// `Player` it was built with must never be reached. `EndTurnPlayer` is that placeholder — if it
    /// were ever consulted the game would end turns immediately and seat 0 would make almost no
    /// decisions, so the assertion below is what detects it.
    fn agent_env<'a>(seed: u64) -> Env<'a> {
        let (deck_a, deck_b) = decks();
        let players = create_players(deck_a, deck_b, vec![PlayerCode::ET, PlayerCode::R]);
        Env::from_players(
            players,
            [SeatPolicy::Agent(AgentId::LEARNER), SeatPolicy::Scripted],
            seed,
        )
    }

    /// A seeded uniform pick over the set bits — the stand-in policy these tests drive the agent
    /// seat with. Always taking `entries[0]` instead would be a degenerate policy, not a cheap one:
    /// it follows the engine's enumeration order, never develops a bench, and stops the game from
    /// reaching whole classes of frame (measured: it produces *zero* reactive frames over 30 games,
    /// where a uniform pick produces ~70).
    fn random_bit(request: &DecisionRequest, rng: &mut StdRng) -> (Head, usize) {
        let entry = &request.mask.entries[rng.gen_range(0..request.mask.entries.len())];
        (entry.head, entry.index)
    }

    /// The env's driving loop must reproduce `Game::play` exactly when nothing is agent-driven:
    /// same seed, same decks, same outcome. This is what says the inversion of control changed the
    /// *caller*, not the game.
    #[test]
    fn scripted_env_matches_the_engines_own_loop() {
        for seed in 0..20u64 {
            let (deck_a, deck_b) = decks();
            let players = create_players(deck_a, deck_b, vec![PlayerCode::R, PlayerCode::R]);
            let mut reference = Game::new(players, seed);
            reference.set_debug(false);
            let expected = reference.play();

            let outcome = scripted_env(seed).run_scripted();
            assert_eq!(outcome.winner, expected, "seed {seed}");
        }
    }

    /// The env only ever surfaces frames that need a policy: never a FORCED one, always for the
    /// seat the request names, and always with a mask that offers a real choice (§1.3.7 invariant
    /// 2).
    ///
    /// Two or more entries, not merely non-empty: `Regime::Forced` is decided on the candidate
    /// count and the projection is a bijection onto the candidates (§1.3.7 invariant 1), so a
    /// yielded frame carrying one bit would mean one of the two had broken. §1.5.6 relies on this
    /// — it is why the forced rate is logged per head and not per frame, a per-frame series being
    /// a flat zero that reads as a measurement.
    #[test]
    fn only_genuine_decisions_of_the_agent_seat_are_yielded() {
        let mut env = agent_env(7);
        let mut rng = env_rng(7, 0);
        let mut yielded = 0;
        loop {
            match env.step() {
                EnvStep::Done(outcome) => {
                    assert_eq!(
                        outcome.decisions[0], yielded,
                        "every yielded request was submitted"
                    );
                    assert_eq!(outcome.decisions[1], 0, "the scripted seat yields nothing");
                    assert!(yielded > 0, "the agent seat never got to decide");
                    break;
                }
                EnvStep::Pending(request) => {
                    assert_eq!(request.actor, 0, "a request for a seat we do not own");
                    assert_eq!(request.agent, AgentId::LEARNER);
                    assert_ne!(
                        request.regime,
                        Regime::Forced,
                        "forced frames are not choices"
                    );
                    assert!(
                        request.mask.entries.len() >= 2,
                        "a yielded frame with {} bit(s) — a forced frame reached the learner",
                        request.mask.entries.len()
                    );
                    let (head, index) = random_bit(&request, &mut rng);
                    yielded += 1;
                    env.submit(head, index).expect("a set bit is legal");
                }
            }
        }
    }

    /// An agent seat's placeholder `Player` is unreachable. `EndTurnPlayer` sits on seat 0; if
    /// `play_tick` ever consulted it, seat 0 would pass every turn and lose nearly always. Playing
    /// the first legal bit instead is not a good policy either, but it is not *that* policy.
    #[test]
    fn never_consults_the_placeholder_player_of_an_agent_seat() {
        let mut turns = 0;
        for seed in 0..10u64 {
            let mut env = agent_env(seed);
            let mut rng = env_rng(seed, 0);
            loop {
                match env.step() {
                    EnvStep::Done(_) => break,
                    EnvStep::Pending(request) => {
                        let (head, index) = random_bit(&request, &mut rng);
                        env.submit(head, index).expect("legal bit");
                        turns += 1;
                    }
                }
            }
        }
        assert!(
            turns > 100,
            "seat 0 made only {turns} decisions over 10 games — the placeholder is being consulted"
        );
    }

    /// Off-turn reactive frames belong to the trajectory (§1.5.1, §1.3.6.1): the env must surface
    /// frames where the actor is not the turn player, or a multi-candidate promotion after a KO —
    /// and every Sabrina-style switch-in — would never be trained on.
    #[test]
    fn yields_frames_the_agent_owns_off_its_own_turn() {
        let mut off_turn = 0;
        let mut total = 0;
        for seed in 0..30u64 {
            let mut env = agent_env(seed);
            let mut rng = env_rng(seed, 0);
            loop {
                match env.step() {
                    EnvStep::Done(_) => break,
                    EnvStep::Pending(request) => {
                        total += 1;
                        if request.actor != env.turn_player() {
                            off_turn += 1;
                        }
                        let (head, index) = random_bit(&request, &mut rng);
                        env.submit(head, index).expect("legal bit");
                    }
                }
            }
        }
        assert!(
            off_turn > 0,
            "no reactive frame over 30 games ({total} decisions)"
        );
    }

    /// Submitting a bit the mask does not set is a contract violation, reported rather than
    /// silently repaired — a mask that disagrees with the engine is exactly the bug §1.3.7 exists
    /// to catch.
    #[test]
    fn rejects_a_bit_the_mask_does_not_set() {
        let mut env = agent_env(3);
        assert_eq!(
            env.submit(Head::EndTurn, 0),
            Err(SubmitError::NothingPending),
            "nothing has been yielded yet"
        );
        let EnvStep::Pending(request) = env.step() else {
            panic!("expected a decision");
        };
        let unset = (0..Head::CandidatePtr.dim())
            .find(|index| !request.mask.is_set(Head::CandidatePtr, *index))
            .expect("some candidate bit is unset");
        assert_eq!(
            env.submit(Head::CandidatePtr, unset),
            Err(SubmitError::IllegalBit {
                head: Head::CandidatePtr,
                index: unset
            })
        );
    }

    /// `poll` batches across envs and reports terminations without resetting them — deck sampling
    /// is §1.5.3's call, not this layer's.
    #[test]
    fn vec_env_polls_every_env_and_reports_terminations() {
        let mut vec_env = VecEnv::new((0..8u64).map(agent_env).collect());
        let mut rng = env_rng(0xB47C, 0);
        let mut done = 0;
        let mut steps = 0;
        while done < 8 {
            let (pending, finished, crashed) = vec_env.poll();
            assert!(crashed.is_empty(), "ordinary play must not panic");
            done += finished.len();
            for slot in finished {
                vec_env.clear(slot.env);
            }
            if pending.is_empty() {
                continue;
            }
            // What a real caller does with this batch: one forward per agent group. Here, the
            // first legal bit of each.
            let choices: Vec<(usize, Head, usize)> = pending
                .iter()
                .map(|slot| {
                    let (head, index) = random_bit(&slot.request, &mut rng);
                    (slot.env, head, index)
                })
                .collect();
            for (env, head, index) in choices {
                vec_env.submit(env, head, index).expect("legal bit");
            }
            steps += 1;
            assert!(steps < 100_000, "envs are not progressing");
        }
    }

    /// A seat that plays normally for a while and then panics, standing in for the engine
    /// invariants that fail deep inside `apply_action` (`Active Pokemon should be there`, and its
    /// siblings). What is under test is the **unwinding path** — `Env::step` calls `play_tick`,
    /// and by the time [`catch`] sees the panic, where in the call stack it was raised is not
    /// something the recovery can or should distinguish. Injected rather than provoked because a
    /// state that actually breaks the engine is a *bug*, with its own fix and its own lifetime;
    /// pinning these tests to one would make them expire the day it is fixed.
    #[derive(Debug)]
    struct PanicsAfter {
        inner: crate::players::RandomPlayer,
        remaining: usize,
    }

    impl Player for PanicsAfter {
        fn get_deck(&self) -> Deck {
            self.inner.get_deck()
        }

        fn decision_fn(
            &mut self,
            rng: &mut StdRng,
            state: &State,
            possible_actions: &[Action],
        ) -> Action {
            if self.remaining == 0 {
                panic!("Active Pokemon should be there");
            }
            self.remaining -= 1;
            self.inner.decision_fn(rng, state, possible_actions)
        }
    }

    fn breaking_env<'a>(seed: u64, after: usize) -> Env<'a> {
        let (deck_a, deck_b) = decks();
        let players: Vec<Box<dyn Player>> = vec![
            Box::new(PanicsAfter {
                inner: crate::players::RandomPlayer { deck: deck_a },
                remaining: after,
            }),
            Box::new(crate::players::RandomPlayer { deck: deck_b }),
        ];
        Env::from_players(players, [SeatPolicy::Scripted, SeatPolicy::Scripted], seed)
    }

    /// A game the engine cannot continue is reported, not fatal — and it comes back with the
    /// message, so a run's crash dump can say what broke rather than that something did.
    #[test]
    fn an_engine_panic_is_caught_and_reported_against_its_env() {
        let mut vec_env = VecEnv::new(vec![breaking_env(5, 3), agent_env(6)]);

        let (pending, finished, crashed) = vec_env.poll();

        assert_eq!(crashed.len(), 1, "exactly one env was broken");
        assert_eq!(crashed[0].env, 0);
        assert!(
            crashed[0].panic.message.contains("Active Pokemon"),
            "unexpected panic: {}",
            crashed[0].panic.message
        );
        assert!(finished.is_empty());
        // The whole point of guarding per env rather than per poll: the healthy env still
        // produced its decision.
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].env, 1);
    }

    /// What recovery means concretely: the slot takes a fresh game and the batch carries on. The
    /// crashed env is left in place until then, which is what lets the caller dump it.
    #[test]
    fn replacing_a_crashed_env_resumes_the_batch() {
        let mut vec_env = VecEnv::new(vec![breaking_env(8, 3)]);

        let (_, _, crashed) = vec_env.poll();
        assert_eq!(crashed.len(), 1);
        // Still there, still broken — a second poll reports it again rather than skipping it.
        assert_eq!(vec_env.poll().2.len(), 1);
        assert!(
            vec_env.get(0).is_some(),
            "the crashed env is kept for the dump"
        );

        vec_env.replace(0, agent_env(9));
        let (pending, _, crashed) = vec_env.poll();
        assert!(crashed.is_empty());
        assert_eq!(pending.len(), 1, "the fresh game decides normally");
    }

    /// A crashed env still answers the forensic questions: which action was being applied, and
    /// from what `(seed, decks)` the game can be replayed. Without those a dump is a broken state
    /// with no way back to it.
    #[test]
    fn a_crashed_env_still_names_the_action_and_the_seed() {
        let mut env = breaking_env(12, 6);
        let panicked = catch(|| env.step()).is_err();

        assert!(panicked);
        assert_eq!(env.seed(), 12);
        assert!(
            env.last_action().is_some(),
            "the game applied actions before it broke, and the last of them is the lead"
        );
        assert!(env.state().turn_count > 0, "the state is still readable");
    }

    /// Reproducibility (§1.5.5): child seeds are a pure function of `(master, index)`, distinct
    /// across envs, and a run's envs are re-derivable one by one.
    #[test]
    fn child_seeds_are_pure_and_distinct() {
        let seeds: Vec<u64> = (0..64).map(|i| split_seed(0xDECC_6414, i)).collect();
        let again: Vec<u64> = (0..64).map(|i| split_seed(0xDECC_6414, i)).collect();
        assert_eq!(seeds, again, "child seeds are not reproducible");
        let mut sorted = seeds.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seeds.len(), "child seeds collide");
        assert_ne!(
            split_seed(1, 0),
            split_seed(2, 0),
            "runs share their first env's seed"
        );
    }

    /// Longest complete game over `seeds` seeds of every example-deck pairing, and the pairing that
    /// produced it. A fully scripted env resolves a whole game in one `step`, so this *is* the worst
    /// case [`FORCED_FRAME_LIMIT`] has to clear.
    fn longest_legal_game(seeds: u64) -> (usize, String) {
        let mut paths: Vec<String> = std::fs::read_dir("example_decks")
            .expect("example_decks")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "txt"))
            .map(|path| path.display().to_string())
            .collect();
        paths.sort();

        let mut worst = 0usize;
        let mut worst_label = String::new();
        for (i, a) in paths.iter().enumerate() {
            for b in &paths[i..] {
                for seed in 0..seeds {
                    // A deck the loader rejects is not this test's subject — it is never a deck a
                    // run plays, so it cannot be the one that trips the guard.
                    let (Ok(deck_a), Ok(deck_b)) = (Deck::from_file(a), Deck::from_file(b)) else {
                        continue;
                    };
                    let players =
                        create_players(deck_a, deck_b, vec![PlayerCode::R, PlayerCode::R]);
                    let mut game = Game::new(players, seed);
                    game.set_debug(false);
                    let mut frames = 0usize;
                    while !game.is_game_over() {
                        game.play_tick();
                        frames += 1;
                        assert!(
                            frames <= FORCED_FRAME_LIMIT,
                            "{a} vs {b} seed {seed} is a legal game the guard would kill"
                        );
                    }
                    if frames > worst {
                        worst = frames;
                        worst_label = format!("{a} vs {b} seed {seed}");
                    }
                }
            }
        }
        (worst, worst_label)
    }

    /// The margin [`FORCED_FRAME_LIMIT`] is set on, kept visible.
    ///
    /// What matters is the maximum over decks, not the mean: one combo deck chaining further than
    /// the bound would lose games to a guard nobody would think to suspect. So if a new card makes
    /// legal chains materially longer, raise the bound rather than loosening the ratio.
    #[test]
    fn no_legal_game_comes_close_to_the_frame_limit() {
        let (worst, label) = longest_legal_game(1);
        assert!(worst > 0, "no deck pairing played");
        assert!(
            worst * 4 <= FORCED_FRAME_LIMIT,
            "the longest legal game ({worst} frames, {label}) is within 4x of the \
             {FORCED_FRAME_LIMIT}-frame guard — the margin has eroded"
        );
    }

    /// The sweep the 400-frame figure in [`FORCED_FRAME_LIMIT`]'s doc comes from. Same assertion as
    /// the cheap one, six times the seeds — too slow for the default suite in a debug build.
    #[test]
    #[ignore = "full deck sweep; run with --release -- --ignored --nocapture"]
    fn the_measured_worst_case_behind_the_frame_limit() {
        let (worst, label) = longest_legal_game(6);
        println!("longest legal game: {worst} frames ({label})");
        assert!(worst * 4 <= FORCED_FRAME_LIMIT);
    }

    /// The guard fires, and the panic it fires is the one the recovery path already knows how to
    /// carry: caught per env, reported, and carrying the frames that repeat.
    ///
    /// The limit is lowered instead of building a cycle because the cycle this exists for has not
    /// been identified yet — and the guard cannot tell a cycle from an over-long game anyway, which
    /// is exactly why [`no_legal_game_comes_close_to_the_frame_limit`] measures the gap between the
    /// two.
    #[test]
    fn a_runaway_game_panics_with_the_frames_that_repeat() {
        let mut env = scripted_env(3);
        env.set_forced_frame_limit(8);
        let mut vec_env = VecEnv::new(vec![env]);

        let (pending, finished, crashed) = vec_env.poll();

        assert!(pending.is_empty() && finished.is_empty());
        assert_eq!(crashed.len(), 1, "the runaway env was not caught");
        let message = &crashed[0].panic.message;
        assert!(
            message.contains("frame cycle"),
            "the panic does not name the guard: {message}"
        );
        assert_eq!(
            message.matches(" -> ").count(),
            FORCED_FRAME_TRACE - 1,
            "the panic must quote {FORCED_FRAME_TRACE} actions: {message}"
        );
    }

    /// The time budget takes the same path, so a game that stopped advancing is dumped and replaced
    /// rather than left spinning. A zero limit stands in for the slow game: what the guard has to
    /// prove is that it reaches the recovery path, and no board is known that is slow on purpose.
    #[test]
    fn a_game_over_its_time_budget_panics_into_the_recovery_path() {
        let mut env = scripted_env(3);
        env.set_engine_time_limit(Duration::ZERO);
        let mut vec_env = VecEnv::new(vec![env]);

        let (pending, finished, crashed) = vec_env.poll();

        assert!(pending.is_empty() && finished.is_empty());
        assert_eq!(crashed.len(), 1, "the over-budget env was not caught");
        let message = &crashed[0].panic.message;
        assert!(
            message.contains("engine time budget"),
            "the panic does not name the guard: {message}"
        );
    }

    /// The budget is the whole game's, not one `step` call's — a game paying a little too much on
    /// every frame has to trip it as surely as one paying everything at once.
    ///
    /// The limit is set to exactly what the first step spent, so the second one fires on the *sum*
    /// and on nothing else: a per-call budget would have reset and let it through. Deterministic
    /// where an absolute limit would be a race against whatever the machine is doing.
    #[test]
    fn the_time_budget_accumulates_across_steps() {
        let mut env = agent_env(3);
        let mut rng = env_rng(3, 0);

        let EnvStep::Pending(request) = env.step() else {
            panic!("expected a decision");
        };
        let (head, index) = random_bit(&request, &mut rng);
        env.submit(head, index).expect("legal bit");

        let spent = env.engine_time_spent();
        assert!(spent > Duration::ZERO, "the first step charged nothing");
        env.set_engine_time_limit(spent);

        let Err(panic) = catch(|| env.step()) else {
            panic!("the second step must exhaust the budget");
        };
        assert!(
            panic.message.contains("engine time budget"),
            "the panic does not name the guard: {}",
            panic.message
        );
    }
}
