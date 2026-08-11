//! Rollout collection — the data path of `RL_ARCHITECTURE.md` §1.5.1.
//!
//! Turns a model plus the §1.5.3 deck sampler into the trajectories §1.5.1 updates on. The
//! §1.5.5 inversion of control is what this module exists to exploit: [`VecEnv::poll`] hands back
//! every env's pending decision at once, one forward answers all of them, and the answers go back
//! through [`VecEnv::submit`].
//!
//! **Whole episodes only.** §1.5.1 has terminal reward and `γ = 1`, so a truncated trajectory
//! carries no return at all. [`Collector`] therefore keeps its envs *across*
//! calls: a game still running when the frame budget is met stays in flight, with its frames, and
//! is emitted by a later call. The budget is a floor on **finished** frames, never a truncation.
//!
//! **The learner sits at seat 0, always.** Going first is worth real winrate, but the seat is not
//! what decides it: `State::initialize` draws `current_player` from the game's own RNG.
//!
//! **The opponent seat is either scripted or a model** ([`super::opponent`]). A heuristic is
//! resolved in-process and costs no forward; a §1.5.2 pool member or baked model is answered by its
//! own frozen network, and `poll`'s decisions are grouped by [`AgentId`] so each network pays one
//! batched forward per poll rather than one per env.
//!
//! **Only the learner's frames are kept.** An opponent's decisions advance the game and reach
//! nothing else — not the on-policy batch, not GAE, and not §1.5.1b's reservoir, where they would
//! make the magnet an average of a frozen checkpoint's play recorded under the best-response's name.
//! Nothing downstream could detect that, so `an_opponent_seats_frames_never_reach_the_batch` pins it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;

use super::crash::{CrashBudget, CrashLog};
use super::harvest::Harvest;
use super::opponent::{Assignment, OpponentModels, OpponentSeat};
use super::rating::{OpponentId, LEARNER as LEARNER_PILOT};
use super::sampler::DeckSampler;
use crate::players::{create_players, PlayerCode};
use crate::rl::action_mask::{ActionMask, Head, MaskEntry, ACTION_MASK_DIM, HEADS};
use crate::rl::env::{env_rng, split_seed, AgentId, Env, SeatPolicy, SubmitFault, VecEnv};
use crate::rl::model::config::ModelConfig;
use crate::rl::model::input::{DecisionPoint, ModelInput};
use crate::rl::model::RlModel;
use crate::rl::observation::Observation;
use crate::rl::recover::EnginePanic;

/// One learner decision frame, kept in the form the update re-encodes from.
///
/// The observation and mask are stored rather than the model input they produce: the §1.5.1 step
/// is a *gradient* step, so the forward has to be replayed under autodiff whatever we keep, and
/// these are the compact end of the choice.
pub struct Frame {
    pub observation: Observation,
    pub mask: ActionMask,
    /// Index into the flat `ACTION_MASK_DIM` policy vector — what the update gathers its
    /// log-probability at.
    pub chosen_bit: usize,
    /// Behaviour log-probability of `chosen_bit`, at collection time.
    pub logprob: f32,
    /// The value head's estimate at this frame, the GAE baseline.
    pub value: f32,
}

/// One finished game, from the learner's side.
pub struct Episode {
    pub frames: Vec<Frame>,
    /// Terminal reward for the learner's seat: win `+1` / loss `−1` / tie `0` (§1.5.1).
    pub reward: f32,
    /// Game turns the episode lasted (§1.5.6). Not `frames.len()`: a turn can hold several
    /// decisions, and off-turn reactive frames belong to the opponent's.
    pub turns: u8,
    /// The §1.5.2 opponent this game was played against.
    ///
    /// Carried so the winrate can be reported *per* opponent ([`super::eval::PanelWindow`]) and so
    /// the result can be rated ([`super::rating::RatingTable::record`]). A mixed winrate over a
    /// panel is the average of matchups that move independently — an agent that beats the random
    /// player and stalls against the weighted one has a flat mixed curve, and the collector is the
    /// only place that still knows which game was which.
    ///
    /// An [`OpponentId`] rather than a `PlayerCode`, because a pool clone and a baked model are
    /// neither of them player codes and all three are rated, sampled and logged identically.
    pub opponent: OpponentId,
}

/// What one [`Collector::collect`] did, for the §1.5.6 standard log.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RolloutStats {
    pub games: usize,
    pub frames: usize,
    /// Batched forwards run **for the learner**. `frames / forwards` is the mean batch size, which
    /// is what says whether the §1.5.5 inversion is actually paying — it decays as envs finish out
    /// of step. Opponent forwards are counted separately so that ratio keeps meaning what it says.
    pub forwards: usize,
    /// Batched forwards run for model-driven opponent seats (§1.5.2). Zero for a run whose panel is
    /// entirely scripted, and the number that says what the pool costs: an env holds one pending
    /// decision at a time and the actor alternates, so a model on the other seat roughly doubles
    /// the forwards for the same games.
    pub opponent_forwards: usize,
    /// §1.5.6's per-head entropy, accumulated here rather than in the update because the policy
    /// row is already read back to the CPU at this point — measuring it in the learner would mean
    /// a per-head GPU readback per micro-batch for a diagnostic.
    pub head_entropy: HeadEntropy,
    /// Games this collection lost to an engine panic (§1.5.5). Reported rather than merely
    /// counted: a run that quietly drops games is a run whose winrate is conditioned on something
    /// nobody wrote down.
    pub crashes: Vec<CrashReport>,
}

/// One game dropped because the engine panicked mid-play.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrashReport {
    pub env: usize,
    /// The panic's own message — enough for the loop's stdout line to be worth reading without
    /// opening the dump.
    pub message: String,
    /// Where the state was written, or `None` past the run's dump cap.
    pub dump: Option<PathBuf>,
}

/// Entropy of the masked policy *restricted to each head*, summed over the frames where that head
/// had a choice to make — and, beside it, the count of frames where it had no choice.
///
/// Restricted, and only over frames with two or more legal bits: a head with one legal bit has
/// zero entropy by arithmetic, not by policy, and averaging those in would report a collapse that
/// is really the mask's doing. `forced` is what that restriction threw away, so §1.5.6's
/// `head_forced/*` and `head_entropy/*` are two readings of the same pool of frames rather than
/// two independent measurements that happen to be logged together.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadEntropy {
    pub sum: [f64; HEADS.len()],
    pub frames: [u64; HEADS.len()],
    /// Frames where the head carried exactly one legal bit. Per head, not per frame: the whole
    /// frame is never forced (§1.3.6.3 frames are resolved without a forward and never reach the
    /// learner), but a frame with eight legal bits spread over four heads leaves several of them
    /// with a single bit apiece.
    pub forced: [u64; HEADS.len()],
}

impl Default for HeadEntropy {
    fn default() -> Self {
        HeadEntropy {
            sum: [0.0; HEADS.len()],
            frames: [0; HEADS.len()],
            forced: [0; HEADS.len()],
        }
    }
}

impl HeadEntropy {
    /// Mean entropy of `head`, or `None` where the head never faced a real choice.
    pub fn mean(&self, head: Head) -> Option<f64> {
        let index = HEADS.iter().position(|candidate| *candidate == head)?;
        (self.frames[index] > 0).then(|| self.sum[index] / self.frames[index] as f64)
    }

    /// Share of the frames offering `head` at all on which it offered exactly one bit, or `None`
    /// where the head never appeared.
    ///
    /// The denominator is [`Self::mean`]'s own frames plus these, which is what makes the pair
    /// readable: entropy that falls while this climbs is the mask narrowing, not the policy
    /// collapsing.
    pub fn forced_rate(&self, head: Head) -> Option<f64> {
        let index = HEADS.iter().position(|candidate| *candidate == head)?;
        let offered = self.frames[index] + self.forced[index];
        (offered > 0).then(|| self.forced[index] as f64 / offered as f64)
    }

    /// Folds one frame's masked policy row in.
    fn observe(&mut self, mask: &ActionMask, probs: &[f32]) {
        for (index, head) in HEADS.iter().enumerate() {
            let bits: Vec<f64> = mask
                .entries
                .iter()
                .filter(|entry| entry.head == *head)
                .map(|entry| probs[entry.head.offset() + entry.index].max(0.0) as f64)
                .collect();
            if bits.len() < 2 {
                if bits.len() == 1 {
                    self.forced[index] += 1;
                }
                continue;
            }
            let total: f64 = bits.iter().sum();
            if total <= 0.0 {
                continue;
            }
            let entropy = -bits
                .iter()
                .map(|p| p / total)
                .filter(|p| *p > 0.0)
                .map(|p| p * p.ln())
                .sum::<f64>();
            self.sum[index] += entropy;
            self.frames[index] += 1;
        }
    }
}

pub struct RolloutConfig {
    pub envs: usize,
    /// The §1.5.2 frozen panel, drawn from uniformly per game.
    pub opponents: Vec<PlayerCode>,
    /// Engine panics tolerated within one [`Collector::collect`] before it gives up (see
    /// [`CrashBudget`]).
    pub max_crashes_per_batch: usize,
}

/// The seat the learner occupies, in every game. See the module docs: the seats are symmetric, so
/// this is a convention rather than a choice.
///
/// Shared with [`super::eval`] rather than restated there: the two read `reward_for` off the same
/// convention, and a second copy that drifted would silently flip the sign of every eval winrate.
pub const LEARNER_SEAT: usize = 0;

/// Stream tags, mixed into the master seed so the three consumers below draw from independent
/// streams. Sharing one generator would make every state a function of how many draws every other
/// consumer had made — which is exactly what a resume cannot reconstruct.
const STREAM_DRAW: u64 = 0x5052_4157_0000_0001;
const STREAM_ACTION: u64 = 0x4143_5449_0000_0002;

pub struct Collector {
    envs: VecEnv<'static>,
    /// Frames of the game currently in flight in each env slot.
    open: Vec<Vec<Frame>>,
    /// The opponent each slot's game is being played against, parallel to `open`.
    opponents: Vec<OpponentId>,
    sampler: DeckSampler,
    config: RolloutConfig,
    /// Deck and opponent draws. Reseeded per game from `started`, so its state *is* `started`.
    draw_rng: StdRng,
    /// Action sampling. Reseeded per [`Collector::collect`] from the batch index, so its state
    /// *is* that index.
    action_rng: StdRng,
    master_seed: u64,
    /// Games started, ever. Also the child-seed index, so a game is replayable from
    /// `(master_seed, game_index)` alone (§1.5.5).
    started: u64,
    /// The §1.5.7 harvest, when the run asks for one. Its sampling draw rides the deck stream, so
    /// which games are harvested is a function of `(master_seed, game_index)` like everything else
    /// about a game.
    harvest: Option<Harvest>,
    /// Where a crashed game's state is written (§1.5.5). Optional because the collector recovers
    /// with or without it — dumping is forensics, recovery is the loop staying alive.
    crashes: Option<CrashLog>,
    budget: CrashBudget,
    /// Who the opponent seat is, per env group (§1.5.2). Read at [`Collector::spawn`] and nowhere
    /// else, which is what makes replacing it mid-run safe.
    assignment: Assignment,
}

/// One decision, drawn but not yet submitted.
///
/// Materialized because every answer in a poll is drawn before any is submitted: a submit can panic
/// and recycle its env, and unwinding a half-consumed readback afterwards is not something the
/// collector should have to be able to do.
struct Answer {
    head: Head,
    index: usize,
    logprob: f32,
    value: f32,
    /// Whether this frame belongs to the network being trained. The one bit that decides if it
    /// reaches the on-policy batch at all.
    learner: bool,
}

impl Collector {
    pub fn new(
        sampler: DeckSampler,
        config: RolloutConfig,
        master_seed: u64,
        harvest: Option<Harvest>,
    ) -> Result<Self, String> {
        if config.envs == 0 {
            return Err("a rollout needs at least one env".to_string());
        }
        if config.opponents.is_empty() {
            return Err("a rollout needs at least one opponent (§1.5.2 panel)".to_string());
        }

        let budget = CrashBudget::new(config.max_crashes_per_batch);
        let assignment = Assignment::PerGame(config.opponents.clone());
        let mut collector = Collector {
            envs: VecEnv::new(Vec::new()),
            open: Vec::new(),
            opponents: Vec::new(),
            sampler,
            config,
            draw_rng: env_rng(master_seed, STREAM_DRAW),
            action_rng: env_rng(master_seed, STREAM_ACTION),
            master_seed,
            started: 0,
            harvest,
            crashes: None,
            budget,
            assignment,
        };

        let mut envs = Vec::with_capacity(collector.config.envs);
        for slot in 0..collector.config.envs {
            let (env, opponent) = collector.spawn(slot)?;
            envs.push(env);
            collector.open.push(Vec::new());
            collector.opponents.push(opponent);
        }
        collector.envs = VecEnv::new(envs);
        Ok(collector)
    }

    /// Attaches the run's crash dumps (§1.5.5). A collector without one still recovers; it just
    /// leaves nothing to debug with.
    pub fn with_crash_log(mut self, log: CrashLog) -> Self {
        self.crashes = Some(log);
        self
    }

    /// Sets the §1.5.2 assignment **before the run starts**, and re-spawns the envs against it.
    ///
    /// Separate from [`Collector::set_assignment`] because the two differ in what they do to the
    /// games already in flight, and both behaviours are wanted. Construction fills every env, so a
    /// plain `set_assignment` right after `new` would leave the first collection playing the panel
    /// the config named — this rewinds to game index 0 and starts over, which is exact precisely
    /// because nothing has been collected yet.
    pub fn with_assignment(mut self, assignment: Assignment) -> Result<Self, String> {
        self.assignment = assignment;
        self.restore(0)?;
        Ok(self)
    }

    /// Replaces the §1.5.2 opponent assignment, for the games spawned from here on.
    ///
    /// Deliberately *not* applied to the games in flight: they finish against the opponent they
    /// started with, so an episode's reward stays attributable to one matchup. The caller must have
    /// loaded the models the assignment names before calling this — [`Collector::collect`] errors
    /// on an agent it was not given, rather than substituting one.
    pub fn set_assignment(&mut self, assignment: Assignment) {
        self.assignment = assignment;
    }

    pub fn assignment(&self) -> &Assignment {
        &self.assignment
    }

    /// Collect until at least `min_frames` frames sit in **finished** episodes.
    ///
    /// Games still running keep their frames and carry over to the next call, so no frame is ever
    /// discarded and none is used without its terminal reward.
    pub fn collect<B: Backend>(
        &mut self,
        model: &RlModel<B>,
        opponents: &OpponentModels<B>,
        model_config: &ModelConfig,
        device: &B::Device,
        min_frames: usize,
        batch: u64,
    ) -> Result<(Vec<Episode>, RolloutStats), String> {
        let mut episodes = Vec::new();
        let mut stats = RolloutStats::default();

        // Same reason as the draw stream: the batch index is what a resume has, the number of
        // actions sampled since the run began is not.
        self.action_rng = env_rng(self.master_seed, split_seed(STREAM_ACTION, batch));
        // Per collection, not per run: §1.5.5's guard is against a *reproducible* crash spinning
        // this loop, and a long run meeting one bad game an hour must not accumulate its way into
        // the limit.
        self.budget.reset();

        while stats.frames < min_frames {
            let (pending, finished, crashed) = self.envs.poll();

            for fault in crashed {
                let report = self.discard(fault.env, &fault.panic, batch)?;
                stats.crashes.push(report);
            }

            for done in finished {
                // Closed before the slot is replaced: `on_game_end` needs the terminal state, and
                // the replacement drops the game holding it.
                if let Some(harvest) = &mut self.harvest {
                    let closed = self
                        .envs
                        .get_mut(done.env)
                        .and_then(|env| env.close_handler());
                    if let Some(handler) = closed {
                        harvest.close_game(&handler)?;
                    }
                }

                let frames = std::mem::take(&mut self.open[done.env]);
                stats.games += 1;
                stats.frames += frames.len();
                episodes.push(Episode {
                    reward: done.outcome.reward_for(LEARNER_SEAT),
                    turns: done.outcome.turns,
                    opponent: self.opponents[done.env].clone(),
                    frames,
                });
                let (fresh, opponent) = self.spawn(done.env)?;
                self.opponents[done.env] = opponent;
                self.envs.replace(done.env, fresh);
            }

            if pending.is_empty() {
                // Every env finished on this poll; the next one re-fills them.
                continue;
            }

            // Grouped by the model that owes the answer — the whole reason [`AgentId`] exists.
            // `BTreeMap` and not a hash: the forwards run in id order, so the action stream is
            // consumed in an order that does not depend on a hasher's seed.
            let mut by_agent: BTreeMap<AgentId, Vec<usize>> = BTreeMap::new();
            for (row, slot) in pending.iter().enumerate() {
                by_agent.entry(slot.request.agent).or_default().push(row);
            }

            // Every answer is drawn before any is submitted. A submit can panic and recycle its
            // env, which invalidates that env's pending row — deciding first means the panic
            // cannot land in the middle of a readback it would have to be unwound from.
            let mut answers: Vec<Option<Answer>> = (0..pending.len()).map(|_| None).collect();
            for (agent, rows) in &by_agent {
                let model = if *agent == AgentId::LEARNER {
                    stats.forwards += 1;
                    model
                } else {
                    stats.opponent_forwards += 1;
                    opponents.get(*agent).ok_or_else(|| {
                        format!(
                            "env asked for opponent model {agent:?}, which the collector was not \
                             given — the assignment and the loaded models must be set together"
                        )
                    })?
                };

                let points: Vec<DecisionPoint<'_>> = rows
                    .iter()
                    .map(|row| DecisionPoint {
                        observation: &pending[*row].request.observation,
                        mask: &pending[*row].request.mask,
                    })
                    .collect();
                let input = ModelInput::<B>::from_points(&points, model_config, device);
                let output = model.forward(&input);
                let policy = output
                    .policy
                    .to_data()
                    .to_vec::<f32>()
                    .map_err(|err| format!("policy readback failed: {err:?}"))?;
                let values = output
                    .value
                    .to_data()
                    .to_vec::<f32>()
                    .map_err(|err| format!("value readback failed: {err:?}"))?;
                drop(points);

                let learner = *agent == AgentId::LEARNER;
                for (offset, row) in rows.iter().enumerate() {
                    let mask = &pending[*row].request.mask;
                    let row_probs =
                        &policy[offset * ACTION_MASK_DIM..(offset + 1) * ACTION_MASK_DIM];
                    // A diagnostic of the policy being *trained*: folding a frozen opponent's
                    // entropy in would report a collapse that is a checkpoint's, not the agent's.
                    if learner {
                        stats.head_entropy.observe(mask, row_probs);
                    }
                    let (entry, logprob) = sample_entry(mask, row_probs, &mut self.action_rng);
                    answers[*row] = Some(Answer {
                        head: entry.head,
                        index: entry.index,
                        logprob,
                        value: values[offset],
                        learner,
                    });
                }
            }

            for (row, slot) in pending.into_iter().enumerate() {
                let Some(answer) = answers[row].take() else {
                    continue;
                };
                let (head, index) = (answer.head, answer.index);

                // A rejected bit is a masking bug and stays fatal (§1.3.7 invariant 3); a panic
                // while the action resolves costs this one game and nothing else.
                match self.envs.submit(slot.env, head, index) {
                    Ok(()) => {
                        // **Only the learner's frames are kept.** The opponent's decisions advance
                        // the game and nothing else: they are not the policy being trained, so
                        // they belong neither in the on-policy batch nor in §1.5.1b's reservoir,
                        // where they would make the magnet the average of a frozen checkpoint's
                        // play under the best-response's name.
                        if answer.learner {
                            self.open[slot.env].push(Frame {
                                chosen_bit: head.offset() + index,
                                observation: slot.request.observation,
                                mask: slot.request.mask,
                                logprob: answer.logprob,
                                value: answer.value,
                            });
                        }
                    }
                    Err(SubmitFault::Panicked(panic)) => {
                        let report = self.discard(slot.env, &panic, batch)?;
                        stats.crashes.push(report);
                    }
                    Err(SubmitFault::Rejected(err)) => {
                        return Err(format!(
                            "env {} rejected {head:?}[{index}]: {err:?}",
                            slot.env
                        ))
                    }
                }
            }
        }

        Ok((episodes, stats))
    }

    /// The §1.5.7 harvest, for the loop's flush cadence.
    pub fn harvest_mut(&mut self) -> Option<&mut Harvest> {
        self.harvest.as_mut()
    }

    pub fn sampler(&self) -> &DeckSampler {
        &self.sampler
    }

    /// Replaces the deck sampler, for the games spawned from here on — a curriculum stage
    /// transition (§1.5.4) changing the deck DB/archetype subset.
    ///
    /// Deliberately *not* applied to games in flight, mirroring [`Collector::set_assignment`]
    /// exactly and for the same reason: the sampler is read only inside [`Collector::spawn`], so a
    /// game already dealt keeps the deck it was dealt and only a newly spawned one draws from the
    /// new sampler.
    pub fn set_sampler(&mut self, sampler: DeckSampler) {
        self.sampler = sampler;
    }

    /// Games started ever — the collector's entire persistent state (§1.5.5 checkpointing).
    pub fn games_started(&self) -> u64 {
        self.started
    }

    /// Restores the draw stream to a checkpointed position and refills the envs.
    ///
    /// Games that were in flight when the checkpoint was written are **not** restored: a
    /// truncated trajectory earns no return under γ = 1 (§1.5.1), so it is worth less than the
    /// engine-state serialization it would cost. Resuming therefore draws game indices from
    /// `started` onward and never replays one.
    pub fn restore(&mut self, started: u64) -> Result<(), String> {
        self.started = started;
        for slot in 0..self.config.envs {
            let (fresh, opponent) = self.spawn(slot)?;
            self.envs.replace(slot, fresh);
            self.opponents[slot] = opponent;
            self.open[slot].clear();
        }
        Ok(())
    }

    /// Throws away a game the engine panicked on, and starts another in its slot (§1.5.5).
    ///
    /// The frames it had accumulated go with it, which is the only correct thing to do: §1.5.1's
    /// γ = 1 gives them no return, and a truncated trajectory scored as if it had ended would
    /// teach the policy that the position it crashed the engine in was worth zero.
    ///
    /// The env's §1.5.7 harvest handler is dropped rather than closed — `on_game_end` reads the
    /// terminal state, and this game has no terminal state, only a broken one. So a crashed game
    /// contributes no labels either.
    fn discard(
        &mut self,
        slot: usize,
        panic: &EnginePanic,
        batch: u64,
    ) -> Result<CrashReport, String> {
        self.budget.charge()?;

        // Dumped before the slot is replaced: the state that panicked is what the fresh game is
        // about to overwrite.
        let dump = match (self.crashes.as_mut(), self.envs.get(slot)) {
            (Some(log), Some(env)) => log.record(panic, env, batch, slot)?,
            _ => None,
        };

        self.open[slot].clear();
        let (fresh, opponent) = self.spawn(slot)?;
        self.opponents[slot] = opponent;
        self.envs.replace(slot, fresh);

        Ok(CrashReport {
            env: slot,
            message: panic.to_string(),
            dump,
        })
    }

    /// A fresh game: decks from §1.5.3, opponent from the §1.5.2 panel. The drawn opponent comes
    /// back with the env, because it is what the finished episode will be attributed to.
    fn spawn(&mut self, slot: usize) -> Result<(Env<'static>, OpponentId), String> {
        // Reseeded per game rather than advanced: a resume knows `started`, but not how many
        // draws the sampler's rejection loops consumed getting there.
        self.draw_rng = env_rng(self.master_seed, split_seed(STREAM_DRAW, self.started));
        let [first, second] = self.sampler.sample(&mut self.draw_rng)?;

        // The opponent is decided *here*, when the game starts, and never again: an assignment set
        // mid-collection reaches the games spawned after it and leaves the ones in flight alone,
        // so an episode's reward is always attributable to the opponent that actually played it.
        let (id, seat) = match &self.assignment {
            Assignment::PerGame(panel) => {
                let code = panel[self.draw_rng.gen_range(0..panel.len())].clone();
                (
                    OpponentId::Heuristic(code.clone()),
                    OpponentSeat::Scripted(code),
                )
            }
            Assignment::Grouped(groups) => {
                let group = self.assignment.group_of(slot, self.config.envs);
                groups[group].clone()
            }
        };

        // Both seats need a `Player` to carry their deck. An agent seat's is never consulted (the
        // env yields its frames instead), so the cheapest one will do for the learner and for a
        // model-driven opponent alike.
        let placeholder = PlayerCode::ET;
        let mut codes = vec![placeholder.clone(); 2];
        let mut seats = [SeatPolicy::Scripted, SeatPolicy::Scripted];
        seats[LEARNER_SEAT] = SeatPolicy::Agent(AgentId::LEARNER);
        match &seat {
            OpponentSeat::Scripted(code) => codes[1 - LEARNER_SEAT] = code.clone(),
            OpponentSeat::Model(agent) => seats[1 - LEARNER_SEAT] = SeatPolicy::Agent(*agent),
        }

        let players = create_players(first.deck, second.deck, codes);
        let seed = split_seed(self.master_seed, self.started);
        self.started += 1;

        let mut env = Env::from_players(players, seats, seed);
        // Drawn off the deck stream, so whether a game is harvested is a function of
        // `(master_seed, game_index)` like everything else about it — a resume harvests the same
        // games it would have without the interruption.
        let sampling = self.harvest.as_ref().map(Harvest::sampling);
        if sampling.is_some_and(|sampling| sampling.draws(&mut self.draw_rng)) {
            // Named per seat, because the far seat's labels are only worth what its pilot is
            // (§1.5.7): the same decklist harvested under `er` and under a plateau checkpoint are
            // two different measurements, and only the column can tell them apart afterwards.
            let mut pilots = [LEARNER_PILOT.to_string(), LEARNER_PILOT.to_string()];
            pilots[1 - LEARNER_SEAT] = id.to_string();
            env.open_handler(Harvest::new_handler(pilots));
        }
        Ok((env, id))
    }
}

/// Draw one legal action from the masked policy.
///
/// The policy is already an exact masked softmax over the set argument bits, so this is a plain
/// categorical over `mask.entries` — the engine's own enumeration is the support, and no bit
/// outside it can be drawn however the probabilities came out. The residual renormalization is
/// float defence, not a correction: the row sums to 1 by construction.
///
/// Shared with [`super::eval`], [`crate::rl::play`] and the out-of-loop harnesses in `examples/`:
/// anything that draws its actions differently is measuring a different agent than the one the run
/// trains.
pub fn sample_entry<'a>(
    mask: &'a ActionMask,
    probs: &[f32],
    rng: &mut StdRng,
) -> (&'a MaskEntry, f32) {
    debug_assert!(!mask.entries.is_empty(), "a decision frame has candidates");

    let weight = |entry: &MaskEntry| {
        debug_assert_ne!(
            entry.head,
            Head::ActionType,
            "entries are argument bits; the family block carries marginals"
        );
        probs[entry.head.offset() + entry.index].max(0.0)
    };

    let total: f32 = mask.entries.iter().map(weight).sum();
    if total <= 0.0 {
        // A degenerate row (an untrained model can emit one) still has to answer legally.
        let entry = &mask.entries[rng.gen_range(0..mask.entries.len())];
        return (entry, (1.0 / mask.entries.len() as f32).ln());
    }

    let mut target = rng.gen::<f32>() * total;
    for entry in &mask.entries {
        target -= weight(entry);
        if target <= 0.0 {
            return (entry, (weight(entry) / total).ln());
        }
    }
    let entry = mask.entries.last().expect("non-empty");
    (entry, (weight(entry) / total).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::model::config::ModelConfig;
    use crate::rl::text_embedding::TextEmbeddings;
    use crate::rl::train::deck_db::DeckDb;
    use crate::rl::train::sampler::SamplerConfig;
    use burn::backend::NdArray;
    use std::path::Path;

    type B = NdArray;

    /// A mask holding `bits` bits on each of `heads`, uniform probabilities behind them.
    fn head_mask(heads: &[(Head, usize)]) -> (ActionMask, Vec<f32>) {
        let mut mask = ActionMask {
            actor: 0,
            regime: crate::rl::action_mask::Regime::FreePlay,
            family: [false; crate::rl::action_mask::ACTION_TYPE_DIM],
            entries: Vec::new(),
        };
        let mut probs = vec![0.0f32; ACTION_MASK_DIM];
        for (head, bits) in heads {
            for index in 0..*bits {
                mask.entries.push(MaskEntry {
                    head: *head,
                    index,
                    action: crate::actions::SimpleAction::EndTurn,
                    is_stack: false,
                });
                probs[head.offset() + index] = 0.5;
            }
        }
        (mask, probs)
    }

    /// The two §1.5.6 series read off one pool of frames: every frame that offered a head lands in
    /// exactly one of `mean`'s denominator and `forced_rate`'s numerator. A head that never
    /// appeared reports neither, which is what keeps it off the log entirely rather than logged as
    /// a zero.
    #[test]
    fn the_forced_rate_and_the_entropy_share_a_denominator() {
        let mut fold = HeadEntropy::default();
        for _ in 0..3 {
            let (mask, probs) = head_mask(&[(Head::HandPtr, 1), (Head::SlotPtrSelf, 4)]);
            fold.observe(&mask, &probs);
        }
        let (mask, probs) = head_mask(&[(Head::HandPtr, 2), (Head::SlotPtrSelf, 2)]);
        fold.observe(&mask, &probs);

        assert_eq!(fold.forced_rate(Head::HandPtr), Some(0.75));
        assert_eq!(fold.forced_rate(Head::SlotPtrSelf), Some(0.0));
        assert_eq!(fold.forced_rate(Head::Attack), None);
        assert_eq!(fold.mean(Head::Attack), None);

        // One measurement of `HandPtr`, and it is the frame the forced rate did *not* count.
        assert_eq!(
            fold.frames[HEADS
                .iter()
                .position(|h| *h == Head::HandPtr)
                .expect("head")],
            1
        );
        assert!(fold.mean(Head::HandPtr).expect("measured") > 0.0);
    }

    fn collector(envs: usize, seed: u64) -> Collector {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let sampler = DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.05,
                mirror: 0.10,
                archetypes: vec!["beginner".to_string()],
            },
        )
        .expect("sampler");
        Collector::new(
            sampler,
            RolloutConfig {
                envs,
                opponents: vec![PlayerCode::R, PlayerCode::W],
                max_crashes_per_batch: 8,
            },
            seed,
            None,
        )
        .expect("collector")
    }

    /// A deliberately tiny encoder. Nothing here is a claim about the model — these tests assert
    /// what the collector does with a forward's output, and §1.4.3's real sizes make a debug build
    /// spend minutes per test on arithmetic none of them read. [`rollout_throughput`], which *is*
    /// a claim about cost, runs the §1.4.3 defaults.
    fn small_config() -> ModelConfig {
        ModelConfig {
            d_model: 24,
            num_blocks: 1,
            num_heads: 2,
            d_ff: 32,
            d_id: 8,
            d_head_hidden: 16,
            max_scored_candidates: 24,
            ..ModelConfig::default()
        }
    }

    fn model(
        config: ModelConfig,
    ) -> (
        RlModel<B>,
        ModelConfig,
        burn::backend::ndarray::NdArrayDevice,
    ) {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = RlModel::<B>::new(&config, &TextEmbeddings::zeros(), &device);
        (model, config, device)
    }

    /// **The load-bearing property of the model-driven opponent seat.** A frozen checkpoint's
    /// decisions advance the game and must reach nothing else: not the on-policy batch, not GAE,
    /// and above all not §1.5.1b's reservoir, where they would make the magnet an average of a
    /// frozen checkpoint's play recorded under the best-response's name. Nothing downstream could
    /// tell that had happened, so it is asserted here.
    ///
    /// Both seats run the *same* weights, which is the strongest form of the test: if routing were
    /// wrong the frames would still be well-formed and legal, and only the count would give it away.
    #[test]
    fn an_opponent_seats_frames_never_reach_the_batch() {
        let (model, config, device) = model(small_config());
        let mut opponents = OpponentModels::new();
        let agent = opponents.insert(
            OpponentId::Pool(7),
            RlModel::<B>::new(&config, &TextEmbeddings::zeros(), &device),
        );

        let mut collector = collector(4, 11)
            .with_assignment(Assignment::uniform(
                OpponentId::Pool(7),
                OpponentSeat::Model(agent),
            ))
            .expect("assignment");

        let (episodes, stats) = collector
            .collect(&model, &opponents, &config, &device, 60, 0)
            .expect("rollout");

        assert!(stats.games > 0);
        // The opponent seat was answered by a model, so it cost forwards of its own.
        assert!(
            stats.opponent_forwards > 0,
            "the opponent seat should have been answered by a model"
        );
        // And every frame kept is one of the learner's: `decisions` counts what each seat was
        // asked, so the episode's frame count must match the learner's half and not the total.
        let kept: usize = episodes.iter().map(|episode| episode.frames.len()).sum();
        assert_eq!(kept, stats.frames);
        for episode in &episodes {
            assert_eq!(episode.opponent, OpponentId::Pool(7));
            assert!(!episode.frames.is_empty());
        }
    }

    /// A model-driven assignment naming weights the collector was not given is a configuration
    /// fault, and it must be one — substituting the learner would train it against itself while the
    /// log said otherwise.
    #[test]
    fn an_unloaded_opponent_model_is_an_error() {
        let (model, config, device) = model(small_config());
        let mut collector = collector(2, 3)
            .with_assignment(Assignment::uniform(
                OpponentId::Pool(1),
                OpponentSeat::Model(AgentId(1)),
            ))
            .expect("assignment");

        let err = match collector.collect(&model, &OpponentModels::new(), &config, &device, 20, 0) {
            Err(err) => err,
            Ok(_) => panic!("a collection against unloaded weights must refuse"),
        };
        assert!(err.contains("not given"), "{err}");
    }

    /// A new assignment reaches the games spawned after it and leaves the ones in flight alone.
    #[test]
    fn a_new_assignment_does_not_touch_the_games_in_flight() {
        let (model, config, device) = model(small_config());
        // Start everything against `r`, then switch before any game has finished.
        let mut collector = collector(4, 5)
            .with_assignment(Assignment::uniform(
                OpponentId::Heuristic(PlayerCode::R),
                OpponentSeat::Scripted(PlayerCode::R),
            ))
            .expect("assignment");
        let (first, _) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 1, 0)
            .expect("rollout");
        collector.set_assignment(Assignment::uniform(
            OpponentId::Heuristic(PlayerCode::W),
            OpponentSeat::Scripted(PlayerCode::W),
        ));
        // Long enough that the games spawned after the switch also *finish* inside this call — a
        // game started against `w` is not in the returned episodes until it ends.
        let (second, _) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 1_000, 1)
            .expect("rollout");

        assert!(first
            .iter()
            .all(|episode| episode.opponent == OpponentId::Heuristic(PlayerCode::R)));
        // The second collection finishes games from both eras, so both labels appear — and every
        // label is one of the two, never a mix.
        assert!(second
            .iter()
            .any(|episode| episode.opponent == OpponentId::Heuristic(PlayerCode::W)));
        assert!(second.iter().all(|episode| {
            episode.opponent == OpponentId::Heuristic(PlayerCode::R)
                || episode.opponent == OpponentId::Heuristic(PlayerCode::W)
        }));
    }

    /// `set_sampler` replaces what `Collector::sampler` reports immediately — a plain field
    /// round-trip, deliberately not an end-to-end proof through completed games. An advanced-tier
    /// game against the `r`/`w` heuristics can run far longer than a beginner one (stalling
    /// against a scripted opponent, observed empirically: 1 000 collected frames across 4 envs was
    /// not always enough to finish even one), which makes "some game reflects the new archetype"
    /// an expensive and occasionally slow thing to prove by simulation. The property that matters
    /// — that only games spawned *after* the switch see it — does not need re-proving by
    /// simulation either: it falls out of the exact same code shape
    /// `a_new_assignment_does_not_touch_the_games_in_flight` already proves for
    /// `set_assignment`, since [`Collector::set_sampler`]'s doc records that `self.sampler` is
    /// read only inside `spawn`, the identical single call site `self.assignment` is.
    #[test]
    fn set_sampler_replaces_the_sampler_collector_reports() {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let beginner = DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.0,
                mirror: 0.0,
                archetypes: vec!["beginner".to_string()],
            },
        )
        .expect("sampler");

        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let advanced = DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.0,
                mirror: 0.0,
                archetypes: vec!["advanced".to_string()],
            },
        )
        .expect("sampler");

        let mut collector = collector(2, 41);

        collector.set_sampler(beginner);
        let mut rng = env_rng(1, 0);
        let [drawn_beginner, _] = collector.sampler().sample(&mut rng).expect("draw");

        collector.set_sampler(advanced);
        let mut rng = env_rng(1, 0);
        let [drawn_advanced, _] = collector.sampler().sample(&mut rng).expect("draw");

        assert_eq!(drawn_beginner.archetype, "beginner");
        assert_eq!(drawn_advanced.archetype, "advanced");
        assert_ne!(drawn_beginner.id, drawn_advanced.id);
    }

    /// The load-bearing safety property: whatever the policy emits, the action submitted is one the
    /// engine enumerated. `VecEnv::submit` rejects an unset bit, so a collection that returns at all
    /// has already proved this — what is asserted here is that the recorded `chosen_bit` names the
    /// same action, since the update gathers its log-probability at that index.
    #[test]
    fn every_recorded_bit_is_a_legal_bit() {
        let (model, config, device) = model(small_config());
        let mut collector = collector(4, 1);
        let (episodes, stats) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 40, 0)
            .expect("rollout");

        assert!(stats.games > 0 && stats.frames >= 40);
        for episode in &episodes {
            assert!(
                [-1.0, 0.0, 1.0].contains(&episode.reward),
                "terminal reward only, got {}",
                episode.reward
            );
            for frame in &episode.frames {
                let legal = frame
                    .mask
                    .entries
                    .iter()
                    .any(|entry| entry.head.offset() + entry.index == frame.chosen_bit);
                assert!(legal, "bit {} is not in the mask", frame.chosen_bit);
                // The env resolves §1.3.6.3's forced frames itself, so a recorded frame always
                // offered a choice. §1.5.6 logs the forced rate per head *because* of this.
                assert!(
                    frame.mask.entries.len() >= 2,
                    "a recorded frame with {} bit(s)",
                    frame.mask.entries.len()
                );
                assert_eq!(frame.observation.perspective, frame.mask.actor);
                assert!(frame.logprob <= 0.0, "logprob {} > 0", frame.logprob);
            }
        }

        // §1.5.6's diagnostics over a real batch: the shares have to partition the frames, or the
        // curves they draw are measuring something other than what they are named for.
        let scalars = crate::rl::train::diagnostics::diagnostics(&episodes, &stats.head_entropy);
        let shares: f64 = scalars
            .iter()
            .filter(|(name, _)| name.starts_with("head_share/"))
            .map(|(_, value)| *value)
            .sum();
        assert!(
            (shares - 1.0).abs() < 1.0e-9,
            "head shares sum to {shares}, not 1"
        );
        for (name, value) in &scalars {
            assert!(value.is_finite(), "{name} is not finite");
            if name.starts_with("head_entropy/") {
                assert!(*value > 0.0, "{name} is {value}, but a head with two or more legal bits cannot have zero entropy");
            }
            if name.starts_with("head_forced/") {
                assert!((0.0..=1.0).contains(value), "{name} is {value}, not a rate");
            }
        }

        // The pair §1.5.6 asks for: a head that was measured at all is measured twice, on one
        // denominator. A `head_entropy/x` without its `head_forced/x` would be an entropy whose
        // skipped frames nobody counted.
        let entropies: Vec<_> = scalars
            .iter()
            .filter_map(|(name, _)| name.strip_prefix("head_entropy/"))
            .collect();
        assert!(!entropies.is_empty(), "no head was ever measured");
        for head in entropies {
            assert!(
                scalars
                    .iter()
                    .any(|(name, _)| name == &format!("head_forced/{head}")),
                "head_entropy/{head} without its companion"
            );
        }
    }

    /// Episodes are whole: the collector must never hand back frames without their terminal reward,
    /// which is the whole reason it keeps envs across calls. A game in flight at the budget must
    /// come back from a *later* call, with the frames it accumulated before the cut.
    #[test]
    fn games_in_flight_survive_the_frame_budget() {
        let (model, config, device) = model(small_config());
        let mut collector = collector(4, 2);

        let (first, _) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 20, 0)
            .expect("first");
        let carried: usize = collector.open.iter().map(|frames| frames.len()).sum();
        assert!(carried > 0, "no game was in flight at the budget");

        let (second, _) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 20, 1)
            .expect("second");
        for episode in first.iter().chain(&second) {
            assert!(!episode.frames.is_empty(), "an episode with no decisions");
        }
        assert!(
            second.iter().any(|episode| episode.frames.len() > 20),
            "no carried-over episode came back longer than one budget"
        );
    }

    /// What [`Collector::restore`] actually promises: two resumes from one checkpointed position
    /// collect the same thing. It deliberately does *not* promise equality with the uninterrupted
    /// run — the games in flight at the save are dropped, so the batch after a resume differs.
    #[test]
    fn restoring_to_a_position_twice_collects_the_same_rollout() {
        let (model, config, device) = model(small_config());
        let resume = || {
            let mut collector = collector(4, 11);
            collector.restore(64).expect("restore");
            let (episodes, stats) = collector
                .collect(&model, &OpponentModels::new(), &config, &device, 40, 7)
                .expect("rollout");
            let shape: Vec<_> = episodes
                .iter()
                .map(|episode| (episode.frames.len(), episode.reward))
                .collect();
            (shape, stats)
        };
        assert_eq!(resume(), resume());
    }

    /// The §1.5.7 harvest end to end: sampled at the env, merged across parallel games, written
    /// as shards. Worth an integration test rather than unit ones because the two failure modes
    /// are both invisible in isolation — `GameplayStatsCollector::merge` *panics* on a foreign
    /// type, and a per-game collector shared across envs would silently interleave four games'
    /// board diffs into one.
    #[test]
    fn a_harvested_rollout_writes_shards_that_hold_the_1_5_7_invariants() {
        use crate::rl::train::harvest::{Harvest, Sampling};

        let dir = std::env::temp_dir().join("deckgym-harvest-rollout");
        let _ = std::fs::remove_dir_all(&dir);
        let harvest = Harvest::new(&dir, Sampling::All(true)).expect("harvest");

        let (model, config, device) = model(small_config());
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let sampler = DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.05,
                mirror: 0.10,
                archetypes: vec!["beginner".to_string()],
            },
        )
        .expect("sampler");
        let mut collector = Collector::new(
            sampler,
            RolloutConfig {
                envs: 4,
                opponents: vec![PlayerCode::R, PlayerCode::W],
                max_crashes_per_batch: 8,
            },
            21,
            Some(harvest),
        )
        .expect("collector");

        let (_, stats) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 120, 0)
            .expect("rollout");
        assert!(stats.games > 0);

        let harvest = collector.harvest_mut().expect("harvest");
        assert_eq!(
            harvest.pending_games(),
            // Both seats are harvested: the collector keys on the deck, and a game teaches
            // §1.5.7 as much about the opponent's list as about the learner's.
            stats.games as u32,
            "every finished game has to reach the shard"
        );
        let shard = harvest.flush().expect("flush").expect("a shard");

        let cards = std::fs::read_to_string(shard.join("cards.jsonl")).expect("cards");
        let decks = std::fs::read_to_string(shard.join("decks.jsonl")).expect("decks");
        assert!(!cards.is_empty() && !decks.is_empty());

        // §1.5.7's free calibration test: per-card damage has to sum exactly to the deck's total.
        // A row-level bug in attribution shows up here and nowhere else. The join is on the whole
        // key — a mirror pairing puts the same decklist on both seats, under two different pilots.
        for line in decks.lines() {
            let deck: serde_json::Value = serde_json::from_str(line).expect("deck row");
            let key = |row: &serde_json::Value| {
                (
                    row["deck_id"].clone(),
                    row["pilot"].clone(),
                    row["opponent_pilot"].clone(),
                )
            };
            // One seat is the learner and the other is drawn from the panel, so every row names a
            // learner on exactly one side — the property that lets the offline pass drop the
            // labels a weak heuristic produced.
            assert_ne!(
                deck["pilot"] == "learner",
                deck["opponent_pilot"] == "learner",
                "row {deck} names the learner on neither side or on both"
            );
            let total: u64 = cards
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("card row"))
                .filter(|card| key(card) == key(&deck))
                .map(|card| card["damage_dealt"].as_u64().expect("damage"))
                .sum();
            assert_eq!(
                total,
                deck["damage_dealt_total"].as_u64().expect("total"),
                "deck {}: per-card damage does not sum to the deck total",
                deck["deck_id"]
            );
        }

        // A second flush with nothing pending must not leave an empty shard behind.
        assert_eq!(
            collector.harvest_mut().expect("h").flush().expect("flush"),
            None
        );
    }

    /// What a dropped game costs, and what it must not cost (§1.5.5). The panic is injected at the
    /// collector's own recovery entry point rather than provoked inside the engine: which state
    /// breaks the simulator is a bug of the day, and [`crate::rl::env`] already pins down that a
    /// panic in `step`/`submit` arrives here at all.
    #[test]
    fn a_crashed_game_loses_its_frames_and_nothing_else() {
        let dir = std::env::temp_dir().join("deckgym-rollout-crash");
        let _ = std::fs::remove_dir_all(&dir);

        let (model, config, device) = model(small_config());
        let mut collector = collector(4, 31).with_crash_log(CrashLog::new(&dir, 4));
        collector
            .collect(&model, &OpponentModels::new(), &config, &device, 30, 0)
            .expect("rollout");

        let broken = (0..4)
            .find(|slot| !collector.open[*slot].is_empty())
            .expect("a game in flight");
        let before = collector.games_started();
        let panic = crate::rl::recover::catch(|| panic!("Active Pokemon should be there"))
            .expect_err("a panic");

        let report = collector.discard(broken, &panic, 9).expect("recovery");

        assert_eq!(report.env, broken);
        assert!(report.message.contains("Active Pokemon"));
        assert!(report.dump.expect("a dump").is_file());
        // The frames go with the game: γ = 1 gives a truncated trajectory no return, so keeping
        // them would teach the policy that the position it crashed in was worth exactly zero.
        assert!(collector.open[broken].is_empty());
        // And the slot is playing again — recovery is a replacement, not a hole in the batch.
        assert_eq!(collector.games_started(), before + 1);
        let (episodes, stats) = collector
            .collect(&model, &OpponentModels::new(), &config, &device, 30, 1)
            .expect("the batch carries on");
        assert!(!episodes.is_empty());
        assert!(stats.crashes.is_empty(), "nothing else broke");
    }

    /// The guard against a *reproducible* crash: without it the collector replaces a game that
    /// panics on its first frame with another that does the same, forever, never finishing an
    /// episode and never returning.
    #[test]
    fn a_crash_storm_stops_the_run_instead_of_spinning() {
        let mut collector = collector(4, 32);
        collector.budget = CrashBudget::new(2);
        let panic = crate::rl::recover::catch(|| panic!("broken")).expect_err("a panic");

        assert!(collector.discard(0, &panic, 0).is_ok());
        assert!(collector.discard(0, &panic, 0).is_ok());
        let stopped = collector.discard(0, &panic, 0).expect_err("past the limit");
        assert!(stopped.contains("engine panics"), "unhelpful: {stopped}");
    }

    /// §1.5.5's reproducibility has to reach the whole data path, not just the deck draw: same
    /// master seed, same model, same games in the same order.
    #[test]
    fn the_same_seed_collects_the_same_rollout() {
        let (model, config, device) = model(small_config());
        let run = |seed| {
            let mut collector = collector(4, seed);
            let (episodes, stats) = collector
                .collect(&model, &OpponentModels::new(), &config, &device, 40, 0)
                .expect("rollout");
            let shape: Vec<_> = episodes
                .iter()
                .map(|episode| (episode.frames.len(), episode.reward))
                .collect();
            (shape, stats)
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3).0, run(4).0);
    }

    /// What §1.5.5's inversion of control is worth, end to end, at the §1.4.3 model sizes.
    ///
    /// The number to read is not games/s alone but games/s **against the env count**. Batching
    /// only pays where there is a fixed per-call latency to hide; if throughput is flat in the env
    /// count, the batch is buying nothing and the §1.4.3 case for the inversion does not hold on
    /// that backend.
    fn throughput<Bk: Backend>(label: &str, device: &Bk::Device) {
        let config = ModelConfig::default();
        let model = RlModel::<Bk>::new(&config, &TextEmbeddings::zeros(), device);
        for envs in [1, 8, 32, 64] {
            let mut collector = collector(envs, 9);
            // Warm the envs so the measured window is steady-state, not first-game setup.
            collector
                .collect(&model, &OpponentModels::new(), &config, device, 50, 0)
                .expect("warmup");

            let start = std::time::Instant::now();
            let (_, stats) = collector
                .collect(&model, &OpponentModels::new(), &config, device, 2000, 1)
                .expect("rollout");
            let elapsed = start.elapsed();

            println!(
                "{label} envs {envs:3}: {:6.2} games/s, {:7.1} frames/s, mean batch {:5.1}, {} forwards",
                stats.games as f64 / elapsed.as_secs_f64(),
                stats.frames as f64 / elapsed.as_secs_f64(),
                stats.frames as f64 / stats.forwards as f64,
                stats.forwards,
            );
        }
    }

    #[test]
    #[ignore = "throughput is only meaningful in release; run with --release --features rl-model -- --ignored"]
    fn rollout_throughput() {
        throughput::<NdArray>("ndarray", &Default::default());
    }

    /// The configuration §1.4.3 draws its ≈ 9.5 games/s per model-driven seat from.
    #[cfg(feature = "rl-model-cuda")]
    #[test]
    #[ignore = "GPU throughput; run with --release --features rl-model-cuda -- --ignored"]
    fn rollout_throughput_cuda() {
        throughput::<burn::backend::Cuda>("cuda   ", &Default::default());
    }
}
