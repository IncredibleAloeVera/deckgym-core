//! Winrate measurement — `RL_ARCHITECTURE.md` §1.5.6, and the quantity §1.5.4's floor reads.
//!
//! Two sources, answering different questions:
//!
//! - [`PanelWindow`] — folded off the *training* rollout over a rolling window. Free, on the deck
//!   distribution the curriculum defines, and high-volume enough (≈ ±2 %) to be the continuous
//!   curve. It lags: the window averages that many model versions.
//! - [`Evaluator`] — dedicated games against anchors the run does **not** train against. §1.5.2
//!   makes winrate-vs-panel a saturation signal, so only a held-out anchor measures generalization.
//!
//! Per opponent, never mixed: a mixed winrate averages matchups that move independently. Decks
//! sweep across evaluations rather than repeat, so the level stays unbiased. The policy is sampled
//! and not argmaxed — §1.5.1 optimizes the stochastic policy, and its greedy projection is a
//! different agent. Rationale for all three, and for the [`EvalGate`] screen: `NOTES.md`.
//!
//! §1.5.4's curriculum owns the opponent set and the deck restriction per stage — one
//! [`PanelWindow`]/[`EvalGate`]/[`Evaluator`] triple per stage, rebuilt on every transition — which
//! is why [`EvalConfig`] is a value rather than a constant; §1.5.2's checkpoint pool joins as one
//! more labelled opponent, which is why the reports key on a label rather than on a [`PlayerCode`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use burn::prelude::*;

use crate::players::{create_players, PlayerCode};
use crate::rl::action_mask::ACTION_MASK_DIM;
use crate::rl::env::{env_rng, split_seed, AgentId, Env, SeatPolicy, SubmitFault, VecEnv};
use crate::rl::model::config::ModelConfig;
use crate::rl::model::input::{DecisionPoint, EncodeFault, ModelInput};
use crate::rl::model::RlModel;

use super::config::EvalTrigger;
use super::diagnostics::Scalar;
use super::rollout::{sample_entry, Episode, LEARNER_SEAT};
use super::sampler::DeckSampler;

pub const REPORT_FILE: &str = "report.jsonl";

/// Stream tags. Evaluating must not shift which games the *run* plays, or it would perturb what it
/// measures.
const STREAM_EVAL_DRAW: u64 = 0x4556_414C_0000_0001;
const STREAM_EVAL_GAME: u64 = 0x4556_414C_0000_0002;
const STREAM_EVAL_ACTION: u64 = 0x4556_414C_0000_0003;

/// One anchor's result. Counts, never rates (§1.5.7's rule): an offline reader wanting a different
/// conditioning — score with ties as halves, say — has the raw numbers.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpponentReport {
    pub label: String,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    /// Decision frames and game turns, summed. Not a rate either: they are what a per-anchor cost
    /// is read against, since evaluation throughput is set by frames and not by games.
    pub decisions: u64,
    pub turns: u64,
    /// Games dropped to an engine panic (§1.5.5), missing from every count above. Always 0 for a
    /// [`PanelWindow`] fold — the rollout drops a crashed game before it becomes an episode.
    pub crashes: usize,
}

impl OpponentReport {
    /// Games that reached a terminal state — the denominator of every rate here.
    pub fn games(&self) -> usize {
        self.wins + self.losses + self.ties
    }

    /// Ties count as neither, so this stays comparable with `panel/winrate`.
    pub fn winrate(&self) -> f64 {
        match self.games() {
            0 => 0.0,
            games => self.wins as f64 / games as f64,
        }
    }

    pub fn tierate(&self) -> f64 {
        match self.games() {
            0 => 0.0,
            games => self.ties as f64 / games as f64,
        }
    }

    /// Binomial standard error of this estimate. Distinct from [`EvalReport::winrate_std`], which is
    /// the spread across anchors.
    pub fn standard_error(&self) -> f64 {
        let games = self.games();
        if games == 0 {
            return 0.0;
        }
        let p = self.winrate();
        (p * (1.0 - p) / games as f64).sqrt()
    }

    /// Decision frames per game — what an anchor's evaluation cost is actually set by.
    pub fn decisions_per_game(&self) -> f64 {
        match self.games() {
            0 => 0.0,
            games => self.decisions as f64 / games as f64,
        }
    }

    pub fn turns_per_game(&self) -> f64 {
        match self.games() {
            0 => 0.0,
            games => self.turns as f64 / games as f64,
        }
    }

    fn record(&mut self, reward: f32, decisions: u32, turns: u8) {
        if reward > 0.0 {
            self.wins += 1;
        } else if reward < 0.0 {
            self.losses += 1;
        } else {
            self.ties += 1;
        }
        self.decisions += decisions as u64;
        self.turns += turns as u64;
    }

    fn merge(&mut self, other: &OpponentReport) {
        self.wins += other.wins;
        self.losses += other.losses;
        self.ties += other.ties;
        self.decisions += other.decisions;
        self.turns += other.turns;
        self.crashes += other.crashes;
    }
}

/// One measurement over a panel, whichever of the two sources produced it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvalReport {
    pub opponents: Vec<OpponentReport>,
}

impl EvalReport {
    /// Unweighted mean of the per-anchor winrates: once game counts differ — a random opponent draw,
    /// §1.5.2's PFSP weighting — the mean of the *matchups* is the question, and weighting by games
    /// would let one over-sampled opponent own the headline.
    pub fn winrate_mean(&self) -> f64 {
        let rated: Vec<f64> = self
            .opponents
            .iter()
            .filter(|o| o.games() > 0)
            .map(|o| o.winrate())
            .collect();
        if rated.is_empty() {
            return 0.0;
        }
        rated.iter().sum::<f64>() / rated.len() as f64
    }

    /// Population standard deviation across opponents — the spread of the panel, not the error on
    /// any one estimate. 0 for a single anchor, by arithmetic rather than by result.
    pub fn winrate_std(&self) -> f64 {
        let rated: Vec<f64> = self
            .opponents
            .iter()
            .filter(|o| o.games() > 0)
            .map(|o| o.winrate())
            .collect();
        if rated.len() < 2 {
            return 0.0;
        }
        let mean = rated.iter().sum::<f64>() / rated.len() as f64;
        (rated.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / rated.len() as f64).sqrt()
    }

    /// The worst matchup, or `None` while no anchor has been played. What §1.5.4's floor reads: the
    /// mean would let a strong matchup pay for a weak one.
    pub fn winrate_min(&self) -> Option<f64> {
        self.opponents
            .iter()
            .filter(|o| o.games() > 0)
            .map(|o| o.winrate())
            .fold(None, |worst: Option<f64>, w| {
                Some(worst.map_or(w, |m| m.min(w)))
            })
    }

    /// [`Self::winrate_min`] restricted to `labels`, ignoring every other opponent in the report.
    ///
    /// What §1.5.4's screen reads, so the free window and the held-out evaluation it gates answer
    /// the same question. Unrestricted, the worst label is almost always a pool clone: §1.5.2
    /// clones the learner itself, so that matchup sits near 50 % by construction and would hold the
    /// screen shut under any floor worth setting — a run would never leave its first stage, and the
    /// held-out evaluation that decides the advance would never run once.
    pub fn winrate_min_among(&self, labels: &BTreeSet<String>) -> Option<f64> {
        self.opponents
            .iter()
            .filter(|o| o.games() > 0 && labels.contains(&o.label))
            .map(|o| o.winrate())
            .fold(None, |worst: Option<f64>, w| {
                Some(worst.map_or(w, |m| m.min(w)))
            })
    }

    /// [`Self::winrate_mean`] restricted to `labels`, and `None` while none of them has been
    /// played — a reading the caller must skip rather than fold in as a zero.
    ///
    /// The same restriction [`Self::winrate_min_among`] makes, for the same reason and against a
    /// sharper failure: §1.5.4's plateau-stop reads a *mean*, and once §1.5.2's pool is full the
    /// clones outnumber the anchors in the window. PFSP aims those matchups at ~50 %, so they hold
    /// the mean still by construction — the metric flattens when the pool fills, not when the
    /// learning stops, and no epsilon separates the two. Incident and numbers: NOTES.md, "Le
    /// plateau dilué par le pool".
    pub fn winrate_mean_among(&self, labels: &BTreeSet<String>) -> Option<f64> {
        let rated: Vec<f64> = self
            .opponents
            .iter()
            .filter(|o| o.games() > 0 && labels.contains(&o.label))
            .map(|o| o.winrate())
            .collect();
        if rated.is_empty() {
            return None;
        }
        Some(rated.iter().sum::<f64>() / rated.len() as f64)
    }

    pub fn games(&self) -> usize {
        self.opponents.iter().map(|o| o.games()).sum()
    }

    pub fn crashes(&self) -> usize {
        self.opponents.iter().map(|o| o.crashes).sum()
    }

    /// The flat §1.5.6 view. `prefix` keeps the two sources apart (`panel/window`, `eval`): one is
    /// on-distribution and the other held out, and averaging them would mean nothing.
    pub fn scalars(&self, prefix: &str) -> Vec<Scalar> {
        let mut out = Vec::with_capacity(self.opponents.len() * 4 + 2);
        for opponent in &self.opponents {
            let label = &opponent.label;
            out.push((format!("{prefix}/winrate/{label}"), opponent.winrate()));
            out.push((format!("{prefix}/tierate/{label}"), opponent.tierate()));
            out.push((
                format!("{prefix}/winrate_se/{label}"),
                opponent.standard_error(),
            ));
            out.push((format!("{prefix}/games/{label}"), opponent.games() as f64));
        }
        out.push((format!("{prefix}/winrate_mean"), self.winrate_mean()));
        out.push((format!("{prefix}/winrate_std"), self.winrate_std()));
        out
    }

    /// A compact `r 55.0%±5.0, w 31.4%±4.6` for the loop's stdout line.
    pub fn summary(&self) -> String {
        self.opponents
            .iter()
            .map(|o| {
                format!(
                    "{} {:.1}%±{:.1}",
                    o.label,
                    100.0 * o.winrate(),
                    100.0 * o.standard_error()
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn json(&self, batch: u64, source: &str) -> serde_json::Value {
        serde_json::json!({
            "batch": batch,
            "source": source,
            "winrate_mean": self.winrate_mean(),
            "winrate_std": self.winrate_std(),
            "opponents": self.opponents.iter().map(|o| serde_json::json!({
                "label": o.label,
                "games": o.games(),
                "wins": o.wins,
                "losses": o.losses,
                "ties": o.ties,
                "crashes": o.crashes,
                "winrate": o.winrate(),
                "tierate": o.tierate(),
                "winrate_se": o.standard_error(),
                "decisions_per_game": o.decisions_per_game(),
                "turns_per_game": o.turns_per_game(),
            })).collect::<Vec<_>>(),
        })
    }
}

/// The per-opponent winrate folded off the training rollout, over the last `capacity` batches.
///
/// A window rather than a running total: a cumulative winrate converges to the average of *every*
/// version of the agent, including the untrained ones, and can only rise slowly however good the
/// current one is.
pub struct PanelWindow {
    capacity: usize,
    /// Ordered, so the report's rows — and therefore the metric series — do not permute between
    /// batches.
    batches: VecDeque<BTreeMap<String, OpponentReport>>,
}

impl PanelWindow {
    pub fn new(capacity: usize) -> Self {
        PanelWindow {
            capacity: capacity.max(1),
            batches: VecDeque::new(),
        }
    }

    /// Folds one collected batch in, evicting the oldest once the window is full.
    pub fn observe(&mut self, episodes: &[Episode]) {
        let mut tally: BTreeMap<String, OpponentReport> = BTreeMap::new();
        for episode in episodes {
            let label = episode.opponent.to_string();
            tally
                .entry(label.clone())
                .or_insert_with(|| OpponentReport {
                    label,
                    ..Default::default()
                })
                .record(episode.reward, episode.frames.len() as u32, episode.turns);
        }
        self.batches.push_back(tally);
        while self.batches.len() > self.capacity {
            self.batches.pop_front();
        }
    }

    /// Batches currently held — how much of the curve is real yet.
    pub fn batches(&self) -> usize {
        self.batches.len()
    }

    /// [`EvalGate`] screens on this: a partial window carries a wider interval than its length
    /// claims, so a floor read off one fires on noise.
    pub fn is_full(&self) -> bool {
        self.batches.len() >= self.capacity
    }

    pub fn report(&self) -> EvalReport {
        let mut merged: BTreeMap<String, OpponentReport> = BTreeMap::new();
        for batch in &self.batches {
            for (label, tally) in batch {
                merged
                    .entry(label.clone())
                    .or_insert_with(|| OpponentReport {
                        label: label.clone(),
                        ..Default::default()
                    })
                    .merge(tally);
            }
        }
        EvalReport {
            opponents: merged.into_values().collect(),
        }
    }
}

/// Decides, once per batch, whether the held-out evaluation runs. Why it screens on the free window
/// rather than a fixed cadence: `NOTES.md`.
///
/// Not checkpointed — a resumed run rebuilding an empty window evaluates a little later than an
/// uninterrupted one, which is the safe direction and cheaper than making §1.5.5's checkpoint format
/// carry a gate.
pub struct EvalGate {
    trigger: EvalTrigger,
    /// Labels the screen is read over, or every label in the window when `None` — see
    /// [`EvalReport::winrate_min_among`] for why a curriculum stage must not screen on the whole
    /// window.
    screen: Option<BTreeSet<String>>,
    /// Consecutive batches the floor has held over a full window. Any batch that fails it resets.
    holding: usize,
    /// Batch of the last evaluation, for the cooldown to measure from.
    last: Option<u64>,
    /// Evaluations run. This, not the batch index, advances the deck sweep, so a cadence changed
    /// mid-run cannot land two evaluations on the same games.
    index: u64,
}

impl EvalGate {
    pub fn new(trigger: EvalTrigger) -> Self {
        EvalGate {
            trigger,
            screen: None,
            holding: 0,
            last: None,
            index: 0,
        }
    }

    /// A gate screening on `labels` alone. An empty set is refused rather than treated as "no
    /// restriction": it would silently widen the screen back to the whole window, which is the
    /// failure this constructor exists to prevent.
    pub fn screening_on(
        trigger: EvalTrigger,
        labels: impl IntoIterator<Item = String>,
    ) -> Result<Self, String> {
        let screen: BTreeSet<String> = labels.into_iter().collect();
        if screen.is_empty() {
            return Err("a screened gate needs at least one label to screen on".to_string());
        }
        Ok(EvalGate {
            screen: Some(screen),
            ..EvalGate::new(trigger)
        })
    }

    /// The labels this gate screens on, or `None` when it reads the whole window. Exposed so
    /// §1.5.4's plateau-stop can restrict its own reading to the same set rather than keeping a
    /// second copy of it that could drift from this one.
    pub fn screen_labels(&self) -> Option<&BTreeSet<String>> {
        self.screen.as_ref()
    }

    /// The floor this gate screens on. `None` for a plain cadence — an evaluation not gated on a
    /// threshold has no threshold to confirm.
    pub fn floor(&self) -> Option<f64> {
        match &self.trigger {
            EvalTrigger::Cadence(_) => None,
            EvalTrigger::Floor(spec) => Some(spec.winrate),
        }
    }

    /// The quantity the screen compares against [`Self::floor`] — the window read over the labels
    /// this gate screens on. `None` under a cadence, which screens on nothing.
    pub fn screen(&self, window: &PanelWindow) -> Option<f64> {
        match &self.trigger {
            EvalTrigger::Cadence(_) => None,
            EvalTrigger::Floor(_) => {
                let report = window.report();
                match &self.screen {
                    Some(labels) => report.winrate_min_among(labels),
                    None => report.winrate_min(),
                }
            }
        }
    }

    /// How far into `hold` the floor has held, out of how many batches it must.
    pub fn holding(&self) -> Option<(usize, usize)> {
        match &self.trigger {
            EvalTrigger::Cadence(_) => None,
            EvalTrigger::Floor(spec) => Some((self.holding, spec.hold.max(1))),
        }
    }

    /// Batches still owed to `cooldown` before another evaluation may fire. `0` once it is spent,
    /// and on a gate that has never fired.
    pub fn cooldown_remaining(&self, batch: u64) -> u64 {
        match (&self.trigger, self.last) {
            (EvalTrigger::Floor(spec), Some(last)) => {
                spec.cooldown.saturating_sub(batch.saturating_sub(last))
            }
            _ => 0,
        }
    }

    /// Call once per batch, **after** the window has taken that batch's episodes. Returns the
    /// evaluation index to run at, or `None`.
    pub fn arm(&mut self, batch: u64, window: &PanelWindow) -> Option<u64> {
        let screened = self.screen(window);
        let due = match &self.trigger {
            EvalTrigger::Cadence(0) => false,
            EvalTrigger::Cadence(every) => batch.is_multiple_of(*every),
            EvalTrigger::Floor(spec) => {
                let met = window.is_full() && screened.is_some_and(|worst| worst >= spec.winrate);
                self.holding = if met { self.holding + 1 } else { 0 };
                let held = self.holding >= spec.hold.max(1);
                let cooled = self
                    .last
                    .is_none_or(|last| batch.saturating_sub(last) >= spec.cooldown);
                held && cooled
            }
        };

        if !due {
            return None;
        }
        // The next evaluation needs its own run of `hold` batches rather than the ones that armed
        // this one — otherwise a `cooldown` below `hold` lets the same evidence pay twice. Effective
        // spacing is `max(hold, cooldown)`.
        self.holding = 0;
        self.last = Some(batch);
        let index = self.index;
        self.index += 1;
        Some(index)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalConfig {
    /// Parallel envs. §1.5.5's batching argument applies unchanged.
    pub envs: usize,
    /// Games against **each** opponent: 100 puts a winrate near 50 % at ±10 % (95 %), 400 at ±5 %.
    pub games_per_opponent: usize,
    /// The anchors, in the order their series appear. §1.5.2's pool extends this list.
    pub opponents: Vec<PlayerCode>,
    /// Engine panics tolerated across one whole evaluation (§1.5.5).
    pub max_crashes: usize,
}

impl EvalConfig {
    fn validate(&self) -> Result<(), String> {
        if self.envs == 0 {
            return Err("an evaluation needs at least one env".to_string());
        }
        if self.games_per_opponent == 0 {
            return Err("an evaluation needs at least one game per opponent".to_string());
        }
        if self.opponents.is_empty() {
            return Err("an evaluation needs at least one opponent".to_string());
        }
        Ok(())
    }
}

pub struct Evaluator {
    config: EvalConfig,
    /// Cloned from the training sampler rather than rebuilt, so the two cannot drift onto different
    /// deck distributions — and a `meta` run does not pay a second 70 k-deck load.
    sampler: DeckSampler,
    seed: u64,
}

impl Evaluator {
    /// `seed` is the run's master seed (§1.5.5); the stream tags keep this off the rollout's
    /// generators.
    pub fn new(sampler: DeckSampler, config: EvalConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        Ok(Evaluator {
            config,
            sampler,
            seed,
        })
    }

    pub fn config(&self) -> &EvalConfig {
        &self.config
    }

    /// Play the whole panel. `index` counts evaluations, not batches: it advances the deck sweep, so
    /// a cadence changed mid-run cannot land two evaluations on the same decks.
    pub fn evaluate<B: Backend>(
        &self,
        model: &RlModel<B>,
        model_config: &ModelConfig,
        device: &B::Device,
        index: u64,
    ) -> Result<EvalReport, String> {
        let mut budget = self.config.max_crashes;
        let mut report = EvalReport::default();
        for opponent in &self.config.opponents {
            report.opponents.push(self.play(
                opponent,
                index,
                model,
                model_config,
                device,
                &mut budget,
            )?);
        }
        Ok(report)
    }

    /// `games_per_opponent` games against one anchor, at most `envs` of them in flight.
    fn play<B: Backend>(
        &self,
        opponent: &PlayerCode,
        index: u64,
        model: &RlModel<B>,
        model_config: &ModelConfig,
        device: &B::Device,
        budget: &mut usize,
    ) -> Result<OpponentReport, String> {
        let total = self.config.games_per_opponent;
        let parallel = self.config.envs.min(total);
        let mut report = OpponentReport {
            label: opponent.to_string(),
            ..Default::default()
        };

        // Both keyed on the evaluation index alone — not on the opponent, not on its position in
        // the panel. So the anchors of one evaluation face the same decks and the same uniforms,
        // and adding an opponent to the `.toml` cannot move every other anchor's number.
        let first_game = index.saturating_mul(total as u64);
        let mut action_rng = env_rng(self.seed, split_seed(STREAM_EVAL_ACTION, index));

        let mut dealt = 0u64;
        let mut envs = Vec::with_capacity(parallel);
        for _ in 0..parallel {
            envs.push(self.spawn(opponent, first_game + dealt)?);
            dealt += 1;
        }
        let mut vec_env = VecEnv::new(envs);

        while report.games() + report.crashes < total {
            let settled = report.games() + report.crashes;
            let (pending, finished, crashed) = vec_env.poll();

            for fault in crashed {
                self.charge(budget, &mut report, &fault.panic.to_string())?;
                self.refill(
                    &mut vec_env,
                    fault.env,
                    opponent,
                    first_game,
                    &mut dealt,
                    total,
                )?;
            }

            for done in finished {
                report.record(
                    done.outcome.reward_for(LEARNER_SEAT),
                    done.outcome.decisions[LEARNER_SEAT],
                    done.outcome.turns,
                );
                self.refill(
                    &mut vec_env,
                    done.env,
                    opponent,
                    first_game,
                    &mut dealt,
                    total,
                )?;
            }

            if pending.is_empty() {
                // Every live env terminated on this poll. If none did either, no env can progress
                // and the loop would spin on its own emptiness.
                if report.games() + report.crashes == settled {
                    return Err(format!(
                        "evaluation stalled against {} at {settled}/{total} games",
                        report.label
                    ));
                }
                continue;
            }

            // Encoding a frame can panic on a wire cap (§1.3.8) the way playing one can panic on an
            // engine invariant, and it is charged the same way — an evaluation that ends the
            // process reports nothing at all, which is worse than one that reports a crash.
            let mut live: Vec<usize> = (0..pending.len()).collect();
            let input = loop {
                if live.is_empty() {
                    break None;
                }
                let points: Vec<DecisionPoint<'_>> = live
                    .iter()
                    .map(|row| DecisionPoint {
                        observation: &pending[*row].request.observation,
                        mask: &pending[*row].request.mask,
                    })
                    .collect();
                match ModelInput::<B>::try_from_points(&points, model_config, device) {
                    Ok(input) => break Some(input),
                    Err(EncodeFault::Row { row, panic }) => {
                        let slot = pending[live[row]].env;
                        self.charge(budget, &mut report, &panic.to_string())?;
                        self.refill(&mut vec_env, slot, opponent, first_game, &mut dealt, total)?;
                        live.remove(row);
                    }
                    Err(EncodeFault::Batch(panic)) => {
                        return Err(format!(
                            "encoding a {}-point batch against {} panicked without a frame to \
                             blame: {panic}",
                            live.len(),
                            report.label
                        ))
                    }
                }
            };
            // Every pending frame crashed; the refilled slots are asked again on the next poll.
            let Some(input) = input else { continue };
            let policy = model
                .forward(&input)
                .policy
                .to_data()
                .to_vec::<f32>()
                .map_err(|err| format!("policy readback failed: {err:?}"))?;

            for (offset, row) in live.iter().enumerate() {
                let pending = &pending[*row];
                let row_probs = &policy[offset * ACTION_MASK_DIM..(offset + 1) * ACTION_MASK_DIM];
                let (entry, _) = sample_entry(&pending.request.mask, row_probs, &mut action_rng);
                let (head, arg) = (entry.head, entry.index);
                match vec_env.submit(pending.env, head, arg) {
                    Ok(()) => {}
                    Err(SubmitFault::Panicked(panic)) => {
                        self.charge(budget, &mut report, &panic.to_string())?;
                        self.refill(
                            &mut vec_env,
                            pending.env,
                            opponent,
                            first_game,
                            &mut dealt,
                            total,
                        )?;
                    }
                    // A rejected bit is a masking bug and stays fatal here for the same reason it
                    // does in the rollout (§1.3.7 invariant 3).
                    Err(SubmitFault::Rejected(err)) => {
                        return Err(format!(
                            "env {} rejected {head:?}[{arg}]: {err:?}",
                            pending.env
                        ))
                    }
                }
            }
        }

        Ok(report)
    }

    /// Charge one dropped game against the panic budget. Tighter than the rollout's: a rollout
    /// trades broken games for progress, an evaluation trades them for a skewed denominator.
    fn charge(
        &self,
        budget: &mut usize,
        report: &mut OpponentReport,
        panic: &str,
    ) -> Result<(), String> {
        if *budget == 0 {
            return Err(format!(
                "evaluation gave up after {} engine panics: {panic}",
                self.config.max_crashes
            ));
        }
        *budget -= 1;
        report.crashes += 1;
        Ok(())
    }

    /// Put the next game in a slot, or retire the slot once every game has been dealt out.
    fn refill(
        &self,
        vec_env: &mut VecEnv<'static>,
        slot: usize,
        opponent: &PlayerCode,
        first_game: u64,
        dealt: &mut u64,
        total: usize,
    ) -> Result<(), String> {
        if *dealt as usize >= total {
            vec_env.clear(slot);
            return Ok(());
        }
        let env = self.spawn(opponent, first_game + *dealt)?;
        *dealt += 1;
        vec_env.replace(slot, env);
        Ok(())
    }

    /// Game `index` of the sweep. Deck and seed are pure functions of it, so two runs of one config
    /// — and a resumed run — evaluate on identical games.
    fn spawn(&self, opponent: &PlayerCode, index: u64) -> Result<Env<'static>, String> {
        let mut draw_rng = env_rng(self.seed, split_seed(STREAM_EVAL_DRAW, index));
        let [first, second] = self.sampler.sample(&mut draw_rng)?;

        let mut codes = vec![opponent.clone(); 2];
        codes[LEARNER_SEAT] = PlayerCode::ET;
        let mut seats = [SeatPolicy::Scripted, SeatPolicy::Scripted];
        seats[LEARNER_SEAT] = SeatPolicy::Agent(AgentId::LEARNER);

        let players = create_players(first.deck, second.deck, codes);
        Ok(Env::from_players(
            players,
            seats,
            split_seed(self.seed, split_seed(STREAM_EVAL_GAME, index)),
        ))
    }
}

/// `runs/<name>/eval/report.jsonl` — one record per measurement, both sources, tagged.
///
/// Nested where [`super::logger::MetricLog`] is flat: the flat projection already reaches
/// TensorBoard through the batch line, so this file is the breakdown a person or a dataframe reads.
pub struct EvalLog {
    file: File,
}

impl EvalLog {
    pub fn open(eval: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(eval)
            .map_err(|err| format!("failed to create {}: {err}", eval.display()))?;
        let path = eval.join(REPORT_FILE);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
        Ok(EvalLog { file })
    }

    pub fn record(&mut self, batch: u64, source: &str, report: &EvalReport) -> Result<(), String> {
        let line = serde_json::to_string(&report.json(batch, source))
            .map_err(|err| format!("failed to encode eval report: {err}"))?;
        writeln!(self.file, "{line}")
            .map_err(|err| format!("failed to write eval report: {err}"))?;
        self.file
            .flush()
            .map_err(|err| format!("failed to flush eval report: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::text_embedding::TextEmbeddings;
    use crate::rl::train::config::FloorSpec;
    use crate::rl::train::deck_db::DeckDb;
    use crate::rl::train::rollout::Frame;
    use crate::rl::train::sampler::SamplerConfig;
    use burn::backend::NdArray;

    type B = NdArray;

    fn sampler() -> DeckSampler {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.05,
                mirror: 0.10,
                archetypes: vec!["beginner".to_string()],
            },
        )
        .expect("sampler")
    }

    fn evaluator(games: usize, envs: usize, opponents: Vec<PlayerCode>) -> Evaluator {
        Evaluator::new(
            sampler(),
            EvalConfig {
                envs,
                games_per_opponent: games,
                opponents,
                max_crashes: 16,
            },
            77,
        )
        .expect("evaluator")
    }

    /// The same deliberately tiny encoder [`super::super::rollout`]'s tests use: nothing here is a
    /// claim about the model, and §1.4.3's real sizes make a debug build spend minutes on
    /// arithmetic none of these assertions read.
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

    /// Seeded, like §1.5.5's loop seeds parameter init. An unseeded model is a different policy in
    /// every process, so anything read off its play — a winrate, a decision count — cannot be
    /// compared with the same measurement taken yesterday.
    fn model_with(
        config: ModelConfig,
    ) -> (
        RlModel<B>,
        ModelConfig,
        burn::backend::ndarray::NdArrayDevice,
    ) {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        B::seed(&device, 0xE7A1);
        let model = RlModel::<B>::new(&config, &TextEmbeddings::zeros(), &device);
        (model, config, device)
    }

    fn model() -> (
        RlModel<B>,
        ModelConfig,
        burn::backend::ndarray::NdArrayDevice,
    ) {
        model_with(small_config())
    }

    fn episode(opponent: PlayerCode, reward: f32) -> Episode {
        Episode {
            frames: Vec::<Frame>::new(),
            reward,
            turns: 10,
            opponent: crate::rl::train::rating::OpponentId::Heuristic(opponent),
        }
    }

    /// The shape of the report: one row per anchor, each over exactly the games it was asked for.
    /// A mixed winrate is precisely what this harness exists not to produce, so the per-opponent
    /// rows are the contract.
    #[test]
    fn every_anchor_gets_its_own_row_over_the_games_it_was_asked_for() {
        let (model, config, device) = model();
        let report = evaluator(6, 4, vec![PlayerCode::R, PlayerCode::W])
            .evaluate(&model, &config, &device, 0)
            .expect("evaluation");

        assert_eq!(report.opponents.len(), 2);
        let labels: Vec<_> = report.opponents.iter().map(|o| o.label.clone()).collect();
        assert_eq!(labels, ["r", "w"]);
        for opponent in &report.opponents {
            assert_eq!(
                opponent.games() + opponent.crashes,
                6,
                "{} played {} games, not 6",
                opponent.label,
                opponent.games()
            );
            assert!((0.0..=1.0).contains(&opponent.winrate()));
        }
        for (name, value) in report.scalars("eval") {
            assert!(value.is_finite(), "{name} is not finite");
            assert!(name.starts_with("eval/"));
        }
    }

    /// One index, one report — decks, game seeds and action draws are all pure functions of
    /// `(eval seed, index)`, so an unchanged model measures the same thing twice and every move in
    /// the curve is the model's. Without this the whole harness reports noise it cannot name, and
    /// two runs of one config are not comparable.
    #[test]
    fn an_index_evaluates_to_the_same_report_every_time() {
        let (model, config, device) = model();
        let evaluator = evaluator(6, 4, vec![PlayerCode::R, PlayerCode::W]);

        let first = evaluator
            .evaluate(&model, &config, &device, 2)
            .expect("first");
        let second = evaluator
            .evaluate(&model, &config, &device, 2)
            .expect("second");
        assert_eq!(first, second);

        // And a *different* index is a different slice of the sweep, or the curve would be
        // re-measuring one frozen sample of the deck distribution.
        let next = evaluator
            .evaluate(&model, &config, &device, 3)
            .expect("next");
        assert_ne!(first, next);
    }

    /// The sweep, at the deck draw itself: consecutive evaluations are dealt different decks, while
    /// a given index stays reproducible.
    #[test]
    fn evaluations_sweep_the_decks_and_stay_reproducible_per_index() {
        let evaluator = evaluator(4, 2, vec![PlayerCode::R]);

        let decks = |index: u64| -> Vec<String> {
            (0..4)
                .map(|game| {
                    let mut rng = env_rng(
                        evaluator.seed,
                        split_seed(STREAM_EVAL_DRAW, index * 4 + game),
                    );
                    let [first, second] = evaluator.sampler.sample(&mut rng).expect("draw");
                    format!("{}|{}", first.id, second.id)
                })
                .collect()
        };

        assert_eq!(decks(0), decks(0), "one index has to be reproducible");
        assert_ne!(
            decks(0),
            decks(1),
            "consecutive evaluations replayed the same decks"
        );
    }

    /// An anchor's result depends on the anchor and the evaluation index, and on nothing else about
    /// the panel it sits in. Decks and action draws are both keyed on the evaluation alone, so
    /// adding an opponent to the `.toml` cannot silently move every other anchor's curve — and the
    /// per-anchor winrates of one evaluation differ by the opponent rather than by what each was
    /// dealt.
    #[test]
    fn an_anchors_result_does_not_depend_on_the_rest_of_the_panel() {
        let (model, config, device) = model();

        let alone = evaluator(6, 4, vec![PlayerCode::W])
            .evaluate(&model, &config, &device, 3)
            .expect("alone");
        let beside = evaluator(6, 4, vec![PlayerCode::R, PlayerCode::W])
            .evaluate(&model, &config, &device, 3)
            .expect("beside");

        assert_eq!(alone.opponents[0], beside.opponents[1]);
    }

    /// The free, on-distribution signal: the rollout's own games, split per anchor. The mixed
    /// winrate the standard line reports is the average of these, and the average is what hides an
    /// agent that beats one anchor and stalls against another.
    #[test]
    fn the_window_splits_the_rollouts_games_by_anchor() {
        let mut window = PanelWindow::new(4);
        window.observe(&[
            episode(PlayerCode::R, 1.0),
            episode(PlayerCode::R, 1.0),
            episode(PlayerCode::R, 0.0),
            episode(PlayerCode::W, -1.0),
            episode(PlayerCode::W, -1.0),
        ]);

        let report = window.report();
        assert_eq!(report.opponents.len(), 2);
        assert_eq!(report.opponents[0].label, "r");
        assert_eq!(report.opponents[0].winrate(), 2.0 / 3.0);
        assert_eq!(report.opponents[0].tierate(), 1.0 / 3.0);
        assert_eq!(report.opponents[1].label, "w");
        assert_eq!(report.opponents[1].winrate(), 0.0);
        // Unweighted over anchors, so the three-game anchor does not outvote the two-game one.
        assert_eq!(report.winrate_mean(), 1.0 / 3.0);
    }

    /// The window is what makes the fold precise enough to replace a dedicated eval: it accumulates
    /// across batches, and it *forgets*, so the curve measures the recent agent rather than the
    /// average of every version the run has had.
    #[test]
    fn the_window_accumulates_across_batches_and_then_forgets() {
        let mut window = PanelWindow::new(2);

        window.observe(&[episode(PlayerCode::R, -1.0), episode(PlayerCode::R, -1.0)]);
        assert_eq!(window.report().opponents[0].winrate(), 0.0);

        window.observe(&[episode(PlayerCode::R, 1.0), episode(PlayerCode::R, 1.0)]);
        assert_eq!(window.batches(), 2);
        assert_eq!(window.report().opponents[0].games(), 4);
        assert_eq!(window.report().opponents[0].winrate(), 0.5);

        // Past the capacity the first batch is gone, and with it the losses it contributed.
        window.observe(&[episode(PlayerCode::R, 1.0), episode(PlayerCode::R, 1.0)]);
        assert_eq!(window.batches(), 2);
        assert_eq!(window.report().opponents[0].games(), 4);
        assert_eq!(window.report().opponents[0].winrate(), 1.0);
    }

    /// `winrate_std` is the spread *across* anchors, not the error on one of them — the two answer
    /// different questions and the headline pair would be misleading if they were confused.
    #[test]
    fn the_aggregate_separates_panel_spread_from_estimate_error() {
        let report = EvalReport {
            opponents: vec![
                OpponentReport {
                    label: "r".to_string(),
                    wins: 100,
                    losses: 0,
                    ties: 0,
                    crashes: 0,
                    ..Default::default()
                },
                OpponentReport {
                    label: "w".to_string(),
                    wins: 0,
                    losses: 100,
                    ties: 0,
                    crashes: 0,
                    ..Default::default()
                },
            ],
        };

        assert_eq!(report.winrate_mean(), 0.5);
        // A mixed winrate would report 0.5 and say nothing; the spread is what names the split.
        assert_eq!(report.winrate_std(), 0.5);
        // And each estimate is individually certain, which is the opposite of what the spread says.
        assert_eq!(report.opponents[0].standard_error(), 0.0);

        let lone = EvalReport {
            opponents: vec![report.opponents[0].clone()],
        };
        assert_eq!(lone.winrate_std(), 0.0);
    }

    /// An anchor with no games yet must not be averaged in as a 0 % winrate: early in a run the
    /// rollout has not drawn every opponent, and a phantom zero would drag the headline down and
    /// invent a spread that is missing data rather than a result.
    #[test]
    fn an_anchor_with_no_games_is_left_out_of_the_aggregate() {
        let report = EvalReport {
            opponents: vec![
                OpponentReport {
                    label: "r".to_string(),
                    wins: 6,
                    losses: 4,
                    ties: 0,
                    crashes: 0,
                    ..Default::default()
                },
                OpponentReport {
                    label: "w".to_string(),
                    ..Default::default()
                },
            ],
        };

        assert_eq!(report.winrate_mean(), 0.6);
        assert_eq!(report.winrate_std(), 0.0);
    }

    /// Ties are their own outcome: counted, excluded from the winrate numerator, and reported. A
    /// deck pool that ties often would otherwise read as one that loses often.
    #[test]
    fn ties_are_neither_wins_nor_losses() {
        let mut opponent = OpponentReport::default();
        for reward in [1.0, 1.0, 0.0, -1.0] {
            opponent.record(reward, 30, 25);
        }

        assert_eq!((opponent.wins, opponent.losses, opponent.ties), (2, 1, 1));
        assert_eq!(opponent.games(), 4);
        assert_eq!(opponent.winrate(), 0.5);
        assert_eq!(opponent.tierate(), 0.25);
    }

    /// The report file carries both sources, tagged, and a resume extends it rather than restarting
    /// it — one run is one series however many times it was interrupted.
    #[test]
    fn the_report_file_appends_one_tagged_record_per_measurement() {
        let dir = std::env::temp_dir().join("deckgym-eval-log");
        let _ = std::fs::remove_dir_all(&dir);

        let report = EvalReport {
            opponents: vec![OpponentReport {
                label: "r".to_string(),
                wins: 7,
                losses: 2,
                ties: 1,
                crashes: 0,
                ..Default::default()
            }],
        };

        let mut log = EvalLog::open(&dir).expect("open");
        log.record(0, "panel_window", &report).expect("record");
        drop(log);
        let mut reopened = EvalLog::open(&dir).expect("reopen");
        reopened.record(20, "eval", &report).expect("record");

        let lines: Vec<serde_json::Value> = std::fs::read_to_string(dir.join(REPORT_FILE))
            .expect("report")
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid json per line"))
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["batch"], 0);
        assert_eq!(lines[0]["source"], "panel_window");
        assert_eq!(lines[1]["source"], "eval");
        assert_eq!(lines[0]["opponents"][0]["games"], 10);
        assert_eq!(lines[0]["winrate_mean"], 0.7);
    }

    /// The floor is the **worst** matchup, so a strong anchor cannot pay for a weak one — the whole
    /// point of splitting the panel in the first place.
    #[test]
    fn the_floor_reads_the_worst_matchup_not_the_average() {
        let report = EvalReport {
            opponents: vec![
                OpponentReport {
                    label: "r".to_string(),
                    wins: 95,
                    losses: 5,
                    ties: 0,
                    crashes: 0,
                    ..Default::default()
                },
                OpponentReport {
                    label: "w".to_string(),
                    wins: 45,
                    losses: 55,
                    ties: 0,
                    crashes: 0,
                    ..Default::default()
                },
            ],
        };

        assert_eq!(report.winrate_mean(), 0.70);
        // A mean-based floor of 0.70 would pass this panel; the min says what is actually true.
        assert_eq!(report.winrate_min(), Some(0.45));
        assert_eq!(EvalReport::default().winrate_min(), None);
    }

    fn losing(count: usize) -> Vec<Episode> {
        (0..count).map(|_| episode(PlayerCode::R, -1.0)).collect()
    }

    fn winning(count: usize) -> Vec<Episode> {
        (0..count).map(|_| episode(PlayerCode::R, 1.0)).collect()
    }

    /// Half won, half lost — what §1.5.2's clone of the learner looks like in the window.
    fn even_against_a_clone(batch: u64, count: usize) -> Vec<Episode> {
        (0..count)
            .map(|i| Episode {
                frames: Vec::<Frame>::new(),
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
                turns: 10,
                opponent: crate::rl::train::rating::OpponentId::Pool(batch),
            })
            .collect()
    }

    /// A stage screens on its own anchors, because the window it shares with §1.5.2 holds clones
    /// whose winrate is near 50 % by construction. Unscreened, that clone is the worst label
    /// forever and the advance it guards is never decided.
    #[test]
    fn a_screened_gate_ignores_the_pool_clone_that_shares_its_window() {
        let spec = FloorSpec {
            winrate: 0.70,
            hold: 1,
            cooldown: 0,
        };
        let mut screened =
            EvalGate::screening_on(EvalTrigger::Floor(spec.clone()), ["r".to_string()])
                .expect("gate");
        let mut unscreened = EvalGate::new(EvalTrigger::Floor(spec));

        let mut window = PanelWindow::new(1);
        let mut batch = winning(10);
        batch.extend(even_against_a_clone(400, 10));
        window.observe(&batch);

        assert_eq!(screened.arm(0, &window), Some(0));
        assert!(
            unscreened.arm(0, &window).is_none(),
            "the clone's 50 % is the worst label, and it holds the unscreened gate shut"
        );
    }

    /// An empty label set would widen the screen back to the whole window — the exact failure the
    /// screened constructor exists to prevent, so it is refused rather than accepted as "no filter".
    #[test]
    fn a_screened_gate_needs_something_to_screen_on() {
        assert!(EvalGate::screening_on(EvalTrigger::Cadence(1), []).is_err());
    }

    /// A plain cadence, including the per-batch non-regression mode — the same mechanism as the
    /// floor rather than a second one, which is why `trigger = 1` needs no code of its own.
    #[test]
    fn a_cadence_fires_on_its_multiples_and_zero_never_fires() {
        let window = PanelWindow::new(1);

        let mut off = EvalGate::new(EvalTrigger::Cadence(0));
        let mut every_third = EvalGate::new(EvalTrigger::Cadence(3));
        let mut every_batch = EvalGate::new(EvalTrigger::Cadence(1));

        let fired: Vec<u64> = (0..7)
            .filter(|batch| every_third.arm(*batch, &window).is_some())
            .collect();

        assert_eq!(fired, [0, 3, 6]);
        assert!((0..7).all(|batch| off.arm(batch, &window).is_none()));
        assert!((0..7).all(|batch| every_batch.arm(batch, &window).is_some()));
        // The index counts evaluations, not batches, so the deck sweep never revisits a slice.
        assert_eq!(every_third.index, 3);
        assert_eq!(off.floor(), None, "a cadence has no verdict to confirm");
    }

    /// The screen: nothing fires while the agent is below the floor, and nothing fires on a window
    /// that is not full yet — a partial window carries a wider interval than its length claims.
    #[test]
    fn the_floor_waits_for_a_full_window_above_the_threshold() {
        let mut gate = EvalGate::new(EvalTrigger::Floor(FloorSpec {
            winrate: 0.70,
            hold: 1,
            cooldown: 0,
        }));
        let mut window = PanelWindow::new(3);

        // Winning from the first batch, but the window is not full: still no evaluation.
        for batch in 0..2 {
            window.observe(&winning(10));
            assert!(!window.is_full());
            assert!(gate.arm(batch, &window).is_none(), "batch {batch}");
        }

        window.observe(&winning(10));
        assert!(window.is_full());
        assert_eq!(gate.arm(2, &window), Some(0));
    }

    /// `hold` is the guard against a single noisy touch of the threshold, and it *resets*: a run of
    /// batches above the floor interrupted by one below has to start over.
    #[test]
    fn the_floor_must_hold_and_a_single_bad_batch_resets_it() {
        let mut gate = EvalGate::new(EvalTrigger::Floor(FloorSpec {
            winrate: 0.70,
            hold: 3,
            cooldown: 0,
        }));
        let mut window = PanelWindow::new(1);

        window.observe(&winning(10));
        assert!(gate.arm(0, &window).is_none(), "one batch is not a level");
        window.observe(&winning(10));
        assert!(gate.arm(1, &window).is_none());

        // A dip below the floor, and the count starts again from zero.
        window.observe(&losing(10));
        assert!(gate.arm(2, &window).is_none());
        window.observe(&winning(10));
        assert!(gate.arm(3, &window).is_none(), "the run restarted");
        window.observe(&winning(10));
        assert!(gate.arm(4, &window).is_none());
        window.observe(&winning(10));
        assert_eq!(gate.arm(5, &window), Some(0));
    }

    /// `cooldown` is the guard against repeated testing: a window sitting on the floor would
    /// otherwise re-test every batch, and enough independent tests eventually pass on noise alone.
    #[test]
    fn the_cooldown_spaces_out_repeated_tests() {
        let mut gate = EvalGate::new(EvalTrigger::Floor(FloorSpec {
            winrate: 0.70,
            hold: 1,
            cooldown: 5,
        }));
        let mut window = PanelWindow::new(1);

        let fired: Vec<u64> = (0..12)
            .filter(|batch| {
                window.observe(&winning(10));
                gate.arm(*batch, &window).is_some()
            })
            .collect();

        assert_eq!(fired, [0, 5, 10]);
    }

    /// A `cooldown` shorter than `hold` must not let one run of batches pay for two evaluations:
    /// firing resets the hold, so consecutive screens read disjoint evidence and the spacing is
    /// `max(hold, cooldown)` however the two are written.
    #[test]
    fn a_cooldown_below_the_hold_does_not_reuse_the_same_batches() {
        let mut gate = EvalGate::new(EvalTrigger::Floor(FloorSpec {
            winrate: 0.70,
            hold: 4,
            cooldown: 1,
        }));
        let mut window = PanelWindow::new(1);

        let fired: Vec<u64> = (0..12)
            .filter(|batch| {
                window.observe(&winning(10));
                gate.arm(*batch, &window).is_some()
            })
            .collect();

        assert_eq!(fired, [3, 7, 11], "spacing collapsed to the cooldown");
    }

    #[test]
    fn a_panel_with_no_opponents_is_refused() {
        assert!(Evaluator::new(
            sampler(),
            EvalConfig {
                envs: 4,
                games_per_opponent: 4,
                opponents: vec![],
                max_crashes: 0,
            },
            0,
        )
        .is_err());
    }

    /// What an evaluation costs per anchor, and why. Expectiminimax is the reason this exists: it is
    /// the only anchor outside the training mix, so it is the only held-out signal, and whether a
    /// few hundred games against it is a budget decision or a rounding error is not guessable.
    ///
    /// `dec/game` is printed beside games/s because the cost is set by *frames*, not games — one
    /// model forward per learner decision — so an anchor that ends games sooner is cheaper to
    /// evaluate against however expensive its own search is.
    #[test]
    #[ignore = "cost measurement; run with --release --features rl-model -- --ignored --nocapture"]
    fn anchor_evaluation_cost() {
        let (model, config, device) = model_with(ModelConfig::default());
        let games = 16;

        println!(
            "{:>3}  {:>8} {:>8} {:>9} {:>9} {:>8}",
            "", "games/s", "dec/s", "dec/game", "turns/gm", "winrate"
        );
        for anchor in [
            PlayerCode::R,
            PlayerCode::W,
            PlayerCode::E { max_depth: 2 },
            PlayerCode::E { max_depth: 3 },
        ] {
            let evaluator = evaluator(games, 16, vec![anchor.clone()]);
            let start = std::time::Instant::now();
            let report = evaluator
                .evaluate(&model, &config, &device, 0)
                .expect("evaluation");
            let elapsed = start.elapsed().as_secs_f64();
            let anchor_report = &report.opponents[0];
            println!(
                "{:>3}: {:8.2} {:8.1} {:9.1} {:9.1} {:7.1}%",
                anchor.to_string(),
                games as f64 / elapsed,
                anchor_report.decisions as f64 / elapsed,
                anchor_report.decisions_per_game(),
                anchor_report.turns_per_game(),
                100.0 * anchor_report.winrate(),
            );
        }
    }
}
