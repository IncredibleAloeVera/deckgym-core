//! §1.5.4 — curriculum & stop.
//!
//! A stage is `(deck DB + archetype subset, opponent set, magnet source)`. Advancing and stopping
//! read two *different* signals, deliberately not one:
//!
//! - **Advance** reuses [`super::eval`]'s already-built "screen cheaply, confirm independently"
//!   harness wholesale: the free [`PanelWindow`] gates an [`EvalGate`]/[`FloorSpec`], which — once
//!   held for `hold` batches — fires a dedicated [`Evaluator`]. The advance verdict reads
//!   [`EvalReport::winrate_min`] off that held-out evaluation: the *worst* anchor, never the mean
//!   (§1.5.4: "the current anchor is the worst one"). Screen and verdict are read over the same
//!   opponents — [`Stage::eval_anchors`] — so a screen that arms is evidence about the question the
//!   verdict will be asked, rather than about §1.5.2's clones, which share the window but decide
//!   nothing here.
//! - **Plateau-stop** reads `panel/window`'s own `winrate_mean` over the stage's anchors — the
//!   free, on-distribution signal the metric is quite literally named after ("winrate-**vs-panel**
//!   plateau"), restricted the same way the screen above is and for the same reason: §1.5.2's
//!   clones share the window and PFSP holds them near 50 %, so an unrestricted mean goes flat when
//!   the pool fills rather than when the learning stops. It is sampled once
//!   per window turnover (every `window_batches`, the same cadence the loop already uses to record
//!   a `panel_window` line to `eval/report.jsonl`), never per batch: consecutive batches of one
//!   rolling window overlap in almost all their games, so sampling every batch would make "K
//!   consecutive readings within ε" trip on autocorrelation rather than on genuine stagnation.
//!
//!   This is a deliberate correction on the first draft of this module, which read the plateau off
//!   the *held-out* evaluation instead. That coupled the plateau check to the advance screen's
//!   `Floor` trigger, which only ever arms once the free window has *already* cleared the stage's
//!   70 % floor — a run stuck at, say, 40 % would never trigger a single held-out evaluation, so
//!   the plateau tracker would never observe anything and could never stop such a run. Reading the
//!   free window directly instead needs no floor to have been cleared, costs nothing extra (those
//!   games are collected anyway), and is the more literal reading of "winrate-**vs-panel**" besides.
//!
//! A stage transition rebuilds the window/gate/evaluator from scratch (a fresh screen for a fresh
//! deck/opponent distribution) and resets the plateau tracker — a transition is a deliberate level
//! shift, not stagnation, and comparing means across two different opponent sets would not be a
//! meaningful Δ. It does **not** touch [`super::pool::Pool`]'s slots/archive or the rating table
//! (see [`super::pool::Pool::retarget`]) — those are what [`super::panel::Panel::retarget`] carries
//! across the transition unchanged.
//!
//! Reaching the last stage's floor is a milestone, not a stop condition: [`Curriculum::poll`]
//! reports [`CurriculumEvent::None`] and the run keeps training until the plateau or the step
//! budget ends it.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;

use crate::players::PlayerCode;
use crate::rl::env::split_seed;
use crate::rl::model::config::ModelConfig;
use crate::rl::model::RlModel;

use super::anchor::AnchorConfig;
use super::config::{EvalTrigger, FloorSpec};
use super::deck_db::DeckDb;
use super::eval::{EvalConfig, EvalGate, EvalReport, Evaluator, PanelWindow};
use super::harvest::Sampling;
use super::pool::Permanent;
use super::rollout::Episode;
use super::sampler::{DeckSampler, DeckSource, SourceSpec};

/// Keeps a stage's held-out evaluator on its own stream, mixed with the stage index so two stages
/// never share a seed.
const STREAM_CURRICULUM: u64 = 0x4355_5252_0000_0001;

/// One stage: what §1.5.3's sampler draws from, who the opponent is, and how the magnet is
/// partially reseeded on entry. Built by [`super::config::TrainConfig::curriculum_stages`], which
/// is where the `.toml` shape is validated — this type carries only what is already known-good.
#[derive(Debug, Clone)]
pub struct Stage {
    /// For logging only — printed and recorded on a transition, never parsed back.
    pub name: String,
    /// The DBs this stage draws from, with their shares — one entry in the ordinary case, several
    /// when the stage mixes (§1.5.3). Loaded at the transition into the stage, not at config load.
    pub sources: Vec<SourceSpec>,
    pub pure_mirror: f64,
    pub mirror: f64,
    pub panel: StagePanel,
    /// The heuristic subset of `panel`, for the held-out advance-eval — [`Evaluator`] only plays
    /// scripted seats (see the "out of scope" note in `RL_ARCHITECTURE.md` §1.5.4's build plan).
    pub eval_anchors: Vec<PlayerCode>,
    pub advance: FloorSpec,
    pub games_per_opponent: usize,
    pub magnet_seed: Option<AnchorConfig>,
    /// Fraction of the magnet's reservoir evicted before the reseed (§1.5.4's "partial, not
    /// total" reseed) — see [`super::reservoir::Reservoir::evict_fraction`].
    pub evict_fraction: f64,
    pub harvest_log: Option<Sampling>,
}

/// A stage's opponent set — the pool's permanent membership when `[pool]` is on, or the scripted
/// panel when it is off. Mirrors the two ways `RL_ARCHITECTURE.md` §1.5.2 already lets a run name
/// its opponents.
#[derive(Debug, Clone)]
pub enum StagePanel {
    /// `[pool].enabled = true` — the pool's permanent members for this stage
    /// ([`super::panel::Panel::retarget`]).
    Pool(Vec<Permanent>),
    /// `[pool].enabled = false` — the scripted panel for this stage
    /// ([`super::rollout::Assignment::PerGame`]).
    Scripted(Vec<PlayerCode>),
}

/// What happened on one [`Curriculum::poll`] call.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CurriculumEvent {
    #[default]
    None,
    Advanced {
        from: usize,
        to: usize,
    },
    Plateaued {
        spread: f64,
    },
}

/// The full outcome of a poll.
#[derive(Debug, Clone, Default)]
pub struct CurriculumOutcome {
    /// The held-out advance-eval's report, when one ran this batch. `None` on a batch the gate did
    /// not arm, and on a plateau stop — which reads the free window and never runs one.
    pub report: Option<EvalReport>,
    /// Whether the stage's floor was confirmed — `None` exactly when `report` is `None`.
    pub confirmed: Option<bool>,
    pub event: CurriculumEvent,
}

/// Consecutive `panel/window` `winrate_mean` readings within `epsilon` of each other, one per
/// window turnover — Part 1's global stop ("winrate-vs-panel plateau").
struct PlateauTracker {
    epsilon: f64,
    k: usize,
    history: VecDeque<f64>,
}

impl PlateauTracker {
    fn new(k: usize, epsilon: f64) -> Self {
        PlateauTracker {
            epsilon,
            k: k.max(1),
            history: VecDeque::new(),
        }
    }

    /// Pushes one observation, truncating to the trailing `k`. Returns the spread (`max - min`)
    /// once `k` have accumulated, regardless of whether it is under `epsilon` — the caller decides
    /// what to do with it, so this stays a pure measurement.
    fn observe(&mut self, mean: f64) -> Option<f64> {
        self.history.push_back(mean);
        while self.history.len() > self.k {
            self.history.pop_front();
        }
        if self.history.len() < self.k {
            return None;
        }
        let min = self.history.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self
            .history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Some(max - min)
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// A stage transition is a deliberate level shift, not stagnation — comparing a mean measured
    /// against one stage's decks/opponents to one measured against the next stage's is not a
    /// meaningful Δ, so the history starts over.
    fn reset(&mut self) {
        self.history.clear();
    }
}

/// The stage/plateau state machine driving `examples/train_player.rs`'s loop.
pub struct Curriculum {
    stages: Vec<Stage>,
    deck_root: PathBuf,
    window_batches: usize,
    eval_envs: usize,
    max_crashes: usize,
    master_seed: u64,
    index: usize,
    sampler: DeckSampler,
    window: PanelWindow,
    gate: EvalGate,
    evaluator: Evaluator,
    plateau: PlateauTracker,
}

impl Curriculum {
    /// `resume_at` is `LoopState.stage` on a resume, `0` on a fresh run — clamped to the last
    /// stage so a `.toml` shortened after a checkpoint was written still loads.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stages: Vec<Stage>,
        deck_root: PathBuf,
        window_batches: usize,
        eval_envs: usize,
        max_crashes: usize,
        plateau_k: usize,
        plateau_epsilon: f64,
        master_seed: u64,
        resume_at: usize,
    ) -> Result<Self, String> {
        if stages.is_empty() {
            return Err("a curriculum needs at least one stage".to_string());
        }
        let index = resume_at.min(stages.len() - 1);
        let (sampler, window, gate, evaluator) = build_stage(
            &stages[index],
            &deck_root,
            window_batches,
            eval_envs,
            max_crashes,
            master_seed,
            index,
        )?;
        Ok(Curriculum {
            stages,
            deck_root,
            window_batches,
            eval_envs,
            max_crashes,
            master_seed,
            index,
            sampler,
            window,
            gate,
            evaluator,
            plateau: PlateauTracker::new(plateau_k, plateau_epsilon),
        })
    }

    pub fn stage(&self) -> &Stage {
        &self.stages[self.index]
    }

    pub fn stage_index(&self) -> usize {
        self.index
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// The current stage's sampler — what the loop hands to
    /// [`super::rollout::Collector::set_sampler`] right after construction/resume, and again on
    /// every [`CurriculumEvent::Advanced`].
    pub fn sampler(&self) -> &DeckSampler {
        &self.sampler
    }

    /// The harvest sampling rate in force at the current stage: the last `harvest_log` any stage
    /// up to and including this one set — a stage without one keeps the rate already in force
    /// rather than reverting to the run-wide `[harvest]` rate, and `None` means no stage has
    /// spoken yet, leaving the run-wide rate in charge. Folding from the start is what keeps the
    /// rate a pure function of `(config, stage index)` like the sampler and the evaluator, so a
    /// resume reapplies it instead of checkpointing it.
    pub fn harvest_sampling(&self) -> Option<Sampling> {
        self.stages[..=self.index]
            .iter()
            .rev()
            .find_map(|stage| stage.harvest_log)
    }

    /// The current stage's free rolling window — what the loop reads for the same
    /// `panel/window/*` stdout line and scalars a non-curriculum run reads off its own
    /// stand-alone [`PanelWindow`], so the two modes leave the same metric names behind.
    pub fn window(&self) -> &PanelWindow {
        &self.window
    }

    /// The current stage's advance gate — read for display only. How close the screen is to firing
    /// is the one thing about a stage that a watcher cannot infer from the winrate curve, since the
    /// `hold` run resets silently on any batch that dips.
    pub fn gate(&self) -> &EvalGate {
        &self.gate
    }

    /// Folds this batch's episodes into the stage's free rolling window, checks the plateau off
    /// that window on its own turnover cadence, and — only on the batches the resulting
    /// [`EvalGate`] arms — runs the held-out evaluation and applies the advance verdict.
    pub fn poll<B: Backend>(
        &mut self,
        batch: u64,
        episodes: &[Episode],
        model: &RlModel<B>,
        model_config: &ModelConfig,
        device: &B::Device,
    ) -> Result<CurriculumOutcome, String> {
        self.window.observe(episodes);

        // The plateau reads the free window on its own turnover cadence — decoupled from the
        // advance gate below, so a stage stuck under its floor still gets a chance to stop the run
        // rather than running out the step budget in silence.
        let window_batches = self.window_batches.max(1) as u64;
        if batch > 0 && batch.is_multiple_of(window_batches) && self.window.is_full() {
            let report = self.window.report();
            // Over the stage's anchors, like the screen below — see `EvalReport::winrate_mean_among`
            // for what the unrestricted mean does once the pool is full. `None` is a window in which
            // no anchor was drawn: not a reading of zero, and not a reading at all.
            let reading = match self.gate.screen_labels() {
                Some(labels) => report.winrate_mean_among(labels),
                None => Some(report.winrate_mean()),
            };
            if let Some(spread) = reading.and_then(|mean| self.plateau.observe(mean)) {
                if spread < self.plateau.epsilon() {
                    return Ok(CurriculumOutcome {
                        report: None,
                        confirmed: None,
                        event: CurriculumEvent::Plateaued { spread },
                    });
                }
            }
        }

        let Some(eval_index) = self.gate.arm(batch, &self.window) else {
            return Ok(CurriculumOutcome::default());
        };

        let report = self
            .evaluator
            .evaluate(model, model_config, device, eval_index)?;
        let floor = self
            .gate
            .floor()
            .expect("this gate was built from EvalTrigger::Floor, so it always has one");
        let confirmed = report.winrate_min().is_some_and(|worst| worst >= floor);

        if confirmed && self.index + 1 < self.stages.len() {
            let from = self.index;
            let to = from + 1;
            let (sampler, window, gate, evaluator) = build_stage(
                &self.stages[to],
                &self.deck_root,
                self.window_batches,
                self.eval_envs,
                self.max_crashes,
                self.master_seed,
                to,
            )?;
            self.index = to;
            self.sampler = sampler;
            self.window = window;
            self.gate = gate;
            self.evaluator = evaluator;
            self.plateau.reset();
            return Ok(CurriculumOutcome {
                report: Some(report),
                confirmed: Some(confirmed),
                event: CurriculumEvent::Advanced { from, to },
            });
        }

        Ok(CurriculumOutcome {
            report: Some(report),
            confirmed: Some(confirmed),
            event: CurriculumEvent::None,
        })
    }
}

/// Builds one stage's deck sampler, free window, advance gate and held-out evaluator — shared by
/// [`Curriculum::new`] (the starting/resumed stage) and [`Curriculum::poll`] (the next one on an
/// advance), so the two can never construct a stage differently.
#[allow(clippy::too_many_arguments)]
fn build_stage(
    stage: &Stage,
    deck_root: &Path,
    window_batches: usize,
    eval_envs: usize,
    max_crashes: usize,
    master_seed: u64,
    index: usize,
) -> Result<(DeckSampler, PanelWindow, EvalGate, Evaluator), String> {
    let context = |err: String| format!("curriculum stage {:?}: {err}", stage.name);

    let mut sources = Vec::with_capacity(stage.sources.len());
    for spec in &stage.sources {
        sources.push(DeckSource {
            db: DeckDb::load(&deck_root.join(&spec.db)).map_err(&context)?,
            share: spec.share,
            archetypes: spec.archetypes.clone(),
        });
    }
    let sampler = DeckSampler::mixed(sources, stage.pure_mirror, stage.mirror).map_err(&context)?;
    let window = PanelWindow::new(window_batches);
    // The screen reads the stage's eval anchors and nothing else: the window also holds the pool's
    // clones, whose winrate is near 50 % by construction, and the worst label over the whole window
    // would never clear a floor worth setting. See `EvalReport::winrate_min_among`.
    let gate = EvalGate::screening_on(
        EvalTrigger::Floor(stage.advance.clone()),
        stage.eval_anchors.iter().map(|code| code.to_string()),
    )
    .map_err(&context)?;
    let evaluator = Evaluator::new(
        sampler.clone(),
        EvalConfig {
            envs: eval_envs,
            games_per_opponent: stage.games_per_opponent,
            opponents: stage.eval_anchors.clone(),
            max_crashes,
        },
        split_seed(master_seed, split_seed(STREAM_CURRICULUM, index as u64)),
    )
    .map_err(&context)?;

    Ok((sampler, window, gate, evaluator))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::text_embedding::TextEmbeddings;
    use crate::rl::train::rating::OpponentId;
    use burn::backend::NdArray;

    type B = NdArray;

    fn stage(name: &str, archetype: &str, advance: FloorSpec) -> Stage {
        Stage {
            name: name.to_string(),
            sources: vec![SourceSpec {
                db: "tutorial".to_string(),
                share: 1.0,
                archetypes: vec![archetype.to_string()],
            }],
            pure_mirror: 0.0,
            mirror: 0.0,
            panel: StagePanel::Scripted(vec![PlayerCode::R]),
            eval_anchors: vec![PlayerCode::R],
            advance,
            games_per_opponent: 4,
            magnet_seed: None,
            evict_fraction: 0.3,
            harvest_log: None,
        }
    }

    fn floor(winrate: f64) -> FloorSpec {
        FloorSpec {
            winrate,
            hold: 1,
            cooldown: 0,
        }
    }

    fn model() -> (
        RlModel<B>,
        ModelConfig,
        burn::backend::ndarray::NdArrayDevice,
    ) {
        let config = ModelConfig {
            d_model: 24,
            num_blocks: 1,
            num_heads: 2,
            d_ff: 32,
            d_id: 8,
            d_head_hidden: 16,
            max_scored_candidates: 24,
            ..ModelConfig::default()
        };
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = RlModel::<B>::new(&config, &TextEmbeddings::zeros(), &device);
        (model, config, device)
    }

    fn curriculum(stages: Vec<Stage>, plateau_k: usize, plateau_epsilon: f64) -> Curriculum {
        Curriculum::new(
            stages,
            PathBuf::from("decks"),
            /* window_batches */ 1,
            /* eval_envs */ 2,
            /* max_crashes */ 8,
            plateau_k,
            plateau_epsilon,
            /* master_seed */ 42,
            /* resume_at */ 0,
        )
        .expect("curriculum")
    }

    /// One episode, so the free window has a real `winrate_min` to screen the advance gate on —
    /// `EvalGate::arm` reads the window, not the held-out eval, so an empty batch can never arm it.
    fn one_win_over(opponent: PlayerCode) -> Vec<Episode> {
        vec![Episode {
            frames: Vec::new(),
            reward: 1.0,
            turns: 10,
            opponent: OpponentId::Heuristic(opponent),
        }]
    }

    /// Half won, half lost against a clone — what §1.5.2's pool contributes to the window a stage
    /// screens on, and by construction: the clone *is* the learner, a few hundred batches ago.
    fn even_against_a_clone(count: usize) -> Vec<Episode> {
        (0..count)
            .map(|i| Episode {
                frames: Vec::new(),
                reward: if i % 2 == 0 { 1.0 } else { -1.0 },
                turns: 10,
                opponent: OpponentId::Pool(400),
            })
            .collect()
    }

    /// The stage's floor is a claim about its anchors, so the screen that decides when to test it
    /// must be one too. A clone sitting at 50 % in the same window is not evidence against the
    /// claim, and before the screen was restricted it silently blocked every advance in a pooled
    /// run — the gate never armed, so the held-out evaluation never ran even once.
    #[test]
    fn a_pool_clone_in_the_window_does_not_hold_the_advance_screen_shut() {
        let mut curriculum = curriculum(
            vec![
                stage("beginner", "beginner", floor(0.70)),
                stage("advanced", "advanced", floor(0.70)),
            ],
            100,
            -1.0,
        );
        let (model, config, device) = model();

        let mut episodes = vec![];
        for _ in 0..10 {
            episodes.extend(one_win_over(PlayerCode::R));
        }
        episodes.extend(even_against_a_clone(10));

        let outcome = curriculum
            .poll::<B>(0, &episodes, &model, &config, &device)
            .expect("poll");

        // Whether the held-out evaluation *confirms* depends on how a randomly initialized model
        // actually plays; that the gate armed and ran one at all is what the screen decides.
        assert!(
            outcome.confirmed.is_some(),
            "the screen should read the anchor's 100 %, not the clone's 50 %"
        );
    }

    /// `wins` won and `losses` lost against one anchor, so a turnover carries a chosen winrate
    /// rather than the 0 or 1 a single episode can express.
    fn record_over(opponent: PlayerCode, wins: usize, losses: usize) -> Vec<Episode> {
        (0..wins + losses)
            .map(|i| Episode {
                frames: Vec::new(),
                reward: if i < wins { 1.0 } else { -1.0 },
                turns: 10,
                opponent: OpponentId::Heuristic(opponent.clone()),
            })
            .collect()
    }

    /// The plateau is a claim about the anchors too, and the pool dilutes it: the mean is unweighted
    /// per opponent, so every clone sitting at its PFSP-intended 50 % divides the anchors' real
    /// movement by one more. Incident and numbers: NOTES.md, "Le plateau dilué par le pool".
    #[test]
    fn a_pool_clone_in_the_window_does_not_fake_a_plateau() {
        // A floor of 1.1 can never be cleared, so the advance gate never arms and the plateau is
        // the only thing this poll can report.
        let mut curriculum = curriculum(vec![stage("stuck", "beginner", floor(1.1))], 3, 0.3);
        let (model, config, device) = model();

        let mut last = CurriculumEvent::None;
        // The anchor climbs 0.5 -> 0.7 -> 0.9, a spread of 0.4 that is nobody's plateau. Halved by
        // one clone at 50 %, it reads as 0.2 — under the epsilon above, which is the bug.
        for (batch, (wins, losses)) in [(5, 5), (7, 3), (9, 1)].into_iter().enumerate() {
            let mut episodes = record_over(PlayerCode::R, wins, losses);
            episodes.extend(even_against_a_clone(10));
            last = curriculum
                .poll::<B>(batch as u64 + 1, &episodes, &model, &config, &device)
                .expect("poll")
                .event;
        }

        assert!(
            !matches!(last, CurriculumEvent::Plateaued { .. }),
            "a 0.4 climb on the stage's own anchor is not a plateau, got {last:?}"
        );
    }

    /// A stage without `harvest_log` keeps the rate the previous stage set — it does not revert to
    /// the run-wide `[harvest]` rate — and the fold reads the same on a resume as it did live,
    /// which is what lets `examples/train_player.rs` reapply it instead of checkpointing it.
    #[test]
    fn harvest_sampling_is_the_last_stage_that_spoke_not_the_current_one() {
        let mut first = stage("beginner", "beginner", floor(0.70));
        first.harvest_log = Some(Sampling::Fraction(0.5));
        let second = stage("advanced", "advanced", floor(0.70));

        let at_second = Curriculum::new(
            vec![first.clone(), second.clone()],
            PathBuf::from("decks"),
            1,
            2,
            8,
            5,
            0.02,
            42,
            /* resume_at */ 1,
        )
        .expect("curriculum");
        assert_eq!(at_second.harvest_sampling(), Some(Sampling::Fraction(0.5)));

        let no_stage_spoke = curriculum(vec![second, first], 5, 0.02);
        assert_eq!(no_stage_spoke.harvest_sampling(), None);
    }

    #[test]
    fn a_fresh_curriculum_starts_at_stage_zero() {
        let curriculum = curriculum(
            vec![
                stage("beginner", "beginner", floor(0.70)),
                stage("advanced", "advanced", floor(0.70)),
            ],
            5,
            0.02,
        );
        assert_eq!(curriculum.stage_index(), 0);
        assert_eq!(curriculum.stage_count(), 2);
        assert_eq!(curriculum.stage().name, "beginner");
    }

    /// Reaching the last stage's floor is a milestone, not a stop condition — nothing in
    /// `RL_ARCHITECTURE.md` §1.5.4 says a curriculum ends when its stages run out, only when the
    /// plateau or the step budget does.
    #[test]
    fn confirming_the_floor_on_the_last_stage_has_nowhere_to_advance_to() {
        // A floor of 0.0 always confirms (both the window screen and the held-out result): every
        // game is a win, loss or tie, none of which can fall under a floor of 0 %.
        let mut curriculum = curriculum(vec![stage("only", "beginner", floor(0.0))], 100, -1.0);
        let (model, config, device) = model();

        let outcome = curriculum
            .poll::<B>(0, &one_win_over(PlayerCode::R), &model, &config, &device)
            .expect("poll");

        assert_eq!(outcome.confirmed, Some(true));
        assert_eq!(outcome.event, CurriculumEvent::None);
        assert_eq!(curriculum.stage_index(), 0, "there is nowhere else to go");
    }

    /// The behavioral contract of an advance: the index moves, the sampler/window/gate/evaluator
    /// are rebuilt for the new stage (a fresh, empty screen), and the plateau history — which
    /// compared means across what is now a different opponent/deck distribution — is discarded.
    #[test]
    fn advancing_resets_the_window_the_gate_and_the_plateau_history() {
        let mut curriculum = curriculum(
            vec![
                stage("beginner", "beginner", floor(0.0)),
                stage("advanced", "advanced", floor(0.0)),
            ],
            2,
            -1.0, // never plateaus, so only the advance path is exercised
        );
        let (model, config, device) = model();

        let outcome = curriculum
            .poll::<B>(0, &one_win_over(PlayerCode::R), &model, &config, &device)
            .expect("poll");

        assert_eq!(outcome.event, CurriculumEvent::Advanced { from: 0, to: 1 });
        assert_eq!(curriculum.stage_index(), 1);
        assert_eq!(curriculum.stage().name, "advanced");
        assert_eq!(curriculum.plateau.history.len(), 0);
        assert_eq!(
            curriculum.window.batches(),
            0,
            "the new stage's window starts empty"
        );
    }

    /// `Δ < ε` over `k` consecutive window turnovers stops the run, off the free `panel/window`
    /// signal alone — no held-out evaluation has to run for this to fire, which is the whole point:
    /// a stage stuck under its advance floor (so its `EvalGate` never arms) must still be able to
    /// trigger the global stop.
    #[test]
    fn the_plateau_fires_on_the_free_window_alone_without_ever_arming_the_advance_gate() {
        // A floor of 1.1 can never be confirmed by the window screen (no winrate exceeds 100%), so
        // `EvalGate::arm` never returns `Some` and the held-out evaluator never runs.
        let mut curriculum = curriculum(vec![stage("stuck", "beginner", floor(1.1))], 3, 0.5);
        let (model, config, device) = model();

        let mut last = CurriculumEvent::None;
        // window_batches = 1, so every batch is a turnover; three identical outcomes give the
        // plateau tracker three (nearly) identical means to compare.
        for batch in 1..=3u64 {
            let outcome = curriculum
                .poll::<B>(
                    batch,
                    &one_win_over(PlayerCode::R),
                    &model,
                    &config,
                    &device,
                )
                .expect("poll");
            assert_eq!(
                outcome.confirmed, None,
                "the advance gate must never have armed"
            );
            last = outcome.event;
        }

        assert!(
            matches!(last, CurriculumEvent::Plateaued { .. }),
            "three identical-winrate window turnovers should read as a plateau, got {last:?}"
        );
    }
}
