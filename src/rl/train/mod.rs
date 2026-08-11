//! Training loop (v1) — `RL_ARCHITECTURE.md` §1.5.
//!
//! Built: the §1.5.3 deck sampler, the §1.5.1 MMD step ([`gae`] + [`update`]) with its magnet half
//! ([`magnet`], [`reservoir`], [`anchor`]), over the §1.5.5 rollout ([`rollout`]), and the run's
//! `.toml` and directory ([`config`], [`run_dir`]). §1.5.5's environment lives one level up, in
//! [`crate::rl::env`].
//!
//! §1.5.2 is built and wired behind `[pool] enabled`: [`rating`] scores, [`pool`] decides
//! membership, [`baked`] is a frozen model on disk, [`opponent`] gets one onto the far seat, and
//! [`panel`] is what the loop calls.
//!
//! §1.5.4 is built: [`curriculum`] is the stage/plateau state machine, reusing [`eval`]'s
//! screen-then-confirm harness for the advance rule and `panel/window` alone for the plateau stop.

#[cfg(feature = "rl-model")]
pub mod anchor;
#[cfg(feature = "rl-model")]
pub mod baked;
#[cfg(feature = "rl-model")]
pub mod checkpoint;
pub mod config;
pub mod crash;
#[cfg(feature = "rl-model")]
pub mod curriculum;
pub mod dashboard;
pub mod deck_db;
#[cfg(feature = "rl-model")]
pub mod diagnostics;
#[cfg(feature = "rl-model")]
pub mod eval;
pub mod gae;
pub mod harvest;
#[cfg(feature = "rl-model")]
pub mod logger;
#[cfg(feature = "rl-model")]
pub mod magnet;
#[cfg(feature = "rl-model")]
pub mod opponent;
#[cfg(feature = "rl-model")]
pub mod panel;
#[cfg(feature = "rl-model")]
pub mod pause;
pub mod pool;
pub mod rating;
#[cfg(feature = "rl-model")]
pub mod reservoir;
#[cfg(feature = "rl-model")]
pub mod rollout;
pub mod run_dir;
pub mod sampler;
pub mod schedule;
#[cfg(feature = "rl-model")]
pub mod update;

#[cfg(feature = "rl-model")]
pub use anchor::{AnchorConfig, AnchorSeed, AnchorShare, AnchorStats};
#[cfg(feature = "rl-model")]
pub use baked::{load_model as load_baked, Baked, BakedMeta, BakedRating, Provenance};
#[cfg(feature = "rl-model")]
pub use checkpoint::{
    has_magnet, latest_hot, load_cold, load_hot, load_magnet, load_reservoir, save_cold, save_hot,
    LoopState, SideState,
};
pub use config::{
    DeckSection, EvalSection, EvalTrigger, FloorSpec, PoolSection, RecoverySection, RunSection,
    TrainConfig,
};
pub use crash::{CrashBudget, CrashLog};
#[cfg(feature = "rl-model")]
pub use curriculum::{Curriculum, CurriculumEvent, CurriculumOutcome, Stage, StagePanel};
pub use dashboard::{Dashboard, Frame as DashboardFrame};
pub use deck_db::{Archetype, DeckDb, DeckEntry};
#[cfg(feature = "rl-model")]
pub use diagnostics::attention as attention_scalars;
#[cfg(feature = "rl-model")]
pub use diagnostics::curriculum as curriculum_scalars;
#[cfg(feature = "rl-model")]
pub use diagnostics::magnet as magnet_scalars;
#[cfg(feature = "rl-model")]
pub use diagnostics::{diagnostics, standard, Scalar};
#[cfg(feature = "rl-model")]
pub use eval::{EvalConfig, EvalGate, EvalLog, EvalReport, Evaluator, OpponentReport, PanelWindow};
pub use gae::{batch_targets, episode_targets, Target, LAMBDA};
pub use harvest::{Harvest, Sampling};
#[cfg(feature = "rl-model")]
pub use logger::MetricLog;
#[cfg(feature = "rl-model")]
pub use magnet::{Magnet, MagnetConfig, MagnetMetrics};
#[cfg(feature = "rl-model")]
pub use opponent::{Assignment, OpponentModels, OpponentSeat};
#[cfg(feature = "rl-model")]
pub use panel::{check_eval_disjoint, Panel, PanelState};
#[cfg(feature = "rl-model")]
pub use pause::Pause;
pub use pool::{HistoryDraw, Permanent, Pool, PoolConfig, PoolRow, Refresh, Role};
pub use rating::{
    score_from_reward, win_probability, Entry, OpponentId, Rating, RatingConfig, RatingTable,
};
#[cfg(feature = "rl-model")]
pub use reservoir::{Reservoir, Sample};
#[cfg(feature = "rl-model")]
pub use rollout::{Collector, Episode, Frame, HeadEntropy, RolloutConfig, RolloutStats};
pub use run_dir::RunDir;
pub use sampler::{DeckSampler, DeckSource, SampledDeck, SamplerConfig, SourceSpec};
pub use schedule::{Schedule, ScheduleSpec, Shape};
#[cfg(feature = "rl-model")]
pub use update::{Learner, StepConfig, StepMetrics};
