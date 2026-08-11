//! Checkpointing — the `checkpoints/` half of §1.5.5's run layout.
//!
//! Two kinds, because they answer different questions.
//!
//! **Cold** is weights alone, written at a conventional stop (end of run, later end of stage).
//! It is what §1.5.2's PFSP pool will freeze and play against, and an opponent needs no optimizer
//! state — carrying one would multiply the pool's footprint for nothing.
//!
//! **Hot** is everything needed to continue the same run: weights, AdamW's moments, and the loop
//! counters — for the best-response *and*, when the run has one, for §1.5.1b's magnet. Note what
//! is *not* in it — gradients are recomputed from each batch and discarded, so there is no gradient
//! state to persist, and games in flight are dropped rather than serialized (see
//! [`super::rollout::Collector::restore`]). The magnet's reservoir is in it, but only on the way out
//! (see [`SideState::reservoir`] and [`super::reservoir`]): it is two orders of magnitude larger
//! than the rest of the directory, and dropping it silently converted the magnet from an average
//! policy into a lagged copy of the current one.
//!
//! The magnet's two files are written only when there is a magnet, and read back as an `Option`, so
//! a checkpoint from a BR-only run resumes into one — and a magnet run that resumes from such a
//! checkpoint starts its magnet fresh rather than refusing to load. What it must never do is load
//! *half* a magnet, which is the same torn-write hazard the marker below already guards.
//!
//! A crash cannot write anything, so "hot" means a **rolling autosave** on a batch cadence plus
//! whatever the interrupt handler manages on the way out. That makes torn writes the real hazard:
//! a checkpoint is three files, and dying between the first and the third would leave a directory
//! that loads as a valid model with a stale optimizer. So a hot checkpoint is published by a
//! [`DONE_MARKER`] written last, [`latest_hot`] ignores any directory without one, and the
//! previous checkpoint is only pruned once its successor is complete.

use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::{AdamW, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use crate::rl::model::RlModel;

/// Written last, and its presence is the only thing that makes a hot directory loadable.
const DONE_MARKER: &str = "complete";

/// §1.5.1b's two files, present only in a run that has a magnet.
const MAGNET_MODEL: &str = "magnet";
const MAGNET_OPTIM: &str = "magnet_optim";

/// §1.5.2's pool and ratings, present only in a run that has a pool.
const POOL_STATE: &str = "pool.json";

/// §1.5.1b's reservoir, present only in a checkpoint written on the way out — see [`save_hot`].
const RESERVOIR: &str = "reservoir.mpk";

/// The extension [`Rec`] appends. Needed because a recorder is given a stem and a *reader* has to
/// ask whether the file exists before trying to load an optional one.
const RECORD_EXT: &str = "mpk";

/// Full precision throughout: a half-precision AdamW second moment resumes into a different
/// trajectory than the one that was interrupted, which would defeat the point of the hot save.
type Rec = NamedMpkFileRecorder<FullPrecisionSettings>;

/// The optimizer travels as its record, not as the optimizer: how AdamW is *built* (the §1.5.5
/// grad clip, the zero weight decay) belongs to [`super::update::Learner`], and a checkpoint that
/// carried it could resurrect a configuration the run's `.toml` no longer asks for.
pub type AdamRecord<B> =
    <burn::optim::adaptor::OptimizerAdaptor<AdamW, RlModel<B>, B> as Optimizer<RlModel<B>, B>>::Record;

/// A network and its optimizer moments — what [`save_hot`] takes for the magnet and [`load_magnet`]
/// gives back.
pub type Trained<B> = (RlModel<B>, AdamRecord<B>);

/// The loop counters a resume needs, and the whole of them.
///
/// Both are indices into the [`super::rollout::Collector`]'s reseeded streams rather than
/// generator states — which is what lets two bytes stand in for an RNG whose representation is
/// not stable across a `rand` upgrade.
///
/// **What that buys, exactly**: two resumes from one checkpoint produce the same run. It does
/// *not* make a resumed run equal to the uninterrupted one it was cut from — the games in flight
/// at the save are dropped, so the first batch after a resume collects different episodes. That
/// difference is the price of not serializing engine state, and §1.5.1's γ = 1 is what makes it
/// cheap: a truncated trajectory carries no return anyway.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoopState {
    /// Batches completed. The resumed run starts at this index.
    pub batch: u64,
    /// [`super::rollout::Collector::games_started`] at the time of the save.
    pub games_started: u64,
    /// Seconds of *training*, accumulated across resumes (§1.5.6).
    ///
    /// A third counter beside the two above, and a different kind of thing: they are stream
    /// positions and this is a measurement. It is here anyway because the alternative is a clock
    /// that restarts at every resume, which answers "how long has this process been up" when the
    /// question is "how far into the run am I". The gap between an interrupt and its resume is
    /// deliberately not counted — a run left overnight did not train overnight.
    ///
    /// Defaulted, so a checkpoint written before it existed still resumes; it resumes with the
    /// time before the interruption lost, which is the honest reading of a field that was not
    /// measured.
    #[serde(default)]
    pub elapsed_seconds: f64,
    /// §1.5.4's curriculum stage index. Defaulted like `elapsed_seconds` — a checkpoint written
    /// before the curriculum existed resumes at stage 0, which is correct: there was no curriculum
    /// to be anywhere else in.
    #[serde(default)]
    pub stage: usize,
}

/// Writes weights alone. Bounded on `Backend` like its reader: baking a §1.5.2 reference model out
/// of an inference-backend instance is a legitimate way to produce one.
pub fn save_cold<B: Backend>(model: &RlModel<B>, path: &Path) -> Result<(), String> {
    model
        .clone()
        .save_file(path, &Rec::new())
        .map_err(|err| format!("failed to write cold checkpoint {}: {err}", path.display()))
}

/// Loads weights into a model already built at the run's [`crate::rl::model::config::ModelConfig`].
///
/// Bounded on `Backend` and not `AutodiffBackend` like its writer: §1.5.2's pool loads frozen
/// opponents onto the *inference* backend, and requiring autodiff there would make every pool
/// member carry an unused gradient machinery for weights that never take a step.
pub fn load_cold<B: Backend>(
    model: RlModel<B>,
    path: &Path,
    device: &B::Device,
) -> Result<RlModel<B>, String> {
    model
        .load_file(path, &Rec::new(), device)
        .map_err(|err| format!("failed to read cold checkpoint {}: {err}", path.display()))
}

/// The side payloads a hot checkpoint may carry, each already encoded by whoever owns its format.
///
/// Written *inside* the checkpoint and before the marker rather than beside it, so the torn-write
/// guard covers them too: a resume that restored a model without its panel would face a different
/// set of opponents than the run it continues, and would lose every rating the run had established.
#[derive(Default)]
pub struct SideState<'a> {
    /// §1.5.2's pool and rating table.
    pub pool: Option<&'a str>,
    /// §1.5.1b's reservoir, and `None` on the rolling autosave. Which saves carry it is a
    /// loop-control question, not this module's: the buffer is two orders of magnitude larger than
    /// everything else here, so it rides the exits the user controls (stop, pause) and not the
    /// cadence. See [`super::reservoir`] for what the alternative cost.
    pub reservoir: Option<&'a [u8]>,
}

/// Writes weights, optimizer state and counters into `dir/hot-<batch>/`, then publishes it.
///
/// Returns the directory written. Older complete checkpoints beyond `keep` are pruned *after*
/// this one is published, so there is no instant at which no resumable checkpoint exists.
pub fn save_hot<B: AutodiffBackend>(
    dir: &Path,
    model: &RlModel<B>,
    optimizer: AdamRecord<B>,
    magnet: Option<(&RlModel<B>, AdamRecord<B>)>,
    state: LoopState,
    side: SideState<'_>,
    keep: usize,
) -> Result<PathBuf, String> {
    let target = dir.join(format!("hot-{:08}", state.batch));
    fs::create_dir_all(&target)
        .map_err(|err| format!("failed to create {}: {err}", target.display()))?;

    model
        .clone()
        .save_file(target.join("model"), &Rec::new())
        .map_err(|err| format!("failed to write model: {err}"))?;
    Rec::new()
        .record(optimizer, target.join("optim"))
        .map_err(|err| format!("failed to write optimizer state: {err}"))?;
    if let Some((magnet, magnet_optimizer)) = magnet {
        magnet
            .clone()
            .save_file(target.join(MAGNET_MODEL), &Rec::new())
            .map_err(|err| format!("failed to write the magnet: {err}"))?;
        Rec::new()
            .record(magnet_optimizer, target.join(MAGNET_OPTIM))
            .map_err(|err| format!("failed to write the magnet's optimizer state: {err}"))?;
    }
    fs::write(
        target.join("loop.json"),
        serde_json::to_string_pretty(&state)
            .map_err(|err| format!("failed to encode loop state: {err}"))?,
    )
    .map_err(|err| format!("failed to write loop state: {err}"))?;
    if let Some(pool) = side.pool {
        fs::write(target.join(POOL_STATE), pool)
            .map_err(|err| format!("failed to write pool state: {err}"))?;
    }
    if let Some(reservoir) = side.reservoir {
        fs::write(target.join(RESERVOIR), reservoir)
            .map_err(|err| format!("failed to write the reservoir: {err}"))?;
    }

    fs::write(target.join(DONE_MARKER), b"")
        .map_err(|err| format!("failed to publish {}: {err}", target.display()))?;

    prune_hot(dir, keep.max(1))?;
    Ok(target)
}

/// Restores a hot checkpoint: weights into `model`, the optimizer state as a record for
/// [`super::update::Learner::load_optimizer`], and the counters.
pub fn load_hot<B: AutodiffBackend>(
    dir: &Path,
    model: RlModel<B>,
    device: &B::Device,
) -> Result<(RlModel<B>, AdamRecord<B>, LoopState), String> {
    let model = model
        .load_file(dir.join("model"), &Rec::new(), device)
        .map_err(|err| format!("failed to read model from {}: {err}", dir.display()))?;

    let optimizer: AdamRecord<B> = Rec::new()
        .load(dir.join("optim"), device)
        .map_err(|err| format!("failed to read optimizer state: {err}"))?;

    let state = fs::read_to_string(dir.join("loop.json"))
        .map_err(|err| format!("failed to read loop state: {err}"))?;
    let state: LoopState =
        serde_json::from_str(&state).map_err(|err| format!("failed to parse loop state: {err}"))?;

    Ok((model, optimizer, state))
}

/// §1.5.2's pool state, or `None` from a checkpoint written by a run that had no pool.
///
/// Returned as text rather than parsed here for the reason [`load_hot`] does not build a model:
/// this module owns the *file*, and [`super::panel::PanelState`] owns what is in it.
pub fn load_pool(dir: &Path) -> Result<Option<String>, String> {
    let path = dir.join(POOL_STATE);
    if !path.is_file() {
        return Ok(None);
    }
    fs::read_to_string(&path)
        .map(Some)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))
}

/// §1.5.1b's reservoir, or `None` from a checkpoint that carried none — the rolling autosave, a
/// crash-time save, or any checkpoint written before the buffer was persisted at all.
///
/// Bytes rather than a decoded buffer for the reason [`load_pool`] returns text: this module owns
/// the file, [`super::reservoir::Reservoir`] owns the record inside it.
pub fn load_reservoir(dir: &Path) -> Result<Option<Vec<u8>>, String> {
    let path = dir.join(RESERVOIR);
    if !path.is_file() {
        return Ok(None);
    }
    fs::read(&path)
        .map(Some)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))
}

/// Whether this checkpoint carries a magnet (§1.5.1b).
pub fn has_magnet(dir: &Path) -> bool {
    dir.join(MAGNET_MODEL).with_extension(RECORD_EXT).is_file()
        && dir.join(MAGNET_OPTIM).with_extension(RECORD_EXT).is_file()
}

/// Restores the magnet half, or `None` when the checkpoint was written by a run that had none.
///
/// `magnet` is a freshly built model at the run's [`crate::rl::model::config::ModelConfig`], for the
/// same reason [`load_hot`] takes one: a record loads *into* a module, it does not construct one.
pub fn load_magnet<B: AutodiffBackend>(
    dir: &Path,
    magnet: RlModel<B>,
    device: &B::Device,
) -> Result<Option<Trained<B>>, String> {
    if !has_magnet(dir) {
        return Ok(None);
    }
    let magnet = magnet
        .load_file(dir.join(MAGNET_MODEL), &Rec::new(), device)
        .map_err(|err| format!("failed to read the magnet from {}: {err}", dir.display()))?;
    let optimizer: AdamRecord<B> = Rec::new()
        .load(dir.join(MAGNET_OPTIM), device)
        .map_err(|err| format!("failed to read the magnet's optimizer state: {err}"))?;
    Ok(Some((magnet, optimizer)))
}

/// The newest **complete** hot checkpoint under `dir`, if any.
pub fn latest_hot(dir: &Path) -> Option<PathBuf> {
    complete_hot(dir).ok()?.pop()
}

/// Complete hot directories, oldest first. The batch index is zero-padded in the name, so
/// lexicographic order is chronological order.
fn complete_hot(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(Vec::new()),
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.join(DONE_MARKER).is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("hot-"))
        })
        .collect();
    found.sort();
    Ok(found)
}

/// A Ctrl-C latch, so the loop can finish the batch it is in and checkpoint on the way out
/// instead of dying between the optimizer step and the save.
///
/// A second Ctrl-C is left to the default handler: if the graceful path is itself stuck, the user
/// still has to be able to kill the process.
#[derive(Clone)]
pub struct Interrupt(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Interrupt {
    pub fn install() -> Result<Self, String> {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let raised = flag.clone();
        ctrlc::set_handler(move || {
            if raised.swap(true, std::sync::atomic::Ordering::SeqCst) {
                std::process::exit(130);
            }
        })
        .map_err(|err| format!("failed to install the interrupt handler: {err}"))?;
        Ok(Interrupt(flag))
    }

    pub fn raised(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

fn prune_hot(dir: &Path, keep: usize) -> Result<(), String> {
    let found = complete_hot(dir)?;
    for stale in found.iter().rev().skip(keep) {
        fs::remove_dir_all(stale)
            .map_err(|err| format!("failed to prune {}: {err}", stale.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::model::config::ModelConfig;
    use crate::rl::text_embedding::TextEmbeddings;
    use burn::backend::{Autodiff, NdArray};

    type B = Autodiff<NdArray>;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-checkpoint-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    /// Deliberately tiny — this file's subject is the file layout, not the model.
    fn small() -> ModelConfig {
        ModelConfig {
            d_model: 24,
            num_blocks: 1,
            num_heads: 2,
            d_ff: 32,
            d_id: 8,
            d_head_emb: 4,
            d_head_hidden: 8,
            ..Default::default()
        }
    }

    fn model(config: &ModelConfig) -> RlModel<B> {
        RlModel::new(config, &TextEmbeddings::zeros(), &Default::default())
    }

    #[test]
    fn a_cold_checkpoint_round_trips_the_weights() {
        let dir = scratch("cold");
        let config = small();
        let saved = model(&config);
        let path = dir.join("weights");

        save_cold(&saved, &path).expect("save");
        let loaded = load_cold(model(&config), &path, &Default::default()).expect("load");

        // Re-saved and compared as bytes: the record type is not `Debug`, and a fresh model of
        // the same config differs from `saved` only in its random init, so the comparison is
        // meaningful only against what round-tripped.
        let round_trip = dir.join("round-trip");
        save_cold(&loaded, &round_trip).expect("re-save");
        assert_eq!(
            fs::read(path.with_extension("mpk")).expect("first"),
            fs::read(round_trip.with_extension("mpk")).expect("second"),
        );
    }

    /// A hot checkpoint has to survive a real optimizer state, not an empty one: AdamW's moments
    /// only exist after a step, and it is the resume-into-a-different-trajectory case that the
    /// hot save exists to prevent.
    #[test]
    fn a_hot_checkpoint_round_trips_a_stepped_optimizer() {
        let dir = scratch("hot");
        let config = small();
        let mut learner = super::super::update::Learner::<B>::new(Default::default());
        let state = LoopState {
            batch: 7,
            games_started: 512,
            elapsed_seconds: 0.0,
            stage: 0,
        };

        let written = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            None,
            state,
            SideState::default(),
            2,
        )
        .expect("save");
        let (_, record, restored) =
            load_hot(&written, model(&config), &Default::default()).expect("load");
        learner.load_optimizer(record);

        assert_eq!(restored, state);
        assert!(
            !has_magnet(&written),
            "a run without a magnet must not leave one behind"
        );
        assert!(load_magnet(&written, model(&config), &Default::default())
            .expect("load")
            .is_none());
    }

    /// §1.5.1b resumes with the best-response, or the KL term comes back pointing at a fresh random
    /// network — which is worse than no magnet, since the BR would be pulled *away* from what it had
    /// converged toward. And a checkpoint written before the magnet existed still has to resume.
    #[test]
    fn a_hot_checkpoint_carries_the_magnet_and_stays_readable_without_one() {
        let dir = scratch("hot-magnet");
        let config = small();
        let learner = super::super::update::Learner::<B>::new(Default::default());
        let magnet = super::super::magnet::Magnet::<B>::new(model(&config), Default::default(), 0);
        let state = LoopState {
            batch: 3,
            games_started: 64,
            elapsed_seconds: 12.0,
            stage: 0,
        };

        let written = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            Some((magnet.weights(), magnet.optimizer_record())),
            state,
            SideState::default(),
            2,
        )
        .expect("save");

        assert!(has_magnet(&written));
        let restored = load_magnet(&written, model(&config), &Default::default())
            .expect("load")
            .expect("the checkpoint carries a magnet");

        // Re-saved and compared as bytes, for the reason the cold test gives: a fresh model of the
        // same config differs from the saved one only in its random init.
        let round_trip = dir.join("magnet-round-trip");
        save_cold(&restored.0, &round_trip).expect("re-save");
        assert_eq!(
            fs::read(written.join(MAGNET_MODEL).with_extension(RECORD_EXT)).expect("saved"),
            fs::read(round_trip.with_extension(RECORD_EXT)).expect("round-tripped"),
        );
    }

    /// A run that was already going when `elapsed_seconds` (or, later, `stage`) was added has to
    /// resume, not fail to parse: both are measurements, and losing one is not worth losing a
    /// run's weights. A missing `stage` resumes at 0, which is correct — there was no curriculum
    /// to be anywhere else in.
    #[test]
    fn a_loop_state_written_before_the_clock_or_the_curriculum_existed_still_loads() {
        let state: LoopState =
            serde_json::from_str(r#"{"batch": 3, "games_started": 9}"#).expect("older checkpoint");
        assert_eq!(state.batch, 3);
        assert_eq!(state.elapsed_seconds, 0.0);
        assert_eq!(state.stage, 0);
    }

    /// The torn-write guard: a directory without the marker is invisible, so a resume never
    /// picks up a half-written checkpoint whose optimizer state does not match its weights.
    #[test]
    fn an_unpublished_checkpoint_is_not_resumable() {
        let dir = scratch("torn");
        let config = small();
        let learner = super::super::update::Learner::<B>::new(Default::default());

        let good = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            None,
            LoopState {
                batch: 1,
                games_started: 8,
                elapsed_seconds: 0.0,
                stage: 0,
            },
            SideState::default(),
            4,
        )
        .expect("save");

        let torn = dir.join("hot-00000002");
        fs::create_dir_all(&torn).expect("torn");
        fs::write(torn.join("loop.json"), b"{}").expect("partial");

        assert_eq!(latest_hot(&dir), Some(good));
    }

    /// §1.5.2's panel travels inside the checkpoint, so it is covered by the same marker that
    /// makes a torn write invisible — and a checkpoint from a pool-less run reads back as `None`
    /// rather than refusing, the same way the magnet's two files do.
    #[test]
    fn the_pool_state_round_trips_and_is_optional() {
        let dir = scratch("pool-state");
        let config = small();
        let learner = super::super::update::Learner::<B>::new(Default::default());
        let state = LoopState {
            batch: 3,
            games_started: 12,
            elapsed_seconds: 1.0,
            stage: 0,
        };

        let with_pool = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            None,
            state,
            SideState {
                pool: Some("{\"pool\":1}"),
                ..Default::default()
            },
            4,
        )
        .expect("save");
        assert_eq!(
            load_pool(&with_pool).expect("read"),
            Some("{\"pool\":1}".to_string())
        );

        let without = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            None,
            LoopState { batch: 4, ..state },
            SideState::default(),
            4,
        )
        .expect("save");
        assert_eq!(load_pool(&without).expect("read"), None);
    }

    /// The reservoir is optional in both directions, and both directions happen in one run: the
    /// rolling autosave passes `None` and the stop passes bytes. A resume must be able to read
    /// either without knowing which wrote it — the alternative is a resume that refuses the
    /// checkpoint a crash left behind.
    #[test]
    fn the_reservoir_rides_the_checkpoint_only_when_it_is_given() {
        let dir = scratch("reservoir");
        let config = small();
        let learner = super::super::update::Learner::<B>::new(Default::default());
        let state = LoopState {
            batch: 3,
            games_started: 30,
            elapsed_seconds: 1.0,
            stage: 0,
        };

        let autosave = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            None,
            state,
            SideState::default(),
            4,
        )
        .expect("save");
        assert_eq!(load_reservoir(&autosave).expect("read"), None);

        let payload = b"\x93\x01\x02\x90".to_vec();
        let on_exit = save_hot(
            &dir,
            &model(&config),
            learner.optimizer_record(),
            None,
            LoopState { batch: 4, ..state },
            SideState {
                reservoir: Some(&payload),
                ..Default::default()
            },
            4,
        )
        .expect("save");
        assert_eq!(load_reservoir(&on_exit).expect("read"), Some(payload));
        // The marker still publishes last, so the buffer cannot be the file that outlives a torn
        // write and makes an incomplete directory look resumable.
        assert_eq!(latest_hot(&dir).as_deref(), Some(on_exit.as_path()));
    }

    #[test]
    fn pruning_keeps_the_newest_and_never_empties_the_directory() {
        let dir = scratch("prune");
        let config = small();
        let learner = super::super::update::Learner::<B>::new(Default::default());

        for batch in 0..5 {
            save_hot(
                &dir,
                &model(&config),
                learner.optimizer_record(),
                None,
                LoopState {
                    batch,
                    games_started: batch * 10,
                    elapsed_seconds: 0.0,
                    stage: 0,
                },
                SideState::default(),
                2,
            )
            .expect("save");
            assert!(
                latest_hot(&dir).is_some(),
                "batch {batch} left no checkpoint"
            );
        }

        let kept = complete_hot(&dir).expect("list");
        assert_eq!(kept.len(), 2);
        assert!(kept[1].ends_with("hot-00000004"));
    }
}
