//! Starting a new run from an existing one's weights — §1.5.5's third entry point.
//!
//! [`super::run_dir::RunDir::create`] starts from a random init and `--resume` continues a run from
//! its own last checkpoint. This is the third case: a **new** run directory, a batch counter at 0,
//! its own log and its own config clone, whose starting weights are somebody else's.
//!
//! The axis is how much of the source comes with the weights.
//!
//! - **Cold** ([`InitMode::Cold`]) is the weights alone. AdamW starts at zero moments, so the first
//!   steps run before the second moment has any estimate of curvature — on weights that are already
//!   specialized, which is why the `.toml`'s warmup phase still matters here even though the model
//!   does not look like a fresh one.
//! - **Warm** ([`InitMode::Warm`]) adds AdamW's moments and the magnet — its network, its optimizer,
//!   and its reservoir when the checkpoint carries one. It is what a resume restores minus the loop
//!   counters, which is the entire difference between continuing a run and starting one from it.
//!
//! **The pool is a separate question, because a clone is a file in the source run.** §1.5.2 keeps a
//! clone's weights in `runs/<source>/pool/b<batch>.mpk` and its bookkeeping in the checkpoint's
//! `pool.json`; restoring the second without the first gives a panel whose members cannot be loaded.
//! Referencing them in place is not the fix — the source run's directory would then have to outlive
//! every run descended from it, and §1.5.5's promise is that a run directory is self-contained. So
//! carrying a pool *copies*, and [`PoolCarry`] is how much:
//!
//! - [`PoolCarry::Empty`] — nothing. The pool refills from scratch, which takes
//!   `(best_slots + history_slots) × clone_every` batches whose only opponents are clones of the
//!   model as it is right now. A strong model against an empty pool trains against one difficulty.
//! - [`PoolCarry::Partial`] — the clones holding a slot at the checkpoint. The archive is narrowed
//!   to them ([`super::pool::Pool::restrict_to_slots`]) and the ratings with it, so the run does not
//!   promise a history draw it cannot serve.
//! - [`PoolCarry::Full`] — the whole archive, every historical draw included.
//!
//! Ratings travel with the members in both non-empty cases, which means the new run's elo continues
//! the source's scale rather than restarting at 1500. That is the point when the source is a run one
//! is deliberately continuing, and a trap when it is not: two runs whose elo curves share an origin
//! are comparable to each other and to nothing else.
//!
//! **What is checked, and what cannot be.** A baked source is validated by
//! [`super::baked::BakedMeta::check_schema`], the same check §1.5.2's panel is held to. A hot
//! checkpoint sitting inside a run directory is checked against that run's own cloned `config.toml`
//! — `[model]` and `text_embeddings`, because burn validates the *length* of a layer `Vec` and not
//! the shapes inside it: a source that trained without `candidate_cross_attention` loads into a
//! config that has it, reports success, and leaves the scorer at its fresh init (TODO.md, and the
//! characterization test in `src/rl/model/heads.rs`). A loose `.mpk` says nothing about itself and
//! is loaded on trust, with a note. No source is checked for the *observation* schema the way a
//! baked model is: a run directory records no fingerprint, so a checkpoint from before a width moved
//! is caught by the load failing on shapes, and one from before a width *changed meaning* is not
//! caught at all. Bake it if that matters — `meta.toml` is where that fact is written down.

use std::fs;
use std::path::{Path, PathBuf};

use burn::tensor::backend::AutodiffBackend;

use super::baked::Baked;
use super::checkpoint::{
    load_cold, load_hot, load_magnet, load_pool, load_reservoir, AdamRecord, Trained,
};
use super::config::{InitMode, InitSection, PoolCarry, TrainConfig};
use super::panel::{clone_stem, PanelState};
use crate::rl::model::RlModel;

/// The weights file a hot checkpoint holds, without the extension the recorder appends.
const HOT_MODEL: &str = "model";
/// A run's clone archive, relative to its directory.
const POOL_DIR: &str = "pool";

/// What `[init] from` turned out to point at.
///
/// Read off the contents rather than declared, because the three are already distinguishable on
/// disk and a `kind = ` field would be one more thing a `.toml` could get wrong about a path it
/// also has to get right.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// A `hot-<batch>/` directory. The only source a warm init or a pool carry can use.
    Hot {
        dir: PathBuf,
        /// The run it belongs to, when it is still where it was written — `<run>/checkpoints/hot-*`.
        /// `None` is a checkpoint that was moved, which costs the compatibility check and the pool,
        /// not the weights.
        run: Option<PathBuf>,
    },
    /// A `models/<name>/` directory, already schema-checked by [`Baked::load`].
    Baked(Box<Baked>),
    /// A loose `.mpk`, carrying nothing but tensors.
    Weights(PathBuf),
}

impl Source {
    /// Classifies `from`.
    pub fn resolve(from: &Path) -> Result<Self, String> {
        // Said first, because a hot checkpoint is pruned by the run that wrote it (`keep_hot`) —
        // the batch a `.toml` names today is gone tomorrow, and "not a checkpoint directory" would
        // send the reader looking at the wrong thing entirely.
        if !from.exists() {
            return Err(format!(
                "{} does not exist — a hot checkpoint is pruned once [checkpoint] keep_hot newer \
                 ones exist, so check which batches are still in the run's checkpoints/",
                from.display()
            ));
        }
        if from.is_dir() {
            if from.join(HOT_MODEL).with_extension("mpk").is_file() {
                return Ok(Source::Hot {
                    dir: from.to_path_buf(),
                    run: source_run(from),
                });
            }
            if from.join("meta.toml").is_file() {
                let (root, name) = split_baked(from)?;
                return Ok(Source::Baked(Box::new(Baked::load(&root, &name)?)));
            }
            return Err(format!(
                "{} is a directory but holds neither model.mpk (a hot checkpoint) nor meta.toml \
                 (a baked model)",
                from.display()
            ));
        }
        if from.extension().is_some_and(|ext| ext == "mpk") {
            if !from.is_file() {
                return Err(format!("{} does not exist", from.display()));
            }
            return Ok(Source::Weights(from.to_path_buf()));
        }
        Err(format!(
            "{} is not a hot checkpoint directory, a baked model directory, or a .mpk",
            from.display()
        ))
    }

    /// The path handed to the recorder, without the `.mpk` it appends.
    fn stem(&self) -> PathBuf {
        match self {
            Source::Hot { dir, .. } => dir.join(HOT_MODEL),
            Source::Baked(baked) => baked.weights(),
            Source::Weights(path) => path.with_extension(""),
        }
    }
}

/// `<run>` for a `<run>/checkpoints/hot-<batch>` that is still in place.
fn source_run(hot: &Path) -> Option<PathBuf> {
    let run = hot
        .parent()
        .filter(|p| p.ends_with("checkpoints"))?
        .parent()?;
    run.join("config.toml").is_file().then(|| run.to_path_buf())
}

/// `models/<name>` split the way [`Baked::load`] wants it.
fn split_baked(dir: &Path) -> Result<(PathBuf, String), String> {
    let name = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "{} has no directory name to read a model name off",
                dir.display()
            )
        })?;
    let root = dir.parent().unwrap_or(Path::new(".")).to_path_buf();
    Ok((root, name.to_string()))
}

/// Everything an init produced, for the loop to install.
///
/// Every field past `model` is `None` under a cold init — which is the mode's definition, not a
/// failure to find them.
pub struct Loaded<B: AutodiffBackend> {
    pub model: RlModel<B>,
    /// Boxed, and so is the magnet below: an [`AdamRecord`] is a moment pair per parameter tensor,
    /// so a `Loaded` holding two of them inline is a stack frame the size of three models. It
    /// overflows a debug test thread before it reaches the loop, and moving it by pointer costs an
    /// allocation on a path that runs once per run.
    pub optimizer: Option<Box<AdamRecord<B>>>,
    pub magnet: Option<Box<Trained<B>>>,
    pub reservoir: Option<Vec<u8>>,
    /// Already narrowed to what was copied, so it can go straight into
    /// [`super::panel::Panel::restore`].
    pub panel: Option<PanelState>,
    /// The clones copied into the new run, ascending.
    pub clones: Vec<u64>,
    /// What the caller should print. Collected rather than printed here because this module is
    /// also called from tests, and a check that reports by side effect cannot be asserted on.
    pub notes: Vec<String>,
}

/// Loads `init` into `model`, copying whatever the pool carry asks for into `pool_dir`.
///
/// `magnet` is a freshly built network for the warm restore to load into, and `None` in a run with
/// no magnet — in which case a magnet in the source is left where it is rather than being an error:
/// what a run runs is its own `.toml`'s business, and §1.5.1b turned off is a different algorithm,
/// not a lost payload.
pub fn load<B: AutodiffBackend>(
    init: &InitSection,
    config: &TrainConfig,
    model: RlModel<B>,
    magnet: Option<RlModel<B>>,
    pool_dir: &Path,
    device: &B::Device,
) -> Result<Loaded<B>, String> {
    let source = Source::resolve(&init.from)?;
    let carry = init.pool_carry();
    let mut notes = check(&source, config)?;

    if init.mode == InitMode::Warm && !matches!(source, Source::Hot { .. }) {
        return Err(format!(
            "{} carries weights but no optimizer state, so it can only serve a cold init — point \
             `mode = \"warm\"` at a hot checkpoint directory, or ask for `mode = \"cold\"`",
            init.from.display()
        ));
    }
    if carry != PoolCarry::Empty && !matches!(source, Source::Hot { .. }) {
        return Err(format!(
            "{} is not a hot checkpoint, so there is no pool to carry — drop `pool` or set it to \
             \"empty\"",
            init.from.display()
        ));
    }

    let mut loaded = Loaded {
        model: load_cold(model, &source.stem(), device)?,
        optimizer: None,
        magnet: None,
        reservoir: None,
        panel: None,
        clones: Vec::new(),
        notes: Vec::new(),
    };

    if let Source::Hot { dir, run } = &source {
        if init.mode == InitMode::Warm {
            let warmed = warm_payload::<B>(dir, config, magnet, device)?;
            loaded.optimizer = Some(warmed.optimizer);
            loaded.magnet = warmed.magnet;
            loaded.reservoir = warmed.reservoir;
            notes.extend(warmed.notes);
        }
        if carry != PoolCarry::Empty {
            let run = run.as_ref().ok_or_else(|| {
                format!(
                    "{} is not inside a run directory, so its pool/ cannot be found — a checkpoint \
                     moved away from its run can still serve the weights, not the panel",
                    dir.display()
                )
            })?;
            let (panel, clones) = carry_pool(dir, run, pool_dir, carry)?;
            notes.push(format!(
                "pool carried ({carry:?}) — {} clone(s), {} rating(s)",
                clones.len(),
                panel.ratings.table().len(),
            ));
            // A history slot draws from the archive minus what the slots already hold, and a
            // partial carry's archive *is* what the slots hold. So slots added beyond the carried
            // set can only be filled by fresh clones, one per `clone_every` — the run asks for more
            // adversaries and gets copies of itself, staggered. Said here because the pool reaches
            // the requested size eventually and nothing downstream would report how.
            let asked = config.pool.best_slots + config.pool.history_slots;
            if carry == PoolCarry::Partial && asked > clones.len() {
                notes.push(format!(
                    "this run asks for {asked} slots and {} were carried — the {} extra will fill \
                     from fresh clones at one per {} batches, not from history, because a partial \
                     carry's archive holds only what was in a slot. Carry `full` to fill them from \
                     the source's archive instead",
                    clones.len(),
                    asked - clones.len(),
                    config.pool.clone_every,
                ));
            }
            loaded.panel = Some(panel);
            loaded.clones = clones;
        }
    }

    loaded.notes = notes;
    Ok(loaded)
}

/// The warm half of a [`Loaded`], before it is folded into one.
struct Warm<B: AutodiffBackend> {
    optimizer: Box<AdamRecord<B>>,
    magnet: Option<Box<Trained<B>>>,
    reservoir: Option<Vec<u8>>,
    notes: Vec<String>,
}

/// Reads a hot checkpoint's optimizer and magnet.
///
/// Its own function for a reason that is about the machine and not the design: a frame holding the
/// tuple `load_hot` returns *and* the `Loaded` being built is three model-sized values deep, which
/// overflows a debug thread's stack. Split, each frame holds one and pops.
fn warm_payload<B: AutodiffBackend>(
    dir: &Path,
    config: &TrainConfig,
    magnet: Option<RlModel<B>>,
    device: &B::Device,
) -> Result<Warm<B>, String> {
    // Through `load_hot` rather than the record alone: the weights are re-read and dropped, which
    // costs one file read and keeps this off a private spelling of how a record is stored.
    let (_, optimizer, saved) = load_hot::<B>(
        dir,
        RlModel::<B>::new(&config.model, &Default::default(), device),
        device,
    )?;
    let mut warm = Warm {
        optimizer: Box::new(optimizer),
        magnet: None,
        reservoir: None,
        notes: vec![format!(
            "warm — AdamW's moments as of batch {} of the source run",
            saved.batch
        )],
    };

    match magnet {
        Some(shell) => match load_magnet(dir, shell, device)? {
            Some(restored) => {
                warm.magnet = Some(Box::new(restored));
                warm.reservoir = load_reservoir(dir)?;
                warm.notes.push(match &warm.reservoir {
                    Some(bytes) => format!("magnet carried, reservoir {} bytes", bytes.len()),
                    // The rolling autosave writes no reservoir (§1.5.5), so this is the common
                    // case and not an exotic one. `loss/kl_magnet` then falls for bookkeeping
                    // reasons — the average is re-taken over a stream that starts here — and
                    // reading that fall as an improvement is the whole reason it is said out loud.
                    None => "magnet carried, but no reservoir in this checkpoint — the average \
                             policy restarts from empty and loss/kl_magnet will fall for that \
                             reason alone"
                        .to_string(),
                });
            }
            None => warm.notes.push(
                "no magnet in the source checkpoint — this run's magnet starts fresh and is \
                 seeded like a fresh run's"
                    .to_string(),
            ),
        },
        None => warm
            .notes
            .push("[magnet] is off in this run, so the source's magnet is left behind".into()),
    }
    Ok(warm)
}

/// Restores the source's panel state, narrows it to the carry, and copies the clones it keeps.
fn carry_pool(
    hot: &Path,
    run: &Path,
    destination: &Path,
    carry: PoolCarry,
) -> Result<(PanelState, Vec<u64>), String> {
    let encoded = load_pool(hot)?.ok_or_else(|| {
        format!(
            "{} carries no pool.json — the source run had no pool, so there is nothing to carry",
            hot.display()
        )
    })?;
    let mut state: PanelState = serde_json::from_str(&encoded)
        .map_err(|err| format!("failed to decode the source pool state: {err}"))?;

    let clones = match carry {
        PoolCarry::Empty => Vec::new(),
        PoolCarry::Partial => {
            let kept = state.pool.restrict_to_slots();
            state.ratings.retain_clones(&kept);
            kept
        }
        PoolCarry::Full => state.pool.archive().to_vec(),
    };
    // At the new run's batch 0, since that is where the run this is feeding starts. Without it the
    // carried slots' tenure is measured against the source's batch numbering and reads as a grace
    // period that never expires — see `Pool::readmit_slots`.
    state.pool.readmit_slots(&state.ratings, 0);

    let archive = run.join(POOL_DIR);
    fs::create_dir_all(destination)
        .map_err(|err| format!("failed to create {}: {err}", destination.display()))?;
    for batch in &clones {
        let from = clone_stem(&archive, *batch).with_extension("mpk");
        let to = clone_stem(destination, *batch).with_extension("mpk");
        fs::copy(&from, &to).map_err(|err| {
            format!(
                "failed to copy clone {} to {}: {err}",
                from.display(),
                to.display()
            )
        })?;
    }
    Ok((state, clones))
}

/// Compares the source against the run that is about to load it, as far as each kind of source
/// allows. Returns what the caller should say about it.
fn check(source: &Source, config: &TrainConfig) -> Result<Vec<String>, String> {
    match source {
        Source::Baked(baked) => {
            // `Baked::load` already refused a schema mismatch; the sizes are this module's
            // business, and unlike a pool member's they are not free to differ.
            same_model(&baked.meta.model, &config.model, &baked.dir)?;
            Ok(vec![format!(
                "source: baked model `{}` (schema checked)",
                baked.name
            )])
        }
        Source::Hot {
            dir,
            run: Some(run),
        } => {
            let theirs = TrainConfig::from_file(&run.join("config.toml"))?;
            same_model(&theirs.model, &config.model, run)?;
            if theirs.text_embeddings != config.text_embeddings {
                return Err(format!(
                    "{} trained on text embeddings {:?} and this run reads {:?} — §1.2.9's tables \
                     are not in a checkpoint, so the weights would be read against features they \
                     were never trained on and nothing downstream could tell",
                    run.display(),
                    theirs.text_embeddings,
                    config.text_embeddings,
                ));
            }
            Ok(vec![format!(
                "source: {} — [model] and text embeddings match {}",
                dir.display(),
                run.join("config.toml").display()
            )])
        }
        Source::Hot { dir, run: None } => Ok(vec![format!(
            "source: {} — not inside a run directory, so its [model] table could not be checked \
             against this one",
            dir.display()
        )]),
        Source::Weights(path) => Ok(vec![format!(
            "source: {} — a loose record says nothing about the sizes or the schema it was trained \
             at, and is loaded on trust",
            path.display()
        )]),
    }
}

fn same_model(
    theirs: &crate::rl::model::config::ModelConfig,
    ours: &crate::rl::model::config::ModelConfig,
    origin: &Path,
) -> Result<(), String> {
    if theirs == ours {
        return Ok(());
    }
    let render = |config: &crate::rl::model::config::ModelConfig| {
        toml::to_string(config).unwrap_or_else(|err| format!("<unprintable: {err}>"))
    };
    Err(format!(
        "the source's [model] table differs from this run's, and burn validates the length of a \
         layer Vec rather than the shapes inside it — so a mismatch here is as likely to load \
         quietly into a wrong network as it is to fail.\n{} says:\n{}\nthis run says:\n{}",
        origin.display(),
        render(theirs),
        render(ours),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::model::config::ModelConfig;
    use crate::rl::text_embedding::TextEmbeddings;
    use burn::backend::{Autodiff, NdArray};

    type B = Autodiff<NdArray>;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-init-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch");
        dir
    }

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

    /// The `.toml` a run clones, at `small()`'s sizes and no text tables — the two things
    /// [`check`] compares.
    fn config_text(name: &str) -> String {
        format!(
            "text_embeddings = \"\"\n\
             [run]\nseed = 1\nname = \"{name}\"\nroot = \"runs\"\n\
             [decks]\nroot = \"decks\"\ndb = \"tutorial\"\npure_mirror = 0.0\nmirror = 0.0\n\
             [rollout]\nenvs = 1\nopponents = [\"r\"]\nframes_per_batch = 1\nbatches = 1\n\
             [step]\nlearning_rate = 3e-4\nvalue_coeff = 0.5\nentropy_coeff = 0.01\n\
             residual_decay = 1e-4\ngrad_clip = 0.5\nmicro_batch = 8\n\
             [model]\nd_model = 24\nnum_blocks = 1\nnum_heads = 2\nd_ff = 32\nd_id = 8\n\
             d_head_emb = 4\nd_head_hidden = 8\n"
        )
    }

    /// A run directory with one hot checkpoint in it, the shape every non-trivial case needs.
    fn run_with_checkpoint(root: &Path, name: &str, batch: u64) -> PathBuf {
        let run = root.join(name);
        let checkpoints = run.join("checkpoints");
        fs::create_dir_all(&checkpoints).expect("checkpoints");
        fs::write(run.join("config.toml"), config_text(name)).expect("config");
        let learner = super::super::update::Learner::<B>::new(Default::default());
        super::super::checkpoint::save_hot(
            &checkpoints,
            &model(&small()),
            learner.optimizer_record(),
            None,
            super::super::checkpoint::LoopState {
                batch,
                games_started: 10,
                elapsed_seconds: 1.0,
                stage: 0,
            },
            Default::default(),
            2,
        )
        .expect("save")
    }

    fn config_for(dir: &Path, name: &str) -> TrainConfig {
        let path = dir.join(format!("{name}.toml"));
        fs::write(&path, config_text(name)).expect("config");
        TrainConfig::from_file(&path).expect("parse")
    }

    #[test]
    fn a_hot_checkpoint_in_a_run_is_recognized_with_its_run() {
        let dir = scratch("resolve-hot");
        let hot = run_with_checkpoint(&dir, "src", 12);
        match Source::resolve(&hot).expect("resolve") {
            Source::Hot { run, .. } => assert_eq!(run, Some(dir.join("src"))),
            other => panic!("{other:?}"),
        }
    }

    /// A checkpoint copied out of its run still serves the weights. It is the compatibility check
    /// and the pool that need the run, and losing those is a smaller loss than refusing the load.
    #[test]
    fn a_hot_checkpoint_moved_out_of_its_run_still_resolves() {
        let dir = scratch("resolve-moved");
        let hot = run_with_checkpoint(&dir, "src", 3);
        let moved = dir.join("elsewhere");
        fs::create_dir_all(&moved).expect("dir");
        for file in fs::read_dir(&hot).expect("read") {
            let file = file.expect("entry").path();
            fs::copy(&file, moved.join(file.file_name().expect("name"))).expect("copy");
        }
        match Source::resolve(&moved).expect("resolve") {
            Source::Hot { run: None, .. } => {}
            other => panic!("{other:?}"),
        }
    }

    /// The mode/source pairing is the one error a user can make from the `.toml` alone, so it is
    /// refused before any file is read rather than surfacing as a missing `optim.mpk`.
    #[test]
    fn a_warm_init_off_a_baked_model_is_refused() {
        let dir = scratch("warm-baked");
        let models = dir.join("models");
        let baked = models.join("veteran");
        fs::create_dir_all(&baked).expect("dir");
        super::super::baked::Baked::write_meta(
            &baked,
            &super::super::baked::BakedMeta::current(small()),
        )
        .expect("meta");
        super::super::checkpoint::save_cold(&model(&small()), &baked.join("weights"))
            .expect("weights");

        let config = config_for(&dir, "target");
        let init = InitSection {
            mode: InitMode::Warm,
            from: baked.clone(),
            pool: None,
        };
        let err = load::<B>(
            &init,
            &config,
            model(&small()),
            None,
            &dir.join("pool"),
            &Default::default(),
        )
        .map(|_| ())
        .expect_err("a baked model has no optimizer state");
        assert!(err.contains("cold init"), "{err}");
    }

    /// The silent case the module docs name: a `[model]` table that differs loads *successfully*
    /// into the wrong network, so the refusal has to happen here or not at all.
    #[test]
    fn a_source_trained_at_other_sizes_is_refused() {
        let dir = scratch("model-mismatch");
        let hot = run_with_checkpoint(&dir, "src", 5);
        let mut config = config_for(&dir, "target");
        config.model.d_ff = 64;

        let init = InitSection {
            mode: InitMode::Cold,
            from: hot,
            pool: None,
        };
        let err = load::<B>(
            &init,
            &config,
            model(&config.model),
            None,
            &dir.join("pool"),
            &Default::default(),
        )
        .map(|_| ())
        .expect_err("a differing [model] table must be refused");
        assert!(err.contains("[model] table differs"), "{err}");
    }

    /// §1.2.9's tables are not in a checkpoint, so the same weights read different features under
    /// a different artifact — and nothing downstream can tell.
    #[test]
    fn a_source_trained_on_other_text_embeddings_is_refused() {
        let dir = scratch("text-mismatch");
        let hot = run_with_checkpoint(&dir, "src", 5);
        let mut config = config_for(&dir, "target");
        config.text_embeddings = "auxiliaries/text_embeddings/out/text_embeddings.json".to_string();

        let init = InitSection {
            mode: InitMode::Cold,
            from: hot,
            pool: None,
        };
        let err = load::<B>(
            &init,
            &config,
            model(&small()),
            None,
            &dir.join("pool"),
            &Default::default(),
        )
        .map(|_| ())
        .expect_err("a differing text artifact must be refused");
        assert!(err.contains("§1.2.9"), "{err}");
    }

    /// A cold init takes the weights and nothing else — including from a source that has an
    /// optimizer sitting right beside them.
    #[test]
    fn a_cold_init_off_a_hot_checkpoint_leaves_the_optimizer_behind() {
        let dir = scratch("cold-hot");
        let hot = run_with_checkpoint(&dir, "src", 9);
        let config = config_for(&dir, "target");

        let loaded = load::<B>(
            &InitSection {
                mode: InitMode::Cold,
                from: hot,
                pool: None,
            },
            &config,
            model(&small()),
            None,
            &dir.join("pool"),
            &Default::default(),
        )
        .expect("cold init");

        assert!(loaded.optimizer.is_none());
        assert!(loaded.magnet.is_none());
        assert!(loaded.panel.is_none());
    }

    /// On its own thread with an explicit stack, and only this one: reading an optimizer record
    /// back is the deepest frame in the module, and an unoptimized build overflows the harness's
    /// default on it. The release binary never comes close — this buys the assertion, not the
    /// feature.
    #[test]
    fn a_warm_init_carries_the_optimizer() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let dir = scratch("warm-hot");
                let hot = run_with_checkpoint(&dir, "src", 9);
                let config = config_for(&dir, "target");

                let loaded = load::<B>(
                    &InitSection {
                        mode: InitMode::Warm,
                        from: hot,
                        pool: Some(PoolCarry::Empty),
                    },
                    &config,
                    model(&small()),
                    None,
                    &dir.join("pool"),
                    &Default::default(),
                )
                .expect("warm init");

                assert!(loaded.optimizer.is_some());
                assert!(loaded.clones.is_empty());
            })
            .expect("spawn")
            .join()
            .expect("warm init thread");
    }

    /// The carry defaults are per-mode, and they are the difference between a continuation that
    /// has opponents and one that spends its first hundreds of batches making some.
    #[test]
    fn the_pool_carry_defaults_to_the_slots_when_warm_and_to_nothing_when_cold() {
        let warm = InitSection {
            mode: InitMode::Warm,
            from: PathBuf::from("x"),
            pool: None,
        };
        let cold = InitSection {
            mode: InitMode::Cold,
            from: PathBuf::from("x"),
            pool: None,
        };
        assert_eq!(warm.pool_carry(), PoolCarry::Partial);
        assert_eq!(cold.pool_carry(), PoolCarry::Empty);
        assert_eq!(
            InitSection {
                pool: Some(PoolCarry::Full),
                ..warm
            }
            .pool_carry(),
            PoolCarry::Full
        );
    }

    /// A pool carry is a file copy, and a partial one must not leave the archive promising draws
    /// whose weights were not copied — the failure would land hours in, at the first refresh that
    /// happens to draw one.
    #[test]
    fn a_partial_carry_copies_the_slots_and_narrows_the_archive_to_them() {
        use super::super::pool::{Permanent, Pool, PoolConfig};
        use super::super::rating::{Entry, OpponentId, RatingConfig, RatingTable};

        let dir = scratch("carry-partial");
        let run = dir.join("src");
        let checkpoints = run.join("checkpoints");
        let archive = run.join(POOL_DIR);
        fs::create_dir_all(&checkpoints).expect("checkpoints");
        fs::create_dir_all(&archive).expect("archive");
        fs::write(run.join("config.toml"), config_text("src")).expect("config");

        // Four clones written, two of them still in slots: the case the narrowing exists for.
        let mut pool = Pool::new(
            PoolConfig {
                best_slots: 1,
                history_slots: 1,
                ..Default::default()
            },
            vec![Permanent::heuristic(crate::players::PlayerCode::R).pinned()],
        )
        .expect("pool");
        let mut ratings = RatingTable::new(RatingConfig::default()).expect("ratings");
        for batch in [10u64, 20, 30, 40] {
            super::super::checkpoint::save_cold(&model(&small()), &clone_stem(&archive, batch))
                .expect("clone");
            ratings.ensure(OpponentId::Pool(batch), Entry::fresh());
            pool.admit_clone(batch, &mut ratings);
        }
        let in_slots = pool.slot_batches();
        assert_eq!(in_slots.len(), 2, "the fixture wants two occupied slots");

        let hot = super::super::checkpoint::save_hot(
            &checkpoints,
            &model(&small()),
            super::super::update::Learner::<B>::new(Default::default()).optimizer_record(),
            None,
            super::super::checkpoint::LoopState {
                batch: 200,
                games_started: 10,
                elapsed_seconds: 1.0,
                stage: 0,
            },
            super::super::checkpoint::SideState {
                pool: Some(
                    &serde_json::to_string(&PanelState {
                        pool: pool.clone(),
                        ratings: ratings.clone(),
                    })
                    .expect("encode"),
                ),
                ..Default::default()
            },
            2,
        )
        .expect("save");

        let destination = dir.join("dst").join(POOL_DIR);
        let (state, clones) =
            carry_pool(&hot, &run, &destination, PoolCarry::Partial).expect("carry");

        assert_eq!(clones, in_slots);
        assert_eq!(state.pool.archive(), in_slots.as_slice());
        for batch in [10u64, 20, 30, 40] {
            let copied = clone_stem(&destination, batch)
                .with_extension("mpk")
                .is_file();
            assert_eq!(copied, in_slots.contains(&batch), "clone {batch}");
            let rated = state.ratings.get(&OpponentId::Pool(batch)).is_some();
            assert_eq!(rated, in_slots.contains(&batch), "rating {batch}");
        }
    }

    /// The full carry is the other end of the same axis: every clone the source could draw, the
    /// new run can draw too.
    #[test]
    fn a_full_carry_copies_the_whole_archive() {
        use super::super::pool::{Permanent, Pool, PoolConfig};
        use super::super::rating::{Entry, OpponentId, RatingConfig, RatingTable};

        let dir = scratch("carry-full");
        let run = dir.join("src");
        let checkpoints = run.join("checkpoints");
        let archive = run.join(POOL_DIR);
        fs::create_dir_all(&checkpoints).expect("checkpoints");
        fs::create_dir_all(&archive).expect("archive");
        fs::write(run.join("config.toml"), config_text("src")).expect("config");

        let mut pool = Pool::new(
            PoolConfig {
                best_slots: 1,
                history_slots: 1,
                ..Default::default()
            },
            vec![Permanent::heuristic(crate::players::PlayerCode::R).pinned()],
        )
        .expect("pool");
        let mut ratings = RatingTable::new(RatingConfig::default()).expect("ratings");
        for batch in [10u64, 20, 30, 40] {
            super::super::checkpoint::save_cold(&model(&small()), &clone_stem(&archive, batch))
                .expect("clone");
            ratings.ensure(OpponentId::Pool(batch), Entry::fresh());
            pool.admit_clone(batch, &mut ratings);
        }

        let hot = super::super::checkpoint::save_hot(
            &checkpoints,
            &model(&small()),
            super::super::update::Learner::<B>::new(Default::default()).optimizer_record(),
            None,
            super::super::checkpoint::LoopState {
                batch: 200,
                games_started: 10,
                elapsed_seconds: 1.0,
                stage: 0,
            },
            super::super::checkpoint::SideState {
                pool: Some(
                    &serde_json::to_string(&PanelState {
                        pool: pool.clone(),
                        ratings: ratings.clone(),
                    })
                    .expect("encode"),
                ),
                ..Default::default()
            },
            2,
        )
        .expect("save");

        let destination = dir.join("dst").join(POOL_DIR);
        let (state, clones) = carry_pool(&hot, &run, &destination, PoolCarry::Full).expect("carry");

        assert_eq!(clones, vec![10, 20, 30, 40]);
        assert_eq!(state.pool.archive(), &[10, 20, 30, 40]);
        for batch in [10u64, 20, 30, 40] {
            assert!(clone_stem(&destination, batch)
                .with_extension("mpk")
                .is_file());
        }
    }
}
