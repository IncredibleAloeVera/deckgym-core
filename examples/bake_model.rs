//! Promotes a checkpoint into a §1.5.2 baked model, and verifies the ones already there.
//!
//! A baked model is a directory, not a file ([`deckgym::rl::train::baked`]): the weights alone do
//! not say what network to build them into, nor which observation layout they were trained to read.
//! This writes the second half and proves the pair loads before claiming it does — a `meta.toml`
//! whose `[model]` table is a guess produces a model that loads into the wrong shape, or worse,
//! into the right shape on the wrong projection.
//!
//! ```text
//! cargo run --release --features rl-model --example bake_model -- --verify
//! cargo run --release --features rl-model --example bake_model -- \
//!     --from runs/<run>/checkpoints/<hot-dir>
//! cargo run --release --features rl-model --example bake_model -- \
//!     --from runs/my_run/checkpoints/loose.mpk --name my_model --note "MMD prototype"
//! ```
//!
//! **`--from` a hot checkpoint directory, not a loose `.mpk`, whenever there is one.** A baked
//! model's `meta.toml` carries a rating it did not earn in this file — it earned it in the run that
//! produced the weights, and that rating is the entire reason the file exists beyond the schema
//! check. Retyping it by hand is not a chore, it is the one error in this whole path that nothing
//! downstream can catch: a wrong rating does not fail to load, it silently bends the next run's elo
//! curve. Everything needed is already in the checkpoint, so a directory `--from` reads it instead:
//! the batch out of `loop.json`, the learner's rating, deviation and game count out of `pool.json`,
//! and the `[model]` table out of the run's own cloned `config.toml` — the only one guaranteed to be
//! what the weights were trained at. A loose `.mpk` still works and still bakes at the default
//! rating, because a file on its own genuinely does not know any of this.
//!
//! `--config` names the `.toml` whose `[model]` table the weights were trained at (default: the
//! run's own on a directory `--from`, `config/default.toml` otherwise); `--root` the models
//! directory (default `models`). The source file is **copied, not moved**, so a wrong `--config`
//! costs a retry rather than the weights.

use std::path::{Path, PathBuf};

use burn::backend::NdArray;

use deckgym::rl::text_embedding::TextEmbeddings;
use deckgym::rl::train::baked::{load_model, Baked, BakedMeta, BakedRating};
use deckgym::rl::train::checkpoint::LoopState;
use deckgym::rl::train::panel::PanelState;
use deckgym::rl::train::TrainConfig;

/// The verification is a load, which needs no gradients — and on CPU, because a few megabytes of
/// weights read once is not what a GPU is for.
type B = NdArray<f32>;

fn flag(name: &str) -> Option<String> {
    std::env::args()
        .skip_while(|arg| arg != name)
        .nth(1)
        .filter(|value| !value.starts_with("--"))
}

/// What a hot checkpoint says about the weights beside it.
struct FromRun {
    /// The run's own cloned `config.toml`, when it is still there — a run directory can be moved or
    /// pruned, and a missing clone is a reason to fall back to `--config`, not to refuse.
    config: Option<PathBuf>,
    batch: u64,
    rating: BakedRating,
}

/// What `--from` resolved to.
struct Source {
    weights: PathBuf,
    /// `None` when `--from` named a loose `.mpk`.
    run: Option<FromRun>,
}

impl Source {
    fn resolve(from: &Path) -> Result<Self, String> {
        if from.is_file() {
            return Ok(Source {
                weights: from.to_path_buf(),
                run: None,
            });
        }
        if !from.is_dir() {
            return Err(format!(
                "{} is neither a file nor a directory",
                from.display()
            ));
        }

        // `save_hot` writes the learner's weights under this name with the same recorder
        // `save_cold` uses, so the file is directly bakeable — there is no conversion step here,
        // only a different place to look.
        let weights = from.join("model.mpk");
        if !weights.is_file() {
            return Err(format!(
                "{} holds no model.mpk — --from takes either a hot checkpoint directory or a loose \
                 weights file",
                from.display()
            ));
        }

        let state = read_json::<LoopState>(&from.join("loop.json"))?;
        // A run with `[pool] enabled = false` never rated its learner, so there is nothing to
        // migrate and the default rating is the honest answer — not an error, and not a zero.
        let rating = match read_json::<PanelState>(&from.join("pool.json")) {
            Ok(panel) => {
                let learner = panel.ratings.learner();
                BakedRating {
                    rating: learner.rating.rating,
                    deviation: learner.rating.deviation,
                    // Deliberately *not* migrated. Volatility describes how erratically an entity's
                    // strength moves, and a baked entry never moves — `Baked::entry` builds it with
                    // `drifts: false`, so the field is read by nothing. Carrying over the learner's
                    // (long_v1 ended at 0.91) would put a pathological number in a file people read
                    // to judge a model, implying a drift that cannot happen.
                    volatility: BakedRating::default().volatility,
                    games: learner.games,
                }
            }
            Err(_) => BakedRating::default(),
        };

        // `runs/<name>/checkpoints/hot-<batch>/` — two levels up is the run directory, where
        // `RunDir::create` left the config the run was launched with.
        let config = from
            .parent()
            .and_then(|checkpoints| checkpoints.parent())
            .map(|run| run.join("config.toml"))
            .filter(|config| config.is_file());

        Ok(Source {
            weights,
            run: Some(FromRun {
                config,
                batch: state.batch,
                rating,
            }),
        })
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn main() {
    env_logger::init();
    let root = PathBuf::from(flag("--root").unwrap_or_else(|| "models".to_string()));

    if std::env::args().any(|arg| arg == "--verify") {
        verify(&root);
        return;
    }

    let Some(from) = flag("--from") else {
        eprintln!(
            "usage: bake_model --from <hot-checkpoint-dir|weights.mpk> [--name <name>] \
             [--config <toml>] [--root <dir>] [--note <text>]\n       bake_model --verify \
             [--root <dir>]"
        );
        std::process::exit(2);
    };
    let from = PathBuf::from(&from);
    let source = match Source::resolve(&from) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let config_path = flag("--config")
        .or_else(|| {
            source
                .run
                .as_ref()
                .and_then(|run| run.config.as_ref())
                .map(|config| config.display().to_string())
        })
        .unwrap_or_else(|| "config/default.toml".to_string());
    let config = TrainConfig::from_file(Path::new(&config_path)).expect("config");

    // `<run>_b<batch>` rather than the directory's own stem, which is `hot-00001423` — a name that
    // says which checkpoint it was and not which run, and collides with every other run's.
    let name = flag("--name").unwrap_or_else(|| match &source.run {
        Some(run) => format!("{}_b{}", config.run.name, run.batch),
        None => from
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .expect("--name, or a --from with a file stem"),
    });

    bake(&root, &name, &source, &config_path, &config);
}

fn bake(root: &Path, name: &str, source: &Source, config_path: &str, config: &TrainConfig) {
    let dir = root.join(name);
    if dir.exists() {
        eprintln!(
            "{} already exists — remove it first rather than baking over a model something may \
             already be rated against",
            dir.display()
        );
        std::process::exit(1);
    }
    std::fs::create_dir_all(&dir).expect("model directory");
    std::fs::copy(&source.weights, dir.join("weights.mpk")).expect("copy weights");

    let mut meta = BakedMeta::current(config.model.clone());
    // Separators normalized: this field is tracked, read by people and diffed across machines, and
    // half of a Windows-built path is already forward slashes because it came from `--from`.
    meta.provenance.source = Some(source.weights.display().to_string().replace('\\', "/"));
    meta.provenance.run = Some(config.run.name.clone());
    meta.provenance.created = Some(chrono::Utc::now().format("%Y-%m-%d").to_string());
    meta.provenance.note = flag("--note");
    if let Some(run) = &source.run {
        meta.provenance.batch = Some(run.batch);
        meta.rating = run.rating.clone();
    }
    Baked::write_meta(&dir, &meta).expect("meta");

    // The claim is "this loads", so it is made by loading — not by the two files existing.
    let baked = Baked::load(root, name).expect("baked model");
    match load_model::<B>(&baked, &TextEmbeddings::zeros(), &Default::default()) {
        Ok(_) => {
            println!(
                "baked {} from {} at the [model] table of {config_path}",
                dir.display(),
                source.weights.display()
            );
            // Printed rather than left to be discovered in the file: the rating is the one thing
            // here nothing downstream will ever contradict, so it is the one thing worth reading
            // back to whoever ran the command.
            match &source.run {
                Some(run) => println!(
                    "  migrated from the checkpoint: batch {}, rating {:.1} ± {:.1} over {} rated \
                     games",
                    run.batch, run.rating.rating, run.rating.deviation, run.rating.games
                ),
                None => println!(
                    "  no run beside these weights — baked at the default {:.0} ± {:.0}, which \
                     ranks it against nothing until it plays",
                    meta.rating.rating, meta.rating.deviation
                ),
            }
        }
        Err(err) => {
            std::fs::remove_dir_all(&dir).ok();
            eprintln!(
                "{err}\nthe weights do not fit the [model] table of {config_path} — bake with the \
                 config they were trained at. Nothing was written."
            );
            std::process::exit(1);
        }
    }
}

fn verify(root: &Path) {
    let models = match Baked::discover(root) {
        Ok(models) => models,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    if models.is_empty() {
        println!("no baked models under {}", root.display());
        return;
    }
    let mut failed = 0;
    for baked in &models {
        match load_model::<B>(baked, &TextEmbeddings::zeros(), &Default::default()) {
            Ok(_) => println!(
                "ok    {:<24} d_model={} blocks={} rating={:.0}±{:.0} games={}",
                baked.name,
                baked.meta.model.d_model,
                baked.meta.model.num_blocks,
                baked.meta.rating.rating,
                baked.meta.rating.deviation,
                baked.meta.rating.games,
            ),
            Err(err) => {
                failed += 1;
                println!("FAIL  {:<24} {err}", baked.name);
            }
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
