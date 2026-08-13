//! Does the whitening of §1.2.9's text blocks decide *at which depth* an attack head can form?
//!
//! `long_v4` grew an attack head in block 0 — `b0h2`, 1.65x chance by batch 25 and 1.80x by 50,
//! before its block 1 had moved at all. `long_v5` never grew one: no block-0 head exceeds 1.21x on
//! the attack family at any point of the run, and all the selection landed in block 1 instead. The
//! two runs differ in seed, in environment, and in the whitening — and a difference already visible
//! at batch 25, with the learning-rate warmup only just finished, is more plausibly in the features
//! than in where SGD wandered.
//!
//! That is testable **without training anything**, which no comparison between the two runs can be.
//! Same `init_seed`, same encoder init (seeded here explicitly, since the trainer does not seed
//! burn's global RNG — see below), same frames, and the text blocks as the only difference. If the
//! depth-1 attack preference is present at init with the unwhitened blocks and absent with the
//! whitened ones, the mechanism is the features and not the trajectory.
//!
//! ```text
//! uv run --no-project --with numpy auxiliaries/text_embeddings/unwhiten.py
//! cargo run --release --features rl-model --example text_scale_ablation -- \
//!     --from runs/<run>/checkpoints/<hot-dir>
//! ```
//!
//! **Why `--from` a checkpoint when the models under test are untrained.** The frames have to be
//! board states the question is about. An init model plays at random, and its boards are not ones
//! either run ever trained on — so the rollout is driven by the checkpoint, and the two *init*
//! models are then read on those same frames. Neither arm plays; both only look.
//!
//! **The encoder init is not reproducible from the config**, only the frozen tables are
//! (`[model] init_seed`, via `FrozenTables`). Nothing in `src/rl/train` seeds burn's global RNG, so
//! the blocks of any two runs start from different weights whatever the config says. Here both arms
//! are seeded to the same value per repeat, which is what makes them a paired comparison rather
//! than two draws.

#[cfg(not(feature = "rl-model"))]
fn main() {
    eprintln!("build with --features rl-model (or rl-model-cuda) --release");
}

#[cfg(feature = "rl-model")]
fn main() {
    use std::path::{Path, PathBuf};

    use burn::backend::NdArray;
    use burn::prelude::Backend;

    use deckgym::players::parse_player_code;
    use deckgym::rl::model::input::ModelInput;
    use deckgym::rl::model::introspect::FAMILY_NAMES;
    use deckgym::rl::model::RlModel;
    use deckgym::rl::text_embedding::TextEmbeddings;
    use deckgym::rl::train::checkpoint::{load_cold, LoopState};
    use deckgym::rl::train::deck_db::DeckDb;
    use deckgym::rl::train::diagnostics::probe_points;
    use deckgym::rl::train::opponent::OpponentModels;
    use deckgym::rl::train::rollout::{Collector, RolloutConfig};
    use deckgym::rl::train::sampler::{DeckSampler, DeckSource};
    use deckgym::rl::train::TrainConfig;

    type B = NdArray<f32>;

    fn flag(name: &str) -> Option<String> {
        std::env::args()
            .skip_while(|arg| arg != name)
            .nth(1)
            .filter(|value| !value.starts_with("--"))
    }

    env_logger::init();
    let Some(from) = flag("--from") else {
        eprintln!(
            "usage: text_scale_ablation --from <hot-checkpoint-dir> [--repeats <n>] \
             [--unwhitened <json>]"
        );
        std::process::exit(2);
    };
    let from = PathBuf::from(&from);
    let repeats: u64 = flag("--repeats")
        .and_then(|value| value.parse().ok())
        .unwrap_or(6);
    let unwhitened = flag("--unwhitened").unwrap_or_else(|| {
        "auxiliaries/text_embeddings/out/text_embeddings_unwhitened.json".to_string()
    });

    let raw = std::fs::read_to_string(from.join("loop.json")).expect("loop.json");
    let state: LoopState = serde_json::from_str(&raw).expect("loop.json");
    let config_path = from
        .parent()
        .and_then(|checkpoints| checkpoints.parent())
        .map(|run| run.join("config.toml"))
        .filter(|config| config.is_file())
        .expect("a run config two levels up");
    let config = TrainConfig::from_file(&config_path).expect("config");
    let device = Default::default();

    let arms = [
        ("whitened", config.text_embeddings().expect("whitened")),
        (
            "unwhitened",
            TextEmbeddings::from_json_file(Path::new(&unwhitened)).unwrap_or_else(|err| {
                panic!("{err}\nbuild it with auxiliaries/text_embeddings/unwhiten.py")
            }),
        ),
    ];

    // Frames the question is about: the paused run's own stage, played by the paused run's own
    // weights.
    let stage = &config.curriculum.stages[state.stage];
    let sources: Vec<DeckSource> = stage
        .mix
        .iter()
        .map(|spec| DeckSource {
            db: DeckDb::load(&config.decks.root.join(&spec.db)).expect("deck db"),
            share: spec.share,
            archetypes: spec.archetypes.clone(),
        })
        .collect();
    let driver = load_cold::<B>(
        RlModel::new(&config.model, &arms[0].1, &device),
        &from.join("model.mpk"),
        &device,
    )
    .expect("checkpoint");
    let mut collector = Collector::new(
        DeckSampler::mixed(sources, stage.pure_mirror, stage.mirror).expect("sampler"),
        RolloutConfig {
            envs: config.rollout.envs,
            opponents: config
                .rollout
                .opponents
                .iter()
                .map(|code| parse_player_code(code).expect("opponent code"))
                .collect(),
            max_crashes_per_batch: 64,
        },
        config.run.seed ^ (0x7EA7_u64 << 16),
        None,
    )
    .expect("collector");
    let (episodes, _) = collector
        .collect::<B>(
            &driver,
            &OpponentModels::new(),
            &config.model,
            &device,
            4096,
            0,
        )
        .expect("rollout");
    let points = probe_points(&episodes, 64);
    let input = ModelInput::from_points(&points, &config.model, &device);

    println!(
        "{} — batch {}, stage {:?}, {} frames, {repeats} encoder inits\n",
        from.display(),
        state.batch,
        stage.name,
        points.len()
    );

    let attack = FAMILY_NAMES
        .iter()
        .position(|family| *family == "attack")
        .expect("attack family");
    println!("block-0 attack focus, best of the six heads (x chance), per encoder init");
    println!("  seed        whitened      unwhitened");
    let mut totals = [0.0f64; 2];
    for repeat in 0..repeats {
        let mut best = [0.0f64; 2];
        for (arm, (_, embeddings)) in arms.iter().enumerate() {
            // Both arms from the same seed, so the encoder they start from is the same weights and
            // the only difference left is the frozen descriptors the embeddings feed.
            B::seed(&device, 0xB10C_0000 + repeat);
            let model = RlModel::<B>::new(&config.model, embeddings, &device);
            let stats = model.attention_stats(&input);
            best[arm] = stats
                .heads
                .iter()
                .filter(|head| head.block == 0)
                .map(|head| head.family_mass[attack] / stats.family_share[attack])
                .fold(0.0f64, f64::max);
            totals[arm] += best[arm];
        }
        println!("  {repeat:<10}  {:6.3}          {:6.3}", best[0], best[1]);
    }
    println!(
        "  {:<10}  {:6.3}          {:6.3}",
        "mean",
        totals[0] / repeats as f64,
        totals[1] / repeats as f64
    );
}
