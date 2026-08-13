//! What a block's attention is worth, in the only unit that settles it: winrate against `er`.
//!
//! The §1.5.6 read-out says `long_v5`'s block 0 writes 4.2x the stream it was given, so it is not
//! near-identity; the drift measurement says its query and key never acquired a direction, so what
//! it looks at is a random walk. Neither says whether the policy would notice losing it. This
//! plays the held-out evaluation with the sublayer disabled and reads the difference.
//!
//! ```text
//! cargo run --release --features rl-model --example block_ablation -- \
//!     --from runs/<run>/checkpoints/<hot-dir>
//! ```
//!
//! Three arms per block, all on the same decks and the same seed, so the only thing that moves is
//! the model:
//!
//! - **baseline** — the checkpoint as saved.
//! - **uniform** — query and key zeroed, so the softmax is exactly uniform over the real tokens.
//!   The block still pools and still writes; it has only stopped choosing where. This is the arm
//!   the drift measurement predicts is free.
//! - **silent** — the output projection zeroed, so the attention sublayer writes nothing and the
//!   block is its feed-forward alone. The control: it should cost something, or the whole sublayer
//!   is dead weight and not merely its pattern.
//!
//! **What this cannot say.** It measures whether the *trained* policy needs the sublayer, not
//! whether the *training* did. A model trained from scratch against a uniform block 0 might be
//! better or worse than this one — the weights here were fitted with the pattern in place, and
//! every other parameter has had 1252 batches to come to depend on it. A large drop falsifies "dead
//! weight" immediately; a small one licenses trying the cheaper architecture, not shipping it.

#[cfg(not(feature = "rl-model"))]
fn main() {
    eprintln!("build with --features rl-model (or rl-model-cuda) --release");
}

#[cfg(feature = "rl-model")]
fn main() {
    use std::path::PathBuf;

    #[cfg(not(feature = "rl-model-cuda"))]
    use burn::backend::NdArray;

    use deckgym::players::parse_player_code;
    use deckgym::rl::model::encoder::AttentionAblation;
    use deckgym::rl::model::RlModel;
    use deckgym::rl::train::checkpoint::{load_cold, LoopState};
    use deckgym::rl::train::deck_db::DeckDb;
    use deckgym::rl::train::eval::{EvalConfig, Evaluator};
    use deckgym::rl::train::sampler::{DeckSampler, DeckSource};
    use deckgym::rl::train::TrainConfig;

    #[cfg(not(feature = "rl-model-cuda"))]
    type B = NdArray<f32>;
    #[cfg(feature = "rl-model-cuda")]
    type B = burn::backend::Cuda;

    fn flag(name: &str) -> Option<String> {
        std::env::args()
            .skip_while(|arg| arg != name)
            .nth(1)
            .filter(|value| !value.starts_with("--"))
    }

    env_logger::init();
    let Some(from) = flag("--from") else {
        eprintln!("usage: block_ablation --from <hot-checkpoint-dir> [--games <n>] [--block <k>]");
        std::process::exit(2);
    };
    let from = PathBuf::from(&from);
    // `all` silences every block at once, which is the only arm that answers "is the attention
    // mechanism needed": ablating one block leaves the other free to carry the mixing, so two
    // separate nulls do not add up to one.
    let target = flag("--block").unwrap_or_else(|| "0".to_string());

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
    let embeddings = config.text_embeddings().expect("text embeddings");

    let games: usize = flag("--games")
        .and_then(|value| value.parse().ok())
        .unwrap_or(config.eval.games_per_opponent);

    // The stage's own deck mix and the stage's own anchors — the arms are compared against the
    // number the run itself reports, so they have to be the same measurement.
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
    let evaluator = Evaluator::new(
        DeckSampler::mixed(sources, stage.pure_mirror, stage.mirror).expect("sampler"),
        EvalConfig {
            envs: config.rollout.envs,
            games_per_opponent: games,
            opponents: stage
                .anchors
                .iter()
                .map(|code| parse_player_code(code).expect("anchor"))
                .collect(),
            max_crashes: config.eval.max_crashes,
        },
        config.run.seed,
    )
    .expect("evaluator");

    let blocks: Vec<usize> = if target == "all" {
        (0..config.model.num_blocks).collect()
    } else {
        vec![target
            .parse()
            .expect("--block takes a block index or `all`")]
    };
    println!(
        "{} — batch {}, stage {:?}, blocks {blocks:?}, {games} games per anchor\n",
        from.display(),
        state.batch,
        stage.name
    );
    println!("arm         anchor    winrate       se     ties");

    for (name, ablation) in [
        ("baseline", None),
        ("uniform", Some(AttentionAblation::UniformPattern)),
        ("silent", Some(AttentionAblation::Silent)),
    ] {
        // Reloaded per arm rather than cloned-and-patched: the surgery is destructive, and an arm
        // reading a model an earlier arm had already zeroed would look like a very clear result.
        let mut model = load_cold::<B>(
            RlModel::new(&config.model, &embeddings, &device),
            &from.join("model.mpk"),
            &device,
        )
        .expect("checkpoint");
        if let Some(ablation) = ablation {
            for block in &blocks {
                model.ablate(*block, ablation);
            }
        }

        // The same evaluation index for every arm, so all three sweep the same decks.
        let report = evaluator
            .evaluate(&model, &config.model, &device, 0)
            .expect("evaluation");
        for opponent in &report.opponents {
            let winrate = opponent.winrate();
            let played = opponent.games().max(1) as f64;
            println!(
                "{name:<10}  {:<8}  {winrate:6.3}   {:6.3}   {:6.3}",
                opponent.label,
                (winrate * (1.0 - winrate) / played).sqrt(),
                opponent.tierate(),
            );
        }
    }
}
