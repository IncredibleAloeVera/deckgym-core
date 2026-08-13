//! Plays an ablated checkpoint against its unablated self.
//!
//! Silencing every block's attention costs ~4.5 points of winrate against `er` — real, but small
//! enough that two readings survive it. Either the attention genuinely does little, or `er` cannot
//! see it: at a 0.795 winrate most games are not close, and a mechanism that only decides hard
//! positions moves a heuristic matchup very little. A heuristic cannot settle that, and there is no
//! harder one — `w`, `v` and `aa` all sit at 0.84–0.94, which is easier still.
//!
//! The model's own unablated self is the hardest available opponent and the one whose skill is, by
//! construction, exactly matched. If the ablated model holds ~50 % here, the attention is not
//! deciding games; if it collapses, `er` was the wrong instrument and every winrate above it is a
//! floor effect.
//!
//! ```text
//! cargo run --release --features rl-model-cuda --example head_to_head -- \
//!     --from runs/<run>/checkpoints/<hot-dir>
//! ```
//!
//! **The first arm is the control, and it is not optional.** The learner always occupies
//! `LEARNER_SEAT`, so a seat advantage — going first in a game where tempo decides a lot — lands
//! entirely on one side of every number below. Baseline against baseline measures it directly, and
//! the ablated arm has to be read as a departure from *that*, never from 0.5.

#[cfg(not(feature = "rl-model"))]
fn main() {
    eprintln!("build with --features rl-model (or rl-model-cuda) --release");
}

#[cfg(feature = "rl-model")]
fn main() {
    use std::path::PathBuf;

    #[cfg(not(feature = "rl-model-cuda"))]
    use burn::backend::NdArray;

    use deckgym::rl::model::encoder::AttentionAblation;
    use deckgym::rl::model::RlModel;
    use deckgym::rl::train::checkpoint::{load_cold, LoopState};
    use deckgym::rl::train::deck_db::DeckDb;
    use deckgym::rl::train::opponent::{Assignment, OpponentModels, OpponentSeat};
    use deckgym::rl::train::rating::OpponentId;
    use deckgym::rl::train::rollout::{Collector, RolloutConfig};
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
        eprintln!("usage: head_to_head --from <hot-checkpoint-dir> [--games <n>]");
        std::process::exit(2);
    };
    let from = PathBuf::from(&from);
    let target_games: usize = flag("--games")
        .and_then(|value| value.parse().ok())
        .unwrap_or(600);

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

    let load = || {
        load_cold::<B>(
            RlModel::new(&config.model, &embeddings, &device),
            &from.join("model.mpk"),
            &device,
        )
        .expect("checkpoint")
    };

    let stage = &config.curriculum.stages[state.stage];
    let sources = || -> Vec<DeckSource> {
        stage
            .mix
            .iter()
            .map(|spec| DeckSource {
                db: DeckDb::load(&config.decks.root.join(&spec.db)).expect("deck db"),
                share: spec.share,
                archetypes: spec.archetypes.clone(),
            })
            .collect()
    };

    println!(
        "{} — batch {}, stage {:?}, ~{target_games} games per arm\n",
        from.display(),
        state.batch,
        stage.name
    );
    println!("learner                        games    wins   ties  winrate      se");

    for (name, ablation) in [
        ("baseline (seat control)", None),
        (
            "uniform, every block",
            Some(AttentionAblation::UniformPattern),
        ),
        ("silent, every block", Some(AttentionAblation::Silent)),
    ] {
        let mut learner = load();
        if let Some(ablation) = ablation {
            for block in 0..config.model.num_blocks {
                learner.ablate(block, ablation);
            }
        }

        // The opponent seat is always the unablated checkpoint, so every arm is measured against
        // one fixed reference rather than against whatever the previous arm was.
        let mut opponents = OpponentModels::new();
        let agent = opponents.insert(OpponentId::Pool(state.batch), load());

        let mut collector = Collector::new(
            DeckSampler::mixed(sources(), stage.pure_mirror, stage.mirror).expect("sampler"),
            RolloutConfig {
                envs: config.rollout.envs,
                // Never drawn from — `with_assignment` below replaces the panel with a single
                // grouped model seat — but `Collector::new` refuses an empty one, and rightly:
                // everywhere else an empty panel is a config that would silently play nobody.
                opponents: config
                    .rollout
                    .opponents
                    .iter()
                    .map(|code| deckgym::players::parse_player_code(code).expect("opponent code"))
                    .collect(),
                max_crashes_per_batch: 64,
            },
            config.run.seed ^ (0x11EA_D2_u64 << 8),
            None,
        )
        .expect("collector")
        .with_assignment(Assignment::uniform(
            OpponentId::Pool(state.batch),
            OpponentSeat::Model(agent),
        ))
        .expect("assignment");

        // `collect` is sized in frames, not games, so it is called until the games are in — and the
        // batch index advances with it, since it reseeds the action stream and a repeated index
        // would replay one batch of games over and over.
        let (mut games, mut wins, mut ties) = (0usize, 0usize, 0usize);
        let mut batch = 0;
        while games < target_games {
            let (episodes, _) = collector
                .collect::<B>(
                    &learner,
                    &opponents,
                    &config.model,
                    &device,
                    config.rollout.frames_per_batch,
                    batch,
                )
                .expect("rollout");
            for episode in &episodes {
                games += 1;
                if episode.reward > 0.0 {
                    wins += 1;
                } else if episode.reward == 0.0 {
                    ties += 1;
                }
            }
            batch += 1;
        }

        let winrate = wins as f64 / games as f64;
        println!(
            "{name:<28}  {games:5}   {wins:5}  {ties:5}   {winrate:6.3}  {:6.3}",
            (winrate * (1.0 - winrate) / games as f64).sqrt()
        );
    }
}
