//! Runs §1.5.6's attention read-out against a checkpoint, off the training loop.
//!
//! The probe lives inside the trainer, so a series added today starts at the next batch and says
//! nothing about the run already on disk. This reads the same numbers off weights that have stopped
//! moving — which is also the only way to get them without restarting a run.
//!
//! ```text
//! cargo run --release --features rl-model --example attention_probe -- \
//!     --from runs/<run>/checkpoints/<hot-dir>
//! ```
//!
//! **Every arm is repeated on `--repeats` independent draws**, and the spread is printed beside the
//! mean. The probe reads 64 frames of one collection: a single number off a single draw cannot say
//! whether a head moved or the deck lottery did, and the whole point of a read-out taken *after* a
//! run is that nobody is watching a curve that would average the noise out. A difference smaller
//! than the spread printed here is not a difference.
//!
//! What it reproduces from the run, and what it cannot. The weights, the `[model]` table, the text
//! embeddings and the stage's deck mix all come from the run's own files — the embeddings
//! especially, since loading zeros against weights trained on real ones would change the token
//! features and so the attention. The **opponent distribution cannot be reproduced**: the training
//! probe reads frames collected against `[pool]`'s clones as well as the scripted panel, and those
//! clones are the run's own history. This plays `[rollout] opponents` alone. Boards differ, so read
//! a family mass from here against one from the log with that in mind; the pairwise divergences are
//! a property of the weights and travel better.

#[cfg(not(feature = "rl-model"))]
fn main() {
    eprintln!("build with --features rl-model (or rl-model-cuda) --release");
}

#[cfg(feature = "rl-model")]
fn main() {
    use std::path::{Path, PathBuf};

    use burn::backend::NdArray;

    use deckgym::players::parse_player_code;
    use deckgym::rl::model::input::ModelInput;
    use deckgym::rl::model::introspect::{AttentionStats, BlockWrite, FAMILY_NAMES};
    use deckgym::rl::model::RlModel;
    use deckgym::rl::train::checkpoint::{load_cold, LoopState};
    use deckgym::rl::train::deck_db::DeckDb;
    use deckgym::rl::train::diagnostics::probe_points;
    use deckgym::rl::train::opponent::OpponentModels;
    use deckgym::rl::train::rollout::{Collector, RolloutConfig};
    use deckgym::rl::train::sampler::{DeckSampler, DeckSource};
    use deckgym::rl::train::TrainConfig;

    // A read-out is a forward with no gradients, and a few megabytes of weights read once is not
    // what a GPU is for.
    type B = NdArray<f32>;

    fn flag(name: &str) -> Option<String> {
        std::env::args()
            .skip_while(|arg| arg != name)
            .nth(1)
            .filter(|value| !value.starts_with("--"))
    }

    fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        serde_json::from_str(&raw)
            .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()))
    }

    env_logger::init();
    let Some(from) = flag("--from") else {
        eprintln!(
            "usage: attention_probe --from <hot-checkpoint-dir> [--repeats <n>] [--frames <n>]"
        );
        std::process::exit(2);
    };
    let from = PathBuf::from(&from);
    let repeats: u64 = flag("--repeats")
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let frames: usize = flag("--frames")
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);

    let state: LoopState = read_json(&from.join("loop.json"));
    // `runs/<name>/checkpoints/hot-<batch>/` — the config two levels up is the one guaranteed to be
    // what these weights were trained at, same reasoning as `bake_model`'s `--from` on a directory.
    let config_path = from
        .parent()
        .and_then(|checkpoints| checkpoints.parent())
        .map(|run| run.join("config.toml"))
        .filter(|config| config.is_file())
        .unwrap_or_else(|| panic!("{} has no run config two levels up", from.display()));
    let config = TrainConfig::from_file(&config_path).expect("config");
    let device = Default::default();

    let embeddings = config.text_embeddings().expect("text embeddings");
    let model = load_cold::<B>(
        RlModel::new(&config.model, &embeddings, &device),
        &from.join("model.mpk"),
        &device,
    )
    .expect("checkpoint");

    // The stage the checkpoint was paused in, not the last one in the file: a run stopped mid
    // curriculum would otherwise be probed on a deck distribution it has never played.
    let stage = &config.curriculum.stages[state.stage];
    let sources: Vec<DeckSource> = if stage.mix.is_empty() {
        vec![DeckSource {
            db: DeckDb::load(&config.decks.root.join(&stage.db)).expect("deck db"),
            share: 1.0,
            archetypes: stage.archetypes.clone(),
        }]
    } else {
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
        "{} — batch {}, stage {:?}, {} frames x {repeats} draws",
        from.display(),
        state.batch,
        stage.name,
        frames
    );

    let mut draws: Vec<AttentionStats> = Vec::new();
    for repeat in 0..repeats {
        let sampler = DeckSampler::mixed(
            sources
                .iter()
                .map(|source| DeckSource {
                    db: source.db.clone(),
                    share: source.share,
                    archetypes: source.archetypes.clone(),
                })
                .collect(),
            stage.pure_mirror,
            stage.mirror,
        )
        .expect("sampler");
        // A seed the run never used, so the draws are independent of it as well as of each other —
        // reusing `[run] seed` would reproduce the frames the checkpoint was last trained on.
        let mut collector = Collector::new(
            sampler,
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
            config.run.seed ^ (0xA77E_u64 << 32),
            None,
        )
        .expect("collector");

        let (episodes, _) = collector
            .collect::<B>(
                &model,
                &OpponentModels::new(),
                &config.model,
                &device,
                frames,
                repeat,
            )
            .expect("rollout");
        let points = probe_points(&episodes, 64);
        draws.push(model.attention_stats(&ModelInput::from_points(
            &points,
            &config.model,
            &device,
        )));
    }

    // Mean and half-range: with 4 draws the range is the honest summary and a standard deviation
    // would dress up three degrees of freedom as a distribution.
    let spread = |values: &[f64]| -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let low = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, (high - low) / 2.0)
    };

    println!("\nhead     entropy          {}", {
        let mut header = String::new();
        for family in FAMILY_NAMES {
            header.push_str(&format!("{family:>16}"));
        }
        header
    });
    for (index, head) in draws[0].heads.iter().enumerate() {
        let (entropy, entropy_spread) = spread(
            &draws
                .iter()
                .map(|draw| draw.heads[index].entropy)
                .collect::<Vec<_>>(),
        );
        let mut line = format!(
            "b{}h{}  {entropy:5.2} ±{entropy_spread:4.2}  ",
            head.block, head.head
        );
        for family in 0..FAMILY_NAMES.len() {
            let (focus, focus_spread) = spread(
                &draws
                    .iter()
                    .map(|draw| draw.heads[index].family_mass[family] / draw.family_share[family])
                    .collect::<Vec<_>>(),
            );
            line.push_str(&format!("  {focus:6.2} ±{focus_spread:4.2}"));
        }
        println!("{line}");
    }

    println!("\nresidual write, as a fraction of the stream written into");
    println!("block    attention        whole block       stream norm");
    for (index, write) in draws[0].writes.iter().enumerate() {
        let column = |pick: fn(&BlockWrite) -> f64| {
            spread(
                &draws
                    .iter()
                    .map(|draw| pick(&draw.writes[index]))
                    .collect::<Vec<_>>(),
            )
        };
        let (attention, attention_spread) = column(|write| write.attention);
        let (total, total_spread) = column(|write| write.total);
        let (residual, residual_spread) = column(|write| write.residual);
        println!(
            "  b{}    {attention:6.3} ±{attention_spread:5.3}    {total:6.3} ±{total_spread:5.3}   \
             {residual:7.2} ±{residual_spread:5.2}",
            write.block
        );
    }

    println!(
        "\npairwise Jensen-Shannon divergence, nats (0 = one head twice, ln 2 = 0.693 = disjoint)"
    );
    for block in 0..config.model.num_blocks {
        println!("\n  block {block}");
        print!("        ");
        for head in 1..config.model.num_heads {
            print!("{:>14}", format!("h{head}"));
        }
        println!();
        for low in 0..config.model.num_heads - 1 {
            print!("     h{low}");
            for high in 1..config.model.num_heads {
                if high <= low {
                    print!("{:>14}", "");
                    continue;
                }
                let (divergence, divergence_spread) = spread(
                    &draws
                        .iter()
                        .map(|draw| {
                            draw.pairs
                                .iter()
                                .find(|pair| {
                                    pair.block == block && pair.low == low && pair.high == high
                                })
                                .expect("pair")
                                .divergence
                        })
                        .collect::<Vec<_>>(),
                );
                print!("{:>14}", format!("{divergence:.3} ±{divergence_spread:.3}"));
            }
            println!();
        }
    }
}
