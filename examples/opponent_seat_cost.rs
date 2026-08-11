//! What a model on the opponent seat actually costs the rollout (§1.5.2).
//!
//! `73b4dc6` claims the pool "roughly doubles the forwards per game" and that the learner's own
//! batch halves whatever the grouping, because an env holds one pending decision at a time and the
//! actor alternates. Both are arguments from the loop's shape; this measures them.
//!
//! Three arms per env count, all against the same decks and seed:
//! - `scripted` — `Assignment::PerGame`, the pre-pool baseline. No opponent forward at all.
//! - `model x1` — one contiguous group, every env against one frozen model.
//! - `model x4` — four groups, which is what fragmentation costs on top of the seat itself.
//!
//! Read `mean batch` (learner frames per learner forward) against the scripted arm for the halving,
//! and `opp fwd / fwd` for the doubling. `games/s` is what either one is worth in the end.

#[cfg(not(feature = "rl-model"))]
fn main() {
    eprintln!("build with --features rl-model (or rl-model-cuda) --release");
}

#[cfg(feature = "rl-model")]
fn main() {
    use burn::tensor::backend::Backend;
    use deckgym::players::PlayerCode;
    use deckgym::rl::model::config::ModelConfig;
    use deckgym::rl::model::RlModel;
    use deckgym::rl::text_embedding::TextEmbeddings;
    use deckgym::rl::train::deck_db::DeckDb;
    use deckgym::rl::train::opponent::{Assignment, OpponentModels, OpponentSeat};
    use deckgym::rl::train::rating::OpponentId;
    use deckgym::rl::train::rollout::{Collector, RolloutConfig};
    use deckgym::rl::train::sampler::{DeckSampler, SamplerConfig};
    use std::path::Path;
    use std::time::Instant;

    const SEED: u64 = 9;

    // `frames / forwards` only converges to the mean batch over a window long enough that the
    // games carried in from the warmup are amortized — a short window counts frames whose forward
    // was paid before it started, and reports a batch wider than `envs` can physically be.
    let frames = |name: &str, fallback: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    };
    let warmup_frames = frames("WARMUP_FRAMES", 200);
    let measure_frames = frames("MEASURE_FRAMES", 1500);

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

    fn collector(envs: usize) -> Collector {
        Collector::new(
            sampler(),
            RolloutConfig {
                envs,
                opponents: vec![PlayerCode::R, PlayerCode::W],
                max_crashes_per_batch: 32,
            },
            SEED,
            None,
        )
        .expect("collector")
    }

    fn sweep<B: Backend>(
        label: &str,
        device: &B::Device,
        warmup_frames: usize,
        measure_frames: usize,
    ) {
        let config = ModelConfig::default();
        let model = RlModel::<B>::new(&config, &TextEmbeddings::zeros(), device);

        println!(
            "\n{label}  ({} frames measured per cell, {} warmup)",
            measure_frames, warmup_frames
        );
        println!(
            "{:>5}  {:<10}  {:>8}  {:>9}  {:>10}  {:>9}  {:>9}",
            "envs", "opponent", "games/s", "frames/s", "mean batch", "fwd", "opp fwd"
        );

        for envs in [16, 32, 64, 128] {
            for arm in ["scripted", "model x1", "model x4"] {
                // A fresh registry per arm: the ids an assignment names have to be the ids the
                // collector was given, and the arms name different ones.
                let mut opponents = OpponentModels::<B>::new();
                let assignment = match arm {
                    "scripted" => Assignment::PerGame(vec![PlayerCode::R, PlayerCode::W]),
                    "model x1" => {
                        let agent = opponents.insert(
                            OpponentId::Pool(1),
                            RlModel::<B>::new(&config, &TextEmbeddings::zeros(), device),
                        );
                        Assignment::uniform(OpponentId::Pool(1), OpponentSeat::Model(agent))
                    }
                    _ => {
                        // Four *distinct* registrations, not one id in four groups: fragmenting
                        // the opponent forward is the whole point of this arm, and one shared id
                        // would batch back into a single call.
                        let groups = (1..=4)
                            .map(|slot| {
                                let id = OpponentId::Pool(slot);
                                let agent = opponents.insert(
                                    id.clone(),
                                    RlModel::<B>::new(&config, &TextEmbeddings::zeros(), device),
                                );
                                (id, OpponentSeat::Model(agent))
                            })
                            .collect();
                        Assignment::grouped(groups).expect("grouped")
                    }
                };

                let mut collector = collector(envs)
                    .with_assignment(assignment)
                    .expect("assignment");

                collector
                    .collect(&model, &opponents, &config, device, warmup_frames, 0)
                    .expect("warmup");

                let start = Instant::now();
                let (_, stats) = collector
                    .collect(&model, &opponents, &config, device, measure_frames, 1)
                    .expect("rollout");
                let elapsed = start.elapsed().as_secs_f64();

                println!(
                    "{:>5}  {:<10}  {:>8.2}  {:>9.1}  {:>10.1}  {:>9}  {:>9}",
                    envs,
                    arm,
                    stats.games as f64 / elapsed,
                    stats.frames as f64 / elapsed,
                    stats.frames as f64 / stats.forwards as f64,
                    stats.forwards,
                    stats.opponent_forwards,
                );
                if !stats.crashes.is_empty() {
                    println!(
                        "        ({} games lost to engine panics)",
                        stats.crashes.len()
                    );
                }
            }
        }
    }

    #[cfg(feature = "rl-model-cuda")]
    sweep::<burn::backend::Cuda>("cuda", &Default::default(), warmup_frames, measure_frames);
    #[cfg(not(feature = "rl-model-cuda"))]
    sweep::<burn::backend::NdArray>(
        "ndarray",
        &Default::default(),
        warmup_frames,
        measure_frames,
    );
}
