//! How far each encoder block's parameters have actually moved over a run.
//!
//! The §1.5.6 write measurement says `long_v5`'s block 0 writes 4.2x the stream it was given, so it
//! is not near-identity. It cannot say whether those weights were *learned*: a block can write hard
//! through a near-random fixed projection that the next block learned to read, and the norm of what
//! it writes is identical in both cases. A random feature map is informative — that is why the
//! question needs the parameters and not the activations.
//!
//! ```text
//! cargo run --release --features rl-model --example block_drift -- --run runs/long_v5
//! ```
//!
//! **Read against the earliest clone, not against the initialization.** The init is not
//! recoverable: `[model] init_seed` fixes the frozen tables, but nothing in `src/rl/train` seeds
//! burn's global RNG, so the blocks' starting weights are gone the moment the process ends. What is
//! on disk is §1.5.2's pool — a clone every `[pool] clone_every` batches — and the earliest of those
//! is the reference here. So a drift of `0.30` means "moved 30 % of its norm **since batch 50**",
//! not since batch 0, and the first stretch of training is missing from every number below.
//!
//! The per-interval column is the one that says whether a block is still moving; the cumulative one
//! is a sum of steps that may have cancelled, and a block wandering in a circle reads high on the
//! first and low on the second.

#[cfg(not(feature = "rl-model"))]
fn main() {
    eprintln!("build with --features rl-model (or rl-model-cuda) --release");
}

#[cfg(feature = "rl-model")]
fn main() {
    use std::path::{Path, PathBuf};

    use burn::backend::NdArray;

    use deckgym::rl::model::RlModel;
    use deckgym::rl::train::checkpoint::load_cold;
    use deckgym::rl::train::TrainConfig;

    type B = NdArray<f32>;

    fn flag(name: &str) -> Option<String> {
        std::env::args()
            .skip_while(|arg| arg != name)
            .nth(1)
            .filter(|value| !value.starts_with("--"))
    }

    env_logger::init();
    let Some(run) = flag("--run") else {
        eprintln!("usage: block_drift --run <run-dir>");
        std::process::exit(2);
    };
    let run = PathBuf::from(&run);
    let config = TrainConfig::from_file(&run.join("config.toml")).expect("config");
    let device = Default::default();
    let embeddings = config.text_embeddings().expect("text embeddings");

    // `(batch, weights)`, oldest first. The pool names its clones by the batch they were cloned at
    // and zero-pads them, so a lexical sort is chronological — the property `OpponentId::Pool`'s
    // `Display` was written for.
    let mut points: Vec<(u64, PathBuf)> = std::fs::read_dir(run.join("pool"))
        .expect("pool directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_stem()?.to_str()?.to_string();
            let batch = name.strip_prefix('b')?.parse::<u64>().ok()?;
            (path.extension()? == "mpk").then_some((batch, path))
        })
        .collect();
    points.sort_by_key(|(batch, _)| *batch);
    for hot in std::fs::read_dir(run.join("checkpoints")).expect("checkpoints") {
        let dir = hot.expect("checkpoint entry").path();
        let name = dir.file_name().and_then(|name| name.to_str()).unwrap_or("");
        if let Some(batch) = name.strip_prefix("hot-").and_then(|b| b.parse().ok()) {
            points.push((batch, dir.join("model.mpk")));
        }
    }
    points.sort_by_key(|(batch, _)| *batch);
    points.dedup_by_key(|(batch, _)| *batch);
    if points.len() < 2 {
        eprintln!("{} holds fewer than two saved models", run.display());
        std::process::exit(1);
    }

    let load = |path: &Path| {
        load_cold::<B>(
            RlModel::new(&config.model, &embeddings, &device),
            path,
            &device,
        )
        .unwrap_or_else(|err| panic!("{err}"))
    };

    println!(
        "{} — {} saved models, batches {} to {}\n",
        run.display(),
        points.len(),
        points[0].0,
        points[points.len() - 1].0
    );
    println!(
        "relative parameter drift since b{}, ‖Δ‖/‖reference‖",
        points[0].0
    );
    // Widened from the block count the run actually has, not from the two this was written
    // against: `[model] num_blocks` is what long_v6 varies, so a hard-coded pair indexes past the
    // end of exactly the run the series exists to read.
    let mut header = String::from(" batch ");
    for block in 0..config.model.num_blocks {
        header.push_str(&format!("|  b{block} q,k   v,o     ffn   "));
    }
    println!("{header}");

    let reference = load(&points[0].1);
    for (batch, path) in &points[1..] {
        let mut row = format!("{batch:6} ");
        for block in load(path).block_drift(&reference) {
            row.push_str(&format!(
                "|  {:6.4}  {:6.4}  {:6.4}  ",
                block.pattern, block.value, block.feed_forward
            ));
        }
        println!("{row}");
    }
}
