//! Decision throughput of every seat, and the size of the models among them.
//!
//! Writes `PERFORMANCE.md`, which is the number every other document quotes.
//!
//! ```text
//! cargo run --release --example benchmark_players
//! cargo run --release --features rl-model-cuda --example benchmark_players -- \
//!     --from runs/my_run/checkpoints/hot-00002121 --models my_model
//! ```
//!
//! `--models` names baked directories under `--models-root`, `--from` run checkpoints; a checkpoint
//! is rebuilt from its run's `config.toml`, a baked model from its own `meta.toml`. The backend
//! follows the feature, since a CPU arm is minutes where a GPU arm is seconds.
//!
//! **Decisions per second, never games per second.** A game is not a fixed amount of work: a
//! heuristic that ends the game in 20 decisions and a random policy that drags it to 100 do the
//! same work per decision and differ sevenfold per game. Games/s is reported beside it, and is
//! what the ratio of the two columns explains.
//!
//! **A decision is a frame offering two or more legal actions.** One-candidate frames are resolved
//! inside `src/rl/env.rs` without a forward, so counting them would credit the model with work it
//! never did and the heuristics with work they barely did.
//!
//! Every seat is measured in a mirror — the same player on both sides — so every decision in the
//! window belongs to the seat being measured, and single-threaded, so the number is per seat and
//! not per machine.

use std::cell::Cell;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::fs;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;

use deckgym::actions::Action;
use deckgym::players::{create_players, parse_player_code, Player, PlayerCode};
use deckgym::{Deck, Game, State};

const DEFAULT_PLAYERS: &str = "aa,er,w,r,et,v,e2,e3";
const DEFAULT_DECKS: &str = "example_decks/venusaur-exeggutor.txt,example_decks/weezing-arbok.txt";
const DEFAULT_OUT: &str = "PERFORMANCE.md";

/// One row of either table: what was measured, and enough of the denominators to re-read it.
struct Arm {
    label: String,
    decisions: u64,
    games: u64,
    elapsed: Duration,
}

impl Arm {
    fn per_second(&self) -> f64 {
        self.decisions as f64 / self.elapsed.as_secs_f64()
    }

    fn per_game(&self) -> f64 {
        self.decisions as f64 / self.games.max(1) as f64
    }

    fn games_per_second(&self) -> f64 {
        self.games as f64 / self.elapsed.as_secs_f64()
    }
}

/// Counts the frames its inner player actually decided.
///
/// A shared `Cell` rather than a return value: the engine owns the players for the whole game and
/// hands nothing back, so the count has to outlive them.
struct Counting {
    inner: Box<dyn Player>,
    decisions: Rc<Cell<u64>>,
}

impl Debug for Counting {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.inner.fmt(f)
    }
}

impl Player for Counting {
    fn get_deck(&self) -> Deck {
        self.inner.get_deck()
    }

    fn decision_fn(&mut self, rng: &mut StdRng, state: &State, actions: &[Action]) -> Action {
        if actions.len() > 1 {
            self.decisions.set(self.decisions.get() + 1);
        }
        self.inner.decision_fn(rng, state, actions)
    }
}

fn flag(name: &str) -> Option<String> {
    std::env::args()
        .skip_while(|arg| arg != name)
        .nth(1)
        .filter(|value| !value.starts_with("--"))
}

fn flag_or<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    flag(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} does not take `{value}`"))
        })
        .unwrap_or(fallback)
}

fn decks(spec: &str) -> [Deck; 2] {
    let paths: Vec<&str> = spec.split(',').collect();
    assert_eq!(paths.len(), 2, "--decks takes two comma-separated files");
    [
        Deck::from_file(paths[0]).expect("deck A"),
        Deck::from_file(paths[1]).expect("deck B"),
    ]
}

/// Plays mirrors of `code` until the budget is spent, and reports what the seat got through.
///
/// The budget is checked between games, never inside one: a game cut in half would count the
/// decisions of an opening (cheap, few legal actions) without the mid-game they lead to.
fn measure_heuristic(code: &PlayerCode, decks: &[Deck; 2], budget: Duration, seed: u64) -> Arm {
    let decisions = Rc::new(Cell::new(0u64));
    let mut games = 0u64;
    let start = Instant::now();
    while start.elapsed() < budget {
        let players = create_players(
            decks[0].clone(),
            decks[1].clone(),
            vec![code.clone(), code.clone()],
        )
        .into_iter()
        .map(|inner| {
            Box::new(Counting {
                inner,
                decisions: Rc::clone(&decisions),
            }) as Box<dyn Player>
        })
        .collect();
        Game::new(players, seed.wrapping_add(games)).play();
        games += 1;
    }
    Arm {
        label: format!("`{code}`"),
        decisions: decisions.get(),
        games,
        elapsed: start.elapsed(),
    }
}

fn table(arms: &[Arm], first_column: &str) -> String {
    let mut out = format!("| {first_column} | Decisions/s | Decisions/game | Games/s |\n");
    out.push_str("| --- | ---: | ---: | ---: |\n");
    for arm in arms {
        // A window too short to finish a game says nothing about game length, and a decisions/game
        // read off zero games is the seat's whole window mistaken for one game.
        let (per_game, games) = if arm.games == 0 {
            ("—".to_string(), "—".to_string())
        } else {
            (
                format!("{:.1}", arm.per_game()),
                format!("{:.1}", arm.games_per_second()),
            )
        };
        out.push_str(&format!(
            "| {} | {:.0} | {per_game} | {games} |\n",
            arm.label,
            arm.per_second(),
        ));
    }
    out
}

fn main() {
    env_logger::init();

    let budget = Duration::from_secs_f64(flag_or("--seconds", 5.0));
    let seed: u64 = flag_or("--seed", 0xB0BA_0001);
    let out = flag("--out").unwrap_or_else(|| DEFAULT_OUT.to_string());
    let deck_spec = flag("--decks").unwrap_or_else(|| DEFAULT_DECKS.to_string());
    let decks = decks(&deck_spec);

    let codes: Vec<PlayerCode> = flag("--players")
        .unwrap_or_else(|| DEFAULT_PLAYERS.to_string())
        .split(',')
        .map(|code| parse_player_code(code).expect("player code"))
        .collect();

    let mut heuristics = Vec::new();
    for code in &codes {
        eprintln!("measuring {code}…");
        let arm = measure_heuristic(code, &decks, budget, seed);
        eprintln!(
            "  {:.0} decisions/s over {} games",
            arm.per_second(),
            arm.games
        );
        heuristics.push(arm);
    }

    let models = flag("--models").unwrap_or_default();
    let mut specs: Vec<(bool, &str)> = models
        .split(',')
        .filter(|name| !name.is_empty())
        .map(|name| (true, name))
        .collect();
    let from = flag("--from").unwrap_or_default();
    specs.extend(
        from.split(',')
            .filter(|dir| !dir.is_empty())
            .map(|dir| (false, dir)),
    );
    let model_section = model_section(&specs, &decks, budget, seed);

    let mut document = String::new();
    document.push_str("# Performance\n\n");
    document.push_str(
        "Generated by `cargo run --release --example benchmark_players` — do not edit by hand.\n\n",
    );
    document.push_str(&format!(
        "Measured {}. Decks `{deck_spec}`, seed `{seed:#x}`, {:.0} s per arm, one thread. A \
         decision is a frame offering two or more legal actions; every seat plays a mirror of \
         itself, so every decision counted is its own.\n\n",
        chrono::Local::now().format("%Y-%m-%d"),
        budget.as_secs_f64(),
    ));
    document.push_str("## Seats\n\n");
    document.push_str(&table(&heuristics, "Seat"));
    document.push_str(&model_section);

    fs::write(&out, document).expect("write the report");
    println!("wrote {out}");
}

#[cfg(not(feature = "rl-model"))]
fn model_section(specs: &[(bool, &str)], _: &[Deck; 2], _: Duration, _: u64) -> String {
    assert!(
        specs.is_empty(),
        "--models / --from need --features rl-model (or rl-model-cuda)"
    );
    String::new()
}

#[cfg(feature = "rl-model")]
fn model_section(specs: &[(bool, &str)], decks: &[Deck; 2], budget: Duration, seed: u64) -> String {
    use std::path::{Path, PathBuf};

    use deckgym::rl::model::RlModel;
    use deckgym::rl::text_embedding::TextEmbeddings;
    use deckgym::rl::train::baked::{load_model, Baked};
    use deckgym::rl::train::checkpoint::load_cold;
    use deckgym::rl::train::TrainConfig;

    if specs.is_empty() {
        return String::new();
    }

    #[cfg(not(feature = "rl-model-cuda"))]
    type B = burn::backend::NdArray<f32>;
    #[cfg(feature = "rl-model-cuda")]
    type B = burn::backend::Cuda;

    let device = Default::default();
    let backend = if cfg!(feature = "rl-model-cuda") {
        "CUDA"
    } else {
        "NdArray CPU"
    };
    let envs: usize = flag_or("--envs", 64);
    let root = flag("--models-root").unwrap_or_else(|| "models".to_string());
    let root = Path::new(&root);

    let mut arms = Vec::new();
    let mut sizes = String::new();
    for (is_baked, spec) in specs {
        eprintln!("measuring {spec}…");
        // A baked model carries its own `[model]` table; a hot checkpoint carries none, so its
        // run's config is what says how to rebuild the network the weights fit.
        let (label, model, config) = if *is_baked {
            let baked = Baked::load(root, spec).expect("baked model");
            let model =
                load_model::<B>(&baked, &TextEmbeddings::zeros(), &device).expect("weights");
            (format!("rl:{spec}"), model, baked.meta.model.clone())
        } else {
            let dir = PathBuf::from(spec);
            let run = dir
                .parent()
                .and_then(|checkpoints| checkpoints.parent())
                .expect("a checkpoint two levels under a run");
            let config = TrainConfig::from_file(&run.join("config.toml")).expect("run config");
            let embeddings = config.text_embeddings().expect("text embeddings");
            let model = load_cold::<B>(
                RlModel::new(&config.model, &embeddings, &device),
                &dir.join("model.mpk"),
                &device,
            )
            .expect("checkpoint");
            (spec.to_string(), model, config.model.clone())
        };

        let arm = measure_model::<B>(&label, &model, &config, decks, envs, budget, seed, &device);
        eprintln!(
            "  {:.0} decisions/s over {} games",
            arm.per_second(),
            arm.games
        );
        arms.push(arm);
        for (component, params) in model.parameter_breakdown() {
            sizes.push_str(&format!("| `{label}` | {component} | {params} |\n"));
        }
    }

    let mut section = format!("\n## Models — {backend}, `envs = {envs}`\n\n");
    section.push_str(&table(&arms, "Model"));
    section.push_str(
        "\nBatching is the whole difference on a GPU backend and none of it on CPU, where the \
         forward is GEMM-bound and flat in the batch: `envs` is the width of each forward, and a \
         mirror splits it in two because an env holds one pending decision at a time.\n",
    );
    section.push_str("\n## Model sizes\n\n");
    section.push_str("| Model | Component | Parameters |\n| --- | --- | ---: |\n");
    section.push_str(&sizes);
    section.push_str(
        "\nThe frozen tables are gathered, never trained, and are not in these counts.\n",
    );
    section
}

/// Runs a model against itself through the batched runner, counting the frames it answered.
#[cfg(feature = "rl-model")]
#[allow(clippy::too_many_arguments)]
fn measure_model<B: burn::tensor::backend::Backend>(
    label: &str,
    model: &deckgym::rl::model::RlModel<B>,
    config: &deckgym::rl::model::config::ModelConfig,
    decks: &[Deck; 2],
    envs: usize,
    budget: Duration,
    seed: u64,
    device: &B::Device,
) -> Arm {
    use std::collections::BTreeMap;

    use deckgym::players::create_players;
    use deckgym::rl::action_mask::ACTION_MASK_DIM;
    use deckgym::rl::env::{env_rng, split_seed, AgentId, Env, SeatPolicy, VecEnv};
    use deckgym::rl::model::input::{DecisionPoint, ModelInput};
    use deckgym::rl::train::rollout::sample_entry;

    let code = PlayerCode::RL {
        name: label.to_string(),
    };

    let spawn = |index: u64| -> Env<'static> {
        let players = create_players(
            decks[0].clone(),
            decks[1].clone(),
            vec![code.clone(), code.clone()],
        );
        Env::from_players(
            players,
            [SeatPolicy::Agent(AgentId(0)), SeatPolicy::Agent(AgentId(1))],
            split_seed(seed, index),
        )
    };

    let mut dealt = 0u64;
    let mut vec_env = VecEnv::new(
        (0..envs.max(1))
            .map(|_| {
                let env = spawn(dealt);
                dealt += 1;
                env
            })
            .collect(),
    );
    let mut action_rng = env_rng(seed, 0x0BEE_0001);

    // The pool warms up on the first forwards (allocator, kernels), and the envs all start on turn
    // one. Timing starts once both have settled, and only the games finished after that count.
    let mut decisions = 0u64;
    let mut games = 0u64;
    let mut start = Instant::now();
    let mut warm = false;
    loop {
        if !warm && start.elapsed() >= budget.min(Duration::from_secs(1)) {
            warm = true;
            decisions = 0;
            games = 0;
            start = Instant::now();
        }
        if warm && start.elapsed() >= budget {
            break;
        }

        let (pending, finished, crashed) = vec_env.poll();
        for slot in finished {
            games += 1;
            vec_env.replace(slot.env, spawn(dealt));
            dealt += 1;
        }
        for slot in crashed {
            vec_env.replace(slot.env, spawn(dealt));
            dealt += 1;
        }
        if pending.is_empty() {
            continue;
        }
        decisions += pending.len() as u64;

        let mut rows_by_agent: BTreeMap<AgentId, Vec<usize>> = BTreeMap::new();
        for (row, slot) in pending.iter().enumerate() {
            rows_by_agent
                .entry(slot.request.agent)
                .or_default()
                .push(row);
        }
        let mut answers = vec![None; pending.len()];
        for (_, rows) in rows_by_agent {
            let points: Vec<DecisionPoint<'_>> = rows
                .iter()
                .map(|&row| DecisionPoint {
                    observation: &pending[row].request.observation,
                    mask: &pending[row].request.mask,
                })
                .collect();
            let policy = model
                .forward(&ModelInput::<B>::from_points(&points, config, device))
                .policy
                .to_data()
                .to_vec::<f32>()
                .expect("policy readback");
            drop(points);
            for (batch_row, &row) in rows.iter().enumerate() {
                let probs = &policy[batch_row * ACTION_MASK_DIM..(batch_row + 1) * ACTION_MASK_DIM];
                let (entry, _) = sample_entry(&pending[row].request.mask, probs, &mut action_rng);
                answers[row] = Some((entry.head, entry.index));
            }
        }
        for (row, slot) in pending.into_iter().enumerate() {
            let (head, index) = answers[row].expect("every row answered");
            if vec_env.submit(slot.env, head, index).is_err() {
                vec_env.replace(slot.env, spawn(dealt));
                dealt += 1;
            }
        }
    }

    Arm {
        label: format!("`{label}`"),
        decisions,
        games,
        elapsed: start.elapsed(),
    }
}
