//! Baked models on the seats of an ordinary `simulate` run.
//!
//! The CLI's other player codes are heuristics behind `Player::decision_fn`, and `simulate` plays
//! them one game at a time (or one per rayon thread). A model cannot go there: `decision_fn` is
//! blocking, so it would forward at batch 1, which §1.4.3 measures at ≈ 40× the per-sample cost of
//! a saturated one. This module is therefore a second runner over the same engine — [`VecEnv`],
//! exactly as §1.5.5's rollout drives it — with `envs` games in flight and one forward per model
//! per poll.
//!
//! `envs` is the whole reason a GPU pays here, and it is *not* the same knob as `--parallel`: the
//! games advance on one thread, interleaved, and what widens is the tensor, not the thread pool.
//! The two do not compose (rayon would fan out games this loop already holds), so the RL path
//! ignores `--parallel`.
//!
//! What it keeps from `simulate`, because a runner nobody can read the output of is not a
//! substitute: the engine's `-v` log (per game, [`Env::set_debug`]) and the [`StatsCollector`]
//! table, folded per game through the same handler the parallel path merges across threads.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::tensor::backend::Backend;
use log::warn;

use crate::data_exporter::DataExporter;
use crate::players::{create_players, PlayerCode};
use crate::rl::action_mask::{Head, ACTION_MASK_DIM};
use crate::rl::env::{env_rng, split_seed, AgentId, Env, SeatPolicy, SubmitFault, VecEnv};
use crate::rl::model::config::ModelConfig;
use crate::rl::model::input::{DecisionPoint, ModelInput};
use crate::rl::model::RlModel;
use crate::rl::text_embedding::TextEmbeddings;
use crate::rl::train::baked::{load_model, Baked};
use crate::rl::train::rollout::sample_entry;
use crate::simulation_event_handler::{
    CompositeSimulationEventHandler, SimulationEventHandler, StatsCollector,
};
use crate::Deck;

/// Stream tags, so the game seeds and the action sampling cannot alias each other off one `--seed`.
const STREAM_PLAY_GAME: u64 = 0x504C_4159_0000_0001;
const STREAM_PLAY_ACTION: u64 = 0x504C_4159_0000_0002;

pub struct PlayConfig {
    pub decks: [Deck; 2],
    pub codes: [PlayerCode; 2],
    pub games: usize,
    /// Games in flight. The batch a forward actually sees is about half of it: an env holds one
    /// pending decision at a time and the seats alternate, so at two model seats each one gets
    /// roughly `envs / 2` rows per poll.
    pub envs: usize,
    pub seed: u64,
    pub models_root: PathBuf,
    /// Where `--data-output` writes its (state, action) pairs, if it was asked for.
    pub data_output: Option<PathBuf>,
    /// The engine's per-frame log. Left to the caller because it is a verbosity decision, not a
    /// correctness one.
    pub debug: bool,
}

/// How many games in a row may die to an engine panic before the run is called off.
///
/// A dropped game is missing from the denominator, so a winrate printed over a run that lost half
/// its games to panics is a different quantity than the one the header promised. The budget scales
/// with the request and stays generous: this is a person's simulation, not a training run.
fn crash_budget(games: usize) -> usize {
    (games / 20).max(5)
}

/// Play `config.games` games and return the merged stats. `on_game` is ticked once per finished
/// game, for the progress bar.
pub fn run<B: Backend>(
    config: &PlayConfig,
    device: &B::Device,
    on_game: &dyn Fn(),
) -> Result<StatsCollector, String> {
    // `AgentId(seat)`: the env tags each request with the agent that owes it an answer, and letting
    // the seat *be* that id is what makes a mirror (`rl:x,rl:x`) two independent entries rather
    // than one shared model answering both sides of its own game.
    let mut models: [Option<(RlModel<B>, ModelConfig)>; 2] = [None, None];
    for (seat, slot) in models.iter_mut().enumerate() {
        let PlayerCode::RL { name } = &config.codes[seat] else {
            continue;
        };
        let baked = Baked::load(&config.models_root, name)?;
        let model = load_model::<B>(&baked, &TextEmbeddings::zeros(), device)?;
        warn!(
            "\tSeat {seat}: model {name} ({}), rating {:.0}",
            baked
                .meta
                .provenance
                .run
                .as_deref()
                .unwrap_or("unknown run"),
            baked.meta.rating.rating,
        );
        *slot = Some((model, baked.meta.model.clone()));
    }
    if models.iter().all(Option::is_none) {
        return Err("the batched runner was called with no model seat".to_string());
    }

    let spawn = |index: u64| -> Env<'static> {
        let players = create_players(
            config.decks[0].clone(),
            config.decks[1].clone(),
            config.codes.to_vec(),
        );
        let seat_policy = |seat: usize| match &config.codes[seat] {
            PlayerCode::RL { .. } => SeatPolicy::Agent(AgentId(seat as u16)),
            _ => SeatPolicy::Scripted,
        };
        let mut env = Env::from_players(
            players,
            [seat_policy(0), seat_policy(1)],
            split_seed(config.seed, split_seed(STREAM_PLAY_GAME, index)),
        );
        env.set_debug(config.debug);
        let mut handlers: Vec<Box<dyn SimulationEventHandler>> =
            vec![Box::new(StatsCollector::new())];
        if let Some(output) = &config.data_output {
            handlers.push(Box::new(DataExporter::new(output.clone())));
        }
        env.open_handler(CompositeSimulationEventHandler::new(handlers));
        env
    };

    let mut stats = StatsCollector::new();
    let mut played = 0usize;
    let mut crashes = 0usize;
    let mut budget = crash_budget(config.games);
    let mut action_rng = env_rng(config.seed, STREAM_PLAY_ACTION);

    let mut dealt = 0u64;
    let parallel = config.envs.min(config.games).max(1);
    let mut vec_env = VecEnv::new(
        (0..parallel)
            .map(|_| {
                let env = spawn(dealt);
                dealt += 1;
                env
            })
            .collect(),
    );

    while played + crashes < config.games {
        let settled = played + crashes;
        let (pending, finished, crashed) = vec_env.poll();

        for fault in crashed {
            if budget == 0 {
                return Err(format!(
                    "gave up after {} engine panics, last: {}",
                    crash_budget(config.games),
                    fault.panic
                ));
            }
            budget -= 1;
            crashes += 1;
            warn!("game dropped on an engine panic: {}", fault.panic);
            refill(&mut vec_env, fault.env, &mut dealt, config.games, &spawn);
        }

        for done in finished {
            if let Some(handler) = vec_env.get_mut(done.env).and_then(Env::close_handler) {
                if let Some(collector) = handler.get_handler::<StatsCollector>() {
                    stats.merge(collector);
                }
            }
            played += 1;
            on_game();
            refill(&mut vec_env, done.env, &mut dealt, config.games, &spawn);
        }

        if pending.is_empty() {
            // Nothing to answer and nothing settled means no env can move: without this the loop
            // spins forever on its own emptiness rather than saying what went wrong.
            if played + crashes == settled {
                return Err(format!(
                    "simulation stalled at {settled}/{} games",
                    config.games
                ));
            }
            continue;
        }

        // One forward per model, not per env: grouping is the entire point of this runner.
        let mut rows_by_agent: BTreeMap<AgentId, Vec<usize>> = BTreeMap::new();
        for (row, slot) in pending.iter().enumerate() {
            rows_by_agent
                .entry(slot.request.agent)
                .or_default()
                .push(row);
        }

        let mut answers: Vec<Option<(Head, usize)>> = vec![None; pending.len()];
        for (agent, rows) in rows_by_agent {
            let (model, model_config) = models[agent.0 as usize]
                .as_ref()
                .ok_or_else(|| format!("no model loaded for {agent:?}"))?;
            let points: Vec<DecisionPoint<'_>> = rows
                .iter()
                .map(|&row| DecisionPoint {
                    observation: &pending[row].request.observation,
                    mask: &pending[row].request.mask,
                })
                .collect();
            let policy = model
                .forward(&ModelInput::<B>::from_points(&points, model_config, device))
                .policy
                .to_data()
                .to_vec::<f32>()
                .map_err(|err| format!("policy readback failed: {err:?}"))?;
            drop(points);

            for (batch_row, &row) in rows.iter().enumerate() {
                let probs = &policy[batch_row * ACTION_MASK_DIM..(batch_row + 1) * ACTION_MASK_DIM];
                let (entry, _) = sample_entry(&pending[row].request.mask, probs, &mut action_rng);
                answers[row] = Some((entry.head, entry.index));
            }
        }

        for (row, slot) in pending.into_iter().enumerate() {
            let (head, index) = answers[row].expect("every pending row was answered");
            match vec_env.submit(slot.env, head, index) {
                Ok(()) => {}
                Err(SubmitFault::Panicked(panic)) => {
                    if budget == 0 {
                        return Err(format!(
                            "gave up after {} engine panics, last: {panic}",
                            crash_budget(config.games)
                        ));
                    }
                    budget -= 1;
                    crashes += 1;
                    warn!("game dropped on an engine panic: {panic}");
                    refill(&mut vec_env, slot.env, &mut dealt, config.games, &spawn);
                }
                // A bit the mask set but the engine refuses is a masking bug (§1.3.7 invariant 3),
                // not a broken game — it stays fatal here as it does in training.
                Err(SubmitFault::Rejected(err)) => {
                    return Err(format!(
                        "env {} rejected {head:?}[{index}]: {err:?}",
                        slot.env
                    ))
                }
            }
        }
    }

    if crashes > 0 {
        warn!("{crashes} game(s) dropped to engine panics and are not in the counts below");
    }
    Ok(stats)
}

fn refill(
    vec_env: &mut VecEnv<'static>,
    slot: usize,
    dealt: &mut u64,
    total: usize,
    spawn: &dyn Fn(u64) -> Env<'static>,
) {
    if *dealt as usize >= total {
        vec_env.clear(slot);
        return;
    }
    let env = spawn(*dealt);
    *dealt += 1;
    vec_env.replace(slot, env);
}
