//! The §1.5.1 training loop: collect → GAE → one MMD step + one magnet SL step → repeat.
//!
//! The best-response network trains against the §1.5.2 panel, with §1.5.1b's magnet as the KL
//! target when the run's `.toml` asks for one. `[pool] enabled` decides what that panel is: off, it
//! is `[rollout] opponents` as scripted seats; on, it is the PFSP pool of frozen clones, baked
//! models and heuristic anchors, rated with Glicko-2. §1.5.4's curriculum, when
//! `[[curriculum.stages]]` is non-empty, drives the deck DB/archetype subset, the opponent set and
//! the magnet's partial reseed through stage transitions on top of that — otherwise this is a
//! single stage, exactly as it was before §1.5.4 existed. With `[magnet] enabled = false` it is
//! §1.1.6 stage 1 — *validate that the agent learns at all* — on the best-response alone.
//!
//! Run (CPU, slow — §1.5.5 measures batching as a GPU-only win):
//!
//! ```text
//! cargo run --release --features rl-model --example train_player
//! cargo run --release --features rl-model-cuda --example train_player -- --cuda
//! cargo run --release --features rl-model-cuda --example train_player -- --cuda --resume
//! ```
//!
//! Three things end the loop: the `[rollout] batches` budget, Ctrl-C, and §1.5.4's plateau; the
//! last two checkpoint on the way out. `p` is not one of them — it pauses in place, checkpoints,
//! and waits for the same key.

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::AutodiffBackend;

use deckgym::rl::model::input::ModelInput;
use deckgym::rl::model::RlModel;
use deckgym::rl::train::anchor::AnchorSeed;
use deckgym::rl::train::checkpoint::{self, Interrupt, LoopState};
use deckgym::rl::train::curriculum::{Curriculum, CurriculumEvent, StagePanel};
use deckgym::rl::train::dashboard::{clock, Dashboard, Frame as DashboardFrame};
use deckgym::rl::train::diagnostics::{
    attention as attention_scalars, curriculum as curriculum_scalars, diagnostics,
    magnet as magnet_scalars, probe_points, standard,
};
use deckgym::rl::train::eval::EvalLog;
use deckgym::rl::train::logger::MetricLog;
use deckgym::rl::train::magnet::Magnet;
use deckgym::rl::train::opponent::{Assignment, OpponentModels};
use deckgym::rl::train::panel::{check_eval_disjoint, Panel, PanelState, PoolLog};
use deckgym::rl::train::pause::Pause;
use deckgym::rl::train::rollout::Collector;
use deckgym::rl::train::update::{inference_model, Learner};
use deckgym::rl::train::TrainConfig;

fn main() {
    env_logger::init();
    let cuda = std::env::args().any(|arg| arg == "--cuda");
    let resume = std::env::args().any(|arg| arg == "--resume");

    let path = std::env::args()
        .skip_while(|arg| arg != "--config")
        .nth(1)
        .unwrap_or_else(|| "config/default.toml".to_string());
    let config = TrainConfig::from_file(Path::new(&path)).expect("config");

    if cuda {
        #[cfg(feature = "rl-model-cuda")]
        {
            run::<Autodiff<burn::backend::Cuda>>(&config, &Default::default(), resume);
            return;
        }
        #[cfg(not(feature = "rl-model-cuda"))]
        panic!("--cuda needs --features rl-model-cuda");
    }
    run::<Autodiff<NdArray>>(&config, &Default::default(), resume);
}

fn run<B: AutodiffBackend>(config: &TrainConfig, device: &B::Device, resume: bool) {
    let interrupt = Interrupt::install().expect("interrupt handler");
    let pause = Pause::install();

    let run_dir = if resume {
        config.open_run().expect("run to resume")
    } else {
        config.create_run().expect("run directory")
    };
    println!("run {}", run_dir.root().display());

    // §1.5.5's master seed has to reach the *parameter init* too, not just the env and draw
    // streams: the learned weights come from the backend's generator, so without this two runs of
    // one config start from different models and nothing downstream is comparable.
    B::seed(device, config.run.seed);

    let model_config = &config.model;
    // §1.2.9's frozen text tables, resolved once and handed to every model the run builds. They are
    // not carried by a checkpoint, so two models built from different tables read the same weights
    // against different features — which is why this is one value and not three call sites.
    let embeddings = config.text_embeddings().expect("text embeddings");
    println!(
        "text embeddings: {}",
        if embeddings.is_empty() {
            "none — training on zeros (§1.2.9 ablation)".to_string()
        } else {
            config.text_embeddings.clone()
        }
    );
    let mut model = RlModel::<B>::new(model_config, &embeddings, device);
    let step_config = config.step_config().expect("step schedules");
    let mut learner = Learner::<B>::new(step_config.clone());
    for (name, schedule) in [
        ("learning_rate", &step_config.learning_rate),
        ("entropy_coeff", &step_config.entropy_coeff),
        ("value_coeff", &step_config.value_coeff),
        ("residual_decay", &step_config.residual_decay),
    ] {
        println!("{name}: {}", schedule.describe());
    }

    // §1.5.1b. Built before the collector so the seed below can borrow the same deck sampler, and
    // `None` in a best-response-only run — in which case the KL term never enters the loss and its
    // series never reach the log.
    let mut magnet = config
        .magnet_config()
        .expect("magnet config")
        .map(|magnet_config| {
            println!(
                "magnet: reservoir {} (fill floor {}), SL batch {} at {}",
                magnet_config.capacity,
                magnet_config.min_fill,
                magnet_config.batch,
                magnet_config.learning_rate.describe(),
            );
            Magnet::<B>::new(
                RlModel::<B>::new(model_config, &embeddings, device),
                magnet_config,
                config.run.seed,
            )
        });

    // §1.5.4. Resolved once — `resume` may later rebuild `curriculum` at a different stage, but
    // never re-resolves the `.toml`. Empty `stages` (the default) leaves `curriculum` `None` and
    // every line below that reads it inert, so a run without `[[curriculum.stages]]` is untouched.
    let curriculum_stages = config.curriculum_stages().expect("curriculum stages");
    let mut curriculum: Option<Curriculum> = if curriculum_stages.is_empty() {
        None
    } else {
        let curriculum = Curriculum::new(
            curriculum_stages.clone(),
            config.decks.root.clone(),
            config.eval.window_batches.max(1),
            config.eval.envs.unwrap_or(config.rollout.envs),
            config.eval.max_crashes,
            config.curriculum.plateau_k,
            config.curriculum.plateau_epsilon,
            config.run.seed,
            0,
        )
        .expect("curriculum");
        println!(
            "curriculum: {} stage(s), starting at {:?} (advance ≥ {:.0}%, plateau ε={} over {} evals)",
            curriculum.stage_count(),
            curriculum.stage().name,
            100.0 * curriculum.stage().advance.winrate,
            config.curriculum.plateau_epsilon,
            config.curriculum.plateau_k,
        );
        Some(curriculum)
    };

    // The training sampler: the curriculum's starting stage when one is configured, or the run's
    // flat `[decks]` fields otherwise — the two are mutually exclusive by construction
    // (`TrainConfig::curriculum_stages` only returns stages when `[[curriculum.stages]]` is
    // non-empty).
    let sampler = match &curriculum {
        Some(curriculum) => curriculum.sampler().clone(),
        None => config.deck_sampler().expect("deck sampler"),
    };
    println!(
        "db {} — {} decks drawable",
        sampler
            .dbs()
            .map(|db| format!("{} ({} archetypes)", db.name, db.archetypes.len()))
            .collect::<Vec<_>>()
            .join(" + "),
        sampler.deck_count(),
    );

    // A curriculum stage's scripted panel (`[pool].enabled = false`) overrides `[rollout]
    // opponents` for the run's starting opponent set; `[rollout] opponents` itself stays a
    // required field regardless (§1.5.4's curriculum supersedes it, but does not remove the need
    // for a syntactically valid run-global fallback for a config that predates the curriculum).
    let mut rollout_config = config.rollout_config().expect("panel");
    if let Some(StagePanel::Scripted(opponents)) = curriculum.as_ref().map(|c| &c.stage().panel) {
        rollout_config.opponents = opponents.clone();
    }

    let mut collector = Collector::new(
        sampler,
        rollout_config,
        config.run.seed,
        config.harvest(&run_dir).expect("harvest"),
    )
    .expect("collector")
    .with_crash_log(config.crash_log(&run_dir));

    // The starting stage's harvest rate (§1.5.4) — set here and not only on a transition, or a
    // `harvest_log` on the first stage would silently never apply. Only the rate: a run with
    // `[harvest] log = false` built no `Harvest`, so there is nothing for a stage to turn on.
    if let (Some(harvest), Some(sampling)) = (
        collector.harvest_mut(),
        curriculum.as_ref().and_then(Curriculum::harvest_sampling),
    ) {
        harvest.set_sampling(sampling);
    }

    // §1.5.2's pool. `None` leaves `[rollout] opponents` in charge, resolved as scripted seats —
    // which is what every run before the pool existed did, and what its `.toml` still describes.
    let mut panel = if config.pool.enabled {
        let mut panel = Panel::<B::InnerBackend>::new(
            &config.pool,
            run_dir.pool(),
            model_config.clone(),
            embeddings.clone(),
        )
        .expect("pool");
        // §1.5.6 holds the evaluation out so it is not a saturation signal, and an overlap is
        // silent otherwise: the evaluation still runs and still reports.
        if let Some(held_out) = config.held_out_opponents().expect("eval panel") {
            check_eval_disjoint(&panel.permanent_ids(), &held_out).expect("eval panel");
        }
        panel.load(device).expect("opponent weights");
        // A curriculum stage's pool membership (§1.5.4) overrides `[pool] anchors`/`baked` for the
        // run's starting permanent list — `[pool]`'s own fields stay required regardless, as the
        // seed a fresh `Panel` needs before anything can retarget it.
        if let Some(StagePanel::Pool(permanent)) = curriculum.as_ref().map(|c| &c.stage().panel) {
            panel
                .retarget(permanent.clone(), device)
                .expect("curriculum stage retarget");
        }
        println!(
            "pool: {} best + {} history over {} permanent, refresh every {} batches (clone every \
             {}), {} opponent(s) in flight",
            config.pool.best_slots,
            config.pool.history_slots,
            panel.permanent_ids().len(),
            config.pool.refresh_every,
            config.pool.clone_every,
            config.pool.concurrent_opponents,
        );
        Some(panel)
    } else {
        None
    };
    // The draws the pool makes: its own stream, keyed by batch, so turning the pool on does not
    // shift the deck or action streams of an otherwise identical run.
    let mut pool_rng = deckgym::rl::env::env_rng(config.run.seed, 0x504F_4F4C_0000_0001);
    // §1.5.4's own stream: reservoir eviction and the per-stage magnet reseed, kept off every
    // other consumer for the same reason.
    let mut curriculum_rng = deckgym::rl::env::env_rng(config.run.seed, 0x4355_5252_0000_0002);
    // Opened only by a run that has a pool: §1.5.5 makes the *existence* of these directories the
    // signal that the run used the thing they hold, so an empty `pool/` beside a scripted run would
    // be a lie about what it did.
    let mut pool_log = panel
        .is_some()
        .then(|| PoolLog::open(&run_dir.pool()).expect("pool log"));
    let no_opponents = OpponentModels::<B::InnerBackend>::new();

    // The continuous curve and the held-out harness, when the curriculum does *not* own them —
    // §1.5.4 owns a window/gate/evaluator per stage instead, rebuilt on every transition, so a
    // curriculum run never touches these.
    let mut window = curriculum.is_none().then(|| config.panel_window());
    let evaluator = curriculum
        .is_none()
        .then(|| config.evaluator(collector.sampler().clone()))
        .transpose()
        .expect("evaluator")
        .flatten();
    let mut gate = curriculum.is_none().then(|| config.eval_gate());
    let mut eval_log = EvalLog::open(&run_dir.eval()).expect("eval log");
    if curriculum.is_none() {
        println!(
            "panel window: {} batches",
            config.eval.window_batches.max(1)
        );
        if let Some(evaluator) = &evaluator {
            let panel: Vec<String> = evaluator
                .config()
                .opponents
                .iter()
                .map(|code| code.to_string())
                .collect();
            println!(
                "held-out eval: {:?} — {} games vs each of [{}]",
                config.eval.trigger,
                evaluator.config().games_per_opponent,
                panel.join(", ")
            );
        }
    }

    let mut state = LoopState {
        batch: 0,
        games_started: collector.games_started(),
        elapsed_seconds: 0.0,
        stage: curriculum
            .as_ref()
            .map(Curriculum::stage_index)
            .unwrap_or(0),
    };
    // Whether the KL already has a target with history behind it, which is not the same question
    // as whether this is a resume — see the seed below.
    let mut restored_magnet = false;
    if resume {
        let hot = checkpoint::latest_hot(&run_dir.checkpoints())
            .expect("no complete hot checkpoint to resume from");
        let (restored, optimizer, saved) =
            checkpoint::load_hot(&hot, model, device).expect("hot checkpoint");
        model = restored;
        learner.load_optimizer(optimizer);
        collector.restore(saved.games_started).expect("envs");
        state = saved;
        if let Some(magnet) = &mut magnet {
            let fresh = RlModel::<B>::new(model_config, &embeddings, device);
            match checkpoint::load_magnet(&hot, fresh, device).expect("magnet checkpoint") {
                Some((weights, optimizer)) => {
                    magnet.restore(weights, optimizer);
                    restored_magnet = true;
                    // Said out loud either way, because the two resumes are different runs. With
                    // the buffer, the magnet keeps averaging over the whole history. Without it,
                    // `seen` restarts at 0 and the average is re-taken over the post-resume stream
                    // — `loss/kl_magnet` will fall for that reason and not because anything
                    // improved (see `train::reservoir`).
                    match checkpoint::load_reservoir(&hot).expect("reservoir state") {
                        Some(encoded) => {
                            magnet
                                .restore_reservoir(&encoded)
                                .expect("decode the reservoir");
                            println!(
                                "magnet resumed — reservoir restored, {} frame(s) over {} offer(s)",
                                magnet.reservoir().len(),
                                magnet.reservoir().seen(),
                            );
                        }
                        None => println!(
                            "magnet resumed — no reservoir in {}, so it refills from empty and \
                             the average policy restarts at this batch",
                            hot.display()
                        ),
                    }
                }
                // A magnet turned on mid-run, or a checkpoint from before §1.5.1b existed. The
                // seed below then runs on a resume, which it otherwise never does: `η` is charged
                // from the next batch on, and an unseeded magnet here would have the KL pull a
                // trained best-response toward a random init — worse than no magnet at all.
                None => println!(
                    "no magnet in {} — seeding it as if this were a fresh run",
                    hot.display()
                ),
            }
        }
        if let Some(panel) = &mut panel {
            match checkpoint::load_pool(&hot).expect("pool state") {
                Some(encoded) => {
                    let restored: PanelState =
                        serde_json::from_str(&encoded).expect("decode the pool state");
                    panel
                        .restore(restored, &config.pool, device)
                        .expect("pool restore");
                    println!(
                        "pool resumed — {} member(s), {} archived, {} rating period(s)",
                        panel.pool().active().len(),
                        panel.pool().archive().len(),
                        panel.ratings().periods(),
                    );
                }
                // A pool turned on mid-run, or a checkpoint from before §1.5.2 existed. Starting
                // fresh is the only option and it is a real discontinuity: every rating the run
                // had established is gone, so the elo curve restarts from the default rather than
                // continuing. Said out loud, because a silently reset curve reads as a collapse.
                None => println!(
                    "no pool in {} — starting the panel and its ratings from scratch",
                    hot.display()
                ),
            }
        }
        // §1.5.4: a resume into a stage other than 0 rebuilds the curriculum at that stage —
        // sampler/window/gate/evaluator and the harvest rate are pure functions of
        // `(config, stage index)`, so nothing about them is checkpointed. The panel's permanent list is *not* retargeted again here:
        // `panel.restore` above already reloaded the checkpointed (already-retargeted) permanent
        // list, ratings, slots and archive as one unit, and retargeting a second time would be
        // redundant. The scripted case has no such restore, so its assignment is set directly.
        if let Some(curriculum) = &mut curriculum {
            if state.stage != curriculum.stage_index() {
                *curriculum = Curriculum::new(
                    curriculum_stages.clone(),
                    config.decks.root.clone(),
                    config.eval.window_batches.max(1),
                    config.eval.envs.unwrap_or(config.rollout.envs),
                    config.eval.max_crashes,
                    config.curriculum.plateau_k,
                    config.curriculum.plateau_epsilon,
                    config.run.seed,
                    state.stage,
                )
                .expect("curriculum resume");
                collector.set_sampler(curriculum.sampler().clone());
                if let StagePanel::Scripted(opponents) = &curriculum.stage().panel {
                    collector.set_assignment(Assignment::PerGame(opponents.clone()));
                }
                if let (Some(harvest), Some(sampling)) =
                    (collector.harvest_mut(), curriculum.harvest_sampling())
                {
                    harvest.set_sampling(sampling);
                }
                println!(
                    "curriculum resumed at stage {} ({})",
                    curriculum.stage_index(),
                    curriculum.stage().name
                );
            }
        }
        println!(
            "resumed {} at batch {}, {} of training so far",
            hot.display(),
            state.batch,
            clock(state.elapsed_seconds)
        );
    }

    // §1.1.3's heuristic anchor as the initial magnet. Skipped only when a magnet was *restored*:
    // that one has already been trained past the anchor, and refilling its buffer with heuristic
    // play would drag the KL target backwards to where the run started. A resume that found no
    // magnet to restore is a fresh magnet, and gets seeded like one. This is the run's *starting*
    // stage's seed regardless of §1.5.4 — a curriculum stage's own `magnet_seed` only ever fires
    // on a later transition (see the main loop), never here.
    if let (false, Some(magnet)) = (restored_magnet, &mut magnet) {
        if let Some(seed) = config
            .magnet_seed(collector.sampler().clone())
            .expect("magnet seed")
        {
            let started = std::time::Instant::now();
            let stats = seed.fill(magnet.reservoir_mut()).expect("heuristic seed");
            // The realized mix, not the configured one — a component that answered nothing is not
            // in the magnet whatever the `.toml` asked for.
            let mix: Vec<String> = seed
                .config()
                .anchors
                .iter()
                .zip(&stats.per_anchor)
                .map(|(anchor, frames)| format!("{}×{frames}", anchor.player))
                .collect();
            println!(
                "magnet seed: {} frames over {} games [{}] ({} unmatched, {} crashed) [{:.1}s]",
                stats.frames,
                stats.games,
                mix.join(" "),
                stats.unmatched,
                stats.crashes,
                started.elapsed().as_secs_f64(),
            );
            let steps = seed.config().steps;
            let pretrained = magnet.pretrain(model_config, device, steps);
            println!(
                "magnet seeded — {steps} cloning steps, loss {}",
                match (pretrained, steps) {
                    (Some(metrics), _) => format!("{:.4}", metrics.loss),
                    (None, 0) => "not run (no pretraining steps asked for)".to_string(),
                    (None, _) => "not run (buffer under the fill floor)".to_string(),
                }
            );
        }
    }

    // The full §1.5.6 line, standard and diagnostic, goes to the JSONL log; stdout gets the
    // subset a human watches a run with. The two are not the same audience.
    let mut log = MetricLog::open(&run_dir.logs()).expect("metric log");

    // `win%` and `±` are the *windowed* per-anchor mean and its spread across anchors, not the
    // batch's own winrate: at ~60 games a batch that one carries a ±13 % interval and reads as
    // noise. The per-anchor breakdown is on the JSONL line, which is where one goes to find out
    // which anchor moved.
    let mut dashboard = Dashboard::new();
    dashboard.header();
    if pause.is_some() {
        println!("press p to pause (checkpoints on the way in), p again to resume");
    }
    // Held across batches: it only refreshes when the gate fires, and a row that blanked in
    // between would read as "no evaluation has ever run" rather than "none since the last one".
    let mut last_eval: Option<(u64, f64, Option<bool>)> = None;

    while state.batch < config.rollout.batches as u64 {
        let start = std::time::Instant::now();

        // The §1.5.2 draw, before the collection it governs. It reaches the games spawned from
        // here on and leaves the ones in flight alone, so a reward is always attributable to the
        // opponent that actually played it.
        if let Some(panel) = &panel {
            let assignment = panel
                .assignment(state.batch, &mut pool_rng)
                .expect("opponent assignment");
            collector.set_assignment(assignment);
        }
        let (episodes, stats) = collector
            .collect(
                &inference_model(&model),
                panel
                    .as_ref()
                    .map(|panel| panel.models())
                    .unwrap_or(&no_opponents),
                model_config,
                device,
                config.rollout.frames_per_batch,
                state.batch,
            )
            .expect("rollout");
        if let Some(panel) = &mut panel {
            panel.record(&episodes);
        }
        let collected = start.elapsed();

        // The KL target, refreshed per batch: the magnet moved on the previous one, and a stale
        // target is a proximal step toward a policy that no longer exists.
        let target = magnet.as_ref().map(|magnet| magnet.target());
        let (next, metrics) = learner.step(
            model,
            &episodes,
            target.as_ref(),
            model_config,
            device,
            state.batch,
        );
        model = next;
        drop(target);

        // Fold the batch into the reservoir and clone from it — §1.5.5's "one MMD step + one magnet
        // SL step". After the BR step so the KL target is dropped first: the two models and the
        // inference twin are not all resident at once on the card §1.4.3 sizes for.
        let magnet_metrics = magnet.as_mut().and_then(|magnet| {
            let accepted = magnet.observe(&episodes, state.batch);
            magnet.step(model_config, device, state.batch, accepted)
        });

        // The whole batch, collection *and* step: what this answers is where the run is, not
        // which of the two phases is the bottleneck — §1.5.6's `rollout/games_per_second` is what
        // separates them.
        let batch_seconds = start.elapsed().as_secs_f64();
        state.elapsed_seconds += batch_seconds;

        // §1.5.4 drives its own window/gate/evaluator per stage when a curriculum is configured;
        // otherwise this is the pre-§1.5.4 fold + gate, byte-for-byte.
        let mut eval_scalars = Vec::new();
        let mut windowed_scalars = Vec::new();
        let mut windowed_summary = (0.0, 0.0); // (winrate_mean, winrate_std), for the stdout line
                                               // Set on `CurriculumEvent::Plateaued` below — Part 1's global stop, alongside the step
                                               // budget and Ctrl-C.
        let mut curriculum_plateaued = false;
        if let Some(curriculum) = &mut curriculum {
            let outcome = curriculum
                .poll(
                    state.batch,
                    &episodes,
                    &inference_model(&model),
                    model_config,
                    device,
                )
                .expect("curriculum poll");

            // Same metric names a non-curriculum run leaves behind, off the *stage's own* window
            // — so `panel/window/*` and the stdout `win%`/`±` columns stay meaningful (and
            // dashboards built against the non-curriculum series keep working) either way.
            let windowed = curriculum.window().report();
            windowed_summary = (windowed.winrate_mean(), windowed.winrate_std());
            windowed_scalars = windowed.scalars("panel/window");
            windowed_scalars.push((
                "panel/window/batches".to_string(),
                curriculum.window().batches() as f64,
            ));
            if let Some(worst) = windowed.winrate_min() {
                windowed_scalars.push(("panel/window/winrate_min".to_string(), worst));
            }

            if let Some(report) = &outcome.report {
                eval_log
                    .record(state.batch, "eval", report)
                    .expect("eval report");
                eval_scalars = report.scalars("eval");
                eval_scalars.push((
                    "eval/winrate_min".to_string(),
                    report.winrate_min().unwrap_or(0.0),
                ));
                if let Some(confirmed) = outcome.confirmed {
                    eval_scalars.push((
                        "eval/floor_confirmed".to_string(),
                        if confirmed { 1.0 } else { 0.0 },
                    ));
                }
                last_eval = Some((
                    state.batch,
                    report.winrate_min().unwrap_or(0.0),
                    outcome.confirmed,
                ));
                dashboard.event(format!(
                    "held-out eval @ batch {} (stage {:?}): worst {:.1}% — mean {:.1}% (std {:.1}) — {}{}",
                    state.batch,
                    curriculum.stage().name,
                    100.0 * report.winrate_min().unwrap_or(0.0),
                    100.0 * report.winrate_mean(),
                    100.0 * report.winrate_std(),
                    report.summary(),
                    match outcome.confirmed {
                        Some(true) => "  FLOOR CONFIRMED",
                        Some(false) => "  floor not confirmed",
                        None => "",
                    },
                ));
            }

            match outcome.event {
                CurriculumEvent::Advanced { from, to } => {
                    collector.set_sampler(curriculum.sampler().clone());
                    match &curriculum.stage().panel {
                        StagePanel::Pool(permanent) => {
                            panel
                                .as_mut()
                                .expect(
                                    "[pool].enabled and every curriculum stage's opponent kind \
                                     agree by construction (TrainConfig::curriculum_stages)",
                                )
                                .retarget(permanent.clone(), device)
                                .expect("curriculum stage retarget");
                        }
                        StagePanel::Scripted(opponents) => {
                            collector.set_assignment(Assignment::PerGame(opponents.clone()));
                        }
                    }
                    if let (Some(harvest), Some(sampling)) =
                        (collector.harvest_mut(), curriculum.harvest_sampling())
                    {
                        harvest.set_sampling(sampling);
                    }
                    // The magnet's partial reseed (§1.5.4): evict a fraction of the reservoir, then
                    // top the freed capacity back up from the new stage's own heuristic mixture.
                    // Only the starting stage skips this — its seed is the ordinary run-start one
                    // above, not this per-transition one.
                    if let (Some(magnet), Some(seed_config)) =
                        (&mut magnet, curriculum.stage().magnet_seed.clone())
                    {
                        let evicted = magnet
                            .reservoir_mut()
                            .evict_fraction(curriculum.stage().evict_fraction, &mut curriculum_rng);
                        let seed_stream = deckgym::rl::env::split_seed(
                            deckgym::rl::env::split_seed(config.run.seed, 0x4D41_474E_4554_0001),
                            to as u64,
                        );
                        let seed =
                            AnchorSeed::new(curriculum.sampler().clone(), seed_config, seed_stream)
                                .expect("curriculum magnet reseed");
                        let stats = seed
                            .fill(magnet.reservoir_mut())
                            .expect("curriculum magnet reseed fill");
                        dashboard.event(format!(
                            "curriculum stage {from} -> {to} ({:?}): evicted {evicted}, reseeded \
                             {} frames over {} games",
                            curriculum.stage().name,
                            stats.frames,
                            stats.games,
                        ));
                    } else {
                        dashboard.event(format!(
                            "curriculum stage {from} -> {to} ({:?})",
                            curriculum.stage().name
                        ));
                    }
                    state.stage = to;
                }
                CurriculumEvent::Plateaued { spread } => {
                    dashboard.event(format!(
                        "curriculum plateau @ batch {} (stage {:?}): spread {:.4} < ε — stopping",
                        state.batch,
                        curriculum.stage().name,
                        spread
                    ));
                    curriculum_plateaued = true;
                }
                CurriculumEvent::None => {}
            }
        } else if let (Some(window), Some(gate)) = (&mut window, &mut gate) {
            window.observe(&episodes);
            let windowed = window.report();
            windowed_summary = (windowed.winrate_mean(), windowed.winrate_std());
            windowed_scalars = windowed.scalars("panel/window");
            windowed_scalars.push(("panel/window/batches".to_string(), window.batches() as f64));
            if let Some(worst) = windowed.winrate_min() {
                windowed_scalars.push(("panel/window/winrate_min".to_string(), worst));
            }

            // The gate screens on the window, which is why it is armed only once the batch is
            // folded in. Under a cadence it fires at batch 0 too, deliberately: the untrained
            // model is the baseline every later point is read against, and it costs one
            // evaluation to have it.
            let due = gate.arm(state.batch, window);
            if let (Some(evaluator), Some(index)) = (evaluator.as_ref(), due) {
                let started = std::time::Instant::now();
                let report = evaluator
                    .evaluate(&inference_model(&model), model_config, device, index)
                    .expect("evaluation");
                let seconds = started.elapsed().as_secs_f64();
                // Counted as run time but kept out of `batch_seconds`: the ETA is read off the
                // training batch, and folding an occasional evaluation into it would make every
                // batch look longer than it is.
                state.elapsed_seconds += seconds;

                eval_log
                    .record(state.batch, "eval", &report)
                    .expect("eval report");

                // The verdict §1.5.4 will read. Only when the gate screened on a floor: an
                // evaluation that was not triggered by a threshold has no threshold to confirm,
                // and a series pinned to zero would read as a measurement.
                let verdict = gate
                    .floor()
                    .map(|floor| report.winrate_min().is_some_and(|worst| worst >= floor));
                last_eval = Some((state.batch, report.winrate_min().unwrap_or(0.0), verdict));
                dashboard.event(format!(
                    "held-out eval @ batch {}: worst {:.1}% — mean {:.1}% (std {:.1}) — {} [{:.1}s]{}",
                    state.batch,
                    100.0 * report.winrate_min().unwrap_or(0.0),
                    100.0 * report.winrate_mean(),
                    100.0 * report.winrate_std(),
                    report.summary(),
                    seconds,
                    match verdict {
                        Some(true) => "  FLOOR CONFIRMED",
                        Some(false) => "  floor not confirmed",
                        None => "",
                    },
                ));

                eval_scalars = report.scalars("eval");
                eval_scalars.push((
                    "eval/winrate_min".to_string(),
                    report.winrate_min().unwrap_or(0.0),
                ));
                eval_scalars.push(("eval/seconds".to_string(), seconds));
                if let Some(confirmed) = verdict {
                    eval_scalars.push((
                        "eval/floor_confirmed".to_string(),
                        if confirmed { 1.0 } else { 0.0 },
                    ));
                }
            }
        }

        let mut scalars = standard(&stats, &metrics, &episodes);
        if let Some(magnet_metrics) = &magnet_metrics {
            scalars.extend(magnet_scalars(magnet_metrics));
        }
        scalars.extend(windowed_scalars);
        scalars.extend(eval_scalars);
        if let Some(curriculum) = &curriculum {
            scalars.extend(curriculum_scalars(
                curriculum.stage_index(),
                curriculum.stage_count(),
            ));
        }
        scalars.extend(diagnostics(&episodes, &stats.head_entropy));
        // The encoder's attention, on `[step] attn_probe_every`. Read off the inference model and
        // one micro-batch of the frames just collected: what each head looks at is a property of
        // the weights, so it needs neither the autodiff graph nor the whole batch — and the frames
        // are already here, which is what keeps the probe to a single forward.
        if step_config.attn_probe_every > 0
            && state.batch.is_multiple_of(step_config.attn_probe_every)
        {
            let points = probe_points(&episodes, step_config.micro_batch);
            if !points.is_empty() {
                let input = ModelInput::from_points(&points, model_config, device);
                scalars.extend(attention_scalars(
                    &inference_model(&model).attention_stats(&input),
                ));
            }
        }
        scalars.push((
            "rollout/games_per_second".to_string(),
            stats.games as f64 / collected.as_secs_f64(),
        ));
        // Absolute seconds since the run began, resumes included — the axis every other curve is
        // read against once one wants to know what a plateau costs rather than how many batches
        // it lasted. `batch_seconds` beside it is what turns the remaining batches into an ETA.
        scalars.push(("time/elapsed_seconds".to_string(), state.elapsed_seconds));
        scalars.push(("time/batch_seconds".to_string(), batch_seconds));
        // Both pool artefacts are written *before* the cadences below mutate anything, so the
        // scalar line and the table describe the same thing: the panel that played this batch, not
        // the one the next batch will face.
        if let Some(panel) = &panel {
            scalars.extend(panel.scalars());
            if let Some(pool_log) = &mut pool_log {
                pool_log
                    .record(state.batch, &panel.rows(state.batch))
                    .expect("pool table");
            }
        }
        log.record(state.batch, &scalars).expect("metrics");

        for crash in &stats.crashes {
            dashboard.event(format!(
                "engine panic in env {} — game dropped: {}{}",
                crash.env,
                crash.message,
                match &crash.dump {
                    Some(path) => format!("\n  dumped to {}", path.display()),
                    None => String::new(),
                }
            ));
        }

        // The mean over every batch the run has done, resumes included — an ETA off this batch
        // alone swings with whatever else the machine was doing for those few seconds.
        let mean_batch_seconds = state.elapsed_seconds / (state.batch + 1) as f64;
        let window_state = window
            .as_ref()
            .map(|window| window.batches())
            .or_else(|| curriculum.as_ref().map(|c| c.window().batches()))
            .unwrap_or(0);
        // Kept, because the pause below redraws this same frame rather than assembling a second
        // one: a block that disagreed with the last running one about anything but the pause
        // would leave the watcher wondering what else moved.
        let frame = DashboardFrame {
            run: config.run.name.clone(),
            batch: state.batch,
            total_batches: config.rollout.batches as u64,
            elapsed_seconds: state.elapsed_seconds,
            mean_batch_seconds,
            games_per_second: stats.games as f64 / collected.as_secs_f64(),
            games: stats.games,
            stage: curriculum.as_ref().map(|curriculum| {
                (
                    curriculum.stage_index(),
                    curriculum.stage_count(),
                    curriculum.stage().name.clone(),
                )
            }),
            floor: curriculum
                .as_ref()
                .and_then(|curriculum| curriculum.gate().floor()),
            screen: curriculum
                .as_ref()
                .and_then(|curriculum| curriculum.gate().screen(curriculum.window())),
            holding: curriculum
                .as_ref()
                .and_then(|curriculum| curriculum.gate().holding()),
            cooldown_remaining: curriculum
                .as_ref()
                .map(|curriculum| curriculum.gate().cooldown_remaining(state.batch))
                .unwrap_or(0),
            window_mean: windowed_summary.0,
            window_std: windowed_summary.1,
            window_batches: window_state,
            window_capacity: config.eval.window_batches.max(1),
            last_eval,
            pool: panel.as_ref().map(|panel| {
                (
                    panel.pool().active().len(),
                    panel.pool().archive().len(),
                    panel.ratings().periods(),
                    panel.ratings().learner().rating.rating,
                )
            }),
            policy_loss: metrics.policy_loss,
            value_loss: metrics.value_loss,
            entropy: metrics.entropy,
            grad_norm: metrics.grad_norm,
            kl_magnet: metrics.kl_magnet,
            magnet_loss: magnet_metrics.map(|magnet| magnet.loss),
            paused: false,
        };
        dashboard.draw(&frame);

        state.batch += 1;
        state.games_started = collector.games_started();

        // One durable record per window turnover, so `eval/` holds the raw per-anchor counts even
        // in the default run where the held-out harness is off — §1.5.5 makes a directory's
        // contents the evidence a run leaves, and flat scalars in the metrics line are the
        // projection, not the record. Either window — the free-standing one, or the current
        // curriculum stage's own — is read the same way here.
        if state
            .batch
            .is_multiple_of(config.eval.window_batches.max(1) as u64)
        {
            let report = window.as_ref().map(|window| window.report()).or_else(|| {
                curriculum
                    .as_ref()
                    .map(|curriculum| curriculum.window().report())
            });
            if let Some(report) = report {
                eval_log
                    .record(state.batch, "panel_window", &report)
                    .expect("eval report");
            }
        }

        // The plateau is Part 1's global stop, alongside the step budget and Ctrl-C: it takes the
        // same checkpoint-on-the-way-out path as an interrupt, so a plateaued run is resumable
        // (into more batches, or a `.toml` with a raised epsilon) exactly like an interrupted one.
        let stopping = interrupt.raised() || curriculum_plateaued;
        // Read once, so the rest of the batch tail agrees with itself about whether this is a
        // pause — the key is pressed by a human at an arbitrary point in it.
        let paused = pause.as_ref().is_some_and(Pause::is_paused);

        // Flushed on the same occasions as the checkpoint, and always on the way out: the shard
        // sits in memory until it is written, so an unflushed one is the harvest a crash costs.
        if stopping || paused || state.batch.is_multiple_of(config.harvest.every_batches) {
            if let Some(harvest) = collector.harvest_mut() {
                if let Some(shard) = harvest.flush().expect("harvest shard") {
                    dashboard.event(format!("harvest {}", shard.display()));
                }
            }
        }

        // §1.5.2's two cadences, in order: a clone is archived first so an id the pool holds is
        // always an id whose weights exist, then the period closes and the slots are re-decided.
        // `refresh` reloads the models in the same call, because clearing them invalidates every
        // `AgentId` the assignment could still be naming.
        if let Some(panel) = &mut panel {
            if panel.should_clone(state.batch) {
                let id = panel
                    .admit(state.batch, &inference_model(&model))
                    .expect("pool clone");
                dashboard.event(format!("pool + {id}"));
            }
            if panel.should_refresh(state.batch) {
                let refresh = panel
                    .refresh(state.batch, &mut pool_rng, device)
                    .expect("pool refresh");
                for (id, role) in &refresh.admitted {
                    dashboard.event(format!("pool + {id} ({role:?})"));
                }
                for id in &refresh.released {
                    dashboard.event(format!("pool - {id}"));
                }
            }
        }

        if stopping || paused || state.batch.is_multiple_of(config.checkpoint.every_batches) {
            let magnet_state = magnet
                .as_ref()
                .map(|magnet| (magnet.weights(), magnet.optimizer_record()));
            let pool_state = panel
                .as_ref()
                .map(|panel| serde_json::to_string(&panel.state()).expect("encode the pool state"));
            // Only on the way out. The buffer dwarfs everything else in the directory, and on the
            // rolling cadence the write would cost more than the batch it interrupts — while a
            // resume from an autosave is a crash resume, which has already lost the games in
            // flight. Stop and pause are the exits the user chose, and the ones §1.5.1b's average
            // policy is worth preserving across.
            let reservoir = (stopping || paused)
                .then(|| {
                    magnet
                        .as_ref()
                        .and_then(|magnet| magnet.encoded_reservoir())
                })
                .flatten()
                .transpose()
                .expect("encode the reservoir");
            let written = checkpoint::save_hot(
                &run_dir.checkpoints(),
                &model,
                learner.optimizer_record(),
                magnet_state,
                state,
                checkpoint::SideState {
                    pool: pool_state.as_deref(),
                    reservoir: reservoir.as_deref(),
                },
                config.checkpoint.keep_hot,
            )
            .expect("hot checkpoint");
            if stopping {
                dashboard.finish();
                println!(
                    "{} — {}",
                    if curriculum_plateaued {
                        "stopped on plateau"
                    } else {
                        "interrupted"
                    },
                    written.display()
                );
                return;
            }
            if paused {
                dashboard.event(format!(
                    "paused at batch {} — {}",
                    state.batch,
                    written.display()
                ));
            }
        }

        // Held here, at the very bottom of the batch: `state.elapsed_seconds` is already closed for
        // this one and the next `Instant::now` is taken after the wait, so a pause costs the ETA
        // nothing. The model, the optimizer, the pool and the envs stay resident — that is the
        // difference between this and the stop above, which §1.5.5 says a resume cannot undo.
        if paused {
            let held = std::time::Instant::now();
            let pause = pause
                .as_ref()
                .expect("`paused` is only true when the key reader exists");
            dashboard.draw(&DashboardFrame {
                paused: true,
                batch: state.batch,
                ..frame.clone()
            });
            if !pause.wait(|| interrupt.raised()) {
                dashboard.finish();
                println!("interrupted while paused — checkpoint already written");
                return;
            }
            dashboard.event(format!(
                "resumed at batch {} after {}",
                state.batch,
                clock(held.elapsed().as_secs_f64())
            ));
        }
    }

    // The block described a batch that is now over; leaving it on screen under the closing lines
    // would keep showing an ETA for a run that has finished.
    dashboard.finish();
    if let Some(harvest) = collector.harvest_mut() {
        if let Some(shard) = harvest.flush().expect("harvest shard") {
            println!("harvest {}", shard.display());
        }
    }

    // Weights alone: a finished run is what §1.5.2's pool will play against, and an opponent has
    // no use for AdamW's moments.
    let final_weights = run_dir.checkpoints().join("final");
    checkpoint::save_cold(&model, &final_weights).expect("cold checkpoint");
    println!(
        "done in {} — {}",
        clock(state.elapsed_seconds),
        final_weights.display()
    );
}
