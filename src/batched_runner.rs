//! Batched Game Runner for GPU-Optimized ONNX Simulation
//!
//! This module provides `BatchedGameRunner`, which runs multiple games simultaneously
//! and batches all ONNX inference calls for maximum GPU utilization.
//!
//! Optimizations:
//! - Pre-allocated observation and mask buffers (avoid allocation per step)
//! - Rayon parallelization for observation collection and action application
//! - Cache-friendly memory access patterns

use indicatif::ProgressBar;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
#[cfg(feature = "onnx")]
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{
    actions::apply_action,
    deck::Deck,
    generate_possible_actions,
    players::{create_players, Player, PlayerCode},
    rl::{
        get_action_mask, get_indexed_actions, get_observation_tensor, ACTION_SPACE_SIZE,
        OBSERVATION_SIZE,
    },
    simulate::create_progress_bar,
    state::{GameOutcome, State},
};

#[cfg(feature = "onnx")]
use crate::players::BatchedOnnxInference;

/// Configuration for a player in the batched runner
#[derive(Clone)]
#[allow(dead_code)]
enum PlayerConfig {
    /// ONNX model with path and device
    #[cfg(feature = "onnx")]
    Onnx { path: String, device: String },
    /// CPU-based player with code
    Cpu { code: PlayerCode },
}

/// A single game instance within the batched runner
struct GameInstance {
    /// Current game state
    state: State,
    /// RNG for this game
    rng: StdRng,
    /// CPU players (only populated if using CPU players)
    cpu_players: Option<Vec<Box<dyn Player>>>,
    /// Track total actions/steps in this game to detect infinite loops
    step_count: usize,
}

// Safety: GameInstance can be sent between threads because State and StdRng are Send
// The cpu_players field contains trait objects but they're only accessed from the main thread
unsafe impl Send for GameInstance {}
unsafe impl Sync for GameInstance {}

impl GameInstance {
    fn new(deck_a: Deck, deck_b: Deck, seed: u64, cpu_codes: &[Option<PlayerCode>; 2]) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let state = State::initialize(&deck_a, &deck_b, &mut rng);

        // Create CPU players if needed
        let cpu_players = if cpu_codes[0].is_some() || cpu_codes[1].is_some() {
            let player_codes = vec![
                cpu_codes[0].clone().unwrap_or(PlayerCode::R),
                cpu_codes[1].clone().unwrap_or(PlayerCode::R),
            ];
            Some(
                create_players(deck_a, deck_b, player_codes).expect("Failed to create CPU players"),
            )
        } else {
            None
        };

        Self {
            state,
            rng,
            cpu_players,
            step_count: 0,
        }
    }

    #[inline]
    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    #[inline]
    fn get_outcome(&self) -> Option<GameOutcome> {
        self.state.winner
    }

    #[inline]
    fn current_player(&self) -> usize {
        self.state.current_player
    }
}

/// Pre-allocated buffers for batched operations
struct BatchBuffers {
    /// Observation buffer: n_games * OBSERVATION_SIZE
    #[allow(dead_code)]
    observations: Vec<f32>,
    /// Action mask buffer: n_games * ACTION_SPACE_SIZE
    #[allow(dead_code)]
    masks: Vec<bool>,
    /// Active game indices buffer
    active_indices: Vec<usize>,
    /// Actions result buffer
    actions: Vec<usize>,
}

impl BatchBuffers {
    fn new(n_games: usize) -> Self {
        Self {
            observations: vec![0.0; n_games * OBSERVATION_SIZE],
            masks: vec![false; n_games * ACTION_SPACE_SIZE],
            active_indices: Vec::with_capacity(n_games),
            actions: Vec::with_capacity(n_games),
        }
    }

    fn clear(&mut self) {
        self.active_indices.clear();
        self.actions.clear();
    }
}

/// Batched game runner for GPU-optimized ONNX simulation
pub struct BatchedGameRunner {
    /// All game instances
    games: Vec<GameInstance>,
    /// Number of games
    n_games: usize,
    /// Player configurations
    player_configs: [PlayerConfig; 2],
    /// ONNX sessions keyed by (path, device) - allows sharing when same model+device
    #[cfg(feature = "onnx")]
    onnx_sessions: HashMap<(String, String), BatchedOnnxInference>,
    /// Keys into onnx_sessions for each player (None if CPU player)
    #[cfg(feature = "onnx")]
    player_onnx_keys: [Option<(String, String)>; 2],
    /// Shared RNG for ONNX sampling
    #[cfg(feature = "onnx")]
    onnx_rng: StdRng,
    /// Pre-allocated buffers for batched operations
    buffers: BatchBuffers,
    /// Progress bar
    progress: Option<ProgressBar>,
    /// Track completed games
    completed_count: usize,
    /// Offset for global progress (used in matrix simulations)
    progress_offset: usize,
    /// Total games for global progress
    progress_total: usize,
    /// Timing stats for performance analysis
    timing_stats: TimingStats,
}

/// Accumulated timing statistics
#[derive(Default)]
struct TimingStats {
    obs_collection: Duration,
    mask_collection: Duration,
    onnx_inference: Duration,
    action_resolution: Duration,
    action_application: Duration,
}

impl BatchedGameRunner {
    /// Create a new batched game runner with multiple deck pairs
    pub fn new(
        deck_pairs: Vec<(Deck, Deck)>,
        player_codes: Vec<PlayerCode>,
        seed: Option<u64>,
        progress_offset: usize,
        progress_total: usize,
    ) -> Result<Self, String> {
        let n_games = deck_pairs.len();
        let base_seed = seed.unwrap_or_else(rand::random);

        // Parse player configurations
        let (player_configs, cpu_codes) = Self::parse_player_configs(&player_codes)?;

        // Initialize ONNX sessions if needed
        #[cfg(feature = "onnx")]
        let (onnx_sessions, player_onnx_keys) =
            Self::init_onnx_sessions(&player_configs, base_seed)?;

        // Create game instances in parallel
        let games: Vec<GameInstance> = deck_pairs
            .into_par_iter()
            .enumerate()
            .map(|(i, (deck_a, deck_b))| {
                GameInstance::new(deck_a, deck_b, base_seed.wrapping_add(i as u64), &cpu_codes)
            })
            .collect();

        // Pre-allocate buffers
        let buffers = BatchBuffers::new(n_games);

        Ok(Self {
            games,
            n_games,
            player_configs,
            #[cfg(feature = "onnx")]
            onnx_sessions,
            #[cfg(feature = "onnx")]
            player_onnx_keys,
            #[cfg(feature = "onnx")]
            onnx_rng: StdRng::seed_from_u64(base_seed.wrapping_add(999999)),
            buffers,
            progress: None,
            completed_count: 0,
            progress_offset,
            progress_total,
            timing_stats: TimingStats::default(),
        })
    }

    /// Parse player codes into configurations
    fn parse_player_configs(
        player_codes: &[PlayerCode],
    ) -> Result<([PlayerConfig; 2], [Option<PlayerCode>; 2]), String> {
        if player_codes.len() != 2 {
            return Err("Exactly 2 player codes required".to_string());
        }

        let mut configs = Vec::with_capacity(2);
        let mut cpu_codes: [Option<PlayerCode>; 2] = [None, None];

        for (i, code) in player_codes.iter().enumerate() {
            match code {
                #[cfg(feature = "onnx")]
                PlayerCode::Onnx { path, device } => {
                    configs.push(PlayerConfig::Onnx {
                        path: path.clone(),
                        device: device.clone(),
                    });
                }
                #[cfg(feature = "onnx")]
                PlayerCode::O { index, device } => {
                    let path = crate::players::get_nth_onnx_model(*index)
                        .map_err(|e| format!("Failed to find model o{}: {}", index, e))?;
                    configs.push(PlayerConfig::Onnx {
                        path,
                        device: device.clone(),
                    });
                }
                _ => {
                    // CPU player
                    configs.push(PlayerConfig::Cpu { code: code.clone() });
                    cpu_codes[i] = Some(code.clone());
                }
            }
        }

        Ok(([configs[0].clone(), configs[1].clone()], cpu_codes))
    }

    /// Initialize ONNX sessions, sharing when path and device match
    #[cfg(feature = "onnx")]
    fn init_onnx_sessions(
        player_configs: &[PlayerConfig; 2],
        _base_seed: u64,
    ) -> Result<
        (
            HashMap<(String, String), BatchedOnnxInference>,
            [Option<(String, String)>; 2],
        ),
        String,
    > {
        let mut sessions: HashMap<(String, String), BatchedOnnxInference> = HashMap::new();
        let mut keys: [Option<(String, String)>; 2] = [None, None];

        for (i, config) in player_configs.iter().enumerate() {
            if let PlayerConfig::Onnx { path, device } = config {
                let key = (path.clone(), device.clone());

                // Only create session if not already cached
                if !sessions.contains_key(&key) {
                    let session = BatchedOnnxInference::new(path, false, device)
                        .map_err(|e| format!("Failed to create ONNX session: {}", e))?;
                    sessions.insert(key.clone(), session);
                }

                keys[i] = Some(key);
            }
        }

        Ok((sessions, keys))
    }

    /// Run all games to completion
    pub fn run_all(mut self) -> Vec<Option<GameOutcome>> {
        // Create progress bar
        self.progress = Some(create_progress_bar(self.n_games as u64));

        let mut total_steps = 0u64;
        let mut total_actions = 0u64;

        // Main loop: step all games until all complete
        loop {
            // Clear buffers for this iteration
            self.buffers.clear();

            // Find active games for each player and collect indices
            let (p0_active, p1_active) = self.find_active_games_partitioned();

            if p0_active.is_empty() && p1_active.is_empty() {
                // All games done
                break;
            }

            total_steps += 1;
            total_actions += (p0_active.len() + p1_active.len()) as u64;

            if total_steps % 10 == 0 {
                log::info!(
                    "[{}] Iteration {}, active games: {}",
                    "INFO",
                    total_steps,
                    p0_active.len() + p1_active.len()
                );
            }

            // Step player 0's games
            if !p0_active.is_empty() {
                for &idx in &p0_active {
                    self.games[idx].step_count += 1;
                }
                self.step_player(0, &p0_active);
            }

            // Step player 1's games
            if !p1_active.is_empty() {
                for &idx in &p1_active {
                    self.games[idx].step_count += 1;
                }
                self.step_player(1, &p1_active);
            }

            // Safety check for infinite loops
            const MAX_STEPS_PER_GAME: usize = 2000;
            for game in &mut self.games {
                if !game.is_game_over() && game.step_count > MAX_STEPS_PER_GAME {
                    log::warn!(
                        "[WARNING] Game exceeded {} steps. Forcing Tie.",
                        MAX_STEPS_PER_GAME
                    );
                    game.state.winner = Some(GameOutcome::Tie);
                }
            }

            // Update progress for newly completed games
            let new_completed = self.count_completed_games();
            if new_completed > self.completed_count {
                if let Some(ref pb) = self.progress {
                    pb.set_position(new_completed as u64);
                }

                // Print a parsable progress line for Python UI
                let global_completed = self.progress_offset + new_completed;
                log::info!(
                    "[{}] [#######] {}/{}",
                    "PROGRESS",
                    global_completed,
                    self.progress_total
                );

                self.completed_count = new_completed;
            }
        }

        // Finish progress bar
        if let Some(pb) = self.progress.take() {
            pb.finish_with_message("Simulation complete!");
        }

        // Print stats (timing breakdown available via ONNX_DEBUG=1)
        let ts = &self.timing_stats;
        let total_time = ts.obs_collection
            + ts.mask_collection
            + ts.onnx_inference
            + ts.action_resolution
            + ts.action_application;

        eprintln!(
            "  [Batched] {} iterations, avg batch {:.0}, {:.1}% in inference",
            total_steps,
            total_actions as f64 / total_steps as f64,
            100.0 * ts.onnx_inference.as_secs_f64() / total_time.as_secs_f64()
        );

        // Detailed timing breakdown if ONNX_DEBUG is set
        if std::env::var("ONNX_DEBUG").is_ok() {
            eprintln!(
                "  [Timing] obs:{:.2}s mask:{:.2}s infer:{:.2}s resolve:{:.2}s apply:{:.2}s",
                ts.obs_collection.as_secs_f64(),
                ts.mask_collection.as_secs_f64(),
                ts.onnx_inference.as_secs_f64(),
                ts.action_resolution.as_secs_f64(),
                ts.action_application.as_secs_f64()
            );
        }

        // Collect outcomes
        self.games.iter().map(|g| g.get_outcome()).collect()
    }

    /// Find active games partitioned by current player (more efficient than two passes)
    fn find_active_games_partitioned(&self) -> (Vec<usize>, Vec<usize>) {
        let mut p0_active = Vec::with_capacity(self.n_games / 2);
        let mut p1_active = Vec::with_capacity(self.n_games / 2);

        for (i, game) in self.games.iter().enumerate() {
            if !game.is_game_over() {
                if game.current_player() == 0 {
                    p0_active.push(i);
                } else {
                    p1_active.push(i);
                }
            }
        }

        (p0_active, p1_active)
    }

    /// Count completed games efficiently
    #[inline]
    fn count_completed_games(&self) -> usize {
        self.games.iter().filter(|g| g.is_game_over()).count()
    }

    /// Step all games in the given indices for the specified player
    fn step_player(&mut self, player: usize, game_indices: &[usize]) {
        match &self.player_configs[player] {
            #[cfg(feature = "onnx")]
            PlayerConfig::Onnx { .. } => {
                self.step_onnx_player_optimized(player, game_indices);
            }
            PlayerConfig::Cpu { .. } => {
                self.step_cpu_player_parallel(player, game_indices);
            }
        }
    }

    /// Step ONNX player with optimized batched inference
    #[cfg(feature = "onnx")]
    fn step_onnx_player_optimized(&mut self, player: usize, game_indices: &[usize]) {
        if game_indices.is_empty() {
            return;
        }

        let key = match &self.player_onnx_keys[player] {
            Some(k) => k.clone(),
            None => return,
        };

        let n_active = game_indices.len();

        // Parallel observation collection into pre-allocated buffer
        let t0 = Instant::now();
        game_indices
            .par_iter()
            .enumerate()
            .for_each(|(batch_idx, &game_idx)| {
                let obs = get_observation_tensor(&self.games[game_idx].state, player);
                let start = batch_idx * OBSERVATION_SIZE;
                // Safe because each batch_idx is unique and we pre-allocated enough space
                unsafe {
                    let ptr = self.buffers.observations.as_ptr() as *mut f32;
                    std::ptr::copy_nonoverlapping(obs.as_ptr(), ptr.add(start), OBSERVATION_SIZE);
                }
            });
        self.timing_stats.obs_collection += t0.elapsed();

        // Parallel mask collection into pre-allocated buffer
        let t1 = Instant::now();
        game_indices
            .par_iter()
            .enumerate()
            .for_each(|(batch_idx, &game_idx)| {
                let mask = get_action_mask(&self.games[game_idx].state);
                let start = batch_idx * ACTION_SPACE_SIZE;
                unsafe {
                    let ptr = self.buffers.masks.as_ptr() as *mut bool;
                    std::ptr::copy_nonoverlapping(mask.as_ptr(), ptr.add(start), ACTION_SPACE_SIZE);
                }
            });
        self.timing_stats.mask_collection += t1.elapsed();

        // Get the slice of observations and masks we actually need
        let obs_slice = &self.buffers.observations[..n_active * OBSERVATION_SIZE];
        let mask_slice = &self.buffers.masks[..n_active * ACTION_SPACE_SIZE];

        // Run batched inference (single GPU call for all active games)
        let t2 = Instant::now();
        let actions = {
            let session = self.onnx_sessions.get_mut(&key).unwrap();
            session.predict_batch(obs_slice, mask_slice, n_active, &mut self.onnx_rng)
        };
        self.timing_stats.onnx_inference += t2.elapsed();

        // Apply actions in parallel
        // Need to collect indexed actions first since apply_action needs mutable access
        let t3 = Instant::now();
        let action_data: Vec<_> = game_indices
            .par_iter()
            .enumerate()
            .map(|(batch_idx, &game_idx)| {
                let action_idx = actions[batch_idx];
                let indexed_actions = get_indexed_actions(&self.games[game_idx].state);

                // Find the action to apply
                let action = if let Some((_, act)) =
                    indexed_actions.iter().find(|(idx, _)| *idx == action_idx)
                {
                    act.clone()
                } else if let Some((_, act)) = indexed_actions.first() {
                    act.clone()
                } else {
                    return None;
                };

                Some((game_idx, action))
            })
            .collect();
        self.timing_stats.action_resolution += t3.elapsed();

        // Apply actions in parallel (apply_action needs mutable state, but each game is independent)
        let t4 = Instant::now();
        let mut actions_to_apply = vec![None; self.games.len()];
        for data in action_data {
            if let Some((idx, action)) = data {
                actions_to_apply[idx] = Some(action);
            }
        }

        self.games.par_iter_mut().enumerate().for_each(|(i, game)| {
            if let Some(action) = &actions_to_apply[i] {
                apply_action(&mut game.rng, &mut game.state, action);
            }
        });
        self.timing_stats.action_application += t4.elapsed();
    }

    /// Step CPU player with parallel execution
    fn step_cpu_player_parallel(&mut self, player: usize, game_indices: &[usize]) {
        if game_indices.is_empty() {
            return;
        }

        let t0 = Instant::now();

        // Create a mask for active games for fast lookup in par_iter
        let mut mask = vec![false; self.games.len()];
        for &idx in game_indices {
            mask[idx] = true;
        }

        // Use Rayon to process all games in parallel
        // Each GameInstance owns its RNG and CPU players, so this is safe and fast
        self.games
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .for_each(|(_, game)| {
                // Get possible actions
                let (_, actions) = generate_possible_actions(&game.state);
                if actions.is_empty() {
                    return;
                }

                // Use CPU player to select action
                let action = if actions.len() == 1 {
                    actions[0].clone()
                } else if let Some(ref mut players) = game.cpu_players {
                    players[player].decision_fn(&mut game.rng, &game.state, &actions)
                } else {
                    // Fallback to random
                    use rand::seq::SliceRandom;
                    actions.choose(&mut game.rng).unwrap().clone()
                };

                apply_action(&mut game.rng, &mut game.state, &action);
            });

        // We'll attribute this to action_resolution for now in the stats
        self.timing_stats.action_resolution += t0.elapsed();
    }
}

/// Check if a player code is an ONNX player
#[cfg(feature = "onnx")]
pub fn is_onnx_player(code: &PlayerCode) -> bool {
    matches!(code, PlayerCode::Onnx { .. } | PlayerCode::O { .. })
}

#[cfg(not(feature = "onnx"))]
pub fn is_onnx_player(_code: &PlayerCode) -> bool {
    false
}
