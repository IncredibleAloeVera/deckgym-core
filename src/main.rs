use clap::{ArgAction, Parser, Subcommand};
use colored::Colorize;
use deckgym::optimize::{ParallelConfig, SimulationConfig};
use deckgym::players::{fill_code_array, parse_player_code, PlayerCode};
use deckgym::simulate::initialize_logger;
use deckgym::state::GameOutcome;
use deckgym::{cli_optimize, is_onnx_player, simulate, simulate_batched, BatchedGameRunner, Deck};
use log::warn;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Simulate games between two decks (or one deck against multiple decks in a folder)
    Simulate {
        /// Path to the first deck file
        deck_a: String,

        /// Path to the second deck file or folder containing multiple deck files
        deck_b_or_folder: String,

        /// Players' strategies as a comma-separated list (e.g., "e2,e4" or "r,e5")
        /// Available codes: aa, et, r, h, w, m, v, e<depth>, er
        /// Example: e2 = ExpectiMiniMax with depth 2
        #[arg(long, value_delimiter = ',', value_parser = parse_player_code)]
        players: Option<Vec<PlayerCode>>,

        /// Number of simulations to run
        #[arg(short, long)]
        num: u32,

        /// Seed for random number generation
        #[arg(short, long)]
        seed: Option<u64>,

        /// Run simulations in parallel
        #[arg(short, long, default_value_t = false)]
        parallel: bool,

        /// Number of threads to use (defaults to number of CPU cores if not specified)
        #[arg(short = 'j', long)]
        threads: Option<usize>,

        /// Increase verbosity (-v, -vv, -vvv, etc.)
        #[arg(short, long, action = ArgAction::Count, default_value_t = 1)]
        verbose: u8,

        /// Output folder for exporting (state, action) pairs in JSON format
        #[arg(long)]
        data_output: Option<String>,
    },
    /// Optimize an incomplete deck against enemy decks
    Optimize {
        /// Path to the incomplete deck file (missing up to 4 cards)
        incomplete_deck: String,

        /// Comma-separated list of candidate card IDs for completion
        candidate_cards: String,

        /// Folder containing enemy deck files
        enemy_decks_folder: String,

        /// Number of simulations to run per enemy deck for each combination
        #[arg(short, long)]
        num: u32,

        /// Players' strategies as a comma-separated list (e.g., "e2,e4" or "r,e5")
        /// Available codes: aa, et, r, h, w, m, v, e<depth>, er
        /// Example: e2 = ExpectiMiniMax with depth 2
        #[arg(long, value_delimiter = ',', value_parser = parse_player_code)]
        players: Option<Vec<PlayerCode>>,

        /// Seed for random number generation
        #[arg(short, long)]
        seed: Option<u64>,

        /// Run simulations in parallel
        #[arg(short, long, default_value_t = false)]
        parallel: bool,

        /// Number of threads to use (defaults to number of CPU cores if not specified)
        #[arg(short = 'j', long)]
        threads: Option<usize>,

        /// Increase verbosity (-v, -vv, -vvv, etc.)
        #[arg(short, long, action = ArgAction::Count, default_value_t = 1)]
        verbose: u8,
    },
}

fn simulate_against_folder(
    path_a: &str,
    path_b: &str,
    players: Option<Vec<PlayerCode>>,
    num_games_total: u32,
    seed: Option<u64>,
    parallel: bool,
    num_threads: Option<usize>,
) {
    let player_codes = fill_code_array(players.clone());
    let has_onnx = player_codes.iter().any(|p| is_onnx_player(p));

    // Helper to get all decks in a path (file or folder)
    let get_decks = |path: &str| -> Vec<String> {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            std::fs::read_dir(p)
                .expect("Failed to read directory")
                .filter_map(|e| {
                    let path = e.ok()?.path();
                    if path.is_file()
                        && (path.extension() == Some(std::ffi::OsStr::new("txt"))
                            || path.extension() == Some(std::ffi::OsStr::new("json")))
                    {
                        Some(path.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![path.to_string()]
        }
    };

    let decks_a_paths = get_decks(path_a);
    let decks_b_paths = get_decks(path_b);

    if decks_a_paths.is_empty() || decks_b_paths.is_empty() {
        warn!("No decks found in directory!");
        return;
    }

    // Pre-load all decks to avoid redundant Disk I/O during matrix generation
    println!(
        "Loading {} decks...",
        decks_a_paths.len() + decks_b_paths.len()
    );
    let decks_a: Vec<Deck> = decks_a_paths
        .iter()
        .map(|p| Deck::from_file(p).expect("Failed to load deck A"))
        .collect();
    let decks_b: Vec<Deck> = decks_b_paths
        .iter()
        .map(|p| Deck::from_file(p).expect("Failed to load deck B"))
        .collect();

    let num_matchups = (decks_a.len() * decks_b.len()) as u32;
    let games_per_matchup = num_games_total / num_matchups;
    let remainder = num_games_total % num_matchups;

    warn!(
        "Running {} total games across {} matchups ({}/{} decks)...",
        num_games_total,
        num_matchups,
        decks_a.len(),
        decks_b.len()
    );

    if has_onnx {
        // Optimized case: Load all matchups into a single batch
        let mut deck_pairs = Vec::with_capacity(num_games_total as usize);
        let mut matchup_idx = 0;
        for deck_a in &decks_a {
            for deck_b in &decks_b {
                let count = if matchup_idx < remainder {
                    games_per_matchup + 1
                } else {
                    games_per_matchup
                };

                if count > 0 {
                    for _ in 0..count {
                        deck_pairs.push((deck_a.clone(), deck_b.clone()));
                    }
                }
                matchup_idx += 1;
            }
        }

        if !deck_pairs.is_empty() {
            let mut outcomes = Vec::with_capacity(deck_pairs.len());

            let mut completed_before = 0;
            // Split into chunks to avoid CUDA OOM and improve progress feedback
            const MAX_BATCH_SIZE: usize = 1024;
            for chunk in deck_pairs.chunks(MAX_BATCH_SIZE) {
                let chunk_vec = chunk.to_vec();
                let runner = BatchedGameRunner::new(
                    chunk_vec,
                    player_codes.clone(),
                    seed,
                    completed_before,
                    num_games_total as usize,
                )
                .expect("Failed to create BatchRunner");
                let chunk_outcomes = runner.run_all();
                outcomes.extend(chunk_outcomes);
                completed_before += chunk.len();
            }

            // Collect results back into matchups
            let mut matchup_idx = 0;
            let mut outcome_idx = 0;
            let mut a_wins_total = 0;
            let mut b_wins_total = 0;
            let mut ties_total = 0;

            for (i, _) in decks_a.iter().enumerate() {
                for (j, _) in decks_b.iter().enumerate() {
                    let count = if matchup_idx < remainder {
                        games_per_matchup + 1
                    } else {
                        games_per_matchup
                    };

                    if count > 0 {
                        let mut w = 0;
                        let mut l = 0;
                        let mut t = 0;
                        for _ in 0..count {
                            match outcomes[outcome_idx] {
                                Some(GameOutcome::Win(0)) => w += 1,
                                Some(GameOutcome::Win(1)) => l += 1,
                                Some(GameOutcome::Tie) | None => t += 1,
                                _ => {}
                            }
                            outcome_idx += 1;
                        }

                        // Print parseable result for Python
                        warn!(
                            "RESULT: {} VS {} | WINS: {} LOSSES: {} TIES: {}",
                            decks_a_paths[i], decks_b_paths[j], w, l, t
                        );

                        a_wins_total += w;
                        b_wins_total += l;
                        ties_total += t;
                    }
                    matchup_idx += 1;
                }
            }

            let total = outcomes.len() as f64;
            warn!("\nSummary Results (All Matchups Combined):");
            warn!(
                "Player 0 won: {} ({:.2}%)",
                a_wins_total,
                (a_wins_total as f64 / total) * 100.0
            );
            warn!(
                "Player 1 won: {} ({:.2}%)",
                b_wins_total,
                (b_wins_total as f64 / total) * 100.0
            );
            warn!(
                "Draws: {} ({:.2}%)",
                ties_total,
                (ties_total as f64 / total) * 100.0
            );
        }
    } else {
        // Sequential/CPU fallback (though rarely used now for scientific eval)
        let mut matchup_idx = 0;
        for (i, _) in decks_a.iter().enumerate() {
            for (j, _) in decks_b.iter().enumerate() {
                let count = if matchup_idx < remainder {
                    games_per_matchup + 1
                } else {
                    games_per_matchup
                };

                if count > 0 {
                    warn!(
                        "\nMatchup {}/{}: {} vs {}",
                        matchup_idx + 1,
                        num_matchups,
                        decks_a_paths[i],
                        decks_b_paths[j]
                    );
                    simulate(
                        &decks_a_paths[i],
                        &decks_b_paths[j],
                        SimulationConfig {
                            num_games: count,
                            players: players.clone(),
                            seed,
                            data_output: None,
                        },
                        ParallelConfig {
                            enabled: parallel,
                            num_threads,
                        },
                    );
                }
                matchup_idx += 1;
            }
        }
    }

    warn!("\n{}", "=".repeat(60));
    warn!("All simulations complete!");
    warn!("{}", "=".repeat(60));
}

fn main() {
    let cli = Cli::parse();

    // Branch depending on the chosen subcommand.
    match cli.command {
        Commands::Simulate {
            deck_a,
            deck_b_or_folder,
            players,
            num,
            seed,
            parallel,
            threads,
            verbose,
            data_output,
        } => {
            initialize_logger(verbose);

            warn!("Welcome to {} simulation!", "deckgym".blue().bold());
            #[cfg(feature = "onnx")]
            deckgym::players::print_available_providers();

            // Fill in default players if not specified
            let player_codes = fill_code_array(players.clone());

            // Check if any player uses ONNX - use batched mode for GPU efficiency
            let has_onnx = player_codes.iter().any(|p| is_onnx_player(p));

            // Check if either is a directory
            if std::path::Path::new(&deck_a).is_dir()
                || std::path::Path::new(&deck_b_or_folder).is_dir()
            {
                simulate_against_folder(
                    &deck_a,
                    &deck_b_or_folder,
                    Some(player_codes),
                    num,
                    seed,
                    parallel,
                    threads,
                );
            } else if has_onnx {
                // ONNX detected: use batched inference for GPU efficiency
                simulate_batched(&deck_a, &deck_b_or_folder, player_codes, num, seed);
            } else {
                // CPU-only: use standard rayon-based parallel simulation
                simulate(
                    &deck_a,
                    &deck_b_or_folder,
                    SimulationConfig {
                        num_games: num,
                        players,
                        seed,
                        data_output,
                    },
                    ParallelConfig {
                        enabled: parallel,
                        num_threads: threads,
                    },
                );
            }
        }
        Commands::Optimize {
            incomplete_deck,
            candidate_cards,
            enemy_decks_folder,
            num,
            players,
            seed,
            parallel,
            threads,
            verbose,
        } => {
            initialize_logger(verbose);

            warn!("Welcome to {} optimizer!", "deckgym".blue().bold());

            let sim_config = SimulationConfig {
                num_games: num,
                players,
                seed,
                data_output: None,
            };
            let parallel_config = ParallelConfig {
                enabled: parallel,
                num_threads: threads,
            };

            cli_optimize(
                &incomplete_deck,
                &candidate_cards,
                &enemy_decks_folder,
                sim_config,
                parallel_config,
            );
        }
    }
}
