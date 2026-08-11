use deckgym::{
    gameplay_stats_collector::{DeckSlotKey, DeckStats, GameplayStatsCollector},
    players::PlayerCode,
    simulate::initialize_logger,
    simulation_event_handler::StatsCollector,
    Simulation,
};
use log::warn;
use num_format::{Locale, ToFormattedString};

/// Example showing how to harvest deckbuilder-facing gameplay statistics
/// (RL_ARCHITECTURE §1.5.7).
///
/// The collector stores **counts next to their denominators**, never rates, and keys every card by
/// `card_id@copies_in_deck` — a card run in 1 copy and the same card run in 2 copies are different
/// entities. This example forms a few of the ratios at print time to show which denominator goes
/// with which count.
///
/// Run with: cargo run --example gameplay_stats
fn main() {
    let num_simulations = 100;
    let deck_a_path = "example_decks/venusaur-exeggutor.txt";
    let deck_b_path = "example_decks/weezing-arbok.txt";
    let player_codes = vec![
        PlayerCode::E { max_depth: 2 },
        PlayerCode::E { max_depth: 2 },
    ];

    let pilots = [player_codes[0].to_string(), player_codes[1].to_string()];

    initialize_logger(1);

    println!("Running {num_simulations} simulations to harvest gameplay statistics...");
    println!("Deck A: {deck_a_path}");
    println!("Deck B: {deck_b_path}");
    println!();

    let mut simulation = Simulation::new(
        deck_a_path,
        deck_b_path,
        player_codes,
        num_simulations,
        None,
        true, // parallel
        None, // use default number of threads
    )
    .expect("Failed to create simulation")
    .register::<StatsCollector>()
    // Named rather than `register::<GameplayStatsCollector>()`: the labels are only worth what the
    // pilot is (§1.5.7), and the seat order here is the order `player_codes` was given in.
    .register_with_closure(move || {
        let mut collector = GameplayStatsCollector::new();
        collector.set_pilots([pilots[0].to_string(), pilots[1].to_string()]);
        Box::new(collector)
    });

    simulation.run();

    let Some(collector) = simulation.get_event_handler::<GameplayStatsCollector>() else {
        eprintln!("Failed to retrieve GameplayStatsCollector");
        return;
    };

    warn!("=== Gameplay Harvest ===");
    warn!(
        "Total games: {}",
        collector.total_games().to_formatted_string(&Locale::en)
    );

    let mut decks: Vec<_> = collector.decks().iter().collect();
    decks.sort_by_key(|(key, _)| (*key).clone());
    for (key, stats) in decks {
        warn!("");
        print_deck(key, stats);
    }
}

fn print_deck(key: &DeckSlotKey, stats: &DeckStats) {
    let games = stats.games.max(1) as f64;
    warn!(
        "--- Deck {:016x} piloted by {} vs deck {:016x} piloted by {} ---",
        key.deck, key.pilot, key.opponent_deck, key.opponent_pilot
    );
    warn!(
        "  {} games: {}W / {}L / {}T  (winrate {:.1}%)",
        stats.games,
        stats.wins,
        stats.losses,
        stats.ties,
        100.0 * stats.wins as f64 / games
    );
    warn!(
        "  avg turns {:.1} | points {}-{} | total damage dealt {}",
        stats.turns_sum as f64 / games,
        stats.points_scored,
        stats.points_conceded,
        stats.damage_dealt_total
    );
    if stats.deck_out_games > 0 {
        warn!(
            "  decked out in {} games, avg turn {:.1}",
            stats.deck_out_games,
            stats.deck_out_turn_sum as f64 / stats.deck_out_games as f64
        );
    }
    if stats.hand_size_samples > 0 {
        warn!(
            "  avg hand size {:.2}",
            stats.hand_size_sum as f64 / stats.hand_size_samples as f64
        );
    }

    // Dead-card ranking: drawn and never played is the cheapest "does the deck function" signal
    // there is — it resolves in tens of games where a synergy lift needs thousands.
    let mut cards: Vec<_> = stats.cards.iter().collect();
    cards.sort_by(|a, b| {
        b.1.games_drawn_never_played
            .cmp(&a.1.games_drawn_never_played)
    });
    warn!("  card                 copies  drawn/g  played/g  dead  dmg  KOs  tanked");
    for (key, card) in cards {
        let n = card.games.max(1) as f64;
        let tanked = if card.base_hp > 0 {
            format!(
                "{:.2}x",
                (card.damage_taken_active + card.damage_taken_bench) as f64
                    / (card.base_hp as f64 * n)
            )
        } else {
            "-".to_string()
        };
        warn!(
            "  {:<20} {:>6}  {:>7.2}  {:>8.2}  {:>4}  {:>3}  {:>3}  {:>6}",
            key.card_id,
            key.copies_in_deck,
            card.copies_drawn as f64 / n,
            card.times_played as f64 / n,
            card.games_drawn_never_played,
            card.damage_dealt,
            card.kos_dealt,
            tanked,
        );
    }

    // Abilities that were on offer and left unused — the same signal, one level down.
    for (key, card) in &stats.cards {
        if card.turns_ability_available > 0 {
            warn!(
                "  ability {}: used {} / offered on {} turns",
                key.card_id, card.ability_activations, card.turns_ability_available
            );
        }
    }
}
