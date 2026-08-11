//! The §1.5.7 label harvest, wired to the training loop.
//!
//! Pretraining runs the games anyway; logging them is the cheapest label source Part 6 will get.
//! This module is the plumbing — [`crate::gameplay_stats_collector`] is the measurement.
//!
//! **Additive shards, never a snapshot.** A flush writes only the games since the last one, into
//! `harvest/shard-NNNNNN/`. The counters sum, so the offline merge is a group-by; a full snapshot
//! would rewrite the whole `meta` DB's worth of rows every flush for a few thousand new games.
//! It also makes a crash a non-event: a resume opens a new shard and nothing has to reconcile.
//!
//! **Three flat tables, not nested maps.** `HashMap<CardSlotKey, CardStats>` has a struct key, and
//! JSON object keys must be strings — flattening the key into two columns avoids the problem
//! rather than encoding around it, and a flat row is what pandas or polars reads in one call.
//!
//! **JSONL rather than CSV or Parquet.** Naming the fields in every row costs roughly 3× the
//! bytes, which at this scale is a few hundred MB and no object. What it buys is tolerance to
//! schema drift: §1.5.7's field list is still moving, and a shard written last month has to stay
//! readable when a counter is added.
//!
//! **Sampling is per game, never per row.** Dropping rows would break `games_never_drawn` and
//! every denominator §1.5.7 is built on — the rule that every card gets a row every game, drawn or
//! not, is exactly what stops the downstream ratios being conditioned on "was drawn".

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::gameplay_stats_collector::GameplayStatsCollector;
use crate::simulation_event_handler::{CompositeSimulationEventHandler, SimulationEventHandler};

/// How much of a run's play is harvested (§1.5.7's per-stage `log` flag).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Sampling {
    /// `false` = none, `true` = every game.
    All(bool),
    /// A probability in `[0, 1]`, drawn per game.
    Fraction(f64),
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling::All(false)
    }
}

impl Sampling {
    pub fn draws(&self, rng: &mut StdRng) -> bool {
        match self {
            Sampling::All(all) => *all,
            Sampling::Fraction(p) => rng.gen::<f64>() < *p,
        }
    }

    pub fn is_off(&self) -> bool {
        matches!(self, Sampling::All(false)) || matches!(self, Sampling::Fraction(p) if *p <= 0.0)
    }
}

/// One row of `decks.jsonl`.
#[derive(Debug, Clone, Serialize)]
struct DeckRow {
    deck_id: String,
    pilot: String,
    opponent_deck: String,
    opponent_pilot: String,
    games: u32,
    wins: u32,
    losses: u32,
    ties: u32,
    games_on_the_play: u32,
    points_scored: u32,
    points_conceded: u32,
    turns_sum: u32,
    damage_dealt_total: u32,
    deck_out_games: u32,
    deck_out_turn_sum: u32,
    hand_size_sum: u32,
    hand_size_samples: u32,
}

/// One row of `cards.jsonl`. The `(deck_id, card_id, copies_in_deck)` triple is the key §1.5.7
/// fixes as the identity: stats attach to the printed card in a given copy count, never to a
/// physical copy. `(pilot, opponent_deck, opponent_pilot)` carry over from the deck row: every
/// counter below is a measurement of play, so the matchup that produced it stays on the row that
/// reports it.
#[derive(Debug, Clone, Serialize)]
struct CardRow {
    deck_id: String,
    pilot: String,
    opponent_deck: String,
    opponent_pilot: String,
    card_id: String,
    copies_in_deck: u8,
    games: u32,
    base_hp: u32,
    copies_drawn: u32,
    times_played: u32,
    games_never_drawn: u32,
    games_drawn_never_played: u32,
    ended_in_hand: u32,
    first_play_turn_sum: u32,
    first_play_games: u32,
    ability_activations: u32,
    turns_ability_available: u32,
    attacks_used: u32,
    damage_dealt: u32,
    kos_dealt: u32,
    turns_active: u32,
    turns_benched: u32,
    damage_taken_active: u32,
    damage_taken_bench: u32,
    healing_received: u32,
    times_koed: u32,
}

#[derive(Debug, Clone, Serialize)]
struct DictionaryRow {
    deck_id: String,
    cards: Vec<String>,
}

/// Accumulates finished games and writes them out in shards.
pub struct Harvest {
    root: PathBuf,
    sampling: Sampling,
    /// Merged counters since the last flush.
    pending: GameplayStatsCollector,
    pending_games: u32,
    shard: u64,
    /// Deck ids already written to a dictionary shard. Kept so the decklist is emitted once
    /// rather than in every shard it appears in — on the `meta` DB that is the difference between
    /// a dictionary and a re-transcription of the whole DB per flush.
    described: std::collections::HashSet<String>,
}

impl Harvest {
    pub fn new(root: &Path, sampling: Sampling) -> Result<Self, String> {
        fs::create_dir_all(root)
            .map_err(|err| format!("failed to create {}: {err}", root.display()))?;
        // A resume must not reuse a shard index, and the directory is the only record of what was
        // already written — the loop state deliberately does not carry harvest bookkeeping.
        let shard = fs::read_dir(root)
            .map_err(|err| format!("failed to read {}: {err}", root.display()))?
            .filter_map(Result::ok)
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_prefix("shard-").map(str::to_string))
            })
            .filter_map(|index| index.parse::<u64>().ok())
            .max()
            .map_or(0, |last| last + 1);

        Ok(Harvest {
            root: root.to_path_buf(),
            sampling,
            pending: GameplayStatsCollector::new(),
            pending_games: 0,
            shard,
            described: std::collections::HashSet::new(),
        })
    }

    pub fn sampling(&self) -> Sampling {
        self.sampling
    }

    /// Changes the sampling rate for games spawned from here on — a curriculum stage's
    /// `harvest_log` (§1.5.4). Only the rate can change this way: a run that never built a
    /// [`Harvest`] at all (`[harvest] log = false` for the whole run) has nothing here to call
    /// this on, so a stage cannot turn harvesting on from nothing.
    pub fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    /// A fresh handler for one game, told who is sitting in each seat.
    ///
    /// One collector per game rather than one shared: the collector carries per-game state (the
    /// board snapshot it diffs damage against, the last acting card), so a single instance cannot
    /// interleave the parallel envs. The pilots are per game for the same reason — the panel
    /// redraws the far seat every spawn.
    pub fn new_handler(pilots: [String; 2]) -> CompositeSimulationEventHandler {
        let mut collector = GameplayStatsCollector::new();
        collector.set_pilots(pilots);
        CompositeSimulationEventHandler::new(vec![Box::new(collector)])
    }

    /// Folds a finished game's handler into the pending shard.
    ///
    /// The inner collector is extracted first: `GameplayStatsCollector::merge` downcasts and
    /// *panics* on a foreign type, so handing it the composite wrapper would abort the run
    /// mid-rollout.
    pub fn close_game(&mut self, handler: &CompositeSimulationEventHandler) -> Result<(), String> {
        let collector = handler
            .get_handler::<GameplayStatsCollector>()
            .ok_or("a harvested game came back without its collector")?;
        self.pending.merge(collector);
        self.pending_games += 1;
        Ok(())
    }

    pub fn pending_games(&self) -> u32 {
        self.pending_games
    }

    /// Writes the pending games as a new shard and starts a fresh one. A flush with nothing
    /// pending writes nothing, so an empty shard never appears.
    pub fn flush(&mut self) -> Result<Option<PathBuf>, String> {
        if self.pending_games == 0 {
            return Ok(None);
        }

        let dir = self.root.join(format!("shard-{:06}", self.shard));
        fs::create_dir_all(&dir)
            .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;

        let mut decks = Vec::new();
        let mut cards = Vec::new();
        let mut dictionary = Vec::new();

        for (key, stats) in self.pending.decks() {
            let deck_id = key.deck.to_string();
            let opponent_deck = key.opponent_deck.to_string();
            decks.push(DeckRow {
                deck_id: deck_id.clone(),
                pilot: key.pilot.clone(),
                opponent_deck: opponent_deck.clone(),
                opponent_pilot: key.opponent_pilot.clone(),
                games: stats.games,
                wins: stats.wins,
                losses: stats.losses,
                ties: stats.ties,
                games_on_the_play: stats.games_on_the_play,
                points_scored: stats.points_scored,
                points_conceded: stats.points_conceded,
                turns_sum: stats.turns_sum,
                damage_dealt_total: stats.damage_dealt_total,
                deck_out_games: stats.deck_out_games,
                deck_out_turn_sum: stats.deck_out_turn_sum,
                hand_size_sum: stats.hand_size_sum,
                hand_size_samples: stats.hand_size_samples,
            });

            for (card_key, card) in &stats.cards {
                cards.push(CardRow {
                    deck_id: deck_id.clone(),
                    pilot: key.pilot.clone(),
                    opponent_deck: opponent_deck.clone(),
                    opponent_pilot: key.opponent_pilot.clone(),
                    card_id: card_key.card_id.clone(),
                    copies_in_deck: card_key.copies_in_deck,
                    games: card.games,
                    base_hp: card.base_hp,
                    copies_drawn: card.copies_drawn,
                    times_played: card.times_played,
                    games_never_drawn: card.games_never_drawn,
                    games_drawn_never_played: card.games_drawn_never_played,
                    ended_in_hand: card.ended_in_hand,
                    first_play_turn_sum: card.first_play_turn_sum,
                    first_play_games: card.first_play_games,
                    ability_activations: card.ability_activations,
                    turns_ability_available: card.turns_ability_available,
                    attacks_used: card.attacks_used,
                    damage_dealt: card.damage_dealt,
                    kos_dealt: card.kos_dealt,
                    turns_active: card.turns_active,
                    turns_benched: card.turns_benched,
                    damage_taken_active: card.damage_taken_active,
                    damage_taken_bench: card.damage_taken_bench,
                    healing_received: card.healing_received,
                    times_koed: card.times_koed,
                });
            }
        }

        for (deck_id, deck) in self.pending.deck_dictionary() {
            let deck_id = deck_id.to_string();
            if self.described.insert(deck_id.clone()) {
                dictionary.push(DictionaryRow {
                    deck_id,
                    cards: deck.cards.iter().map(|card| card.get_id()).collect(),
                });
            }
        }

        write_jsonl(&dir.join("decks.jsonl"), &decks)?;
        write_jsonl(&dir.join("cards.jsonl"), &cards)?;
        write_jsonl(&dir.join("dictionary.jsonl"), &dictionary)?;

        self.pending = GameplayStatsCollector::new();
        self.pending_games = 0;
        self.shard += 1;
        Ok(Some(dir))
    }
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), String> {
    let mut file = fs::File::create(path)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    for row in rows {
        let line = serde_json::to_string(row)
            .map_err(|err| format!("failed to encode a row of {}: {err}", path.display()))?;
        writeln!(file, "{line}")
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-harvest-{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    /// A curriculum stage (§1.5.4) changing how much of the run it harvests.
    #[test]
    fn set_sampling_changes_the_rate_for_games_spawned_from_here_on() {
        let mut harvest =
            Harvest::new(&scratch("set-sampling"), Sampling::All(false)).expect("harvest");
        assert!(harvest.sampling().is_off());

        harvest.set_sampling(Sampling::Fraction(0.5));
        assert!(!harvest.sampling().is_off());
        assert_eq!(harvest.sampling(), Sampling::Fraction(0.5));
    }

    #[test]
    fn sampling_reads_a_bool_or_a_fraction() {
        assert!(Sampling::All(false).is_off());
        assert!(Sampling::Fraction(0.0).is_off());
        assert!(!Sampling::All(true).is_off());
        assert!(!Sampling::Fraction(0.5).is_off());

        let mut rng = rng();
        assert!(Sampling::All(true).draws(&mut rng));
        assert!(!Sampling::All(false).draws(&mut rng));

        let drawn = (0..1000)
            .filter(|_| Sampling::Fraction(0.25).draws(&mut rng))
            .count();
        assert!((200..300).contains(&drawn), "{drawn} of 1000 at p = 0.25");
    }

    #[test]
    fn a_flush_with_nothing_pending_writes_no_shard() {
        let dir = scratch("empty");
        let mut harvest = Harvest::new(&dir, Sampling::All(true)).expect("harvest");

        assert_eq!(harvest.flush().expect("flush"), None);
        assert_eq!(fs::read_dir(&dir).expect("dir").count(), 0);
    }

    /// A resume must not overwrite the shards the interrupted run already wrote. The directory is
    /// the only record — the hot checkpoint deliberately carries no harvest bookkeeping.
    #[test]
    fn reopening_continues_after_the_last_shard() {
        let dir = scratch("resume");
        fs::create_dir_all(dir.join("shard-000000")).expect("shard");
        fs::create_dir_all(dir.join("shard-000007")).expect("shard");

        let harvest = Harvest::new(&dir, Sampling::All(true)).expect("harvest");
        assert_eq!(harvest.shard, 8);
    }
}
