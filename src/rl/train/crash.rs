//! Crash dumps — the `crashes/` directory of §1.5.5's run layout.
//!
//! [`crate::rl::recover`] keeps a run alive through an engine panic; this is what makes that
//! survivable rather than merely quiet. A dropped game is a bug the run just walked past, and the
//! rollout is the only place it was ever seen: the state that produced it exists for the length of
//! one `poll` and is then overwritten by a fresh game.
//!
//! A dump is therefore written to be **read months later, without the run**. One JSON file per
//! crash, holding three layers:
//!
//! - the panic (message, location, forced backtrace) — where the engine gave up;
//! - the action being applied and the full serialized `State` — what it gave up *on*;
//! - `(seed, decks)` — enough to replay the game from the start, since `State` is deserializable
//!   but a state reached mid-effect is not a position anyone should trust as a starting point.
//!
//! **Capped, not rolled.** Past `keep` dumps the writer stops rather than overwriting the oldest:
//! the first occurrences are the ones worth having (a panic that fires 400 times fires the same
//! way each time), and a run that finds a systematic one must not spend its disk proving it.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::rl::env::Env;
use crate::rl::recover::EnginePanic;

/// One crash, in the shape it is written.
#[derive(Debug, Serialize)]
struct Dump<'a> {
    batch: u64,
    env: usize,
    /// Uuid of the game, so a dump can be lined up with a §1.5.7 harvest row.
    game: String,
    /// With `decks`, what replays the game.
    seed: u64,
    turn_count: u8,
    current_player: usize,
    panic_message: &'a str,
    panic_location: Option<&'a str>,
    backtrace: &'a str,
    /// `Debug` rather than the `Serialize` impl `Action` has: what one reads first is a one-line
    /// name of the action, and the structured form is in `state` anyway for anything deeper.
    last_action: Option<String>,
    decks: [&'a crate::Deck; 2],
    state: &'a crate::State,
}

/// Where a run's crashes land, and how many of them it will keep.
pub struct CrashLog {
    dir: PathBuf,
    keep: usize,
    written: usize,
    /// Crashes seen, including the ones past `keep`. This is the number the §1.5.6 log carries —
    /// the dump cap must not silently deflate the rate a run reports.
    seen: usize,
}

impl CrashLog {
    /// The directory is created lazily on the first dump: a run that never crashes should not
    /// leave an empty `crashes/` suggesting it might have.
    pub fn new(dir: &Path, keep: usize) -> Self {
        CrashLog {
            dir: dir.to_path_buf(),
            keep,
            written: 0,
            seen: 0,
        }
    }

    /// Dumps a crashed env. Returns the file written, or `None` once `keep` is reached.
    ///
    /// The env must be the one that crashed and must not have been replaced yet — the state it
    /// still holds is the subject.
    pub fn record(
        &mut self,
        panic: &EnginePanic,
        env: &Env<'_>,
        batch: u64,
        slot: usize,
    ) -> Result<Option<PathBuf>, String> {
        self.seen += 1;
        if self.written >= self.keep {
            return Ok(None);
        }

        let state = env.state();
        let dump = Dump {
            batch,
            env: slot,
            game: env.game_id().to_string(),
            seed: env.seed(),
            turn_count: state.turn_count,
            current_player: state.current_player,
            panic_message: &panic.message,
            panic_location: panic.location.as_deref(),
            backtrace: &panic.backtrace,
            last_action: env.last_action().map(|action| format!("{action:?}")),
            decks: env.decks(),
            state,
        };

        fs::create_dir_all(&self.dir)
            .map_err(|err| format!("failed to create {}: {err}", self.dir.display()))?;
        // Named by the game id rather than by a counter: a resumed run reopens this log with its
        // counter back at zero, and two dumps of two different games must not collide.
        let path = self
            .dir
            .join(format!("crash-{:08}-{}.json", batch, dump.game));
        let encoded = serde_json::to_string_pretty(&dump)
            .map_err(|err| format!("failed to encode the crash dump: {err}"))?;
        fs::write(&path, encoded)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;

        self.written += 1;
        Ok(Some(path))
    }

    /// Crashes seen by this log, dumped or not.
    pub fn seen(&self) -> usize {
        self.seen
    }
}

/// How many crashes one collection is allowed before the run stops.
///
/// The failure this guards is not the rare panic — it is the **systematic** one: a game that
/// panics on its first frame is replaced by another that does the same, and the collector spins
/// forever without ever finishing an episode. Counted per collection rather than for the run, so
/// a long run that meets one bad game an hour is not eventually killed by the sum of them.
#[derive(Debug, Clone, Copy)]
pub struct CrashBudget {
    limit: usize,
    spent: usize,
}

impl CrashBudget {
    pub fn new(limit: usize) -> Self {
        CrashBudget { limit, spent: 0 }
    }

    /// Opens a fresh collection.
    pub fn reset(&mut self) {
        self.spent = 0;
    }

    /// Charges one crash. `Err` means the run should stop — the message is what the loop reports.
    pub fn charge(&mut self) -> Result<(), String> {
        self.spent += 1;
        if self.spent > self.limit {
            return Err(format!(
                "{} engine panics in one collection (limit {}) — this is a reproducible crash, \
                 not attrition; see the run's crashes/ directory",
                self.spent, self.limit
            ));
        }
        Ok(())
    }

    pub fn spent(&self) -> usize {
        self.spent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::{create_players, PlayerCode};
    use crate::rl::env::SeatPolicy;
    use crate::rl::recover::catch;
    use crate::{Deck, Game, State};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-crash-{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// A game a few actions in, plus a panic — the two halves a dump joins.
    ///
    /// They are produced separately here on purpose: what this module does is *serialize* an env
    /// beside a panic, and whether the panic came out of that env is [`crate::rl::env`]'s subject,
    /// not this one's.
    fn crashed_env<'a>() -> (Env<'a>, EnginePanic) {
        let players = create_players(
            Deck::from_file("example_decks/venusaur-exeggutor.txt").expect("deck a"),
            Deck::from_file("example_decks/weezing-arbok.txt").expect("deck b"),
            vec![PlayerCode::R, PlayerCode::R],
        );
        let mut game = Game::new(players, 4);
        game.set_debug(false);
        game.play_until_stable();

        let env = Env::new(game, [SeatPolicy::Scripted, SeatPolicy::Scripted]);
        let empty = State::new(&Deck::default(), &Deck::default());
        let panic = catch(|| empty.get_active(0).get_remaining_hp())
            .expect_err("an empty board has no active");
        (env, panic)
    }

    /// What the dump has to answer: what broke, where, on which action, and how to get back to it.
    /// Asserted field by field because a dump is only ever read after the run that could have
    /// answered these questions is gone.
    #[test]
    fn a_dump_carries_the_panic_the_action_and_what_replays_the_game() {
        let dir = scratch("fields");
        let (env, panic) = crashed_env();

        let path = CrashLog::new(&dir, 8)
            .record(&panic, &env, 17, 3)
            .expect("write")
            .expect("a dump");

        let dump: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
        assert_eq!(dump["batch"], 17);
        assert_eq!(dump["env"], 3);
        assert_eq!(dump["seed"], 4);
        assert!(dump["panic_message"]
            .as_str()
            .expect("message")
            .contains("Active Pokemon"));
        assert!(dump["panic_location"]
            .as_str()
            .expect("location")
            .contains("state"));
        assert!(!dump["backtrace"].as_str().expect("backtrace").is_empty());
        assert!(dump["last_action"].is_string());
        assert_eq!(dump["decks"].as_array().expect("decks").len(), 2);
        // The state is the payload, and it has to come back as a `State` rather than as some
        // JSON that merely looks like one — a dump nobody can deserialize cannot be replayed.
        let state: State = serde_json::from_value(dump["state"].clone()).expect("state");
        assert_eq!(state.turn_count, env.state().turn_count);
        assert_eq!(state.hands[0].len(), env.state().hands[0].len());
    }

    /// The cap stops the writing, never the counting: §1.5.6's crash rate is read off `seen`, and
    /// a run whose disk quota silently flattened its own error curve is worse than one with no
    /// dumps at all.
    #[test]
    fn past_the_cap_dumps_stop_but_crashes_are_still_counted() {
        let dir = scratch("cap");
        let (env, panic) = crashed_env();
        let mut log = CrashLog::new(&dir, 2);

        assert!(log.record(&panic, &env, 0, 0).expect("first").is_some());
        assert!(log.record(&panic, &env, 1, 0).expect("second").is_some());
        assert!(log.record(&panic, &env, 2, 0).expect("third").is_none());

        assert_eq!(log.seen(), 3);
        assert_eq!(fs::read_dir(&dir).expect("dir").count(), 2);
    }

    /// A run that never crashes must not leave the directory behind — an empty `crashes/` reads
    /// as a run that had some.
    #[test]
    fn the_directory_is_only_created_by_a_crash() {
        let dir = scratch("lazy");
        let log = CrashLog::new(&dir, 4);
        assert!(!dir.exists());
        drop(log);
    }

    #[test]
    fn the_budget_stops_a_reproducible_crash_and_tolerates_attrition() {
        let mut budget = CrashBudget::new(2);
        assert!(budget.charge().is_ok());
        assert!(budget.charge().is_ok());
        assert!(budget.charge().is_err(), "past the limit the run must stop");

        budget.reset();
        assert_eq!(budget.spent(), 0);
        assert!(budget.charge().is_ok(), "a new collection starts fresh");
    }
}
