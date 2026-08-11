//! The two deck DBs of `RL_ARCHITECTURE.md` §1.5.3, on disk and in memory.
//!
//! §1.5.3 has no static dataset: the "dataset" is the deck sampler, and what it samples from is
//! two directories built by `auxiliaries/build_deck_dbs.py` out of the JSON archives —
//! **`meta`** (O(100k) decks) and **`tutorial`** (O(1k)). Only `playable` decks are compiled in:
//! a deck carrying a `blockers` entry names a card the engine cannot run, so drawing one would
//! fail at game construction.
//!
//! **One file per archetype, many decks per file.** The archetype is a deck label in `meta` and
//! the difficulty tier in `tutorial`. Three things want the grouping: §1.5.3's mirror quota needs
//! "another deck of the same archetype" at draw time; a run needs to draw one tutorial tier
//! rather than mixing beginner with expert; and 70k loose files is a directory walk no run should
//! pay at startup.
//!
//! **Decks are held as text and parsed on draw.** A [`Deck`] owns its 20 [`Card`](crate::Card)s
//! by value, descriptors and attack strings included, so materializing the meta DB eagerly means
//! 1.4M cloned cards resident for the whole run. The text block is a couple hundred bytes, and
//! parsing one costs nothing against a game that takes ~100 ms.
//!
//! The block format is the ordinary [`Deck::from_string`] text format with a `# <id>` header line
//! prepended, blocks separated by a blank line. The header is stripped here rather than taught to
//! `deck.rs`, which stays the single-deck parser it is.

use std::fs;
use std::path::Path;

use crate::Deck;

/// One deck in a DB: its archive id (a hash of the card multiset, assigned upstream) and the text
/// it was written as. [`DeckEntry::build`] is what turns it into a playable [`Deck`].
#[derive(Debug, Clone)]
pub struct DeckEntry {
    pub id: String,
    text: String,
}

impl DeckEntry {
    pub fn build(&self) -> Result<Deck, String> {
        Deck::from_string(&self.text).map_err(|err| format!("deck {}: {err}", self.id))
    }
}

/// The decks sharing an archetype label — the unit §1.5.3's mirror quota draws twice from.
#[derive(Debug, Clone)]
pub struct Archetype {
    pub name: String,
    pub decks: Vec<DeckEntry>,
}

/// One compiled deck DB: `decks/meta` or `decks/tutorial`.
#[derive(Debug, Clone)]
pub struct DeckDb {
    pub name: String,
    pub archetypes: Vec<Archetype>,
}

impl DeckDb {
    /// Loads every `*.txt` under `dir`, one archetype per file, named after the file stem.
    ///
    /// Archetypes are ordered by name and decks by their order in the file, both of which the
    /// generator fixes, so a given directory always yields the same indices — §1.5.5's
    /// reproducibility guarantee reaches the deck draw only if the draw's index space is stable.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let name = dir
            .file_name()
            .map(|stem| stem.to_string_lossy().into_owned())
            .ok_or_else(|| format!("deck db path has no final component: {}", dir.display()))?;

        let files: Vec<_> = fs::read_dir(dir)
            .map_err(|err| format!("failed to read deck db {}: {err}", dir.display()))?
            .map(|entry| entry.map_err(|err| format!("failed to walk {}: {err}", dir.display())))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "txt"))
            .collect();

        let mut archetypes = Vec::with_capacity(files.len());
        for path in files {
            let label = path
                .file_stem()
                .expect("a *.txt path has a stem")
                .to_string_lossy()
                .into_owned();
            let contents = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            let decks = parse_blocks(&contents, &label)?;
            if decks.is_empty() {
                return Err(format!("archetype file {} is empty", path.display()));
            }
            archetypes.push(Archetype { name: label, decks });
        }

        if archetypes.is_empty() {
            return Err(format!("deck db {} contains no *.txt file", dir.display()));
        }
        archetypes.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(DeckDb { name, archetypes })
    }

    /// Total decks across archetypes — the population a deck-uniform draw runs over.
    pub fn deck_count(&self) -> usize {
        self.archetypes.iter().map(|a| a.decks.len()).sum()
    }
}

/// Splits an archetype file into `# <id>` headed blocks, in file order.
///
/// Line endings are normalized first: the DBs are versioned text, so a checkout with
/// `core.autocrlf` on hands this function CRLF whatever the generator wrote.
fn parse_blocks(contents: &str, label: &str) -> Result<Vec<DeckEntry>, String> {
    let contents = contents.replace("\r\n", "\n");
    let mut entries: Vec<DeckEntry> = Vec::new();
    for (position, block) in contents.split("\n\n").enumerate() {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let (header, body) = block.split_once('\n').ok_or_else(|| {
            format!("{label} block {position}: expected a `# <id>` header and a deck body")
        })?;
        let id = header.strip_prefix("# ").ok_or_else(|| {
            format!("{label} block {position}: expected a `# <id>` header, got {header:?}")
        })?;
        let id = id.trim().to_string();
        check_energy(body, label, &id)?;
        entries.push(DeckEntry {
            id,
            text: body.to_string(),
        });
    }
    Ok(entries)
}

/// A compiled deck must declare its energy explicitly, and only energy a deck can actually hold.
///
/// `Colorless` and `Dragon` are attack costs and Pokémon types, never deck energy — `get_color`
/// in `game.rs` asserts as much with a `todo!()`, so one such deck reaching a logged game panics
/// mid-run. A few preset decks carry one because the upstream extractor read the headline
/// Pokémon's type as the deck's energy; the generator rewrites those (Dragon runs on Water +
/// Lightning, Colorless drops out or becomes Water), and this rejects any that a hand-edit or a
/// future archive puts back.
///
/// An absent `Energy:` line is rejected for the same reason: [`Deck::from_string`] would fall
/// back to deriving the energy from the cards, which is that same guess.
fn check_energy(body: &str, label: &str, id: &str) -> Result<(), String> {
    let declared = body
        .lines()
        .find_map(|line| line.trim().strip_prefix("Energy:"))
        .ok_or_else(|| format!("{label} deck {id}: no `Energy:` line"))?;

    for energy in declared.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        if matches!(energy, "Colorless" | "Dragon") {
            return Err(format!("{label} deck {id}: {energy} is not a deck energy"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> DeckDb {
        DeckDb::load(Path::new("decks/meta")).expect("meta db")
    }

    #[test]
    fn tutorial_db_loads_and_every_deck_builds() {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        assert_eq!(db.name, "tutorial");
        assert!(db.deck_count() > 100, "tutorial is O(1k) decks");

        for archetype in &db.archetypes {
            for entry in &archetype.decks {
                let deck = entry.build().expect("every compiled deck is playable");
                assert_eq!(deck.cards.len(), 20, "deck {} is not 20 cards", entry.id);
            }
        }
    }

    /// The meta DB is too large to build in full here; the invariants that matter at load are the
    /// shape ones, and a sample covers whether the block format survived the generator.
    #[test]
    fn meta_db_loads_with_stable_indices() {
        let db = meta();
        assert!(db.deck_count() > 10_000, "meta is O(100k) decks");
        assert!(db
            .archetypes
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name));

        let again = meta();
        let ids = |db: &DeckDb| {
            db.archetypes[0]
                .decks
                .iter()
                .map(|d| d.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&db), ids(&again));

        for archetype in db.archetypes.iter().take(20) {
            let entry = &archetype.decks[0];
            assert_eq!(entry.build().expect("playable").cards.len(), 20);
        }
    }

    #[test]
    fn a_block_without_a_header_is_rejected() {
        let err = parse_blocks("Energy: Fire\n2 A1 042", "x").expect_err("no header");
        assert!(err.contains("header"), "{err}");
    }

    #[test]
    fn a_deck_declaring_a_non_deck_energy_is_rejected() {
        for energy in ["Colorless", "Dragon", "Psychic, Colorless"] {
            let block = format!("# abc\nEnergy: {energy}\n2 A1 042");
            let err = parse_blocks(&block, "x").expect_err("illegal energy");
            assert!(err.contains("not a deck energy"), "{err}");
        }
        parse_blocks("# abc\nEnergy: Psychic, Fire\n2 A1 042", "x").expect("legal energy");
    }

    #[test]
    fn a_deck_without_an_energy_line_is_rejected() {
        let err = parse_blocks("# abc\n2 A1 042", "x").expect_err("no energy");
        assert!(err.contains("`Energy:` line"), "{err}");
    }
}
