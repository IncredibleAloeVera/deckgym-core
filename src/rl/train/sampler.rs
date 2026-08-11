//! The deck sampler of `RL_ARCHITECTURE.md` §1.5.3 — what stands in for a dataset.
//!
//! Experience is generated on the fly, so the only thing that shapes the distribution the player
//! trains on is which pair of decks each game gets. §1.5.3 fixes three draws:
//!
//! - **DB draw** — an ordinary pair, both seats independent.
//! - **Mirror quota** — both seats the same *archetype*, so the player meets the matchup where
//!   card advantage cancels and only piloting separates the seats.
//! - **Pure-mirror quota** — both seats the *same deck*. §1.5.7 calls this a free calibration
//!   test: over enough games it has to fit to ≈ 50 % (modulo `on_the_play`), and it does not if
//!   the reward, the seat symmetry or the RNG split is wrong.
//!
//! The DB draw is **deck-uniform, not archetype-uniform**: an archetype's deck count in the
//! archive is roughly how many meta refreshes it survived, so weighting by it is the closest
//! thing to meta-tiering available without a hand-written tier table. It can be restricted to a
//! subset of archetypes ([`DeckSource::archetypes`]), which is how a run draws one tutorial
//! difficulty tier rather than all of them at once.
//!
//! **The draw may run over several DBs at once**, each an explicit share of it ([`DeckSource`]).
//! That share is the one place deck-uniformity is overridden on purpose: concatenating `tutorial`
//! into `meta` would leave it 262 of 70 884 decks — 0.4 % of the draw, effectively absent — when
//! the reason to mix it in is that its beginner tier holds the weak decks `meta` has none of.
//! Inside a source the draw stays deck-uniform.
//!
//! **The source is rolled per seat, independently.** Rolling it once per game would give `meta`
//! self-play and `tutorial` self-play in proportion and never one across the table from the
//! other, which is the matchup the mix exists to buy: §1.1.3 wants a result attributable to the
//! deck, and that is not learnable from a distribution where both seats are always equally
//! strong. The two mirror quotas are the exception and stay inside one source, an archetype being
//! a per-DB label.
//!
//! §1.5.3's permanent uniform-deck quota is **not implemented here**: a legal-random-deck
//! generator is deckbuilder-side work (Part 6) and nothing in Part 5 needs it.

use rand::Rng;

use super::deck_db::DeckDb;
use crate::Deck;

/// A deck as drawn: the built deck plus the identity §1.5.7 keys its labels on.
#[derive(Debug, Clone)]
pub struct SampledDeck {
    pub archetype: String,
    pub id: String,
    pub deck: Deck,
}

/// The two quotas, as fractions of games, and the slice of the DB they run over.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    pub pure_mirror: f64,
    pub mirror: f64,
    /// Archetypes the draw is restricted to, or the whole DB when empty.
    ///
    /// `tutorial`'s archetype is the difficulty tier, so this is what keeps an early run off
    /// expert decks — drawing beginner and expert uniformly is the thing the grouping exists to
    /// prevent. A curriculum stage (§1.5.4) sets this per stage, one `SamplerConfig` per
    /// `DeckSampler` rebuilt on every transition.
    pub archetypes: Vec<String>,
}

impl SamplerConfig {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [("pure_mirror", self.pure_mirror), ("mirror", self.mirror)] {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} quota must be in [0, 1], got {value}"));
            }
        }
        if self.pure_mirror + self.mirror > 1.0 {
            return Err(format!(
                "quotas sum above 1: pure_mirror {} + mirror {}",
                self.pure_mirror, self.mirror
            ));
        }
        Ok(())
    }
}

/// One DB in the draw, its slice, and how much of the draw it takes.
#[derive(Debug, Clone)]
pub struct DeckSource {
    pub db: DeckDb,
    /// Relative weight of this source in a seat's roll, normalized against the other sources —
    /// the same "shares need not sum to 1" convention `[magnet.seed]`'s anchors already use.
    pub share: f64,
    /// Archetypes of *this* DB the draw is restricted to, or all of them when empty.
    pub archetypes: Vec<String>,
}

/// The by-name form of [`DeckSource`], as a stage's `.toml` carries it — the DB is a directory
/// under `[decks] root` and is only loaded at the transition into the stage that names it.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSpec {
    pub db: String,
    pub share: f64,
    pub archetypes: Vec<String>,
}

/// One source, resolved: the DB plus the index space a deck-uniform draw over its selected
/// archetypes runs on.
#[derive(Debug, Clone)]
struct Source {
    db: DeckDb,
    /// `(archetype, deck)` for every selected deck, so a deck-uniform draw is one index roll
    /// instead of a walk over cumulative archetype sizes.
    flat: Vec<(u32, u32)>,
    /// Normalized cumulative share, so picking a source is one roll and one scan.
    cumulative: f64,
}

impl Source {
    fn build(&self, (archetype, deck): (u32, u32)) -> Result<SampledDeck, String> {
        let archetype = &self.db.archetypes[archetype as usize];
        let entry = &archetype.decks[deck as usize];
        Ok(SampledDeck {
            archetype: archetype.name.clone(),
            id: entry.id.clone(),
            deck: entry.build()?,
        })
    }
}

/// Draws the deck pair for a game out of one or more DBs.
#[derive(Debug, Clone)]
pub struct DeckSampler {
    sources: Vec<Source>,
    pure_mirror: f64,
    mirror: f64,
}

impl DeckSampler {
    /// One DB taking the whole draw — the shape every non-mixed run and every test wants.
    pub fn new(db: DeckDb, config: SamplerConfig) -> Result<Self, String> {
        let SamplerConfig {
            pure_mirror,
            mirror,
            archetypes,
        } = config;
        DeckSampler::mixed(
            vec![DeckSource {
                db,
                share: 1.0,
                archetypes,
            }],
            pure_mirror,
            mirror,
        )
    }

    /// Several DBs, each an explicit share of the seat roll.
    pub fn mixed(sources: Vec<DeckSource>, pure_mirror: f64, mirror: f64) -> Result<Self, String> {
        SamplerConfig {
            pure_mirror,
            mirror,
            archetypes: Vec::new(),
        }
        .validate()?;

        if sources.is_empty() {
            return Err("a sampler needs at least one deck source".to_string());
        }
        for source in &sources {
            if !(source.share.is_finite() && source.share > 0.0) {
                return Err(format!(
                    "deck db {} has share {}, which must be finite and above zero",
                    source.db.name, source.share
                ));
            }
        }
        let total: f64 = sources.iter().map(|source| source.share).sum();

        let mut resolved = Vec::with_capacity(sources.len());
        let mut running = 0.0;
        for source in sources {
            // An unknown name is an error, not an empty selection: a typo in the run's `.toml`
            // would otherwise silently narrow the training distribution instead of failing at
            // startup.
            for wanted in &source.archetypes {
                if !source.db.archetypes.iter().any(|a| &a.name == wanted) {
                    return Err(format!(
                        "deck db {} has no archetype {wanted:?}",
                        source.db.name
                    ));
                }
            }

            let selected =
                |name: &String| source.archetypes.is_empty() || source.archetypes.contains(name);
            let mut flat = Vec::with_capacity(source.db.deck_count());
            for (index, archetype) in source.db.archetypes.iter().enumerate() {
                if !selected(&archetype.name) {
                    continue;
                }
                for deck in 0..archetype.decks.len() {
                    flat.push((index as u32, deck as u32));
                }
            }

            running += source.share / total;
            resolved.push(Source {
                db: source.db,
                flat,
                cumulative: running,
            });
        }
        Ok(DeckSampler {
            sources: resolved,
            pure_mirror,
            mirror,
        })
    }

    /// The DBs behind the draw, in config order.
    pub fn dbs(&self) -> impl Iterator<Item = &DeckDb> {
        self.sources.iter().map(|source| &source.db)
    }

    /// Decks the draw can actually reach, over every source and after the archetype restrictions.
    pub fn deck_count(&self) -> usize {
        self.sources.iter().map(|source| source.flat.len()).sum()
    }

    /// The deck pair for one game, in seat order.
    pub fn sample(&self, rng: &mut impl Rng) -> Result<[SampledDeck; 2], String> {
        let roll: f64 = rng.gen();
        if roll < self.pure_mirror {
            let source = self.pick(rng);
            let drawn = source.build(source.flat[rng.gen_range(0..source.flat.len())])?;
            return Ok([drawn.clone(), drawn]);
        }
        if roll < self.pure_mirror + self.mirror {
            // Weighted by deck count, like the ordinary draw: picking the archetype through a
            // deck-uniform roll keeps the mirror quota on the same distribution as the rest.
            let source = self.pick(rng);
            let (archetype, _) = source.flat[rng.gen_range(0..source.flat.len())];
            let size = source.db.archetypes[archetype as usize].decks.len();
            return Ok([
                source.build((archetype, rng.gen_range(0..size) as u32))?,
                source.build((archetype, rng.gen_range(0..size) as u32))?,
            ]);
        }
        Ok([self.draw(rng)?, self.draw(rng)?])
    }

    /// One seat's source. A single-source sampler consumes **no** randomness here: the pre-mix
    /// sampler had no such roll, and §1.5.5's reproducibility means a resumed run has to keep
    /// drawing the decks it was drawing.
    fn pick(&self, rng: &mut impl Rng) -> &Source {
        if self.sources.len() == 1 {
            return &self.sources[0];
        }
        let roll: f64 = rng.gen();
        // The last cumulative is 1.0 only up to the rounding of the normalization, so the scan
        // has to be able to fall through rather than index past the end.
        self.sources
            .iter()
            .find(|source| roll < source.cumulative)
            .unwrap_or(&self.sources[self.sources.len() - 1])
    }

    fn draw(&self, rng: &mut impl Rng) -> Result<SampledDeck, String> {
        let source = self.pick(rng);
        source.build(source.flat[rng.gen_range(0..source.flat.len())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::env::env_rng;
    use std::path::Path;

    fn sampler(config: SamplerConfig) -> DeckSampler {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        DeckSampler::new(db, config).expect("sampler")
    }

    fn quotas(pure_mirror: f64, mirror: f64) -> SamplerConfig {
        SamplerConfig {
            pure_mirror,
            mirror,
            archetypes: Vec::new(),
        }
    }

    #[test]
    fn pure_mirror_gives_both_seats_the_same_deck() {
        let sampler = sampler(quotas(1.0, 0.0));
        let mut rng = env_rng(7, 0);
        for _ in 0..50 {
            let [a, b] = sampler.sample(&mut rng).expect("draw");
            assert_eq!(a.id, b.id);
            assert_eq!(a.deck, b.deck);
        }
    }

    #[test]
    fn mirror_gives_both_seats_the_same_archetype() {
        let sampler = sampler(quotas(0.0, 1.0));
        let mut rng = env_rng(7, 1);
        for _ in 0..50 {
            let [a, b] = sampler.sample(&mut rng).expect("draw");
            assert_eq!(a.archetype, b.archetype);
        }
    }

    /// Reproducibility is a §1.5.5 guarantee and it has to survive the deck draw, not just the
    /// game: two runs on the same child seed must meet the same decks in the same order.
    #[test]
    fn the_same_seed_draws_the_same_decks() {
        let sampler = sampler(quotas(0.05, 0.1));
        let draw = |seed| {
            let mut rng = env_rng(seed, 3);
            (0..30)
                .map(|_| {
                    let [a, b] = sampler.sample(&mut rng).expect("draw");
                    (a.id, b.id)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(11), draw(11));
        assert_ne!(draw(11), draw(12));
    }

    /// The remainder of the quotas is the ordinary draw, and it has to actually reach across
    /// archetypes — a sampler that silently collapsed to one archetype would pass every test above.
    #[test]
    fn the_db_draw_spreads_over_archetypes_and_decks() {
        let sampler = sampler(quotas(0.0, 0.0));
        let mut rng = env_rng(7, 2);
        let mut archetypes = std::collections::HashSet::new();
        let mut decks = std::collections::HashSet::new();
        for _ in 0..200 {
            for drawn in sampler.sample(&mut rng).expect("draw") {
                archetypes.insert(drawn.archetype);
                decks.insert(drawn.id);
            }
        }
        assert_eq!(
            archetypes.len(),
            sampler.dbs().next().expect("one db").archetypes.len()
        );
        assert!(decks.len() > 100, "only {} distinct decks", decks.len());
    }

    /// The tutorial DB is keyed on difficulty tier precisely so a run can draw one; a restriction
    /// that leaked a single expert deck would defeat the point.
    #[test]
    fn restricting_to_an_archetype_draws_only_from_it() {
        let mut config = quotas(0.05, 0.1);
        config.archetypes = vec!["beginner".to_string()];
        let sampler = sampler(config);
        let mut rng = env_rng(7, 4);
        for _ in 0..200 {
            for drawn in sampler.sample(&mut rng).expect("draw") {
                assert_eq!(drawn.archetype, "beginner");
            }
        }
        assert!(
            sampler.dbs().next().expect("one db").archetypes.len() > 1,
            "the DB has other tiers"
        );
    }

    #[test]
    fn an_unknown_archetype_is_rejected() {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let mut config = quotas(0.0, 0.0);
        config.archetypes = vec!["begginer".to_string()];
        let err = DeckSampler::new(db, config).expect_err("typo");
        assert!(err.contains("no archetype"), "{err}");
    }

    #[test]
    fn quotas_summing_above_one_are_rejected() {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let err = DeckSampler::new(db, quotas(0.7, 0.7)).expect_err("invalid quotas");
        assert!(err.contains("sum above 1"), "{err}");
    }

    /// Two tiers of one DB stand in for two DBs: the mix only ever sees `(db, archetypes)` pairs,
    /// and this keeps the 70k-deck `meta` load out of a unit test.
    fn tiers(beginner_share: f64, expert_share: f64) -> Vec<DeckSource> {
        let load = || DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        vec![
            DeckSource {
                db: load(),
                share: beginner_share,
                archetypes: vec!["beginner".to_string()],
            },
            DeckSource {
                db: load(),
                share: expert_share,
                archetypes: vec!["expert".to_string()],
            },
        ]
    }

    /// The reason the mix exists: a weak deck across the table from a strong one. Rolling the
    /// source once per game instead of once per seat would make this count exactly zero.
    #[test]
    fn the_ordinary_draw_puts_the_two_sources_against_each_other() {
        let sampler = DeckSampler::mixed(tiers(0.5, 0.5), 0.0, 0.0).expect("mixed");
        let mut rng = env_rng(7, 5);
        let mut crossed = 0;
        for _ in 0..400 {
            let [a, b] = sampler.sample(&mut rng).expect("draw");
            if a.archetype != b.archetype {
                crossed += 1;
            }
        }
        // 2·p·q = 50 % at equal shares; the bound is loose enough to be a shape test, not a
        // distribution test.
        assert!(
            (150..=250).contains(&crossed),
            "{crossed}/400 cross-source games, expected ≈ 200"
        );
    }

    /// An archetype is a per-DB label, so a mirror that reached across sources would be pairing
    /// two decks that share nothing but a name.
    #[test]
    fn the_mirror_quotas_stay_inside_one_source() {
        for (pure_mirror, mirror) in [(1.0, 0.0), (0.0, 1.0)] {
            let sampler = DeckSampler::mixed(tiers(0.5, 0.5), pure_mirror, mirror).expect("mixed");
            let mut rng = env_rng(7, 6);
            for _ in 0..200 {
                let [a, b] = sampler.sample(&mut rng).expect("draw");
                assert_eq!(a.archetype, b.archetype);
            }
        }
    }

    /// The share is a real knob, not a label.
    #[test]
    fn the_shares_set_the_proportion_each_source_is_drawn_at() {
        let sampler = DeckSampler::mixed(tiers(0.8, 0.2), 0.0, 0.0).expect("mixed");
        let mut rng = env_rng(7, 7);
        let mut beginner = 0;
        let mut total = 0;
        for _ in 0..500 {
            for drawn in sampler.sample(&mut rng).expect("draw") {
                beginner += usize::from(drawn.archetype == "beginner");
                total += 1;
            }
        }
        let realized = beginner as f64 / total as f64;
        assert!(
            (0.75..=0.85).contains(&realized),
            "beginner share realized at {realized:.3}, asked 0.80"
        );
    }

    /// §1.5.5's reproducibility reaches the deck draw, and adding the mix must not move a
    /// single-source run's stream: the source roll has to be skipped, not rolled and discarded.
    #[test]
    fn a_single_source_draws_exactly_what_the_unmixed_sampler_draws() {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        let stream = |sampler: &DeckSampler| {
            let mut rng = env_rng(11, 8);
            (0..40)
                .map(|_| {
                    let [a, b] = sampler.sample(&mut rng).expect("draw");
                    (a.id, b.id)
                })
                .collect::<Vec<_>>()
        };
        let plain = DeckSampler::new(db.clone(), quotas(0.05, 0.1)).expect("plain");
        // A share other than 1.0 normalizes to the same thing on its own, and would still consume
        // a roll if `pick` were unconditional.
        let mixed = DeckSampler::mixed(
            vec![DeckSource {
                db,
                share: 0.37,
                archetypes: Vec::new(),
            }],
            0.05,
            0.1,
        )
        .expect("mixed");
        assert_eq!(stream(&plain), stream(&mixed));
    }

    #[test]
    fn a_source_without_a_usable_share_is_rejected() {
        for share in [0.0, -1.0, f64::NAN] {
            let mut sources = tiers(0.5, 0.5);
            sources[1].share = share;
            let err = DeckSampler::mixed(sources, 0.0, 0.0).expect_err("bad share");
            assert!(err.contains("share"), "{err}");
        }
    }

    /// The restriction is per source, so the error has to name the DB that does not have it.
    #[test]
    fn an_unknown_archetype_in_a_mixed_source_is_rejected() {
        let mut sources = tiers(0.5, 0.5);
        sources[1].archetypes = vec!["expurt".to_string()];
        let err = DeckSampler::mixed(sources, 0.0, 0.0).expect_err("typo");
        assert!(err.contains("no archetype"), "{err}");
    }

    #[test]
    fn a_sampler_without_a_source_is_rejected() {
        let err = DeckSampler::mixed(Vec::new(), 0.0, 0.0).expect_err("no source");
        assert!(err.contains("at least one deck source"), "{err}");
    }
}
