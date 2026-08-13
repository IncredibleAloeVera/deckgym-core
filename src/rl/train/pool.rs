//! The PFSP opponent pool — `RL_ARCHITECTURE.md` §1.5.2.
//!
//! Three kinds of opponent share one pool, one rating table ([`super::rating`]) and one sampling
//! distribution, differing only in tenure:
//!
//! - **Permanent** — the frozen heuristic panel and the curriculum's "baked" models. Never evicted;
//!   one of them is the rating scale's origin, and also carries the uniform floor.
//! - **Best slots** (`X`) — the frozen clones that are hardest for the current best-response.
//! - **History slots** (`Y`) — clones drawn back out of the archive, whatever their age.
//!
//! Selection reads the *conservative* rating ([`Rating::conservative`]), never the window's
//! winrate — see [`super::rating`] for why. Eviction frees a slot but never deletes weights: the
//! historical draw needs an archive to redraw from. A new member is protected from eviction and
//! guaranteed a share of the sampling mass until it clears `grace_games`, since it enters holding
//! its parent's — stale — rating.
//!
//! Full design rationale (why the clone/refresh cadences are decoupled, the archive's disk cost,
//! why grace is a game count with only a batch cap, the `X`/`Y` split): NOTES.md, "Système PFSP".

use std::collections::HashSet;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::rating::{win_probability, Entry, OpponentId, RatingTable};
use crate::players::PlayerCode;

/// Where the historical slots draw from. Exposed because there is no defensible universal answer:
/// the three differ in *which* timescale they guarantee coverage of, and which one matters depends
/// on whether the run is cycling.
///
/// Written in the `.toml` as a bare string for the two that take no argument (`"uniform"`,
/// `"log_age"`) and as a table for the one that does (`{ kind = "recent", max_age = 500 }`). Serde
/// cannot derive that shape — an internally tagged enum insists on the table even for a unit
/// variant — and forcing `{ kind = "uniform" }` would be ceremony on the common case.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum HistoryDraw {
    /// Uniform over the whole archive. Since an archive is mostly old, this leans old — which is
    /// the direction that counters the best slots' pull toward the newest clones.
    #[default]
    Uniform,
    /// Equal mass per age octave: roughly one very recent, one mid-run and one ancient. Covers
    /// every timescale at the cost of covering none of them densely.
    LogAge,
    /// Uniform over clones no older than `max_age` batches. The original §1.5.2 sketch, kept
    /// because a run that is *not* cycling spends nothing on ancient opponents — but note it pulls
    /// the same way the best slots already do, so it makes the pool as a whole recency-biased.
    Recent { max_age: u64 },
}

impl<'de> Deserialize<'de> for HistoryDraw {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Named(String),
            Tagged { kind: String, max_age: Option<u64> },
        }

        let (kind, max_age) = match Raw::deserialize(deserializer)? {
            Raw::Named(kind) => (kind, None),
            Raw::Tagged { kind, max_age } => (kind, max_age),
        };
        match kind.as_str() {
            "uniform" => Ok(HistoryDraw::Uniform),
            "log_age" => Ok(HistoryDraw::LogAge),
            "recent" => max_age
                .map(|max_age| HistoryDraw::Recent { max_age })
                .ok_or_else(|| {
                    serde::de::Error::custom(
                        "[pool] history_draw = \"recent\" needs a horizon: \
                     { kind = \"recent\", max_age = 500 }",
                    )
                }),
            other => Err(serde::de::Error::custom(format!(
                "[pool] unknown history_draw `{other}`: expected \"uniform\", \"log_age\", or \
                 {{ kind = \"recent\", max_age = <batches> }}"
            ))),
        }
    }
}

/// An opponent that is in the mix for the run's life: the heuristic panel and the curriculum's
/// baked models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Permanent {
    pub id: OpponentId,
    /// The scale's origin. At most one across the pool, enforced by [`Pool::new`] rather than left
    /// to the config — two fixed points over-determine a relative scale, and the second would be
    /// dragged off its stated value by nothing at all.
    #[serde(default)]
    pub pinned: bool,
}

impl Permanent {
    pub fn heuristic(code: PlayerCode) -> Self {
        Permanent {
            id: OpponentId::Heuristic(code),
            pinned: false,
        }
    }

    pub fn baked(name: impl Into<String>) -> Self {
        Permanent {
            id: OpponentId::Baked(name.into()),
            pinned: false,
        }
    }

    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }
}

/// §1.5.2's `.toml` half.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolConfig {
    /// `X` — slots held by the hardest clones.
    pub best_slots: usize,
    /// `Y` — slots held by clones drawn back out of the archive.
    pub history_slots: usize,
    /// `B` — batches per refresh, and the rating period (the decision and the measurement it reads
    /// have to close on the same boundary). Sizing against the noise floor and the archive's disk
    /// cost: NOTES.md, "Système PFSP".
    pub refresh_every: u64,
    /// `C` — batches per clone. Smaller than `refresh_every`, or the pool cannot fill.
    pub clone_every: u64,
    /// Games a new member is guaranteed, and below which it cannot be evicted.
    pub grace_games: u64,
    /// Upper bound on how long the guarantee may hold.
    pub grace_batches: u64,
    /// Share of the sampling mass reserved for members still in grace, split equally. Must be large
    /// enough that `grace_games` actually fits in `grace_batches` — [`PoolConfig::validate`] does
    /// not check that (it does not know the batch's game count), but the defaults are sized for it:
    /// 20 % of 10 batches × ~60 games ≈ 120 against a floor of 60.
    pub grace_share: f64,
    /// Share of the mass spread uniformly over every active member, whatever PFSP thinks of it.
    /// This is what keeps the pinned anchor playing, and with it the rating scale anchored.
    pub uniform_floor: f64,
    /// Exponent on PFSP's `p(1−p)`. Higher concentrates harder on the ~50 % matchups; `0` is
    /// uniform sampling.
    pub pfsp_sharpness: f64,
    pub history_draw: HistoryDraw,
}

/// Sized for the 200-batch run of `config/default.toml`, and deliberately history-heavy
/// (`X = 2` / `Y = 6`, not an even split) — see NOTES.md, "Système PFSP".
impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            best_slots: 2,
            history_slots: 6,
            refresh_every: 40,
            clone_every: 10,
            grace_games: 60,
            grace_batches: 10,
            grace_share: 0.2,
            uniform_floor: 0.15,
            pfsp_sharpness: 1.0,
            history_draw: HistoryDraw::default(),
        }
    }
}

impl PoolConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.refresh_every == 0 {
            return Err("[pool] refresh_every must be > 0".to_string());
        }
        if self.clone_every == 0 {
            return Err("[pool] clone_every must be > 0".to_string());
        }
        if self.clone_every > self.refresh_every {
            return Err(format!(
                "[pool] clone_every ({}) must be ≤ refresh_every ({}): a pool that clones less \
                 often than it refreshes never fills its slots",
                self.clone_every, self.refresh_every
            ));
        }
        if self.grace_batches > self.refresh_every {
            return Err(format!(
                "[pool] grace_batches ({}) must be ≤ refresh_every ({}): grace exists to protect a \
                 member until the refresh that judges it",
                self.grace_batches, self.refresh_every
            ));
        }
        for (name, share) in [
            ("grace_share", self.grace_share),
            ("uniform_floor", self.uniform_floor),
        ] {
            if !(0.0..=1.0).contains(&share) {
                return Err(format!("[pool] {name} must be in [0, 1], got {share}"));
            }
        }
        if self.grace_share + self.uniform_floor > 1.0 {
            return Err(format!(
                "[pool] grace_share ({}) + uniform_floor ({}) must be ≤ 1",
                self.grace_share, self.uniform_floor
            ));
        }
        if self.pfsp_sharpness < 0.0 {
            return Err(format!(
                "[pool] pfsp_sharpness must be ≥ 0, got {}",
                self.pfsp_sharpness
            ));
        }
        Ok(())
    }
}

/// Which kind of slot a member occupies. Logged, because "this opponent is here because it is hard"
/// and "this opponent is here because it is old" are different readings of the same winrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Permanent,
    Best,
    History,
}

/// A clone occupying a slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Slot {
    batch: u64,
    role: Role,
    /// Batch at which it took this slot. Grace runs from here, not from the clone's own batch — a
    /// member drawn back out of the archive is new to the *learner* again.
    admitted: u64,
    /// The member's lifetime game count when it was admitted, so grace measures games in this
    /// stint rather than games ever.
    games_at_admission: u64,
}

impl Slot {
    fn id(&self) -> OpponentId {
        OpponentId::Pool(self.batch)
    }

    /// Sort key, so the slot vector's order is a property of the pool rather than of the order
    /// refreshes happened to push things in.
    fn role_order(&self) -> u8 {
        match self.role {
            Role::Permanent => 0,
            Role::Best => 1,
            Role::History => 2,
        }
    }
}

/// What a refresh did, for the log and for the caller that has to keep files around.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Refresh {
    pub admitted: Vec<(OpponentId, Role)>,
    /// Slots freed. The weights stay on disk — see the module docs.
    pub released: Vec<OpponentId>,
}

/// One row of the pool's own JSONL table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PoolRow {
    pub id: String,
    pub role: Role,
    pub rating: f64,
    pub deviation: f64,
    pub volatility: f64,
    pub games: u64,
    /// Sampling probability as of the row's batch, so a surprising winrate can be read against how
    /// often the matchup actually came up.
    pub weight: f64,
    /// `null` for a permanent member, which has no age.
    pub age_batches: Option<u64>,
    pub in_grace: bool,
}

/// Shared by [`Pool::new`] and [`Pool::retarget`]: non-empty, no duplicate id, no clone id
/// masquerading as permanent, at most one pin.
fn validate_permanent(permanent: &[Permanent]) -> Result<(), String> {
    if permanent.is_empty() {
        return Err(
            "[pool] needs at least one permanent opponent: an empty pool has nothing to play \
             against before the first clone, and no origin for the rating scale"
                .to_string(),
        );
    }
    let mut seen = HashSet::new();
    for member in permanent {
        if !seen.insert(member.id.clone()) {
            return Err(format!(
                "[pool] duplicate permanent opponent `{}`",
                member.id
            ));
        }
        if matches!(member.id, OpponentId::Pool(_)) {
            return Err(format!(
                "[pool] `{}` is a clone id and cannot be a permanent member",
                member.id
            ));
        }
    }
    let pinned: Vec<&Permanent> = permanent.iter().filter(|member| member.pinned).collect();
    if pinned.len() > 1 {
        return Err(format!(
            "[pool] exactly one opponent may be pinned as the rating origin, got {}: {}",
            pinned.len(),
            pinned
                .iter()
                .map(|member| member.id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(())
}

/// The pool itself.
///
/// Serialized whole into §1.5.5's hot checkpoint, beside the rating table: a resume that re-rolled
/// its slots would face a different panel than the run it continues, and §1.5.5's promise is that
/// two resumes from one checkpoint agree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pool {
    #[serde(skip)]
    config: PoolConfig,
    permanent: Vec<Permanent>,
    slots: Vec<Slot>,
    /// Every clone ever written, by batch, ascending. Never pruned.
    archive: Vec<u64>,
}

impl Pool {
    pub fn new(config: PoolConfig, permanent: Vec<Permanent>) -> Result<Self, String> {
        config.validate()?;
        validate_permanent(&permanent)?;
        Ok(Pool {
            config,
            permanent,
            slots: Vec::new(),
            archive: Vec::new(),
        })
    }

    /// Swaps the permanent membership — a curriculum stage transition (§1.5.4) changing which
    /// anchors/baked models are in the mix. `slots` and `archive` are untouched: [`Pool::active`]
    /// already derives from both fields independently, so a member dropped from `permanent` just
    /// stops appearing in `active()`/sampling, exactly like an evicted clone's rating already sits
    /// unused rather than deleted — nothing here needs to special-case the transition.
    ///
    /// **Does not touch the pin.** [`RatingTable::pin`] overwrites the previous pin rather than
    /// refusing a different id, so a stage that dropped the pinned anchor would silently move the
    /// rating scale's origin — and with it the elo curve §1.5.6 reads as the generalization signal.
    /// The guard against that lives at config-load time (every stage's anchors must include
    /// `[pool] pinned`), not here: `retarget` only re-affirms whatever the incoming list says,
    /// same as [`Pool::register`] already does after every ordinary refresh.
    pub fn retarget(
        &mut self,
        permanent: Vec<Permanent>,
        ratings: &mut RatingTable,
    ) -> Result<(), String> {
        validate_permanent(&permanent)?;
        self.permanent = permanent;
        self.register(ratings);
        Ok(())
    }

    /// Re-attaches the `.toml`'s parameters after a checkpoint load, for the reason
    /// [`RatingTable::with_config`] gives.
    pub fn with_config(mut self, config: PoolConfig) -> Result<Self, String> {
        config.validate()?;
        self.config = config;
        Ok(self)
    }

    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Declares every current member to the rating table, and applies the pin. Idempotent, and
    /// called after a refresh and after a resume — a member that is already rated keeps its rating,
    /// which is what makes a re-drawn checkpoint resume rather than restart.
    pub fn register(&self, ratings: &mut RatingTable) {
        for member in &self.permanent {
            ratings.ensure(member.id.clone(), Entry::fresh());
            if member.pinned {
                ratings.pin(
                    member.id.clone(),
                    super::rating::DEFAULT_RATING,
                    ratings.config().min_deviation,
                );
            }
        }
        for slot in &self.slots {
            ratings.ensure(slot.id(), Entry::fresh());
        }
    }

    /// Everything currently in the mix, permanents first.
    pub fn active(&self) -> Vec<OpponentId> {
        self.permanent
            .iter()
            .map(|member| member.id.clone())
            .chain(self.slots.iter().map(Slot::id))
            .collect()
    }

    pub fn archive(&self) -> &[u64] {
        &self.archive
    }

    /// The membership a resume has to re-adopt: unlike [`Pool::active`] this keeps the `Permanent`
    /// wrapper, which is what says whether an id is scripted, baked, or the pin.
    pub fn permanent(&self) -> &[Permanent] {
        &self.permanent
    }

    /// Every clone currently holding a slot, ascending. The membership a `partial` carry
    /// ([`super::init`]) has to copy, since those are the only weights the next batch can ask for.
    pub fn slot_batches(&self) -> Vec<u64> {
        let mut batches: Vec<u64> = self.slots.iter().map(|slot| slot.batch).collect();
        batches.sort_unstable();
        batches
    }

    /// Re-admits every occupied slot at `batch`, as if it had just been drawn.
    ///
    /// For carrying a pool into a run whose batch counter starts over ([`super::init`]). Tenure is
    /// stored as the batch a member was admitted at and its game count at that moment, and both are
    /// read as *differences* against the current batch and game count — so a slot admitted at the
    /// source's batch 2150 lands in a run at batch 50 with `50 − 2150` saturating to zero. It then
    /// reads as freshly admitted at every batch until the new run's counter passes 2150, which is a
    /// grace period that cannot expire: the member holds its slot against every ranking, and keeps
    /// the sampling share the grace reservation gives it. Re-admitting at the new run's own batch is
    /// what makes the tenure mean the same thing on both sides of the copy.
    pub fn readmit_slots(&mut self, ratings: &RatingTable, batch: u64) {
        for slot in &mut self.slots {
            slot.admitted = batch;
            slot.games_at_admission = ratings
                .get(&OpponentId::Pool(slot.batch))
                .map(|entry| entry.games)
                .unwrap_or(0);
        }
    }

    /// Narrows the archive to the clones currently in slots, and returns them.
    ///
    /// For carrying a pool into a *different* run directory ([`super::init`]): the archive is a
    /// list of batch numbers that a history draw will happily hand back, and a run whose
    /// `pool/` holds fewer files than that list promises would fail at the refresh that draws one
    /// — hours in, and for a file that was never copied rather than for anything that happened.
    /// The slots themselves are untouched, so the panel the next batch faces is the one the source
    /// run stopped on.
    pub fn restrict_to_slots(&mut self) -> Vec<u64> {
        let kept = self.slot_batches();
        self.archive.retain(|batch| kept.contains(batch));
        kept
    }

    fn slots_full(&self) -> bool {
        self.slots.len() >= self.config.best_slots + self.config.history_slots
    }

    fn in_grace(&self, slot: &Slot, ratings: &RatingTable, batch: u64) -> bool {
        let games = ratings
            .get(&slot.id())
            .map(|entry| entry.games)
            .unwrap_or(0)
            .saturating_sub(slot.games_at_admission);
        games < self.config.grace_games
            && batch.saturating_sub(slot.admitted) < self.config.grace_batches
    }

    /// The sampling distribution over [`Pool::active`], in the same order.
    ///
    /// Three layers, in this order: PFSP's `(p(1−p))^k` prioritizing the ~50 % matchups, a uniform
    /// floor so nothing (the pinned anchor least of all) can be starved out, and the grace
    /// reservation on top. `p` is the *predicted* score from the ratings rather than a measured
    /// winrate, which is what lets a member that has never played have a well-defined weight from
    /// its first game instead of after its first hundred.
    pub fn weights(&self, ratings: &RatingTable, batch: u64) -> Vec<(OpponentId, f64)> {
        let active = self.active();
        if active.is_empty() {
            return Vec::new();
        }
        let learner = ratings.learner().rating;
        let raw: Vec<f64> = active
            .iter()
            .map(|id| {
                let p = win_probability(&learner, &ratings.rating_of(id));
                (p * (1.0 - p)).powf(self.config.pfsp_sharpness)
            })
            .collect();

        let total: f64 = raw.iter().sum();
        let n = active.len() as f64;
        // A pool the learner beats (or loses to) unanimously flattens `p(1−p)` to zero and PFSP has
        // nothing to say. Uniform is the honest fallback, and the situation is transient by
        // construction: those results are exactly what moves the ratings apart.
        let base: Vec<f64> = if total > 0.0 && total.is_finite() {
            raw.iter().map(|w| w / total).collect()
        } else {
            vec![1.0 / n; active.len()]
        };

        let floor = self.config.uniform_floor;
        let mut weights: Vec<f64> = base.iter().map(|w| (1.0 - floor) * w + floor / n).collect();

        let graced: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| self.in_grace(slot, ratings, batch))
            .map(|(index, _)| self.permanent.len() + index)
            .collect();
        if !graced.is_empty() {
            let share = self.config.grace_share;
            let each = share / graced.len() as f64;
            for weight in weights.iter_mut() {
                *weight *= 1.0 - share;
            }
            for index in graced {
                weights[index] += each;
            }
        }

        active.into_iter().zip(weights).collect()
    }

    /// Draws one opponent for one game.
    ///
    /// Takes the caller's generator rather than owning one: §1.5.5 keys every draw to a stream so a
    /// resume can reconstruct it, and a pool with a private generator would be a fourth consumer
    /// nobody could replay.
    pub fn sample<R: Rng + ?Sized>(
        &self,
        ratings: &RatingTable,
        batch: u64,
        rng: &mut R,
    ) -> OpponentId {
        let weights = self.weights(ratings, batch);
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        let mut point = rng.gen_range(0.0..total.max(f64::MIN_POSITIVE));
        for (id, weight) in &weights {
            point -= weight;
            if point <= 0.0 {
                return id.clone();
            }
        }
        // Floating-point residue only; the loop above consumes the mass in every real case.
        weights
            .last()
            .map(|(id, _)| id.clone())
            .expect("a pool always has at least one permanent member")
    }

    pub fn should_clone(&self, batch: u64) -> bool {
        batch > 0 && batch.is_multiple_of(self.config.clone_every)
    }

    pub fn should_refresh(&self, batch: u64) -> bool {
        batch > 0 && batch.is_multiple_of(self.config.refresh_every)
    }

    /// Records a clone taken at `batch` and gives it its parent's rating (§1.5.2).
    ///
    /// The caller writes the weights; the pool only decides identity and tenure. That split is what
    /// keeps this module free of a backend, and with it of the `rl-model` feature gate.
    ///
    /// While slots are still free the clone takes one immediately — that is the fill phase, and it
    /// is why the pool is populated in `(X + Y) × clone_every` batches rather than
    /// `(X + Y) × refresh_every`.
    pub fn admit_clone(&mut self, batch: u64, ratings: &mut RatingTable) -> OpponentId {
        let id = OpponentId::Pool(batch);
        if !self.archive.contains(&batch) {
            self.archive.push(batch);
            self.archive.sort_unstable();
        }
        ratings.ensure(id.clone(), Entry::cloned_from(&ratings.learner().rating));

        if !self.slots_full() && !self.slots.iter().any(|slot| slot.batch == batch) {
            let role = if self
                .slots
                .iter()
                .filter(|slot| slot.role == Role::Best)
                .count()
                < self.config.best_slots
            {
                Role::Best
            } else {
                Role::History
            };
            let games_at_admission = ratings.get(&id).map(|entry| entry.games).unwrap_or(0);
            self.slots.push(Slot {
                batch,
                role,
                admitted: batch,
                games_at_admission,
            });
        }
        id
    }

    /// Re-decides the slots. Call after [`RatingTable::close_period`], so it reads a closed period
    /// rather than a half-filled one.
    ///
    /// The best slots are ranked on the conservative rating; anything still in grace holds its slot
    /// regardless of rank. The history slots are re-drawn from the archive every time, which is
    /// what stops them from silently turning into a second set of best slots.
    pub fn refresh<R: Rng + ?Sized>(
        &mut self,
        ratings: &mut RatingTable,
        batch: u64,
        rng: &mut R,
    ) -> Refresh {
        let before: HashSet<OpponentId> = self.slots.iter().map(Slot::id).collect();

        // Taken out first: the partition below reads `self` immutably, which it cannot do while
        // draining a field of it.
        let slots = std::mem::take(&mut self.slots);
        let (protected, contested): (Vec<Slot>, Vec<Slot>) = slots
            .into_iter()
            .partition(|slot| self.in_grace(slot, ratings, batch));

        // A protected member keeps the slot it holds rather than being promoted into a best slot:
        // grace says "not yet judged", which is not the same claim as "among the hardest".
        let (mut best, kept_history): (Vec<Slot>, Vec<Slot>) = protected
            .into_iter()
            .partition(|slot| slot.role == Role::Best);

        let mut ranked = contested;
        ranked.sort_by(|a, b| {
            let key = |slot: &Slot| ratings.rating_of(&slot.id()).conservative();
            key(b)
                .partial_cmp(&key(a))
                .unwrap_or(std::cmp::Ordering::Equal)
                // Ties broken by age so the order does not depend on how the vector happened to be
                // laid out: a refresh has to be a function of the ratings alone.
                .then_with(|| b.batch.cmp(&a.batch))
        });
        for mut slot in ranked {
            if best.len() >= self.config.best_slots {
                break;
            }
            slot.role = Role::Best;
            best.push(slot);
        }

        let mut held: HashSet<u64> = best.iter().map(|slot| slot.batch).collect();
        held.extend(kept_history.iter().map(|slot| slot.batch));
        let free_history = self.config.history_slots.saturating_sub(kept_history.len());
        let drawn = self.draw_history(&held, batch, free_history, rng);

        self.slots = best;
        self.slots.extend(kept_history);
        for batch_id in drawn {
            let games_at_admission = ratings
                .get(&OpponentId::Pool(batch_id))
                .map(|entry| entry.games)
                .unwrap_or(0);
            self.slots.push(Slot {
                batch: batch_id,
                role: Role::History,
                admitted: batch,
                games_at_admission,
            });
        }
        self.slots
            .sort_by_key(|slot| (slot.role_order(), slot.batch));
        self.register(ratings);

        let after: HashSet<OpponentId> = self.slots.iter().map(Slot::id).collect();
        Refresh {
            admitted: self
                .slots
                .iter()
                .filter(|slot| !before.contains(&slot.id()))
                .map(|slot| (slot.id(), slot.role))
                .collect(),
            released: before.difference(&after).cloned().collect(),
        }
    }

    /// Fills `wanted` history slots from the archive, without replacement and skipping anything a
    /// slot already holds.
    fn draw_history<R: Rng + ?Sized>(
        &self,
        held: &HashSet<u64>,
        batch: u64,
        wanted: usize,
        rng: &mut R,
    ) -> Vec<u64> {
        let mut candidates: Vec<u64> = self
            .archive
            .iter()
            .copied()
            .filter(|entry| !held.contains(entry))
            .filter(|entry| match self.config.history_draw {
                HistoryDraw::Recent { max_age } => batch.saturating_sub(*entry) <= max_age,
                _ => true,
            })
            .collect();

        let mut drawn = Vec::new();
        while drawn.len() < wanted && !candidates.is_empty() {
            let weights: Vec<f64> = candidates
                .iter()
                .map(|entry| match self.config.history_draw {
                    HistoryDraw::Uniform | HistoryDraw::Recent { .. } => 1.0,
                    // Equal mass per age octave: `1/age` in age-space integrates to a constant per
                    // doubling, so a run's whole history is covered at every scale rather than
                    // being dominated by whichever stretch happens to be longest.
                    HistoryDraw::LogAge => 1.0 / (1.0 + batch.saturating_sub(*entry) as f64),
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let mut point = rng.gen_range(0.0..total.max(f64::MIN_POSITIVE));
            let mut chosen = candidates.len() - 1;
            for (index, weight) in weights.iter().enumerate() {
                point -= weight;
                if point <= 0.0 {
                    chosen = index;
                    break;
                }
            }
            drawn.push(candidates.remove(chosen));
        }
        drawn.sort_unstable();
        drawn
    }

    /// The pool's own table, one row per active member — §1.5.6's per-member view, kept out of
    /// `metrics.jsonl` because a scalar series per clone leaves a dead curve behind for every
    /// eviction.
    pub fn rows(&self, ratings: &RatingTable, batch: u64) -> Vec<PoolRow> {
        let weights = self.weights(ratings, batch);
        weights
            .into_iter()
            .map(|(id, weight)| {
                let entry = ratings.get(&id).cloned().unwrap_or_else(Entry::fresh);
                let slot = self.slots.iter().find(|slot| slot.id() == id);
                PoolRow {
                    id: id.to_string(),
                    role: slot.map(|slot| slot.role).unwrap_or(Role::Permanent),
                    rating: entry.rating.rating,
                    deviation: entry.rating.deviation,
                    volatility: entry.rating.volatility,
                    games: entry.games,
                    weight,
                    age_batches: slot.map(|slot| batch.saturating_sub(slot.batch)),
                    in_grace: slot
                        .map(|slot| self.in_grace(slot, ratings, batch))
                        .unwrap_or(false),
                }
            })
            .collect()
    }

    /// The scalars §1.5.6 puts on the per-batch line. Per category, never per member.
    pub fn scalars(&self) -> Vec<(String, f64)> {
        vec![
            ("pool/active".to_string(), self.active().len() as f64),
            ("pool/archive".to_string(), self.archive.len() as f64),
            (
                "pool/best_slots".to_string(),
                self.slots
                    .iter()
                    .filter(|slot| slot.role == Role::Best)
                    .count() as f64,
            ),
            (
                "pool/history_slots".to_string(),
                self.slots
                    .iter()
                    .filter(|slot| slot.role == Role::History)
                    .count() as f64,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::train::rating::{Rating, RatingConfig};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(0xC0FFEE)
    }

    fn ratings() -> RatingTable {
        RatingTable::new(RatingConfig::default()).expect("config")
    }

    fn panel() -> Vec<Permanent> {
        vec![
            Permanent::heuristic(PlayerCode::ER).pinned(),
            Permanent::heuristic(PlayerCode::R),
            Permanent::baked("Cliff"),
        ]
    }

    fn pool(config: PoolConfig) -> Pool {
        Pool::new(config, panel()).expect("pool")
    }

    /// Drives a pool forward, playing every game as a learner win so the ratings actually separate.
    fn run(pool: &mut Pool, ratings: &mut RatingTable, batches: u64, games_per_batch: usize) {
        let mut rng = rng();
        for batch in 1..=batches {
            for _ in 0..games_per_batch {
                let opponent = pool.sample(ratings, batch, &mut rng);
                ratings.record(opponent, 1.0);
            }
            if pool.should_clone(batch) {
                pool.admit_clone(batch, ratings);
            }
            if pool.should_refresh(batch) {
                ratings.close_period();
                pool.refresh(ratings, batch, &mut rng);
            }
        }
    }

    /// Carried tenure has to be re-based, or a slot admitted late in the source run is protected
    /// for as long as it takes the new run's counter to reach the source's — thousands of batches
    /// during which no ranking can touch it.
    #[test]
    fn a_carried_slot_re_admitted_at_the_new_runs_batch_can_be_evicted_again() {
        let config = PoolConfig {
            best_slots: 1,
            history_slots: 1,
            ..PoolConfig::default()
        };
        let mut ratings = ratings();
        let mut pool = pool(PoolConfig {
            best_slots: 2,
            history_slots: 2,
            ..config.clone()
        });
        pool.register(&mut ratings);
        // Admitted late in a long source run, and never played since — the case where tenure by
        // games cannot expire the grace either.
        for batch in [1000u64, 1100, 1200, 1300] {
            pool.admit_clone(batch, &mut ratings);
        }
        assert_eq!(pool.slot_batches().len(), 4);

        let mut carried = pool.clone().with_config(config).expect("narrowed");
        carried.readmit_slots(&ratings, 0);
        ratings.close_period();
        carried.refresh(&mut ratings, 50, &mut rng());

        // Two slots for a config that asks for two. Without the re-admission every carried slot
        // reads as in grace at batch 50 — `50 − 1000` saturates — and all four hold.
        assert_eq!(carried.slot_batches().len(), 2);
    }

    #[test]
    fn the_fill_phase_runs_on_the_clone_cadence_not_the_refresh_cadence() {
        let config = PoolConfig::default();
        let (slots, clone_every) = (config.best_slots + config.history_slots, config.clone_every);
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);

        run(&mut pool, &mut ratings, slots as u64 * clone_every, 60);

        // Full after (X + Y) × clone_every batches — 200 at the defaults, against the 800 that
        // cloning on the refresh cadence would have cost.
        assert_eq!(pool.active().len(), panel().len() + slots);
    }

    #[test]
    fn a_clone_inherits_the_learners_rating() {
        let mut pool = pool(PoolConfig::default());
        let mut ratings = ratings();
        pool.register(&mut ratings);
        for _ in 0..300 {
            ratings.record(OpponentId::Heuristic(PlayerCode::R), 1.0);
        }
        ratings.close_period();

        let parent = ratings.learner().rating;
        let id = pool.admit_clone(25, &mut ratings);
        assert_eq!(ratings.get(&id).expect("rated").rating, parent);
    }

    #[test]
    fn eviction_frees_a_slot_and_keeps_the_weights() {
        let config = PoolConfig {
            best_slots: 1,
            history_slots: 1,
            grace_games: 0,
            grace_batches: 0,
            ..Default::default()
        };
        let clone_every = config.clone_every;
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 400, 60);

        assert_eq!(pool.active().len(), panel().len() + 2);
        // Every clone ever taken is still on the archive's books, including the evicted ones —
        // otherwise the historical draw has nothing to draw from.
        assert_eq!(pool.archive().len(), (400 / clone_every) as usize);
        assert!(pool.archive().len() > pool.active().len());
    }

    /// The reason the pool keeps evicted ratings: a checkpoint drawn back in resumes from where it
    /// left off rather than from 1500.
    #[test]
    fn a_redrawn_checkpoint_keeps_the_rating_it_had() {
        let config = PoolConfig {
            best_slots: 1,
            history_slots: 1,
            grace_games: 0,
            grace_batches: 0,
            ..Default::default()
        };
        let first_clone = config.clone_every;
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 200, 60);

        let evicted = OpponentId::Pool(first_clone);
        let rating = ratings.get(&evicted).expect("rated").rating;
        assert!(rating.rating < crate::rl::train::rating::CENTRE);
        assert!(ratings.get(&evicted).expect("rated").games > 0);

        // Re-registering the whole pool must not reset it.
        pool.register(&mut ratings);
        assert_eq!(ratings.get(&evicted).expect("rated").rating, rating);
    }

    #[test]
    fn a_member_in_grace_is_not_evicted_however_badly_it_is_rated() {
        let config = PoolConfig {
            best_slots: 1,
            history_slots: 0,
            refresh_every: 10,
            clone_every: 5,
            grace_games: 10_000,
            grace_batches: 10,
            ..Default::default()
        };
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);

        let mut rng = rng();
        pool.admit_clone(5, &mut ratings);
        // Crush it: on rating alone it would lose its slot at the next refresh.
        for _ in 0..400 {
            ratings.record(OpponentId::Pool(5), 1.0);
        }
        ratings.close_period();
        pool.refresh(&mut ratings, 10, &mut rng);

        assert!(pool.active().contains(&OpponentId::Pool(5)));
    }

    #[test]
    fn grace_reserves_sampling_mass_and_releases_it_afterwards() {
        let config = PoolConfig {
            best_slots: 2,
            history_slots: 0,
            refresh_every: 50,
            clone_every: 5,
            grace_games: 200,
            grace_batches: 20,
            grace_share: 0.2,
            ..Default::default()
        };
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);
        pool.admit_clone(5, &mut ratings);

        let weight_of = |pool: &Pool, ratings: &RatingTable, batch: u64| {
            pool.weights(ratings, batch)
                .into_iter()
                .find(|(id, _)| *id == OpponentId::Pool(5))
                .map(|(_, w)| w)
                .expect("in the pool")
        };
        let in_grace = weight_of(&pool, &ratings, 6);
        // Past `grace_batches` the reservation is gone, whatever the game count.
        let after = weight_of(&pool, &ratings, 5 + 20);
        assert!(
            in_grace > after,
            "grace weight {in_grace} should exceed the settled weight {after}"
        );

        let total: f64 = pool.weights(&ratings, 6).iter().map(|(_, w)| w).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "weights must sum to 1, got {total}"
        );
    }

    #[test]
    fn pfsp_prefers_the_even_matchup_and_the_floor_keeps_the_rest_alive() {
        let mut pool = pool(PoolConfig {
            best_slots: 3,
            history_slots: 0,
            uniform_floor: 0.15,
            grace_games: 0,
            grace_batches: 0,
            ..Default::default()
        });
        let mut ratings = ratings();
        pool.register(&mut ratings);
        for batch in [10u64, 20, 30] {
            pool.admit_clone(batch, &mut ratings);
        }

        // Hand-place the three: hopeless, even, unbeatable.
        for (batch, rating) in [(10u64, 900.0), (20, 1500.0), (30, 2100.0)] {
            let id = OpponentId::Pool(batch);
            let entry = Entry {
                rating: Rating {
                    rating,
                    deviation: 30.0,
                    volatility: 0.06,
                },
                pinned: false,
                drifts: false,
                games: 1_000,
            };
            // `set` and not `ensure`: the admission already registered a placeholder, and `ensure`
            // is idempotent by design.
            ratings.set(id, entry);
        }

        let weights = pool.weights(&ratings, 100);
        let weight = |batch: u64| {
            weights
                .iter()
                .find(|(id, _)| *id == OpponentId::Pool(batch))
                .map(|(_, w)| *w)
                .expect("present")
        };
        assert!(weight(20) > weight(10));
        assert!(weight(20) > weight(30));
        // Nothing is starved: the floor is what keeps the pinned anchor playing, and with it the
        // rating scale anchored.
        assert!(weight(10) > 0.0 && weight(30) > 0.0);
    }

    #[test]
    fn the_pinned_anchor_keeps_getting_games() {
        let mut pool = pool(PoolConfig::default());
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 300, 60);

        let anchor = OpponentId::Heuristic(PlayerCode::ER);
        assert_eq!(ratings.pinned(), Some(&anchor));
        assert!(
            ratings.get(&anchor).expect("rated").games > 0,
            "an origin that stops playing stops anchoring anything"
        );
        assert_eq!(ratings.get(&anchor).expect("rated").rating.rating, 1500.0);
    }

    #[test]
    fn the_history_draw_reaches_further_back_than_the_best_slots() {
        let config = PoolConfig {
            best_slots: 2,
            history_slots: 2,
            refresh_every: 50,
            clone_every: 10,
            grace_games: 0,
            grace_batches: 0,
            history_draw: HistoryDraw::Uniform,
            ..Default::default()
        };
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 1_000, 40);

        let oldest_held = pool
            .active()
            .iter()
            .filter_map(|id| match id {
                OpponentId::Pool(batch) => Some(*batch),
                _ => None,
            })
            .min()
            .expect("clones in the pool");
        // With 100 clones on the books and a uniform draw, holding nothing older than the last few
        // hundred batches would mean the historical slots are not doing their job.
        assert!(
            oldest_held < 800,
            "oldest held clone was from batch {oldest_held}"
        );
    }

    #[test]
    fn the_recent_draw_refuses_anything_past_its_horizon() {
        let config = PoolConfig {
            best_slots: 0,
            history_slots: 2,
            refresh_every: 50,
            clone_every: 10,
            grace_games: 0,
            grace_batches: 0,
            history_draw: HistoryDraw::Recent { max_age: 100 },
            ..Default::default()
        };
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 500, 40);

        for id in pool.active() {
            if let OpponentId::Pool(batch) = id {
                assert!(
                    500 - batch <= 100,
                    "clone from batch {batch} is past the horizon"
                );
            }
        }
    }

    #[test]
    fn a_refresh_is_a_function_of_the_ratings_not_of_the_vector_layout() {
        let config = PoolConfig {
            best_slots: 2,
            history_slots: 0,
            refresh_every: 20,
            clone_every: 5,
            grace_games: 0,
            grace_batches: 0,
            ..Default::default()
        };
        let mut first = pool(config.clone());
        let mut second = pool(config);
        let (mut ratings_a, mut ratings_b) = (ratings(), ratings());
        first.register(&mut ratings_a);
        second.register(&mut ratings_b);

        run(&mut first, &mut ratings_a, 200, 40);
        run(&mut second, &mut ratings_b, 200, 40);

        assert_eq!(first.active(), second.active());
    }

    #[test]
    fn the_pool_round_trips_through_json() {
        let mut pool = pool(PoolConfig::default());
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 260, 60);

        let json = serde_json::to_string(&pool).expect("serialize");
        let restored: Pool = serde_json::from_str(&json).expect("deserialize");
        let restored = restored.with_config(PoolConfig::default()).expect("config");

        assert_eq!(restored.active(), pool.active());
        assert_eq!(restored.archive(), pool.archive());
        assert_eq!(restored.weights(&ratings, 260), pool.weights(&ratings, 260));
    }

    #[test]
    fn a_config_that_cannot_fill_its_slots_is_refused() {
        let config = PoolConfig {
            clone_every: 200,
            refresh_every: 100,
            ..Default::default()
        };
        assert!(Pool::new(config, panel()).is_err());
    }

    #[test]
    fn two_pinned_anchors_are_refused() {
        let permanent = vec![
            Permanent::heuristic(PlayerCode::ER).pinned(),
            Permanent::heuristic(PlayerCode::R).pinned(),
        ];
        assert!(Pool::new(PoolConfig::default(), permanent).is_err());
    }

    #[test]
    fn an_empty_panel_is_refused() {
        assert!(Pool::new(PoolConfig::default(), Vec::new()).is_err());
    }

    /// A curriculum stage transition (§1.5.4): the permanent roster changes, and a member that
    /// stays in the new list — the pinned anchor least of all — keeps exactly the rating it had.
    #[test]
    fn retarget_replaces_membership_and_keeps_the_pinned_anchors_rating() {
        let mut pool = pool(PoolConfig::default());
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 60, 40);

        let anchor = OpponentId::Heuristic(PlayerCode::ER);
        let anchor_rating = ratings.get(&anchor).expect("rated").rating;

        let next = vec![
            Permanent::heuristic(PlayerCode::ER).pinned(),
            Permanent::heuristic(PlayerCode::W),
        ];
        pool.retarget(next.clone(), &mut ratings).expect("retarget");

        assert!(!pool
            .active()
            .contains(&OpponentId::Baked("Cliff".to_string())));
        assert!(pool
            .active()
            .contains(&OpponentId::Heuristic(PlayerCode::W)));
        assert_eq!(
            ratings.get(&anchor).expect("still rated").rating,
            anchor_rating
        );
        assert_eq!(ratings.pinned(), Some(&anchor));
    }

    /// The whole reason `retarget` exists rather than a fresh `Pool`: the clone archive and the
    /// currently active best/history slots must survive a permanent-membership change untouched.
    #[test]
    fn retarget_leaves_slots_and_archive_untouched() {
        let config = PoolConfig {
            best_slots: 1,
            history_slots: 1,
            ..Default::default()
        };
        let mut pool = pool(config);
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 200, 40);

        let slots_before: Vec<OpponentId> = pool
            .active()
            .into_iter()
            .filter(|id| matches!(id, OpponentId::Pool(_)))
            .collect();
        let archive_before = pool.archive().to_vec();

        pool.retarget(
            vec![Permanent::heuristic(PlayerCode::ER).pinned()],
            &mut ratings,
        )
        .expect("retarget");

        let slots_after: Vec<OpponentId> = pool
            .active()
            .into_iter()
            .filter(|id| matches!(id, OpponentId::Pool(_)))
            .collect();
        assert_eq!(slots_before, slots_after);
        assert_eq!(pool.archive(), archive_before.as_slice());
    }

    #[test]
    fn retarget_refuses_a_second_pin_or_a_duplicate_id() {
        let mut pool = pool(PoolConfig::default());
        let mut ratings = ratings();
        pool.register(&mut ratings);

        let two_pins = vec![
            Permanent::heuristic(PlayerCode::ER).pinned(),
            Permanent::heuristic(PlayerCode::R).pinned(),
        ];
        assert!(pool.retarget(two_pins, &mut ratings).is_err());

        let duplicate = vec![
            Permanent::heuristic(PlayerCode::ER).pinned(),
            Permanent::heuristic(PlayerCode::ER),
        ];
        assert!(pool.retarget(duplicate, &mut ratings).is_err());
    }

    /// The `.toml` spelling: a bare string for the argument-free draws, a table for the one that
    /// takes a horizon, and a named error for anything else.
    #[test]
    fn history_draw_parses_both_of_its_toml_shapes() {
        #[derive(serde::Deserialize)]
        struct Holder {
            draw: HistoryDraw,
        }
        let parse = |text: &str| toml::from_str::<Holder>(text).map(|holder| holder.draw);

        assert_eq!(
            parse("draw = \"uniform\"").expect("bare"),
            HistoryDraw::Uniform
        );
        assert_eq!(
            parse("draw = \"log_age\"").expect("bare"),
            HistoryDraw::LogAge
        );
        assert_eq!(
            parse("draw = { kind = \"recent\", max_age = 500 }").expect("table"),
            HistoryDraw::Recent { max_age: 500 }
        );
        // A horizon-less `recent` is the mistake worth naming: it would otherwise have to invent
        // one, and any value it invented would silently reshape the pool.
        assert!(parse("draw = { kind = \"recent\" }").is_err());
        assert!(parse("draw = \"newest\"").is_err());
    }

    #[test]
    fn rows_cover_every_active_member_and_their_weights_sum_to_one() {
        let mut pool = pool(PoolConfig::default());
        let mut ratings = ratings();
        pool.register(&mut ratings);
        run(&mut pool, &mut ratings, 260, 60);

        let rows = pool.rows(&ratings, 260);
        assert_eq!(rows.len(), pool.active().len());
        let total: f64 = rows.iter().map(|row| row.weight).sum();
        assert!((total - 1.0).abs() < 1e-9, "weights summed to {total}");
        assert!(rows.iter().any(|row| row.role == Role::Best));
        assert!(rows.iter().any(|row| row.role == Role::History));
        assert!(rows
            .iter()
            .any(|row| row.role == Role::Permanent && row.age_batches.is_none()));
    }
}
