//! Glicko-2 ratings — §1.5.6's "self-play elo", and the quantity §1.5.2's pool selects on.
//!
//! **Why a rating and not the window winrate.** The obvious way to rank pool members is to total
//! their winrate over the refresh window, and it is wrong three times over. PFSP samples
//! non-uniformly *and endogenously*, so the member that is being crushed is the one drawn least
//! often — its estimate is noisiest exactly when its eviction is being decided. The measurement is
//! taken against a learner that moves, so member A's early-window games and member B's late-window
//! games are not games against the same opponent. And a raw winrate carries no uncertainty, so five
//! games can evict six hundred. A rating handles all three: unequal game counts are what the
//! deviation is for, a moving opponent is carried by that opponent's own rating, and the deviation
//! is a first-class output rather than a footnote.
//!
//! **The scale needs an origin, and the grace period is not it.** Ratings are relative, so a
//! population whose members all move — or all rotate out — can drift hundreds of points with no
//! match result having changed. §1.5.6 wants the elo curve to be *the* generalization signal, which
//! it cannot be if the axis floats. So one entity is [`Entry::pinned`]: never updated, by
//! construction rather than by precaution. It is `er` and not `r`, because an origin placed on a
//! very weak player pushes the whole population to +600 and wastes the resolution where it is
//! actually read. A fixed point only fixes something if it plays, so the pool's uniform floor has to
//! keep feeding it games.
//!
//! **Only the learner drifts.** Glicko-2's volatility, and the `φ* = √(φ² + σ²)` step that consumes
//! it, model a strength that changes between rating periods. Frozen weights do not change: a pool
//! member's true rating on an `er`-pinned scale is a constant, and inflating its deviation over an
//! idle period would say the opposite — it would re-enter as an unknown after a quiet stretch and
//! throw away everything it had established. So [`Entry::drifts`] is set for the best-response
//! alone, which is also what gives it the wide deviation §1.5.2 wants it to carry into each period:
//! it learned all through the last one, and Glicko-2 assumes constant strength *within* a period.
//!
//! **The result graph is a star.** Every rated game is learner-vs-opponent — opponents never play
//! each other — so a period's results are a single centre and its spokes. That is why
//! [`RatingTable::record`] takes one score and why the whole period closes in one pass: each
//! opponent's update reads the learner's rating *as it stood at the period's start*, and the
//! learner's reads theirs. Updating in place as results arrive would make a member's rating depend
//! on where in the window it was drawn.

use std::collections::HashMap;
use std::fmt;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::players::{parse_player_code, PlayerCode};

/// Glicko-2's internal scale factor: rating points per unit of `μ`.
const SCALE: f64 = 173.7178;

/// The centre of the visible scale. Not a tunable — it is the constant Glicko-2's conversion is
/// written around, and moving it would only relabel every rating in the run.
pub const CENTRE: f64 = 1500.0;

/// Where an entity with no history starts. §1.5.2: every initialization gets the same one, so a
/// curriculum anchor introduced mid-run is not implicitly ranked by the order it appeared in.
pub const DEFAULT_RATING: f64 = CENTRE;
pub const DEFAULT_DEVIATION: f64 = 350.0;
pub const DEFAULT_VOLATILITY: f64 = 0.06;

/// Illinois-method tolerance on `ln σ²`, from Glickman's own write-up.
const CONVERGENCE: f64 = 1e-6;

/// Guards the volatility search against a pathological period (an entity that went from unbeaten to
/// unbeatable in one window) walking `B` out to infinity. Glickman's step 5 grows the bracket by
/// `τ` per try and expects a handful; a hundred is far past "the model does not fit".
const MAX_BRACKET_STEPS: u32 = 100;

/// Who a rated game was against. The three variants are rated, sampled and logged identically —
/// what differs is only who *runs* them, which is [`super::pool`]'s problem and not this module's.
///
/// The string form is the identity: it names metric series and keys the pool's on-disk state, so it
/// is what gets serialized, never the enum shape. `PlayerCode`'s own `Display` exists for the same
/// reason (`players/mod.rs`), and this extends the guarantee to the two variants it does not cover.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpponentId {
    /// A repo heuristic, run as a scripted seat.
    Heuristic(PlayerCode),
    /// A "baked" model from `models/<name>/`, frozen for the run's life and owned by the curriculum.
    Baked(String),
    /// A frozen best-response clone, named by the batch it was cloned at — which is both unique and
    /// the age the historical draw sorts on.
    Pool(u64),
}

impl fmt::Display for OpponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpponentId::Heuristic(code) => write!(f, "{code}"),
            OpponentId::Baked(name) => write!(f, "baked:{name}"),
            // Zero-padded so a lexical sort of the series names is a chronological one.
            OpponentId::Pool(batch) => write!(f, "pool:b{batch:09}"),
        }
    }
}

impl std::str::FromStr for OpponentId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(name) = s.strip_prefix("baked:") {
            if name.is_empty() {
                return Err("a baked opponent needs a name: `baked:<name>`".to_string());
            }
            return Ok(OpponentId::Baked(name.to_string()));
        }
        if let Some(batch) = s.strip_prefix("pool:b") {
            return batch
                .parse::<u64>()
                .map(OpponentId::Pool)
                .map_err(|_| format!("`{s}` is not a pool id: expected `pool:b<batch>`"));
        }
        parse_player_code(s).map(OpponentId::Heuristic)
    }
}

impl Serialize for OpponentId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for OpponentId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(D::Error::custom)
    }
}

/// One entity's rating, on the visible scale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rating {
    pub rating: f64,
    /// The uncertainty of `rating`, not its spread — roughly, the true value is within `±2·deviation`.
    pub deviation: f64,
    /// Glicko-2's `σ`: how erratic the entity's results have been. Only meaningful for something
    /// that can actually change (see the module docs), but carried by everything so a clone can
    /// inherit its parent's whole state in one move.
    pub volatility: f64,
}

impl Default for Rating {
    fn default() -> Self {
        Rating {
            rating: DEFAULT_RATING,
            deviation: DEFAULT_DEVIATION,
            volatility: DEFAULT_VOLATILITY,
        }
    }
}

impl Rating {
    /// Glicko-2's internal `(μ, φ)`.
    fn internal(&self) -> (f64, f64) {
        ((self.rating - CENTRE) / SCALE, self.deviation / SCALE)
    }

    fn from_internal(mu: f64, phi: f64, volatility: f64) -> Self {
        Rating {
            rating: mu * SCALE + CENTRE,
            deviation: phi * SCALE,
            volatility,
        }
    }

    /// A conservative one-number summary: the rating a result would be worth betting on. Used for
    /// ranking when the deviations differ by an order of magnitude, which is exactly the case a
    /// fresh clone creates.
    pub fn conservative(&self) -> f64 {
        self.rating - 2.0 * self.deviation
    }
}

/// The `.toml`-tunable half. `tau` is the only genuinely free parameter of Glicko-2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatingConfig {
    /// Constrains how much the volatility may move in one period. Glickman suggests 0.3–1.2; the
    /// smaller the value, the less a single surprising window is allowed to reopen a settled rating.
    pub tau: f64,
    /// Floor on the deviation. Without one a long-lived member's deviation tends to zero and its
    /// rating stops responding at all — which for a frozen entity is *nearly* right, but leaves no
    /// room for the scale itself to be corrected.
    pub min_deviation: f64,
    /// Ceiling on the deviation, and the value a fresh entity starts at.
    pub max_deviation: f64,
}

impl Default for RatingConfig {
    fn default() -> Self {
        RatingConfig {
            tau: 0.5,
            min_deviation: 30.0,
            max_deviation: DEFAULT_DEVIATION,
        }
    }
}

impl RatingConfig {
    pub fn validate(&self) -> Result<(), String> {
        // Written to catch NaN as well as zero: a NaN parameter would otherwise poison every rating
        // in the table silently, and the config is the last place it can still be named.
        if self.tau.is_nan() || self.tau <= 0.0 {
            return Err(format!("[rating] tau must be > 0, got {}", self.tau));
        }
        if self.min_deviation.is_nan() || self.min_deviation <= 0.0 {
            return Err(format!(
                "[rating] min_deviation must be > 0, got {}",
                self.min_deviation
            ));
        }
        if self.max_deviation < self.min_deviation {
            return Err(format!(
                "[rating] max_deviation ({}) must be ≥ min_deviation ({})",
                self.max_deviation, self.min_deviation
            ));
        }
        Ok(())
    }

    fn clamp(&self, deviation: f64) -> f64 {
        deviation.clamp(self.min_deviation, self.max_deviation)
    }
}

/// One rated entity and the flags that decide how the period treats it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub rating: Rating,
    /// Never updated. Exactly one entry carries this, and it is what makes the axis an axis.
    #[serde(default)]
    pub pinned: bool,
    /// Whether the entity's parameters change between periods. True for the best-response alone —
    /// see the module docs for why a frozen checkpoint must *not* have its deviation inflated by
    /// sitting out.
    #[serde(default)]
    pub drifts: bool,
    /// Rated games played, all periods. The pool's eviction floor reads this, and a rating with no
    /// games behind it should never decide anything.
    #[serde(default)]
    pub games: u64,
}

impl Entry {
    /// A frozen entity with no history.
    pub fn fresh() -> Self {
        Entry {
            rating: Rating::default(),
            pinned: false,
            drifts: false,
            games: 0,
        }
    }

    /// The best-response: the only entity whose strength changes between periods.
    pub fn learner() -> Self {
        Entry {
            drifts: true,
            ..Entry::fresh()
        }
    }

    /// The scale's origin: fixed rating, fixed deviation, never updated.
    pub fn pinned(rating: f64, deviation: f64) -> Self {
        Entry {
            rating: Rating {
                rating,
                deviation,
                volatility: DEFAULT_VOLATILITY,
            },
            pinned: true,
            drifts: false,
            games: 0,
        }
    }

    /// What a clone inherits (§1.5.2): the parent's rating *and* its uncertainty, frozen from here
    /// on. The rating is systematically stale — it was earned over the window by policies weaker
    /// than the one being cloned — which is why the pool protects a new member from eviction by a
    /// game count rather than trusting this number on arrival.
    pub fn cloned_from(parent: &Rating) -> Self {
        Entry {
            rating: *parent,
            pinned: false,
            drifts: false,
            games: 0,
        }
    }
}

/// The learner's key. It is rated like everything else, but it is never an *opponent*, so it lives
/// outside [`OpponentId`]. Public because the §1.5.7 harvest names pilots in the same vocabulary,
/// and a run's ratings and its labels have to agree on what the learner is called.
pub const LEARNER: &str = "learner";

/// Glicko-2's `g(φ)`: how much weight an opponent's result carries, discounted by how unsure we are
/// of that opponent.
fn g(phi: f64) -> f64 {
    1.0 / (1.0 + 3.0 * phi * phi / (std::f64::consts::PI * std::f64::consts::PI)).sqrt()
}

/// Expected score of `mu` against `(mu_j, phi_j)`.
fn expected(mu: f64, mu_j: f64, phi_j: f64) -> f64 {
    1.0 / (1.0 + (-g(phi_j) * (mu - mu_j)).exp())
}

/// The probability `a` scores against `b`, accounting for *both* uncertainties.
///
/// This is what §1.5.2's PFSP weight is computed from, rather than a measured winrate: a member
/// that has just entered the pool has no measurement, but it does have a rating, so its sampling
/// weight is defined from its first game instead of after its first hundred.
pub fn win_probability(a: &Rating, b: &Rating) -> f64 {
    let (mu_a, phi_a) = a.internal();
    let (mu_b, phi_b) = b.internal();
    expected(mu_a, mu_b, (phi_a * phi_a + phi_b * phi_b).sqrt())
}

/// One game's contribution to a period, seen from the entity being updated.
#[derive(Debug, Clone, Copy)]
struct Match {
    opponent: Rating,
    /// `1.0` win, `0.5` tie, `0.0` loss. Glicko-2 scores a tie as half a win, which is the right
    /// reading *here* and deliberately not the one §1.5.6 uses for its winrate series — a tie there
    /// is counted and reported separately, never folded in.
    score: f64,
}

/// Glickman's step 5: solve for the new volatility by Illinois-method bisection on `ln σ²`.
fn solve_volatility(config: &RatingConfig, phi: f64, sigma: f64, v: f64, delta: f64) -> f64 {
    let a = (sigma * sigma).ln();
    let tau_sq = config.tau * config.tau;
    let delta_sq = delta * delta;
    let phi_sq = phi * phi;

    let f = |x: f64| {
        let ex = x.exp();
        let denom = phi_sq + v + ex;
        (ex * (delta_sq - phi_sq - v - ex)) / (2.0 * denom * denom) - (x - a) / tau_sq
    };

    let mut lo = a;
    // The bracket's upper end has a closed form when the period's surprise exceeds the uncertainty
    // that could explain it; otherwise it is walked down until the sign flips.
    let mut hi = if delta_sq > phi_sq + v {
        (delta_sq - phi_sq - v).ln()
    } else {
        let mut k = 1.0;
        let mut candidate = a - k * config.tau;
        let mut steps = 0;
        while f(candidate) < 0.0 && steps < MAX_BRACKET_STEPS {
            k += 1.0;
            candidate = a - k * config.tau;
            steps += 1;
        }
        candidate
    };

    let mut f_lo = f(lo);
    let mut f_hi = f(hi);
    let mut steps = 0;
    while (hi - lo).abs() > CONVERGENCE && steps < MAX_BRACKET_STEPS {
        let mid = lo + (lo - hi) * f_lo / (f_hi - f_lo);
        let f_mid = f(mid);
        if f_mid * f_hi <= 0.0 {
            lo = hi;
            f_lo = f_hi;
        } else {
            // Illinois: halving the retained endpoint's value is what stops one stubborn side from
            // holding the bracket open for hundreds of iterations.
            f_lo /= 2.0;
        }
        hi = mid;
        f_hi = f_mid;
        steps += 1;
    }
    (hi / 2.0).exp()
}

/// One entity's period update. `matches` is empty for an entity that sat the period out.
fn advance(config: &RatingConfig, entry: &Entry, matches: &[Match]) -> Rating {
    if entry.pinned {
        return entry.rating;
    }
    let (mu, phi) = entry.rating.internal();
    let sigma = entry.rating.volatility;

    if matches.is_empty() {
        // A frozen entity that did not play learned nothing and forgot nothing. Only a drifting one
        // becomes less certain for having sat still.
        let phi_next = if entry.drifts {
            (phi * phi + sigma * sigma).sqrt()
        } else {
            phi
        };
        return Rating {
            deviation: config.clamp(phi_next * SCALE),
            ..entry.rating
        };
    }

    let mut v_inv = 0.0;
    let mut delta_sum = 0.0;
    for m in matches {
        let (mu_j, phi_j) = m.opponent.internal();
        let g_j = g(phi_j);
        let e_j = expected(mu, mu_j, phi_j);
        v_inv += g_j * g_j * e_j * (1.0 - e_j);
        delta_sum += g_j * (m.score - e_j);
    }

    // A period in which every game was a foregone conclusion carries no information about the
    // rating, and `v` diverges. Leaving the rating where it was is the honest answer; the
    // alternative is a division that produces an infinity and poisons the table.
    if v_inv <= 0.0 || !v_inv.is_finite() {
        return entry.rating;
    }

    let v = 1.0 / v_inv;
    let delta = v * delta_sum;

    let sigma_next = if entry.drifts {
        solve_volatility(config, phi, sigma, v, delta)
    } else {
        sigma
    };
    let phi_star = if entry.drifts {
        (phi * phi + sigma_next * sigma_next).sqrt()
    } else {
        phi
    };

    let phi_next = 1.0 / (1.0 / (phi_star * phi_star) + v_inv).sqrt();
    let mu_next = mu + phi_next * phi_next * delta_sum;

    let mut next = Rating::from_internal(mu_next, phi_next, sigma_next);
    next.deviation = config.clamp(next.deviation);
    next
}

/// Results awaiting the close of the current period, and the ratings they will be scored against.
///
/// §1.5.2's rating period *is* the pool's refresh window, so this fills for `refresh_every` batches
/// and empties in one pass. Glicko-2 is not an incremental algorithm — the deviation and the
/// volatility are defined over a period, not over a game — so there is deliberately no way to ask
/// this table for a rating that reflects a game played five minutes ago.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Pending {
    /// Learner-seat score per game, keyed by opponent: a star graph, one spoke per game.
    scores: Vec<(OpponentId, f64)>,
}

impl Pending {
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

/// Every rated entity in the run, plus the period being accumulated.
///
/// Serialized whole into §1.5.5's hot checkpoint: a resume that re-rolled its ratings would lose
/// the history that §1.5.2's eviction reads, and would also lose the ratings of *evicted* members —
/// which the spec keeps on purpose, so a checkpoint drawn back into the pool later resumes from
/// where it left off rather than from 1500.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatingTable {
    #[serde(skip)]
    config: RatingConfig,
    learner: Entry,
    /// Keyed by the string form, because that is the identity that survives a rename of the enum.
    entries: HashMap<OpponentId, Entry>,
    pending: Pending,
    /// Periods closed. Only diagnostic — a rating that has been through two periods and one that
    /// has been through fifty read identically otherwise.
    periods: u64,
}

impl RatingTable {
    pub fn new(config: RatingConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(RatingTable {
            config,
            learner: Entry::learner(),
            entries: HashMap::new(),
            pending: Pending::default(),
            periods: 0,
        })
    }

    /// Re-attaches the `.toml`'s parameters to a table read back from a checkpoint. The config is
    /// not serialized with the table for the reason §1.5.5 gives about AdamW: a stored parameter
    /// could resurrect a value the run's `.toml` no longer asks for.
    pub fn with_config(mut self, config: RatingConfig) -> Result<Self, String> {
        config.validate()?;
        self.config = config;
        Ok(self)
    }

    pub fn config(&self) -> &RatingConfig {
        &self.config
    }

    pub fn periods(&self) -> u64 {
        self.periods
    }

    pub fn learner(&self) -> &Entry {
        &self.learner
    }

    pub fn get(&self, id: &OpponentId) -> Option<&Entry> {
        self.entries.get(id)
    }

    /// The rating an unknown opponent is treated as having. §1.5.2: one default for every
    /// initialization, so a curriculum anchor added at batch 90 000 is not implicitly ranked by
    /// having arrived late.
    pub fn rating_of(&self, id: &OpponentId) -> Rating {
        self.entries
            .get(id)
            .map(|entry| entry.rating)
            .unwrap_or_default()
    }

    /// Registers an entity, or leaves it alone if it is already known. Idempotent, because the pool
    /// re-declares its slots on every refresh and a resume re-declares the whole panel.
    pub fn ensure(&mut self, id: OpponentId, entry: Entry) -> &Entry {
        self.entries.entry(id).or_insert(entry)
    }

    /// Registers an entity, overwriting whatever was there.
    ///
    /// Separate from [`RatingTable::ensure`] because the two answer opposite questions and the
    /// difference is load-bearing: re-declaring the panel must *not* reset a rating, while a baked
    /// model arriving with a rating recorded in its `meta.toml` must not be silently ignored for
    /// having been touched first.
    /// Drops the rating of every clone outside `keep`, leaving heuristics and baked models alone.
    ///
    /// The one caller is [`super::init`]'s partial pool carry, and the reason is [`Self::scalars`]:
    /// `elo/pool_mean` and `elo/pool_best` average over every `Pool` entry, evicted ones included.
    /// A table carrying ratings for clones whose weights were not copied would report a pool the
    /// run does not have — the entries are unreachable, so the lie is only in the curve, which is
    /// exactly where it would be believed.
    pub fn retain_clones(&mut self, keep: &[u64]) {
        self.entries.retain(|id, _| match id {
            OpponentId::Pool(batch) => keep.contains(batch),
            _ => true,
        });
    }

    pub fn set(&mut self, id: OpponentId, entry: Entry) {
        self.entries.insert(id, entry);
    }

    /// Pins `id` as the scale's origin, replacing any previous pin.
    ///
    /// Enforced as "at most one" rather than left to the config: two fixed points do not define a
    /// scale, they over-determine it, and the second one would be silently dragged off its stated
    /// value by nothing at all.
    pub fn pin(&mut self, id: OpponentId, rating: f64, deviation: f64) {
        for entry in self.entries.values_mut() {
            entry.pinned = false;
        }
        let games = self.entries.get(&id).map(|entry| entry.games).unwrap_or(0);
        let mut pinned = Entry::pinned(rating, deviation);
        pinned.games = games;
        self.entries.insert(id, pinned);
    }

    pub fn pinned(&self) -> Option<&OpponentId> {
        self.entries
            .iter()
            .find(|(_, entry)| entry.pinned)
            .map(|(id, _)| id)
    }

    /// One finished game, from the learner's seat: `1.0` win, `0.5` tie, `0.0` loss.
    ///
    /// The env's terminal reward is `−1 / 0 / +1` (§1.5.1), so callers convert with
    /// [`score_from_reward`] rather than passing the reward through — the two conventions differ and
    /// a sign error here would be invisible until the whole pool inverted.
    pub fn record(&mut self, opponent: OpponentId, score: f64) {
        self.entries
            .entry(opponent.clone())
            .or_insert_with(Entry::fresh);
        self.pending.scores.push((opponent, score));
    }

    pub fn pending(&self) -> &Pending {
        &self.pending
    }

    /// Closes the period: every rating moves at once, read against the ratings as they stood when
    /// the period opened.
    ///
    /// Returns the number of games it consumed, which is zero for an empty period — and an empty
    /// period is still a period for a drifting entity, whose deviation grows for having sat it out.
    pub fn close_period(&mut self) -> u64 {
        let games = self.pending.scores.len() as u64;

        // Snapshot first: an opponent's update must read the learner as it was, not as it will be.
        let learner_before = self.learner.rating;
        let opponents_before: HashMap<OpponentId, Rating> = self
            .entries
            .iter()
            .map(|(id, entry)| (id.clone(), entry.rating))
            .collect();

        let mut learner_matches = Vec::with_capacity(self.pending.scores.len());
        let mut per_opponent: HashMap<&OpponentId, Vec<Match>> = HashMap::new();
        for (id, score) in &self.pending.scores {
            let opponent = opponents_before.get(id).copied().unwrap_or_default();
            learner_matches.push(Match {
                opponent,
                score: *score,
            });
            per_opponent.entry(id).or_default().push(Match {
                opponent: learner_before,
                score: 1.0 - *score,
            });
        }

        let learner_next = advance(&self.config, &self.learner, &learner_matches);
        let updates: Vec<(OpponentId, Rating, u64)> = self
            .entries
            .iter()
            .map(|(id, entry)| {
                let matches = per_opponent.get(id).map(Vec::as_slice).unwrap_or(&[]);
                (
                    id.clone(),
                    advance(&self.config, entry, matches),
                    matches.len() as u64,
                )
            })
            .collect();

        self.learner.rating = learner_next;
        self.learner.games += games;
        for (id, rating, played) in updates {
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.rating = rating;
                entry.games += played;
            }
        }

        self.pending.scores.clear();
        self.periods += 1;
        games
    }

    /// Every rated opponent, sorted by identity so a log line is stable across runs.
    pub fn table(&self) -> Vec<(OpponentId, Entry)> {
        let mut rows: Vec<(OpponentId, Entry)> = self
            .entries
            .iter()
            .map(|(id, entry)| (id.clone(), entry.clone()))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        rows
    }

    /// The scalar series §1.5.6 asks for, named for the logger.
    ///
    /// Per-*category* rather than per-checkpoint: a series per pool member would leave hundreds of
    /// dead curves behind over a run, one per evicted clone. The pool's own table (written as JSONL
    /// beside the metrics) is where a member is followed individually.
    pub fn scalars(&self) -> Vec<(String, f64)> {
        let mut out = vec![
            (format!("elo/{LEARNER}"), self.learner.rating.rating),
            (
                format!("elo/{LEARNER}_deviation"),
                self.learner.rating.deviation,
            ),
            (
                format!("elo/{LEARNER}_volatility"),
                self.learner.rating.volatility,
            ),
        ];
        let pool: Vec<f64> = self
            .entries
            .iter()
            .filter(|(id, _)| matches!(id, OpponentId::Pool(_)))
            .map(|(_, entry)| entry.rating.rating)
            .collect();
        if !pool.is_empty() {
            let mean = pool.iter().sum::<f64>() / pool.len() as f64;
            let best = pool.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            out.push(("elo/pool_mean".to_string(), mean));
            out.push(("elo/pool_best".to_string(), best));
        }
        out
    }
}

/// §1.5.1's terminal reward, as a Glicko score. The two scales differ (`−1/0/+1` against `0/½/1`)
/// and nothing else in the loop needs the conversion, so it lives here rather than at each call.
pub fn score_from_reward(reward: f32) -> f64 {
    (reward as f64 + 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> RatingTable {
        RatingTable::new(RatingConfig::default()).expect("config")
    }

    /// Glickman's own worked example ("Example calculation", *The Glicko-2 system*): a player at
    /// 1500/200 with τ = 0.5 plays three opponents and lands at ≈ 1464.06 / 151.52 / 0.05999.
    ///
    /// Run through [`advance`] directly rather than the table, because the published example is a
    /// single player's update against fixed opponents — which is what this function is.
    #[test]
    fn glickman_worked_example_reproduces_the_published_numbers() {
        let config = RatingConfig {
            tau: 0.5,
            min_deviation: 0.0001,
            max_deviation: 10_000.0,
        };
        let entry = Entry {
            rating: Rating {
                rating: 1500.0,
                deviation: 200.0,
                volatility: 0.06,
            },
            pinned: false,
            drifts: true,
            games: 0,
        };
        let matches = [
            (1400.0, 30.0, 1.0),
            (1550.0, 100.0, 0.0),
            (1700.0, 300.0, 0.0),
        ]
        .map(|(rating, deviation, score)| Match {
            opponent: Rating {
                rating,
                deviation,
                volatility: 0.06,
            },
            score,
        });

        let next = advance(&config, &entry, &matches);
        assert!(
            (next.rating - 1464.06).abs() < 0.01,
            "rating was {}",
            next.rating
        );
        assert!(
            (next.deviation - 151.52).abs() < 0.01,
            "deviation was {}",
            next.deviation
        );
        assert!(
            (next.volatility - 0.05999).abs() < 0.00001,
            "volatility was {}",
            next.volatility
        );
    }

    /// The asymmetry the module docs argue for: an idle period costs the learner certainty and
    /// costs a frozen checkpoint nothing.
    #[test]
    fn an_idle_period_widens_the_learner_and_leaves_frozen_entities_alone() {
        let mut ratings = table();
        ratings.ensure(OpponentId::Heuristic(PlayerCode::R), Entry::fresh());
        // A played period first: everything starts at the deviation ceiling, where the drift step
        // has nowhere to go and the clamp would hide the asymmetry this test is about.
        for _ in 0..100 {
            ratings.record(OpponentId::Heuristic(PlayerCode::R), 1.0);
        }
        ratings.close_period();

        let learner_before = ratings.learner().rating.deviation;
        let frozen_before = ratings
            .get(&OpponentId::Heuristic(PlayerCode::R))
            .expect("registered")
            .rating
            .deviation;

        assert_eq!(ratings.close_period(), 0);

        assert!(ratings.learner().rating.deviation > learner_before);
        assert_eq!(
            ratings
                .get(&OpponentId::Heuristic(PlayerCode::R))
                .expect("registered")
                .rating
                .deviation,
            frozen_before
        );
    }

    #[test]
    fn the_pinned_anchor_never_moves() {
        let mut ratings = table();
        let anchor = OpponentId::Heuristic(PlayerCode::ER);
        ratings.pin(anchor.clone(), 1500.0, 30.0);

        for _ in 0..200 {
            ratings.record(anchor.clone(), 1.0);
        }
        ratings.close_period();

        let entry = ratings.get(&anchor).expect("pinned");
        assert_eq!(entry.rating.rating, 1500.0);
        assert_eq!(entry.rating.deviation, 30.0);
        // It still counts its games: the eviction floor and the uniform floor both read them.
        assert_eq!(entry.games, 200);
        // And the learner moves against it, which is the whole point of having an origin.
        assert!(ratings.learner().rating.rating > 1500.0);
    }

    #[test]
    fn pinning_a_second_anchor_unpins_the_first() {
        let mut ratings = table();
        let first = OpponentId::Heuristic(PlayerCode::R);
        let second = OpponentId::Heuristic(PlayerCode::ER);
        ratings.pin(first.clone(), 1500.0, 30.0);
        ratings.pin(second.clone(), 1500.0, 30.0);

        assert_eq!(ratings.pinned(), Some(&second));
        assert!(!ratings.get(&first).expect("known").pinned);
    }

    /// The star graph: what the learner gains, its opponents lose.
    #[test]
    fn a_period_moves_the_learner_and_its_opponents_in_opposite_directions() {
        let mut ratings = table();
        let opponent = OpponentId::Pool(400);
        ratings.ensure(opponent.clone(), Entry::fresh());
        for _ in 0..100 {
            ratings.record(opponent.clone(), 1.0);
        }
        assert_eq!(ratings.close_period(), 100);

        assert!(ratings.learner().rating.rating > CENTRE);
        assert!(ratings.get(&opponent).expect("known").rating.rating < CENTRE);
        assert_eq!(ratings.get(&opponent).expect("known").games, 100);
    }

    /// Unequal game counts are the case the raw window winrate gets wrong: the member with five
    /// games must not end up ranked above the member with five hundred on the strength of a fluke.
    #[test]
    fn a_thin_sample_moves_a_rating_less_than_a_thick_one() {
        let mut ratings = table();
        let thin = OpponentId::Pool(1);
        let thick = OpponentId::Pool(2);
        ratings.ensure(thin.clone(), Entry::fresh());
        ratings.ensure(thick.clone(), Entry::fresh());
        for _ in 0..5 {
            ratings.record(thin.clone(), 0.0);
        }
        for _ in 0..500 {
            ratings.record(thick.clone(), 0.0);
        }
        ratings.close_period();

        let thin = ratings.get(&thin).expect("known").rating;
        let thick = ratings.get(&thick).expect("known").rating;
        // Both beat the learner every game; only the thick one has established it.
        assert!(thick.rating > thin.rating);
        assert!(thick.deviation < thin.deviation);
    }

    #[test]
    fn ties_score_half() {
        let mut ratings = table();
        let opponent = OpponentId::Pool(7);
        ratings.ensure(opponent.clone(), Entry::fresh());
        for _ in 0..300 {
            ratings.record(opponent.clone(), score_from_reward(0.0));
        }
        ratings.close_period();

        // Two entities at the same rating drawing every game learn nothing about the gap between
        // them, so only the certainty moves.
        assert!((ratings.learner().rating.rating - CENTRE).abs() < 1.0);
        assert!(ratings.get(&opponent).expect("known").rating.deviation < DEFAULT_DEVIATION);
    }

    #[test]
    fn reward_maps_onto_the_glicko_scale() {
        assert_eq!(score_from_reward(1.0), 1.0);
        assert_eq!(score_from_reward(0.0), 0.5);
        assert_eq!(score_from_reward(-1.0), 0.0);
    }

    #[test]
    fn the_deviation_is_clamped_at_both_ends() {
        let config = RatingConfig {
            tau: 0.5,
            min_deviation: 100.0,
            max_deviation: 120.0,
        };
        let mut ratings = RatingTable::new(config).expect("config");
        let opponent = OpponentId::Pool(3);
        ratings.ensure(opponent.clone(), Entry::fresh());
        for _ in 0..5_000 {
            ratings.record(opponent.clone(), 1.0);
        }
        ratings.close_period();

        assert!(ratings.learner().rating.deviation >= 100.0);
        assert!(ratings.learner().rating.deviation <= 120.0);
    }

    #[test]
    fn a_clone_inherits_its_parents_rating_and_stops_drifting() {
        let mut ratings = table();
        let sparring = OpponentId::Heuristic(PlayerCode::R);
        ratings.ensure(sparring.clone(), Entry::fresh());
        for _ in 0..200 {
            ratings.record(sparring.clone(), 1.0);
        }
        ratings.close_period();

        let parent = ratings.learner().rating;
        let clone = Entry::cloned_from(&parent);
        assert_eq!(clone.rating, parent);
        assert!(!clone.drifts);
        assert_eq!(clone.games, 0);
    }

    #[test]
    fn win_probability_is_monotone_in_the_gap_and_symmetric_at_parity() {
        let even = Rating::default();
        assert!((win_probability(&even, &even) - 0.5).abs() < 1e-12);

        let strong = Rating {
            rating: 1800.0,
            deviation: 50.0,
            volatility: 0.06,
        };
        let weak = Rating {
            rating: 1200.0,
            deviation: 50.0,
            volatility: 0.06,
        };
        assert!(win_probability(&strong, &weak) > 0.5);
        assert!(win_probability(&weak, &strong) < 0.5);
        assert!(
            (win_probability(&strong, &weak) + win_probability(&weak, &strong) - 1.0).abs() < 1e-12
        );
    }

    /// A wide deviation on either side pulls the prediction back toward a coin flip, which is what
    /// makes this usable as PFSP's weight for a member that has never played.
    #[test]
    fn uncertainty_flattens_the_prediction() {
        let strong = Rating {
            rating: 1800.0,
            deviation: 50.0,
            volatility: 0.06,
        };
        let sure = Rating {
            rating: 1200.0,
            deviation: 50.0,
            volatility: 0.06,
        };
        let unsure = Rating {
            rating: 1200.0,
            deviation: 350.0,
            volatility: 0.06,
        };
        assert!(win_probability(&strong, &unsure) < win_probability(&strong, &sure));
    }

    #[test]
    fn opponent_ids_round_trip_through_their_string_form() {
        let cases = [
            OpponentId::Heuristic(PlayerCode::ER),
            OpponentId::Heuristic(PlayerCode::E { max_depth: 2 }),
            OpponentId::Baked("Cliff".to_string()),
            OpponentId::Pool(12_400),
        ];
        for id in cases {
            let text = id.to_string();
            let back: OpponentId = text.parse().expect("parses");
            assert_eq!(back, id, "round trip of `{text}`");
            let json = serde_json::to_string(&id).expect("serialize");
            let from_json: OpponentId = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(from_json, id);
        }
    }

    /// Zero-padded, so the historical draw and the metric names agree on what "older" means.
    #[test]
    fn pool_ids_sort_chronologically_as_strings() {
        let mut names = [
            OpponentId::Pool(1000),
            OpponentId::Pool(25),
            OpponentId::Pool(3),
        ]
        .map(|id| id.to_string());
        names.sort();
        assert_eq!(
            names,
            ["pool:b000000003", "pool:b000000025", "pool:b000001000"]
        );
    }

    #[test]
    fn a_table_round_trips_through_json_with_its_pending_period() {
        let mut ratings = table();
        let opponent = OpponentId::Pool(64);
        ratings.pin(OpponentId::Heuristic(PlayerCode::ER), 1500.0, 30.0);
        ratings.ensure(opponent.clone(), Entry::fresh());
        for _ in 0..50 {
            ratings.record(opponent.clone(), 1.0);
        }
        ratings.close_period();
        ratings.record(opponent.clone(), 0.0);

        let json = serde_json::to_string(&ratings).expect("serialize");
        let restored: RatingTable = serde_json::from_str(&json).expect("deserialize");
        let restored = restored
            .with_config(RatingConfig::default())
            .expect("config");

        assert_eq!(restored.periods(), ratings.periods());
        assert_eq!(restored.learner().rating, ratings.learner().rating);
        assert_eq!(restored.pending().len(), 1);
        assert_eq!(restored.get(&opponent), ratings.get(&opponent));
        assert_eq!(
            restored.pinned(),
            Some(&OpponentId::Heuristic(PlayerCode::ER))
        );
    }

    #[test]
    fn an_invalid_config_is_refused() {
        assert!(RatingTable::new(RatingConfig {
            tau: 0.0,
            ..Default::default()
        })
        .is_err());
        assert!(RatingTable::new(RatingConfig {
            min_deviation: 100.0,
            max_deviation: 50.0,
            ..Default::default()
        })
        .is_err());
    }
}
