//! The run's `.toml` — `RL_ARCHITECTURE.md` §1.5.5, "`config/*.toml` (sources)".
//!
//! Deliberately **only what exists**. §1.5 says everything in Part 5 is a `.toml` default, but
//! writing the opponent-panel and curriculum sections before those systems exist would fix their
//! interfaces from the outside, sight unseen. So this file grows one section per part as the
//! parts land.
//!
//! One rule holds across all of them: **this file is the whole run.** A hyperparameter left in
//! Rust — the Part 4 sizes were, until [`ModelConfig`] got its section here — makes the
//! [`super::run_dir::RunDir`] clone a partial record, and a run whose artefacts do not describe
//! the run that made them is not reproducible whatever the seed says.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::deck_db::DeckDb;
use super::harvest::Sampling;
use super::run_dir::RunDir;
use super::sampler::{DeckSampler, DeckSource, SourceSpec};
use super::schedule::{Schedule, ScheduleSpec};

/// `deny_unknown_fields`, here and on every section below: a misspelled or misplaced key that
/// silently falls back to its default is a run whose `.toml` no longer describes it — the §1.5.5
/// reproducibility contract inverted. Refusing at load names the key while fixing it is still
/// free. The tolerance this forgoes is *forward* only (an old binary refusing a newer config),
/// which is the refusal that should happen; resuming an older clone stays fine, since older
/// files only ever have fewer fields.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub run: RunSection,
    pub decks: DeckSection,
    pub rollout: RolloutSection,
    pub step: StepSection,
    #[serde(default)]
    pub checkpoint: CheckpointSection,
    #[serde(default)]
    pub harvest: HarvestSection,
    #[serde(default)]
    pub recovery: RecoverySection,
    #[serde(default)]
    pub eval: EvalSection,
    /// §1.5.4's curriculum. Absent or empty `stages` means the run stays on its flat
    /// `[decks]`/`[eval]`/`[pool]`/`[magnet.seed]` fields exactly as before this section existed
    /// — see [`CurriculumSection`].
    #[serde(default)]
    pub curriculum: CurriculumSection,
    /// §1.5.1b's magnet. Absent means the section's `Default`, which is **off** — a config written
    /// before the magnet existed has to resume into the run it described, not into a different
    /// algorithm.
    #[serde(default)]
    pub magnet: MagnetSection,
    /// §1.5.2's opponent pool. Absent means the section's `Default`, which is **off** — the panel
    /// stays `[rollout] opponents`, resolved as scripted seats, which is what a config written
    /// before the pool existed described.
    #[serde(default)]
    pub pool: PoolSection,
    /// The Part 4 sizes (§1.4.3). Absent means the v1 defaults.
    #[cfg(feature = "rl-model")]
    #[serde(default)]
    pub model: crate::rl::model::config::ModelConfig,
    /// `[model]` has to parse under every feature set: without `rl-model` the sizes have no
    /// consumer, but a build that cannot use them must not refuse a config that carries them —
    /// which `deny_unknown_fields` would otherwise do, since the real field above is cfg-gated.
    #[cfg(not(feature = "rl-model"))]
    #[serde(default, rename = "model")]
    #[allow(dead_code)]
    model_unused: Option<toml::Value>,
    /// The frozen text-encoder artifact (§1.2.9), as a path to the JSON
    /// `auxiliaries/text_embeddings` writes.
    ///
    /// A path with a default rather than an `Option`, and a missing file is an error, because the
    /// silent zero is what this field exists to make impossible (incident: NOTES.md, "Schéma
    /// d'observation — historique des versions"). The zero-text ablation is still available, and is
    /// now something a config has to *say*, by setting this to the empty string.
    #[serde(default = "default_text_embeddings")]
    pub text_embeddings: String,
    /// Where this was parsed from, so the run clone cannot drift from what was loaded.
    #[serde(skip)]
    source: PathBuf,
}

fn default_text_embeddings() -> String {
    "auxiliaries/text_embeddings/out/text_embeddings.json".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    /// Master seed. §1.5.5 derives every per-env child seed from it, so this one number is the
    /// whole reproducibility contract.
    pub seed: u64,
    /// Names `runs/<name>/`. Deliberately not overridable from the command line — see
    /// [`super::run_dir`].
    pub name: String,
    /// Where the run directories live.
    #[serde(default = "default_runs_root")]
    pub root: PathBuf,
}

fn default_runs_root() -> PathBuf {
    PathBuf::from("runs")
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeckSection {
    /// Directory holding the compiled DBs, built by `auxiliaries/build_deck_dbs.py`.
    pub root: PathBuf,
    /// Which DB this run draws from — `meta` or `tutorial` (§1.5.3). Read only to build the run's
    /// very first sampler; a non-empty `[[curriculum.stages]]` (§1.5.4) owns this choice per
    /// stage and supersedes it from there on.
    #[serde(default)]
    pub db: String,
    /// Archetypes to restrict the draw to; absent or empty means the whole DB. For `tutorial`
    /// the archetype is the difficulty tier.
    #[serde(default)]
    pub archetypes: Vec<String>,
    /// Several DBs at once, each with its own share and archetype slice (§1.5.3) — the
    /// alternative to `db`, never an addition to it.
    #[serde(default)]
    pub mix: Vec<MixSection>,
    pub pure_mirror: f64,
    pub mirror: f64,
}

/// One `{ db, share, archetypes }` of a `mix`.
///
/// The share is relative and normalized against its siblings, the convention `[magnet.seed]`'s
/// anchors already use. It has to be stated: a mix is precisely the place deck-uniformity stops
/// being the right default, since `tutorial` concatenated into `meta` is 0.4 % of the decks.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixSection {
    pub db: String,
    pub share: f64,
    #[serde(default)]
    pub archetypes: Vec<String>,
}

/// Resolves the `db` + `archetypes` shorthand, or the explicit `mix`, into the sampler's sources.
///
/// The two are alternatives rather than layers: a file setting both would be saying two different
/// things about one draw, and resolving that silently is how a run ends up training on a
/// distribution nobody asked for.
fn resolve_sources(
    db: &str,
    archetypes: &[String],
    mix: &[MixSection],
) -> Result<Vec<SourceSpec>, String> {
    match (db.is_empty(), mix.is_empty()) {
        (true, true) => Err("needs either db = \"…\" or a non-empty mix = […]".to_string()),
        (false, false) => Err(format!(
            "sets both db = {db:?} and mix = […], which are alternatives, not layers"
        )),
        (false, true) => Ok(vec![SourceSpec {
            db: db.to_string(),
            share: 1.0,
            archetypes: archetypes.to_vec(),
        }]),
        (true, false) => {
            // Beside a mix this would be silently dropped, and the run would train on every
            // archetype of every source without a word about it.
            if !archetypes.is_empty() {
                return Err(
                    "archetypes = […] belongs inside each mix entry, not beside the mix"
                        .to_string(),
                );
            }
            // Checked here and not just at the transition, because it costs nothing: the DB
            // behind a late stage is not loaded until the run reaches it, but a share is a number
            // in the file.
            for entry in mix {
                if !(entry.share.is_finite() && entry.share > 0.0) {
                    return Err(format!(
                        "mix entry {:?} has share {}, which must be finite and above zero",
                        entry.db, entry.share
                    ));
                }
            }
            Ok(mix
                .iter()
                .map(|entry| SourceSpec {
                    db: entry.db.clone(),
                    share: entry.share,
                    archetypes: entry.archetypes.clone(),
                })
                .collect())
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutSection {
    /// Parallel envs. The §1.5.5 measurement makes this the single most consequential number in
    /// the file on GPU and an irrelevance on CPU.
    pub envs: usize,
    /// The §1.5.2 frozen panel, as the CLI's player codes (`r`, `w`, `e2`, …).
    pub opponents: Vec<String>,
    /// Frames per on-policy batch — a floor on *finished* frames (§1.5.1).
    pub frames_per_batch: usize,
    /// Batches to run — Part 1's step-budget stop. §1.5.4 adds the other one, a winrate-vs-panel
    /// plateau, alongside it rather than instead of it: whichever fires first ends the run.
    pub batches: usize,
}

/// §1.5.5 checkpointing. The cadence is a bet on how much rollout a crash may cost — the hot save
/// is worth roughly one batch of collection, so saving every batch would be pure overhead.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CheckpointSection {
    pub every_batches: u64,
    /// Complete hot checkpoints retained. Two is the floor worth having: one to resume from, one
    /// as the fallback if the newest turns out to have been written from a poisoned state.
    pub keep_hot: usize,
}

impl Default for CheckpointSection {
    fn default() -> Self {
        CheckpointSection {
            every_batches: 20,
            keep_hot: 2,
        }
    }
}

/// §1.5.7's label harvest. `log` is the run-wide rate; a curriculum stage's `harvest_log` (§1.5.4)
/// can vary it from there, but only while this section builds a [`super::harvest::Harvest`] at
/// all — see [`super::harvest::Harvest::set_sampling`]'s doc for why a stage cannot turn it on
/// from nothing.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HarvestSection {
    /// `false` / `true` / a probability. Sampled per *game*, never per row — §1.5.7's denominators
    /// depend on every card of a harvested deck getting a row whether it was drawn or not.
    pub log: Sampling,
    /// Batches between shard flushes. A shard is written from memory, so this trades peak memory
    /// against how much of a harvest a crash costs.
    pub every_batches: u64,
}

impl Default for HarvestSection {
    fn default() -> Self {
        HarvestSection {
            log: Sampling::All(false),
            every_batches: 20,
        }
    }
}

/// §1.5.5's engine-panic recovery. What a crashed game costs and what it leaves behind.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecoverySection {
    /// Crash dumps kept, per run. Zero turns dumping off without turning *recovery* off — the two
    /// are separate decisions, and a run that only wants the counter should not have to choose
    /// between a full disk and a fatal panic.
    pub keep_dumps: usize,
    /// Engine panics tolerated within one collection before the run stops. The guard is against a
    /// *reproducible* crash, which would otherwise spin the collector forever without finishing an
    /// episode; ordinary attrition never comes near it.
    pub max_per_batch: usize,
}

impl Default for RecoverySection {
    fn default() -> Self {
        RecoverySection {
            keep_dumps: 32,
            max_per_batch: 64,
        }
    }
}

/// The two measurements of §1.5.5's `eval/` — the rollout fold and the held-out harness.
///
/// They are one section because they are one question asked twice, but they cost differently:
/// `window_batches` is free (the games are collected anyway), `every_batches` buys games the run
/// would not otherwise play. `window_batches`/`envs`/`max_crashes` stay run-global mechanics; a
/// curriculum stage's own `advance`/`games_per_opponent` (§1.5.4) replace `trigger`/
/// `games_per_opponent` per stage, and its heuristic anchors replace `opponents`.
/// §1.5.2's pool, as the `.toml` spells it.
///
/// `enabled = false` is not the same as an empty pool: it leaves `[rollout] opponents` in charge and
/// costs nothing, which is what every run before this section existed did. Turning it on changes the
/// opponent *and* the cost — a model on the far seat roughly doubles the forwards per game — so it
/// is deliberately not the default.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoolSection {
    pub enabled: bool,
    /// Slots held by the hardest clones, and by clones drawn back out of the archive.
    ///
    /// History-heavy on purpose: the best slots converge on the newest clones, so an even split
    /// makes half the panel the last few versions of the learner — the plain self-play the history
    /// slots exist to break up.
    pub best_slots: usize,
    pub history_slots: usize,
    /// Batches per slot refresh, which is also the Glicko rating period: the eviction and the
    /// measurement it reads have to close on the same boundary.
    pub refresh_every: u64,
    /// Batches per clone. Below `refresh_every`, or the pool cannot fill.
    pub clone_every: u64,
    pub grace_games: u64,
    pub grace_batches: u64,
    pub grace_share: f64,
    pub uniform_floor: f64,
    pub pfsp_sharpness: f64,
    /// `"uniform"`, `"log_age"`, or `{ kind = "recent", max_age = 500 }`.
    pub history_draw: crate::rl::train::pool::HistoryDraw,
    /// Distinct opponents in flight per collection. **Not** a throughput knob: an env holds one
    /// pending decision at a time and the actor alternates, so a model on the far seat halves the
    /// learner's own batch whatever this is set to (raise `[rollout] envs` for that). What it sets
    /// is how many opponents PFSP samples per collection, and so how fast the ratings fill in.
    pub concurrent_opponents: usize,
    /// Heuristics that sit in the mix for the run's life, as CLI player codes.
    pub anchors: Vec<String>,
    /// Which of `anchors` is the rating scale's origin — never updated, so the elo curve has a fixed
    /// point to be read against. `er` and not `r`: an origin on a very weak player pushes the whole
    /// population to +600 and wastes the resolution where the curve is actually read.
    pub pinned: String,
    /// Baked models from `models/<name>/` that join the permanent panel (§1.5.2's curriculum-owned
    /// references).
    pub baked: Vec<String>,
    /// Where the baked models live.
    pub models_root: String,
    /// Glicko-2's `τ`, and the bounds on a rating deviation.
    pub tau: f64,
    pub min_deviation: f64,
    pub max_deviation: f64,
}

impl Default for PoolSection {
    fn default() -> Self {
        let pool = crate::rl::train::pool::PoolConfig::default();
        let rating = crate::rl::train::rating::RatingConfig::default();
        PoolSection {
            enabled: false,
            best_slots: pool.best_slots,
            history_slots: pool.history_slots,
            refresh_every: pool.refresh_every,
            clone_every: pool.clone_every,
            grace_games: pool.grace_games,
            grace_batches: pool.grace_batches,
            grace_share: pool.grace_share,
            uniform_floor: pool.uniform_floor,
            pfsp_sharpness: pool.pfsp_sharpness,
            history_draw: pool.history_draw,
            concurrent_opponents: 1,
            anchors: vec!["er".to_string(), "w".to_string()],
            pinned: "er".to_string(),
            baked: Vec::new(),
            models_root: "models".to_string(),
            tau: rating.tau,
            min_deviation: rating.min_deviation,
            max_deviation: rating.max_deviation,
        }
    }
}

impl PoolSection {
    pub fn pool_config(&self) -> Result<crate::rl::train::pool::PoolConfig, String> {
        let config = crate::rl::train::pool::PoolConfig {
            best_slots: self.best_slots,
            history_slots: self.history_slots,
            refresh_every: self.refresh_every,
            clone_every: self.clone_every,
            grace_games: self.grace_games,
            grace_batches: self.grace_batches,
            grace_share: self.grace_share,
            uniform_floor: self.uniform_floor,
            pfsp_sharpness: self.pfsp_sharpness,
            history_draw: self.history_draw,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn rating_config(&self) -> Result<crate::rl::train::rating::RatingConfig, String> {
        let config = crate::rl::train::rating::RatingConfig {
            tau: self.tau,
            min_deviation: self.min_deviation,
            max_deviation: self.max_deviation,
        };
        config.validate()?;
        Ok(config)
    }

    /// The permanent members: the heuristic anchors plus the baked models, with exactly one pin.
    ///
    /// The pin is resolved by *name* against the anchors rather than taken on faith, because a
    /// `pinned` naming an anchor that is not in the mix would leave the scale with no origin at all
    /// — and the symptom of that is a slowly drifting elo curve, which reads as a result.
    pub fn permanent(&self) -> Result<Vec<crate::rl::train::pool::Permanent>, String> {
        if self.concurrent_opponents == 0 {
            return Err("[pool] concurrent_opponents must be > 0".to_string());
        }
        resolve_permanent(&self.anchors, &self.baked, &self.pinned)
    }
}

/// Shared by [`PoolSection::permanent`] and [`TrainConfig::curriculum_stages`]: anchors + baked
/// names + a pin into a validated [`Permanent`] list. Not a `PoolSection` method because a
/// curriculum stage has its own `anchors`/`baked` but never its own `pinned` — §1.5.4's rating
/// scale origin is a run-global setting a stage cannot move, so every caller resolves against
/// `[pool].pinned`.
fn resolve_permanent(
    anchors: &[String],
    baked: &[String],
    pinned: &str,
) -> Result<Vec<crate::rl::train::pool::Permanent>, String> {
    use crate::rl::train::pool::Permanent;

    if anchors.is_empty() && baked.is_empty() {
        return Err(
            "needs at least one anchor or baked model: an empty permanent panel has nothing to \
             play before the first clone, and no origin for the rating scale"
                .to_string(),
        );
    }
    let pin =
        crate::players::parse_player_code(pinned).map_err(|err| format!("[pool] pinned: {err}"))?;

    let mut members = Vec::new();
    let mut found_pin = false;
    for code in anchors {
        let parsed =
            crate::players::parse_player_code(code).map_err(|err| format!("anchors: {err}"))?;
        let member = Permanent::heuristic(parsed.clone());
        if parsed == pin {
            found_pin = true;
            members.push(member.pinned());
        } else {
            members.push(member);
        }
    }
    if !found_pin {
        return Err(format!(
            "pinned = \"{pinned}\" is not in anchors = {anchors:?}: the rating origin has to \
             play, or it anchors nothing and the elo curve drifts without saying so"
        ));
    }
    for name in baked {
        members.push(Permanent::baked(name.clone()));
    }
    Ok(members)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvalSection {
    /// Batches folded into the per-opponent rolling winrate. The precision dial of the continuous
    /// curve, and its lag: longer averages more games (≈ ±2 % at 20 batches of a 64-env rollout)
    /// but over more versions of the model.
    pub window_batches: usize,
    /// When the held-out evaluation runs. An integer is a batch cadence (`0` = never); a table is a
    /// floor screened off the rolling window. See [`EvalTrigger`].
    pub trigger: EvalTrigger,
    /// Games against **each** held-out opponent — the denominator of every interval it reports.
    pub games_per_opponent: usize,
    /// Absent means `[rollout] envs`. Only worth setting apart to evaluate at a different batch
    /// size than the rollout collects at.
    pub envs: Option<usize>,
    /// Absent means the `[rollout]` panel. The point of setting it is anchors the run does **not**
    /// train on: §1.5.2 warns that training against the panel makes winrate-vs-panel a saturation
    /// signal, so an opponent inside the mix measures nothing the fold does not measure better.
    pub opponents: Option<Vec<String>>,
    /// Engine panics tolerated across one whole evaluation. Lower than `[recovery] max_per_batch`
    /// on purpose: a rollout trades broken games for progress, an evaluation trades them for a
    /// denominator that no longer describes the panel.
    pub max_crashes: usize,
}

impl Default for EvalSection {
    fn default() -> Self {
        EvalSection {
            window_batches: 20,
            trigger: EvalTrigger::Cadence(0),
            games_per_opponent: 100,
            envs: None,
            opponents: None,
            max_crashes: 8,
        }
    }
}

/// When the held-out evaluation runs: a batch cadence, or a floor on the rolling window.
///
/// Untagged, so the TOML *type* decides — `trigger = 20` and a `[eval.trigger]` table with a
/// `winrate` are both this field, like [`Coefficient`]. Full reasoning for the floor design (why a
/// floor beats a cadence, why the screen is not circular, the repeated-testing risk
/// [`FloorSpec::hold`]/[`FloorSpec::cooldown`] guard against): NOTES.md, "Mesure du winrate".
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum EvalTrigger {
    /// Every N batches; `0` never. `1` is the per-batch non-regression mode, which is the same
    /// mechanism rather than a second one.
    Cadence(u64),
    Floor(FloorSpec),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloorSpec {
    /// The floor, on the **minimum** winrate over the anchors — §1.5.4's "winrate vs the current
    /// anchor" read conservatively. The mean would let a 95 % against the random player pay for a
    /// 45 % against the weighted one, which is the averaging the per-anchor split exists to undo.
    pub winrate: f64,
    /// Consecutive batches the floor must hold, over a **full** window, before the evaluation fires.
    /// One batch touching the threshold is a fluctuation; a run of them is a level.
    #[serde(default = "default_hold")]
    pub hold: usize,
    /// Minimum batches between two held-out evaluations. Without it a window oscillating around the
    /// floor re-tests every batch, and enough tests eventually pass on noise alone.
    #[serde(default = "default_cooldown")]
    pub cooldown: u64,
}

fn default_hold() -> usize {
    20
}

fn default_cooldown() -> u64 {
    20
}

/// §1.5.4 — an ordered curriculum of stages, each owning its own deck DB/archetype subset,
/// opponent set and magnet reseed, advanced through on the shared eval harness's floor-confirm
/// mechanism (`EvalGate`/[`FloorSpec`]), plus the run's global stop-by-plateau knobs.
///
/// **Empty `stages` (the default) leaves every pre-§1.5.4 code path untouched.** A non-empty list
/// *supersedes* the run's flat `[decks]`/`[eval]`/`[pool] anchors,baked`/`[magnet.seed]` fields
/// for the run rather than merging with them — an ambiguous "flat fields are stage 0" merge is not
/// attempted; those sections simply go unread once `curriculum.stages` is non-empty (mechanics
/// that stay run-global regardless — `[pool] best_slots`, `tau`, `models_root`, … — are the
/// exception, since §1.5.4's stage triple names *membership*, not mechanics).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurriculumSection {
    pub stages: Vec<StageSection>,
    /// Consecutive `panel/window` `winrate_mean` readings — one per window turnover, off the free
    /// training-rollout fold, **not** the held-out evaluation — that must sit within
    /// `plateau_epsilon` of each other before the whole run stops (Part 1's global stop, alongside
    /// the step budget). Deliberately independent of the advance rule's held-out harness: that one
    /// only ever runs once the free window has already cleared the stage's floor, so a stage stuck
    /// *below* its floor would never trigger it — the free window needs no floor cleared to be
    /// read. Reset at every stage transition — a transition is a deliberate level shift, not
    /// stagnation, and comparing means across two different opponent sets would not be a
    /// meaningful Δ. See `curriculum.rs`'s module docs for the full reasoning.
    pub plateau_k: usize,
    pub plateau_epsilon: f64,
}

impl Default for CurriculumSection {
    fn default() -> Self {
        CurriculumSection {
            stages: Vec::new(),
            plateau_k: 5,
            plateau_epsilon: 0.02,
        }
    }
}

/// One `[[curriculum.stages]]` table.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSection {
    /// For logging only — printed and recorded on a transition, never parsed back.
    pub name: String,
    #[serde(default)]
    pub db: String,
    #[serde(default)]
    pub archetypes: Vec<String>,
    /// Several DBs at once (§1.5.3) — the alternative to `db`. A stage mixing `meta` with
    /// `tutorial` is how the run gets asymmetric matchups: `meta` alone is ~60 effective
    /// archetypes all of which are tournament-viable, so both seats are always about as strong.
    #[serde(default)]
    pub mix: Vec<MixSection>,
    #[serde(default)]
    pub pure_mirror: f64,
    #[serde(default)]
    pub mirror: f64,
    /// The pool's permanent membership for this stage, when `[pool].enabled`. `[pool].pinned`
    /// must be among them — the rating scale's origin cannot move between stages (see
    /// `Pool::retarget`'s doc); [`resolve_permanent`] enforces this the same way it does for
    /// `[pool]` itself.
    #[serde(default)]
    pub anchors: Vec<String>,
    #[serde(default)]
    pub baked: Vec<String>,
    /// The scripted panel for this stage, when `[pool]` is disabled.
    #[serde(default)]
    pub opponents: Vec<String>,
    /// §1.5.4's 70%-vs-the-worst-anchor advance rule, screened by the free rolling window and
    /// confirmed by a held-out evaluation — reuses [`FloorSpec`] wholesale.
    #[serde(default = "default_advance")]
    pub advance: FloorSpec,
    #[serde(default = "default_stage_games_per_opponent")]
    pub games_per_opponent: usize,
    /// §1.1.3's heuristic seed, replayed into the magnet's reservoir at this stage's transition —
    /// only into whatever capacity `evict_fraction` freed up, never a full reset.
    #[serde(default)]
    pub magnet_seed: Option<MagnetSeedSection>,
    /// Fraction of the magnet's reservoir evicted before the reseed, read from the stage being
    /// *entered* — and inert without a `magnet_seed` on that stage, since eviction exists to free
    /// capacity for the reseed. The landscape shifts a lot between stages, but a full reset would
    /// throw away real signal — this is why the reseed is partial rather than a clear.
    #[serde(default = "default_evict_fraction")]
    pub evict_fraction: f64,
    /// A new harvest rate from this stage on. Absent keeps whatever rate is already in force —
    /// the previous stage's, or `[harvest]`'s if no stage ever spoke — rather than reverting; see
    /// [`super::curriculum::Curriculum::harvest_sampling`].
    #[serde(default)]
    pub harvest_log: Option<super::harvest::Sampling>,
}

fn default_advance() -> FloorSpec {
    FloorSpec {
        winrate: 0.70,
        hold: default_hold(),
        cooldown: default_cooldown(),
    }
}

fn default_stage_games_per_opponent() -> usize {
    100
}

fn default_evict_fraction() -> f64 {
    0.3
}

impl TrainConfig {
    /// Resolves `[[curriculum.stages]]` into typed stages, validating everything checkable before
    /// a game is played:
    /// - every stage's `anchors` include `[pool].pinned` when `[pool]` is enabled — enforced by
    ///   [`resolve_permanent`], the same check `[pool]` itself is held to, and what guarantees the
    ///   permanent list is never without a heuristic member either (the pin always resolves to
    ///   one);
    /// - the `.toml` needs at least one `opponents` entry when `[pool]` is disabled — which is
    ///   also what guarantees a non-empty heuristic panel there;
    /// - every stage's `baked` names resolve off `[pool] models_root` ([`super::baked::Baked`]'s
    ///   own load, minus the weights reaching a network) — [`super::panel::Panel::retarget`] only
    ///   loads a stage's model at the transition into it, so without this check a misspelled name
    ///   in a late stage kills the run hours in rather than before it starts.
    ///
    /// Both branches end up with at least one heuristic in the mix, which is what
    /// [`super::eval::Evaluator`] needs for the held-out advance-eval (it only plays scripted
    /// seats) — so there is no separate "needs a heuristic anchor" check to write.
    #[cfg(feature = "rl-model")]
    pub fn curriculum_stages(&self) -> Result<Vec<super::curriculum::Stage>, String> {
        use super::curriculum::{Stage, StagePanel};
        use crate::rl::train::rating::OpponentId;

        let mut stages = Vec::with_capacity(self.curriculum.stages.len());
        for section in &self.curriculum.stages {
            let (panel, eval_anchors) = if self.pool.enabled {
                let permanent =
                    resolve_permanent(&section.anchors, &section.baked, &self.pool.pinned)
                        .map_err(|err| {
                            format!("[[curriculum.stages]] {:?}: {err}", section.name)
                        })?;
                for name in &section.baked {
                    super::baked::Baked::load(Path::new(&self.pool.models_root), name).map_err(
                        |err| format!("[[curriculum.stages]] {:?} baked: {err}", section.name),
                    )?;
                }
                let heuristics: Vec<crate::players::PlayerCode> = permanent
                    .iter()
                    .filter_map(|member| match &member.id {
                        OpponentId::Heuristic(code) => Some(code.clone()),
                        _ => None,
                    })
                    .collect();
                (StagePanel::Pool(permanent), heuristics)
            } else {
                let opponents = section
                    .opponents
                    .iter()
                    .map(|code| crate::players::parse_player_code(code))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| {
                        format!("[[curriculum.stages]] {:?} opponents: {err}", section.name)
                    })?;
                if opponents.is_empty() {
                    return Err(format!(
                        "[[curriculum.stages]] {:?}: needs at least one `opponents` entry when \
                         [pool] is disabled",
                        section.name
                    ));
                }
                (StagePanel::Scripted(opponents.clone()), opponents)
            };

            let magnet_seed = section
                .magnet_seed
                .as_ref()
                .map(|seed| -> Result<super::anchor::AnchorConfig, String> {
                    let anchors = seed
                        .anchors
                        .iter()
                        .map(|anchor| {
                            Ok(super::anchor::AnchorShare {
                                player: crate::players::parse_player_code(&anchor.player)?,
                                share: anchor.share,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    Ok(super::anchor::AnchorConfig {
                        anchors,
                        envs: seed.envs,
                        steps: seed.steps,
                        max_crashes: seed.max_crashes,
                    })
                })
                .transpose()
                .map_err(|err| {
                    format!(
                        "[[curriculum.stages]] {:?} magnet_seed: {err}",
                        section.name
                    )
                })?;

            let sources = resolve_sources(&section.db, &section.archetypes, &section.mix)
                .map_err(|err| format!("[[curriculum.stages]] {:?}: {err}", section.name))?;

            stages.push(Stage {
                name: section.name.clone(),
                sources,
                pure_mirror: section.pure_mirror,
                mirror: section.mirror,
                panel,
                eval_anchors,
                advance: section.advance.clone(),
                games_per_opponent: section.games_per_opponent,
                magnet_seed,
                evict_fraction: section.evict_fraction,
                harvest_log: section.harvest_log,
            });
        }
        Ok(stages)
    }
}

/// §1.5.1b — the magnet clone, its reservoir, its SL step, and the `η` of the KL term the
/// best-response pays for it.
///
/// One section for one system, `enabled` included: **MMD without the magnet is policy gradient**,
/// so switching it off is switching algorithms, and that decision belongs in one place rather than
/// spread over the file as an `η = 0` beside a live second network.
///
/// `eta` lives here rather than in `[step]` for the same reason: it is the BR's coupling to this
/// system, and a coefficient whose target does not exist is not a hyperparameter.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MagnetSection {
    pub enabled: bool,
    /// `η`. A number or a schedule table, like every `[step]` coefficient.
    pub eta: Coefficient,
    pub learning_rate: Coefficient,
    pub residual_decay: Coefficient,
    pub grad_clip: f32,
    /// Frames the reservoir holds — **host RAM**, storing observations and masks. One on-policy
    /// batch of `[rollout] frames_per_batch` holds objects of the same kind, so the buffer costs
    /// about `capacity / frames_per_batch` batches' worth of them.
    pub capacity: usize,
    /// Frames the buffer must hold before the SL step runs (see [`super::magnet::MagnetConfig`]).
    pub min_fill: usize,
    /// Frames per SL step.
    pub batch: usize,
    /// Absent means `[step] micro_batch`. A VRAM bound, and the two models are the same shape, so
    /// splitting them apart is only worth it to trade one against the other.
    pub micro_batch: Option<usize>,
    /// The §1.1.3 heuristic seed. Absent means the magnet starts at its own random init.
    pub seed: Option<MagnetSeedSection>,
}

impl Default for MagnetSection {
    fn default() -> Self {
        MagnetSection {
            enabled: false,
            eta: Coefficient::Fixed(0.05),
            learning_rate: Coefficient::Fixed(1.0e-3),
            residual_decay: Coefficient::Fixed(1.0e-4),
            grad_clip: 0.5,
            capacity: 20_000,
            min_fill: 4_000,
            batch: 256,
            micro_batch: None,
            seed: None,
        }
    }
}

/// Seeding the magnet on a weighted mixture of repo heuristics before the loop starts
/// ([`super::anchor`]).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MagnetSeedSection {
    /// The mixture. An array rather than a table keyed by player code, because the order is part of
    /// the run's reproducibility: it is the order the components are drawn against.
    pub anchors: Vec<AnchorShareSection>,
    pub envs: usize,
    /// Cloning steps run against the seeded buffer before batch 0.
    pub steps: usize,
    pub max_crashes: usize,
}

/// One `{ player, share }` of [`MagnetSeedSection::anchors`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorShareSection {
    /// A CLI player code (`r`, `w`, `v`, `e2`, …).
    pub player: String,
    /// Relative weight; the array is normalized, so these need not sum to anything.
    pub share: f64,
}

impl Default for MagnetSeedSection {
    fn default() -> Self {
        MagnetSeedSection {
            anchors: [("w", 0.5), ("v", 0.3), ("e2", 0.2)]
                .into_iter()
                .map(|(player, share)| AnchorShareSection {
                    player: player.to_string(),
                    share,
                })
                .collect(),
            envs: 16,
            steps: 200,
            max_crashes: 8,
        }
    }
}

/// A coefficient: a bare number, or a [`ScheduleSpec`] table.
///
/// Untagged, so the TOML *type* decides — `learning_rate = 3e-4` and a `[step.learning_rate]`
/// table with phases are both this field, and nothing has to name which form was used.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Coefficient {
    Fixed(f64),
    Phased(ScheduleSpec),
}

impl Coefficient {
    fn resolve(&self, batches: u64) -> Result<Schedule, String> {
        match self {
            Coefficient::Fixed(value) => Ok(Schedule::constant(*value)),
            Coefficient::Phased(spec) => Schedule::resolve(spec, batches),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepSection {
    pub learning_rate: Coefficient,
    pub value_coeff: Coefficient,
    pub entropy_coeff: Coefficient,
    pub residual_decay: Coefficient,
    /// Not schedulable — see [`super::update::StepConfig`].
    pub grad_clip: f32,
    pub micro_batch: usize,
    /// Batches between two per-term gradient probes; `0` is off. Defaulted rather than required so
    /// an existing run's `config.toml` still parses — the probe is a diagnostic, and a resume that
    /// refused to load over a missing one would be a worse trade than a default cadence.
    #[serde(default = "default_grad_probe_every")]
    pub grad_probe_every: u64,
    /// Batches between two attention read-outs; `0` is off. Defaulted for the same reason as the
    /// gradient probe above: an existing `config.toml` predates it and must still resume.
    #[serde(default = "default_attn_probe_every")]
    pub attn_probe_every: u64,
}

fn default_grad_probe_every() -> u64 {
    50
}

fn default_attn_probe_every() -> u64 {
    25
}

impl RolloutSection {
    /// Player codes, parsed through the same parser the CLI uses.
    pub fn panel(&self) -> Result<Vec<crate::players::PlayerCode>, String> {
        self.opponents
            .iter()
            .map(|code| crate::players::parse_player_code(code))
            .collect()
    }
}

impl TrainConfig {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let contents = fs::read_to_string(path)
            .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
        let mut config: Self = toml::from_str(&contents)
            .map_err(|err| format!("failed to parse config {}: {err}", path.display()))?;
        config.source = path.to_path_buf();
        Ok(config)
    }

    /// Lays out `runs/<name>/` and clones this config into it (§1.5.5).
    pub fn create_run(&self) -> Result<RunDir, String> {
        RunDir::create(&self.run.root, &self.run.name, &self.source)
    }

    /// Reopens `runs/<name>/`, for resume.
    pub fn open_run(&self) -> Result<RunDir, String> {
        RunDir::open(&self.run.root, &self.run.name)
    }

    /// The frozen text tables every model in the run is built on — the learner's, the magnet's,
    /// the pool clones', and the baked opponents'.
    ///
    /// One table for all of them, resolved once, because the tables are *not* in a checkpoint:
    /// burn records a bare `Tensor` field as an `EmptyRecord` whose `load_record` is a no-op, so
    /// [`crate::rl::model::tables::FrozenTables`] is rebuilt from whatever is passed at
    /// construction. Two models built from different tables read the same weights against
    /// different features, and nothing downstream can tell.
    pub fn text_embeddings(&self) -> Result<crate::rl::text_embedding::TextEmbeddings, String> {
        use crate::rl::text_embedding::TextEmbeddings;
        if self.text_embeddings.is_empty() {
            return Ok(TextEmbeddings::zeros());
        }
        TextEmbeddings::from_json_file(&self.text_embeddings).map_err(|err| {
            format!(
                "{err}\nSet `text_embeddings = \"\"` to train on zeros deliberately (§1.2.9); \
                 build the artifact with auxiliaries/text_embeddings/build_embeddings.py"
            )
        })
    }

    /// Loads the named DBs and wires them to the quotas.
    pub fn deck_sampler(&self) -> Result<DeckSampler, String> {
        let specs = resolve_sources(&self.decks.db, &self.decks.archetypes, &self.decks.mix)
            .map_err(|err| format!("[decks] {err}"))?;
        let mut sources = Vec::with_capacity(specs.len());
        for spec in specs {
            sources.push(DeckSource {
                db: DeckDb::load(&self.decks.root.join(&spec.db))?,
                share: spec.share,
                archetypes: spec.archetypes,
            });
        }
        DeckSampler::mixed(sources, self.decks.pure_mirror, self.decks.mirror)
    }

    /// The §1.5.7 harvest for this run, or `None` when the run does not log labels.
    pub fn harvest(&self, run: &RunDir) -> Result<Option<super::harvest::Harvest>, String> {
        if self.harvest.log.is_off() {
            return Ok(None);
        }
        super::harvest::Harvest::new(&run.harvest(), self.harvest.log).map(Some)
    }

    #[cfg(feature = "rl-model")]
    pub fn rollout_config(&self) -> Result<super::rollout::RolloutConfig, String> {
        Ok(super::rollout::RolloutConfig {
            envs: self.rollout.envs,
            opponents: self.rollout.panel()?,
            max_crashes_per_batch: self.recovery.max_per_batch,
        })
    }

    /// The per-opponent rolling winrate folded off the training rollout. Always on: it costs a fold
    /// over episodes the run already collected.
    #[cfg(feature = "rl-model")]
    pub fn panel_window(&self) -> super::eval::PanelWindow {
        super::eval::PanelWindow::new(self.eval.window_batches)
    }

    /// The gate deciding when the held-out evaluation runs (§1.5.4's floor, or a plain cadence).
    #[cfg(feature = "rl-model")]
    pub fn eval_gate(&self) -> super::eval::EvalGate {
        super::eval::EvalGate::new(self.eval.trigger.clone())
    }

    /// The anchors §1.5.6's evaluation plays *when the `.toml` names them*, and `None` when it
    /// falls back to the training panel.
    ///
    /// The distinction is what makes the §1.5.2 disjointness check meaningful: an evaluation that
    /// defaults to the training panel is already known not to be held out — §1.5.6 says so — and
    /// refusing that configuration would refuse the documented default. Only an *explicit* list
    /// claims to be held out, and only an explicit list can break that claim.
    pub fn held_out_opponents(&self) -> Result<Option<Vec<crate::players::PlayerCode>>, String> {
        match &self.eval.opponents {
            Some(codes) => codes
                .iter()
                .map(|code| crate::players::parse_player_code(code))
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
            None => Ok(None),
        }
    }

    fn eval_opponents(&self) -> Result<Vec<crate::players::PlayerCode>, String> {
        match self.held_out_opponents()? {
            Some(codes) => Ok(codes),
            None => self.rollout.panel(),
        }
    }

    /// The held-out eval harness for this run, or `None` when the trigger never fires.
    ///
    /// The sampler is the caller's — cloned off the collector's, so eval and training cannot end up
    /// on different deck distributions without the `.toml` saying so.
    #[cfg(feature = "rl-model")]
    pub fn evaluator(
        &self,
        sampler: super::sampler::DeckSampler,
    ) -> Result<Option<super::eval::Evaluator>, String> {
        if self.eval.trigger == EvalTrigger::Cadence(0) {
            return Ok(None);
        }
        let opponents = self.eval_opponents()?;
        super::eval::Evaluator::new(
            sampler,
            super::eval::EvalConfig {
                envs: self.eval.envs.unwrap_or(self.rollout.envs),
                games_per_opponent: self.eval.games_per_opponent,
                opponents,
                max_crashes: self.eval.max_crashes,
            },
            self.run.seed,
        )
        .map(Some)
    }

    /// Where this run dumps the games an engine panic cost (§1.5.5).
    pub fn crash_log(&self, run: &RunDir) -> super::crash::CrashLog {
        super::crash::CrashLog::new(&run.crashes(), self.recovery.keep_dumps)
    }

    /// Resolves the `[step]` coefficients against this run's batch count.
    ///
    /// Resolution needs `batches`, which is why it happens here and not in serde: a `"5%"` phase
    /// is meaningless until the run's length is known, and reading it from another section is
    /// exactly the coupling that a `Deserialize` impl cannot express.
    #[cfg(feature = "rl-model")]
    pub fn step_config(&self) -> Result<super::update::StepConfig, String> {
        let batches = self.rollout.batches as u64;
        let resolve = |name: &str, coefficient: &Coefficient| {
            coefficient
                .resolve(batches)
                .map_err(|err| format!("[step] {name}: {err}"))
        };
        Ok(super::update::StepConfig {
            learning_rate: resolve("learning_rate", &self.step.learning_rate)?,
            value_coeff: resolve("value_coeff", &self.step.value_coeff)?,
            entropy_coeff: resolve("entropy_coeff", &self.step.entropy_coeff)?,
            // The magnetic term is a property of the pair, so `η` is resolved only when there is a
            // magnet: a schedule left over from a disabled section must not reach the loss.
            eta: match self.magnet.enabled {
                true => Some(
                    self.magnet
                        .eta
                        .resolve(batches)
                        .map_err(|err| format!("[magnet] eta: {err}"))?,
                ),
                false => None,
            },
            residual_decay: resolve("residual_decay", &self.step.residual_decay)?,
            grad_clip: self.step.grad_clip,
            micro_batch: self.step.micro_batch,
            grad_probe_every: self.step.grad_probe_every,
            attn_probe_every: self.step.attn_probe_every,
        })
    }

    /// §1.5.1b's magnet configuration, or `None` for the best-response-only run.
    #[cfg(feature = "rl-model")]
    pub fn magnet_config(&self) -> Result<Option<super::magnet::MagnetConfig>, String> {
        if !self.magnet.enabled {
            return Ok(None);
        }
        let batches = self.rollout.batches as u64;
        let resolve = |name: &str, coefficient: &Coefficient| {
            coefficient
                .resolve(batches)
                .map_err(|err| format!("[magnet] {name}: {err}"))
        };
        // A zero SL batch is an enabled magnet that never moves, which reads downstream as a
        // magnet that converged: `magnet/*` is emitted, `magnet/loss` sits at 0 and the KL points
        // at the seed forever. Refused rather than clamped — there is no batch size this meant.
        if self.magnet.batch == 0 {
            return Err("[magnet] batch: an enabled magnet needs a non-zero SL batch".to_string());
        }
        Ok(Some(super::magnet::MagnetConfig {
            capacity: self.magnet.capacity,
            min_fill: self.magnet.min_fill.min(self.magnet.capacity),
            batch: self.magnet.batch,
            micro_batch: self.magnet.micro_batch.unwrap_or(self.step.micro_batch),
            learning_rate: resolve("learning_rate", &self.magnet.learning_rate)?,
            residual_decay: resolve("residual_decay", &self.magnet.residual_decay)?,
            grad_clip: self.magnet.grad_clip,
        }))
    }

    /// The §1.1.3 heuristic seed for this run's magnet, or `None` when the magnet is off or starts
    /// at its own init.
    ///
    /// The sampler is the caller's — cloned off the collector's, like the evaluator's, so the seed
    /// cannot draw its decks from a distribution the run does not train on.
    #[cfg(feature = "rl-model")]
    pub fn magnet_seed(
        &self,
        sampler: super::sampler::DeckSampler,
    ) -> Result<Option<super::anchor::AnchorSeed>, String> {
        let (true, Some(seed)) = (self.magnet.enabled, &self.magnet.seed) else {
            return Ok(None);
        };
        let anchors = seed
            .anchors
            .iter()
            .map(|anchor| {
                Ok(super::anchor::AnchorShare {
                    player: crate::players::parse_player_code(&anchor.player)?,
                    share: anchor.share,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        super::anchor::AnchorSeed::new(
            sampler,
            super::anchor::AnchorConfig {
                anchors,
                envs: seed.envs,
                steps: seed.steps,
                max_crashes: seed.max_crashes,
            },
            self.run.seed,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_parses_and_builds_its_sampler() {
        let config = TrainConfig::from_file(Path::new("config/default.toml")).expect("config");
        assert_eq!(config.decks.db, "tutorial");
        assert_eq!(config.decks.archetypes, ["beginner"]);
        let sampler = config.deck_sampler().expect("sampler");
        assert_eq!(sampler.dbs().next().expect("one db").name, "tutorial");
    }

    /// `[recovery]` has to be optional: the configs of runs that started before it existed are
    /// cloned into their run directories, and a resume must not fail to parse its own record.
    #[test]
    fn the_recovery_section_is_optional_and_defaults_to_recovering() {
        let config: TrainConfig = toml::from_str(
            "[run]\nseed = 0\nname = \"t\"\n\
             [decks]\nroot = \"decks\"\ndb = \"tutorial\"\npure_mirror = 0.0\nmirror = 0.0\n\
             [rollout]\nenvs = 1\nopponents = [\"r\"]\nframes_per_batch = 1\nbatches = 1\n\
             [step]\nlearning_rate = 3e-4\nvalue_coeff = 0.5\nentropy_coeff = 0.01\n\
             residual_decay = 1e-4\ngrad_clip = 0.5\nmicro_batch = 8\n",
        )
        .expect("config without [recovery]");

        assert!(config.recovery.keep_dumps > 0);
        assert!(config.recovery.max_per_batch > 0);
    }

    /// The two forms of a coefficient are told apart by the TOML type alone, and both have to
    /// reach the learner — a scalar silently becoming a one-phase schedule, or a schedule silently
    /// collapsing to its `start`, would each look correct in the file and be wrong in the run.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_coefficient_is_a_scalar_or_a_schedule_and_both_resolve() {
        let config = TrainConfig::from_file(Path::new("config/default.toml")).expect("config");
        let step = config.step_config().expect("schedules");
        let batches = config.rollout.batches as u64;

        // The default's learning rate is §1.5.5's warmup: it starts at zero and is not there by
        // the end. A scalar coefficient beside it must not have moved at all.
        assert_eq!(step.learning_rate.at(0), 0.0);
        assert!(step.learning_rate.at(batches - 1) > 0.0);
        assert_eq!(step.learning_rate.span(), batches);
        assert_eq!(step.value_coeff.at(0), step.value_coeff.at(batches - 1));
    }

    /// A percentage is resolved against `[rollout] batches`, so the same schedule text has to
    /// produce different boundaries in a shorter run.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_relative_phase_follows_the_run_length() {
        let base = "[run]\nseed = 0\nname = \"t\"\n\
             [decks]\nroot = \"decks\"\ndb = \"tutorial\"\npure_mirror = 0.0\nmirror = 0.0\n\
             [step]\nlearning_rate = { start = 0.0, phases = [{ over = \"10%\", to = 1.0 }] }\n\
             value_coeff = 0.5\nentropy_coeff = 0.01\nresidual_decay = 1e-4\n\
             grad_clip = 0.5\nmicro_batch = 8\n\
             [rollout]\nenvs = 1\nopponents = [\"r\"]\nframes_per_batch = 1\nbatches = ";

        let short: TrainConfig = toml::from_str(&format!("{base}100")).expect("short");
        let long: TrainConfig = toml::from_str(&format!("{base}5000")).expect("long");

        assert_eq!(short.step_config().expect("short").learning_rate.span(), 10);
        assert_eq!(long.step_config().expect("long").learning_rate.span(), 500);
    }

    /// The §1.4.3 sizes have to come off the file, not off `Default`, or the run clone would
    /// describe a model the run did not use.
    #[cfg(feature = "rl-model")]
    #[test]
    fn the_model_section_overrides_the_1_4_3_defaults() {
        let config: TrainConfig = toml::from_str(
            "[run]\nseed = 0\nname = \"t\"\n\
             [decks]\nroot = \"decks\"\ndb = \"tutorial\"\npure_mirror = 0.0\nmirror = 0.0\n\
             [rollout]\nenvs = 1\nopponents = [\"r\"]\nframes_per_batch = 1\nbatches = 1\n\
             [step]\nlearning_rate = 3e-4\nvalue_coeff = 0.5\nentropy_coeff = 0.01\n\
             residual_decay = 1e-4\ngrad_clip = 0.5\nmicro_batch = 8\n\
             [model]\nd_model = 384\n",
        )
        .expect("config");

        assert_eq!(config.model.d_model, 384);
        assert_eq!(config.model.num_blocks, 2);
    }

    /// `[magnet]` has to be optional and default to **off**: the configs of runs that started before
    /// §1.5.1b existed are cloned into their run directories, and a resume that silently switched
    /// the run from policy gradient to MMD would be a different experiment under the same name.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_config_without_the_magnet_section_resumes_as_a_best_response_only_run() {
        let config: TrainConfig = toml::from_str(
            "[run]\nseed = 0\nname = \"t\"\n\
             [decks]\nroot = \"decks\"\ndb = \"tutorial\"\npure_mirror = 0.0\nmirror = 0.0\n\
             [rollout]\nenvs = 1\nopponents = [\"r\"]\nframes_per_batch = 1\nbatches = 1\n\
             [step]\nlearning_rate = 3e-4\nvalue_coeff = 0.5\nentropy_coeff = 0.01\n\
             residual_decay = 1e-4\ngrad_clip = 0.5\nmicro_batch = 8\n",
        )
        .expect("config without [magnet]");

        assert!(!config.magnet.enabled);
        assert!(config.magnet_config().expect("magnet").is_none());
        // And `η` does not reach the loss: a coefficient whose target does not exist is not one.
        assert!(config.step_config().expect("step").eta.is_none());
    }

    /// The shipped default *is* §1.5.1's algorithm, magnet included, and every piece of it has to
    /// come off the file — a magnet built from `Default` would make the run clone in
    /// `runs/<name>/` describe a magnet the run did not train.
    #[cfg(feature = "rl-model")]
    #[test]
    fn the_default_config_builds_the_magnet_and_its_seed() {
        let config = TrainConfig::from_file(Path::new("config/default.toml")).expect("config");
        let magnet = config.magnet_config().expect("magnet").expect("enabled");

        assert!(
            magnet.min_fill <= magnet.capacity,
            "an unreachable fill floor"
        );
        // Absent in the file, so it follows `[step]` rather than the Rust default.
        assert_eq!(magnet.micro_batch, config.step.micro_batch);
        assert!(config.step_config().expect("step").eta.is_some());

        let seed = config
            .magnet_seed(config.deck_sampler().expect("sampler"))
            .expect("seed")
            .expect("the default seeds from the anchor mixture");
        // A mixture, not a single heuristic: the magnet is an average policy, and a lone search
        // player's clone is a near-pure strategy for the KL to pull the best-response onto.
        assert!(
            seed.config().anchors.len() > 1,
            "the default seed collapsed to one anchor"
        );
        assert!(seed
            .config()
            .anchors
            .iter()
            .any(|anchor| anchor.player == crate::players::PlayerCode::W));
    }

    /// An enabled magnet that cannot take an SL step is refused at load rather than logged as one
    /// that never learned.
    #[cfg(feature = "rl-model")]
    #[test]
    fn an_enabled_magnet_with_no_sl_batch_is_rejected() {
        let mut config = TrainConfig::from_file(Path::new("config/default.toml")).expect("config");
        config.magnet.batch = 0;
        assert!(config.magnet_config().is_err());
    }

    /// The minimal set of sections every config in this file needs, matching the pattern the
    /// other inline-TOML tests already use — callers append `[pool]`/`[[curriculum.stages]]`.
    fn minimal_config_base() -> String {
        "[run]\nseed = 0\nname = \"t\"\n\
         [decks]\nroot = \"decks\"\ndb = \"tutorial\"\npure_mirror = 0.0\nmirror = 0.0\n\
         [rollout]\nenvs = 1\nopponents = [\"r\"]\nframes_per_batch = 1\nbatches = 1\n\
         [step]\nlearning_rate = 3e-4\nvalue_coeff = 0.5\nentropy_coeff = 0.01\n\
         residual_decay = 1e-4\ngrad_clip = 0.5\nmicro_batch = 8\n"
            .to_string()
    }

    /// `[curriculum]` has to be optional and default to empty: the configs of runs that started
    /// before §1.5.4 existed are cloned into their run directories, and a resume must not fail to
    /// parse its own record, nor silently start behaving like a curriculum run.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_config_without_curriculum_stages_behaves_exactly_as_before() {
        let config: TrainConfig = toml::from_str(&minimal_config_base()).expect("config");

        assert!(config.curriculum.stages.is_empty());
        assert!(config
            .curriculum_stages()
            .expect("empty is fine")
            .is_empty());
    }

    /// §1.5.2's rating scale origin cannot move between stages (`Pool::retarget`'s doc), so a
    /// stage that would drop it is refused at load — long before any game plays it out.
    #[cfg(feature = "rl-model")]
    #[test]
    fn every_stage_must_carry_the_run_global_pinned_anchor() {
        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\", \"r\"]\npinned = \"er\"\n\
             [[curriculum.stages]]\nname = \"drops-the-pin\"\ndb = \"tutorial\"\n\
             archetypes = [\"beginner\"]\nanchors = [\"w\"]\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");

        let err = config
            .curriculum_stages()
            .expect_err("a stage without the pinned anchor must be refused");
        assert!(err.contains("drops-the-pin"), "{err}");
        assert!(err.contains("pinned"), "{err}");
    }

    /// A stage's scripted panel (`[pool].enabled = false`) needs at least one opponent — an empty
    /// one has nothing to play, and no anchor for its held-out advance-eval.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_scripted_stage_needs_at_least_one_opponent() {
        let toml = format!(
            "{}\n\
             [[curriculum.stages]]\nname = \"empty-panel\"\ndb = \"tutorial\"\n\
             archetypes = [\"beginner\"]\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");

        let err = config
            .curriculum_stages()
            .expect_err("a stage with no opponents must be refused");
        assert!(err.contains("empty-panel"), "{err}");
    }

    /// A stage's `.toml` shape mirrors `[pool]`'s own anchor/baked/pinned resolution, and reuses
    /// [`FloorSpec`]'s 70 %-default and `hold`/`cooldown` defaults wholesale.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_stage_resolves_its_advance_floor_and_magnet_seed_defaults() {
        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\"]\npinned = \"er\"\n\
             [[curriculum.stages]]\nname = \"beginner\"\ndb = \"tutorial\"\n\
             archetypes = [\"beginner\"]\nanchors = [\"er\"]\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");
        let stages = config.curriculum_stages().expect("stages");

        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].advance.winrate, 0.70);
        assert_eq!(stages[0].games_per_opponent, 100);
        assert_eq!(stages[0].evict_fraction, 0.3);
        assert_eq!(stages[0].eval_anchors, vec![crate::players::PlayerCode::ER]);
        assert!(
            stages[0].magnet_seed.is_none(),
            "no [curriculum.stages.magnet_seed] table was given"
        );
    }

    /// Every shipped config resolves the text tables it names.
    ///
    /// The check `long_v3` did not have. Its `.toml` was correct and its loop still trained on
    /// zeros, because the artifact was never *asked* for; now that it is, what remains to go wrong
    /// is a path that does not resolve — silent for as long as nobody starts that config, and worth
    /// hours when they do. A run that means to train on zeros says so with `""`, which this admits
    /// and the loop prints.
    #[test]
    fn every_shipped_config_resolves_its_text_embeddings() {
        let configs: Vec<PathBuf> = fs::read_dir("config")
            .expect("config dir")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
            .collect();
        assert!(!configs.is_empty(), "no shipped configs to check");

        for path in configs {
            let config = TrainConfig::from_file(&path)
                .unwrap_or_else(|err| panic!("{} does not parse: {err}", path.display()));
            let table = config.text_embeddings().unwrap_or_else(|err| {
                panic!("{} names unusable text tables: {err}", path.display())
            });
            assert_eq!(
                table.is_empty(),
                config.text_embeddings.is_empty(),
                "{} disagrees with itself on whether it has text",
                path.display()
            );
        }
    }

    /// The two-stage shape the long runs are built on, against the DBs this repository ships.
    /// The shares are the difference between a fifth of the games being asymmetric and none of
    /// them: the source is rolled per seat, so 0.90/0.10 is 2·p·q = 18 % cross-DB, not 10 %.
    /// Builds the sampler rather than only resolving the names, since a share or an archetype the
    /// DBs cannot honour is otherwise only discovered at the transition — hours into a run.
    ///
    /// Written inline rather than read off a run's `.toml`: the shipped configs are `default.toml`
    /// alone, and a test that pins the *stage shape* should not also pin one run's tuning.
    #[cfg(feature = "rl-model")]
    #[test]
    fn the_long_run_shape_resolves_and_builds_its_mixed_stage() {
        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\"]\npinned = \"er\"\n\
             [[curriculum.stages]]\nname = \"warmup\"\ndb = \"tutorial\"\n\
             pure_mirror = 0.05\nmirror = 0.25\nanchors = [\"er\", \"w\", \"aa\", \"v\"]\n\
             games_per_opponent = 600\n\
             [curriculum.stages.advance]\nwinrate = 0.35\nhold = 25\ncooldown = 50\n\
             [[curriculum.stages]]\nname = \"meta\"\n\
             mix = [\n\
             {{ db = \"meta\", share = 0.90 }},\n\
             {{ db = \"tutorial\", share = 0.10 }},\n\
             ]\n\
             pure_mirror = 0.05\nmirror = 0.10\nanchors = [\"er\"]\n\
             games_per_opponent = 600\nharvest_log = 0.33\n\
             [curriculum.stages.advance]\nwinrate = 0.75\nhold = 25\ncooldown = 100\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");
        let stages = config.curriculum_stages().expect("stages");

        assert_eq!(
            stages[1].eval_anchors,
            vec![crate::players::PlayerCode::ER],
            "the mixed stage screens on the pin alone — `w`/`v` saturated at 0.84–0.94 in long_v3 \
             and a saturated anchor cannot move a `min`"
        );

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].sources.len(), 1, "the warmup is one DB");
        assert_eq!(stages[0].advance.winrate, 0.35);

        let meta = &stages[1];
        assert_eq!(meta.sources.len(), 2);
        assert_eq!(meta.sources[0].db, "meta");
        assert_eq!(meta.sources[1].db, "tutorial");
        // The exact split is a judgement call that moves with the DB; what must not silently
        // change is which way round it is, since the shares are per seat and swapping them turns
        // a mostly-`meta` run into a mostly-`tutorial` one.
        assert!(meta.sources[0].share > meta.sources[1].share * 4.0);

        // Every floor in this file is a demand on the pinned `er` and nothing else, which only
        // holds if `er` is in fact among the anchors the screen and the verdict read.
        for stage in &stages {
            assert!(stage.eval_anchors.contains(&crate::players::PlayerCode::ER));
            assert!(
                stage.games_per_opponent >= 550,
                "{}: the confirming eval is sized on the floor's SE",
                stage.name
            );
        }

        let sources = meta
            .sources
            .iter()
            .map(|spec| super::super::sampler::DeckSource {
                db: DeckDb::load(&config.decks.root.join(&spec.db)).expect("db"),
                share: spec.share,
                archetypes: spec.archetypes.clone(),
            })
            .collect();
        let sampler = DeckSampler::mixed(sources, meta.pure_mirror, meta.mirror).expect("sampler");
        // A floor, not a count: the `meta` DB is rebuilt periodically and only grows.
        assert!(sampler.deck_count() > 10_000);
    }

    /// §1.5.3's mix: several DBs on one stage, each with its own share and archetype slice.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_stage_resolves_a_mix_of_several_dbs() {
        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\"]\npinned = \"er\"\n\
             [[curriculum.stages]]\nname = \"meta\"\nanchors = [\"er\"]\n\
             mix = [\n\
             {{ db = \"meta\", share = 0.85 }},\n\
             {{ db = \"tutorial\", share = 0.15, archetypes = [\"beginner\"] }},\n\
             ]\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");
        let stages = config.curriculum_stages().expect("stages");

        assert_eq!(stages[0].sources.len(), 2);
        assert_eq!(stages[0].sources[0].db, "meta");
        assert_eq!(stages[0].sources[0].share, 0.85);
        assert!(stages[0].sources[0].archetypes.is_empty());
        assert_eq!(stages[0].sources[1].db, "tutorial");
        assert_eq!(stages[0].sources[1].archetypes, ["beginner"]);
    }

    /// The `db` shorthand and `mix` describe the same draw, so a file giving both is ambiguous and
    /// a file giving neither says nothing — both are refused rather than resolved to a guess.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_stage_naming_both_db_and_mix_or_neither_is_refused() {
        let stage = |body: &str| {
            let toml = format!(
                "{}\n\
                 [pool]\nenabled = true\nanchors = [\"er\"]\npinned = \"er\"\n\
                 [[curriculum.stages]]\nname = \"ambiguous\"\nanchors = [\"er\"]\n{body}",
                minimal_config_base()
            );
            let config: TrainConfig = toml::from_str(&toml).expect("config");
            config.curriculum_stages().expect_err("must be refused")
        };

        assert!(
            stage("db = \"meta\"\nmix = [{ db = \"tutorial\", share = 1.0 }]\n")
                .contains("alternatives"),
        );
        assert!(stage("").contains("needs either"));
        // Beside a mix this would be dropped without a word, and the stage would silently draw
        // every archetype of every source.
        assert!(
            stage("archetypes = [\"beginner\"]\nmix = [{ db = \"tutorial\", share = 1.0 }]\n")
                .contains("inside each mix entry"),
        );
        assert!(
            stage("mix = [{ db = \"tutorial\", share = 0.0 }]\n").contains("above zero"),
            "a zero share reads as \"this DB is in the mix\" and would not be"
        );
    }

    /// `deny_unknown_fields`, witnessed: a misspelled key refuses the config at load, naming the
    /// key — the alternative is a field silently falling back to its default, a run whose `.toml`
    /// no longer describes it.
    #[test]
    fn a_misspelled_field_is_refused_at_parse_rather_than_silently_defaulted() {
        let toml = format!("{}[recovery]\nkeep_dump = 3\n", minimal_config_base());
        let err = toml::from_str::<TrainConfig>(&toml).expect_err("unknown field");
        assert!(err.to_string().contains("keep_dump"), "{err}");
    }

    /// A stage's baked model is only loaded at the transition into that stage
    /// (`Panel::retarget`), so a name that does not resolve on disk has to be refused at config
    /// load — hours before the transition that would otherwise kill the run.
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_stage_naming_a_missing_baked_model_is_refused_at_load() {
        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\"]\npinned = \"er\"\n\
             [[curriculum.stages]]\nname = \"late\"\ndb = \"tutorial\"\n\
             archetypes = [\"beginner\"]\nanchors = [\"er\"]\nbaked = [\"does_not_exist\"]\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");

        let err = config
            .curriculum_stages()
            .expect_err("a baked name that does not resolve on disk must be refused");
        assert!(err.contains("late"), "{err}");
        assert!(err.contains("does_not_exist"), "{err}");
    }

    /// A config naming a bake from before the text encoder was plugged in is refused at load.
    ///
    /// The assertion worth keeping from the runs this repository no longer ships: plugging the
    /// encoder in changed what 128 dimensions of every effect descriptor *mean* at unchanged
    /// width, so a bake from before it reads its weights against features it never saw.
    /// [`crate::rl::OBS_SCHEMA_VERSION`] is the only thing standing in the way —
    /// `schema_fingerprint` sees widths, and the frozen tables are rebuilt at construction rather
    /// than restored from a record, so nothing else in the load path can notice.
    ///
    /// The stale bake is written here rather than read off `models/`, which ships empty: a
    /// `meta.toml` alone is enough, because the schema is checked before the weights are opened
    /// (`super::baked`'s `a_stale_schema_is_refused_with_both_halves_named` covers that check
    /// itself — this one covers the config load path reaching it).
    #[cfg(feature = "rl-model")]
    #[test]
    fn a_config_naming_a_stale_bake_is_refused() {
        let root = std::env::temp_dir().join("deckgym-config-stale-bake");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("ancient")).expect("scratch");
        let mut meta =
            crate::rl::train::BakedMeta::current(crate::rl::model::config::ModelConfig::default());
        meta.schema_fingerprint = "0x0000000000000001".to_string();
        crate::rl::train::Baked::write_meta(&root.join("ancient"), &meta).expect("meta");

        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\"]\npinned = \"er\"\n\
             models_root = {root:?}\n\
             [[curriculum.stages]]\nname = \"late\"\ndb = \"tutorial\"\n\
             anchors = [\"er\"]\nbaked = [\"ancient\"]\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");
        let err = config
            .curriculum_stages()
            .expect_err("a stale bake must not resolve");

        assert!(err.contains("schema mismatch"), "{err}");
        assert!(err.contains("ancient"), "{err}");
        // The message has to name the way out. A refusal that only says "mismatch" sends the reader
        // to the wrong half of the problem — the sizes, which are free to differ.
        assert!(err.contains("re-bake"), "{err}");
    }

    /// The commented-out stages in `config/default.toml` are a real, runnable curriculum — two
    /// stages, the second with its own `magnet_seed`. This is them uncommented, and every piece
    /// has to resolve the same way the loop resolves it.
    #[cfg(feature = "rl-model")]
    #[test]
    fn the_default_files_commented_curriculum_resolves_two_stages() {
        let toml = format!(
            "{}\n\
             [pool]\nenabled = true\nanchors = [\"er\", \"w\"]\npinned = \"er\"\n\
             [[curriculum.stages]]\nname = \"beginner\"\ndb = \"tutorial\"\n\
             archetypes = [\"beginner\"]\npure_mirror = 0.05\nmirror = 0.10\n\
             anchors = [\"er\", \"w\"]\n\
             [curriculum.stages.advance]\nwinrate = 0.70\nhold = 20\ncooldown = 20\n\
             [[curriculum.stages]]\nname = \"intermediate\"\ndb = \"tutorial\"\n\
             archetypes = [\"intermediate\"]\npure_mirror = 0.05\nmirror = 0.10\n\
             anchors = [\"er\", \"w\", \"e2\"]\nevict_fraction = 0.3\n\
             [curriculum.stages.advance]\nwinrate = 0.70\nhold = 20\ncooldown = 20\n\
             [curriculum.stages.magnet_seed]\n\
             anchors = [{{ player = \"w\", share = 0.3 }}, {{ player = \"v\", share = 0.3 }}, \
             {{ player = \"e2\", share = 0.4 }}]\n\
             envs = 16\nsteps = 200\nmax_crashes = 8\n",
            minimal_config_base()
        );
        let config: TrainConfig = toml::from_str(&toml).expect("config");
        let stages = config.curriculum_stages().expect("stages");

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].name, "beginner");
        assert_eq!(stages[1].name, "intermediate");
        assert!(
            stages[0].magnet_seed.is_none(),
            "stage 1's seed is the run-wide [magnet.seed], not a per-stage one"
        );
        assert!(
            stages[1].magnet_seed.is_some(),
            "stage 2 reseeds the reservoir on its own transition"
        );
        for stage in &stages {
            assert!(
                stage.eval_anchors.contains(&crate::players::PlayerCode::ER),
                "{}: [pool].pinned must be in the mix",
                stage.name
            );
        }
    }
}
