//! The §1.5.2 pool, wired — what the training loop actually calls.
//!
//! [`super::pool`] decides membership, [`super::rating`] scores it and [`super::opponent`] gets a
//! frozen network onto the far seat. None of the three knows about the others, which is what made
//! them testable; this is the piece that does, and it owns exactly the couplings they refused:
//!
//! - **which ids are scripted and which are weights** — a heuristic anchor costs no forward, a clone
//!   and a baked model each cost one, and only the config knows which is which;
//! - **where a clone's weights live** — `runs/<name>/pool/b<batch>.mpk`, written when the pool admits
//!   it and never deleted, because the historical slots draw from that archive;
//! - **when the period closes** — the rating period *is* the refresh window, so closing it and
//!   re-deciding the slots is one operation and cannot be called half-way.
//!
//! **The models are reloaded on every refresh, not patched.** [`OpponentModels::clear`] invalidates
//! every `AgentId` it handed out, so the assignment has to be rebuilt in the same breath — which is
//! why [`Panel::refresh`] does both and there is no way to do one without the other. A stale
//! assignment pointing into a re-numbered model table would silently play the wrong checkpoint under
//! the right name, and the winrate would be attributed to whoever the id now belongs to.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use rand::Rng;

use crate::players::PlayerCode;
use crate::rl::model::config::ModelConfig;
use crate::rl::model::RlModel;
use crate::rl::text_embedding::TextEmbeddings;

use super::baked::{self, Baked};
use super::checkpoint::{load_cold, save_cold};
use super::config::PoolSection;
use super::opponent::{Assignment, OpponentModels, OpponentSeat};
use super::pool::{Permanent, Pool, PoolRow};
use super::rating::{score_from_reward, OpponentId, RatingTable};
use super::rollout::Episode;

/// A clone's weights inside a run's `pool/`, without the `.mpk` the recorder appends.
///
/// Free-standing because [`super::init`] copies these files between runs without owning a
/// [`Panel`], and a second spelling of the name would be a second thing to keep in step.
pub fn clone_stem(archive: &Path, batch: u64) -> PathBuf {
    archive.join(format!("b{batch:09}"))
}

/// Everything the loop needs to checkpoint about the pool. The weights are on disk already; this is
/// the bookkeeping that says which of them matter and what they are worth.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PanelState {
    pub pool: Pool,
    pub ratings: RatingTable,
}

pub struct Panel<B: Backend> {
    pool: Pool,
    ratings: RatingTable,
    models: OpponentModels<B>,
    /// The permanent members that are played in-process. Everything not in here and not a heuristic
    /// is weights.
    scripted: HashMap<OpponentId, PlayerCode>,
    /// Baked models by name, validated at construction so a missing one fails at startup rather
    /// than at the first refresh that happens to draw it.
    baked: HashMap<String, Baked>,
    /// `runs/<name>/pool/`.
    archive: PathBuf,
    /// Where a newly-referenced baked model is loaded from on a [`Panel::retarget`] — `Panel::new`
    /// only needs this once, but a stage transition can introduce a name that was not in the
    /// original `.toml`.
    models_root: PathBuf,
    concurrent: usize,
    /// The run's own sizes, used to build a *clone*. A baked model brings its own.
    model_config: ModelConfig,
    /// The learner's own text tables, carried rather than rebuilt.
    ///
    /// The frozen tables are not in a checkpoint — burn records a bare `Tensor` as an `EmptyRecord`
    /// — so an opponent built from a different table reads its weights against different features
    /// and plays nonsense, at full confidence and under its own rating. This field is what keeps
    /// the panel on the learner's projection instead of the zeros it used to assume.
    embeddings: TextEmbeddings,
}

impl<B: Backend> Panel<B> {
    /// Builds the panel from the `.toml` and validates every part of it that can be validated
    /// before a single game is played: the pool's arithmetic, Glicko's parameters, the anchors, the
    /// pin, and the presence and schema of every baked model named.
    pub fn new(
        section: &PoolSection,
        archive: PathBuf,
        model_config: ModelConfig,
        embeddings: TextEmbeddings,
    ) -> Result<Self, String> {
        let permanent = section.permanent()?;
        let pool = Pool::new(section.pool_config()?, permanent.clone())?;
        let mut ratings = RatingTable::new(section.rating_config()?)?;

        let mut scripted = HashMap::new();
        for member in &permanent {
            if let OpponentId::Heuristic(code) = &member.id {
                scripted.insert(member.id.clone(), code.clone());
            }
        }

        let models_root = Path::new(&section.models_root);
        let mut baked = HashMap::new();
        for name in &section.baked {
            let model = Baked::load(models_root, name)?;
            // A stored rating is why `meta.toml` carries one: a reference model keeps what it
            // established across runs. `set` and not `ensure`, or registering the panel first would
            // silently win.
            ratings.set(model.id(), model.entry());
            baked.insert(name.clone(), model);
        }

        pool.register(&mut ratings);
        Ok(Panel {
            pool,
            ratings,
            models: OpponentModels::new(),
            scripted,
            baked,
            archive,
            models_root: models_root.to_path_buf(),
            concurrent: section.concurrent_opponents,
            model_config,
            embeddings,
        })
    }

    /// Reconfigures the permanent membership mid-run — a curriculum stage transition (§1.5.4)
    /// changing which anchors/baked models are in the mix.
    ///
    /// `slots`/`archive`/every existing rating survive: [`Pool::retarget`] only swaps the
    /// permanent list, never rebuilds the pool or the rating table, for the same reason a
    /// checkpoint resume never re-rolls them. A baked model already cached in `self.baked` is
    /// never re-loaded or re-rated here — baked models accumulate across stages and are never
    /// evicted from the cache, so one reintroduced in a later stage keeps whatever it earned the
    /// first time, instead of being clobbered back to `meta.toml`'s stored rating.
    ///
    /// Reloads every active member's weights in the same call, exactly like [`Panel::refresh`]
    /// does — a retarget has the identical shape and risk profile (`OpponentModels::clear()` plus
    /// a full reload), so it reuses the same safety envelope rather than inventing a new one.
    pub fn retarget(
        &mut self,
        permanent: Vec<Permanent>,
        device: &B::Device,
    ) -> Result<(), String> {
        self.adopt(&permanent)?;
        self.pool.retarget(permanent, &mut self.ratings)?;
        self.load(device)
    }

    /// Caches the weights behind every baked id in `permanent` and rebuilds the scripted map from
    /// it, so [`Panel::load`] and [`Panel::assignment`] can resolve every member of the incoming
    /// list.
    ///
    /// Shared by [`Panel::retarget`] and [`Panel::restore`] because a resume faces the same
    /// situation a stage transition does: the permanent list it adopts can name a baked model the
    /// run's `.toml` never mentioned, since §1.5.4 lets a stage introduce one.
    ///
    /// `ensure` and not `set`: `meta.toml`'s rating is a baked model's *starting* value, so a
    /// resume — or a stage that still wants it in the mix — keeps whatever it earned in the run
    /// rather than rolling it back to what it shipped with.
    fn adopt(&mut self, permanent: &[Permanent]) -> Result<(), String> {
        for member in permanent {
            if let OpponentId::Baked(name) = &member.id {
                if !self.baked.contains_key(name) {
                    let model = Baked::load(&self.models_root, name)?;
                    self.ratings.ensure(model.id(), model.entry());
                    self.baked.insert(name.clone(), model);
                }
            }
        }
        self.scripted = permanent
            .iter()
            .filter_map(|member| match &member.id {
                OpponentId::Heuristic(code) => Some((member.id.clone(), code.clone())),
                _ => None,
            })
            .collect();
        Ok(())
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub fn ratings(&self) -> &RatingTable {
        &self.ratings
    }

    /// Where [`Panel::admit`] writes a clone, and [`Panel::load`] reads one. Without the `.mpk` the
    /// recorder appends.
    fn clone_stem(&self, batch: u64) -> PathBuf {
        clone_stem(&self.archive, batch)
    }

    /// Loads the weights of every model-driven member currently in the pool, dropping whatever was
    /// loaded before.
    ///
    /// Every id is re-inserted from scratch rather than diffed against the previous slate: the
    /// saving is one file read per unchanged member per refresh, and the cost of getting a diff
    /// wrong is an assignment pointing at the wrong network — which plays perfectly well and
    /// reports its results under someone else's name.
    pub fn load(&mut self, device: &B::Device) -> Result<(), String> {
        self.models.clear();
        let embeddings = &self.embeddings;
        for id in self.pool.active() {
            match &id {
                OpponentId::Heuristic(_) => {}
                OpponentId::Baked(name) => {
                    let model = self.baked.get(name).ok_or_else(|| {
                        format!("baked model `{name}` is in the pool but was never loaded")
                    })?;
                    let weights = baked::load_model::<B>(model, embeddings, device)?;
                    self.models.insert(id.clone(), weights);
                }
                OpponentId::Pool(batch) => {
                    let fresh = RlModel::<B>::new(&self.model_config, embeddings, device);
                    let weights = load_cold(fresh, &self.clone_stem(*batch), device)
                        .map_err(|err| format!("pool clone {id} could not be read back: {err}"))?;
                    self.models.insert(id.clone(), weights);
                }
            }
        }
        Ok(())
    }

    pub fn models(&self) -> &OpponentModels<B> {
        &self.models
    }

    /// One collection's opponent layout: `concurrent_opponents` PFSP draws, one per env group.
    ///
    /// Drawn **with** replacement. Two groups landing on the same opponent is not waste — the
    /// collector groups its forwards by `AgentId`, so they merge back into one batch — and drawing
    /// without replacement would distort the very weights the draw exists to respect.
    pub fn assignment<R: Rng + ?Sized>(
        &self,
        batch: u64,
        rng: &mut R,
    ) -> Result<Assignment, String> {
        let mut groups = Vec::with_capacity(self.concurrent);
        for _ in 0..self.concurrent {
            let id = self.pool.sample(&self.ratings, batch, rng);
            let seat = match self.scripted.get(&id) {
                Some(code) => OpponentSeat::Scripted(code.clone()),
                None => {
                    let agent = self.models.agent_of(&id).ok_or_else(|| {
                        format!("pool drew {id}, whose weights are not loaded — call Panel::load")
                    })?;
                    OpponentSeat::Model(agent)
                }
            };
            groups.push((id, seat));
        }
        Assignment::grouped(groups)
    }

    /// Folds one collection's games into the open rating period.
    ///
    /// The score is `(reward + 1) / 2`: §1.5.1's terminal reward is `−1/0/+1` and Glicko wants
    /// `0/½/1`, with a tie worth half — which is right here and deliberately not what §1.5.6's
    /// winrate series does with the same game.
    pub fn record(&mut self, episodes: &[Episode]) {
        for episode in episodes {
            self.ratings
                .record(episode.opponent.clone(), score_from_reward(episode.reward));
        }
    }

    /// Writes a clone of the current best-response into the archive and offers it to the pool.
    ///
    /// The weights are written *before* the pool is told, so an id the pool holds is always an id
    /// whose file exists — the reverse order would leave a refresh able to draw a member that was
    /// never saved.
    ///
    /// **And if the pool seats it, its weights are loaded in the same call.** During the fill phase
    /// `admit_clone` takes a free slot immediately, so the very next assignment can draw the new
    /// member — it has to be playable by then. The weights are taken from the model in hand rather
    /// than read back from the file just written, which costs nothing and cannot disagree with what
    /// was saved. Appending an id never renumbers the ones already handed out, so this is safe
    /// mid-run in a way that [`Panel::load`] is not.
    pub fn admit(&mut self, batch: u64, model: &RlModel<B>) -> Result<OpponentId, String> {
        std::fs::create_dir_all(&self.archive)
            .map_err(|err| format!("failed to create {}: {err}", self.archive.display()))?;
        save_cold(model, &self.clone_stem(batch))?;
        let id = self.pool.admit_clone(batch, &mut self.ratings);
        if self.pool.active().contains(&id) {
            self.models.insert(id.clone(), model.clone());
        }
        Ok(id)
    }

    /// Closes the rating period, re-decides the slots, and reloads the weights — in that order, and
    /// only together.
    ///
    /// Together because [`OpponentModels::clear`] invalidates every `AgentId`: a caller that
    /// refreshed the slots without reloading would hold an assignment pointing into a re-numbered
    /// table, and play the wrong checkpoint under the right name.
    pub fn refresh<R: Rng + ?Sized>(
        &mut self,
        batch: u64,
        rng: &mut R,
        device: &B::Device,
    ) -> Result<super::pool::Refresh, String> {
        self.ratings.close_period();
        let refresh = self.pool.refresh(&mut self.ratings, batch, rng);
        self.load(device)?;
        Ok(refresh)
    }

    pub fn should_clone(&self, batch: u64) -> bool {
        self.pool.should_clone(batch)
    }

    pub fn should_refresh(&self, batch: u64) -> bool {
        self.pool.should_refresh(batch)
    }

    /// §1.5.6's per-batch scalars: the elo curve and the pool's shape, per category and never per
    /// member — a series per clone leaves a dead curve behind at every eviction.
    pub fn scalars(&self) -> Vec<(String, f64)> {
        let mut out = self.ratings.scalars();
        out.extend(self.pool.scalars());
        out
    }

    /// The per-member table, for the JSONL beside the metrics.
    pub fn rows(&self, batch: u64) -> Vec<PoolRow> {
        self.pool.rows(&self.ratings, batch)
    }

    /// What §1.5.5's hot checkpoint carries. A resume that re-rolled its slots would face a
    /// different panel than the run it continues, and would lose the ratings of evicted members —
    /// which §1.5.2 keeps so a checkpoint drawn back in resumes rather than restarts.
    pub fn state(&self) -> PanelState {
        PanelState {
            pool: self.pool.clone(),
            ratings: self.ratings.clone(),
        }
    }

    /// Restores a checkpointed panel and reloads its weights.
    ///
    /// The `.toml`'s parameters are re-attached rather than restored from the state, for the reason
    /// §1.5.5 gives about AdamW: a stored parameter could resurrect a value the run's config no
    /// longer asks for.
    ///
    /// The *membership* comes from the checkpoint alone, and a run interrupted in a later
    /// curriculum stage resumes against that stage's panel — which can name baked models and
    /// heuristics the run's `[pool]` section never listed. So the scripted map and the baked cache
    /// are rebuilt from the restored permanent list rather than left as [`Panel::new`] read them
    /// off the `.toml`.
    pub fn restore(
        &mut self,
        state: PanelState,
        section: &PoolSection,
        device: &B::Device,
    ) -> Result<(), String> {
        self.pool = state.pool.with_config(section.pool_config()?)?;
        self.ratings = state.ratings.with_config(section.rating_config()?)?;
        self.adopt(&self.pool.permanent().to_vec())?;
        self.pool.register(&mut self.ratings);
        self.load(device)
    }

    /// The permanent members, for the startup line and for the disjointness check against §1.5.6's
    /// held-out anchors.
    pub fn permanent_ids(&self) -> Vec<OpponentId> {
        let mut ids: Vec<OpponentId> = self.scripted.keys().cloned().collect();
        ids.extend(self.baked.values().map(Baked::id));
        ids.sort();
        ids
    }
}

/// The per-member table, one JSON object per member per record, appended to
/// `runs/<name>/pool/table.jsonl`.
///
/// Separate from `metrics.jsonl` because the two are different shapes: the metrics line is flat
/// scalars, one per batch, and a series per pool member would leave a dead TensorBoard curve behind
/// at every eviction. This is a table — it grows rows, not columns — and is read by asking it
/// questions, not by plotting it.
pub struct PoolLog {
    file: std::fs::File,
}

impl PoolLog {
    /// Opens the log, creating `pool/` if needed. Appended like §1.5.6's metrics, so a run
    /// interrupted three times is still one table.
    pub fn open(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
        let path = dir.join("table.jsonl");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
        Ok(PoolLog { file })
    }

    pub fn record(&mut self, batch: u64, rows: &[PoolRow]) -> Result<(), String> {
        use std::io::Write;
        for row in rows {
            let mut object = serde_json::to_value(row)
                .map_err(|err| format!("failed to encode a pool row: {err}"))?;
            object["batch"] = serde_json::json!(batch);
            writeln!(self.file, "{object}")
                .map_err(|err| format!("failed to write the pool table: {err}"))?;
        }
        Ok(())
    }
}

/// Refuses a config whose held-out evaluation shares an anchor with the training panel.
///
/// §1.5.6 keeps `eval/*` on opponents the run does **not** train against, because training against
/// one makes its winrate a saturation signal rather than generalization. The overlap is silent
/// otherwise: the evaluation still runs, still reports, and measures nothing the rolling window did
/// not already measure better.
pub fn check_eval_disjoint(panel: &[OpponentId], held_out: &[PlayerCode]) -> Result<(), String> {
    let overlap: Vec<String> = held_out
        .iter()
        .filter(|code| panel.contains(&OpponentId::Heuristic((*code).clone())))
        .map(|code| code.to_string())
        .collect();
    if overlap.is_empty() {
        return Ok(());
    }
    Err(format!(
        "[eval] opponents overlap the training panel: {}. §1.5.6 holds the evaluation out precisely \
         so it is not a saturation signal — an anchor inside the mix measures nothing the rolling \
         `panel/window/*` does not measure better and cheaper.",
        overlap.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::train::rating::CENTRE;
    use burn::backend::ndarray::NdArray;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type B = NdArray<f32>;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-panel-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn section() -> PoolSection {
        PoolSection {
            enabled: true,
            best_slots: 1,
            history_slots: 1,
            refresh_every: 4,
            clone_every: 2,
            grace_games: 0,
            grace_batches: 0,
            concurrent_opponents: 1,
            anchors: vec!["er".to_string(), "r".to_string()],
            pinned: "er".to_string(),
            ..Default::default()
        }
    }

    fn model() -> RlModel<B> {
        RlModel::<B>::new(
            &ModelConfig::default(),
            &TextEmbeddings::zeros(),
            &Default::default(),
        )
    }

    /// A baked model minted at *this* build's schema, in its own models root.
    ///
    /// Tests that need a bake mint one rather than naming something in `models/`: a repo artifact
    /// is baked against whatever schema was current the day it was written, so a test standing on
    /// one fails the moment [`crate::rl::OBS_SCHEMA_VERSION`] moves — reporting a schema guard
    /// doing its job as a broken panel.
    fn minted_bake(tag: &str, name: &str) -> PathBuf {
        let root = scratch(tag);
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("bake dir");
        let meta = super::super::baked::BakedMeta::current(ModelConfig::default());
        Baked::write_meta(&dir, &meta).expect("meta");
        save_cold(&model(), &dir.join("weights")).expect("weights");
        root
    }

    /// A panel whose models root holds one freshly minted bake, with the section that found it —
    /// a restore has to rebuild from the same root, and re-minting would wipe the archive.
    fn panel_with_bake(tag: &str, name: &str) -> (Panel<B>, PoolSection) {
        let mut section = section();
        section.models_root = minted_bake(&format!("{tag}-models"), name)
            .display()
            .to_string();
        let panel = Panel::<B>::new(
            &section,
            scratch(tag),
            ModelConfig::default(),
            TextEmbeddings::zeros(),
        )
        .expect("panel");
        (panel, section)
    }

    fn panel(tag: &str) -> Panel<B> {
        Panel::<B>::new(
            &section(),
            scratch(tag),
            ModelConfig::default(),
            TextEmbeddings::zeros(),
        )
        .expect("panel")
    }

    #[test]
    fn a_fresh_panel_is_all_scripted_and_needs_no_weights() {
        let panel = panel("fresh");
        let mut rng = StdRng::seed_from_u64(1);

        let assignment = panel.assignment(0, &mut rng).expect("assignment");
        assert!(assignment.agents().is_empty());
        // Sorted by `OpponentId`'s `Ord`, which follows `PlayerCode`'s declaration order.
        assert_eq!(
            panel.permanent_ids(),
            vec![
                OpponentId::Heuristic(PlayerCode::R),
                OpponentId::Heuristic(PlayerCode::ER),
            ]
        );
    }

    #[test]
    fn the_pin_is_applied_and_never_moves() {
        let mut panel = panel("pin");
        assert_eq!(
            panel.ratings().pinned(),
            Some(&OpponentId::Heuristic(PlayerCode::ER))
        );

        let episodes: Vec<Episode> = (0..50)
            .map(|_| Episode {
                frames: Vec::new(),
                reward: 1.0,
                turns: 10,
                opponent: OpponentId::Heuristic(PlayerCode::ER),
            })
            .collect();
        panel.record(&episodes);
        panel
            .refresh(4, &mut StdRng::seed_from_u64(2), &Default::default())
            .expect("refresh");

        let anchor = panel
            .ratings()
            .get(&OpponentId::Heuristic(PlayerCode::ER))
            .expect("rated");
        assert_eq!(anchor.rating.rating, CENTRE);
        assert!(panel.ratings().learner().rating.rating > CENTRE);
    }

    /// A clone seated during the fill phase is playable **immediately**, with no `load` in
    /// between: the very next batch's assignment can draw it, and a draw it cannot answer stops
    /// the run. The unit test below used to call `load` explicitly and so never saw this; a
    /// twelve-batch end-to-end run found it on the second batch.
    #[test]
    fn an_admitted_clone_is_playable_without_a_reload() {
        let mut panel = panel("admit-playable");
        let id = panel.admit(2, &model()).expect("admit");

        assert!(
            panel.models().agent_of(&id).is_some(),
            "a seated clone must be loaded by `admit` itself"
        );
        // Draw enough times that the new member is certainly among them.
        let mut rng = StdRng::seed_from_u64(9);
        for batch in 0..100 {
            panel.assignment(batch, &mut rng).expect("assignment");
        }
    }

    /// The end-to-end path: a clone is written, the pool takes it, and its weights load back.
    #[test]
    fn a_clone_is_written_admitted_and_reloadable() {
        let mut panel = panel("clone");
        let id = panel.admit(2, &model()).expect("admit");
        assert_eq!(id, OpponentId::Pool(2));

        panel.load(&Default::default()).expect("load");
        assert_eq!(panel.models().len(), 1);
        assert!(panel.models().agent_of(&OpponentId::Pool(2)).is_some());

        // And it is now drawable as a model seat rather than a scripted one.
        let assignment = panel
            .assignment(3, &mut StdRng::seed_from_u64(3))
            .expect("assignment");
        assert_eq!(assignment.concurrent(), 1);
    }

    /// **The invariant every mutating entry point has to keep**: whatever the pool holds, it can
    /// play. A member the assignment cannot answer stops the run, so this walks the three ways the
    /// pool changes — admission, refresh, and restore — and checks the property after each.
    ///
    /// `assignment` still refuses an unloaded member, but that guard is now unreachable through
    /// this API. It is kept as the last line of defence rather than deleted: the alternative to
    /// failing there is a game played by the wrong network and rated as this one.
    #[test]
    fn every_member_the_pool_holds_can_be_played() {
        let mut panel = panel("invariant");
        let mut rng = StdRng::seed_from_u64(4);

        let playable = |panel: &Panel<B>| {
            panel
                .pool()
                .active()
                .into_iter()
                .filter(|id| !matches!(id, OpponentId::Heuristic(_)))
                .all(|id| panel.models().agent_of(&id).is_some())
        };

        for batch in 1..=12u64 {
            if panel.should_clone(batch) {
                panel.admit(batch, &model()).expect("admit");
                assert!(playable(&panel), "after admitting at batch {batch}");
            }
            if panel.should_refresh(batch) {
                panel
                    .refresh(batch, &mut rng, &Default::default())
                    .expect("refresh");
                assert!(playable(&panel), "after refreshing at batch {batch}");
            }
            // What the loop does next, and what caught this: a draw the panel cannot answer is a
            // hard stop, so it has to hold on *every* batch, not just after a refresh.
            panel.assignment(batch, &mut rng).expect("assignment");
        }

        let state = panel.state();
        let mut restored = Panel::<B>::new(
            &section(),
            panel.archive.clone(),
            ModelConfig::default(),
            TextEmbeddings::zeros(),
        )
        .expect("panel");
        restored
            .restore(state, &section(), &Default::default())
            .expect("restore");
        assert!(playable(&restored), "after a restore");
    }

    /// A curriculum stage transition (§1.5.4) can introduce a baked model the run never named at
    /// startup — it has to load and play immediately, and its rating must survive being
    /// retargeted again rather than being reset back to `meta.toml`'s stored value every time a
    /// stage still wants it in the mix.
    #[test]
    fn retarget_loads_a_newly_referenced_baked_model_and_keeps_its_rating_on_a_second_retarget() {
        let (mut panel, _section) = panel_with_bake("retarget-baked", "veteran");
        let baked = OpponentId::Baked("veteran".to_string());

        panel
            .retarget(
                vec![
                    Permanent::heuristic(PlayerCode::ER).pinned(),
                    Permanent::baked("veteran"),
                ],
                &Default::default(),
            )
            .expect("retarget");

        assert!(panel.pool().active().contains(&baked));
        assert!(
            panel.models().agent_of(&baked).is_some(),
            "a newly retargeted baked model must be loaded, not just registered"
        );

        panel.record(&[Episode {
            frames: Vec::new(),
            reward: 1.0,
            turns: 10,
            opponent: baked.clone(),
        }]);
        panel
            .refresh(4, &mut StdRng::seed_from_u64(11), &Default::default())
            .expect("refresh closes the period the game was recorded in");
        let rating_after_games = panel.ratings().get(&baked).expect("rated").rating;

        // Retargeting again with the same baked model in the list must not re-load-and-re-rate it.
        panel
            .retarget(
                vec![
                    Permanent::heuristic(PlayerCode::ER).pinned(),
                    Permanent::baked("veteran"),
                    Permanent::heuristic(PlayerCode::W),
                ],
                &Default::default(),
            )
            .expect("second retarget");

        assert_eq!(
            panel.ratings().get(&baked).expect("still rated").rating,
            rating_after_games,
            "a cached baked model must not be re-loaded and re-rated"
        );
    }

    /// The pool-level test already covers `Pool::retarget`'s own contract; this checks the same
    /// property survives through `Panel`, whose `scripted`/`baked`/`models` bookkeeping sits on
    /// top of it.
    #[test]
    fn retarget_drops_a_member_from_active_but_its_rating_survives_like_an_evicted_clone() {
        let mut panel = panel("retarget-drop");
        let dropped = OpponentId::Heuristic(PlayerCode::R);

        panel.record(&[Episode {
            frames: Vec::new(),
            reward: -1.0,
            turns: 10,
            opponent: dropped.clone(),
        }]);
        panel
            .refresh(4, &mut StdRng::seed_from_u64(12), &Default::default())
            .expect("refresh closes the period the game was recorded in");
        let rating_before = panel.ratings().get(&dropped).expect("rated").rating;

        panel
            .retarget(
                vec![Permanent::heuristic(PlayerCode::ER).pinned()],
                &Default::default(),
            )
            .expect("retarget");

        assert!(!panel.pool().active().contains(&dropped));
        assert!(!panel.permanent_ids().contains(&dropped));
        assert_eq!(
            panel
                .ratings()
                .get(&dropped)
                .expect("rating kept, not deleted")
                .rating,
            rating_before
        );
    }

    #[test]
    fn a_refresh_reloads_the_models_it_renumbered() {
        let mut panel = panel("refresh");
        for batch in [2u64, 4] {
            panel.admit(batch, &model()).expect("admit");
        }
        panel
            .refresh(4, &mut StdRng::seed_from_u64(5), &Default::default())
            .expect("refresh");

        // Every model-driven member of the refreshed pool has weights, and the assignment that
        // follows resolves against them.
        for id in panel.pool().active() {
            if !matches!(id, OpponentId::Heuristic(_)) {
                assert!(panel.models().agent_of(&id).is_some(), "{id} unloaded");
            }
        }
        panel
            .assignment(5, &mut StdRng::seed_from_u64(6))
            .expect("assignment");
    }

    #[test]
    fn the_state_round_trips_and_keeps_evicted_ratings() {
        let mut panel = panel("state");
        for batch in [2u64, 4, 6] {
            panel.admit(batch, &model()).expect("admit");
            panel.record(&[Episode {
                frames: Vec::new(),
                reward: -1.0,
                turns: 10,
                opponent: OpponentId::Pool(batch),
            }]);
        }
        panel
            .refresh(8, &mut StdRng::seed_from_u64(7), &Default::default())
            .expect("refresh");

        let json = serde_json::to_string(&panel.state()).expect("serialize");
        let state: PanelState = serde_json::from_str(&json).expect("deserialize");

        let mut restored = Panel::<B>::new(
            &section(),
            panel.archive.clone(),
            ModelConfig::default(),
            TextEmbeddings::zeros(),
        )
        .expect("panel");
        restored
            .restore(state, &section(), &Default::default())
            .expect("restore");

        assert_eq!(restored.pool().active(), panel.pool().active());
        assert_eq!(restored.pool().archive(), panel.pool().archive());
        for batch in [2u64, 4, 6] {
            let id = OpponentId::Pool(batch);
            assert_eq!(
                restored.ratings().get(&id).map(|entry| entry.rating),
                panel.ratings().get(&id).map(|entry| entry.rating),
                "{id} lost its rating across the resume"
            );
        }
    }

    /// Resuming a run that was interrupted after a §1.5.4 stage transition: the checkpointed panel
    /// names a baked model and a heuristic that `[pool]` does not, and both have to come back
    /// playable — the pool state is the authority on membership, the `.toml` only on parameters.
    #[test]
    fn a_restore_adopts_a_baked_model_the_config_never_named_and_keeps_its_earned_rating() {
        let (mut panel, bake_section) = panel_with_bake("restore-baked", "veteran");
        let baked = OpponentId::Baked("veteran".to_string());
        let joined = OpponentId::Heuristic(PlayerCode::W);

        panel
            .retarget(
                vec![
                    Permanent::heuristic(PlayerCode::ER).pinned(),
                    Permanent::heuristic(PlayerCode::W),
                    Permanent::baked("veteran"),
                ],
                &Default::default(),
            )
            .expect("retarget");
        panel.record(&[Episode {
            frames: Vec::new(),
            reward: 1.0,
            turns: 10,
            opponent: baked.clone(),
        }]);
        panel
            .refresh(4, &mut StdRng::seed_from_u64(13), &Default::default())
            .expect("refresh closes the period the game was recorded in");
        let earned = panel.ratings().get(&baked).expect("rated").rating;

        let json = serde_json::to_string(&panel.state()).expect("serialize");
        let state: PanelState = serde_json::from_str(&json).expect("deserialize");

        // `bake_section` names neither `w` nor any baked model — only the root one can be found in
        // — exactly like the config of a run whose baked models live in a later curriculum stage.
        let mut restored = Panel::<B>::new(
            &bake_section,
            panel.archive.clone(),
            ModelConfig::default(),
            TextEmbeddings::zeros(),
        )
        .expect("panel");
        restored
            .restore(state, &bake_section, &Default::default())
            .expect("restore");

        assert!(
            restored.models().agent_of(&baked).is_some(),
            "a restored baked model must be loaded, not just registered"
        );
        assert_eq!(
            restored.ratings().get(&baked).expect("still rated").rating,
            earned,
            "a restored baked model must keep what it earned, not `meta.toml`'s starting rating"
        );
        // The heuristic the config does not name is back in the scripted map, so it costs no
        // forward rather than being mistaken for weights the panel cannot find.
        assert!(restored.permanent_ids().contains(&joined));
        restored
            .assignment(5, &mut StdRng::seed_from_u64(14))
            .expect("every restored member resolves");
    }

    #[test]
    fn a_baked_model_joins_with_the_rating_its_meta_recorded() {
        let root = scratch("baked-root");
        let dir = root.join("veteran");
        std::fs::create_dir_all(&dir).expect("dir");
        let mut meta = super::super::baked::BakedMeta::current(ModelConfig::default());
        meta.rating.rating = 1680.0;
        meta.rating.games = 4_000;
        Baked::write_meta(&dir, &meta).expect("meta");
        save_cold(&model(), &dir.join("weights")).expect("weights");

        let mut section = section();
        section.baked = vec!["veteran".to_string()];
        section.models_root = root.display().to_string();

        let panel = Panel::<B>::new(
            &section,
            scratch("baked-archive"),
            ModelConfig::default(),
            TextEmbeddings::zeros(),
        )
        .expect("panel");
        let entry = panel
            .ratings()
            .get(&OpponentId::Baked("veteran".to_string()))
            .expect("rated");
        assert_eq!(entry.rating.rating, 1680.0);
        assert_eq!(entry.games, 4_000);
    }

    #[test]
    fn a_missing_baked_model_fails_at_startup() {
        let mut section = section();
        section.baked = vec!["absent".to_string()];
        section.models_root = scratch("baked-missing").display().to_string();

        assert!(Panel::<B>::new(
            &section,
            scratch("archive"),
            ModelConfig::default(),
            TextEmbeddings::zeros()
        )
        .is_err());
    }

    #[test]
    fn a_pin_outside_the_anchors_is_refused() {
        let mut section = section();
        section.pinned = "w".to_string();
        let err = section.permanent().expect_err("must refuse");
        assert!(err.contains("anchors"), "{err}");
    }

    #[test]
    fn an_eval_anchor_inside_the_panel_is_refused() {
        let panel = panel("disjoint");
        assert!(
            check_eval_disjoint(&panel.permanent_ids(), &[PlayerCode::E { max_depth: 2 }]).is_ok()
        );
        let err =
            check_eval_disjoint(&panel.permanent_ids(), &[PlayerCode::R]).expect_err("must refuse");
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    fn scalars_name_the_elo_curve_and_the_pool_shape() {
        let panel = panel("scalars");
        let names: Vec<String> = panel.scalars().into_iter().map(|(name, _)| name).collect();
        assert!(names.iter().any(|name| name == "elo/learner"));
        assert!(names.iter().any(|name| name == "pool/active"));
        // Never per member: a series per clone leaves a dead curve at every eviction.
        assert!(!names.iter().any(|name| name.contains("pool:b")));
    }
}
