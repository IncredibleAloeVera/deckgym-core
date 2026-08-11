//! Baked models — the frozen, curriculum-owned half of §1.5.2's permanent panel.
//!
//! A baked model is a set of weights that outlives the run that produced it: a prototype worth
//! keeping as a sparring partner, or a previous run's final checkpoint promoted to a fixed
//! reference. Unlike a pool clone it is never evicted and its rating survives in a file rather than
//! in a checkpoint, so it can be compared across runs.
//!
//! **On disk it is a directory, not a file**, because weights alone are not loadable:
//!
//! ```text
//! models/<name>/
//!   weights.mpk   the network
//!   meta.toml     the sizes it was trained at, the schema it reads, its rating and where it came from
//! ```
//!
//! **Sizes may differ; the schema may not.** Nothing requires two pool members to share a shape —
//! each is its own [`RlModel`] instance, forwards are grouped per instance anyway, and at ≈ 5.7 MB
//! a model the residency is free. So `meta.toml`'s `[model]` table is *read* to build the network
//! and never compared against the run's own. What is compared is
//! [`crate::rl::schema_fingerprint`]: a model trained before an observation width moved has exactly
//! the same shape as one trained after, loads without complaint, and plays on a projection that no
//! longer means what it meant. That is the failure this module exists to refuse, and it refuses it
//! **loudly** — a run that quietly dropped an unreadable panel member would train against a
//! different curriculum than the one its `.toml` describes, and nothing downstream would say so.
//!
//! `models/` is **tracked**, weights included — a few megabytes against a curriculum that is
//! otherwise unreproducible, since a run's panel is named in its `.toml` and a panel member nobody
//! else has makes the run impossible to repeat. A missing directory is therefore a real fault
//! (deleted, or misnamed in the config) and errors: silently continuing with a smaller panel is the
//! same lie as silently continuing with an unreadable one.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::rl::model::config::ModelConfig;
use crate::rl::{schema_fingerprint, OBS_SCHEMA_VERSION};

use super::rating::{
    Entry, OpponentId, Rating, DEFAULT_DEVIATION, DEFAULT_RATING, DEFAULT_VOLATILITY,
};

/// The recorder appends `.mpk`, so the *stem* is what [`Baked::weights`] hands to it.
const WEIGHTS_STEM: &str = "weights";
const WEIGHTS_FILE: &str = "weights.mpk";
const META_FILE: &str = "meta.toml";

/// Where a baked model came from. Free-form on purpose: it is read by people, never by the loop.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default)]
    pub source: Option<String>,
    /// The run that produced it, and the batch it was taken at — the two facts that let a stale
    /// reference model be traced back rather than guessed at.
    #[serde(default)]
    pub run: Option<String>,
    #[serde(default)]
    pub batch: Option<u64>,
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

/// The rating a baked model enters the table with.
///
/// Stored rather than defaulted so a reference model keeps what it established across runs; a
/// freshly baked one is written at §1.5.2's single initialization value, which is the same for
/// every entity so that arriving late is not itself a ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BakedRating {
    pub rating: f64,
    pub deviation: f64,
    pub volatility: f64,
    #[serde(default)]
    pub games: u64,
}

impl Default for BakedRating {
    fn default() -> Self {
        BakedRating {
            rating: DEFAULT_RATING,
            deviation: DEFAULT_DEVIATION,
            volatility: DEFAULT_VOLATILITY,
            games: 0,
        }
    }
}

/// `models/<name>/meta.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BakedMeta {
    /// The semantic half of the compatibility check.
    pub schema_version: u32,
    /// The mechanical half, hex with a `0x` prefix so a human reading the file can compare it
    /// against the one an error message printed.
    pub schema_fingerprint: String,
    /// Read to *build* the network. Never compared against the run's own model — see the module
    /// docs on why differing sizes are fine and a differing schema is not.
    pub model: ModelConfig,
    #[serde(default)]
    pub provenance: Provenance,
    #[serde(default)]
    pub rating: BakedRating,
}

impl BakedMeta {
    /// A meta for a model baked by *this* build.
    pub fn current(model: ModelConfig) -> Self {
        BakedMeta {
            schema_version: OBS_SCHEMA_VERSION,
            schema_fingerprint: format!("{:#018x}", schema_fingerprint()),
            model,
            provenance: Provenance::default(),
            rating: BakedRating::default(),
        }
    }

    /// Whether this build can read the model. Both halves are checked and reported together,
    /// because a version bump usually moves the fingerprint too and being told only one of them
    /// sends the reader looking for a second problem that does not exist.
    pub fn check_schema(&self) -> Result<(), String> {
        let expected = format!("{:#018x}", schema_fingerprint());
        if self.schema_version == OBS_SCHEMA_VERSION && self.schema_fingerprint == expected {
            return Ok(());
        }
        Err(format!(
            "schema mismatch: the model was baked against version {} / fingerprint {}, this build \
             reads version {OBS_SCHEMA_VERSION} / fingerprint {expected}. The model's *sizes* are \
             free to differ from the run's; its observation and action layout is not — re-bake it \
             from a checkpoint trained on this build.",
            self.schema_version, self.schema_fingerprint
        ))
    }
}

/// A baked model on disk, validated.
#[derive(Debug, Clone, PartialEq)]
pub struct Baked {
    pub name: String,
    pub dir: PathBuf,
    pub meta: BakedMeta,
}

impl Baked {
    /// Reads and validates `root/<name>/`.
    pub fn load(root: &Path, name: &str) -> Result<Self, String> {
        let dir = root.join(name);
        if !dir.is_dir() {
            return Err(format!(
                "baked model `{name}` not found: {} is not a directory — check the name against \
                 what is in `models/`, bake one with `examples/bake_model.rs`, or drop `{name}` \
                 from the panel.",
                dir.display()
            ));
        }
        let meta_path = dir.join(META_FILE);
        let raw = fs::read_to_string(&meta_path)
            .map_err(|err| format!("failed to read {}: {err}", meta_path.display()))?;
        let meta: BakedMeta = toml::from_str(&raw)
            .map_err(|err| format!("failed to parse {}: {err}", meta_path.display()))?;
        meta.check_schema()
            .map_err(|err| format!("baked model `{name}`: {err}"))?;

        let weights = dir.join(WEIGHTS_FILE);
        if !weights.is_file() {
            return Err(format!(
                "baked model `{name}` has a {META_FILE} but no {WEIGHTS_FILE} at {}",
                weights.display()
            ));
        }
        Ok(Baked {
            name: name.to_string(),
            dir,
            meta,
        })
    }

    /// Every baked model under `root`, by name.
    ///
    /// A directory that fails to load is an **error**, not a skip: the run's panel is named in its
    /// `.toml`, and discovering a broken model is exactly when to say so.
    pub fn discover(root: &Path) -> Result<Vec<Self>, String> {
        if !root.is_dir() {
            return Ok(Vec::new());
        }
        let mut names: Vec<String> = fs::read_dir(root)
            .map_err(|err| format!("failed to list {}: {err}", root.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .collect();
        names.sort();
        names.iter().map(|name| Baked::load(root, name)).collect()
    }

    /// The path handed to the recorder — **without** the `.mpk`, which it appends itself.
    pub fn weights(&self) -> PathBuf {
        self.dir.join(WEIGHTS_STEM)
    }

    pub fn id(&self) -> OpponentId {
        OpponentId::Baked(self.name.clone())
    }

    /// The rating-table entry this model enters with: frozen, and carrying whatever `meta.toml`
    /// recorded. Registered with [`super::rating::RatingTable::set`] rather than `ensure`, since a
    /// stored rating that is silently ignored for having been touched first is the whole reason
    /// storing it would be pointless.
    pub fn entry(&self) -> Entry {
        Entry {
            rating: Rating {
                rating: self.meta.rating.rating,
                deviation: self.meta.rating.deviation,
                volatility: self.meta.rating.volatility,
            },
            pinned: false,
            drifts: false,
            games: self.meta.rating.games,
        }
    }

    /// Writes `meta.toml` into an existing directory. The weights are the caller's to place — this
    /// module owns the metadata, not the recorder.
    pub fn write_meta(dir: &Path, meta: &BakedMeta) -> Result<(), String> {
        fs::create_dir_all(dir)
            .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
        let text = toml::to_string_pretty(meta)
            .map_err(|err| format!("failed to encode {META_FILE}: {err}"))?;
        fs::write(dir.join(META_FILE), text)
            .map_err(|err| format!("failed to write {}: {err}", dir.join(META_FILE).display()))
    }
}

/// Builds the model `meta.toml` describes and loads its weights into it.
///
/// On the *inference* backend: a frozen opponent never takes a step, so the autodiff machinery
/// would be paid for nothing (see [`super::checkpoint::load_cold`]).
pub fn load_model<B: burn::tensor::backend::Backend>(
    baked: &Baked,
    embeddings: &crate::rl::text_embedding::TextEmbeddings,
    device: &B::Device,
) -> Result<crate::rl::model::RlModel<B>, String> {
    let fresh = crate::rl::model::RlModel::<B>::new(&baked.meta.model, embeddings, device);
    super::checkpoint::load_cold(fresh, &baked.weights(), device)
        .map_err(|err| format!("baked model `{}`: {err}", baked.name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::text_embedding::TextEmbeddings;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-baked-{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    /// Bakes a real model so the round trip covers the recorder, not just the metadata.
    fn bake(root: &Path, name: &str, config: ModelConfig) -> Baked {
        let dir = root.join(name);
        fs::create_dir_all(&dir).expect("dir");
        let model = crate::rl::model::RlModel::<B>::new(
            &config,
            &TextEmbeddings::zeros(),
            &Default::default(),
        );
        super::super::checkpoint::save_cold(&model, &dir.join(WEIGHTS_STEM)).expect("weights");
        Baked::write_meta(&dir, &BakedMeta::current(config)).expect("meta");
        Baked::load(root, name).expect("load")
    }

    #[test]
    fn a_baked_model_round_trips_through_its_directory() {
        let root = scratch("round-trip");
        let baked = bake(&root, "prototype", ModelConfig::default());

        assert_eq!(baked.id(), OpponentId::Baked("prototype".to_string()));
        assert_eq!(baked.meta.schema_version, OBS_SCHEMA_VERSION);
        assert_eq!(baked.meta.model, ModelConfig::default());
        assert!(baked.dir.join(WEIGHTS_FILE).is_file());

        let loaded = load_model::<B>(&baked, &TextEmbeddings::zeros(), &Default::default());
        assert!(loaded.is_ok(), "{:?}", loaded.err());
    }

    /// The point of the whole module: a differing *size* is fine, because each member is its own
    /// instance built from its own `[model]` table.
    #[test]
    fn a_model_of_a_different_size_loads_fine() {
        let root = scratch("sizes");
        let wide = ModelConfig {
            d_model: 256,
            num_heads: 8,
            d_ff: 512,
            ..Default::default()
        };
        let baked = bake(&root, "wide", wide.clone());

        assert_eq!(baked.meta.model, wide);
        assert_ne!(baked.meta.model, ModelConfig::default());
        let loaded = load_model::<B>(&baked, &TextEmbeddings::zeros(), &Default::default());
        assert!(loaded.is_ok(), "{:?}", loaded.err());
    }

    /// And the mismatch that is *not* fine, and that no shape check would ever catch.
    #[test]
    fn a_stale_schema_is_refused_with_both_halves_named() {
        let root = scratch("stale-schema");
        let dir = root.join("ancient");
        fs::create_dir_all(&dir).expect("dir");
        let mut meta = BakedMeta::current(ModelConfig::default());
        meta.schema_fingerprint = "0x0000000000000001".to_string();
        Baked::write_meta(&dir, &meta).expect("meta");

        let err = Baked::load(&root, "ancient").expect_err("must refuse");
        assert!(err.contains("schema mismatch"), "{err}");
        assert!(err.contains("0x0000000000000001"), "{err}");
        assert!(
            err.contains(&format!("{:#018x}", schema_fingerprint())),
            "the error must name what this build expects: {err}"
        );
    }

    #[test]
    fn a_stale_version_is_refused_even_when_the_fingerprint_matches() {
        let mut meta = BakedMeta::current(ModelConfig::default());
        meta.schema_version = OBS_SCHEMA_VERSION + 1;
        assert!(meta.check_schema().is_err());
    }

    #[test]
    fn a_meta_without_weights_is_refused() {
        let root = scratch("no-weights");
        let dir = root.join("headless");
        fs::create_dir_all(&dir).expect("dir");
        Baked::write_meta(&dir, &BakedMeta::current(ModelConfig::default())).expect("meta");

        let err = Baked::load(&root, "headless").expect_err("must refuse");
        assert!(err.contains(WEIGHTS_FILE), "{err}");
    }

    /// A panel member named in the `.toml` but absent on disk is a fault, not a smaller panel.
    #[test]
    fn a_missing_directory_is_an_error_not_an_empty_panel() {
        let root = scratch("missing");
        let err = Baked::load(&root, "absent").expect_err("must refuse");
        assert!(err.contains("absent"), "{err}");
        assert!(
            err.contains("bake_model"),
            "the error should say how to produce one: {err}"
        );
    }

    #[test]
    fn discover_lists_every_model_in_name_order() {
        let root = scratch("discover");
        bake(&root, "beta", ModelConfig::default());
        bake(&root, "alpha", ModelConfig::default());

        let found = Baked::discover(&root).expect("discover");
        let names: Vec<&str> = found.iter().map(|baked| baked.name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta"]);
    }

    #[test]
    fn discover_fails_on_a_broken_model_rather_than_skipping_it() {
        let root = scratch("discover-broken");
        bake(&root, "good", ModelConfig::default());
        fs::create_dir_all(root.join("broken")).expect("dir");

        assert!(Baked::discover(&root).is_err());
    }

    #[test]
    fn a_missing_models_root_discovers_nothing_without_erroring() {
        let root = scratch("no-root").join("nowhere");
        assert_eq!(Baked::discover(&root).expect("discover"), Vec::new());
    }

    /// A stored rating is the reason `meta.toml` exists at all beyond the schema check: a reference
    /// model keeps what it established across runs.
    #[test]
    fn a_stored_rating_becomes_a_frozen_entry() {
        let root = scratch("rating");
        let dir = root.join("veteran");
        fs::create_dir_all(&dir).expect("dir");
        let mut meta = BakedMeta::current(ModelConfig::default());
        meta.rating = BakedRating {
            rating: 1712.5,
            deviation: 48.0,
            volatility: 0.055,
            games: 9_100,
        };
        Baked::write_meta(&dir, &meta).expect("meta");
        let model = crate::rl::model::RlModel::<B>::new(
            &ModelConfig::default(),
            &TextEmbeddings::zeros(),
            &Default::default(),
        );
        super::super::checkpoint::save_cold(&model, &dir.join(WEIGHTS_STEM)).expect("weights");

        let baked = Baked::load(&root, "veteran").expect("load");
        let entry = baked.entry();
        assert_eq!(entry.rating.rating, 1712.5);
        assert_eq!(entry.games, 9_100);
        assert!(!entry.drifts, "a baked model is frozen");
        assert!(!entry.pinned);
    }

    #[test]
    fn a_fresh_meta_starts_at_the_single_default_rating() {
        let meta = BakedMeta::current(ModelConfig::default());
        assert_eq!(meta.rating.rating, DEFAULT_RATING);
        assert_eq!(meta.rating.deviation, DEFAULT_DEVIATION);
        assert_eq!(meta.rating.games, 0);
    }
}
