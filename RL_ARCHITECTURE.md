# RL Architecture — DeckGym

> Architecture summary of the Reinforcement Learning pipeline for Pokémon TCG Pocket.

---

## Rust Core (`src/`)

### Game Simulator — `game.rs`, `state/`

Game engine. Simulates full Pokémon TCG Pocket rules, produces `State` objects consumed by the RL layer.

### Observation Tensor — `src/rl/observation.rs` (V5fix)

Generates a flat `Vec<f32>` of size **OBSERVATION_SIZE** = `GLOBAL_FEATURES + MAX_CARDS_IN_OBS × FEATURES_PER_CARD`.

| Constant | Value | Description |
|---|---|---|
| `GLOBAL_FEATURES` | 171 | Turn, points, sizes, energy types, stadium |
| `MAX_CARDS_IN_OBS` | 40 | Max cards encoded per observation |
| `FEATURES_PER_CARD` | 731 | Per-card feature vector dimension |
| **OBSERVATION_SIZE** | **29,411** | Total observation vector size |

#### Global features (171 dims)
- **State core (10)**: Turn count, points, deck/hand/discard sizes.
- **Energy generated (32)**: One-hot encoding of energy units available in deck slots.
- **Stadium (129)**: 1 presence flag + 128-dim text embedding of the active stadium.

#### Per-card features (731 dims)

**Base Features (42 dims)**
- **HP (1)**: Raw remaining HP.
- **Type (11)**: Energy type one-hot (including None).
- **Weakness (11)**: Weakness type one-hot.
- **Flags (8)**: `ex`, `mega`, `is_pokemon`, `is_tool`, `is_trainer`, `is_item`, `is_stadium`, `is_fossil`.
- **Meta (8)**: Evolution line size (4) and stage (3), plus ready flag (1).
- **Retreat (1)**: Normalized retreat cost.
- **Status (4)**: `confused`, `burned`, `asleep`, `paralyzed`.

**Dynamic Features (689 dims)**
- **Attacks (276)**: 2 slots × (1 raw damage + 128-dim embedding + 9-dim energy cost).
- **Ability (128)**: 128-dim text embedding.
- **Position (9)**: Location one-hot (4) + Slot one-hot (4) + Allied flag (1).
- **Supporter (128)**: 128-dim text embedding (Trainer cards only).
- **Attached Tool (128)**: 128-dim text embedding (played cards only, V5fix addition).
- **References (20)**: Text-mined references to types (9) and mechanics (11).

#### Specialized Encodings

**Fossil cards** — Encoded as special Pokemon: HP=40, Colorless type, Fighting weakness, `is_trainer=1 + is_fossil=1`. Text embeddings zeroed out.

**Stadium cards** — When active, text embedding goes in the global feature slot. Per-card: `is_stadium=1` flag, normal text embedding in hand/deck.

Static data loaded from `src/rl/generated/card_features.json` via `lazy_static!`.

### Action Mask — `src/rl/action_mask.rs` (v3.0)

Maps all `SimpleAction` variants to fixed indices. **ACTION_SPACE_SIZE = 179**.

| Section | Indices | Content |
|---|---|---|
| Board Actions | 0–29 | EndTurn, Attack (×2), Retreat (×3), UseAbility (×4), AttachEnergy (×4), ActivateBench (×3), DiscardFossil (×4), Heal (×4), AttachFromDiscard (×4), FlipCoin (×1) |
| Hand Actions | 30–129 | 20 hand slots × 5 interaction targets |
| Resolution Actions | 130–168 | ApplyDamage targets, Place/Evolve from deck, SelectHandCard (×20), DrawCard, Noop, special mechanics |
| Reserved | 169–178 | — |

Key functions: `get_action_mask(state) -> Vec<bool>`, `get_indexed_actions(state) -> Vec<(usize, Action)>`.

### Effect Categories — `src/rl/*_categories.rs`

Multi-hot encodings injected into the observation tensor for semantic card understanding.

| Module | Enum | Dims | Coverage |
|---|---|---|---|
| `ability_categories.rs` | `AbilityEffectCategory` | 17 | 53 ability IDs (Heals, Charge, Damage, Switch, Protect, Buff, EnergyManip, Evolution, Draw, Disrupt, Debuff, Drawback + activation types) |
| `attack_categories.rs` | `AttackEffectCategory` | 13 | Per-mechanic (Heal, StatusInflict, Variance, EnergyGen, EnergyDiscard, CardAdvantage, ConditionalDmg, SpreadDmg, SelfDmg, Protection, Disruption, Movement, BoardDev) |
| `supporter_categories.rs` | `SupporterEffectCategory` | 14 | ~80+ card IDs (Heal, Draw, Search, Energy, DmgBoost, Switch, Retreat, Disrupt, Evolution, Supporter, Tool, Item, Specialized, Generalist) |

### VecGame — `src/vec_game.rs`

Vectorized game environment for batched training. Reduces Python↔Rust FFI overhead from O(n_envs) to O(1) per step.

Key methods:
- `new(deck_pairs, base_seed, opponent_type)` — creates N parallel environments
- `reset_all() -> Vec<f32>` — flattened observations (n_envs × OBSERVATION_SIZE)
- `step_batch(actions: &[usize]) -> BatchStepResult` — core O(1) FFI call
- `get_action_masks() -> Vec<bool>` — flattened (n_envs × ACTION_SPACE_SIZE)

**Reward formula**: `base × speed_factor` where `speed_factor = max(1.0, 1.0 + (13 - turn_count) / 13)`, win base = `1.0 + point_diff/6.0`, loss base = `-1.0 + point_diff/6.0`, draw = `-0.5`.

**ONNX opponent pool** (behind `#[cfg(feature = "onnx")]`): `set_onnx_opponent`, `add_onnx_to_pool`, `add_baseline_to_pool`, `set_env_opponent` — supports PFSP with mixed opponent types per environment.

### Python Bindings — `src/python_bindings.rs`

PyO3 bindings exposing the `deckgym` Python module.

| Class | Purpose |
|---|---|
| `PyGame` | Single-game RL interface (`step_action`, `get_obs`, `get_action_mask`) |
| `PyVecGame` | Batched environment wrapper (all `VecGame` methods + ONNX pool) |
| `PyState` | Game state introspection (hand, board, discard, HP, etc.) |
| `PyCard` / `PyPlayedCard` | Card data and in-play state |
| `PyDeck` | Deck loading from file or string |
| `PySimulationResults` | Bulk simulation results |

Standalone functions: `simulate(...)`, `get_player_types()`.

---

## Python Training Pipeline (`python/deckgym/`)

### Config — `config.py`

Single source of truth for all Python-side constants and hyperparameters. Mirrors Rust constants.

`TrainingConfig` dataclass with defaults: `total_timesteps=30M`, `n_envs=12`, `n_steps=256`, `batch_size=1024`, `n_epochs=8`, `base_lr=1e-5`, `gamma=0.98`, `gae_lambda=0.95`, `ent_coef=0.02`. Supports YAML serialization.

### Environments — `envs/`

| Module | Class | Description |
|---|---|---|
| `base.py` | `DeckGymEnv(gym.Env)` | Single-env Gymnasium wrapper (legacy) |
| `batched.py` | `BatchedDeckGymEnv(VecEnv)` | High-perf vectorized env, drop-in SB3 VecEnv replacement, single FFI call per step |
| `self_play.py` | `SelfPlayEnv` | Self-play variant |

### Models — `models/extractors/`

Transformer-based policy network (V2 with gradient stability).

| Class | Role |
|---|---|
| `OnnxSafeAttention` | ONNX-compatible multi-head self-attention (fused QKV, SDPA/FlashAttention, scaled residual init) |
| `MultiHeadAttentionPooling` | Cross-attention pooling with learned queries |
| `CardAttentionExtractor(BaseFeaturesExtractor)` | Main SB3 feature extractor: global_proj + card_embed → N attention layers (pre-norm) → attention pooling → output_proj → concat |
| `CardAttentionPolicy(ActorCriticPolicy)` | SB3 policy wrapper |

Architecture defaults: `embed_dim=512`, `num_heads=8`, `num_layers=3`, `dropout=0.1`, `pool_queries=4`.

### Callbacks — `callbacks/`

| Module | Class | Description |
|---|---|---|
| `episode_metrics.py` | `EpisodeMetricsCallback` | Logs episode rewards, lengths, turn counts |
| `frozen_opponent.py` | `FrozenOpponentCallback` | Simple frozen self-play opponent update |
| `pfsp.py` | `PFSPCallback` | Prioritized Fictitious Self-Play — manages Rust opponent pool, curriculum stages, TrueSkill ratings, periodic model snapshots + ONNX export |
| `pause_resume.py` | `PauseResumeCallback` | Pause/resume training interactively |

### Deck Loader — `deck_loader.py`

| Class | Description |
|---|---|
| `MetaDeckLoader` | Loads JSON era files, samples by `"uniform"` or `"hierarchical"` (era → archetype → deck) |
| `CurriculumDeckLoader` | Wraps two MetaDeckLoaders, `set_difficulty(0..1)` for progressive training |

### ONNX Export — `onnx_export.py`

`export_policy_to_onnx(model, output_path)` — wraps SB3 policy, exports with dynamic batch axis. Input: `"observation"` `[batch, 29411]`, output: `"action_logits"` `[batch, 179]`.

---

## Embeddings Pipeline (`python/deckgym/embeddings/`)

Pre-computes card text embeddings baked into the Rust binary via `card_features.json`.

| Module | Role |
|---|---|
| `config.py` | `EMBEDDING_DIM=128`, `SENTENCE_MODEL="all-MiniLM-L6-v2"`, paths, TYPE_MAP (9 energy types), MECH_MAP (11 mechanic keywords) |
| `encoder.py` | `EmbeddingEncoder` — SentenceTransformer + PCA to 128 dims |
| `text_cleaner.py` | `TextCleaner` — normalizes effect text, extracts type_refs (9) and mech_refs (11) |
| `pocket_mapper.py` | `PocketMapper` — maps Pocket DB cards to feature dicts |
| `data_loader.py` | `TCGDataLoader` — loads full TCG database for PCA training |
| `generator.py` | `generate_all(...)` — end-to-end: load DB → train embeddings → generate features → save JSON |

Output: `src/rl/generated/card_features.json` — keyed by card ID, each entry contains `{ability, atk1, atk2, supporter}` (each with 128-dim embedding + type_refs + mech_refs) + `line_size` + `is_final_stage`.

---

## Scripts (`python/scripts/`)

| Script | Purpose |
|---|---|
| `train.py` | **Main training entry point.** Loads decks via `MetaDeckLoader`, creates `BatchedDeckGymEnv`, builds `MaskablePPO` with `CardAttentionExtractor`, sets up ONNX frozen opponent, runs with PFSP or frozen opponent callbacks. CLI args: `--config`, `--meta`, `--save`, `--steps`, `--lr`, `--batch-size`, `--n-envs`, `--attention-dim/heads/layers`, `--device`, etc. |
| `evaluate.py` | **Scientific evaluation.** `ScientificEvaluator` with two modes: full matrix (all archetypes aller-retour) and generalization protocol (control vs unseen decks). Outputs JSON + Markdown reports to `eval_reports/`. |
| `evaluate_human.py` | Human-playable evaluation interface |
| `generate_embeddings.py` | Runs the embeddings pipeline to regenerate `card_features.json` |
| `diagnose_model.py` | Model diagnostics (gradient norms, attention stats) |
| `benchmark_players.py` | Benchmarks rule-based player types against each other |
| `clean_archetypes.py` | Cleans/organizes archetype deck files |
