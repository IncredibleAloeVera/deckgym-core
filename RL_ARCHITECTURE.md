# 1. RL_ARCHITECTURE

Version : 1.0.0\
Any content below is subject to change.

## Table of Contents

- [1. RL\_ARCHITECTURE](#1-rl_architecture)
  - [Table of Contents](#table-of-contents)
  - [1.1. RL\_ARCHITECTURE — Part 1: Intent](#11-rl_architecture--part-1-intent)
    - [1.1.1. General framing](#111-general-framing)
    - [1.1.2. Card embeddings](#112-card-embeddings)
    - [1.1.3. The player (deck estimator)](#113-the-player-deck-estimator)
    - [1.1.4. The deckbuilder (deck cartographer)](#114-the-deckbuilder-deck-cartographer)
    - [1.1.5. Simulation budget (order of magnitude)](#115-simulation-budget-order-of-magnitude)
    - [1.1.6. Build order (each stage falsifiable on its own)](#116-build-order-each-stage-falsifiable-on-its-own)
  - [1.2. RL\_ARCHITECTURE — Part 2: Observation (v1)](#12-rl_architecture--part-2-observation-v1)
    - [1.2.1. Core design principles](#121-core-design-principles)
    - [1.2.2. Shared objects](#122-shared-objects)
    - [1.2.3. Global vector — 86 floats + 1 index](#123-global-vector--86-floats--1-index)
    - [1.2.4. Pokémon token](#124-pokémon-token)
    - [1.2.5. Attack token — action-affordance satellite (third token family)](#125-attack-token--action-affordance-satellite-third-token-family)
    - [1.2.6. Trainer token](#126-trainer-token)
    - [1.2.7. History token — opponent action trace (fourth token family)](#127-history-token--opponent-action-trace-fourth-token-family)
    - [1.2.8. Assembly and projected sizes](#128-assembly-and-projected-sizes)
    - [1.2.9. Deferred to later versions (noted, not built now)](#129-deferred-to-later-versions-noted-not-built-now)
  - [1.3. RL\_ARCHITECTURE — Part 3: Action mask (v1)](#13-rl_architecture--part-3-action-mask-v1)
    - [1.3.1. Core design principles](#131-core-design-principles)
    - [1.3.2. Regimes (mutually exclusive; dispatcher asserts exactly one)](#132-regimes-mutually-exclusive-dispatcher-asserts-exactly-one)
    - [1.3.3. §1 — Decision-point taxonomy](#133-1--decision-point-taxonomy)
    - [1.3.4. §2 — Free-play factored heads](#134-2--free-play-factored-heads)
    - [1.3.5. §3 — Stack frames, fully factorized](#135-3--stack-frames-fully-factorized)
    - [1.3.6. §4 — Frames off turn, reveals, forced](#136-4--frames-off-turn-reveals-forced)
      - [1.3.6.1. §4.1 Decoupled from turn ownership](#1361-41-decoupled-from-turn-ownership)
      - [1.3.6.2. §4.2 Reveal effects](#1362-42-reveal-effects)
      - [1.3.6.3. §4.3 Forced, Noop, internal](#1363-43-forced-noop-internal)
    - [1.3.7. §5 — Invariants \& falsifiable tests](#137-5--invariants--falsifiable-tests)
    - [1.3.8. §6 — Egocentric shapes](#138-6--egocentric-shapes)
  - [1.4. RL\_ARCHITECTURE — Part 4: Model (v1)](#14-rl_architecture--part-4-model-v1)
    - [1.4.1. Encoder](#141-encoder)
    - [1.4.2. Heads](#142-heads)
    - [1.4.3. Sizes (v1 default — starting point, `.toml`-tunable)](#143-sizes-v1-default--starting-point-toml-tunable)
  - [1.5. RL\_ARCHITECTURE — Part 5: Training loop (v1)](#15-rl_architecture--part-5-training-loop-v1)
    - [1.5.1. Learning algorithm](#151-learning-algorithm)
    - [1.5.2. Opponents — PFSP + continuous panel](#152-opponents--pfsp--continuous-panel)
    - [1.5.3. Data — self-play, two deck DBs](#153-data--self-play-two-deck-dbs)
    - [1.5.4. Curriculum \& stop](#154-curriculum--stop)
    - [1.5.5. Infrastructure](#155-infrastructure)
    - [1.5.6. Logging](#156-logging)
  - [1.6. RL\_ARCHITECTURE — Part 6: Deckbuilder (sketch)](#16-rl_architecture--part-6-deckbuilder-sketch)
    - [1.6.1. Encoder \& input](#161-encoder--input)
    - [1.6.2. Three scoring heads (EBM)](#162-three-scoring-heads-ebm)
    - [1.6.3. Sampling — GFlowNet](#163-sampling--gflownet)
    - [1.6.4. Feedback loop \& data](#164-feedback-loop--data)
  - [1.7. RL\_ARCHITECTURE — Part 7: Deployment (sketch)](#17-rl_architecture--part-7-deployment-sketch)

## 1.1. RL_ARCHITECTURE — Part 1: Intent

Target description for a reinforcement-learning system built on top of the deckgym-core
simulator (Pokémon TCG Pocket), made of two coupled agents — a **player** (deck estimator)
and a **deckbuilder** (deck-landscape cartographer). Later parts will detail implementation.

### 1.1.1. General framing

- **Closed card pool**: the fork is frozen at the start of training. No generalization to
  future expansions is intended. This is the structuring decision: it allows per-card ID
  embeddings and exhaustive per-card / per-pair statistics.
- **Finite horizon**: 99 turns maximum.
- **Zero-sum game**: win / loss / tie, no other reward signal.
- Cards in play are split into two entity types: **Pokémon** and **Trainer** (distinct
  feature vectors, `len(Pokémon) != len(Trainer)`) — no single chimera vector.
- Card texts go through a learned embedding (frozen, descriptive encoder — see below).
- **Egocentric, role-relative encoding + containerized self-play.** Observations and actions
  are encoded by role (self / opponent), never by absolute player index. Decision points are
  **not aligned with turn ownership** — a player is prompted on their own entities during the
  other's turn (forced promotion after a KO, gust-induced switch-in, forced discard). Self-play
  is containerized: the opponent is a fixed-weight policy in its own container (enabling a
  league), so each agent only resolves the frames it owns. This halves the player dimension of
  both the observation and the action heads and removes self/opp redundancy.

### 1.1.2. Card embeddings

- One freely learned ID vector per card (possible because the pool is closed).
- **Meta-neutral initialization** from raw mechanics: HP, type, costs, damage, effect text
  encoded by a small frozen LM. The prior says "these cards are mechanically similar",
  never "these cards are played together by humans".
- No deckbuilder pretraining of representations on human decks (that would contaminate them with the
  human metagame). Human decks may at most serve as a tutorial-deck distribution early in the
  player's curriculum.
- The player fine-tunes **its own copy** of the embeddings; the deckbuilder receives a
  **truly frozen copy** (not "near-frozen") of the meta-neutral version. The decoupling is
  strict and testable.

### 1.1.3. The player (deck estimator)

Goal: not to be the best, but to **play everything well enough** that its results are
attributable to the deck and cards rather than to the pilot.

- **Model-free**, best-response approach each turn in a highly stochastic game.
- Entity encoder (transformer); the observation includes board, hand **and the remaining
  deck contents** (entities with a zone flag, unordered) — the encoder's pooling is de
  facto the deck embedding, so deck conditioning is implicit.
- **Factorized actor heads**, using the embeddings of the affected/selected entities —
  aligned with the engine's action space (`Place(Card, idx)`, `Attack(Attack)`, `Evolve{..}`).
- **Mandatory action masking.**
- **Magnetic Mirror Descent** for equilibrium approximation under imperfect information.
- **Heuristic anchor** as the initial magnet (support + progress indicator).
- Self-play + past checkpoints (**PFSP**); a full AlphaStar-style league only if cycling is
  observed. Model size kept as small as possible.
- Pretraining stop criterion: **fixed step budget / winrate plateau against a frozen panel**
  (random, weighted-random, expectiminimax from the repo) — no absolute threshold like
  "40% vs expectiminimax", which is not interpretable.
- **Permanent quota of uniformly sampled decks** in the training regime, to counter
  co-evolution collapse (the player plays familiar cards better → biases scores → the
  deckbuilder re-proposes them → feedback loop). This bias comes from the competence
  distribution, not from the embeddings; the remedy is coverage.

### 1.1.4. The deckbuilder (deck cartographer)

**EBM** principle: score the association of up to 40 cards + up to 2 selected energies,
20 allied cards (a legal deck is exactly 20 cards, with ≥ 1 non-fossil Pokémon — hence
≤ 20 Pokémon / ≤ 19 Trainers, taken as 20/20 in practice) and 0 to 20 enemy cards. The goal is **not** to find the
best deck but to build a genuine landscape of decks.

Three scoring heads (final weighting to be defined later):

1. **Strength** — does the cards in the deck tend to win? Label: measured winrate.
2. **Coherence** — does the deck function? Operational definition: **win-conditioned
   synergy lift** (PMI) — winrate(X∧Y together) vs the product of marginal winrates.
   Decorrelated from Strength since it is normalized by individual card strength.
   - Shrinkage is mandatory (beta prior pulling toward the marginal while n is small).
   - Labels at the pairwise level; the EBM head learns higher orders from deck-level
     labels (aggregate of the deck's lifts).
3. **Counter** — does the deck handle the selected enemy cards well? Label: conditional
   winrate.

- **GFlowNet-style sampling**: diversity proportional to energy, no argmax.
- **Exploration bonus** in the GFlowNet reward based on a per-card / per-pair visit
  counter (possible because the pool is closed); the landscape carries the uncertainty of
  rarely visited scores.
- Feedback loop: the deckbuilder proposes decks to the player (which keeps updating its
  weights) and receives the results. Deckbuilder pretraining limited to tutorial decks
  (< 1000) — no contamination by the human meta.

### 1.1.5. Simulation budget (order of magnitude)

- ~100 games/s. Resolving a winrate to ±2.5% (95% CI) ≈ 1500 games ≈ 15 s per deck.
- Strategy: coarse mass labeling (128–256 games, ±4–6%), the EBM denoises by smoothing
  across neighboring decks; fine labels reserved for regions where the GFlowNet
  concentrates its mass.
- Target coverage: on the order of a hundred archetypes, thousands of decks — total budget
  ~10⁷–10⁸ games. Feasible on one machine.

### 1.1.6. Build order (each stage falsifiable on its own)

1. Entity encoder + masking + factorized heads, in simple self-play against the repo's
   panel → **validate that the agent learns at all**.
2. Freeze the embeddings, train the Strength-only EBM on uniform + tutorial decks →
   **validate that the landscape is meaningful** (correlation with measured winrate).
3. Only then: GFlowNet, Coherence/Counter.

## 1.2. RL_ARCHITECTURE — Part 2: Observation (v1)

Frozen specification of the observation consumed by both agents. This **is** version 1 (it
supersedes the `deprecated` branch's V5fix `observation.rs`, which is kept only as a
reference implementation).

### 1.2.1. Core design principles

1. **Identity is an index, not a payload.** The per-step observation carries only *indices*
   (`card_id`, `species_id`, `line_id`, `tool_id`) plus *dynamic* state. The heavy static
   descriptor (HP, types, costs, damage, text embeddings, …) lives in a **frozen table
   gathered in-model** by `card_id`, never serialized per step. This is the decisive break
   from the previous approach, which baked ~700 static floats into every one of 40 card slots
   at every tick.
2. **Two entity types, `len(Pokémon) != len(Trainer)`.** Two static descriptors, two input
   MLPs, projecting `emb(ids) ⊕ static ⊕ dynamic → d_model`. No chimera vector.
3. **Meta-neutral init from the static descriptor.** `card_id[c]` is *initialized* by projecting
   card `c`'s static descriptor (mechanics + frozen text LM) to `d_id`; `species_id`/`line_id`
   are initialized by **mean-pooling the `card_id` inits of their member cards** (≡ pooling the
   member descriptors then projecting, identical since the projection is linear). At runtime the
   descriptor is *also* concatenated — the overlap is intentional: explicit features = exact
   inductive bias, the ID embedding = free residual capacity. The player fine-tunes its copy; the
   deckbuilder gets a strictly frozen copy.
4. **Unordered set with masking.** Entities carry a zone flag and are permutation-invariant;
   the only spatial signal (board `slot`) is a feature, not a sequence position. Variable
   length + padding mask — no destructive `take(40)` truncation.
5. **Imperfect information is respected** (unlike V5fix, which leaked the opponent deck):

   | Entity                                  | Self                                                   | Opponent                            |
   | --------------------------------------- | ------------------------------------------------------ | ----------------------------------- |
   | Board (Pokémon, tools, attached energy) | full                                                   | full (public)                       |
   | Discard pile                            | full                                                   | full (public)                       |
   | Hand                                    | full contents                                          | **size** and partial (contextual)   |
   | Deck / draw pile                        | full contents (unordered → implicit deck conditioning) | **size + declared energy set only** |

   Both players' energy zones (`current` + `next`) are public in TCG Pocket, so both are
   observed.

   This table is the *player-mode* default. The reveal effects that punch holes in it (Silver,
   Mega Absol Ex, …) and the belief layer that maintains and invalidates those holes are a
   separate information-state component (spectator vs player mode, presence-/position-revealed
   bookkeeping). The engine is fully observable and does not implement it yet.

### 1.2.2. Shared objects

- **`Energy` (10-dim)** = `[Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness,
  Metal, Dragon, Colorless]`. The zero-vector encodes "none" — no explicit None slot. Used as
  one-hot (single 1) or as counts (integers).
- **Three ID spaces = three granularities of identity**, each its own embedding table (default
  `d_id = 64`), kept **distinct and concatenated** at the Pokémon MLP input (different natures —
  the network must tell them apart, no summation/composition):
  - `card_id` — the exact **printed card** (finest grain: do not distinguish complete reprints, differing
    HP/attacks).
  - `species_id` — the **named Pokémon across all its printings** (every "Pikachu" card → one id).
  - `line_id` — the **whole evolution lineage** (Charmander/Charmeleon/Charizard + all variants).

  `species_id`/`line_id` come from a precomputed grouping table (`evolves_from` chains + name),
  analogous to `card_features.json`; their embeddings are initialized by pooling member `card_id`
  inits (principle 3).
- **Learning as a regularized bias/modulation, not free re-training.** Each table is
  parametrized as `frozen meta-neutral init ⊕ small learned term` — an additive bias and/or a
  multiplicative gate `(1 + γ)` — learned **only by the player**, with the term's magnitude
  regularized. Rationale: the meta-neutral init is already close to target (it is the exact
  representation the deckbuilder's deck landscape consumes), so the player should *adapt*, not
  drift. This is the concrete form of Part 1's "player fine-tunes its own copy". The
  deckbuilder uses the init alone (no bias term), strictly frozen.
- **Count normalization** (all counts normalized, clamped): attached energy per type `/4`;
  discard energy per type `/12`; attack cost per type `/5`; total energy per attack `/5`; base
  retreat as one-hot(5) over 0..4; `retreat_cost_delta` `/4` (signed).
- **HP buckets**: HP ∈ {30,…,240}, **22 distinct values**. Encoded as **thermometer(22) ⊕
  one-hot(22) = 44** — thermometer for ordinality/survival thresholds, one-hot for the exact
  breakpoints that matter on a TCG (140 HP ≠ 130 ≠ 150 for an EX).
- **Damage buckets**: `fixed_damage` ∈ {0,10,…,180,200,250}, **21 distinct values**. Nominal
  damage encoded as **thermometer(21) ⊕ one-hot(21) = 42**. *Expected* (previsional) damage,
  being continuous, uses **thermometer ⊕ scalar** instead (one-hot is meaningless off-grid).
- **Legality features are a sibling projection, not a derivation.**
  `legal_actions = generate_possible_actions(state)` feeds *both* `get_observation(state,
  perspective, legal_actions)` *and* the Part 3 action mask; the legality features
  (`can_evolve_this_turn`, `ability_activatable_now`, `playable_now`, `attack_readiness`,
  threat) are that same enumeration projected onto tokens — neither derived from the mask nor
  the mask from them. Defined for `perspective = frame.actor`'s own board only (which may
  differ from the turn player — Part 3 §4.1); 0 elsewhere.

### 1.2.3. Global vector — 86 floats + 1 index

| Field                                         | Dims        | Norm / encoding                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `turn_lin`, `turn_log`, `turns_remaining`     | 3           | `t/H`, `ln(1+t)/ln(1+H)` (concave → early turns spread), `(H−t)/H`. **`H = 99`** (finite horizon). ⚠ Engine prerequisite: the current `turn_count > 30` tie cap in [src/state/mod.rs](src/state/mod.rs) must be lifted to 99, else games end at 30 and `turns_remaining` is wrong |
| `on_the_play`, `is_my_turn`, `is_setup_phase` | 3           | bits (`turn_count<=2` for setup)                                                                                                                                                                                                                                                  |
| points self / opp / diff                      | 3           | `/2`                                                                                                                                                                                                                                                                              |
| draw pile self / opp                          | 2           | `/17`                                                                                                                                                                                                                                                                             |
| hand self / opp                               | 2           | `/10`                                                                                                                                                                                                                                                                             |
| discard pile self / opp                       | 2           | `/19`                                                                                                                                                                                                                                                                             |
| energy zone `this`+`next` × self+opp          | 40          | `Energy × 4`                                                                                                                                                                                                                                                                      |
| `energy_already_attached` self / opp          | 2           | bit (turn's generated energy already placed)                                                                                                                                                                                                                                      |
| discard energy self / opp                     | 20          | `Energy × 2`, counts                                                                                                                                                                                                                                                              |
| `has_stadium`                                 | 1           | bit                                                                                                                                                                                                                                                                               |
| `stadium_id`                                  | *(1 index)* | → shared embedding table                                                                                                                                                                                                                                                          |
| `has_played_support` self / opp               | 2           | bits                                                                                                                                                                                                                                                                              |
| `has_retreated` self / opp                    | 2           | bits                                                                                                                                                                                                                                                                              |
| `has_used_stadium` self / opp                 | 2           | bits                                                                                                                                                                                                                                                                              |
| KO by opponent attack this / last turn        | 2           | bits (watchlist for removal)                                                                                                                                                                                                                                                      |

### 1.2.4. Pokémon token

Emitted for every board Pokémon (self + opp) and every Pokémon **and Fossil** in self
hand/deck/discard. Fossils use this schema (HP 40, Colorless type, Fighting weakness, 0
attacks). Static block resolved in-model from `card_id`; dynamic block is on the wire.

**Static descriptor (in-model, gathered) — 565 dims**

| Field                                                                                                             | Dims |
| ----------------------------------------------------------------------------------------------------------------- | ---- |
| `energy_type` (`Energy`)                                                                                          | 10   |
| HP base (thermo 22 ⊕ one-hot 22)                                                                                  | 44   |
| weakness (`Energy`)                                                                                               | 10   |
| stage (one-hot Basic/1/2)                                                                                         | 3    |
| base retreat cost (one-hot 0..4)                                                                                  | 5    |
| `is_ex`, `is_mega`                                                                                                | 2    |
| `has_ability`                                                                                                     | 1    |
| ability: `AbilityMechanic` multi-hot (80) ⊕ text emb (48)                                                         | 128  |
| attacks × 2, each: `fixed_damage` (42) + cost `Energy` (10) + total-energy `/5` (1) + effect text emb (128) = 181 | 362  |

**Dynamic block (on the wire) — 32 floats + 4 indices**

| Field                                                                                 | Dims      |
| ------------------------------------------------------------------------------------- | --------- |
| indices: `card_id`, `species_id`, `line_id`, `tool_id`                                | *(4 idx)* |
| zone (one-hot)                                                                        | 4         |
| allied                                                                                | 1         |
| slot (one-hot) + `is_active`                                                          | 5         |
| remaining-HP ratio                                                                    | 1         |
| attached energy (`Energy` counts `/4`, *Jungle-Totem-aware*: Serperior doubles Grass) | 10        |
| `retreat_cost_delta` (`/4`, signed additional cost from tools/abilities)              | 1         |
| status (poison, paralyze, sleep, burn, confuse)                                       | 5         |
| `can_evolve_this_turn` (mask)                                                         | 1         |
| `ability_used`                                                                        | 1         |
| `ability_activatable_now` (mask)                                                      | 1         |
| `ability_will_proc` (typed start/end-of-turn condition met)                           | 1         |
| `has_tool`                                                                            | 1         |

### 1.2.5. Attack token — action-affordance satellite (third token family)

Not a card entity: an **action-affordance token** aligned with the factorized `Attack(Attack)`
head. One token per **usable attack of each board Pokémon**. This solves the variable-attack-
count problem cleanly: an item that lets a Stage-2 use an earlier stage's attacks (from
`cards_behind`) just emits **extra attack tokens** parented to the Pokémon — no fixed cap, and
the policy can point at the exact attack it selects. (This keeps Part 1's "two *card* entity
types" intact; Attack is an action token, resolved by its own small MLP.)

**Static descriptor (in-model, gathered by `src_card_id` + `attack_slot`) — 181 dims**

| Field                                   | Dims |
| --------------------------------------- | ---- |
| `fixed_damage` (thermo 21 ⊕ one-hot 21) | 42   |
| energy cost (`Energy` counts `/5`)      | 10   |
| total energy `/5`                       | 1    |
| effect text emb                         | 128  |

**Dynamic block (on the wire) — 14 floats + 2 indices**

| Field                                                                                                                                                                                        | Dims      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| indices: `parent_pokemon_ref`, `src_card_id`                                                                                                                                                 | *(2 idx)* |
| `attack_slot` (one-hot, which attack on the source card)                                                                                                                                     | 2         |
| allied                                                                                                                                                                                       | 1         |
| `can_pay`, `deficit`, `surplus` (given parent's current effective energy)                                                                                                                    | 3         |
| **threat matrix (full Q6)**: expected-damage ratio vs each of the 4 opposing board slots (self attack → opp's 4 slots; opp attack → our 4 slots), normalized by that defender's remaining HP | 4         |
| `is_lethal` per opposing slot (guaranteed-KO floor)                                                                                                                                          | 4         |

> `src_card_id` = the card the attack's descriptor comes from — the Pokémon itself, or an
> earlier stage for a borrowed attack. Attack tokens are emitted for **every board Pokémon on
> both sides** (benched attackers included), so the threat matrix gives a full
> our-attacks × their-Pokémon picture (and symmetrically their-attacks × our-Pokémon).
> **Expected damage is 0 when the attack's energy is unmet (`can_pay = 0`) or the slot is
> unreachable** (single-target attack vs a bench slot). The random part uses coin-flip
> expectation; `is_lethal` uses the guaranteed damage floor.

### 1.2.6. Trainer token

Emitted for Item / Supporter / Tool / Stadium in **self** hand/deck/discard (opponent
hand/deck hidden). Attached tools ride on their host Pokémon's `tool_id`.

**Static descriptor (in-model, gathered) — 149 dims**

| Field                                                                                           | Dims |
| ----------------------------------------------------------------------------------------------- | ---- |
| `trainer_type` (one-hot)                                                                        | 5    |
| effect text emb                                                                                 | 128  |
| targeting: type-mask `Energy` (10) + bits `{targets_ex, targets_stage(3), targets_self/opp(2)}` | 16   |

**Dynamic block (on the wire) — 7 floats + 1..(1+K) indices**

| Field                                                                                                                                                                     | Dims      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| index: `card_id`                                                                                                                                                          | *(1 idx)* |
| targeting index **set** `{line_id, species_id}` this card affects → live-gathered + summed into a `d_id` bag in-model (reduces to the Pokémon single-index case when K=1) | *(K idx)* |
| zone (one-hot)                                                                                                                                                            | 4         |
| allied                                                                                                                                                                    | 1         |
| `playable_now` (mask)                                                                                                                                                     | 1         |
| `activation_condition_met` (separate from playability)                                                                                                                    | 1         |

> The target set is **emitted** (not frozen in the static table) because it must index the
> *trainable* line/species embeddings live — same reason the Pokémon token emits its own
> `line_id`/`species_id` rather than baking them in.

### 1.2.7. History token — opponent action trace (fourth token family)

Not a card entity and **not permutation-invariant**: an ordered trace of the opponent's last
`HISTORY_LEN = 20` observable action *choices*. Because there is **no centralized critic**
(hidden information stays part of the game dynamics, never privileged to the value head), this
trace is the model's *only* belief-bearing signal — so it encodes **what the opponent chose**,
never an outcome, and is kept as lean as possible.

Scope rules:

- **Opponent only.** Our own actions are not traced (the agent already knows them).
- **Choices only.** Only genuine opponent *decisions* (the frames a policy resolves) enter. All
  forced / automatic / internal frames — `DrawCard`, `ApplyDamage`,
  `ScheduleDelayedSpotDamage`, single-candidate auto-resolutions — are excluded.
- **No deltas / outcomes.** Only the action identity, never its result.
- **Public-index rule.** The `card_id` is attached **only if the referenced card is public**
  (board / played / discard); a choice referencing a **hidden** card (hand discard/shuffle)
  enters with `card_id = 0`. This single rule keeps the trace a leak-free proto-belief
  (Part 2 §6 info-set).
- **Crosses turns.** FIFO over the 20 most recent qualifying opponent decisions, spanning turn
  boundaries — the tempo/sequence signal is the whole point.

**Dynamic block (on the wire) — 2 floats + 2 indices**

- *(idx)* `card_id` — public referenced card; `0` = none / hidden.
- *(idx)* `head_id` — `discriminant(SimpleAction)`, bucketed per Part 3 §1, resolved in-model to
  a learned head embedding (`d = 16`).
- `recency` (2 floats) — step offset `(t−t_a)/H`, turn offset `(turn−turn_a)/H`.

> No static descriptor and no invented vocabulary: the token is
> `emb(card_id, 64) ⊕ head_emb(16) ⊕ recency(2) = 82`. `head_id` **reuses the engine
> enumeration** (single source of truth, auto-tracks any `SimpleAction` change), symmetric to how
> the policy *emits* its own actions. The `card_id` carries most of the signal; `head_id` only
> disambiguates the families an entity alone cannot (`Place` vs `Evolve` on a hand-Pokémon,
> `Retreat` vs `UseAbility` on a slot) and nullary actions (`EndTurn`, `UseStadium`). **Order
> matters** — hence the recency encoding; consumed as a sequence (Part 4 fixes whether via
> positional encoding inside the shared encoder or a separate temporal stream).

### 1.2.8. Assembly and projected sizes

MLP input token = `emb(ids) ⊕ static_descriptor ⊕ dynamic` (with `d_id = 64`, three ID
embeddings **concatenated** for Pokémon):

| Token   | ID embeddings | Static | Dynamic (+resolved embs) | **MLP input width** |
| ------- | ------------- | ------ | ------------------------ | ------------------- |
| Pokémon | 3 × 64 = 192  | 565    | 32 + 64 (tool emb) = 96  | **≈ 853**           |
| Trainer | 1 × 64 = 64   | 149    | 7 + 64 (target bag) = 71 | **≈ 284**           |
| Attack  | —             | 181    | 14                       | **≈ 195**           |
| History | 64 + 16 = 80  | —      | 2                        | **≈ 82**            |

`len(Pokémon) ≈ 853 ≠ len(Trainer) ≈ 284 ≠ len(Attack) ≈ 195 ≠ len(History) ≈ 82` — four input
MLPs. The first three are permutation-invariant entity/affordance sets; **History is ordered**
(recency-encoded, consumed as a sequence).

**Observation payload on the wire** (`MAX_POKEMON_TOKENS = MAX_TRAINER_TOKENS = 40`,
`MAX_ATTACK_TOKENS = 32`, `HISTORY_LEN = 20`; padded + masked; assert on overflow):

- Global: 86 floats + 1 index
- Pokémon: 40 × (32 floats + 4 idx) = 1280 floats + 160 idx
- Attack: 32 × (14 floats + 2 idx) = 448 floats + 64 idx
- Trainer: 40 × (7 floats + 1 idx, + target-set idx) = 280 floats + 40 idx
- History: 20 × (2 floats + 2 idx) = 40 floats + 40 idx
- **Total ≈ 2 134 floats + ~305 indices ≈ 9 KB/observation** (vs ~30 k floats in V5fix).

Static tables held once in-model: Pokémon `≈ 3233 × 565 × 4 B ≈ 7.3 MB`, Attack descriptors
`≈ 3758 × 181 × 4 B ≈ 2.7 MB`, Trainer negligible, embedding tables `≈ 3520 × 64` per ID
space — all trivial.

### 1.2.9. Deferred to later versions (noted, not built now)

- **Structured attack schema** (authoritative, parsed offline over the frozen pool):
  `{range ∈ self/active/bench/all, coin_flip ∈ none/×N/until-tails, status_inflicted(5),
  self_damage, scales_with_energy, heal, search/draw, discard}`. This is the highest-ROI
  addition — attacks have **no** typed enum in the engine (only `fixed_damage` +
  `energy_required` + free `effect` text), so today the attack text embedding is
  load-bearing and unverified on the numeric/logical structure. Until then, attacks rely on
  the frozen text embedding + `fixed_damage` + cost.
- **Pure damage estimator** (required by the Attack-token threat matrix, now **in v1**):
  `estimate_damage(state, attacker, attack, defender) -> (expected, guaranteed_floor)` —
  side-effect-free and RNG-free, weakness-adjusted, resolving coin-flip expectation
  analytically and returning 0 for unpayable/unreachable pairs. This is the observation's
  heaviest computation (our-attacks × their-Pokémon, both sides, every step); it must not
  mutate state or draw RNG. Higher-order effects it cannot resolve statically fall back to
  `fixed_damage`.
- **`ability_will_proc`** is limited to the 80 typed `AbilityMechanic` variants; text-only
  passive triggers stay at 0 until typed.
- **Text encoder**: the frozen "super-set TCG" descriptive encoder (trained on the full TCG
  card-text corpus, applied to the Pocket subset, 128-dim) is the v1 baseline and is
  meta-neutral as long as it never sees winrate/co-occurrence. Optional later: continue-
  pretrain a small model on the TCG rules DSL (MLM), and/or PCA-compress. The encoder stays
  frozen and identical across player and deckbuilder.
- **Typed per-card in-play effects**: `PlayedCard.effects` (`CardEffect` list) is currently
  not encoded; high-impact effects can later become typed bits (replacing the dropped
  effect-count feature).

## 1.3. RL_ARCHITECTURE — Part 3: Action mask (v1)

Frozen specification of the legal-action mask consumed by the player's factorized actor heads.
Single source of truth: `generate_possible_actions(state) -> (actor, Vec<Action>)`
([src/move_generation/mod.rs](src/move_generation/mod.rs)). Part 3 only *projects* that
enumeration onto the Part 2 token/head structure and states the invariants that make the
projection falsifiable — legality is never reimplemented, only bucketed.

### 1.3.1. Core design principles

1. **Engine is authoritative; the mask is a projection.**
   `mask := project(generate_possible_actions(state))`. Each set head-bit ↔ exactly one
   engine-legal `SimpleAction` (bijection, §5). The observation's legality features are the
   *sibling* projection of the same enumeration (Part 2, "Legality features are a sibling
   projection") — not a separate computation, and not derived from the mask.
2. **Factorize the frequent, point at the rare.** Top-level actions get factorized heads keyed
   on Part-2 tokens (Pokémon / Attack / Trainer). Combinatorial stack frames go through one
   generic candidate-pointer head over the engine's enumerated list — no fixed action space to
   size.
3. **Egocentric by role (Part 1).** Every frame is scored from `frame.actor`'s own
   perspective; heads address self-role or, for genuine cross-side effects, opp-role entities —
   never a player-0/1 index. Containerized self-play → an agent only resolves the frames it owns.

### 1.3.2. Regimes (mutually exclusive; dispatcher asserts exactly one)

- **SETUP** — `turn_count == 0`: only `Place` (active first; then bench + `EndTurn`).
- **STACK** — `move_generation_stack` non-empty: only the top frame's candidate list, for that
  frame's `actor` (which may not be the turn player — §4.1).
- **FREE_PLAY** — stack empty, `end_turn_pending == false`: full turn action set.
- **FORCED** — `end_turn_pending`, or any frame with a single candidate: auto-resolved, no
  learned decision (§4.3).

### 1.3.3. §1 — Decision-point taxonomy

Each `SimpleAction` maps to one head, or is internal-only (never a choice, never masked).
"Target" role is relative to `frame.actor`.

| `SimpleAction`                                                                                                                                    | Regime        | Head                | Target                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------- | ------------------------------------------- |
| `EndTurn`                                                                                                                                         | FREE / FORCED | `END_TURN` (bit)    | —                                           |
| `Place(Card, idx)`                                                                                                                                | FREE / SETUP  | `PLACE`             | self hand-Pokémon ⊗ empty slot              |
| `Evolve{..}`                                                                                                                                      | FREE          | `EVOLVE`            | self hand-evo → compatible slot (bipartite) |
| `Attach{is_turn_energy:true}`                                                                                                                     | FREE          | `ATTACH_ENERGY`     | self slot (type = zone.current)             |
| `Retreat(idx)`                                                                                                                                    | FREE          | `RETREAT`           | self bench                                  |
| `Attack(Attack)`                                                                                                                                  | FREE          | `ATTACK`            | self Attack token                           |
| `UseAbility{idx}`                                                                                                                                 | FREE          | `USE_ABILITY`       | self slot                                   |
| `Play{trainer_card}`                                                                                                                              | FREE          | `PLAY_TRAINER`      | self hand-Trainer token                     |
| `UseStadium`                                                                                                                                      | FREE          | `USE_STADIUM` (bit) | —                                           |
| `DiscardFossil{idx}`                                                                                                                              | FREE          | `DISCARD_FOSSIL`    | self slot                                   |
| `Heal` / `AttachFromDiscard` / `AttachTypedFromDiscard` / `ReturnPokemonToHand` / `ShuffleInPlayPokemonIntoDeck` / `Activate`(promotion)          | STACK         | `SLOT_PTR`          | self slot                                   |
| `Activate`(Cyrus) / `DiscardToolFromPokemon` / gust switch-in                                                                                     | STACK         | `SLOT_PTR`          | **opp** slot                                |
| `MoveEnergy` / `MoveAllDamage`                                                                                                                    | STACK         | `SLOT_PAIR`         | self (from, to)                             |
| `CommunicatePokemon`                                                                                                                              | STACK         | `HAND_PTR`          | self hand-Pokémon                           |
| `ApplyStatusToOpponentActive`                                                                                                                     | STACK         | `STATUS_CAT` (5)    | —                                           |
| `ShuffleOpponentSupporter` / `DiscardOpponentSupporter`                                                                                           | STACK         | `REVEALED_HAND_PTR` | opp revealed set (see §4.2)                 |
| `Attach{is_turn_energy:false}` / `SadaAttach` / `ShufflePokemonIntoDeck` / `ShuffleOwnCardsIntoDeck` / `DiscardOwnCards` / `HealAndDiscardEnergy` | STACK         | `CANDIDATE_PTR`     | pooled entities per candidate               |
| `ApplyEeveeBagDamageBoost` / `HealAllEeveeEvolutions` / `DiscardActiveStadium` / `Noop`                                                           | STACK         | `CANDIDATE_PTR`     | nullary candidates                          |
| `DrawCard` / `ApplyDamage` / `ScheduleDelayedSpotDamage`                                                                                          | —             | internal-only       | engine-resolved, never masked               |

### 1.3.4. §2 — Free-play factored heads

An action-type head (categorical over the 10 free-play families) is masked to families with
≥1 legal instantiation; then the chosen family's argument head(s) are masked. **Every gate
below is already an emitted Part-2 token feature** — the free-play mask is a reshape of the
observation's legality bits (both being sibling projections of `legal_actions`), not new work:

- **PLACE** — outer product `hand_basic[POKEMON_SELF] ⊗ empty_slot[4]` (factorizes exactly:
  any basic → any empty slot).
- **EVOLVE** — **bipartite** `[POKEMON_SELF × 4]` (evolution X is legal only on its
  matching pre-evolution): attention from each hand-evolution token to compatible slots.
  `from_deck` evolutions (Rare Candy) arrive as STACK frames, not here.
- **ATTACH_ENERGY** — `slot[4]`, gated by `can_attach_energy_from_zone(i)` ∧ `zone.current`.
  Energy type is not a choice (it is the zone's current).
- **RETREAT** — `bench[3]`, gated by `can_retreat` ∧ cost payable. If paying the cost requires
  choosing which energy to discard, that becomes a follow-up STACK frame.
- **ATTACK** — pointer over self Attack tokens with `can_pay = 1 ∧ ¬restricted`.
- **USE_ABILITY** — `slot[4]` = `ability_activatable_now`.
- **PLAY_TRAINER** — pointer over self hand-Trainer tokens with `playable_now = 1`.
- **USE_STADIUM** / **END_TURN** — bits. **DISCARD_FOSSIL** — self slot mask.

### 1.3.5. §3 — Stack frames, fully factorized

The top frame `(actor, Vec<SimpleAction>)` dispatches per candidate to a **typed argument
head**, reusing the free-play heads and their entity embeddings (routing in the §1 STACK rows).
`CANDIDATE_PTR` is reserved for families whose candidate is a *set/assignment* of no fixed
shape (energy distributions, `generate_combinations` card-sets, Sada triples, nullary choices).
It encodes each engine-enumerated candidate as
`type_emb ⊕ pool(referenced-entity embeddings) ⊕ scalar_args`, scores them with a shared MLP,
then softmaxes over the padded/masked candidate set. It is keyed on the same embeddings as
every other head, so no head ever sees an opaque action id.

### 1.3.6. §4 — Frames off turn, reveals, forced

#### 1.3.6.1. §4.1 Decoupled from turn ownership

Decision points are **not** aligned with whose turn it is (Part 1). Two shapes, both handled by
egocentric-by-role encoding:

- **Reactive** (`frame.actor ≠ turn player`): a player decides **on their own entities** during
  the other's turn — forced promotion after a KO (`actor = ko_receiver`,
  [apply_action_helpers.rs:639](src/actions/apply_action_helpers.rs#L639)); Sabrina making the
  defender pick their new active. No legal promotion ⇒ game ends.
- **Cross-target** (`frame.actor = turn player`, target on the opponent's board): Cyrus dragging
  up the opponent's damaged bench ([apply_trainer_action.rs:625](src/actions/apply_trainer_action.rs#L625)),
  Field Blower on an opponent tool, gust. The absolute `player` in `Activate{player,…}` /
  `DiscardToolFromPokemon{player,…}` resolves to a self/opp **role**, never a 0/1 index.

**Invariant:** obs perspective = `frame.actor`; heads address `frame.actor`'s own roles.

#### 1.3.6.2. §4.2 Reveal effects

`ShuffleOpponentSupporter` / `DiscardOpponentSupporter` (Silver, Mega Absol Ex) require the
actor to point at a card in the opponent's hand — i.e. they **reveal** it, so this is a learned
head (`REVEALED_HAND_PTR`), not a random resolution. The reveal taxonomy, the belief overlay it
reads (presence- vs position-revealed), and its invalidation on shuffle live in the
information-state component — see `NOTES.md`. The engine is fully observable and does not
implement it yet, so this head is blocked on that side-quest.

#### 1.3.6.3. §4.3 Forced, Noop, internal

- `len(candidates) == 1` (incl. `end_turn_pending`, some setup steps): auto-resolve without a
  network forward; the mask has exactly one entry.
- `Noop` is a real choice ("say no") and stays a candidate whenever the engine offers it.
- `DrawCard` (automatic since commit 2d9244a), `ApplyDamage`, `ScheduleDelayedSpotDamage` are
  engine-internal: never candidates, never masked.

### 1.3.7. §5 — Invariants & falsifiable tests

With `E = generate_possible_actions(state)`:

1. **Bijection.** `unproject(mask) ≡ E` as sets, for every reachable state (property test over
   random-player rollouts).
2. **Non-empty.** `|E| ≥ 1` always (at minimum `EndTurn` or a forced frame).
3. **Round-trip.** The selected `(head, args)` maps back to a `SimpleAction ∈ E` that
   `apply_action` accepts without panic.
4. **Regime exclusivity.** Exactly one of SETUP / STACK / FREE_PLAY / FORCED is active.
5. **Perspective.** For every STACK frame, obs perspective = `frame.actor` (§4.1).

### 1.3.8. §6 — Egocentric shapes

Self-only heads point into **self-scoped slices** of the Part-2 encoder banks, **not** the full
mixed banks — this is the concrete halving the egocentric principle buys. The encoder is
unchanged (it still attends over both boards); only the head pointer domains shrink. A 20-card
deck (Part 1) bounds a player's own tokens: `POKEMON_SELF ≤ 20`, `TRAINER_SELF ≤ 20`,
`ATTACK_SELF ≤ 16` (4 board Pokémon × 2, plus Time-Recall borrows). **No head carries a
player-index dimension; opp-role heads use a 4-slot board index, never an opp token bank.**

| Head                                                                  | Shape            | Role                                                     |
| --------------------------------------------------------------------- | ---------------- | -------------------------------------------------------- |
| `action_type`                                                         | 10               | self                                                     |
| `PLACE` / `EVOLVE`                                                    | POKEMON_SELF × 4 | self (hand Pokémon → slot)                               |
| `ATTACK`                                                              | ATTACK_SELF      | self                                                     |
| `PLAY_TRAINER`                                                        | TRAINER_SELF     | self                                                     |
| `HAND_PTR` (`CommunicatePokemon`)                                     | POKEMON_SELF     | self                                                     |
| `ATTACH_ENERGY` / `USE_ABILITY` / `DISCARD_FOSSIL` / `SLOT_PTR`(self) | 4                | self board                                               |
| `RETREAT`                                                             | 3                | self bench                                               |
| `SLOT_PAIR`                                                           | 4 × 4            | self board                                               |
| `SLOT_PTR`(opp)                                                       | 4                | opp board (cross-target only: Cyrus, Field Blower, gust) |
| `STATUS_CAT`                                                          | 5                | —                                                        |
| `USE_STADIUM` / `END_TURN`                                            | 1                | self                                                     |
| `REVEALED_HAND_PTR`                                                   | K                | opp revealed set (NOTES.md)                              |
| `CANDIDATE_PTR`                                                       | K                | per-frame candidate list                                 |

Versus a naive both-players head the Pokémon and attack pointer dims halve (40 → 20, 32 → 16)
and no head spans both sides; the only opp-role heads use a 4-slot board index for the genuine
cross-side effects.

## 1.4. RL_ARCHITECTURE — Part 4: Model (v1)

Consumes the Part 2 observation, drives the Part 3 heads. One shared encoder; heads read rows of
its output `H : [N × d_model]`, `N ≤ 133`. **No centralized critic** — value and policy share the
same imperfect-information observation. Deckbuilder (Part 6) reuses the encoder on frozen
embeddings and emits 0 History tokens.

### 1.4.1. Encoder

- **Five input projections**, one per family, each a single linear `width_f → d_model`; a learned
  token-type embedding tags the family:

  | Family  | Width |
  | ------- | ----- |
  | Global  | 150   |
  | Pokémon | 853   |
  | Trainer | 284   |
  | Attack  | 195   |
  | History | 82    |

- **Global token**: the Part 2 global vector projected to a token; its output row `H[global]` is
  the state summary for the value and nullary heads. `d_model ≥ 192` (Pokémon carries `3 × d_id`).
- **History fused in-encoder** (not a side stream): attends jointly with entities. Bidirectional,
  **no causal mask**; only History carries a recency signal, the four entity families are
  permutation-invariant.
- Pre-LN transformer blocks, MHA + FFN, padding mask on unused slots.

### 1.4.2. Heads

Egocentric, self-scoped (Part 3 §6); masked by the Part 3 mask.

- **Pointer heads** (`PLACE`, `EVOLVE`, `ATTACK`, `PLAY_TRAINER`, `HAND_PTR`, slot/pair,
  `CANDIDATE_PTR`, …): `logit_i = MLP(H[token_i])`, softmax over the masked candidate rows.
- **Nullary / global** (`action_type`, `END_TURN`, `USE_STADIUM`, `STATUS_CAT`): `MLP(H[global])`.
- **Value** (only use of pooling, value-only): `v = MLP( AttnPool₁(H) ⊕ H[global] ) ∈ [−1, 1]`.
  `AttnPool₁` = one learned query over all rows (History included). Value-loss coefficient scaled
  (`.toml`) so it doesn't distort the shared representation.

### 1.4.3. Sizes (v1 default — starting point, `.toml`-tunable)

- `d_model` = 256, blocks = 4, heads = 8 (32/head), FFN = 1024 (×4)
- `d_id` = 64 (3 concatenated for Pokémon = 192)
- input projections = 5 × linear; value attention-pool = 1 query
- **≈ 4.4 M trainable params** (bulk = per-card embeddings)

Frozen static tables ≈ **12 MB**, gather-only, not parameters. `N ≤ 133` → sub-ms forward, CPU-viable
in Burn.

## 1.5. RL_ARCHITECTURE — Part 5: Training loop (v1)

Player pretraining only (deckbuilder = Part 6). Everything below is a v1 default in the run's
`.toml`.

### 1.5.1. Learning algorithm

**Two networks:**

- **Best-response (BR)** — the Part 4 player, trained by **strict single-step MMD**: one
  mirror-descent proximal step per on-policy batch (no PPO clip, no multi-epoch). Per-step
  objective = policy-gradient with GAE advantage (shared value baseline) `+ η·KL(π_BR ‖ magnet)`
  (magnetic term) `+ τ·entropy`. Terminal reward only (win +1 / loss −1 / tie 0), **γ = 1**
  (finite horizon), GAE `λ = 0.95`.
- **Magnet (average clone)** — a **separate off-policy network**, trained by supervised behavioral
  cloning on a **reservoir buffer** of BR's past `(state, action)` → approximates the NFSP
  time-average policy. Seeded from the heuristic anchor; it is the KL target above.

Division of labour: **PFSP picks the opponent, MMD does the update, the average clone carries the
equilibrium.** Trajectory = the agent's **own decision frames** (off-turn reactive frames included,
Part 3 §4.1), reward propagated from the terminal. No centralized critic (Part 4).

### 1.5.2. Opponents — PFSP + continuous panel

- **Self-play pool** = frozen BR checkpoints, containerized. Sampling: **PFSP** (prioritized toward
  ~50% opponents) + uniform floor + **min games per performer**.
- **Frozen heuristic panel** (random, weighted-random, expectiminimax from the repo) sits
  **continuously in the opponent mix** — non-self playstyles (anti self-play overfit) *and* the
  monitoring probe (double duty). Caveat: training against it makes winrate-vs-panel a *saturation*
  signal, not held-out generalization — **self-play elo** is the cleaner generalization curve.

### 1.5.3. Data — self-play, two deck DBs

No static dataset; experience is generated on the fly. The "dataset" is the **deck sampler**:

- Two DBs: **`meta`** (O(100k) decks, archetype/meta-tiered) and **`tutorial`** (O(1k)). The
  curriculum stage names which DB it draws from.
- Per game: DB draw + **permanent uniform-deck quota** (Part 1, anti-collapse) + forced **mirror**
  (same archetype) and **pure-mirror** (same deck) quotas.
- Buffers: transient per-iteration on-policy rollout (BR + value); persistent **reservoir** (magnet
  clone).

### 1.5.4. Curriculum & stop

- **Stages** in `.toml`: `(deck DB, opponent set, magnet source)`. **Advance** when winrate ≥ **70%**
  vs the current anchor over a min game count.
- **Global stop** (Part 1): **step budget** OR **winrate-vs-panel plateau** (Δ < ε over K
  consecutive evals). No absolute threshold.

### 1.5.5. Infrastructure

- **Vectorized Rust envs**, parallel, batched inference (self = current BR, opp =
  sampled frozen policy in its container).
- **Seeding**: master seed → per-env child seeds (splittable PCG/SplitMix), fully reproducible.
- **Synchronous single learner**: collect multi-env batch → GAE → one MMD step + one magnet SL step
  → repeat.
- **Layout**: `config/*.toml` (sources); `runs/<name>/` = cloned `.toml` + `checkpoints/` +
  `logs/` + `eval/`.
- **Optimizer**: AdamW, `lr = 3e-4` (short warmup + constant), grad-clip `0.5`, weight-decay on the
  player embedding residuals (Part 2).

### 1.5.6. Logging

- **Standard**: winrate vs panel, self-play elo, losses (policy / value / KL-magnet / entropy),
  grad-norm, policy entropy, KL-to-magnet, games/s, curriculum stage.
- **Diagnostic** (pathology detection & real training consequences): per-head action-type
  distribution, turns per episode, per-head entropy, value calibration (predicted vs realized
  outcome), forced/`Noop` rate, legal-mask-size distribution.

## 1.6. RL_ARCHITECTURE — Part 6: Deckbuilder (sketch)

Distant; rationale in Part 1 §1.1.4 — this part fixes only the concrete shape. Uses the **strictly
frozen** meta-neutral embeddings (no player residual).

### 1.6.1. Encoder & input

- Reuses the **Part 4 encoder skeleton** on the frozen embedding copy, but **no game state**: input
  is a *deck as a set of card tokens* (static descriptors only — no dynamic block, no History, no
  Attack affordances). Allied set (≤ 20) + optional enemy set (≤ 20) + ≤ 2 selected energies, each
  tagged allied/enemy + energy flag.
- Pooling → a **deck representation**; the model is an **EBM**: a scalar *energy* over the
  `(deck, enemy)` association.

### 1.6.2. Three scoring heads (EBM)

- **Strength** — label: measured winrate.
- **Coherence** — win-conditioned synergy lift (PMI), pairwise labels + **shrinkage** (beta prior),
  higher orders learned from the deck-level aggregate; decorrelated from Strength (normalized by
  marginal strengths).
- **Counter** — label: conditional winrate vs the enemy set.
- Final head weighting deferred.

### 1.6.3. Sampling — GFlowNet

- Deck built **card-by-card** (sequential-add MDP, legality-masked so the 20-card / ≥ 1-non-fossil
  constraint stays reachable). Objective: **trajectory balance** (default). Terminal reward
  ∝ `exp(−energy)` → sampling **proportional to energy**, no argmax (diversity).
- **Exploration bonus** from a per-card / per-pair **visit counter** (closed pool) — the landscape
  carries the uncertainty of rarely visited regions.

### 1.6.4. Feedback loop & data

- Proposes decks to the player (weights still updating), receives results → labels. Coarse mass
  labeling (128–256 games) + fine labels where the GFlowNet concentrates (Part 1 §1.1.5).
- Pretraining limited to the **`tutorial` DB (< 1000)** — no human-meta contamination.

## 1.7. RL_ARCHITECTURE — Part 7: Deployment (sketch)

- **`Player` trait impl.** The trained best-response is wrapped as an engine `Player`; egocentric
  by role → plays as P1 or P2, vs bots or humans, unchanged (Part 1). Checkpoints are **registered
  by name** so any one is callable as a player.
- **Inference**: single batched forward, action = argmax or sample over the
  masked heads.
- **TUI advisor** (read-only overlay, no engine mutation): given the human's current state, expose
  (a) **suggested action** (policy argmax), (b) **per-action confidence** (masked-softmax over legal
  actions), (c) **state judgment** (the value head output).
- Far future: weight analysis & model dissection (Part 1 deferred).
