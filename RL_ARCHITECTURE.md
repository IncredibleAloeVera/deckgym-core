# 1. RL_ARCHITECTURE

Version: 2.1.0\
Any content below is subject to change.

2.0.0 is an editorial refactor: no specification changed, but every heading was renamed and every
cross-reference rewritten to one form, so anchors published against 1.0.x no longer resolve.

This file states what the system *is*. The reasoning behind it — rationale, measurement protocols,
rejected alternatives, calibration debts — is kept in a companion working document, `NOTES.md`,
which is **untracked (`.gitignore`) and ships with no clone of this repository**. The pointers to it
below are therefore deliberately plain text rather than links: they resolve only in the author's
working tree. Nothing a reader needs in order to implement the specification depends on them; they
mark where the argument went, not where a requirement lives. Publishing `NOTES.md` is possible
later but not planned.

## Table of Contents

- [1. RL\_ARCHITECTURE](#1-rl_architecture)
  - [Table of Contents](#table-of-contents)
  - [1.1. Part 1 — Intent](#11-part-1--intent)
    - [1.1.1. General framing](#111-general-framing)
    - [1.1.2. Card embeddings](#112-card-embeddings)
    - [1.1.3. The player (deck estimator)](#113-the-player-deck-estimator)
    - [1.1.4. The deckbuilder (deck cartographer)](#114-the-deckbuilder-deck-cartographer)
    - [1.1.5. Simulation budget](#115-simulation-budget)
    - [1.1.6. Build order](#116-build-order)
  - [1.2. Part 2 — Observation (v1)](#12-part-2--observation-v1)
    - [1.2.1. Design principles](#121-design-principles)
    - [1.2.2. Shared objects](#122-shared-objects)
    - [1.2.3. Global vector](#123-global-vector)
    - [1.2.4. Pokémon token](#124-pokémon-token)
    - [1.2.5. Attack token](#125-attack-token)
    - [1.2.6. Trainer token](#126-trainer-token)
    - [1.2.7. History token](#127-history-token)
    - [1.2.8. Assembly and sizes](#128-assembly-and-sizes)
    - [1.2.9. Deferred to later versions](#129-deferred-to-later-versions)
  - [1.3. Part 3 — Action mask (v1)](#13-part-3--action-mask-v1)
    - [1.3.1. Design principles](#131-design-principles)
    - [1.3.2. Regimes](#132-regimes)
    - [1.3.3. Decision-point taxonomy](#133-decision-point-taxonomy)
    - [1.3.4. Free-play factored heads](#134-free-play-factored-heads)
    - [1.3.5. Stack frames](#135-stack-frames)
    - [1.3.6. Frames off turn, reveals, forced](#136-frames-off-turn-reveals-forced)
      - [1.3.6.1. Decoupled from turn ownership](#1361-decoupled-from-turn-ownership)
      - [1.3.6.2. Reveal effects](#1362-reveal-effects)
      - [1.3.6.3. Forced, Noop, internal](#1363-forced-noop-internal)
    - [1.3.7. Invariants \& falsifiable tests](#137-invariants--falsifiable-tests)
    - [1.3.8. Egocentric shapes](#138-egocentric-shapes)
  - [1.4. Part 4 — Model (v1)](#14-part-4--model-v1)
    - [1.4.1. Encoder](#141-encoder)
    - [1.4.2. Heads](#142-heads)
    - [1.4.3. Sizes and measured budget](#143-sizes-and-measured-budget)
  - [1.5. Part 5 — Training loop (v1)](#15-part-5--training-loop-v1)
    - [1.5.1. Learning algorithm](#151-learning-algorithm)
    - [1.5.2. Opponents — PFSP + continuous panel](#152-opponents--pfsp--continuous-panel)
    - [1.5.3. Data — self-play, two deck DBs](#153-data--self-play-two-deck-dbs)
    - [1.5.4. Curriculum \& stop](#154-curriculum--stop)
    - [1.5.5. Infrastructure](#155-infrastructure)
    - [1.5.6. Logging](#156-logging)
    - [1.5.7. Label harvest](#157-label-harvest)
  - [1.6. Part 6 — Deckbuilder (sketch)](#16-part-6--deckbuilder-sketch)
    - [1.6.1. Encoder \& input](#161-encoder--input)
    - [1.6.2. Three scoring heads (EBM)](#162-three-scoring-heads-ebm)
    - [1.6.3. Sampling — GFlowNet](#163-sampling--gflownet)
    - [1.6.4. Feedback loop \& data](#164-feedback-loop--data)
    - [1.6.5. Warm start from the harvest](#165-warm-start-from-the-harvest)
  - [1.7. Part 7 — Deployment (sketch)](#17-part-7--deployment-sketch)

## 1.1. Part 1 — Intent

Target description for a reinforcement-learning system built on top of the deckgym-core simulator
(Pokémon TCG Pocket), made of two coupled agents: a *player* (deck estimator) and a *deckbuilder*
(deck-landscape cartographer).

### 1.1.1. General framing

- Closed card pool: the fork is frozen at the start of training. No generalization to future
  expansions is intended. This is the structuring decision — it allows per-card ID embeddings and
  exhaustive per-card / per-pair statistics.
- Finite horizon: 99 turns maximum.
- Zero-sum game: win / loss / tie, no other reward signal.
- Cards in play are split into two entity types, Pokémon and Trainer, with distinct feature vectors
  (`len(Pokémon) != len(Trainer)`). No single chimera vector.
- Card texts go through a learned embedding (frozen descriptive encoder, §1.2.9).
- Egocentric, role-relative encoding with containerized self-play. Observations and actions are
  encoded by role (self / opponent), never by absolute player index. Decision points are not aligned
  with turn ownership — a player is prompted on their own entities during the other's turn (forced
  promotion after a KO, gust-induced switch-in, forced discard). The opponent is a fixed-weight
  policy in its own container, so each agent only resolves the frames it owns. This halves the player
  dimension of both observation and action heads.

### 1.1.2. Card embeddings

- One freely learned ID vector per card (possible because the pool is closed).
- Meta-neutral initialization from raw mechanics: HP, type, costs, damage, effect text encoded by a
  small frozen LM. The prior says "these cards are mechanically similar", never "these cards are
  played together by humans".
- No pretraining of representations on human decks. Human decks may at most serve as a
  tutorial-deck distribution early in the player's curriculum.
- The player fine-tunes its own copy of the embeddings; the deckbuilder receives a strictly frozen
  copy of the meta-neutral version. The decoupling is testable.

### 1.1.3. The player (deck estimator)

Goal: not to be the best, but to play everything well enough that results are attributable to the
deck and the cards rather than to the pilot.

- Model-free, best-response each turn in a highly stochastic game.
- Entity encoder (transformer). The observation includes board, hand and the remaining deck contents
  (entities with a zone flag, unordered), so the encoder's pooling is de facto the deck embedding and
  deck conditioning is implicit.
- Factorized actor heads keyed on the embeddings of the affected entities, aligned with the engine's
  action space (`Place(Card, idx)`, `Attack(Attack)`, `Evolve{..}`).
- Mandatory action masking.
- Magnetic Mirror Descent for equilibrium approximation under imperfect information, with a
  heuristic anchor as the initial magnet (support and progress indicator).
- Self-play against past checkpoints (PFSP); a full AlphaStar-style league only if cycling is
  observed. Model size kept as small as possible.
- Pretraining stop criterion: fixed step budget or winrate plateau against a frozen panel (random,
  weighted-random, expectiminimax from the repo). No absolute threshold such as "40 % vs
  expectiminimax", which is not interpretable.
- Permanent quota of uniformly sampled decks in the training regime, to counter co-evolution
  collapse (the player plays familiar cards better → biases scores → the deckbuilder re-proposes
  them). The bias comes from the competence distribution, not from the embeddings; the remedy is
  coverage. Deckbuilder-side and unbuilt (§1.5.3, §1.6.5): it needs a generator of legal random
  decks, which the repo does not have.

### 1.1.4. The deckbuilder (deck cartographer)

EBM principle: score the association of up to 40 cards plus up to 2 selected energies — 20 allied
cards (a legal deck is exactly 20, with ≥ 1 non-fossil Pokémon, hence ≤ 20 Pokémon / ≤ 19 Trainers,
taken as 20/20 in practice) and 0 to 20 enemy cards. The goal is not to find the best deck but to
build a landscape of decks.

Three scoring heads (final weighting deferred):

1. Strength — do the cards in the deck tend to win? Label: measured winrate.
2. Coherence — does the deck function? Operational definition: win-conditioned synergy lift (PMI),
   winrate(X∧Y together) vs the product of marginal winrates. Decorrelated from Strength since it is
   normalized by individual card strength. Shrinkage is mandatory (beta prior pulling toward the
   marginal while `n` is small). Labels are pairwise; the head learns higher orders from deck-level
   labels.
3. Counter — does the deck handle the selected enemy cards? Label: conditional winrate.

- GFlowNet-style sampling: diversity proportional to energy, no argmax.
- Exploration bonus in the GFlowNet reward based on a per-card / per-pair visit counter (possible
  because the pool is closed); the landscape carries the uncertainty of rarely visited scores.
- Feedback loop: the deckbuilder proposes decks to the player (which keeps updating its weights) and
  receives the results. Deckbuilder pretraining limited to tutorial decks (< 1000).

### 1.1.5. Simulation budget

- ~100 games/s for the bare simulator. Resolving a winrate to ±2.5 % (95 % CI) ≈ 1500 games ≈ 15 s
  per deck. With the v1 player in the loop, §1.4.3's per-decision budget projects ≈ 9.5 games/s per
  model-driven seat on a 4 GB laptop GPU; the end-to-end rollout (§1.5.5) measures ≈ 23 games/s at
  64 parallel envs, batching across envs amortizing the launch latency that the projection paid per
  decision. The labeling budget reads against that number, or against a cheaper policy for the
  mass-labeling passes.
- Strategy: coarse mass labeling (128–256 games, ±4–6 %), the EBM denoising by smoothing across
  neighbouring decks; fine labels reserved for regions where the GFlowNet concentrates its mass.
- Target coverage: order of a hundred archetypes, thousands of decks — total budget ~10⁷–10⁸ games.
  Feasible on one machine.

### 1.1.6. Build order

Each stage is falsifiable on its own.

1. Entity encoder + masking + factorized heads, in simple self-play against the repo's panel →
   validate that the agent learns at all.
2. Freeze the embeddings, train the Strength-only EBM on uniform + tutorial decks → validate that
   the landscape is meaningful (correlation with measured winrate).
3. Only then: GFlowNet, Coherence / Counter.

## 1.2. Part 2 — Observation (v1)

Frozen specification of the observation consumed by both agents. It supersedes the `deprecated`
branch's V5fix `observation.rs`, kept only as a reference implementation.

### 1.2.1. Design principles

1. **Identity is an index, not a payload.** The per-step observation carries only indices
   (`card_id`, `species_id`, `line_id`, `tool_id`) plus dynamic state. The static descriptor (HP,
   types, costs, damage, text embeddings, …) lives in a frozen table gathered in-model by `card_id`,
   never serialized per step. This is the break from the previous approach, which baked ~700 static
   floats into every one of 40 card slots at every tick.
2. **Two entity types.** Two static descriptors, two input MLPs, projecting
   `emb(ids) ⊕ static ⊕ dynamic → d_model`.
3. **Meta-neutral init from the static descriptor.** `card_id[c]` is initialized by projecting card
   `c`'s static descriptor to `d_id`; `species_id` / `line_id` are initialized by mean-pooling the
   `card_id` inits of their members (equivalent to pooling the descriptors then projecting, the
   projection being linear). At runtime the descriptor is also concatenated — the overlap is
   intentional: explicit features give an exact inductive bias, the ID embedding free residual
   capacity.
4. **Unordered set with masking.** Entities carry a zone flag and are permutation-invariant; the
   only spatial signal (board `slot`) is a feature, not a sequence position. Variable length plus
   padding mask — no destructive `take(40)` truncation.
5. **Imperfect information is respected** (unlike V5fix, which leaked the opponent deck):

   | Entity | Self | Opponent |
   | --- | --- | --- |
   | Board (Pokémon, tools, attached energy) | full | full (public) |
   | Discard pile | full | full (public) |
   | Hand | full contents | size, partial (contextual) |
   | Deck / draw pile | full contents (unordered → implicit deck conditioning) | size + the energy types seen so far only |

   Both players' energy zones (`current` + `next`) are public in TCG Pocket, so both are observed.

   This table is the player-mode default. The reveal effects that punch holes in it (Silver, Mega
   Absol Ex, …) are tracked by the belief overlay ([src/belief/](src/belief/)), a separate
   information-state component: the engine stays fully observable (spectator mode) and
   `Game::enable_belief` switches on the per-player bookkeeping, monotone presence and volatile
   position, maintained off typed reveal and movement events. `get_observation` takes that overlay
   as an argument and renders both halves — a known card of the opponent's hand is a token in
   whichever bank its kind belongs to, `allied = false`, `zone = Hand`; a card whose position
   marker went stale (a shuffled-away hand) survives as `zone = Unknown`, the monotone `presence`
   netted against every copy currently visible on their board or in their discard. That netting is
   what keeps a played card from being rendered twice — once face-up in the discard, once as a
   claim that it is still hidden. Both are ordered by `card_index`, so a bank is a function of the
   state and not of a `HashMap` walk. Passing `None` is the table above, unchanged. Taxonomy and
   invalidation rules: `NOTES.md`.

### 1.2.2. Shared objects

- **`Energy` (10-dim)** = `[Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal,
  Dragon, Colorless]`. The zero vector encodes "none". Used as one-hot or as counts.
- **Three ID spaces**, three granularities of identity, each its own embedding table (default
  `d_id = 64`), kept distinct and concatenated at the Pokémon MLP input — different natures, so no
  summation or composition:
  - `card_id` — the exact printed card. Complete reprints are not distinguished; differing HP or
    attacks are.
  - `species_id` — the named Pokémon across all its printings.
  - `line_id` — the whole evolution lineage (Charmander / Charmeleon / Charizard and variants).

  `species_id` / `line_id` come from a precomputed grouping table (`evolves_from` chains + name),
  analogous to `card_features.json`.
- **Learning is a regularized bias, not free re-training.** Each table is parametrized as
  `frozen meta-neutral init ⊕ small learned term` — an additive bias and/or a multiplicative gate
  `(1 + γ)` — learned only by the player, with the term's magnitude regularized. The init is already
  the representation the deckbuilder consumes, so the player adapts rather than drifts. The
  deckbuilder uses the init alone, strictly frozen.
- **Count normalization** (all counts normalized and clamped): attached energy per type `/4`;
  discard energy per type `/12`; attack cost per type `/5`; total energy per attack `/5`; base
  retreat as one-hot(5) over 0..4; `retreat_cost_delta` `/4` (signed).
- **HP buckets**: HP ∈ {30,…,240}, 22 distinct values, encoded as thermometer(22) ⊕ one-hot(22) = 44
  — thermometer for ordinality and survival thresholds, one-hot for the exact breakpoints that
  matter (140 HP ≠ 130 ≠ 150 for an EX).
- **Damage buckets**: `fixed_damage` ∈ {0,10,…,180,200,250}, 21 distinct values, as
  thermometer(21) ⊕ one-hot(21) = 42. Expected (previsional) damage, being continuous, uses
  thermometer ⊕ scalar instead.
- **Legality features are a sibling projection, not a derivation.**
  `legal_actions = generate_possible_actions(state)` feeds both
  `get_observation(state, perspective, legal_actions)` and the Part 3 action mask; the legality
  features (`can_evolve_this_turn`, `ability_activatable_now`, `playable_now`, `attack_readiness`,
  threat) are that enumeration projected onto tokens — neither derived from the mask nor the mask
  from them. Defined for `perspective = frame.actor`'s own board only (§1.3.6.1); 0 elsewhere.

### 1.2.3. Global vector

106 floats + 1 index.

| Field | Dims | Norm / encoding |
| --- | --- | --- |
| `turn_lin`, `turn_log`, `turns_remaining` | 3 | `t/H`, `ln(1+t)/ln(1+H)` (concave → early turns spread), `(H−t)/H`, with `H = 99`. Engine prerequisite: the `turn_count > 30` tie cap in [src/state/mod.rs](src/state/mod.rs) must be lifted to 99, else games end at 30 and `turns_remaining` is wrong |
| `on_the_play`, `is_my_turn`, `is_setup_phase` | 3 | bits (`turn_count <= 2` for setup). `on_the_play` is 0 for both players during setup (`turn_count == 0`): both place their boards simultaneously in the real game, so the engine's placement alternation is an artifact that must not leak |
| points self / opp / diff | 3 | `/2` |
| draw pile self / opp | 2 | `/17` |
| hand self / opp | 2 | `/10` |
| discard pile self / opp | 2 | `/19` |
| deck energies self / opp | 20 | `Energy × 2`, multi-hot. Self is the declared set (≤ 3 types, what drives that player's energy generation). Opp is *not*: the composition is private in Pocket, so this is the monotone set of types seen leaving their Energy Zone, tracked by `BeliefTracker::seen_opponent_energy` and empty without a belief overlay |
| energy zone `this` + `next` × self + opp | 40 | `Energy × 4` |
| `energy_already_attached` self / opp | 2 | bit (turn's generated energy already placed; only ever set for the turn player once generation has happened — turn 1 grants no energy) |
| discard energy self / opp | 20 | `Energy × 2`, counts |
| `has_stadium` | 1 | bit |
| `stadium_id` | *(1 index)* | → shared embedding table |
| `has_played_support` self / opp | 2 | bits |
| `has_retreated` self / opp | 2 | bits |
| `has_used_stadium` self / opp | 2 | bits |
| KO by opponent attack this / last turn | 2 | bits |

### 1.2.4. Pokémon token

Emitted for every board Pokémon (self + opp), every Pokémon and Fossil in self hand / deck /
discard, and every one in the opponent's discard pile (public per §1.2.1; the caps of §1.2.8 absorb
it, a 20-card deck bounding each side's tokens). Fossils use this schema (HP 40, Colorless type,
Fighting weakness, 0 attacks).

Static descriptor (in-model, gathered) — 565 dims. The ability multi-hot tracks the engine's
`AbilityMechanic` enum, so this width moves with it: adding a mechanic widens the descriptor and
invalidates any serialized table.

| Field | Dims |
| --- | --- |
| `energy_type` (`Energy`) | 10 |
| HP base (thermo 22 ⊕ one-hot 22) | 44 |
| weakness (`Energy`) | 10 |
| stage (one-hot Basic/1/2) | 3 |
| base retreat cost (one-hot 0..4) | 5 |
| `is_ex`, `is_mega` | 2 |
| `has_ability` | 1 |
| ability: `AbilityMechanic` multi-hot (80 at spec time) ⊕ text emb (48) | 128 |
| attacks × 2, each `fixed_damage` (42) + cost `Energy` (10) + total energy `/5` (1) + effect text emb (128) = 181 | 362 |

Dynamic block (on the wire) — 33 floats + 4 indices.

| Field | Dims |
| --- | --- |
| indices: `card_id`, `species_id`, `line_id`, `tool_id` | *(4 idx)* |
| zone (one-hot: board, hand, deck, discard, unknown) | 5 |
| allied | 1 |
| slot (one-hot) + `is_active` | 5 |
| remaining-HP ratio | 1 |
| attached energy (`Energy` counts `/4`, Jungle-Totem-aware: Serperior doubles Grass) | 10 |
| `retreat_cost_delta` (`/4`, signed additional cost from tools / abilities) | 1 |
| status (poison, paralyze, sleep, burn, confuse) | 5 |
| `can_evolve_this_turn` (mask) | 1 |
| `ability_used` | 1 |
| `ability_activatable_now` (mask) | 1 |
| `ability_will_proc` (typed start / end-of-turn condition met) | 1 |
| `has_tool` | 1 |

### 1.2.5. Attack token

Third token family, and not a card entity: an action-affordance token aligned with the factorized
`Attack(Attack)` head. One token per usable attack of each board Pokémon. This solves the
variable-attack-count problem — an item letting a Stage-2 use an earlier stage's attacks (from
`cards_behind`) emits extra attack tokens parented to the Pokémon, with no fixed cap, and the policy
can point at the exact attack it selects.

Static descriptor (in-model, gathered by `src_card_id` + `attack_slot`) — 181 dims.

| Field | Dims |
| --- | --- |
| `fixed_damage` (thermo 21 ⊕ one-hot 21) | 42 |
| energy cost (`Energy` counts `/5`) | 10 |
| total energy `/5` | 1 |
| effect text emb | 128 |

Dynamic block (on the wire) — 14 floats + 2 indices.

| Field | Dims |
| --- | --- |
| indices: `parent_pokemon_ref`, `src_card_id` | *(2 idx)* |
| `attack_slot` (one-hot, which attack on the source card) | 2 |
| allied | 1 |
| `can_pay`, `deficit`, `surplus` (given parent's current effective energy) | 3 |
| threat matrix (full Q6): expected-damage ratio vs each of the 4 opposing board slots, normalized by that defender's remaining HP | 4 |
| `is_lethal` per opposing slot (guaranteed-KO floor) | 4 |

`src_card_id` is the card the attack's descriptor comes from — the Pokémon itself, or an earlier
stage for a borrowed attack. Attack tokens are emitted for every board Pokémon on both sides
(benched attackers included), so the threat matrix gives a full our-attacks × their-Pokémon picture
and symmetrically. Expected damage is 0 when the attack's energy is unmet (`can_pay = 0`) or the
slot is unreachable (single-target attack vs a bench slot). The random part uses coin-flip
expectation; `is_lethal` uses the guaranteed damage floor.

### 1.2.6. Trainer token

Emitted for Item / Supporter / Tool / Stadium in self hand / deck / discard and in the opponent's
discard pile (public per §1.2.1; opponent hand and deck stay hidden). Attached tools ride on their
host Pokémon's `tool_id`.

Static descriptor (in-model, gathered) — 149 dims.

| Field | Dims |
| --- | --- |
| `trainer_type` (one-hot) | 5 |
| effect text emb | 128 |
| targeting: type-mask `Energy` (10) + bits `{targets_ex, targets_stage(3), targets_self/opp(2)}` | 16 |

Dynamic block (on the wire) — 8 floats + 1..(1+K) indices.

| Field | Dims |
| --- | --- |
| index: `card_id` | *(1 idx)* |
| targeting index set `{line_id, species_id}` this card affects → live-gathered and summed into a `d_id` bag in-model (reduces to the single-index case when K = 1) | *(K idx)* |
| zone (one-hot: board, hand, deck, discard, unknown) | 5 |
| allied | 1 |
| `playable_now` (mask) | 1 |
| `activation_condition_met` (separate from playability) | 1 |

The target set is emitted rather than frozen in the static table because it must index the trainable
line / species embeddings live — the same reason the Pokémon token emits its own `line_id` /
`species_id`.

### 1.2.7. History token

Fourth token family, not a card entity and not permutation-invariant: an ordered trace of the
opponent's last `HISTORY_LEN = 20` observable action choices. Because there is no centralized critic
(hidden information stays part of the game dynamics, never privileged to the value head), this trace
is the model's only belief-bearing signal — so it encodes what the opponent chose, never an outcome,
and is kept lean.

Scope rules:

- Opponent only. Our own actions are not traced.
- Choices only. Only genuine opponent decisions (the frames a policy resolves) enter; forced,
  automatic and internal frames (`DrawCard`, `ApplyDamage`, `ScheduleDelayedSpotDamage`,
  single-candidate auto-resolutions) are excluded.
- No deltas or outcomes. Only the action identity.
- Public-index rule. The `card_id` is attached only if the referenced card is public (board /
  played / discard); a choice referencing a hidden card (hand discard or shuffle) enters with
  `card_id = 0`. This keeps the trace a leak-free proto-belief.
- Crosses turns. FIFO over the 20 most recent qualifying opponent decisions, spanning turn
  boundaries — the tempo signal is the point.

Dynamic block (on the wire) — 2 floats + 2 indices:

- *(idx)* `card_id` — public referenced card; `0` = none / hidden.
- *(idx)* `head_id` — `discriminant(SimpleAction)`, bucketed per §1.3.3, resolved in-model to a
  learned head embedding (`d = 16`).
- `recency` (2 floats) — step offset `(t−t_a)/H`, turn offset `(turn−turn_a)/H`.

No static descriptor and no invented vocabulary: the token is
`emb(card_id, 64) ⊕ head_emb(16) ⊕ recency(2) = 82`. `head_id` reuses the engine enumeration (single
source of truth, auto-tracking any `SimpleAction` change), symmetric to how the policy emits its own
actions. The `card_id` carries most of the signal; `head_id` only disambiguates the families an
entity alone cannot (`Place` vs `Evolve` on a hand-Pokémon, `Retreat` vs `UseAbility` on a slot) and
nullary actions (`EndTurn`, `UseStadium`). Order matters, hence the recency encoding.

### 1.2.8. Assembly and sizes

MLP input token = `emb(ids) ⊕ static_descriptor ⊕ dynamic`, with `d_id = 64` and three ID embeddings
concatenated for Pokémon:

| Token | ID embeddings | Static | Dynamic (+resolved embs) | MLP input width |
| --- | --- | --- | --- | --- |
| Pokémon | 3 × 64 = 192 | 565 | 33 + 64 (tool emb) = 97 | ≈ 854 |
| Trainer | 1 × 64 = 64 | 149 | 8 + 64 (target bag) = 72 | ≈ 285 |
| Attack | — | 181 | 14 | ≈ 195 |
| History | 64 + 16 = 80 | — | 2 | ≈ 82 |

Four input MLPs, since the four widths differ. The first three families are permutation-invariant
entity / affordance sets; History is ordered.

Observation payload on the wire (`MAX_POKEMON_TOKENS = MAX_TRAINER_TOKENS = 40`,
`MAX_ATTACK_TOKENS = 32`, `HISTORY_LEN = 20`; padded and masked, assert on overflow):

40 is a *proof*, not a margin: two 20-card decks are 40 cards, each card occupies at most one row —
including under the belief render, where the `presence` residual is netted against the copies
already visible, so a card is never both hidden and public. The bound is tight rather than generous:
`the_wire_form_holds_with_the_belief_overlay_on` walks whole games on an all-Pokémon deck and peaks
at 37 of 40, against 20 / 20 for a conventional 10-Pokémon list. Widening the banks would buy
headroom above a ceiling that cannot be crossed, and pay for it in `SEQ_LEN`, which the encoder's
attention is quadratic in.

- Global: 106 floats + 1 index
- Pokémon: 40 × (33 floats + 4 idx) = 1320 floats + 160 idx
- Attack: 32 × (14 floats + 2 idx) = 448 floats + 64 idx
- Trainer: 40 × (8 floats + 1 idx, + target-set idx) = 320 floats + 40 idx
- History: 20 × (2 floats + 2 idx) = 40 floats + 40 idx
- Total ≈ 2234 floats + ~305 indices ≈ 9 KB per observation, against ~30 k floats in V5fix.

Static tables held once in-model: Pokémon ≈ 3233 × 565 × 4 B ≈ 7.3 MB, Attack descriptors
≈ 3758 × 181 × 4 B ≈ 2.7 MB, Trainer negligible, embedding tables ≈ 3520 × 64 per ID space.

### 1.2.9. Deferred to later versions

- **Structured attack schema** (authoritative, parsed offline over the frozen pool):
  `{range ∈ self/active/bench/all, coin_flip ∈ none/×N/until-tails, status_inflicted(5),
  self_damage, scales_with_energy, heal, search/draw, discard}`. Highest-ROI addition: attacks have
  no typed enum in the engine (only `fixed_damage`, `energy_required` and free `effect` text), so
  today the attack text embedding is load-bearing and unverified on numeric and logical structure.
- **Pure damage estimator**, required by the Attack-token threat matrix and therefore already in v1:
  `estimate_damage(state, attacker, attack, defender) -> (expected, guaranteed_floor)` —
  side-effect-free and RNG-free, weakness-adjusted, resolving coin-flip expectation analytically and
  returning 0 for unpayable or unreachable pairs. This is the observation's heaviest computation
  (our-attacks × their-Pokémon, both sides, every step). Higher-order effects it cannot resolve
  statically fall back to `fixed_damage`.
- **`ability_will_proc`** is limited to the 80 typed `AbilityMechanic` variants; text-only passive
  triggers stay at 0 until typed.
- **Text encoder**: the frozen "super-set TCG" descriptive encoder (trained on the full TCG card-text
  corpus, applied to the Pocket subset, 128-dim) is the v1 baseline, meta-neutral as long as it never
  sees winrate or co-occurrence. Optional later: continue-pretrain a small model on the TCG rules DSL
  (MLM). It stays frozen and identical across player and deckbuilder.

  **The PCA block is whitened, and that is part of the frozen artifact** — components divided by
  their corpus standard deviation, block by `sqrt(dim)`. Raw, the table measured an effective rank of
  24/128 and a squared norm of 0.29 against ~7 for the damage thermometer beside it in an Attack
  descriptor: present at full width (71 % of it) and 3.9 % of its energy, which is a channel a single
  linear cannot weigh. Whitened it reads 124/128 and ~1. `OBS_SCHEMA_VERSION = 3` is what refuses
  models trained against the unwhitened values.
  The pre-whitening table stays recoverable exactly
  (`auxiliaries/text_embeddings/unwhiten.py`), which is what makes the two comparable at init on
  identical weights and frames ([examples/text_scale_ablation.rs](examples/text_scale_ablation.rs)).
  It does not decide the depth at which an attack head forms; measurement in `NOTES.md`.
- **Typed per-card in-play effects**: `PlayedCard.effects` (`CardEffect` list) is not encoded;
  high-impact effects can later become typed bits.

## 1.3. Part 3 — Action mask (v1)

Frozen specification of the legal-action mask consumed by the player's factorized actor heads.
Single source of truth: `generate_possible_actions(state) -> (actor, Vec<Action>)`
([src/move_generation/mod.rs](src/move_generation/mod.rs)). Part 3 only projects that enumeration
onto the Part 2 token / head structure and states the invariants that make the projection
falsifiable — legality is never reimplemented, only bucketed.

### 1.3.1. Design principles

1. **Engine is authoritative; the mask is a projection.**
   `mask := project(generate_possible_actions(state))`. Each set head-bit corresponds to exactly one
   engine-legal `SimpleAction` (bijection, §1.3.7). The observation's legality features are the
   sibling projection of the same enumeration (§1.2.2), not a separate computation.
2. **Factorize the frequent, point at the rare.** Top-level actions get factorized heads keyed on
   Part 2 tokens (Pokémon / Attack / Trainer). Combinatorial stack frames go through one generic
   candidate-pointer head over the engine's enumerated list, so there is no fixed action space to
   size.
3. **Egocentric by role.** Every frame is scored from `frame.actor`'s perspective; heads address
   self-role or, for genuine cross-side effects, opp-role entities — never a player-0/1 index.

### 1.3.2. Regimes

Mutually exclusive; the dispatcher asserts exactly one.

- **SETUP** — `turn_count == 0`: only `Place` (active first, then bench + `EndTurn`).
- **STACK** — `move_generation_stack` non-empty: only the top frame's candidate list, for that
  frame's `actor`, which may not be the turn player (§1.3.6.1).
- **FREE_PLAY** — stack empty, `end_turn_pending == false`: full turn action set.
- **FORCED** — `end_turn_pending`, or any frame with a single candidate: auto-resolved, no learned
  decision (§1.3.6.3). Tested first, so it takes precedence over the other three.

### 1.3.3. Decision-point taxonomy

Each `SimpleAction` maps to one head, or is internal-only (never a choice, never masked). "Target"
role is relative to `frame.actor`.

| `SimpleAction` | Regime | Head | Target |
| --- | --- | --- | --- |
| `EndTurn` | FREE / FORCED | `END_TURN` (bit) | — |
| `Place(Card, idx)` | FREE / SETUP | `PLACE` | self hand-Pokémon ⊗ empty slot |
| `Evolve{..}` | FREE | `EVOLVE` | self hand-evo → compatible slot (bipartite) |
| `Attach{is_turn_energy:true}` | FREE | `ATTACH_ENERGY` | self slot (type = zone.current) |
| `Retreat(idx)` | FREE | `RETREAT` | self bench |
| `Attack(Attack)` | FREE | `ATTACK` | self Attack token |
| `UseAbility{idx}` | FREE | `USE_ABILITY` | self slot |
| `Play{trainer_card}` | FREE | `PLAY_TRAINER` | self hand-Trainer token |
| `UseStadium` | FREE | `USE_STADIUM` (bit) | — |
| `DiscardFossil{idx}` | FREE | `DISCARD_FOSSIL` | self slot |
| `Heal` / `AttachFromDiscard` / `AttachTypedFromDiscard` / `ReturnPokemonToHand` / `ShuffleInPlayPokemonIntoDeck` / `Activate`(promotion) | STACK | `SLOT_PTR` | self slot |
| `Activate`(Cyrus) / `DiscardToolFromPokemon` / gust switch-in | STACK | `SLOT_PTR` | opp slot |
| `MoveEnergy` / `MoveAllDamage` | STACK | `SLOT_PAIR` | self (from, to) |
| `CommunicatePokemon` | STACK | `HAND_PTR` | self hand-Pokémon |
| `ApplyStatusToOpponentActive` | STACK | `STATUS_CAT` (5) | — |
| `AttachTool` | STACK | `SLOT_PTR` | self slot (the tool is fixed by the frame) |
| `ApplyDamage`(single target) / `ScheduleDelayedSpotDamage` | STACK | `SLOT_PTR` | opp slot (spot-damage targeting) |
| `ShuffleOpponentSupporter` / `DiscardOpponentSupporter` | STACK | `REVEALED_HAND_PTR` | opp revealed set (§1.3.6.2) |
| `Attach{is_turn_energy:false}` / `SadaAttach` / `ShufflePokemonIntoDeck` / `ShuffleOwnCardsIntoDeck` / `DiscardOwnCards` / `HealAndDiscardEnergy` / `SwitchHandCardForRandomTool` | STACK | `CANDIDATE_PTR` | pooled entities per candidate |
| `ApplyEeveeBagDamageBoost` / `HealAllEeveeEvolutions` / `DiscardActiveStadium` / `DiscardRandomOpponentActiveEnergy` / `Noop` | STACK | `CANDIDATE_PTR` | nullary candidates |
| `DrawCard` | — | internal-only | engine-resolved, never masked |

Spot damage is a choice, not an internal frame: the engine hands the attacker a genuine
multi-candidate frame of `ApplyDamage` / `ScheduleDelayedSpotDamage` when an attack picks its target.
Those are cross-target decisions on a public board and get the opp-role `SLOT_PTR`, exactly like
Cyrus; only `DrawCard` is truly engine-internal. A multi-target `ApplyDamage` payload has no single
slot to point at and falls through to `CANDIDATE_PTR`.

### 1.3.4. Free-play factored heads

An action-type head (categorical over the 10 free-play families) is masked to families with ≥ 1 legal
instantiation; then the chosen family's argument head(s) are masked. Every gate below is already an
emitted Part 2 token feature — the free-play mask is a reshape of the observation's legality bits,
both being sibling projections of `legal_actions`.

- **PLACE** — outer product `hand_basic[POKEMON_SELF] ⊗ empty_slot[4]`; factorizes exactly.
- **EVOLVE** — bipartite `[POKEMON_SELF × 4]`, evolution X being legal only on its matching
  pre-evolution: attention from each hand-evolution token to compatible slots. `from_deck`
  evolutions (Rare Candy) would arrive as STACK frames; the head already routes them (deck slice
  instead of hand slice), and the engine prints no such card today.
- **ATTACH_ENERGY** — `slot[4]`, gated by `can_attach_energy_from_zone(i)` ∧ `zone.current`. Energy
  type is not a choice.
- **RETREAT** — `bench[3]`, gated by `can_retreat` ∧ cost payable. If paying the cost requires
  choosing which energy to discard, that becomes a follow-up STACK frame.
- **ATTACK** — pointer over self Attack tokens with `can_pay = 1 ∧ ¬restricted`.
- **USE_ABILITY** — `slot[4]` = `ability_activatable_now`.
- **PLAY_TRAINER** — pointer over self hand-Trainer tokens with `playable_now = 1`.
- **USE_STADIUM** / **END_TURN** — bits. **DISCARD_FOSSIL** — self slot mask.

### 1.3.5. Stack frames

The top frame `(actor, Vec<SimpleAction>)` dispatches per candidate to a typed argument head,
reusing the free-play heads and their entity embeddings (the STACK rows of §1.3.3). `CANDIDATE_PTR`
is reserved for families whose candidate is a set or assignment of no fixed shape (energy
distributions, `generate_combinations` card-sets, Sada triples, nullary choices). It encodes each
engine-enumerated candidate as `type_emb ⊕ pool(referenced-entity embeddings) ⊕ scalar_args`, scores
them with a shared MLP, then softmaxes over the padded candidate set. It is keyed on the same
embeddings as every other head, so no head ever sees an opaque action id.

Dispatch is per candidate, not per frame: a mixed frame (`Heal(0)`, `Heal(1)`, `Noop`) sets bits on
`SLOT_PTR` and `CANDIDATE_PTR`, and the policy softmaxes over the union of set bits.

`CANDIDATE_PTR` is also the escape hatch. A typed head addresses its argument, not the whole action,
so two distinct candidates in one frame can land on the same bit (two `Heal`s of different amounts on
one slot; a tool frame offering several tools). Whenever that happens the colliding head is demoted
wholesale to `CANDIDATE_PTR` for that frame and the candidate is never dropped. This is what makes
§1.3.7's bijection hold on every frame the engine can produce rather than only the tidy ones, and why
a new `SimpleAction` variant degrades gracefully instead of silently disappearing from the mask.

### 1.3.6. Frames off turn, reveals, forced

#### 1.3.6.1. Decoupled from turn ownership

Decision points are not aligned with whose turn it is (§1.1.1). Two shapes, both handled by
egocentric-by-role encoding:

- **Reactive** (`frame.actor ≠ turn player`): a player decides on their own entities during the
  other's turn — forced promotion after a KO (`actor = ko_receiver`,
  [apply_action_helpers.rs:639](src/actions/apply_action_helpers.rs#L639)); Sabrina making the
  defender pick their new active. No legal promotion ⇒ game ends.
- **Cross-target** (`frame.actor = turn player`, target on the opponent's board): Cyrus dragging up
  the opponent's damaged bench
  ([apply_trainer_action.rs:625](src/actions/apply_trainer_action.rs#L625)), Field Blower on an
  opponent tool, gust. The absolute `player` in `Activate{player,…}` /
  `DiscardToolFromPokemon{player,…}` resolves to a self / opp role, never a 0/1 index.

Invariant: obs perspective = `frame.actor`; heads address `frame.actor`'s own roles.

#### 1.3.6.2. Reveal effects

`ShuffleOpponentSupporter` / `DiscardOpponentSupporter` (Silver, Mega Absol Ex) require the actor to
point at a card in the opponent's hand — they reveal it, so this is a learned head
(`REVEALED_HAND_PTR`), not a random resolution. The reveal taxonomy, the belief overlay it reads
(presence- vs position-revealed, [src/belief/](src/belief/)) and its invalidation on shuffle are the
information-state component: `NOTES.md`.

The head's domain is the frame's own enumerated set, in engine order, which is the revealed subset
(the engine only ever offers the opponent's Supporters here). That domain never had to move: what
was missing was the Part 2 side, and the belief render supplies it — the candidate's reference rows
resolve to the opponent-hand token of the very card it points at, so the head now chooses between
cards the encoder can tell apart. In spectator mode there is no such token and the candidate keeps
an empty encoding, which degrades the choice and never its legality.

#### 1.3.6.3. Forced, Noop, internal

- `len(candidates) == 1` (including `end_turn_pending` and some setup steps): auto-resolve without a
  network forward; the mask has exactly one entry. FORCED wins over the other three regimes.
- `Noop` is a real choice ("say no") and stays a candidate whenever the engine offers it.
- `DrawCard` (automatic since commit 2d9244a) is engine-internal: never a candidate, never a learned
  choice. It still reaches the mask as the single entry of a FORCED frame, so the dispatcher has one
  uniform way to resolve any frame.

### 1.3.7. Invariants & falsifiable tests

With `E = generate_possible_actions(state)`:

1. **Bijection.** `unproject(mask) ≡ E` as sets, for every reachable state (property test over
   random-player rollouts), modulo reprint identity: §1.2.2 collapses complete reprints onto one
   `card_id` row, so a pointer head cannot separate two printings of one card and the equality is
   stated on `canonical_action(·)`. For the same reason the mapping is a surjection where a player
   holds several copies of a card: each copy is its own token row, every copy's bit is legal, and
   they resolve to the same play.
2. **Non-empty.** `|E| ≥ 1` always (at minimum `EndTurn` or a forced frame).
3. **Round-trip.** The selected `(head, args)` maps back to a `SimpleAction ∈ E` that `apply_action`
   accepts without panic.
4. **Regime exclusivity.** Exactly one of SETUP / STACK / FREE_PLAY / FORCED is active.
5. **Perspective.** For every STACK frame, obs perspective = `frame.actor` (§1.3.6.1).
6. **Family consistency.** `action_type[f]` is set iff family `f`'s argument head carries a bit — the
   family head is the argument heads' own emptiness, never a second opinion on legality.
7. **No free-play demotion.** Over ordinary play, no SETUP / FREE_PLAY candidate ever falls back to
   `CANDIDATE_PTR`. A regression here means a family head quietly stopped factorizing.

### 1.3.8. Egocentric shapes

Self-only heads point into self-scoped slices of the Part 2 encoder banks, not the full mixed banks —
this is the concrete halving the egocentric principle buys. The encoder is unchanged (it still
attends over both boards); only the head pointer domains shrink. A 20-card deck bounds a player's own
tokens: `POKEMON_SELF ≤ 20`, `TRAINER_SELF ≤ 20`, `ATTACK_SELF ≤ 16` (4 board Pokémon × 2, plus
Time-Recall borrows). No head carries a player-index dimension; opp-role heads use a 4-slot board
index, never an opp token bank.

| Head | Shape | Role |
| --- | --- | --- |
| `action_type` | 10 | self |
| `PLACE` / `EVOLVE` | POKEMON_SELF × 4 | self (hand Pokémon → slot) |
| `ATTACK` | ATTACK_SELF | self |
| `PLAY_TRAINER` | TRAINER_SELF | self |
| `HAND_PTR` (`CommunicatePokemon`) | POKEMON_SELF | self |
| `ATTACH_ENERGY` / `USE_ABILITY` / `DISCARD_FOSSIL` / `SLOT_PTR`(self) | 4 | self board |
| `RETREAT` | 3 | self bench |
| `SLOT_PAIR` | 4 × 4 | self board |
| `SLOT_PTR`(opp) | 4 | opp board (cross-target only: Cyrus, Field Blower, gust) |
| `STATUS_CAT` | 5 | — |
| `USE_STADIUM` / `END_TURN` | 1 | self |
| `REVEALED_HAND_PTR` | K ≤ 20 | opp revealed set |
| `CANDIDATE_PTR` | K ≤ 512 | per-frame candidate list |

Against a naive both-players head the Pokémon and attack pointer dims halve (40 → 20, 32 → 16) and no
head spans both sides.

A `PLACE` / `EVOLVE` index is `row × 4 + slot`, where `row` is a position in the allied subsequence of
the Part 2 Pokémon bank, so a head index names an encoder row directly with no player dimension to
strip. `EVOLVE` reads the deck slice instead of the hand slice for a `from_deck` evolution: one head,
two source zones.

The two per-frame heads are dynamic, so like the Part 2 banks they are padded to a cap and assert on
overflow at flattening time only — `project` itself is unbounded, so an oversized frame is still
projected correctly and only the wire form complains. Heads are concatenated in the implementation's
fixed `HEADS` wire order — `action_type`, `PLACE`, `EVOLVE`, `ATTACH_ENERGY`, `RETREAT`, `ATTACK`,
`USE_ABILITY`, `PLAY_TRAINER`, `USE_STADIUM`, `END_TURN`, `DISCARD_FOSSIL`, `SLOT_PTR`(self),
`SLOT_PTR`(opp), `SLOT_PAIR`, `HAND_PTR`, `STATUS_CAT`, `REVEALED_HAND_PTR`, `CANDIDATE_PTR` — which
is not the table order above. The flat mask is 804 bits
(`10 + 80 + 80 + 4 + 3 + 16 + 4 + 20 + 1 + 1 + 4 + 4 + 4 + 16 + 20 + 5 + 20 + 512`), of which 512 are
the candidate pointer; measured over the deck shelf, real frames stay two orders of magnitude below
that cap (widest observed: 20 candidates, 2 revealed).

## 1.4. Part 4 — Model (v1)

Consumes the Part 2 observation, drives the Part 3 heads. One shared encoder; heads read rows of its
output `H : [N × d_model]`, `N ≤ 133`. No centralized critic — value and policy share the same
imperfect-information observation. The deckbuilder (Part 6) reuses the encoder on frozen embeddings
and emits 0 History tokens.

### 1.4.1. Encoder

- **Five input projections**, one per family, each a single linear `width_f → d_model`; a learned
  token-type embedding tags the family:

  | Family | Width |
  | --- | --- |
  | Global | 170 |
  | Pokémon | 854 |
  | Trainer | 285 |
  | Attack | 195 |
  | History | 82 |

- **Global token**: the Part 2 global vector projected to a token; its output row `H[global]` is the
  state summary for the value and nullary heads. `d_model ≥ 192`, since Pokémon carries `3 × d_id`.
- **History fused in-encoder**, not a side stream: it attends jointly with entities, bidirectional
  and with no causal mask. Only History carries a recency signal; the four entity families are
  permutation-invariant.
- Pre-LN transformer blocks, MHA + FFN, padding mask on unused slots.

### 1.4.2. Heads

Egocentric and self-scoped (§1.3.8), masked by the Part 3 mask.

- **Pointer heads** (`PLACE`, `EVOLVE`, `ATTACK`, `PLAY_TRAINER`, `HAND_PTR`, slot / pair,
  `CANDIDATE_PTR`, …): `logit_i = MLP(H[token_i])`, softmax over the masked candidate rows.
- **Nullary / global** (`action_type`, `END_TURN`, `USE_STADIUM`, `STATUS_CAT`): `MLP(H[global])`.
- **`CANDIDATE_PTR` cross-attention** (`[model] candidate_cross_attention`, off by default): one
  single-headed `d_model`-wide attention from the candidate descriptor over the whole encoded
  sequence, padded keys masked out of the softmax. Concatenated to §1.3.5's mean, never in place of
  it — the mean identifies the candidate, the attention weighs the rest of the board. The flag adds
  parameters, so a model built with it on cannot read a checkpoint written with it off.
- **Value** (the only use of pooling, value-only):
  `v = MLP( AttnPool₁(H) ⊕ H[global] ) ∈ [−1, 1]`, `AttnPool₁` being one learned query over all rows
  (History included). The value-loss coefficient is scheduled in the `.toml` so it does not distort
  the shared representation.

### 1.4.3. Sizes and measured budget

v1 defaults, `.toml`-tunable:

- `d_model` = 192, blocks = 2, heads = 6 (32 per head), FFN = 384 (×2)
- `d_id` = 64 (3 concatenated for Pokémon = 192, so `d_model` sits exactly on its floor)
- input projections = 5 × linear; value attention-pool = 1 query
- ≈ 1.43 M trainable params (encoder blocks 0.59 M, input projections 0.31 M, the rest heads and
  embedding residuals — the heavy per-card content lives in the frozen tables)
- frozen static tables ≈ 12 MB, gather-only, not parameters; trainable weights ≈ 5.7 MB in f32

A configured model's own split, and the end-to-end decision throughput of every seat it can be
compared against, are regenerated into `PERFORMANCE.md` by
[examples/benchmark_players.rs](examples/benchmark_players.rs) — measured in decisions per second,
never games per second, since a game is not a fixed amount of work and the seats differ sixfold in
how many decisions they spend on one. The figures below are the component budget that explains it.

The first-pass defaults (`d_model` 256, 4 blocks, FFN 1024, ≈ 4.4 M params) measured ≈ 0.95 GFLOP per
forward — no cheaper to run than the `deprecated` branch's 15 M-parameter model, because `N` went
from 40 to 133 tokens and cancelled the width reduction. Parameters are the wrong meter here; FLOPs
are, and they are dominated by `N·d²` and `N·d·d_ff`. Per-block MACs split 61 % FFN / 31 % q,k,v,o /
8 % attention proper, so `d_ff` was cut first (×4 → ×2) and blocks second: ≈ 0.21 GFLOP, 4.5× fewer
FLOPs for 3.1× fewer parameters. Derivation, rejected alternatives and the `SEQ_LEN` bucketing
option: `NOTES.md`.

**Measured inference budget** (Burn 0.21, release; RTX 3050 Laptop 4 GB / single-thread NdArray CPU).
The original "`N ≤ 133` → sub-ms forward, CPU-viable" claim was falsified: sub-ms on one forward needs
a ~1 TFLOPS effective stream, which no CPU sustains and the GPU reaches only batched.

| Backend | batch 1 | saturated forward | fwd + bwd (training proxy) |
| --- | --- | --- | --- |
| CPU NdArray, 1 thread | ≈ 14 ms | ≈ 14.5 ms/sample, flat b1→b256 (GEMM-bound) | — |
| CUDA 13.3 | ≈ 16 ms | ≈ 386 µs/sample (≈ 2590/s, b256); 442 at b128, 605 at b64 | best ≈ 1.71 ms/sample (≈ 586/s) at b256; b512 collapses ×7 |

wgpu is not re-measured at this size. Measurement prerequisites and backend caveats: `NOTES.md`.

**Per-decision budget beyond the forward.** The sweeps above assemble `ModelInput` once, outside the
timed loop. Self-play pays more per decision (8 random rollouts, 256 decision points, CUDA at
batch 64):

| Stage | Cost / decision | Note |
| --- | --- | --- |
| `generate_possible_actions` | ≈ 7 µs | shared — the engine calls it during `play_tick` anyway |
| `get_observation` | ≈ 290 µs | dominated by the §1.2.5 threat matrix |
| `project` (Part 3 mask) | ≈ 12 µs | |
| `ModelInput::from_points` | ≈ 17 µs | wire flattening + host→device |
| forward + readback | ≈ 553 µs | CUDA, batch 64 |
| **RL cost** | **≈ 870 µs** | ≈ 1150 decisions/s |

Policy decisions per game: ≈ 121, measured against an untrained policy and therefore an upper bound.
A lightly trained policy should end turns sooner and land nearer ≈ 40; the figure is kept as the
illustrative worst case, and every games/s number below is the correspondingly pessimistic one
(multiply by ~3 for the trained regime).

Consequences for Part 5 sizing on this hardware class:

- **Inference must be batched across the vectorized envs** (§1.5.5) on a GPU backend: saturated CUDA
  is ≈ 40× a lone forward and ≈ 38× the CPU.
- **The §1.1.5 budget is model-bound**: at ≈ 1150 decisions/s end-to-end and ≈ 121 decisions per
  game, one model-driven seat reaches ≈ 9.5 games/s on a 4 GB laptop GPU (≈ 29 at the trained-policy
  decision count) — halve it again if both seats run the model rather than a cheap panel opponent.
- **`get_observation` is the next target, not the encoder.** It was ≈ 18 % of the per-decision cost
  against the first-pass encoder; at the revised size it is ≈ 33 %. Shrinking the encoder further has
  sharply diminishing returns until the §1.2.9 damage estimator behind the threat matrix is
  optimized.
- **The "training batch ≤ 64" rule was an artifact of the oversized model.** At the revised size the
  knee moved out by 4×: CUDA training peaks at batch 256 (≈ 1.71 ms/sample) and only collapses at 512
  (≈ ×7). What binds is the peak VRAM reservation across phases of the single learner process —
  inference and fwd+bwd share one pool — and exceeding it degrades inference ≈ 25× without recovering
  (mechanism in `NOTES.md`).
- **Measured VRAM of the whole §1.5.5 loop** (RTX 3050 4 GB, peak resident over a 3-batch run,
  `nvidia-smi` sampled throughout). Throughput is deliberately absent from this table: the loop's
  games/s times collection only, so `micro_batch` is not in that path, and three batches is far too
  few to compare env counts (§1.5.5's sweep is the throughput measurement).

  | envs | `micro_batch` | peak VRAM |
  | --- | --- | --- |
  | 64 | 64 | **1 027 MiB** |
  | 8 | 128 | 1 891 MiB |
  | 16 | 128 | 1 891 MiB |
  | 32 | 128 | 2 531 MiB |
  | 64 | 128 | 2 563 MiB |
  | 64 | 256 | 3 907 MiB |

  `micro_batch` owns the budget and `envs` barely moves it — ×8 on the env count costs 35 %, ×2 on
  the micro-batch costs 150 % — because inference retains no activation graph and the backward does.
  The peak fits ≈ 260 MiB resident + 12 MiB per micro-batch frame. The isolated knee of 256 is
  therefore not usable with a rollout in the same process: it takes 95 % of the card.
- **Sizing for the multi-model future.** Part 5 keeps the BR and the §1.5.1 magnet resident, and
  Part 6 adds the deckbuilder, so the run default is `envs = 64`, `micro_batch = 64` — 1 GB, leaving
  ~3 GB for the other two. `micro_batch` is a pure VRAM knob (the step is one step whatever it is set
  to), so this trades only the training step's per-sample efficiency, never the algorithm and never
  collection throughput. That per-sample cost at 64 vs 128 is not measured.

**Attention is ablatable at inference** ([src/rl/model/encoder.rs](src/rl/model/encoder.rs)):
`AttentionAblation::UniformPattern` zeroes query and key, so the softmax is uniform over the real
tokens; `Silent` zeroes the output projection, so the sublayer writes nothing. Weight surgery rather
than a branch in the forward, both falling out of the arithmetic exactly, and the §1.5.6 read-out is
what verifies each does what it claims. `examples/block_ablation.rs` puts an ablated block through
the evaluation and `examples/head_to_head.rs` against the unablated checkpoint; both bound the
trained model only, since its other parameters were fitted with the pattern in place. Measurements:
`NOTES.md`.

## 1.5. Part 5 — Training loop (v1)

Player pretraining only (the deckbuilder is Part 6). Everything below is a v1 default in the run's
`.toml`. Each subsection states the specification, then the build state; the arguments behind the
decisions are in `NOTES.md`.

### 1.5.1. Learning algorithm

Two networks:

- **Best-response (BR)** — the Part 4 player, trained by strict single-step MMD: one mirror-descent
  proximal step per on-policy batch, no PPO clip and no multi-epoch. Per-step objective =
  policy-gradient with GAE advantage (shared value baseline) `+ η·KL(π_BR ‖ magnet)` `+ τ·entropy`.
  Terminal reward only (win +1 / loss −1 / tie 0), `γ = 1` (finite horizon), GAE `λ = 0.95`.
- **Magnet (average clone)** — a separate off-policy network, trained by supervised behavioral
  cloning on a reservoir buffer of BR's past `(state, action)`, approximating the NFSP time-average
  policy. Its objective is cross-entropy on the taken bit and nothing else: no advantage, no return,
  no value target. Seeded from the heuristic anchor (§1.1.3); it is the KL target above.

Division of labour: PFSP picks the opponent, MMD does the update, the average clone carries the
equilibrium. A trajectory is the agent's own decision frames (off-turn reactive frames included,
§1.3.6.1), reward propagated from the terminal. No centralized critic.

Fixed decisions the objective above does not state:

- **"Next state" is the learner's next decision frame**, not the next engine tick — the opponent
  moves and whole turns pass in between. That is what makes "the agent's own decision frames" a
  well-formed MDP.
- **Decay is a loss term, not the optimizer's.** It applies to the embedding residuals only
  (§1.5.5), so AdamW runs at `weight_decay = 0` and the residual L2 is added explicitly.
- **Advantages are normalized over the batch, not per episode.**
- **The KL and the entropy are over the argument bits, not the whole policy vector**: the
  `ACTION_TYPE` block carries the induced family marginals (§1.3.4), which are sums of those same
  argument bits, so including it would count part of the distribution twice. Any per-bit term meant
  to be over the distribution drops that block.
- **The reservoir is uniform over the whole stream** (Vitter's algorithm R), not over recent frames:
  "time-average policy" is the specification, and a ring buffer would make the magnet a lagged copy
  of the current BR.
- **The heuristic anchor is a weighted mixture `Σ wᵢ·πᵢ`, consulted per decision**, and the clone
  fits that mixture. Strength is not what a magnet seed wants — the seed's job is support and
  coverage. Filling the 20 000-frame buffer measures ≈ 5 s of `w`, ≈ 20 s of `v`, ≈ 113 s of `e2`,
  once, before batch 0.
- **The buffer is checkpointed on a stop or a pause, not on the autosave cadence.** `seen` travels
  with the residents, because it is the acceptance denominator: a buffer restored without it re-takes
  the average over the post-resume stream only, which is the lagged copy the point above forbids.
  Measured over `long_v3` (4 restarts) and `long_v4` (1), a reset drops `loss/kl_magnet` ~0.13 → 0.09
  in 80 batches and makes the series incomparable across runs. A crash still resumes from an empty
  buffer, and a resumed run then holds the SL step until it refills past its fill floor.
- **A batch larger than the §1.4.3 knee is split for the forward and accumulated**, staying one step
  for the optimizer: "one step per batch" is an algorithm property, the split is a VRAM one. The
  magnet's forward for the KL runs on the same micro-batch input, read through to the non-autodiff
  backend — one forward, no activations.

**Build state.** Both halves are built: GAE in [src/rl/train/gae.rs](src/rl/train/gae.rs), the step
in [src/rl/train/update.rs](src/rl/train/update.rs), the magnet in
[src/rl/train/magnet.rs](src/rl/train/magnet.rs) over the reservoir of
[src/rl/train/reservoir.rs](src/rl/train/reservoir.rs), its heuristic seed in
[src/rl/train/anchor.rs](src/rl/train/anchor.rs), the loop in
[examples/train_player.rs](examples/train_player.rs). `[magnet] enabled = false` drops the KL and
leaves the loss a plain policy gradient — §1.1.6 stage 1's ablation, kept because MMD without the
magnet is not MMD and telling the two runs apart requires being able to run both. None of the
magnet's constants (η, capacity, seed length) is calibrated yet; see `NOTES.md`.

### 1.5.2. Opponents — PFSP + continuous panel

- **Self-play pool** = frozen BR checkpoints, containerized. Sampling: PFSP (prioritized toward ~50 %
  opponents) + uniform floor + minimum games per performer.
- **Frozen heuristic panel** (random, weighted-random, expectiminimax from the repo) sits
  continuously in the opponent mix: non-self playstyles (anti self-play overfit) and the monitoring
  probe. Training against it makes winrate-vs-panel a saturation signal rather than held-out
  generalization — self-play elo is the cleaner generalization curve.
- **Pool membership** is `X` best slots + `Y` history slots over a permanent panel of heuristics and
  curriculum-owned baked models. Defaults are 2 + 6, not an even split: the best slots converge on
  the newest clones, so half the pool would be "the last few versions of me", which is the plain
  self-play the history slots exist to break up.

Fixed decisions:

- **Selection ranks on `rating − 2·deviation`, never on the window's winrate**: PFSP samples
  non-uniformly and endogenously, so a member's winrate is noisiest exactly when its eviction is
  decided.
- **The rating scale needs a fixed origin, and it is `er`** — one permanent member never updated,
  the strongest non-model heuristic rather than `r`. A fixed point only fixes something if it plays,
  which is what the uniform floor is for.
- **Only the best-response drifts.** Glicko-2's `φ* = √(φ² + σ²)` models strength changing between
  periods; frozen weights do not change, so a pool member's deviation is not inflated for sitting a
  period out.
- **The rating period is the refresh window.** Glicko-2 is not incremental, so no rating reflects the
  last rollout: decision and measurement close on the same boundary.
- **Cloning and refreshing run on separate cadences** — one clone every `clone_every` batches, slots
  re-decided every `refresh_every`. Tied together, the pool would need `(X+Y)` refreshes to fill
  (320 batches at the defaults, which a 200-batch run never reaches); decoupled, it is full by
  batch 80.
- **Grace is a game count with a batch cap.** A slice of the sampling mass is reserved for members
  below `grace_games`, and `grace_batches` only bounds how long the reservation may hold. Below the
  floor a member is also immune to eviction, since a clone enters holding its parent's stale rating.
- **Envs are cut into contiguous groups**, each facing one opponent for a collection, with
  `concurrent_opponents` setting how many are in flight. That knob trades coverage, not batch width:
  an env holds one pending decision at a time and the actor alternates, so a model on the far seat
  halves the learner's own batch whatever the grouping — the fix for that is more envs, which §1.4.3
  measures as nearly free in VRAM. Games in flight keep the opponent they started with. The
  opponent's decision frames are destroyed, not merely unused: they must reach neither the on-policy
  batch nor the reservoir.
- **Sizes may differ between pool members; the observation schema may not.** `meta.toml`'s `[model]`
  table is read, never compared; what is compared is `rl::schema_fingerprint()`, a digest of every
  observation and mask width mixed with a hand-bumped `OBS_SCHEMA_VERSION` for changes of meaning
  that no width would show. A mismatch is fatal, and so is a missing panel member. `models/` is
  tracked, weights included — a few MB against a curriculum that is otherwise unreproducible.
- **Eviction frees a slot; it never deletes weights.** The history slots draw from the archive, and a
  checkpoint drawn back in resumes from its old rating rather than from 1500. At ≈ 5.7 MB a
  checkpoint the default run's archive is ≈ 114 MB; it grows as `batches / clone_every` and not with
  the pool, so a long run raises the cadences rather than pruning. The historical draw is
  `.toml`-selectable (uniform / per-age-octave / bounded-recent), the three covering different
  timescales; bounded-recent pulls the same way the best slots already do.

**Build state.** Built and wired, behind `[pool] enabled` — off by default, because a model on the
far seat roughly doubles the forwards per game and `[rollout] opponents` is what a config written
before the pool described. Glicko-2 in [src/rl/train/rating.rs](src/rl/train/rating.rs); slots,
archive and PFSP weights in [src/rl/train/pool.rs](src/rl/train/pool.rs); the on-disk form of a
frozen model in [src/rl/train/baked.rs](src/rl/train/baked.rs) (`models/<name>/` = `weights.mpk` +
`meta.toml`, written by `examples/bake_model.rs`); the model-driven opponent seat in
[src/rl/train/opponent.rs](src/rl/train/opponent.rs) and
[src/rl/train/rollout.rs](src/rl/train/rollout.rs); the loop binding in
[src/rl/train/panel.rs](src/rl/train/panel.rs). The panel's state travels inside the hot checkpoint
and its per-member table is `runs/<name>/pool/table.jsonl`. Every cadence is sized for
`config/default.toml`'s 200-batch run and scales with `[rollout] batches`.

### 1.5.3. Data — self-play, two deck DBs

No static dataset; experience is generated on the fly. The "dataset" is the deck sampler.

- **Two DBs**, compiled to `decks/` by `auxiliaries/build_deck_dbs.py` and read by
  [src/rl/train/deck_db.rs](src/rl/train/deck_db.rs): `meta` (120 444 decks, archetype = the
  Limitless deck label, 8 849 of them) and `tutorial` (262 decks, archetype = the difficulty tier —
  beginner / intermediate / advanced / expert, plus the untiered rental bucket). Only decks the
  engine can run are compiled in; eight preset decks whose upstream energy list was read off the
  headline Pokémon's type are rewritten, not dropped (Dragon runs on Water + Lightning; Colorless
  drops out beside a real energy, becoming Water alone).
- **The meta DB is tournament data**, built by `auxiliaries/build_meta_decks.py` from the Limitless
  API: every list played in a POCKET event before B4, deduplicated on its card multiset. It replaces
  the tier-list archive, whose lists were generated rather than played. The cutoff is B4 because
  `database.json` stops at B3b.
- **Per game** ([src/rl/train/sampler.rs](src/rl/train/sampler.rs)): DB draw plus forced mirror (same
  archetype) and pure-mirror (same deck) quotas. The DB draw is deck-uniform, so a meta archetype's
  weight is how many distinct lists of it were brought to a tournament — meta-tiering read off real
  play, needing no hand-written tier table. It restricts to a subset of archetypes, which is how a
  run draws one tutorial tier instead of mixing beginner with expert. §1.5.4's curriculum owns both
  the DB and that subset per stage, one `DeckSampler` rebuilt on every transition.
- **A draw may run over several DBs at once**, each an explicit share of it (`mix` in the `.toml`,
  `DeckSource` in code). The share is the one place deck-uniformity is overridden deliberately:
  `tutorial` concatenated into `meta` would be a fraction of a percent of the draw and effectively
  absent — less of one every time `meta` grows — when the reason to mix it in is that its beginner
  tier holds the weak decks `meta` has none of. Deck-uniformity still governs *within* a source.
- **The source is rolled per seat, not per game.** Once per game would give `meta` self-play and
  `tutorial` self-play in proportion and never one across the table from the other, which is the
  matchup the mix exists to buy — §1.1.3 asks for a result attributable to the deck, and that is
  not learnable from a distribution where both seats are always about as strong. The cross-DB rate
  is therefore `2·p·q`: shares of 0.90/0.10 give 18 % mixed games, not 10 %. The two mirror
  quotas are the exception and stay inside one source, an archetype being a per-DB label. A
  single-source sampler consumes no randomness on the source roll, so adding the mix leaves every
  existing run's deck stream bit-identical.
- **The uniform-deck quota** (§1.1.3, anti-collapse) is not in this sampler: a legal-random-deck
  generator is Part 6 work, and the exotic-pair coverage it would buy the §1.5.7 harvest is §1.6.5's
  concern. Nothing fixes that coverage today.
- **Buffers**: transient per-iteration on-policy rollout (BR + value); persistent reservoir (magnet
  clone, [src/rl/train/reservoir.rs](src/rl/train/reservoir.rs)). The reservoir's own deck draw is
  the seed's ([src/rl/train/anchor.rs](src/rl/train/anchor.rs)), cloned off this sampler so the
  magnet is seeded on the distribution the run trains over.

### 1.5.4. Curriculum & stop

- **Stages** in `.toml`: `(deck DB, opponent set, magnet source)`. Advance when winrate ≥ 70 % vs the
  current anchor over a minimum game count.
- **Global stop** (§1.1.3): step budget or winrate-vs-panel plateau (Δ < ε over K consecutive
  evaluations). No absolute threshold.

Fixed decisions:

- **"The current anchor" is the worst one.** Over a panel the floor reads `min` over anchors, never
  the mean: a 95 % against random must not pay for a 45 % against weighted.
- **Screened cheaply, confirmed independently.** The held-out evaluation fires when the free rolling
  window (§1.5.6) says the floor may have been reached, not on a fixed cadence. Triggering on a
  threshold crossing biases the screen upward, but the evaluation it fires is an independent sample
  and carries none of that bias. What the screen does risk is testing until something passes, so it
  is guarded by a hold (the floor must survive K consecutive batches over a full window) and a
  cooldown (a minimum gap between evaluations).
- **The screen lags**: the window averages K model versions, so it under-reports a fast-improving
  agent and fires late. That is the safe direction, and the window length is the only dial on it.
- **The screen reads the stage's anchors, not the whole window.** With the pool on, the window also
  holds the learner's own frozen clones, whose winrate sits near 50 % by construction; taking `min`
  over every label would make a clone the worst one for the run's life and the gate would never arm.
- **The plateau reads the free `panel/window` signal, not the held-out evaluation**, sampled once per
  window turnover rather than per batch (consecutive batches of one rolling window overlap in almost
  all their games, so a per-batch sample would trip on autocorrelation). Feeding the plateau off the
  confirming evaluation would couple it to a trigger that only ever arms once the window has already
  cleared 70 %, so a run stuck at 40 % could never be stopped by it.
- **The pool's permanent membership is retargetable per stage; its mechanics are not.**
  `Pool::retarget` swaps `anchors` / `baked` while the Glicko table, the clone archive and the active
  slots carry over. `best_slots`, `refresh_every`, `tau`, `models_root`, … stay run-global — the
  spec's triple names membership, not mechanics. The pinned origin never moves either: every stage's
  anchors must include `[pool] pinned`, checked once at config load.
- **The magnet's reservoir reseed is partial, not total.** `evict_fraction` of the buffer is evicted
  at random before the new stage's heuristic mixture tops the freed capacity back up.
- **Reaching the last stage's floor is a milestone, not a stop condition.** Only the plateau or the
  step budget end the run.
- **An empty `curriculum.stages` is its own mode, not "stage 0."** A non-empty list supersedes the
  run's flat `[decks]` / `[eval]` / `[pool] anchors,baked` / `[magnet.seed]` fields rather than
  merging with them; those fields stay syntactically required but are read only to build the very
  first `Collector` / `Panel`, which the starting stage retargets immediately.

**Build state.** The stage/plateau state machine is in
[src/rl/train/curriculum.rs](src/rl/train/curriculum.rs) (`Curriculum`, `Stage`, `StagePanel`), with
`Pool::retarget` / `Panel::retarget`, `Collector::set_sampler`
([rollout.rs](src/rl/train/rollout.rs)), `Reservoir::evict_fraction`
([reservoir.rs](src/rl/train/reservoir.rs)), `Harvest::set_sampling`
([harvest.rs](src/rl/train/harvest.rs)), the `[curriculum]` / `[[curriculum.stages]]` `.toml` shape
and `TrainConfig::curriculum_stages` ([config.rs](src/rl/train/config.rs)), wired into
[examples/train_player.rs](examples/train_player.rs) behind `curriculum.stages` being non-empty.
Empty — the default `config/default.toml` still ships — leaves every pre-§1.5.4 code path untouched.
`default.toml` carries a real two-stage run over the `tutorial` tiers as a commented-out
`[[curriculum.stages]]` block: the file doubles as the schema's reference, every field appearing in
it live or commented, and the parser (`deny_unknown_fields`) refuses a field it does not show.

### 1.5.5. Infrastructure

- **Vectorized Rust envs**, parallel, batched inference (self = current BR, opp = sampled frozen
  policy in its container). [src/rl/env.rs](src/rl/env.rs) yields the frames,
  [src/rl/train/rollout.rs](src/rl/train/rollout.rs) turns them into whole episodes.
- **Batching is a GPU-only win, and it decides the machine.** Rollout throughput measured end-to-end
  at the §1.4.3 sizes, against the panel, sweeping the env count:

  | envs | CPU (`NdArray`) | CUDA (RTX 3050, 4 GB) |
  | --- | --- | --- |
  | 1 | 1.24 games/s | 0.62 games/s |
  | 8 | 1.34 | 4.61 |
  | 32 | 1.31 | 14.36 |
  | 64 | 1.18 | **23.03** |

  CPU is flat in the env count: `NdArray` has no fixed per-call latency to amortize, so the batch
  buys nothing and 10⁶ games costs ≈ 9 days. CUDA scales ×37 over the same sweep — batch 1 is worse
  than CPU (the ≈ 16 ms lone forward of §1.4.3 is pure launch latency) and only the batch pays it
  back. At 64 envs, 10⁶ games ≈ 12 h. The sweep has not plateaued at 64; how far it goes is a VRAM
  question, not a scaling one.
- **Seeding**: master seed → per-env child seeds (splittable PCG / SplitMix), fully reproducible.
  Three consumers draw from it, each on its own stream keyed by a counter rather than a shared
  generator: env seeds by game index, deck / opponent draws by game index, action sampling by batch
  index. Sharing one generator would make each state a function of how many draws the others had
  made, unreconstructable at resume. The master seed also seeds parameter init.
- **Checkpoints** ([src/rl/train/checkpoint.rs](src/rl/train/checkpoint.rs)), two kinds, answering
  different questions. Cold = weights alone, at a conventional stop; this is what §1.5.2's pool
  freezes, and an opponent has no use for optimizer state. Hot = weights + AdamW's moments + the two
  counters, on a batch cadence plus a Ctrl-C latch. There is no gradient state to keep, and games in
  flight are dropped rather than serialized — `γ = 1` gives a truncated trajectory no return.
  Consequence: a resumed run is not the uninterrupted run. Two resumes from one checkpoint agree; the
  batch after a resume does not match the batch it replaces. A crash writes nothing, so the hazard
  the format guards is the torn write — a marker file published last, and a checkpoint is invisible
  until it exists.
- **A pause is not a stop** ([src/rl/train/pause.rs](src/rl/train/pause.rs)). `p` between batches
  holds the process — model, optimizer, pool and envs resident, envs therefore *not* dropped — and
  takes a hot checkpoint on the way in against the machine dying during it. That checkpoint carries
  the magnet's reservoir, as a stop's does and the rolling autosave's does not: a pause is where a
  user may decide to kill the process after all. The same key resumes;
  the wait is outside the run clock, so a pause costs the ETA nothing. A Ctrl-C during one is still
  a stop. Keyboard-only, on a run whose stdin is a terminal.
- **Synchronous single learner**: collect multi-env batch → GAE → one MMD step + one magnet SL step →
  repeat. Collection keeps envs across calls, so the frame budget is a floor on finished frames.
- **Engine panics are caught, not fatal** ([src/rl/recover.rs](src/rl/recover.rs)). The simulator
  asserts its invariants with `expect`, and the rollout is the only caller that reaches states
  nothing else does — a policy starting uniform over the legal set plays lines no heuristic player
  produces. A panic raised while advancing a game is caught per env: that game is thrown away with
  its frames, and the slot takes a fresh game. This is not error handling for the engine — it is what
  stops one bad game in 10⁶ from ending a 12-hour run, and the caught panic is still a bug. The
  state, the action being applied, the location and a forced backtrace go to `runs/<name>/crashes/`
  before the slot is reused ([src/rl/train/crash.rs](src/rl/train/crash.rs)), since the rollout is the
  only place that state ever existed; [examples/replay_crash.rs](examples/replay_crash.rs) drives a
  dump again rather than only reading it. Two `.toml` bounds: a dump cap (first occurrences are the
  informative ones) and a per-collection panic limit, guarding the reproducible crash that would
  otherwise respawn into itself forever. A rejected mask bit stays fatal (§1.3.7 invariant 3): that
  one is the caller violating a contract, not the engine giving up.
- **Layout**: `config/*.toml` (sources); `runs/<name>/` = cloned `.toml` + `checkpoints/` + `logs/` +
  `eval/` (§1.5.6's two winrate measurements, one JSONL record per measurement, tagged by source —
  the nested per-anchor counts the flat metrics line cannot carry), laid out by
  [src/rl/train/run_dir.rs](src/rl/train/run_dir.rs), plus `harvest/` (§1.5.7) and `crashes/`, both
  created lazily so their existence is the signal that the run logged labels or hit a panic. The
  clone is what makes a run reproducible, so the `.toml` has to be the whole run: the Part 4 sizes
  live in it, not in Rust. A run is identified by its config — the name is a `.toml` field with no
  CLI override, and `create` refuses an existing directory rather than interleaving two runs'
  checkpoints.
- **Optimizer**: AdamW, `lr = 3e-4` (short warmup + constant), grad-clip `0.5`, weight decay on the
  player embedding residuals (§1.2.2).
- **Schedules** ([src/rl/train/schedule.rs](src/rl/train/schedule.rs)). "Short warmup + constant" is
  one shape among many and the right one is not knowable before the run, so a coefficient is a
  sequence of phases: `lr`, `τ`, `c_v` and the residual decay each take either a number or a
  `{ start, phases }` table, and the TOML type decides which.
  - **Durations are absolute or relative, read off the type**: an integer is batches, `"5%"` a
    fraction of `batches`, `"rest"` what remains. Relative survives changing the run length; absolute
    is what a warmup usually means. Neither is right in general, so the file says which and the
    parser does not guess.
  - **A phase starts where the previous one ended**, so only destinations are written: inserting a
    phase cannot leave a discontinuity, and omitting `to` means hold.
  - **Past its last phase a schedule holds.** The resolved boundaries are printed at startup, so a
    mismatch is visible before the run rather than after.
  - **Grad-clip is not schedulable**: Burn takes it when the optimizer is built, and a clip is a
    stability bound rather than a term one anneals.

  Coefficients are evaluated once per batch, not per micro-batch — one moving between micro-batches
  would make the accumulated gradient the gradient of no single loss — and logged as `sched/*`.

### 1.5.6. Logging

- **Standard**: winrate vs panel, self-play elo, losses (policy / value / KL-magnet / entropy /
  magnet cloning), grad-norm, policy entropy, KL-to-magnet, reservoir fill, games/s, elapsed training
  time, dropped games, curriculum stage.
- **Diagnostic** (pathology detection and real training consequences): per-head action-type
  distribution, turns per episode, per-head entropy, value calibration (predicted vs realized
  outcome), per-head forced rate, legal-mask-size distribution, encoder attention read-out.

Both blocks are built, `elo/*` (§1.5.2) and `curriculum/*` (§1.5.4) included, each emitted only by a
run whose section is on — absent rather than zero, since a flat zero curve reads as a measurement.
The same rule runs inside the magnetic block: `loss/kl_magnet` and `sched/eta` are emitted only by a
run that has a magnet, and the `magnet/*` series have gaps on the batches the SL step was held below
its fill floor. `η` is logged beside the KL rather than the product of the two, which cannot tell a
divergence that collapsed from a coefficient that was annealed. Value calibration is in from the
start, early for a diagnostic, because it is what separates "the agent is not learning" from "the
critic is flat".

**Two readings no loss curve gives.** `loss/value` is an MSE against the λ-return, a target built
from the critic's own predictions, so it has neither a scale nor a meaning independent of the
bootstrap: `value/explained` divides the scale out, `value/mc_explained` and `value/mc_abs_error`
measure against the game's actual result, and `value/calibration/*` bucket predicted against
realized — an aggregate error cannot tell a critic near the Bayes floor of an imperfect-information
game from one predicting the batch mean. The mirror-image trap sits on the other side of the loss: a
normalized advantage puts `loss/policy` near zero however large its gradient, so the terms' loss
magnitudes cannot say which one shapes the shared trunk. `optim/grad_trunk/*` — `|c| · ‖∇L‖` per
term over the parameters both heads read from — is what `value_coeff` is judged on. It costs a
forward and a backward per term, so it runs on `[step] grad_probe_every` and has gaps by design.

**What the encoder attends to** ([src/rl/model/introspect.rs](src/rl/model/introspect.rs)). Per
`(block, head)`: `attn_entropy/*`, the key distribution's entropy in nats, and `attn_focus/*`, the
attention mass spent on each of the five §1.2 token families **divided by** that family's share of
the batch's unmasked tokens (`attn_share/*`, logged beside it). `1.0` is chance. The ratio and not
the mass, because the families are not the same size and do not fill alike — History holds 20 slots
and stays full where Pokémon has 40 mostly padded — so a raw mass cannot separate a preference from
an abundant family. The families being named is what makes the reading semantic at all; an
interpretability pass over text would have to earn the same statement.

**Pokémon and Trainer are read per zone as well** (`attn_focus/*/trainer.hand`, …), because for those
two the family baseline is unreadable: their tokens span hand / deck / discard, so ~11 of a batch's
~14 Trainer tokens are cards nobody can play, and a head spending its mass on the relevant ones alone
scores ~0.5 against chance. `long_v3` and `long_v4` both read all 48 head/family pairs below chance
on Trainer — including the run with no text features, which rules out the text channel as the cause.
The zoned buckets refine the families rather than replace them (asserted: the four zones of a family
sum to it, in share and in mass), and the aggregates stay so earlier runs' curves keep meaning
something.

**Redundancy is a pair's property and not a head's** (`attn_js/*`, keyed `b<block>h<low>h<high>`): the
Jensen-Shannon divergence between two heads' key distributions over the same query rows, in nats,
for every unordered pair *within* a block. `0` is two copies of one head, `ln 2 ≈ 0.693` two heads
whose supports never meet. `attn_focus/*` cannot stand in for it — two heads on one family read as
two equal masses whether they duplicate each other or split the family between them — and the
question is only well posed inside a block, whose heads are concatenated into one projection and are
therefore permutation-symmetric. It rides the same probe, adds no forward, and is quadratic in
`[model] num_heads`.

**How much each block writes** (`attn_write/b<block>`, `block_write/b<block>`, `stream_norm/*`): the
attention sublayer's write and the whole block's, each as a fraction of the residual stream it wrote
into, per real token. The scale factor every reading above is implicitly multiplied by, and what
separates the two things a near-uniform head can mean — a block writing little is near-identity and
its pattern is irrelevant, while a block writing hard through a uniform pattern is pooling the
sequence into every token, which is what lets the next block's queries carry context and be
selective. The first is wasted capacity, the second is what makes the rest possible, and the entropy
alone reads them the same. `stream_norm/*` is logged because pre-LN accumulates with depth, so the
two ratios do not share a denominator across blocks.

Two things it cannot say. A head with nothing to contribute still spends a total mass of 1 and
deposits it on a fixed token — in practice the global row, the only one never padded — so focus
concentrated there is a head with no signal, not a head consulting the global features. And the
series are keyed by position, which is a head's whole identity: changing `[model] num_heads` or
`num_blocks` starts a new set of curves.

One forward over a single micro-batch, on `[step] attn_probe_every`, gaps by design like the
gradient probe. The frames are taken on a stride across the batch's episodes, never off the head of
the flattened list: at ~25 decisions a game that would confine the probe to three of the hundred-odd
games a batch collects, and the panel and the deck sampler make those three unrepresentative — the
reading would track the deck lottery rather than the weights. The cadence is set by resolution
rather than cost (the probe is ~0.6 % of a batch even if run every batch), and what it must
out-resolve is the probe's own sampling noise, which
[src/rl/train/update.rs](src/rl/train/update.rs) measures directly by reading two disjoint halves of
one batch at fixed weights.

**Two read-outs live outside the loop**, since a series added today starts at the next batch and
says nothing about the run already on disk.

- [examples/attention_probe.rs](examples/attention_probe.rs) reads every series above off a
  checkpoint, rebuilding the model from the run's own config, text embeddings and stage. Each
  reading is repeated on independent draws and reported with its spread, one collection being too
  few frames to separate a head moving from the deck lottery. The opponent distribution is not
  reproduced — the training probe also sees `[pool]`'s clones — so family masses travel worse from
  there than the pairwise divergences, which are a property of the weights.
- [examples/block_drift.rs](examples/block_drift.rs) measures the parameters rather than the
  activations, per block and with `q,k` split from `v,o`: a block writing hard through a fixed
  random projection reads the same as one that learned where to look, and only the drift of `q,k`
  separates them. Normalized by `√batch` and referenced to the pool's earliest clone —
  `init_seed` fixes the frozen tables, but nothing in `src/rl/train` seeds burn's global RNG, so the
  blocks' initialization is unrecoverable once the process ends.

**Winrate is two measurements, not one** ([src/rl/train/eval.rs](src/rl/train/eval.rs)), because the
per-batch `panel/winrate` mixes the panel's opponents and, at ~60 games, carries a ±13 % interval.

- **`panel/window/*` — the continuous curve**, folded off the training rollout, split per opponent
  over a rolling window of batches. Free (those games are collected anyway), on the deck distribution
  the curriculum defines, and at ~60 games a batch over 20 batches it reaches ≈ ±2 %, against ≈ ±5 %
  for a few hundred dedicated games. Volume bought the precision that a fixed-deck paired design
  would have bought at the price of a permanently biased level.
- **`eval/*` — a held-out probe**, dedicated games against anchors the run does not train against,
  gated by §1.5.4's trigger. An anchor inside the training mix measures nothing the fold does not
  measure better; expectiminimax is the one that is genuinely held out. Its decks sweep — evaluation
  *n* draws indices from *n·games* onward — so the curve walks the deck distribution rather than
  re-measuring one frozen sample of it.

  Measured, and the reverse of the intuition: `e2` is the cheapest anchor to evaluate against
  (≈ 37 s per 100 games, against ≈ 100 s for the random player). The cost is set by decision frames,
  one forward each, and a strong anchor ends games in 18 turns where the random player drags them to
  78. The search shows up only in decisions/s — ≈ 75 against `r` / `w`, 61 against `e2`, 40 against
  `e3` — and is outweighed three to one by the shorter games. Against a trained agent those games
  lengthen, so this is a floor on the cost, not a ceiling.

Both report the same three quantities per anchor: winrate, its binomial standard error, and — across
anchors — the mean and standard deviation. The last two answer different questions: the SE says
whether a move is worth reading, the std whether a mean of 0.6 is "beats everything moderately" or
"crushes one and loses to the rest". Ties are counted and reported, never scored as half a win.

**The bridge is JSONL, not an event file.** One flat JSON object per batch into
`runs/<name>/logs/metrics.jsonl` ([src/rl/train/logger.rs](src/rl/train/logger.rs)), replayed into a
TensorBoard event file by `auxiliaries/jsonl_to_tensorboard.py`. Writing `tfrecord`-framed protobuf
from Rust would put a protobuf toolchain in the crate for a convenience the loop does not need, and
would make the log unreadable without TensorBoard; the conversion is idempotent, rebuilding the
event file rather than appending to it, and `--serve` repeats it on an interval so a live run needs
no manual refresh. The log is appended and the batch index is in every record, so a run interrupted
three times is still one series.

**Time is a series, not a stopwatch.** `time/elapsed_seconds` accumulates across resumes and is
checkpointed beside the loop counters, so it measures training and not process uptime. Every other
value on the line is per batch, and a batch is not a fixed amount of time: what a plateau costs is
only readable against this axis, and `time/batch_seconds` is what turns the remaining batches into an
ETA. `rollout/engine_panics` is §1.5.5's dropped-game counter on the same line — those games are
missing from every other series, and a rate that climbs is a simulator regression nothing else would
show.

**stdout carries state, not history** ([src/rl/train/dashboard.rs](src/rl/train/dashboard.rs)). Every
series above goes to the JSONL and from there to TensorBoard, which is where a curve is read. So an
interactive run gets a fixed block redrawn in place (position and ETA, curriculum stage, how close
the advance screen is to firing, pool shape, and the few levels worth a glance), with events — stage
transitions, held-out evaluations, pool admissions, engine panics — scrolling above it. A redirected
run gets the line-per-batch format instead: cursor repositioning turns a log file into escape codes.

Two shapes the diagnostics take deliberately
([src/rl/train/diagnostics.rs](src/rl/train/diagnostics.rs)):

- **Distributions become named scalar series, not histograms** — 18 `head_share/*` curves and
  `mask/size_{mean,p50,p90,max}`, because scalars need no bucketing agreement. `ActionType` is
  omitted from the shares: it carries the induced family marginals and is never a chosen bit, so its
  series would be pinned to zero.
- **Per-head entropy is measured at collection, not in the update** — the policy row is already read
  back to the CPU there, so it costs a fold. It is restricted to frames where the head had two or
  more legal bits: a one-bit head has zero entropy by arithmetic, not by policy. `head_forced/*` is
  the companion that says how often that happens, per head and on the same pool of frames: of the
  frames that offered the head at all, the share where it offered a single bit. The two denominators
  partition that pool, so `head_entropy/x` falling while `head_forced/x` climbs is the mask
  narrowing rather than the policy collapsing — which `head_entropy/*` alone cannot distinguish.
  `EndTurn` and `UseStadium` are left out of both: a one-slot domain cannot hold two bits, so their
  rate would be a constant 1 and their entropy is never defined.

  **Forced is a per-head reading because a whole forced frame is an invariant, not an event.**
  §1.3.6.3's one-candidate frames are auto-resolved inside `step`
  ([src/rl/env.rs](src/rl/env.rs)) and never reach the learner, and the projection is a bijection
  onto the candidates (§1.3.7 invariant 1) — so every collected frame carries two or more bits,
  always. A per-frame forced rate is therefore a flat zero, which by this section's own rule reads
  as a measurement rather than as the absence of one; it is asserted over real rollouts in
  `only_genuine_decisions_of_the_agent_seat_are_yielded` and `every_recorded_bit_is_a_legal_bit`
  instead of logged. A head with one bit inside an eight-bit frame is the thing that does happen,
  and is what the entropy fold actually skips.

### 1.5.7. Label harvest

Pretraining runs 10⁶–10⁷ games it would run anyway; logging them is the cheapest label source the
deckbuilder will get. A `SimulationEventHandler`
([src/gameplay_stats_collector.rs](src/gameplay_stats_collector.rs)) serves both the RL loop and the
ordinary mass-labeling passes.

Three rules: counts, never rates — every quantity sits next to the denominator it would be divided
by, so the offline aggregation picks its own conditioning; `card_id@copies` is the identity (§1.6.1),
stats attaching to the printed card, never to a physical copy; and every row names both the matchup's
pilots and its decks, `(deck_id, pilot, opponent_deck, opponent_pilot)` being the key of the deck
table.

The pilot names are the run's own opponent vocabulary
(§1.5.2): `learner`, `w`, `e2`, `pool:b000001234`, `baked:<name>`, or `unknown` for a caller that
set none. The opponent's name and decklist are both on the row for the matchup reason: a winrate is
a property of the pair, and §1.6.2 fits it as one — `opponent_deck` is what keeps that true once a
curriculum stage mixes archetypes on the opposing side, where the pilot name alone no longer
determines which decklist produced a given win.

- **Per deck**: games, W/L/T, points scored / conceded, turns, `damage_dealt_total`, deck-out turn,
  hand sizes.
- **Per `(deck, card_id@copies)`**: `copies_drawn`, `times_played`, `games_never_drawn`,
  `games_drawn_never_played`, `ended_in_hand`, first-play turn; `ability_activations` /
  `turns_ability_available`; `attacks_used`, `damage_dealt`, `kos_dealt`; `turns_active` /
  `turns_benched`, `damage_taken_active` / `damage_taken_bench`, `healing_received`, `times_koed`,
  `base_hp`.

Four constraints hold that list together:

- **Every card gets a row every game, drawn or not** — dropping never-drawn rows silently conditions
  every downstream ratio on "was drawn".
- **Damage is absolute, never a share** of the deck's total (shares are compositional and fabricate
  anti-synergy between attackers). The share stays derivable from `damage_dealt_total`.
- **`turns_ability_available` and `turns_active` are the denominators** that make "dead ability" and
  absorption measurable; the first is read off `playable_actions`, so the engine stays the authority
  (§1.3.1).
- **Winrate labels are fitted as a matchup model, not a marginal** (§1.6.2), and restricted to
  plateau-region checkpoints (§1.5.4), since a harvested winrate depends on the pilot checkpoint. The
  mechanical counts are far less pilot-sensitive and are kept over the whole run.

**Curriculum flag.** Each stage carries `log: none | sampled(p) | full`; a thin sample of early stages
is kept on purpose, so pilot drift can be measured rather than assumed. Until §1.5.4 owns it,
`[harvest] log` is the run-wide setting — `false`, `true`, or a probability, the TOML type saying
which.

**Storage** ([src/rl/train/harvest.rs](src/rl/train/harvest.rs)):
`runs/<name>/harvest/shard-NNNNNN/`, holding three flat JSONL tables — `decks.jsonl` keyed on
`(deck_id, pilot, opponent_deck, opponent_pilot)`, `cards.jsonl` on that key plus `(card_id,
copies_in_deck)`, and an append-once `dictionary.jsonl` (on `deck_id` alone — a decklist does not
depend on who played it, and covers both sides of every matchup, so `opponent_deck` resolves
through the same table as `deck_id`).

- **Shards are additive, never snapshots.** A flush writes only the games since the last one; the
  counters sum, so the offline merge is a group-by and a resume just opens the next shard index.
- **Flat rows, not nested maps.** The per-card map has a struct key and JSON object keys must be
  strings.
- **JSONL over CSV or Parquet.** Repeating field names costs ~3× the bytes — a few hundred MB at this
  scale — and buys tolerance to schema drift, which matters while the field list moves.
- **Sampling is per game, never per row.** Dropping rows would break `games_never_drawn` and every
  denominator above.

**One collector per game, not one per run.** The collector carries per-game state (the board snapshot
it diffs damage against, the last acting card), so a single instance cannot interleave the parallel
envs of §1.5.5 — each env gets its own and they are merged at the flush. Both seats are harvested:
stats key on the deck, and a game teaches as much about the opponent's list — which is also why the
pilot pair matters, the far seat being a heuristic or an archived clone rather than the learner.

**Free calibration tests.** §1.5.3's pure-mirror quota must fit to ≈ 50 % (modulo `on_the_play`), and
per-card `damage_dealt` must sum exactly to `damage_dealt_total`.

Collection details — the counts-vs-rates argument, the two damage-attribution traps, KO vs
board-departure, decklist reconstruction and the de-biasing protocol: `NOTES.md`.

## 1.6. Part 6 — Deckbuilder (sketch)

Distant; rationale in §1.1.4 — this part fixes only the concrete shape. Uses the strictly frozen
meta-neutral embeddings, with no player residual.

### 1.6.1. Encoder & input

- Reuses the Part 4 encoder skeleton on the frozen embedding copy, but with no game state: input is a
  deck as a set of card tokens (static descriptors only — no dynamic block, no History, no Attack
  affordances). Allied set (≤ 20) + optional enemy set (≤ 20) + ≤ 2 selected energies, each tagged
  allied / enemy + energy flag.
- Pooling gives a deck representation; the model is an EBM, a scalar energy over the `(deck, enemy)`
  association.
- **A deck is a multiset, not a set** — `card_id@1` and `card_id@2` are different entities: same
  printed card, different draw dynamics and different marginal value. Two consequences: tokens carry
  a copy index (`copy 1/2`, `copy 2/2`), since identical tokens are invisible as a count under
  pooling and the §1.6.3 GFlowNet can add a 2nd copy; and copy count is part of the label key
  (§1.5.7). A side effect is that a `card_id × card_id` pair can never stand in for a 2-copy card in
  the Coherence head — a pair is an interaction between two distinct cards. Rationale: `NOTES.md`.

### 1.6.2. Three scoring heads (EBM)

- **Strength** — label: measured winrate.
- **Coherence** — win-conditioned synergy lift (PMI), pairwise labels + shrinkage (beta prior),
  higher orders learned from the deck-level aggregate; decorrelated from Strength.
- **Counter** — label: conditional winrate vs the enemy set.
- Final head weighting deferred.

Strength is fitted as a matchup model with a card-factored latent, not as a marginal winrate:
`strength(deck) := E_θ(deck)` inside a Bradley-Terry / bilinear fit over `(deckA, deckB, outcome)`
triples, rather than a free per-deck parameter, which would be unidentifiable at the harvest's
density (≈ 50 games/deck). It also dissolves "winrate against which field?" (the marginal is
recomputable under any field, after the fit) and is the Counter head's target: one fit, two heads.

### 1.6.3. Sampling — GFlowNet

- Deck built card-by-card (sequential-add MDP, legality-masked so the 20-card / ≥ 1-non-fossil
  constraint stays reachable). Objective: trajectory balance (default). Terminal reward
  ∝ `exp(−energy)` → sampling proportional to energy, no argmax.
- Exploration bonus from a per-card / per-pair visit counter (closed pool), so the landscape carries
  the uncertainty of rarely visited regions.

### 1.6.4. Feedback loop & data

- Proposes decks to the player (weights still updating), receives results as labels. Coarse mass
  labeling (128–256 games) plus fine labels where the GFlowNet concentrates (§1.1.5).
- Pretraining limited to the `tutorial` DB (< 1000 decks) — no human-meta contamination.

### 1.6.5. Warm start from the harvest

The §1.5.7 harvest seeds the three heads. This does not contradict §1.1.2, which forbids
contaminating the values: an embedding pretrained on human decks encodes a human opinion, and
distilling the player's pooled deck representation is the same path since it carries a learned
residual — distil labels, never representations. A winrate measured in the simulator is a fact; only
the sampling is skewed. The human meta sets the landscape's initial resolution map, not its initial
preferences: well resolved on card marginals (n ≈ 10⁵) and on meta pairs (n ≈ 10³–10⁴), thin on
exotic pairs (n ≈ 50–100), useless per deck (n ≈ 50).

**The single binding condition: seed the counters, not only the means.** §1.6.3's exploration bonus is
driven by a per-card / per-pair visit counter; seeded honestly, a low-`n` exotic pair reads as high
uncertainty and the GFlowNet is pulled toward it, so a warm start begins by exploring the complement
of the human meta. Lock-in only happens if a mean is seeded without its `n`. Same for the Coherence
beta prior, whose prior mean can come from the value head at a deliberately small pseudo-count.

Consequently the uniform-deck quota has to exist and to sit above the anti-collapse minimum: it is
what buys the exotic-pair coverage. It is deckbuilder-side work — §1.5.3's player sampler does not
carry it, and the generator of legal random decks it needs does not exist yet.

**Adoption is falsifiable.** §1.1.6 stage 2 runs twice — cold (uniform + tutorial) and warm (full
harvest) — compared on held-out winrate; the warm start is adopted only if it wins on exotic held-out
decks. The harvest stays a separate table with its sampler tag, never merged into fresh labels, so
the ablation stays available; the tag is a row annotation, never an input feature.

**Expected failure mode: the EBM collapses to additive + weak pairwise**, which loses exactly the
structure that matters most — combo decks, where energy is low for `{A,B,C}` and high for every
2-subset, and the card-by-card GFlowNet never gets pulled through the valley. Four remedies, cheapest
first: trajectory balance is already the right objective for this reason (§1.6.3); the mechanical
utilization labels of §1.5.7 are the real fix (a broken interaction is observable in 20 games, a
synergy lift needs 10³); the deckbuilder can afford more depth than the player (`N ≤ 42`, one forward
per proposed deck); and the decisive diagnostic is to fit additive, additive+pairwise and the EBM on
the same data — if the EBM does not beat additive+pairwise it learned nothing higher-order.

Why exhaustive higher-order estimation is out of reach, why the frozen mechanical embeddings are the
smoothing kernel that makes higher-order generalization possible at all, the resolution table and the
cold-start cost: `NOTES.md`.

## 1.7. Part 7 — Deployment (sketch)

- **`Player` trait impl.** The trained best-response is wrapped as an engine `Player`; egocentric by
  role, so it plays as P1 or P2, against bots or humans, unchanged. Checkpoints are registered by
  name so any one is callable as a player.
- **Inference**: single batched forward, action = argmax or sample over the masked heads.
- **TUI advisor** (read-only overlay, no engine mutation): given the human's current state, expose
  (a) suggested action (policy argmax), (b) per-action confidence (masked-softmax over legal
  actions), (c) state judgment (the value head output).
- Far future: weight analysis and model dissection.
