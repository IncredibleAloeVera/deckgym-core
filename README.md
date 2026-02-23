<img src="./images/logo.svg" alt="Logo" width="100" height="100">

# deckgym-core: Deep RL for Pokémon TCG Pocket

![Card Implemented](https://img.shields.io/badge/Cards_Implemented-1709_%2F_2408_%2871.0%25%29-yellow)

> Fork of deckgym-core focused on **training a Deep Reinforcement Learning agent** to play Pokémon TCG Pocket at a competitive level.

The Rust simulator serves as a high-performance environment (10k games in ~3s). On top of it, a full **PPO + Transformer** training pipeline produces ONNX policies that play directly inside the engine.

### Key components at a glance

- **Rust game engine** — full TCG Pocket rules, vectorized batched env (`VecGame`), O(1) FFI per step
- **Observation tensor** — 29,411-dim flat vector (171 global + 40 cards × 731 features each), with 128-dim text embeddings (SentenceTransformer + PCA)
- **Action space** — 179 fixed indices covering board, hand, and resolution actions
- **Transformer policy** — `CardAttentionExtractor` (self-attention + learned pooling), exported to ONNX for Rust-side inference
- **Training** — `MaskablePPO` (SB3) with PFSP opponent pool, curriculum deck loading, TrueSkill ratings
- **Evaluation** — full archetype matrix + generalization protocol, JSON/Markdown reports

See [RL_ARCHITECTURE.md](./RL_ARCHITECTURE.md) for the full technical breakdown.

---

## System Requirements

While the core simulation runs on CPU, using ONNX neural networks with GPU acceleration requires the following:

### GPU Acceleration (Optional)

If you plan to use the `onnx` feature with CUDA:
- **NVIDIA GPU** (Compute Capability 7.0+)
- **NVIDIA Driver**
- **CUDA Toolkit** (11.8+)
- **cuDNN** (8.x or 9.x)

## Usage

We already provide several example decks in the repo you can use to get started. For example, to face off a VenusaurEx-ExeggutorEx deck with a Weezing-Arbok deck 1,000 times, run:

```bash
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 -v
```

You can also simulate one deck against multiple decks in a folder. The total games will be distributed evenly across all decks:

```bash
# Simulate your deck against all decks in example_decks folder (1000 games total)
cargo run simulate my_deck.txt example_decks/ --num 1000 -v
```

## Terminal User Interface (TUI)

The TUI provides an interactive way to view and replay games with a visual representation of the game state.

To use the TUI, you need to enable the `tui` feature:

```bash
cargo run --bin tui --features tui -- example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --players e,e
```

### Controls

- **↑/↓/Space**: Navigate between game states (forward/backward)
- **PageUp/PageDown**: Scroll through battle log
- **A/D**: Scroll through your hand cards
- **Shift+A/Shift+D**: Scroll through opponent's hand cards
- **Q/Esc**: Quit

## DRL Pipeline

> Most scripts support `--help` / `-h` for full argument details.

**Setup**

```bash
maturin develop --release --features onnx     # Build Python bindings (with ONNX for self-play)
cargo build --release --features onnx         # Build Rust binary (needed by evaluation scripts)
```

**Training**

```bash
python python/scripts/train.py --config <config.yaml>
python python/scripts/train.py --config <config.yaml> --resume <checkpoint.zip>
```

**Evaluation**

```bash
python python/scripts/evaluate.py chaos <model_code>            # Full archetype matrix
python python/scripts/evaluate.py generalization <model_code>   # Unseen decks protocol
python python/scripts/evaluate_human.py                         # Human vs bot (TUI)
```

**Diagnostics & Data**

```bash
python python/scripts/diagnose_model.py <model.zip>    # Gradient and attention health check
python python/scripts/generate_embeddings.py            # Regenerate card_features.json
python python/scripts/clean_archetypes.py               # Filter archetypes with incomplete cards
```

See [RL_ARCHITECTURE.md](./RL_ARCHITECTURE.md) for the full technical reference.

---

## Contributing

New to Open Source? See [CONTRIBUTING.md](./CONTRIBUTING.md).

The main contribution is to implement more cards, basically their attack and abilities logic. This makes the cards eligible for simulation and thus available for use in https://www.deckgym.com.

See the Claude [SKILL.md](./.claude/skills/implement-cards/SKILL.md) describing how to implement cards.
It's good documentation for humans and AIs alike.


## Appendix: Useful Commands

Once you have Rust installed (see https://www.rust-lang.org/tools/install) you should be able to use the following commands from the root of the repo:

**Running Automated Test Suite**

```bash
cargo test
```

**Running Benchmarks**

```bash
cargo bench
```

**Running Main Script**

```bash
# Simulate between two specific decks
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 --players r,r
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1 --players r,r -vv
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1 --players r,r -vvvv

# Simulate one deck against all decks in a folder (games distributed evenly)
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/ --num 1000 --players r,r -v

# Optimize incomplete decks
cargo run optimize example_decks/incomplete-chari.txt A2147,A2148 example_decks/ --num 10 --players e,e -v
cargo run optimize example_decks/incomplete-chari.txt A2147,A2147,A2148,A2148 example_decks/ --num 1000 --players r,r -v --parallel
```

**Card Search Tool**

The repository includes a search utility that's particularly useful for agentic AI applications, as reading the complete `database.json` file (which contains all card data) often exceeds context limits.

```bash
# Search for cards by name
cargo run --bin search "Charizard"

# Search for cards with specific attacks
cargo run --bin search "Venusaur" --attack "Giant Bloom"
```

**Card Implementation Status Tool**

Check which cards are fully implemented versus which are missing attack effects, abilities, or trainer logic. This tool helps contributors identify cards that need implementation work.

```bash
# Show all cards with their implementation status
cargo run --bin card_status

# Show only incomplete cards
cargo run --bin card_status -- --incomplete-only

# Get the first incomplete card (useful for automation)
cargo run --bin card_status -- --first-incomplete
```

The tool displays a summary showing total cards, completion percentage, and a breakdown of missing implementations by type (attacks, abilities, tools, trainer logic).

Compare decision-making speed of different player strategies (running 1000 games in parallel):

```bash
python python/scripts/benchmark_players.py
```

| Player | Code | Games/s | Notes |
|--------|------|---------|-------|
| AttachAttack | `aa` | ~30300 | Simple heuristic, highly parallelizable |
| EvolutionRusher | `er` | ~20000 | Prioritizes evolutions |
| WeightedRandom | `w` | ~15000 | Slight heuristics overhead |
| Random | `r` | ~10600 | Baseline random policy |
| EndTurn | `et` | ~4200 | Always ends turn immediately |
| ValueFunction | `v` | ~900 | Evaluates board state heuristically |
| **ONNX o1 (GPU)** | `o1g` | **~210** | **Flagship Model** (Attention-based) |
| ExpectiMiniMax (d=2) | `e2` | ~150 | Search tree with depth 2 |
| ExpectiMiniMax (d=3) | `e3` | ~40 | Search tree with depth 3 |
| **ONNX o1 (CPU)** | `o1c` | **~35** | Attention model (15MB) |
| **MCTSPlayer** | `m[n]` | Varies | Omniscient search (lookahead) |

**Temporary Deck Generator**

Generate a valid temporary test deck for a specific card id (it considers the evolution chain of a card and the required energy types if its a pokemon card).

```bash
cargo run --bin temp_deck_generator -- "A1 035"
```

**Card Test Command**

Generate a temporary test deck and run 10,000 games against all decks in `example_decks/` (games distributed evenly) using random players.

```bash
cargo run --bin card_test -- "A1 035"
```

**Setting Up Git Hooks (Optional)**

The repository includes a pre-commit hook that ensures code quality by automatically fixing issues and running tests before each commit. To enable it:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook runs:
1. `cargo clippy --fix --allow-dirty --features tui -- -D warnings` - Auto-fixes linting issues
2. `cargo fmt` - Auto-formats code
3. `git add -u` - Adds clippy and formatting fixes to the commit
4. `cargo test --features tui` - Runs the full test suite (fails commit if tests fail)

This helps maintain code quality and prevents broken commits, but it's optional and each developer can choose whether to enable it.

**Generating database.rs**

Ensure database.json is up-to-date with latest data. Mock the `get_card_by_enum` in `database.rs` with a `_ => panic` so that
it compiles mid-way through the generation.

```bash
cargo run --bin card_enum_generator > tmp.rs && mv tmp.rs src/card_ids.rs && cargo fmt
```

Then temporarily edit `database.rs` for `_` to match Bulbasaur (this is so that the next code can compile-run).

```bash
cargo run --bin card_enum_generator -- --database > tmp.rs && mv tmp.rs src/database.rs && cargo fmt
```

To generate attacks do (first time):
```bash
cargo run --bin card_enum_generator -- --attack-map > tmp.rs && mv tmp.rs src/actions/effect_mechanic_map.rs && cargo fmt
then with each new set of new mechanics, use:
```bash
cargo run --bin card_enum_generator -- --incremental-attack-ma
```
and manually copy-paste into the ever changing `src/actions/effect_mechanic_map.rs`.

**Profiling Main Script**
```bash
sudo cargo flamegraph --root --dev -- simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 && open flamegraph.svg
```
