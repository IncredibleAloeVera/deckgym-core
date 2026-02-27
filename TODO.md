[!] High priority :

- [x] Remove draw-card in the rust simulator as a manual action as it should be automatic. Currently there is 5 actions on average during the DRL training, cutting that by 1 means 20% less steps per turn.

- [x] Encode attached tool description (text embedding) on played Pokemon cards in observation tensor (`src/rl/observation.rs`). Currently tools only get a 1-bit `is_tool` flag but the model has no way to know *what* tool is attached to a Pokemon.
- [ ] Training interpretability logging: add metrics that reveal *what* the model is learning during training, not just whether it is learning.
  - Relative entropy (normalised by number of valid actions) — raw entropy is misleading because the action space varies heavily between steps
  - Per-action-category frequency histograms (attack / retreat / trainer / energy / end-turn) over time — to detect if the model develops recognisable habits or gets stuck in degenerate policies
  - Value head statistics (mean, std, percentiles) per episode outcome — to verify the critic is calibrating correctly

Medium priority :

- [ ] TUI Coach Panel: assist human players in understanding and elevating their play by showing what an expert ONNX agent would do.
  - **Intent**: if/when the trained agent reaches expert or superhuman level, the coach panel lets a human player compare their intuition against the model's evaluation in real time — action by action. Note: the model's quality depends on training completion; the feature is most valuable once the agent demonstrably outperforms skilled human play.
  - Add `predict_batch_logits()` to `BatchedOnnxInference` in `src/players/onnx_player.rs` returning raw logits (not just argmax) so the full distribution is available
  - Update `onnx_export.py` `PolicyWrapper` to also output `state_value` (value head) as a second ONNX output — currently the `_` in `policy_latent, _ = self.mlp_extractor(features)` is discarded
  - Add `coach_model: Option<BatchedOnnxInference>` to `App` in `src/tui/app.rs`, loaded via a `--coach model.onnx` CLI flag
  - Render a probability bar per legal action in `src/tui/ui.rs` footer, sorted by prob descending (e.g. `[████████░░] 68%  Attack Pikachu`)
  - Display the value estimate `v(s) ∈ [-2, 2]` as a position evaluation bar (e.g. `Position: +0.74 — model favours you`)

- [ ] Model Interpretability Module (`python/deckgym/interpretability/`): dual-purpose — diagnose training pipeline weaknesses AND extract a human-readable description of the expert strategy the agent has learned.
  - **Purpose 1 — pipeline diagnosis**: identify if the model exploits reward shaping artefacts, fixates on irrelevant features, or fails to generalise across decks. Informs reward function improvements, observation encoding gaps, and hyperparameter tuning.
  - **Purpose 2 — strategy extraction**: since the emergent expert strategy is not known in advance (and cannot be hand-coded), interpretability tools are the primary way to describe *what the agent actually does* — e.g. "prioritises bench development in turns 1-3", "attacks only when energy advantage ≥ 2", "retreats when HP < 40%".
  - **Attention visualisation** (BerViz): expose per-head attention weights from `OnnxSafeAttention` via PyTorch hooks; display card-to-card heatmaps showing which board elements the model focuses on at each decision point
  - **Feature attribution** (Captum Integrated Gradients): for a given state + chosen action, identify which observation features (HP, energy counts, specific cards) most influenced the decision; requires a label mapping from flat obs indices → card/feature names based on the layout in `RL_ARCHITECTURE.md` and `observation.rs`
  - **Policy latent UMAP/t-SNE**: collect `(policy_latent, value, action_category, turn, point_diff)` across thousands of game positions and project to 2D to reveal and name strategy clusters (e.g. setup phase, aggressive pressure, defensive stall, forced-decision)
  - Entry point script `python/scripts/interpret_model.py`: loads a checkpoint, runs self-play or evaluation games, and produces HTML/image reports

- [ ] Design and Implement AI deckbuilder
- [ ] Use Typer for every python scripts that support CLI

- [ ] Upstream PR preparation: merge simulation, ONNX, and coaching improvements back to upstream `main` without including the private training pipeline.
  - Identify which components are self-contained and valuable upstream: `BatchedGameRunner` (`batched_runner.rs`), `VecGame`/`BatchedDeckGymEnv` ONNX pool, `simulate_batched`, `batched_env.py`, `onnx_export.py`, TUI coach panel
  - Explicitly exclude: `train.py`, PFSP callbacks, `config.py` training hyperparameters, deck archives, etc.
  - Check for any upstream API-breaking changes introduced since fork (deck format, `PyVecGame` bindings, observation schema version)

Low priority :

- [ ] Explore others learning algorithms apart from PPO
- [ ] Explore optimisation across the training pipeline (Rust, Python, PyTorch)