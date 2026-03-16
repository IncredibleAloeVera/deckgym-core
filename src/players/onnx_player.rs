//! ONNX-based neural network player for self-play training.
//!
//! This player uses a pre-trained ONNX model exported from Python to make decisions.
//! Useful for fast self-play training with frozen opponents.

#![allow(unused_imports)]

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

/// Print available execution providers (call once at startup for debugging)
#[cfg(feature = "onnx")]
pub fn print_available_providers() {
    use ort::execution_providers::ExecutionProvider;

    // Check CUDA availability (but actual usability depends on cuDNN)
    let cuda_ep = CUDAExecutionProvider::default();
    if cuda_ep.is_available().unwrap_or(false) {
        eprintln!("  [ONNX] CUDA provider: AVAILABLE");
    } else {
        eprintln!("  [ONNX] CUDA provider: NOT AVAILABLE");
    }
}

use rand::rngs::StdRng;
use rand::Rng;

use crate::{
    actions::Action,
    rl::{
        get_action_mask, get_indexed_actions, get_observation_tensor, ACTION_SPACE_SIZE,
        OBSERVATION_SIZE,
    },
    state::State,
    Deck,
};

#[cfg(feature = "onnx")]
use crate::players::Player;

use std::fmt::Debug;

// Global session cache to avoid reloading ONNX models for every game
#[cfg(feature = "onnx")]
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "onnx")]
type SessionCache = Mutex<HashMap<String, Arc<Mutex<Session>>>>;

#[cfg(feature = "onnx")]
static ONNX_SESSION_CACHE: OnceLock<SessionCache> = OnceLock::new();

#[cfg(feature = "onnx")]
fn get_session_cache() -> &'static SessionCache {
    ONNX_SESSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Clear the global ONNX session cache to free GPU memory.
#[cfg(feature = "onnx")]
pub fn clear_onnx_cache() {
    if let Some(cache) = ONNX_SESSION_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            guard.clear();
            eprintln!("  [ONNX] Session cache cleared (GPU memory released)");
        }
    }
}

/// Remove a specific model from the cache.
#[cfg(feature = "onnx")]
pub fn remove_model_from_cache(model_path: &str, device: &str) {
    if let Some(cache) = ONNX_SESSION_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            let normal_key = format!("{}:{}", model_path, device);
            let batched_key = format!("batched:{}:{}", model_path, device);
            guard.remove(&normal_key);
            guard.remove(&batched_key);
            eprintln!("  [ONNX] Model removed from cache: {}", model_path);
        }
    }
}

/// A player that uses an ONNX model for decision-making.
///
/// The model takes a game observation and outputs action logits.
/// Supports both deterministic (argmax) and stochastic (sampling) modes.
#[cfg(feature = "onnx")]
pub struct OnnxPlayer {
    session: Arc<Mutex<Session>>,
    deterministic: bool,
    deck: Deck,
    model_path: String,
    device: String,
}

#[cfg(feature = "onnx")]
impl Debug for OnnxPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OnnxPlayer {{ model: {}, device: {}, deterministic: {} }}",
            self.model_path, self.device, self.deterministic
        )
    }
}

#[cfg(feature = "onnx")]
fn get_execution_providers(
    device: &str,
) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    match device.to_lowercase().as_str() {
        "cpu" => {
            vec![]
        }
        "auto" | "cuda" => {
            vec![CUDAExecutionProvider::default().build()]
        }
        _ => {
            vec![]
        }
    }
}

#[cfg(feature = "onnx")]
impl OnnxPlayer {
    /// Create a new ONNX player from a model file.
    /// Uses a global session cache to avoid reloading the model for every game.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `deck` - The deck this player uses
    /// * `deterministic` - If true, always pick the best action; if false, sample from distribution
    pub fn new(
        model_path: &str,
        deck: Deck,
        deterministic: bool,
        device: &str,
    ) -> Result<Self, String> {
        // Check cache first
        let cache = get_session_cache();
        let cache_key = format!("{}:{}", model_path, device);

        let session = {
            let mut guard = cache
                .lock()
                .map_err(|e| format!("Cache lock poisoned: {}", e))?;

            if let Some(cached) = guard.get(&cache_key) {
                // Reuse cached session
                Arc::clone(cached)
            } else {
                // Create new session and cache it
                let providers = get_execution_providers(device);
                let new_session = Session::builder()
                    .map_err(|e| format!("Failed to create session builder: {}", e))?
                    .with_execution_providers(providers)
                    .map_err(|e| format!("Failed to set execution providers: {}", e))?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| format!("Failed to set optimization level: {}", e))?
                    .with_intra_threads(1)
                    .map_err(|e| format!("Failed to set thread count: {}", e))?
                    .commit_from_file(model_path)
                    .map_err(|e| {
                        format!("Failed to load ONNX model from '{}': {}", model_path, e)
                    })?;

                let arc_session = Arc::new(Mutex::new(new_session));
                guard.insert(cache_key, Arc::clone(&arc_session));
                arc_session
            }
        };

        Ok(Self {
            session,
            deterministic,
            deck,
            model_path: model_path.to_string(),
            device: device.to_string(),
        })
    }

    /// Run inference on a single observation.
    /// Returns the selected action index.
    fn predict(&self, obs: &[f32], action_mask: &[bool], rng: &mut StdRng) -> usize {
        // Create input tensor as ndarray
        let obs_array = Array2::from_shape_vec((1, OBSERVATION_SIZE), obs.to_vec())
            .expect("Failed to create observation array");

        // Create tensor reference
        let tensor = TensorRef::from_array_view(&obs_array).expect("Failed to create tensor");

        // Run inference and extract logits
        let logits = {
            let mut session = self.session.lock().expect("Session lock poisoned");
            let outputs = session
                .run(ort::inputs![tensor])
                .expect("ONNX inference failed");

            // Get logits output (index 0) and copy before dropping outputs
            let (_shape, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .expect("Failed to extract logits tensor");
            logits_data.to_vec()
        }; // outputs is dropped here

        // Apply mask and select action
        self.select_action(&logits, action_mask, rng)
    }

    /// Select action from logits with masking.
    fn select_action(&self, logits: &[f32], mask: &[bool], rng: &mut StdRng) -> usize {
        // Apply mask: set invalid actions to -inf
        let masked_logits: Vec<f32> = logits
            .iter()
            .zip(mask.iter())
            .map(|(&l, &valid)| if valid { l } else { f32::NEG_INFINITY })
            .collect();

        if self.deterministic {
            // Argmax
            masked_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            // Softmax and sample
            let max_logit = masked_logits
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = masked_logits
                .iter()
                .map(|&l| (l - max_logit).exp())
                .collect();
            let sum: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

            // Sample from distribution
            let mut cumsum = 0.0;
            let rand_val: f32 = rng.gen();
            for (idx, &p) in probs.iter().enumerate() {
                cumsum += p;
                if rand_val < cumsum {
                    return idx;
                }
            }

            // Fallback to first valid action
            mask.iter().position(|&v| v).unwrap_or(0)
        }
    }
}

#[cfg(feature = "onnx")]
impl Player for OnnxPlayer {
    fn get_deck(&self) -> Deck {
        self.deck.clone()
    }

    fn decision_fn(&mut self, rng: &mut StdRng, state: &State, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0].clone();
        }

        // Get observation from current player's perspective
        let current_player = state.current_player;
        let obs = get_observation_tensor(state, current_player);

        // Get action mask
        let action_mask = get_action_mask(state);

        // Get indexed actions to map back from index to Action
        let indexed_actions = get_indexed_actions(state);

        // Run neural network
        let action_idx = self.predict(&obs, &action_mask, rng);

        // Find the action with this index
        for (idx, action) in &indexed_actions {
            if *idx == action_idx {
                return action.clone();
            }
        }

        // Fallback to first action if index not found (shouldn't happen)
        actions[0].clone()
    }
}

/// Batched ONNX inference for VecGame.
/// This is faster than individual calls when processing multiple environments.
#[cfg(feature = "onnx")]
pub struct BatchedOnnxInference {
    session: Arc<Mutex<Session>>,
    deterministic: bool,
    pub model_path: String,
    pub device: String,
}

#[cfg(feature = "onnx")]
impl BatchedOnnxInference {
    pub fn new(model_path: &str, deterministic: bool, device: &str) -> Result<Self, String> {
        let cache = get_session_cache();
        let cache_key = format!("batched:{}:{}", model_path, device);

        let session = {
            let mut guard = cache
                .lock()
                .map_err(|e| format!("Cache lock poisoned: {}", e))?;

            if let Some(cached) = guard.get(&cache_key) {
                // Reuse cached session - we need to extract the inner Session
                // This is a bit tricky because OnnxPlayer stores Arc<Mutex<Session>>
                // We'll just cache Session directly for batched too.
                Arc::clone(cached)
            } else {
                let mut builder = Session::builder()
                    .map_err(|e| format!("Failed to create session builder: {}", e))?;

                // Register execution providers based on device selection
                match device.to_lowercase().as_str() {
                    "cuda" | "auto" => {
                        if let Err(e) = CUDAExecutionProvider::default()
                            .with_device_id(0)
                            .register(&mut builder)
                        {
                            eprintln!("  [ONNX] Warning: Failed to register CUDA provider: {}. Falling back to CPU.", e);
                        } else {
                            eprintln!("  [ONNX] CUDA provider registered successfully.");
                        }
                    }
                    _ => {}
                }

                let new_session = builder
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| format!("Failed to set optimization level: {}", e))?
                    .with_intra_threads(4)
                    .map_err(|e| format!("Failed to set thread count: {}", e))?
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

                let arc_session = Arc::new(Mutex::new(new_session));
                guard.insert(cache_key, Arc::clone(&arc_session));
                arc_session
            }
        };

        Ok(Self {
            session,
            deterministic,
            model_path: model_path.to_string(),
            device: device.to_string(),
        })
    }

    /// Run batched inference on multiple observations.
    ///
    /// # Arguments
    /// * `observations` - Flattened observations [n_envs * OBSERVATION_SIZE]
    /// * `action_masks` - Flattened action masks [n_envs * ACTION_SPACE_SIZE]
    /// * `n_envs` - Number of environments
    /// * `rng` - Random number generator for sampling
    ///
    /// # Returns
    /// Vec of action indices, one per environment
    pub fn predict_batch(
        &mut self,
        observations: &[f32],
        action_masks: &[bool],
        n_envs: usize,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        assert_eq!(observations.len(), n_envs * OBSERVATION_SIZE);
        assert_eq!(action_masks.len(), n_envs * ACTION_SPACE_SIZE);

        // Create batched input tensor
        let obs_array = Array2::from_shape_vec((n_envs, OBSERVATION_SIZE), observations.to_vec())
            .expect("Failed to create observation array");

        // Create tensor reference
        let tensor = TensorRef::from_array_view(&obs_array).expect("Failed to create tensor");

        // Run inference
        let all_logits = {
            let mut session = self.session.lock().expect("Session lock poisoned");
            let outputs = session
                .run(ort::inputs![tensor])
                .expect("Batched ONNX inference failed");

            // Get logits output and copy before dropping outputs
            let (_shape, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .expect("Failed to extract logits tensor");
            logits_data.to_vec()
        }; // outputs is dropped here

        // Select actions for each environment
        let mut actions = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let start_logit = i * ACTION_SPACE_SIZE;
            let end_logit = start_logit + ACTION_SPACE_SIZE;
            let start_mask = i * ACTION_SPACE_SIZE;
            let end_mask = start_mask + ACTION_SPACE_SIZE;

            let logits = &all_logits[start_logit..end_logit];
            let mask = &action_masks[start_mask..end_mask];

            let action = self.select_action(logits, mask, rng);
            actions.push(action);
        }

        actions
    }

    fn select_action(&self, logits: &[f32], mask: &[bool], rng: &mut StdRng) -> usize {
        // Apply mask
        let masked_logits: Vec<f32> = logits
            .iter()
            .zip(mask.iter())
            .map(|(&l, &valid)| if valid { l } else { f32::NEG_INFINITY })
            .collect();

        if self.deterministic {
            masked_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            let max_logit = masked_logits
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = masked_logits
                .iter()
                .map(|&l| (l - max_logit).exp())
                .collect();
            let sum: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

            let mut cumsum = 0.0;
            let rand_val: f32 = rng.gen();
            for (idx, &p) in probs.iter().enumerate() {
                cumsum += p;
                if rand_val < cumsum {
                    return idx;
                }
            }

            mask.iter().position(|&v| v).unwrap_or(0)
        }
    }
}
