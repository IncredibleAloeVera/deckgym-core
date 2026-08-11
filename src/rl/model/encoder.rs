//! The shared encoder (§1.4.1): five input projections, token-type tags, Pre-LN blocks.
//!
//! - **Five input projections**, one single linear per family (`width_f → d_model`), plus a
//!   learned token-type embedding added per family block.
//! - **Bidirectional, no causal mask**: the four entity families are permutation-invariant sets;
//!   History alone carries order, through its recency floats — not through positions.
//! - **Padding mask**: unused slots contribute nothing. The global token is never padded, so no
//!   row is ever fully masked.
//!
//! The blocks themselves are `burn::nn::transformer::TransformerEncoder`, whose `norm_first = true`
//! layer computes exactly `x + MHA(LN(x))` then `x + FFN(LN(x))` — the same graph, the same
//! parameters (4 × `d_model²` attention + 2 × `d_model · d_ff` FFN + 2 LayerNorm per block) and the
//! same default initializer as a hand-rolled version, so there is no reason to carry our own. Two
//! settings are **not** the defaults and are load-bearing:
//!
//! - `dropout = 0.0`. Burn defaults to `0.1`, applies it to the attention *scores before* the
//!   softmax rather than to the weights, and `Dropout::forward` early-returns unless
//!   `B::ad_enabled()` — so the default would be inert during inference and active under
//!   `Autodiff`, i.e. a train/inference divergence visible only once training starts. §1.4.1
//!   specifies no dropout.
//! - `norm_first = true`, with [`Encoder::ln_final`] applied **outside** the stack: burn's pre-norm
//!   encoder has no final normalization of its own, which pre-LN requires.
//!
//! Burn's padding convention is the opposite of the Part-2 wire's: `mask_pad` is `true` where a
//! slot is padding, `seq_mask` is `1.0` where a slot is real. [`Encoder::forward`] inverts it.

use burn::module::ModuleVisitor;
use burn::module::Param;
use burn::nn::attention::MhaInput;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use super::config::ModelConfig;

/// The five token families, in sequence order. `TOKEN_TYPES` tags each block.
pub const TOKEN_TYPES: usize = 5;

/// The encoder: input projections, token-type tags, blocks, final norm.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub project_global: Linear<B>,
    pub project_pokemon: Linear<B>,
    pub project_attack: Linear<B>,
    pub project_trainer: Linear<B>,
    pub project_history: Linear<B>,
    /// `[TOKEN_TYPES × d_model]`, one learned tag per family.
    token_type: Param<Tensor<B, 2>>,
    blocks: TransformerEncoder<B>,
    /// Pre-LN's final normalization, which burn's encoder does not apply itself.
    ln_final: LayerNorm<B>,
}

/// Widths of the five family inputs, computed from the Part-2 modules rather than hard-coded —
/// the spec's 853/284/195/82/170 hold when `|AbilityMechanic| = 80`.
pub struct InputWidths {
    pub global: usize,
    pub pokemon: usize,
    pub attack: usize,
    pub trainer: usize,
    pub history: usize,
}

impl InputWidths {
    pub fn of(config: &ModelConfig) -> Self {
        use crate::rl::history::HISTORY_DYNAMIC_DIM;
        use crate::rl::observation::{
            ATTACK_DYNAMIC_DIM, GLOBAL_DIM, POKEMON_DYNAMIC_DIM, TRAINER_DYNAMIC_DIM,
        };
        use crate::rl::static_tables::{ATTACK_STATIC_DIM, POKEMON_STATIC_DIM, TRAINER_STATIC_DIM};
        Self {
            // global floats ⊕ stadium embedding (shared card table).
            global: GLOBAL_DIM + config.d_id,
            // 3 concatenated IDs ⊕ static ⊕ dynamic ⊕ tool embedding.
            pokemon: 3 * config.d_id + POKEMON_STATIC_DIM + POKEMON_DYNAMIC_DIM + config.d_id,
            attack: ATTACK_STATIC_DIM + ATTACK_DYNAMIC_DIM,
            // card ID ⊕ static ⊕ dynamic ⊕ target-set bag.
            trainer: config.d_id + TRAINER_STATIC_DIM + TRAINER_DYNAMIC_DIM + config.d_id,
            history: config.d_id + config.d_head_emb + HISTORY_DYNAMIC_DIM,
        }
    }
}

impl<B: Backend> Encoder<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        assert!(
            config.d_model >= 3 * config.d_id,
            "d_model must be ≥ 3·d_id (§1.4.1: the Pokémon token carries 3 × d_id)"
        );
        let widths = InputWidths::of(config);
        Self {
            project_global: LinearConfig::new(widths.global, config.d_model).init(device),
            project_pokemon: LinearConfig::new(widths.pokemon, config.d_model).init(device),
            project_attack: LinearConfig::new(widths.attack, config.d_model).init(device),
            project_trainer: LinearConfig::new(widths.trainer, config.d_model).init(device),
            project_history: LinearConfig::new(widths.history, config.d_model).init(device),
            token_type: Param::from_tensor(Tensor::random(
                [TOKEN_TYPES, config.d_model],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: TransformerEncoderConfig::new(
                config.d_model,
                config.d_ff,
                config.num_heads,
                config.num_blocks,
            )
            .with_norm_first(true)
            .with_dropout(0.0)
            .init(device),
            ln_final: LayerNormConfig::new(config.d_model).init(device),
        }
    }

    /// The blocks' own parameter count — see the equivalence test below.
    #[cfg(test)]
    pub fn block_params(&self) -> usize {
        use burn::module::Module;
        self.blocks.num_params()
    }

    /// The token-type tag of one family, broadcastable over its block.
    pub fn type_tag(&self, family: usize) -> Tensor<B, 3> {
        self.token_type
            .val()
            .slice([family..family + 1])
            .unsqueeze_dim(0)
    }

    /// Run the blocks over an assembled sequence. `mask: [batch × n]`, `1.0` for real slots —
    /// inverted here into burn's `mask_pad`, which marks the *padding*.
    pub fn forward(&self, tokens: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let padding = mask.equal_elem(0.0);
        let encoded = self
            .blocks
            .forward(TransformerEncoderInput::new(tokens).mask_pad(padding));
        self.ln_final.forward(encoded)
    }

    /// Every block's forward, in block order, with the halves `forward` discards.
    ///
    /// `TransformerEncoder::forward` keeps only the encoded sequence, so the attention weights have
    /// to be recomputed — one extra attention per block. The alternative, replaying the block body
    /// here and keeping both halves of its output, is what this deliberately does not do: the
    /// residual and FFN path would then exist twice, and a divergence between our copy and burn's
    /// would be invisible, since these are read by diagnostics that have no independent expectation
    /// to fail against. Advancing `x` through burn's own `forward` keeps it the only definition —
    /// and is also why [`BlockTrace::output`] is a result rather than a reconstruction, so the write
    /// read off it is the write that actually happened.
    ///
    /// The normalization mirrors `norm_first = true`, where burn applies `norm_2` before the
    /// attention and `norm_1` before the FFN — the reverse of what the field names suggest.
    pub fn trace(&self, tokens: Tensor<B, 3>, mask: Tensor<B, 2>) -> Vec<BlockTrace<B>> {
        let padding = mask.equal_elem(0.0);
        let layers = &self.blocks.layers;
        let mut x = tokens;
        let mut trace = Vec::with_capacity(layers.len());
        for layer in layers.iter() {
            let normed = layer.norm_2.forward(x.clone());
            let attended = layer
                .mha
                .forward(MhaInput::self_attn(normed).mask_pad(padding.clone()));
            let output = layer.forward(x.clone(), Some(padding.clone()), None);
            trace.push(BlockTrace {
                weights: attended.weights,
                attention: attended.context,
                input: x,
                output: output.clone(),
            });
            x = output;
        }
        trace
    }

    /// Trainable parameters, split where §1.4.3's FLOP argument splits: what projects the five
    /// families in, and what mixes them afterwards.
    pub fn parameter_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "input projections",
                self.project_global.num_params()
                    + self.project_pokemon.num_params()
                    + self.project_attack.num_params()
                    + self.project_trainer.num_params()
                    + self.project_history.num_params()
                    + self.token_type.num_params(),
            ),
            (
                "encoder blocks",
                self.blocks.num_params() + self.ln_final.num_params(),
            ),
        ]
    }

    /// Disable part of one block's attention in place, to measure what it was contributing.
    ///
    /// Weight surgery rather than a branch in [`Self::forward`], because the alternative is a second
    /// copy of the block body — the duplication [`Self::trace`] refuses for the same reason. Both
    /// ablations fall out of the arithmetic exactly:
    ///
    /// - [`AttentionAblation::UniformPattern`] zeroes query and key, so every score is `0`, every
    ///   pair scores alike, and the softmax is *exactly* uniform over the unmasked keys — burn masks
    ///   the padded ones to `min_float` first, so they stay out. Value and output are untouched, so
    ///   this removes only what the drift measurement says block 0 never learned: where it looks.
    /// - [`AttentionAblation::Silent`] zeroes the output projection, so the sublayer writes nothing
    ///   into the residual stream and the block is its feed-forward alone.
    pub fn ablate(&mut self, block: usize, ablation: AttentionAblation) {
        let mha = &mut self.blocks.layers[block].mha;
        let silence = |linear: &mut Linear<B>| {
            linear.weight = Param::from_tensor(linear.weight.val().zeros_like());
            linear.bias = linear
                .bias
                .take()
                .map(|bias| Param::from_tensor(bias.val().zeros_like()));
        };
        match ablation {
            AttentionAblation::UniformPattern => {
                silence(&mut mha.query);
                silence(&mut mha.key);
            }
            AttentionAblation::Silent => silence(&mut mha.output),
        }
    }

    /// Each block's drift from the same block of `reference`, in block order.
    pub fn drift(&self, reference: &Self) -> Vec<BlockDrift> {
        self.blocks
            .layers
            .iter()
            .zip(&reference.blocks.layers)
            .enumerate()
            .map(|(block, (current, reference))| BlockDrift {
                block,
                pattern: relative_drift(&[
                    drift_sums(&current.mha.query, &reference.mha.query),
                    drift_sums(&current.mha.key, &reference.mha.key),
                ]),
                value: relative_drift(&[
                    drift_sums(&current.mha.value, &reference.mha.value),
                    drift_sums(&current.mha.output, &reference.mha.output),
                ]),
                feed_forward: relative_drift(&[drift_sums(&current.pwff, &reference.pwff)]),
            })
            .collect()
    }
}

/// What [`Encoder::ablate`] takes away from a block's attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionAblation {
    /// Keep the sublayer, force its attention uniform over the real tokens.
    UniformPattern,
    /// Keep the pattern, stop the sublayer writing.
    Silent,
}

/// How far one block's parameters have moved from another copy of the same architecture.
///
/// Split three ways because "did this block learn" and "did it learn to *look* somewhere else" are
/// different questions, and a diffuse block answers them differently: query and key decide where
/// the heads look, value and output decide what gets written once they have looked, and a block
/// pooling uniformly can rewrite the second indefinitely while the first sits still. Lumping them
/// would report that block as learning, which is true, and hide what it learned, which is the
/// finding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockDrift {
    pub block: usize,
    /// `‖Δ‖ / ‖reference‖` over the query and key projections together — the attention *pattern*,
    /// and so the half [`super::introspect`]'s entropy read-out describes.
    pub pattern: f64,
    /// The same over the value and output projections.
    pub value: f64,
    pub feed_forward: f64,
}

/// Every float parameter of a subtree, flattened, in visit order — a stable order for two modules
/// of the same shape, and the only thing that makes pairing them meaningful.
struct Flatten<B: Backend> {
    values: Vec<Vec<f32>>,
    backend: core::marker::PhantomData<B>,
}

impl<B: Backend> Default for Flatten<B> {
    fn default() -> Self {
        Flatten {
            values: Vec::new(),
            backend: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for Flatten<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.values.push(
            param
                .val()
                .to_data()
                .to_vec::<f32>()
                .expect("parameters are f32"),
        );
    }
}

/// `(‖current − reference‖², ‖reference‖²)` over a subtree, so callers can pool subtrees before
/// taking the ratio — `‖Δ‖/‖ref‖` of two projections together is not the mean of theirs apart.
fn drift_sums<B: Backend, M: Module<B>>(current: &M, reference: &M) -> (f64, f64) {
    let flatten = |module: &M| {
        let mut visitor = Flatten::<B>::default();
        module.visit(&mut visitor);
        visitor.values
    };
    let (current, reference) = (flatten(current), flatten(reference));

    let (mut delta, mut scale) = (0.0f64, 0.0f64);
    for (current, reference) in current.iter().zip(&reference) {
        for (current, reference) in current.iter().zip(reference) {
            delta += f64::from(current - reference).powi(2);
            scale += f64::from(*reference).powi(2);
        }
    }
    (delta, scale)
}

/// Pooled `‖Δ‖ / ‖reference‖`.
fn relative_drift(sums: &[(f64, f64)]) -> f64 {
    let delta: f64 = sums.iter().map(|(delta, _)| delta).sum();
    let scale: f64 = sums.iter().map(|(_, scale)| scale).sum();
    // A reference of exactly zero has no scale to be relative to. It cannot happen for an
    // initialized block; returning `0` rather than an infinity keeps a caller's table readable if
    // it ever does.
    if scale == 0.0 {
        0.0
    } else {
        (delta / scale).sqrt()
    }
}

/// One block's forward, kept rather than discarded.
///
/// The stream on both sides of the block, so what the block wrote into it is a subtraction and not
/// a second implementation of the block.
pub struct BlockTrace<B: Backend> {
    /// `[batch × heads × query × key]`.
    pub weights: Tensor<B, 4>,
    /// The attention sublayer's output, before it is added back — `[batch × n × d_model]`.
    pub attention: Tensor<B, 3>,
    /// The stream as the block received it.
    pub input: Tensor<B, 3>,
    /// The stream as the block left it.
    pub output: Tensor<B, 3>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    /// Delegating the blocks to `burn::nn::transformer::TransformerEncoder` must not change the
    /// model. Per Pre-LN block: 4 × (d² + d) attention projections, 2 LayerNorm (2d each),
    /// (d·d_ff + d_ff) + (d_ff·d + d) FFN — the arithmetic behind §1.4.3's "bulk = the 4 encoder
    /// blocks ≈ 3.2 M". A drift here means the swap silently changed the architecture.
    #[test]
    fn the_blocks_carry_exactly_the_pre_ln_parameter_count() {
        let config = ModelConfig::default();
        let (d, d_ff) = (config.d_model, config.d_ff);
        let per_block = 4 * (d * d + d) + 2 * (2 * d) + (d * d_ff + d_ff) + (d_ff * d + d);
        assert_eq!(per_block, 297_024);
        let encoder = Encoder::<NdArray>::new(&config, &Default::default());
        assert_eq!(encoder.block_params(), config.num_blocks * per_block);
        assert_eq!(config.num_blocks * per_block, 594_048);
    }
}
