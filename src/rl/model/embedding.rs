//! The three identity-embedding tables (§1.2.2) and the small learned tables beside them.
//!
//! Each ID table is parametrized as **`frozen meta-neutral init ⊕ small learned residual`** — the
//! concrete form of "the player fine-tunes *its own copy*" (Part 1): the init tensor is a constant
//! (the deckbuilder's strictly-frozen copy is exactly this constant, residual-free), the residual
//! is a zero-initialized parameter whose magnitude the trainer regularizes via
//! [`IdEmbedding::residual_l2`] (§1.5.5 "weight-decay on the player embedding residuals").
//!
//! Index 0 is PAD / none / hidden in every space: [`IdEmbedding::embed`] forces row 0 to the zero
//! vector after the residual is added, so "absent" keeps one canonical encoding that no gradient
//! can drift (§1.2.2 — the zero vector encodes "none").

use burn::module::Param;
use burn::prelude::*;

/// One ID space: frozen init + learned residual, gathered by index.
#[derive(Module, Debug)]
pub struct IdEmbedding<B: Backend> {
    /// Constant: the meta-neutral init (not a parameter).
    init: Tensor<B, 2>,
    /// Learned: the player's regularized adaptation, starting at exactly zero.
    residual: Param<Tensor<B, 2>>,
}

impl<B: Backend> IdEmbedding<B> {
    pub fn new(init: Tensor<B, 2>) -> Self {
        let shape = init.dims();
        let device = init.device();
        Self {
            init,
            residual: Param::from_tensor(Tensor::zeros(shape, &device)),
        }
    }

    /// Embedding width.
    pub fn d_id(&self) -> usize {
        self.init.dims()[1]
    }

    /// Gather `[batch × slots]` indices into `[batch × slots × d_id]` embeddings.
    /// PAD (index 0) resolves to the zero vector, residual included.
    ///
    /// The two tables are gathered *then* summed, never summed then gathered: `init + residual`
    /// would materialize the whole `[vocab × d_id]` table on every call — nine times per forward,
    /// at a cost independent of the batch — to keep the ~40 rows a token bank actually indexes.
    /// The backward is a scatter-add either way.
    pub fn embed(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, slots] = indices.dims();
        let d_id = self.d_id();
        let flat = indices.reshape([batch * slots]);
        let rows =
            self.init.clone().select(0, flat.clone()) + self.residual.val().select(0, flat.clone());
        let not_pad = flat.not_equal_elem(0).float().unsqueeze_dim::<2>(1);
        (rows * not_pad).reshape([batch, slots, d_id])
    }

    /// Squared L2 norm of the learned residual — the regularization target of §1.2.2.
    pub fn residual_l2(&self) -> Tensor<B, 1> {
        self.residual.val().powi_scalar(2).sum()
    }
}

/// A plain learned table (the History `head_id` embedding, §1.2.7) — small, freely trained,
/// PAD row forced to zero at gather time like everything else.
#[derive(Module, Debug)]
pub struct LearnedEmbedding<B: Backend> {
    table: Param<Tensor<B, 2>>,
}

impl<B: Backend> LearnedEmbedding<B> {
    pub fn new(size: usize, width: usize, device: &B::Device) -> Self {
        // Small-init like an ordinary embedding table.
        let table = Tensor::random(
            [size, width],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        );
        Self {
            table: Param::from_tensor(table),
        }
    }

    pub fn embed(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, slots] = indices.dims();
        let [_, width] = self.table.dims();
        let flat = indices.reshape([batch * slots]);
        let rows = self.table.val().select(0, flat.clone());
        let not_pad = flat.not_equal_elem(0).float().unsqueeze_dim::<2>(1);
        (rows * not_pad).reshape([batch, slots, width])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    #[test]
    fn embed_gathers_init_plus_residual_and_zeroes_pad() {
        let device = Default::default();
        let init = Tensor::<NdArray, 2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0], [3, 2]),
            &device,
        );
        let table = IdEmbedding::new(init);
        let indices = Tensor::from_data(TensorData::new(vec![0i64, 2, 1], [1, 3]), &device);
        let out = table.embed(indices).to_data().to_vec::<f32>().unwrap();
        assert_eq!(out, vec![0.0, 0.0, 3.0, 4.0, 1.0, 2.0]);
        assert_eq!(
            table.residual_l2().into_scalar(),
            0.0,
            "residual starts at 0"
        );
    }
}
