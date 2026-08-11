//! The frozen in-model tables: static descriptors and the meta-neutral embedding inits.
//!
//! Two kinds of frozen data, both held as **constant tensors** (never parameters — `num_params`
//! does not count them, and no optimizer ever sees them):
//!
//! 1. **Static descriptor tables** (§1.2.1 principle 1): the Part-2 Pokémon / Trainer / Attack
//!    descriptors for the whole pool, gathered by `card_id` (and `(card_id, attack_slot)` for
//!    attacks) at every forward.
//! 2. **Meta-neutral embedding inits** (§1.2.1 principle 3, §1.2.2): `card_id[c]` is initialized
//!    by a *linear projection* of card `c`'s static descriptor to `d_id`; `species_id` / `line_id`
//!    by **mean-pooling the `card_id` inits of their member cards** — identical to pooling the
//!    descriptors then projecting, since the projection is linear. The prior therefore says
//!    "mechanically similar", never "played together by humans".
//!
//! The projections are deterministic (seeded SplitMix64 → Gaussian, variance `1/width`): the init
//! is part of the model identity, reproducible without any serialized artifact, and shared
//! bit-for-bit by the player and the (strictly frozen) deckbuilder copy.

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::database::get_card_by_enum;
use crate::models::Card;

use crate::rl::ids::{
    canonical_cards, card_index, card_table_size, line_index, line_table_size, species_index,
    species_table_size,
};
use crate::rl::static_tables::{
    build_attack_static_table, build_pokemon_static_table, build_trainer_static_table,
    ATTACK_STATIC_DIM, POKEMON_STATIC_DIM, TRAINER_STATIC_DIM,
};
use crate::rl::text_embedding::TextEmbeddings;

/// Deterministic SplitMix64 — the seeding primitive of §1.5.5, reused here for the frozen init.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]` — never 0, so `ln` below is finite.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller.
    fn next_normal(&mut self) -> f64 {
        let (u, v) = (self.next_unit(), self.next_unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A `[width × d_id]` Gaussian projection with variance `1/width`, so a multi-hot descriptor with
/// `k` set bits projects to an init of scale `≈ sqrt(k / width)`.
fn projection(rng: &mut SplitMix64, width: usize, d_id: usize) -> Vec<Vec<f32>> {
    let std = 1.0 / (width as f64).sqrt();
    (0..width)
        .map(|_| {
            (0..d_id)
                .map(|_| (rng.next_normal() * std) as f32)
                .collect()
        })
        .collect()
}

fn project(matrix: &[Vec<f32>], descriptor: &[f32], d_id: usize) -> Vec<f32> {
    let mut out = vec![0.0; d_id];
    for (value, row) in descriptor.iter().zip(matrix) {
        if *value != 0.0 {
            for (accumulator, weight) in out.iter_mut().zip(row) {
                *accumulator += value * weight;
            }
        }
    }
    out
}

/// The frozen tables, on-device. Constant tensors: gathered every forward, updated never.
#[derive(Module, Debug)]
pub struct FrozenTables<B: Backend> {
    /// `[card_table_size × POKEMON_STATIC_DIM]`, PAD row zero, non-Pokémon rows zero.
    pub pokemon_static: Tensor<B, 2>,
    /// `[card_table_size × TRAINER_STATIC_DIM]`, Fossils excluded (they are Pokémon tokens).
    pub trainer_static: Tensor<B, 2>,
    /// `[card_table_size · 2 × ATTACK_STATIC_DIM]`, row = `attack_table_row(card, slot)`.
    pub attack_static: Tensor<B, 2>,
    /// Meta-neutral `card_id` init, `[card_table_size × d_id]`, PAD row zero.
    pub card_init: Tensor<B, 2>,
    /// Meta-neutral `species_id` init (member mean-pool), `[species_table_size × d_id]`.
    pub species_init: Tensor<B, 2>,
    /// Meta-neutral `line_id` init (member mean-pool), `[line_table_size × d_id]`.
    pub line_init: Tensor<B, 2>,
}

impl<B: Backend> FrozenTables<B> {
    /// Build every frozen table from the pool. `embeddings` is the frozen text-encoder artifact
    /// (or [`TextEmbeddings::zeros`] before one is plugged in); `init_seed` fixes the projections.
    pub fn new(
        embeddings: &TextEmbeddings,
        d_id: usize,
        init_seed: u64,
        device: &B::Device,
    ) -> Self {
        let pokemon_rows = build_pokemon_static_table(embeddings);
        let trainer_rows = build_trainer_static_table(embeddings);
        let attack_rows = build_attack_static_table(embeddings);

        // Meta-neutral init: one projection per descriptor kind (the two kinds have different
        // widths, so a shared matrix cannot exist). Fossils ride the Pokémon table, as everywhere.
        let mut rng = SplitMix64(init_seed);
        let pokemon_projection = projection(&mut rng, POKEMON_STATIC_DIM, d_id);
        let trainer_projection = projection(&mut rng, TRAINER_STATIC_DIM, d_id);

        let mut card_init = vec![vec![0.0f32; d_id]; card_table_size()];
        for &card_id in canonical_cards() {
            let row = card_index(card_id) as usize;
            let card = get_card_by_enum(card_id);
            let is_pokemon_token = matches!(&card, Card::Pokemon(_)) || card.is_fossil();
            card_init[row] = if is_pokemon_token {
                project(&pokemon_projection, &pokemon_rows[row], d_id)
            } else {
                project(&trainer_projection, &trainer_rows[row], d_id)
            };
        }

        // species/line init = mean of member card inits (§1.2.2). Members are canonical rows —
        // reprints share their original's row and must not double-count it.
        let pool = |table_size: usize, index_of: fn(crate::card_ids::CardId) -> u32| {
            let mut sums = vec![vec![0.0f32; d_id]; table_size];
            let mut counts = vec![0usize; table_size];
            for &card_id in canonical_cards() {
                let target = index_of(card_id) as usize;
                counts[target] += 1;
                for (accumulator, value) in sums[target]
                    .iter_mut()
                    .zip(&card_init[card_index(card_id) as usize])
                {
                    *accumulator += value;
                }
            }
            for (sum, count) in sums.iter_mut().zip(&counts) {
                if *count > 0 {
                    for value in sum.iter_mut() {
                        *value /= *count as f32;
                    }
                }
            }
            sums
        };
        let species_init = pool(species_table_size(), species_index);
        let line_init = pool(line_table_size(), line_index);

        Self {
            pokemon_static: to_tensor(pokemon_rows, POKEMON_STATIC_DIM, device),
            trainer_static: to_tensor(trainer_rows, TRAINER_STATIC_DIM, device),
            attack_static: to_tensor(attack_rows, ATTACK_STATIC_DIM, device),
            card_init: to_tensor(card_init, d_id, device),
            species_init: to_tensor(species_init, d_id, device),
            line_init: to_tensor(line_init, d_id, device),
        }
    }
}

fn to_tensor<B: Backend>(rows: Vec<Vec<f32>>, width: usize, device: &B::Device) -> Tensor<B, 2> {
    let height = rows.len();
    let mut flat = Vec::with_capacity(height * width);
    for row in rows {
        debug_assert_eq!(row.len(), width);
        flat.extend(row);
    }
    Tensor::from_data(TensorData::new(flat, [height, width]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_ids::CardId;
    use burn::backend::NdArray;

    fn tables() -> FrozenTables<NdArray> {
        FrozenTables::new(&TextEmbeddings::zeros(), 64, 7, &Default::default())
    }

    #[test]
    fn tables_have_the_pool_shapes_and_zero_pad_rows() {
        let tables = tables();
        assert_eq!(
            tables.pokemon_static.dims(),
            [card_table_size(), POKEMON_STATIC_DIM]
        );
        assert_eq!(
            tables.attack_static.dims(),
            [card_table_size() * 2, ATTACK_STATIC_DIM]
        );
        assert_eq!(tables.card_init.dims(), [card_table_size(), 64]);
        assert_eq!(tables.species_init.dims(), [species_table_size(), 64]);
        assert_eq!(tables.line_init.dims(), [line_table_size(), 64]);

        let pad = tables.card_init.clone().slice([0..1]).abs().sum();
        assert_eq!(pad.into_scalar(), 0.0, "PAD init is zero");
    }

    #[test]
    fn the_init_is_deterministic_in_the_seed_and_sensitive_to_it() {
        let a = tables();
        let b = tables();
        assert_eq!(
            a.card_init.to_data().as_slice::<f32>().unwrap(),
            b.card_init.to_data().as_slice::<f32>().unwrap()
        );
        let c: FrozenTables<NdArray> =
            FrozenTables::new(&TextEmbeddings::zeros(), 64, 8, &Default::default());
        assert_ne!(
            a.card_init.to_data().as_slice::<f32>().unwrap(),
            c.card_init.to_data().as_slice::<f32>().unwrap()
        );
    }

    /// Mechanically similar cards start closer than dissimilar ones — the whole point of the
    /// meta-neutral prior. The projection is linear, so embedding distance tracks descriptor
    /// distance (within one descriptor kind): Pikachu must start nearer to Raichu (same line,
    /// same type, adjacent stats) than to Charizard ex (other type, other stage, other extreme
    /// of every thermometer).
    #[test]
    fn the_prior_is_mechanical_similarity() {
        let tables = tables();
        let row = |card_id: CardId| {
            let index = card_index(card_id) as usize;
            tables
                .card_init
                .clone()
                .slice([index..index + 1])
                .to_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let distance = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>()
        };

        let pikachu = row(CardId::A1094Pikachu);
        let raichu = row(CardId::A1095Raichu);
        let charizard_ex = row(CardId::A1036CharizardEx);
        assert!(distance(&pikachu, &raichu) < distance(&pikachu, &charizard_ex));
    }

    /// The species init is the member mean: a single-printing species equals its card init.
    #[test]
    fn a_singleton_species_pools_to_its_only_member() {
        let tables = tables();
        // Find a species with exactly one canonical printing.
        let mut counts = std::collections::HashMap::new();
        for &card_id in canonical_cards() {
            *counts.entry(species_index(card_id)).or_insert(0usize) += 1;
        }
        let (&species, _) = counts
            .iter()
            .find(|(_, count)| **count == 1)
            .expect("some species has a single printing");
        let member = canonical_cards()
            .iter()
            .find(|card_id| species_index(**card_id) == species)
            .unwrap();

        let species_row = tables
            .species_init
            .clone()
            .slice([species as usize..species as usize + 1])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let card_row_index = card_index(*member) as usize;
        let card_row = tables
            .card_init
            .clone()
            .slice([card_row_index..card_row_index + 1])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(species_row, card_row);
    }
}
