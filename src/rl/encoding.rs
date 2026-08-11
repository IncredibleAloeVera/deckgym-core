//! Primitive feature encoders shared by every token family.
//!
//! These are the "shared objects" of `RL_ARCHITECTURE.md` §1.2.2: the `Energy` 10-vector, the
//! bucketed HP / damage encodings, and the count normalizations. Everything here is pure and
//! allocation-light: encoders append to a caller-owned `Vec<f32>` so a whole token can be built in
//! one buffer.

use crate::models::EnergyType;

/// `[Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dragon, Colorless]`.
/// The zero vector encodes "none" — there is no explicit `None` slot.
pub const ENERGY_DIM: usize = 10;

/// The 22 distinct printed HP values, `{30, 40, …, 240}`.
pub const HP_VALUES: [u32; 22] = [
    30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
    230, 240,
];
/// thermometer(22) ⊕ one-hot(22).
pub const HP_DIM: usize = 2 * HP_VALUES.len();

/// The 21 distinct printed `fixed_damage` values, `{0, 10, …, 180, 200, 250}`.
pub const DAMAGE_VALUES: [u32; 21] = [
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 250,
];
/// thermometer(21) ⊕ one-hot(21).
pub const DAMAGE_DIM: usize = 2 * DAMAGE_VALUES.len();

/// thermometer(21) ⊕ scalar — the encoding for *expected* (previsional) damage, which is
/// continuous and therefore has no meaningful one-hot.
pub const EXPECTED_DAMAGE_DIM: usize = DAMAGE_VALUES.len() + 1;

/// One-hot width of a base retreat cost (0..=4).
pub const RETREAT_COST_DIM: usize = 5;

// Count normalizers (§1.2.2 "Count normalization"). All ratios are clamped to `[0, 1]`
// (`[-1, 1]` for the signed retreat delta).
/// Attached energy, per type.
pub const ATTACHED_ENERGY_DENOM: f32 = 4.0;
/// Discard-pile energy, per type.
pub const DISCARD_ENERGY_DENOM: f32 = 12.0;
/// Attack cost, per type, and total energy per attack.
pub const ATTACK_COST_DENOM: f32 = 5.0;
/// Signed additional retreat cost from tools/abilities.
pub const RETREAT_DELTA_DENOM: f32 = 4.0;

/// Position of an energy type in the canonical `Energy` 10-vector.
pub const fn energy_index(energy: EnergyType) -> usize {
    match energy {
        EnergyType::Grass => 0,
        EnergyType::Fire => 1,
        EnergyType::Water => 2,
        EnergyType::Lightning => 3,
        EnergyType::Psychic => 4,
        EnergyType::Fighting => 5,
        EnergyType::Darkness => 6,
        EnergyType::Metal => 7,
        EnergyType::Dragon => 8,
        EnergyType::Colorless => 9,
    }
}

/// Tally an iterator of energies into the canonical 10-vector of raw counts.
pub fn energy_counts<'a>(energies: impl IntoIterator<Item = &'a EnergyType>) -> [u32; ENERGY_DIM] {
    let mut counts = [0u32; ENERGY_DIM];
    for energy in energies {
        counts[energy_index(*energy)] += 1;
    }
    counts
}

/// Append a boolean as `1.0` / `0.0`.
pub fn push_bit(out: &mut Vec<f32>, bit: bool) {
    out.push(if bit { 1.0 } else { 0.0 });
}

/// Append a `[0, 1]`-clamped ratio.
pub fn push_ratio(out: &mut Vec<f32>, value: f32, denominator: f32) {
    out.push((value / denominator).clamp(0.0, 1.0));
}

/// Append a `[-1, 1]`-clamped signed ratio.
pub fn push_signed_ratio(out: &mut Vec<f32>, value: f32, denominator: f32) {
    out.push((value / denominator).clamp(-1.0, 1.0));
}

/// Append a one-hot `Energy` vector; `None` appends the zero vector.
pub fn push_energy_one_hot(out: &mut Vec<f32>, energy: Option<EnergyType>) {
    let base = out.len();
    out.extend(std::iter::repeat_n(0.0, ENERGY_DIM));
    if let Some(energy) = energy {
        out[base + energy_index(energy)] = 1.0;
    }
}

/// Append normalized per-type `Energy` counts.
pub fn push_energy_counts(out: &mut Vec<f32>, counts: &[u32; ENERGY_DIM], denominator: f32) {
    for count in counts {
        push_ratio(out, *count as f32, denominator);
    }
}

/// Append a one-hot over `0..width`; out-of-range appends the zero vector.
pub fn push_one_hot(out: &mut Vec<f32>, index: Option<usize>, width: usize) {
    let base = out.len();
    out.extend(std::iter::repeat_n(0.0, width));
    if let Some(index) = index {
        if index < width {
            out[base + index] = 1.0;
        }
    }
}

/// Append `thermometer(buckets)`: `1.0` for every bucket the value reaches or exceeds.
/// Ordinal by construction — this is what carries "survives a 120-damage hit" style thresholds.
pub fn push_thermometer(out: &mut Vec<f32>, value: u32, buckets: &[u32]) {
    for bucket in buckets {
        push_bit(out, value >= *bucket);
    }
}

/// Append `one_hot(buckets)` on an *exact* bucket match; a value off the grid appends zeros.
pub fn push_bucket_one_hot(out: &mut Vec<f32>, value: u32, buckets: &[u32]) {
    for bucket in buckets {
        push_bit(out, value == *bucket);
    }
}

/// HP encoded as thermometer(22) ⊕ one-hot(22) — ordinality *and* the exact breakpoints that
/// matter on a TCG (140 HP ≠ 130 ≠ 150 for an EX).
pub fn push_hp_buckets(out: &mut Vec<f32>, hp: u32) {
    push_thermometer(out, hp, &HP_VALUES);
    push_bucket_one_hot(out, hp, &HP_VALUES);
}

/// Nominal damage encoded as thermometer(21) ⊕ one-hot(21).
pub fn push_damage_buckets(out: &mut Vec<f32>, damage: u32) {
    push_thermometer(out, damage, &DAMAGE_VALUES);
    push_bucket_one_hot(out, damage, &DAMAGE_VALUES);
}

/// Expected (continuous) damage as thermometer(21) ⊕ scalar. Off-grid values have no meaningful
/// one-hot, so the exact value rides along as a single normalized scalar instead.
pub fn push_expected_damage(out: &mut Vec<f32>, expected: f32) {
    let floor = expected.max(0.0).floor() as u32;
    push_thermometer(out, floor, &DAMAGE_VALUES);
    push_ratio(out, expected, DAMAGE_VALUES[DAMAGE_VALUES.len() - 1] as f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_one_hot_zero_vector_encodes_none() {
        let mut out = vec![];
        push_energy_one_hot(&mut out, None);
        assert_eq!(out, vec![0.0; ENERGY_DIM]);

        let mut out = vec![];
        push_energy_one_hot(&mut out, Some(EnergyType::Psychic));
        assert_eq!(out.iter().sum::<f32>(), 1.0);
        assert_eq!(out[energy_index(EnergyType::Psychic)], 1.0);
    }

    #[test]
    fn hp_buckets_are_thermometer_then_one_hot() {
        let mut out = vec![];
        push_hp_buckets(&mut out, 140);
        assert_eq!(out.len(), HP_DIM);
        // thermometer: every value <= 140 is set (30..=140 → 12 buckets).
        assert_eq!(out[..HP_VALUES.len()].iter().sum::<f32>(), 12.0);
        // one-hot: exactly one.
        assert_eq!(out[HP_VALUES.len()..].iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn damage_buckets_off_grid_value_has_no_one_hot() {
        let mut out = vec![];
        push_damage_buckets(&mut out, 195);
        assert_eq!(out.len(), DAMAGE_DIM);
        assert_eq!(out[DAMAGE_VALUES.len()..].iter().sum::<f32>(), 0.0);
        // 0..=180 (19 buckets) are all reached, 200 and 250 are not.
        assert_eq!(out[..DAMAGE_VALUES.len()].iter().sum::<f32>(), 19.0);
    }

    #[test]
    fn counts_are_clamped() {
        let mut out = vec![];
        let counts = energy_counts([EnergyType::Grass; 9].iter());
        push_energy_counts(&mut out, &counts, ATTACHED_ENERGY_DENOM);
        assert_eq!(out[energy_index(EnergyType::Grass)], 1.0);
    }
}
