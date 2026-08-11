//! The frozen static descriptors, gathered in-model by index — never serialized per step.
//!
//! This is principle 1 of §1.2.1: *identity is an index, not a payload*. The per-step observation
//! carries `card_id` / `species_id` / `line_id` / `tool_id`; the heavy descriptor (HP, types, costs,
//! damage, text embeddings, …) is looked up here once and held as a frozen table. It is also what
//! §1.2.1 principle 3 initializes the ID embeddings from — projecting a card's descriptor gives the
//! meta-neutral prior "these cards are mechanically similar", never "these cards are played
//! together by humans".
//!
//! Three descriptors, three widths ([`POKEMON_STATIC_DIM`] ≠ [`TRAINER_STATIC_DIM`] ≠
//! [`ATTACK_STATIC_DIM`]) — no chimera vector.

use std::collections::HashMap;
use std::sync::LazyLock;

use strum::{EnumCount, IntoEnumIterator};

use crate::actions::abilities::AbilityMechanicDiscriminants;
use crate::actions::get_ability_mechanic;
use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::models::{Attack, Card, EnergyType, TrainerCard, TrainerType};

use super::encoding::*;
use super::ids::{
    canonical_cards, card_index, card_table_size, ids_for_species_key, max_species_key_words,
};
use super::text_embedding::{TextEmbeddings, ABILITY_TEXT_DIM, EFFECT_TEXT_DIM};

/// Typed ability vocabulary width — the `AbilityMechanic` enumeration itself, so the block tracks
/// the engine instead of a hand-copied number. (§1.2.4 quoted 80, the count at the time of writing.)
pub const ABILITY_MECHANIC_DIM: usize = AbilityMechanicDiscriminants::COUNT;
/// `AbilityMechanic` multi-hot ⊕ ability text embedding.
pub const ABILITY_BLOCK_DIM: usize = ABILITY_MECHANIC_DIM + ABILITY_TEXT_DIM;

/// A printed card carries at most two attacks.
pub const MAX_ATTACKS_PER_CARD: usize = 2;

/// `fixed_damage` (42) + energy cost (10) + total energy (1) + effect text (128) = 181.
pub const ATTACK_STATIC_DIM: usize = DAMAGE_DIM + ENERGY_DIM + 1 + EFFECT_TEXT_DIM;

/// energy type (10) + HP (44) + weakness (10) + stage (3) + retreat (5) + ex/mega (2) +
/// has_ability (1) + ability block + 2 attack blocks.
pub const POKEMON_STATIC_DIM: usize = ENERGY_DIM
    + HP_DIM
    + ENERGY_DIM
    + 3
    + RETREAT_COST_DIM
    + 2
    + 1
    + ABILITY_BLOCK_DIM
    + MAX_ATTACKS_PER_CARD * ATTACK_STATIC_DIM;

/// Targeting block: type mask (10) + `targets_ex` (1) + `targets_stage` (3) + self/opp (2).
pub const TRAINER_TARGETING_DIM: usize = ENERGY_DIM + 1 + 3 + 2;
/// Number of `TrainerType` variants.
pub const TRAINER_TYPE_DIM: usize = 5;
/// trainer type (5) + effect text (128) + targeting (16) = 149.
pub const TRAINER_STATIC_DIM: usize = TRAINER_TYPE_DIM + EFFECT_TEXT_DIM + TRAINER_TARGETING_DIM;

/// A Fossil is played "as if it were a 40-HP Basic [C] Pokémon", so it rides the Pokémon schema.
const FOSSIL_HP: u32 = 40;

static MECHANIC_INDEX: LazyLock<HashMap<AbilityMechanicDiscriminants, usize>> =
    LazyLock::new(|| {
        AbilityMechanicDiscriminants::iter()
            .enumerate()
            .map(|(index, discriminant)| (discriminant, index))
            .collect()
    });

/// Position of an ability mechanic in the typed multi-hot.
pub fn ability_mechanic_index(discriminant: AbilityMechanicDiscriminants) -> usize {
    MECHANIC_INDEX[&discriminant]
}

// ---------------------------------------------------------------------------------------------
// Pokémon
// ---------------------------------------------------------------------------------------------

/// Static descriptor of a card emitted as a Pokémon token. Accepts Fossil trainer cards, which use
/// this schema (HP 40, Colorless type, Fighting weakness, no attacks); panics on any other Trainer.
pub fn pokemon_static_descriptor(card: &Card, embeddings: &TextEmbeddings) -> Vec<f32> {
    let mut out = Vec::with_capacity(POKEMON_STATIC_DIM);
    match card {
        Card::Pokemon(pokemon) => {
            push_energy_one_hot(&mut out, Some(pokemon.energy_type));
            push_hp_buckets(&mut out, pokemon.hp);
            push_energy_one_hot(&mut out, pokemon.weakness);
            push_one_hot(&mut out, Some(pokemon.stage as usize), 3);
            push_one_hot(
                &mut out,
                Some(pokemon.retreat_cost.len().min(RETREAT_COST_DIM - 1)),
                RETREAT_COST_DIM,
            );
            push_bit(&mut out, card.is_ex());
            push_bit(&mut out, card.is_mega());
            push_bit(&mut out, pokemon.ability.is_some());
            push_ability_block(&mut out, card, embeddings);
            for slot in 0..MAX_ATTACKS_PER_CARD {
                push_attack_block(&mut out, pokemon.attacks.get(slot), embeddings);
            }
        }
        Card::Trainer(trainer) => {
            assert_eq!(
                trainer.trainer_card_type,
                TrainerType::Fossil,
                "only Fossil trainers are emitted as Pokémon tokens, got {trainer:?}"
            );
            push_energy_one_hot(&mut out, Some(EnergyType::Colorless));
            push_hp_buckets(&mut out, FOSSIL_HP);
            push_energy_one_hot(&mut out, Some(EnergyType::Fighting));
            push_one_hot(&mut out, Some(0), 3); // Basic
            push_one_hot(&mut out, Some(0), RETREAT_COST_DIM); // cannot retreat
            push_bit(&mut out, false);
            push_bit(&mut out, false);
            push_bit(&mut out, false);
            push_ability_block(&mut out, card, embeddings);
            for _ in 0..MAX_ATTACKS_PER_CARD {
                push_attack_block(&mut out, None, embeddings);
            }
        }
    }
    debug_assert_eq!(out.len(), POKEMON_STATIC_DIM);
    out
}

fn push_ability_block(out: &mut Vec<f32>, card: &Card, embeddings: &TextEmbeddings) {
    let base = out.len();
    out.extend(std::iter::repeat_n(0.0, ABILITY_MECHANIC_DIM));
    if let Some(mechanic) = get_ability_mechanic(card) {
        out[base + ability_mechanic_index(mechanic.into())] = 1.0;
    }
    let ability_text = card.get_ability().map(|ability| ability.effect);
    out.extend_from_slice(embeddings.ability(ability_text.as_deref()));
}

// ---------------------------------------------------------------------------------------------
// Attack
// ---------------------------------------------------------------------------------------------

/// Static descriptor of one attack — the action-affordance satellite's frozen half (§1.2.5),
/// gathered by `(src_card_id, attack_slot)`.
pub fn attack_static_descriptor(attack: &Attack, embeddings: &TextEmbeddings) -> Vec<f32> {
    let mut out = Vec::with_capacity(ATTACK_STATIC_DIM);
    push_attack_block(&mut out, Some(attack), embeddings);
    out
}

fn push_attack_block(out: &mut Vec<f32>, attack: Option<&Attack>, embeddings: &TextEmbeddings) {
    let Some(attack) = attack else {
        out.extend(std::iter::repeat_n(0.0, ATTACK_STATIC_DIM));
        return;
    };
    push_damage_buckets(out, attack.fixed_damage);
    let cost = energy_counts(attack.energy_required.iter());
    push_energy_counts(out, &cost, ATTACK_COST_DENOM);
    push_ratio(out, attack.energy_required.len() as f32, ATTACK_COST_DENOM);
    out.extend_from_slice(embeddings.effect(attack.effect.as_deref()));
}

// ---------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------

/// Static descriptor of an Item / Supporter / Tool / Stadium card (§1.2.6).
pub fn trainer_static_descriptor(card: &TrainerCard, embeddings: &TextEmbeddings) -> Vec<f32> {
    let mut out = Vec::with_capacity(TRAINER_STATIC_DIM);
    push_one_hot(
        &mut out,
        Some(trainer_type_index(&card.trainer_card_type)),
        TRAINER_TYPE_DIM,
    );
    out.extend_from_slice(embeddings.effect(Some(&card.effect)));

    let targeting = trainer_targeting(card);
    for reachable in targeting.energy_mask {
        push_bit(&mut out, reachable);
    }
    push_bit(&mut out, targeting.targets_ex);
    for stage in targeting.targets_stage {
        push_bit(&mut out, stage);
    }
    push_bit(&mut out, targeting.targets_self);
    push_bit(&mut out, targeting.targets_opponent);

    debug_assert_eq!(out.len(), TRAINER_STATIC_DIM);
    out
}

/// Position of a trainer type in its one-hot.
pub const fn trainer_type_index(trainer_type: &TrainerType) -> usize {
    match trainer_type {
        TrainerType::Supporter => 0,
        TrainerType::Item => 1,
        TrainerType::Tool => 2,
        TrainerType::Fossil => 3,
        TrainerType::Stadium => 4,
    }
}

/// What a trainer card reaches, read off its effect text.
///
/// Until the structured schema of §1.2.9 exists, this is a **deterministic text heuristic** over
/// the frozen pool, not an authoritative parse: it is a prior, and the effect-text embedding plus
/// the ID embeddings carry the rest. It is computed once per card and cached.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainerTargeting {
    /// Energy types the card names (`[W]`, `[G]`, …).
    pub energy_mask: [bool; ENERGY_DIM],
    /// Mentions "Pokémon ex".
    pub targets_ex: bool,
    /// Mentions Basic / Stage 1 / Stage 2.
    pub targets_stage: [bool; 3],
    /// Acts on the player's own side.
    pub targets_self: bool,
    /// Acts on the opponent's side.
    pub targets_opponent: bool,
    /// The `(species_id, line_id)` pairs the card names explicitly ("your Ninetales, Rapidash, or
    /// Magmar"). Emitted on the wire — it must index the *trainable* embeddings live, which is why
    /// it is not baked into the static block.
    pub target_ids: Vec<(u32, u32)>,
}

static TRAINER_TARGETING: LazyLock<HashMap<CardId, TrainerTargeting>> = LazyLock::new(|| {
    CardId::iter()
        .filter_map(|card_id| match get_card_by_enum(card_id) {
            Card::Trainer(trainer) => Some((card_id, parse_trainer_targeting(&trainer.effect))),
            Card::Pokemon(_) => None,
        })
        .collect()
});

/// Cached targeting of a trainer card.
pub fn trainer_targeting(card: &TrainerCard) -> &'static TrainerTargeting {
    static EMPTY: LazyLock<TrainerTargeting> = LazyLock::new(TrainerTargeting::default);
    CardId::from_card_id(&card.id)
        .and_then(|card_id| TRAINER_TARGETING.get(&card_id))
        .unwrap_or(&EMPTY)
}

fn parse_trainer_targeting(effect: &str) -> TrainerTargeting {
    let mut targeting = TrainerTargeting::default();

    for (code, energy) in ENERGY_CODES {
        if effect.contains(code) {
            targeting.energy_mask[energy_index(energy)] = true;
        }
    }
    targeting.targets_ex = effect.contains("Pokémon ex");
    targeting.targets_stage = [
        effect.contains("Basic"),
        effect.contains("Stage 1"),
        effect.contains("Stage 2"),
    ];
    targeting.targets_opponent = effect.contains("opponent");
    // "your opponent's Active" is not a self-target: strip those mentions before looking for "your".
    targeting.targets_self = effect.replace("your opponent", "").contains("your");
    targeting.target_ids = scan_named_species(effect);
    targeting
}

const ENERGY_CODES: [(&str, EnergyType); ENERGY_DIM] = [
    ("[G]", EnergyType::Grass),
    ("[R]", EnergyType::Fire),
    ("[W]", EnergyType::Water),
    ("[L]", EnergyType::Lightning),
    ("[P]", EnergyType::Psychic),
    ("[F]", EnergyType::Fighting),
    ("[D]", EnergyType::Darkness),
    ("[M]", EnergyType::Metal),
    ("[N]", EnergyType::Dragon),
    ("[C]", EnergyType::Colorless),
];

/// Find every printed Pokémon named in a card's text, longest name first so "Alolan Vulpix" wins
/// over "Vulpix". Matching is on whole words, so "Mew" never matches inside "Mewtwo".
fn scan_named_species(text: &str) -> Vec<(u32, u32)> {
    let words: Vec<&str> = text
        .split(|c: char| c.is_whitespace())
        .map(trim_word)
        .filter(|word| !word.is_empty())
        .collect();
    let window = max_species_key_words();

    let mut found: Vec<(u32, u32)> = Vec::new();
    let mut cursor = 0;
    while cursor < words.len() {
        let mut consumed = 0;
        for length in (1..=window.min(words.len() - cursor)).rev() {
            let candidate = words[cursor..cursor + length].join(" ");
            if let Some(ids) = ids_for_species_key(&candidate) {
                if !found.contains(&ids) {
                    found.push(ids);
                }
                consumed = length;
                break;
            }
        }
        cursor += consumed.max(1);
    }
    found
}

fn trim_word(word: &str) -> &str {
    let word = word.trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | '"'));
    let word = word.trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']'));
    word.strip_suffix("'s").unwrap_or(word)
}

// ---------------------------------------------------------------------------------------------
// Whole-pool tables
// ---------------------------------------------------------------------------------------------

/// Row of the attack static table for `(card index, attack slot)`.
pub const fn attack_table_row(card_index: u32, attack_slot: usize) -> usize {
    card_index as usize * MAX_ATTACKS_PER_CARD + attack_slot
}

/// Build the whole frozen Pokémon descriptor table, indexed by [`card_index`], one row per
/// *canonical* printing (reprints share their original's row). Row 0 is the PAD
/// row (zeros); rows of cards that are not Pokémon-or-Fossil are zero too.
pub fn build_pokemon_static_table(embeddings: &TextEmbeddings) -> Vec<Vec<f32>> {
    let mut table = vec![vec![0.0; POKEMON_STATIC_DIM]; card_table_size()];
    for &card_id in canonical_cards() {
        let card = get_card_by_enum(card_id);
        let is_pokemon_token = matches!(&card, Card::Pokemon(_)) || card.is_fossil();
        if is_pokemon_token {
            table[card_index(card_id) as usize] = pokemon_static_descriptor(&card, embeddings);
        }
    }
    table
}

/// Build the whole frozen Trainer descriptor table, indexed by [`card_index`]. Fossils are excluded
/// (they live in the Pokémon table).
pub fn build_trainer_static_table(embeddings: &TextEmbeddings) -> Vec<Vec<f32>> {
    let mut table = vec![vec![0.0; TRAINER_STATIC_DIM]; card_table_size()];
    for &card_id in canonical_cards() {
        if let Card::Trainer(trainer) = get_card_by_enum(card_id) {
            if trainer.trainer_card_type != TrainerType::Fossil {
                table[card_index(card_id) as usize] =
                    trainer_static_descriptor(&trainer, embeddings);
            }
        }
    }
    table
}

/// Build the whole frozen Attack descriptor table, indexed by [`attack_table_row`].
pub fn build_attack_static_table(embeddings: &TextEmbeddings) -> Vec<Vec<f32>> {
    let mut table = vec![vec![0.0; ATTACK_STATIC_DIM]; card_table_size() * MAX_ATTACKS_PER_CARD];
    for &card_id in canonical_cards() {
        if let Card::Pokemon(pokemon) = get_card_by_enum(card_id) {
            for (slot, attack) in pokemon
                .attacks
                .iter()
                .take(MAX_ATTACKS_PER_CARD)
                .enumerate()
            {
                table[attack_table_row(card_index(card_id), slot)] =
                    attack_static_descriptor(attack, embeddings);
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::ids::canonical_card;

    fn embeddings() -> TextEmbeddings {
        TextEmbeddings::zeros()
    }

    /// `MAX_ATTACKS_PER_CARD` is a frozen-pool fact, not a truncation: the descriptors and the
    /// attack-token emission both `take(2)`, which is only lossless if no printed card exceeds it.
    #[test]
    fn no_card_in_the_pool_has_more_than_two_attacks() {
        for card_id in CardId::iter() {
            if let Card::Pokemon(pokemon) = get_card_by_enum(card_id) {
                assert!(
                    pokemon.attacks.len() <= MAX_ATTACKS_PER_CARD,
                    "{card_id:?} has {} attacks",
                    pokemon.attacks.len()
                );
            }
        }
    }

    #[test]
    fn descriptor_widths_match_the_spec_decomposition() {
        assert_eq!(ATTACK_STATIC_DIM, 181);
        assert_eq!(TRAINER_STATIC_DIM, 149);
        // §1.2.4's 565 assumed an 80-variant ability vocabulary; the block tracks the engine.
        assert_eq!(POKEMON_STATIC_DIM, 485 + ABILITY_MECHANIC_DIM);
    }

    #[test]
    fn every_card_in_the_pool_produces_a_descriptor_of_the_right_width() {
        let embeddings = embeddings();
        for card_id in CardId::iter() {
            let card = get_card_by_enum(card_id);
            match &card {
                Card::Pokemon(_) => {
                    assert_eq!(
                        pokemon_static_descriptor(&card, &embeddings).len(),
                        POKEMON_STATIC_DIM
                    );
                }
                Card::Trainer(trainer) => {
                    if trainer.trainer_card_type == TrainerType::Fossil {
                        assert_eq!(
                            pokemon_static_descriptor(&card, &embeddings).len(),
                            POKEMON_STATIC_DIM
                        );
                    } else {
                        assert_eq!(
                            trainer_static_descriptor(trainer, &embeddings).len(),
                            TRAINER_STATIC_DIM
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ability_multi_hot_marks_the_typed_mechanic() {
        let embeddings = embeddings();
        // Serperior's Jungle Totem is a typed mechanic.
        let serperior = get_card_by_enum(CardId::A1a006Serperior);
        let descriptor = pokemon_static_descriptor(&serperior, &embeddings);
        let base = ENERGY_DIM + HP_DIM + ENERGY_DIM + 3 + RETREAT_COST_DIM + 2 + 1;
        let mechanic_block = &descriptor[base..base + ABILITY_MECHANIC_DIM];
        assert_eq!(mechanic_block.iter().sum::<f32>(), 1.0);
        assert_eq!(
            mechanic_block[ability_mechanic_index(AbilityMechanicDiscriminants::DoubleGrassEnergy)],
            1.0
        );

        // A card without an ability leaves the whole block at zero.
        let bulbasaur = get_card_by_enum(CardId::A1001Bulbasaur);
        let descriptor = pokemon_static_descriptor(&bulbasaur, &embeddings);
        assert_eq!(
            descriptor[base..base + ABILITY_MECHANIC_DIM]
                .iter()
                .sum::<f32>(),
            0.0
        );
    }

    #[test]
    fn fossils_use_the_pokemon_schema() {
        let embeddings = embeddings();
        let old_amber = get_card_by_enum(CardId::A1218OldAmber);
        let descriptor = pokemon_static_descriptor(&old_amber, &embeddings);
        assert_eq!(descriptor.len(), POKEMON_STATIC_DIM);
        assert_eq!(descriptor[energy_index(EnergyType::Colorless)], 1.0);
        // HP 40 is the second bucket: thermometer has exactly two set bits.
        assert_eq!(
            descriptor[ENERGY_DIM..ENERGY_DIM + HP_VALUES.len()]
                .iter()
                .sum::<f32>(),
            2.0
        );
    }

    #[test]
    fn targeting_reads_named_pokemon_and_qualifiers() {
        // Blaine: "attacks used by your Ninetales, Rapidash, or Magmar do +30 damage".
        let blaine = get_card_by_enum(CardId::A1221Blaine).as_trainer();
        let targeting = trainer_targeting(&blaine);
        assert_eq!(targeting.target_ids.len(), 3);
        assert!(targeting.targets_self);
        assert!(targeting.targets_opponent, "mentions opponent's Active");

        // Erika: "Heal 50 damage from 1 of your [G] Pokémon." — one energy, no named species.
        let erika = get_card_by_enum(CardId::A1219Erika).as_trainer();
        let targeting = trainer_targeting(&erika);
        assert!(targeting.energy_mask[energy_index(EnergyType::Grass)]);
        assert!(targeting.target_ids.is_empty());
        assert!(targeting.targets_self);
        assert!(!targeting.targets_opponent);
    }

    #[test]
    fn named_species_matching_respects_word_boundaries() {
        // "Mewtwo" must not resolve to Mew.
        let mew = ids_for_species_key("Mew").expect("Mew is printed");
        assert!(!scan_named_species("Put 1 random Mewtwo into your hand.").contains(&mew));
        assert!(scan_named_species("Put 1 random Mew into your hand.").contains(&mew));
    }

    /// Reprints share one embedding row, so they must share one descriptor. This is what makes
    /// the collapse safe: if the reprint fingerprint ever missed a field a descriptor reads, two
    /// cards would land on the same row with different content and this test would catch it.
    #[test]
    fn every_reprint_shares_its_originals_descriptor() {
        let embeddings = embeddings();
        for card_id in CardId::iter() {
            let canonical = canonical_card(card_id);
            if canonical == card_id {
                continue;
            }
            let card = get_card_by_enum(card_id);
            let original = get_card_by_enum(canonical);
            match (&card, &original) {
                (Card::Pokemon(_), Card::Pokemon(_)) => assert_eq!(
                    pokemon_static_descriptor(&card, &embeddings),
                    pokemon_static_descriptor(&original, &embeddings),
                    "{card_id:?} and its original {canonical:?} disagree"
                ),
                (Card::Trainer(reprint), Card::Trainer(source)) => {
                    if reprint.trainer_card_type == TrainerType::Fossil {
                        assert_eq!(
                            pokemon_static_descriptor(&card, &embeddings),
                            pokemon_static_descriptor(&original, &embeddings),
                            "{card_id:?} and its original {canonical:?} disagree"
                        );
                    } else {
                        assert_eq!(
                            trainer_static_descriptor(reprint, &embeddings),
                            trainer_static_descriptor(source, &embeddings),
                            "{card_id:?} and its original {canonical:?} disagree"
                        );
                    }
                }
                _ => panic!("{card_id:?} collapsed onto a different card kind"),
            }
        }
    }

    /// Same for the attack descriptors, which are gathered by `(card index, attack slot)`.
    #[test]
    fn every_reprint_shares_its_originals_attacks() {
        let embeddings = embeddings();
        for card_id in CardId::iter() {
            let canonical = canonical_card(card_id);
            let (Card::Pokemon(reprint), Card::Pokemon(source)) =
                (get_card_by_enum(card_id), get_card_by_enum(canonical))
            else {
                continue;
            };
            assert_eq!(reprint.attacks.len(), source.attacks.len());
            for (slot, attack) in reprint.attacks.iter().enumerate() {
                assert_eq!(
                    attack_static_descriptor(attack, &embeddings),
                    attack_static_descriptor(&source.attacks[slot], &embeddings),
                    "{card_id:?} attack {slot} differs from its original {canonical:?}"
                );
            }
        }
    }

    #[test]
    fn static_tables_cover_the_pool() {
        let embeddings = embeddings();
        let pokemon = build_pokemon_static_table(&embeddings);
        assert_eq!(pokemon.len(), card_table_size());
        assert!(pokemon[0].iter().all(|value| *value == 0.0), "PAD row");
        assert!(pokemon[card_index(CardId::A1001Bulbasaur) as usize]
            .iter()
            .any(|value| *value != 0.0));

        let attacks = build_attack_static_table(&embeddings);
        assert_eq!(attacks.len(), card_table_size() * MAX_ATTACKS_PER_CARD);
    }
}
