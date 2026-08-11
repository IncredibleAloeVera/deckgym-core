//! The frozen text-embedding side of the static descriptors.
//!
//! §1.2.2 initializes every identity embedding from raw mechanics *plus* a small frozen LM applied
//! to the card's text. That encoder ("super-set TCG" descriptive encoder, 128-dim, deferred in
//! §1.2.9) lives outside the engine: it is trained offline on the card-text corpus and must stay
//! **meta-neutral** — it never sees winrates or co-occurrence — and identical across the player and
//! the deckbuilder.
//!
//! This module is the seam. The static descriptors always reserve the exact slot widths the spec
//! fixes ([`EFFECT_TEXT_DIM`], [`ABILITY_TEXT_DIM`]); the values come from a [`TextEmbeddings`]
//! table that is **all zeros until an encoder is plugged in**. Dimensions are therefore stable from
//! day one, and swapping the encoder in later changes no shapes.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Width of an attack / trainer effect-text embedding.
pub const EFFECT_TEXT_DIM: usize = 128;
/// Width of an ability-text embedding (the ability block also carries a typed multi-hot, so the
/// text half is narrower).
pub const ABILITY_TEXT_DIM: usize = 48;

const ZERO_EFFECT: [f32; EFFECT_TEXT_DIM] = [0.0; EFFECT_TEXT_DIM];
const ZERO_ABILITY: [f32; ABILITY_TEXT_DIM] = [0.0; ABILITY_TEXT_DIM];

/// A frozen lookup from card text to its embedding. Missing entries (and the whole table, before
/// an encoder exists) resolve to zeros.
#[derive(Debug, Clone, Default)]
pub struct TextEmbeddings {
    effect: HashMap<String, Vec<f32>>,
    ability: HashMap<String, Vec<f32>>,
}

impl TextEmbeddings {
    /// The v1 baseline: no encoder plugged in, every text embeds to zeros.
    pub fn zeros() -> Self {
        Self::default()
    }

    /// Build from precomputed tables. Every vector must have the exact spec width.
    pub fn new(
        effect: HashMap<String, Vec<f32>>,
        ability: HashMap<String, Vec<f32>>,
    ) -> Result<Self, String> {
        check_widths(&effect, EFFECT_TEXT_DIM, "effect")?;
        check_widths(&ability, ABILITY_TEXT_DIM, "ability")?;
        Ok(Self { effect, ability })
    }

    /// Load from a JSON file shaped `{"effect": {text: [f32; 128]}, "ability": {text: [f32; 48]}}`.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let raw: HashMap<String, HashMap<String, Vec<f32>>> = serde_json::from_str(&contents)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        let mut raw = raw;
        Self::new(
            raw.remove("effect").unwrap_or_default(),
            raw.remove("ability").unwrap_or_default(),
        )
    }

    /// Effect-text embedding; zeros for absent text or an unknown string.
    pub fn effect(&self, text: Option<&str>) -> &[f32] {
        text.and_then(|text| self.effect.get(text))
            .map(Vec::as_slice)
            .unwrap_or(&ZERO_EFFECT)
    }

    /// Ability-text embedding; zeros for absent text or an unknown string.
    pub fn ability(&self, text: Option<&str>) -> &[f32] {
        text.and_then(|text| self.ability.get(text))
            .map(Vec::as_slice)
            .unwrap_or(&ZERO_ABILITY)
    }

    /// True when no encoder is plugged in (every lookup returns zeros).
    pub fn is_empty(&self) -> bool {
        self.effect.is_empty() && self.ability.is_empty()
    }
}

fn check_widths(
    table: &HashMap<String, Vec<f32>>,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    for (text, vector) in table {
        if vector.len() != expected {
            return Err(format!(
                "{label} embedding for {text:?} has width {}, expected {expected}",
                vector.len()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_text_embeds_to_zeros_of_the_right_width() {
        let table = TextEmbeddings::zeros();
        assert_eq!(table.effect(Some("anything")).len(), EFFECT_TEXT_DIM);
        assert_eq!(table.effect(None).len(), EFFECT_TEXT_DIM);
        assert_eq!(table.ability(None).len(), ABILITY_TEXT_DIM);
        assert!(table.effect(None).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn wrong_width_is_rejected() {
        let mut effect = HashMap::new();
        effect.insert("too short".to_string(), vec![0.0; 4]);
        assert!(TextEmbeddings::new(effect, HashMap::new()).is_err());
    }

    /// The frozen artifact built by `auxiliaries/text_embeddings` must cover every text the
    /// static tables will ever query: all ability effects, attack effects and trainer effects of
    /// the canonical pool resolve to a non-zero vector (a zero hit means a key drifted between
    /// `database.json` and the export).
    #[test]
    fn frozen_artifact_covers_every_pool_text() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/auxiliaries/text_embeddings/out/text_embeddings.json"
        );
        let table = TextEmbeddings::from_json_file(path).expect("frozen artifact loads");
        assert!(!table.is_empty());
        let is_zero = |v: &[f32]| v.iter().all(|x| *x == 0.0);
        for &card_id in crate::rl::ids::canonical_cards() {
            match crate::database::get_card_by_enum(card_id) {
                crate::models::Card::Pokemon(pokemon) => {
                    if let Some(ability) = &pokemon.ability {
                        assert!(
                            !is_zero(table.ability(Some(&ability.effect))),
                            "missing ability embedding for {:?}",
                            ability.effect
                        );
                    }
                    for attack in &pokemon.attacks {
                        if let Some(effect) = &attack.effect {
                            assert!(
                                !is_zero(table.effect(Some(effect))),
                                "missing effect embedding for {effect:?}"
                            );
                        }
                    }
                }
                crate::models::Card::Trainer(trainer) => {
                    if !trainer.effect.is_empty() {
                        assert!(
                            !is_zero(table.effect(Some(&trainer.effect))),
                            "missing effect embedding for {:?}",
                            trainer.effect
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn known_text_resolves() {
        let mut effect = HashMap::new();
        effect.insert("Heal 30 damage.".to_string(), vec![0.5; EFFECT_TEXT_DIM]);
        let table = TextEmbeddings::new(effect, HashMap::new()).expect("valid widths");
        assert_eq!(table.effect(Some("Heal 30 damage."))[0], 0.5);
        assert_eq!(table.effect(Some("Unknown"))[0], 0.0);
    }
}
