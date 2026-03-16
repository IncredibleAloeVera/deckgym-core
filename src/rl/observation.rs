// Observation Tensor Generation - V5fix: Attached Tool Embedding
// V5fix adds 128-dim tool text embedding on played/static cards (attached_tool support).

use lazy_static::lazy_static;
use serde::Deserialize;
use std::collections::HashMap;

use crate::hooks::get_retreat_cost;
use crate::models::{Card, EnergyType, PlayedCard};
use crate::state::State;

// =============================================================================
// STATIC DATA LOADING
// =============================================================================

#[derive(Deserialize)]
struct TextData {
    embedding: Vec<f32>,
    type_refs: Vec<f32>,
    mech_refs: Vec<f32>,
}

#[derive(Deserialize)]
struct CardFeatures {
    ability: TextData,
    atk1: TextData,
    atk2: TextData,
    supporter: TextData,
    line_size: u8,
    is_final_stage: u8,
}

lazy_static! {
    static ref CARD_FEATURES_MAP: HashMap<String, CardFeatures> = {
        let json_str = include_str!("generated/card_features.json");
        serde_json::from_str(json_str).expect("Failed to parse card_features.json")
    };
    static ref EMPTY_EMBEDDING: Vec<f32> = vec![0.0; 128];
    static ref EMPTY_TYPE_REFS: Vec<f32> = vec![0.0; 9];
    static ref EMPTY_MECH_REFS: Vec<f32> = vec![0.0; 11];
}

fn encode_meta(card_id: &str, ready: f32, obs: &mut Vec<f32>) {
    let (ls, fs) = if let Some(feat) = CARD_FEATURES_MAP.get(card_id) {
        (feat.line_size, feat.is_final_stage)
    } else {
        (0, 0)
    };

    // Line size one-hot (None, 1, 2, 3)
    obs.push(if ls == 0 { 1.0 } else { 0.0 });
    obs.push(if ls == 1 { 1.0 } else { 0.0 });
    obs.push(if ls == 2 { 1.0 } else { 0.0 });
    obs.push(if ls == 3 { 1.0 } else { 0.0 });

    // Final stage one-hot (None, False, True)
    obs.push(if fs == 0 { 1.0 } else { 0.0 });
    obs.push(if fs == 1 { 1.0 } else { 0.0 });
    obs.push(if fs == 2 { 1.0 } else { 0.0 });

    obs.push(ready);
}

// =============================================================================
// CONSTANTS
// =============================================================================

pub const MAX_CARDS_IN_OBS: usize = 40;

// --- Feature Dimensions ---

pub const FEATURES_PER_CARD: usize = 731;
// Breakdown:
// 1 (hp) + 11 (types) + 9 (weakness) + 8 (flags: ex, mega, pokemon, tool, trainer, item, stadium, fossil)
// + 8 (meta: line_size (4: None,1,2,3), is_final (3: None,F,T), ready) + 1 (retreat) + 4 (status)
// + 138 (atk1: 1 dmg, 128 emb, 9 en) + 138 (atk2: 1 dmg, 128 emb, 9 en)
// + 128 (talent emb) + 4 (location one-hot) + 4 (slot one-hot) + 1 (allied) + 128 (supporter emb)
// + 128 (attached tool emb) + 9 (type references) + 11 (mechanic references)
// = 731

pub const GLOBAL_FEATURES: usize = 1   // turn count
    + 3                                 // points (diff, self, opponent)
    + 2                                 // deck sizes (self, opponent)
    + 2                                 // hand sizes (self, opponent)
    + 2                                 // discard sizes (self, opponent)
    + 32                                // energy generated (2 players * 2 slots * 8 units)
    + 1                                 // has_stadium flag
    + 128; // stadium text embedding

pub const OBSERVATION_SIZE: usize = GLOBAL_FEATURES + (MAX_CARDS_IN_OBS * FEATURES_PER_CARD);

// Position constants
const LOC_DECK: usize = 0;
const LOC_BOARD: usize = 1;
const LOC_HAND: usize = 2;
const LOC_DISCARD: usize = 3;

// =============================================================================
// MAIN OBSERVATION FUNCTION
// =============================================================================

pub fn get_observation_tensor(state: &State, perspective: usize) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBSERVATION_SIZE);
    let opponent = 1 - perspective;

    // --- Global Features ---
    encode_global_features(state, perspective, &mut obs);

    // --- Collect Cards to Encode ---
    let mut cards_to_encode = Vec::new();

    // 1. Perspective Board (Priority 0-3)
    for slot in 0..4 {
        if let Some(p) = &state.in_play_pokemon[perspective][slot] {
            cards_to_encode.push((Some(p), None, LOC_BOARD, 1.0, Some(slot)));
        }
    }

    // 2. Perspective Hand
    for card in &state.hands[perspective] {
        cards_to_encode.push((None, Some(card), LOC_HAND, 1.0, None));
    }

    // 3. Perspective Discard
    for card in state.discard_piles[perspective].iter().rev().take(10) {
        cards_to_encode.push((None, Some(card), LOC_DISCARD, 1.0, None));
    }

    // 4. Opponent Board
    for slot in 0..4 {
        if let Some(p) = &state.in_play_pokemon[opponent][slot] {
            cards_to_encode.push((Some(p), None, LOC_BOARD, 0.0, Some(slot)));
        }
    }

    // 5. Opponent Discard
    for card in state.discard_piles[opponent].iter().rev().take(10) {
        cards_to_encode.push((None, Some(card), LOC_DISCARD, 0.0, None));
    }

    // 6. Perspective Deck (rest)
    for card in &state.decks[perspective].cards {
        cards_to_encode.push((None, Some(card), LOC_DECK, 1.0, None));
    }

    // 7. Opponent Deck (rest, if visible/needed)
    for card in &state.decks[opponent].cards {
        cards_to_encode.push((None, Some(card), LOC_DECK, 0.0, None));
    }

    // --- Encode 40 Cards ---
    let mut card_count = 0;
    for (played, card, loc, allied, slot) in cards_to_encode.into_iter().take(MAX_CARDS_IN_OBS) {
        if let Some(p) = played {
            encode_played_card(p, state, loc, allied, slot, &mut obs);
        } else if let Some(c) = card {
            encode_static_card(c, loc, allied, slot, &mut obs);
        }
        card_count += 1;
    }

    // Pad remaining
    while card_count < MAX_CARDS_IN_OBS {
        obs.extend(vec![0.0; FEATURES_PER_CARD]);
        card_count += 1;
    }

    assert_eq!(
        obs.len(),
        OBSERVATION_SIZE,
        "Observation size mismatch! Expected {}, got {}",
        OBSERVATION_SIZE,
        obs.len()
    );
    obs
}

// =============================================================================
// ENCODING HELPERS
// =============================================================================

fn encode_global_features(state: &State, perspective: usize, obs: &mut Vec<f32>) {
    let opponent = 1 - perspective;

    // Turn count (norm /30)
    obs.push((state.turn_count as f32 / 30.0).clamp(0.0, 1.0));

    // Points
    let p_pts = state.points[perspective] as f32;
    let o_pts = state.points[opponent] as f32;
    obs.push(((p_pts - o_pts) / 2.0).clamp(-1.0, 1.0));
    obs.push(p_pts / 2.0);
    obs.push(o_pts / 2.0);

    // Sizes
    obs.push((state.decks[perspective].cards.len() as f32 / 15.0).clamp(0.0, 1.0));
    obs.push((state.decks[opponent].cards.len() as f32 / 15.0).clamp(0.0, 1.0));
    obs.push((state.hands[perspective].len() as f32 / 10.0).clamp(0.0, 1.0));
    obs.push((state.hands[opponent].len() as f32 / 10.0).clamp(0.0, 1.0));
    obs.push((state.discard_piles[perspective].len() as f32 / 15.0).clamp(0.0, 1.0));
    obs.push((state.discard_piles[opponent].len() as f32 / 15.0).clamp(0.0, 1.0));

    // Energy Generated (32 dims)
    // 2 players * 2 types max * 8 energies
    for p in [perspective, opponent] {
        let types = &state.decks[p].energy_types;
        for i in 0..2 {
            let e_type = types.get(i);
            for e_check in deck_energy_types() {
                obs.push(if e_type == Some(&e_check) { 1.0 } else { 0.0 });
            }
        }
    }

    // Stadium encoding (129 dims: 1 flag + 128 embedding)
    if let Some(stadium_card) = &state.active_stadium {
        obs.push(1.0); // has_stadium flag
        let card_id = stadium_card.get_id();
        if let Some(f) = CARD_FEATURES_MAP.get(&card_id) {
            obs.extend_from_slice(&f.supporter.embedding);
        } else {
            obs.extend_from_slice(&EMPTY_EMBEDDING);
        }
    } else {
        obs.push(0.0); // no stadium
        obs.extend_from_slice(&EMPTY_EMBEDDING);
    }
}

fn encode_played_card(
    p: &PlayedCard,
    state: &State,
    loc: usize,
    allied: f32,
    slot: Option<usize>,
    obs: &mut Vec<f32>,
) {
    let card_id = p.get_id();
    let mut type_refs = [0.0; 9];
    let mut mech_refs = [0.0; 11];

    // Properties
    obs.push(p.get_remaining_hp() as f32); // HP raw
    encode_type_one_hot(p.get_energy_type(), obs); // Types 11
    encode_weakness_one_hot(p.card.get_weakness(), obs); // Weakness 9

    // Flags 8
    obs.push(if p.card.is_ex() { 1.0 } else { 0.0 });
    obs.push(if p.card.is_mega() { 1.0 } else { 0.0 });
    obs.push(if matches!(p.card, Card::Pokemon(_)) {
        1.0
    } else {
        0.0
    });
    obs.push(p.card.is_tool() as i32 as f32);
    obs.push(if matches!(p.card, Card::Trainer(_)) {
        1.0
    } else {
        0.0
    });
    obs.push(p.card.is_item() as i32 as f32);
    obs.push(p.card.is_stadium() as i32 as f32);
    obs.push(p.card.is_fossil() as i32 as f32);

    // Meta 8
    let ready = if p.played_this_turn { 0.0 } else { 1.0 };
    encode_meta(&card_id, ready, obs);

    obs.push((get_retreat_cost(state, p).len() as f32 / 4.0).clamp(0.0, 1.0)); // Retreat

    // Status 4
    obs.push(if p.confused { 1.0 } else { 0.0 });
    obs.push(if p.burned { 1.0 } else { 0.0 });
    obs.push(if p.asleep { 1.0 } else { 0.0 });
    obs.push(if p.paralyzed { 1.0 } else { 0.0 });

    let features = CARD_FEATURES_MAP.get(&card_id);

    // Attacks 2 x 74 = 148
    let attacks = p.get_attacks();
    for i in 0..2 {
        if let Some(atk) = attacks.get(i) {
            obs.push(atk.fixed_damage as f32);
            let data = match (i, features) {
                (0, Some(f)) => &f.atk1,
                (1, Some(f)) => &f.atk2,
                _ => unreachable!(),
            };
            obs.extend_from_slice(&data.embedding);
            merge_refs(&mut type_refs, &data.type_refs);
            merge_refs(&mut mech_refs, &data.mech_refs);
            encode_energy_cost_9(&atk.energy_required, obs);
        } else {
            obs.extend_from_slice(&[0.0; 138]);
        }
    }

    // Talent 64
    if let Card::Pokemon(_) = &p.card {
        if let Some(f) = features {
            obs.extend_from_slice(&f.ability.embedding);
            merge_refs(&mut type_refs, &f.ability.type_refs);
            merge_refs(&mut mech_refs, &f.ability.mech_refs);
        } else {
            obs.extend_from_slice(&EMPTY_EMBEDDING);
        }
    } else {
        obs.extend_from_slice(&EMPTY_EMBEDDING);
    }

    // Position 4 + 4 + 1 = 9
    for i in 0..4 {
        obs.push(if i == loc { 1.0 } else { 0.0 });
    }
    for i in 0..4 {
        obs.push(if slot == Some(i) { 1.0 } else { 0.0 });
    }
    obs.push(allied);

    // Supporter embedding 128 — skip for fossils (identical text, no information)
    let is_fossil = p.card.is_fossil();
    if !is_fossil {
        if let (Card::Trainer(_), Some(f)) = (&p.card, features) {
            obs.extend_from_slice(&f.supporter.embedding);
            merge_refs(&mut type_refs, &f.supporter.type_refs);
            merge_refs(&mut mech_refs, &f.supporter.mech_refs);
        } else {
            obs.extend_from_slice(&EMPTY_EMBEDDING);
        }
    } else {
        obs.extend_from_slice(&EMPTY_EMBEDDING);
    }

    // Attached tool embedding 128
    if let Some(tool) = &p.attached_tool {
        let tool_id = tool.get_id();
        if let Some(f) = CARD_FEATURES_MAP.get(&tool_id) {
            obs.extend_from_slice(&f.supporter.embedding);
            merge_refs(&mut type_refs, &f.supporter.type_refs);
            merge_refs(&mut mech_refs, &f.supporter.mech_refs);
        } else {
            obs.extend_from_slice(&EMPTY_EMBEDDING);
        }
    } else {
        obs.extend_from_slice(&EMPTY_EMBEDDING);
    }

    // References 9 + 11 = 20
    obs.extend_from_slice(&type_refs);
    obs.extend_from_slice(&mech_refs);
}

fn encode_static_card(c: &Card, loc: usize, allied: f32, slot: Option<usize>, obs: &mut Vec<f32>) {
    let card_id = c.get_id();
    let mut type_refs = [0.0; 9];
    let mut mech_refs = [0.0; 11];

    match c {
        Card::Pokemon(pk) => {
            obs.push(pk.hp as f32);
            encode_type_one_hot(Some(pk.energy_type), obs);
            encode_weakness_one_hot(pk.weakness, obs);

            obs.push(if c.is_ex() { 1.0 } else { 0.0 });
            obs.push(if c.is_mega() { 1.0 } else { 0.0 });
            obs.push(1.0); // is_pokemon
            obs.push(0.0); // is_tool
            obs.push(0.0); // is_trainer
            obs.push(0.0); // is_item
            obs.push(0.0); // is_stadium
            obs.push(0.0); // is_fossil

            // Meta 8
            encode_meta(&card_id, 0.0, obs);
            obs.push((pk.retreat_cost.len() as f32 / 4.0).clamp(0.0, 1.0));
            obs.extend_from_slice(&[0.0; 4]); // no status

            let features = CARD_FEATURES_MAP.get(&card_id);

            for i in 0..2 {
                if let Some(atk) = pk.attacks.get(i) {
                    obs.push(atk.fixed_damage as f32);
                    let data = match (i, features) {
                        (0, Some(f)) => &f.atk1,
                        (1, Some(f)) => &f.atk2,
                        _ => unreachable!(),
                    };
                    obs.extend_from_slice(&data.embedding);
                    merge_refs(&mut type_refs, &data.type_refs);
                    merge_refs(&mut mech_refs, &data.mech_refs);
                    encode_energy_cost_9(&atk.energy_required, obs);
                } else {
                    obs.extend_from_slice(&[0.0; 138]);
                }
            }
            if let Some(f) = features {
                obs.extend_from_slice(&f.ability.embedding);
                merge_refs(&mut type_refs, &f.ability.type_refs);
                merge_refs(&mut mech_refs, &f.ability.mech_refs);
            } else {
                obs.extend_from_slice(&EMPTY_EMBEDDING);
            }
        }
        Card::Trainer(t) if t.trainer_card_type == crate::models::TrainerType::Fossil => {
            // Fossils are special Pokemon: Colorless type, Fighting weakness,
            // fixed 40 HP, no attacks, but meaningful evolution line metadata
            obs.push(40.0); // HP always 40
            encode_type_one_hot(Some(EnergyType::Colorless), obs);
            encode_weakness_one_hot(Some(EnergyType::Fighting), obs);

            obs.push(0.0); // ex
            obs.push(0.0); // mega
            obs.push(0.0); // pokemon
            obs.push(0.0); // tool
            obs.push(1.0); // trainer
            obs.push(0.0); // item
            obs.push(0.0); // stadium
            obs.push(1.0); // fossil

            // Meta 8 — fossils have meaningful line_size and is_final_stage
            encode_meta(&card_id, 0.0, obs);
            obs.push(0.0); // retreat
            obs.extend_from_slice(&[0.0; 4]); // no status
            obs.extend_from_slice(&[0.0; 276]); // no attacks
            obs.extend_from_slice(&EMPTY_EMBEDDING); // no talent
        }
        Card::Trainer(_) => {
            obs.push(0.0); // hp
            encode_type_one_hot(None, obs);
            encode_weakness_one_hot(None, obs);

            obs.push(0.0); // ex
            obs.push(0.0); // mega
            obs.push(0.0); // pokemon
            obs.push(c.is_tool() as i32 as f32);
            obs.push(1.0); // trainer
            obs.push(c.is_item() as i32 as f32);
            obs.push(c.is_stadium() as i32 as f32);
            obs.push(0.0); // fossil

            // Meta 8 + Retreat 1
            encode_meta(&card_id, 0.0, obs);
            obs.push(0.0); // retreat
            obs.extend_from_slice(&[0.0; 4]); // no status
            obs.extend_from_slice(&[0.0; 276]); // no attacks
            obs.extend_from_slice(&EMPTY_EMBEDDING); // no talent
        }
    }

    // Position
    for i in 0..4 {
        obs.push(if i == loc { 1.0 } else { 0.0 });
    }
    for i in 0..4 {
        obs.push(if slot == Some(i) { 1.0 } else { 0.0 });
    }
    obs.push(allied);

    // Supporter embedding 128 — skip for fossils (identical text, no information)
    let is_fossil_card = c.is_fossil();
    if !is_fossil_card {
        if let (Card::Trainer(_), Some(f)) = (c, CARD_FEATURES_MAP.get(&card_id)) {
            obs.extend_from_slice(&f.supporter.embedding);
            merge_refs(&mut type_refs, &f.supporter.type_refs);
            merge_refs(&mut mech_refs, &f.supporter.mech_refs);
        } else {
            obs.extend_from_slice(&EMPTY_EMBEDDING);
        }
    } else {
        obs.extend_from_slice(&EMPTY_EMBEDDING);
    }

    // Attached tool embedding 128 (always empty for static cards)
    obs.extend_from_slice(&EMPTY_EMBEDDING);

    // References 9 + 11 = 20
    obs.extend_from_slice(&type_refs);
    obs.extend_from_slice(&mech_refs);
}

// --- Utils ---

fn merge_refs<const N: usize>(target: &mut [f32; N], source: &[f32]) {
    for i in 0..N.min(source.len()) {
        if source[i] > 0.5 {
            target[i] = 1.0;
        }
    }
}

fn encode_type_one_hot(t: Option<EnergyType>, obs: &mut Vec<f32>) {
    let types = all_energy_types();
    for et in types {
        obs.push(if t == Some(et) { 1.0 } else { 0.0 });
    }
    obs.push(if t.is_none() { 1.0 } else { 0.0 }); // None
}

fn encode_weakness_one_hot(t: Option<EnergyType>, obs: &mut Vec<f32>) {
    let types = deck_energy_types(); // Only 8 types for weakness
    for et in types {
        obs.push(if t == Some(et) { 1.0 } else { 0.0 });
    }
    obs.push(if t.is_none() { 1.0 } else { 0.0 }); // None
}

fn encode_energy_cost_9(costs: &[EnergyType], obs: &mut Vec<f32>) {
    let mut counts = [0.0; 9];
    for e in costs {
        if let Some(idx) = energy_to_index_9(*e) {
            counts[idx] += 1.0;
        }
    }
    for c in counts {
        obs.push(c / 4.0);
    }
}

fn energy_to_index_9(e: EnergyType) -> Option<usize> {
    match e {
        EnergyType::Grass => Some(0),
        EnergyType::Fire => Some(1),
        EnergyType::Water => Some(2),
        EnergyType::Lightning => Some(3),
        EnergyType::Psychic => Some(4),
        EnergyType::Fighting => Some(5),
        EnergyType::Darkness => Some(6),
        EnergyType::Metal => Some(7),
        EnergyType::Colorless => Some(8),
        EnergyType::Dragon => None, // Dragon excluded
    }
}

fn all_energy_types() -> [EnergyType; 10] {
    [
        EnergyType::Grass,
        EnergyType::Fire,
        EnergyType::Water,
        EnergyType::Lightning,
        EnergyType::Psychic,
        EnergyType::Fighting,
        EnergyType::Darkness,
        EnergyType::Metal,
        EnergyType::Dragon,
        EnergyType::Colorless,
    ]
}

fn deck_energy_types() -> [EnergyType; 8] {
    [
        EnergyType::Grass,
        EnergyType::Fire,
        EnergyType::Water,
        EnergyType::Lightning,
        EnergyType::Psychic,
        EnergyType::Fighting,
        EnergyType::Darkness,
        EnergyType::Metal,
    ]
}

// Add these helper methods to Card if missing or implement here
trait CardExt {
    fn is_tool(&self) -> bool;
    fn is_item(&self) -> bool;
    fn is_stadium(&self) -> bool;
    fn get_weakness(&self) -> Option<EnergyType>;
}

impl CardExt for Card {
    fn is_tool(&self) -> bool {
        match self {
            Card::Trainer(t) => t.trainer_card_type == crate::models::TrainerType::Tool,
            _ => false,
        }
    }
    fn is_item(&self) -> bool {
        match self {
            Card::Trainer(t) => t.trainer_card_type == crate::models::TrainerType::Item,
            _ => false,
        }
    }
    fn is_stadium(&self) -> bool {
        match self {
            Card::Trainer(t) => t.trainer_card_type == crate::models::TrainerType::Stadium,
            _ => false,
        }
    }
    fn get_weakness(&self) -> Option<EnergyType> {
        match self {
            Card::Pokemon(pk) => pk.weakness,
            _ => None,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        let state = State::default();
        assert_eq!(get_observation_tensor(&state, 0).len(), OBSERVATION_SIZE);
    }
}
