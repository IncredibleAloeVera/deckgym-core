// Action Mask Generation
//
// Generates a boolean mask indicating which actions are valid for the current state.
// Maps ALL SimpleAction variants to fixed indices for RL compatibility.
//
// =============================================================================
// ACTION SPACE LAYOUT v3.0 (179 indices total)
// =============================================================================
//
// SECTION 1: Board Actions (0-29)
// --------------------------------
// These are actions targeting board positions (in-play Pokemon slots 0-3).
//
//   0       : EndTurn
//   1-2     : Attack (attack index 0, 1)
//   3-5     : Retreat to bench position (1, 2, 3)
//   6-9     : UseAbility on slot (0=Active, 1-3=Bench)
//   10-13   : Attach turn energy to slot (0-3)
//   14-16   : Activate bench Pokemon (slots 1-3, not active)
//   17-20   : DiscardFossil from slot (0-3)
//   21-24   : Heal Pokemon at slot (0-3) - ability-triggered
//   25-28   : AttachFromDiscard to slot (0-3) - ability-triggered
//   29      : (padding)
//
// SECTION 2: Hand Actions (30-129)
// ---------------------------------
// These map hand cards (up to 20) to actions with optional targets.
// Formula: 30 + (hand_index * 5) + interaction_type
//
//   Hand Index 0:  30-34
//   Hand Index 1:  35-39
//   Hand Index 2:  40-44
//   ...
//   Hand Index 19: 125-129
//
// Interaction Types per hand slot:
//   +0 : Play/Evolve (no target or self-target for trainers/evolutions)
//   +1 : Target Active (slot 0)
//   +2 : Target Bench 1 (slot 1)
//   +3 : Target Bench 2 (slot 2)
//   +4 : Target Bench 3 (slot 3)
//
// SECTION 3: Resolution Actions (130-168)
// ----------------------------------------
// These handle complex multi-step actions and special cases.
//
//   130-133 : ApplyDamage to opponent slot (0-3) - attack targeting
//   134-137 : Resolution Place/Evolve to slot (0-3) - from deck search
//   138-157 : Select hand card by index (0-19) - for discard/communicate
//   158     : DrawCard
//   159     : Noop (pass)
//   160     : ApplyEeveeBagDamageBoost
//   161     : HealAllEeveeEvolutions
//   162     : MoveEnergy
//   163     : MoveAllDamage
//   164     : ShuffleOpponentSupporter
//   165     : DiscardOpponentSupporter
//   166     : Attach (non-turn energy, e.g. from ability)
//   167     : HealAndDiscardEnergy
//   168     : ReturnPokemonToHand
//   169     : UseStadium (activated stadiums like Mesagoza)
//   170-171 : UseCopiedAttack (attack_index 0, 1)
//   172-175 : ScheduleDelayedSpotDamage to opponent slot (0-3)
//   176-178 : (reserved for future expansion)
//
// =============================================================================

use std::collections::HashMap;

use crate::actions::{Action, SimpleAction};
use crate::models::Card;
use crate::move_generation::generate_possible_actions;
use crate::state::State;

// --- Constants ---

/// Total size of the discrete action space
pub const ACTION_SPACE_SIZE: usize = 179;

// --- Helper Types ---

/// Maps for resolving hand cards to action indices
struct HandMaps {
    /// Card ID -> hand index
    card_index: HashMap<String, usize>,
    /// Tool card ID -> hand index (for tools specifically)
    tool_index: HashMap<String, usize>,
}

/// Build index maps from the current player's hand.
///
/// This creates two maps:
/// - `card_index`: Maps card IDs to their position in hand
/// - `tool_index`: Maps tool card IDs to their position in hand (first occurrence only)
fn build_hand_maps(hand: &[Card]) -> HandMaps {
    let mut card_index = HashMap::with_capacity(hand.len());
    let mut tool_index = HashMap::new();

    for (idx, card) in hand.iter().enumerate() {
        let card_id = card.get_id();
        card_index.insert(card_id.clone(), idx);

        if let Card::Trainer(t) = card {
            if t.trainer_card_type == crate::models::TrainerType::Tool {
                // First occurrence wins (lowest index)
                tool_index.entry(card_id).or_insert(idx);
            }
        }
    }

    HandMaps {
        card_index,
        tool_index,
    }
}

/// Converts a SimpleAction to a canonical action index.
fn action_to_index(action: &SimpleAction, maps: &HandMaps) -> Option<usize> {
    let index = match action {
        // --- 1. Board Actions (0-29) ---
        SimpleAction::EndTurn => Some(0),
        SimpleAction::Attack(attack_idx) => Some(1 + attack_idx.min(&1)),
        SimpleAction::Retreat(bench_idx) => {
            if *bench_idx >= 1 && *bench_idx <= 3 {
                Some(3 + bench_idx - 1)
            } else {
                None
            }
        }
        SimpleAction::UseAbility { in_play_idx } => {
            if *in_play_idx <= 3 {
                Some(6 + in_play_idx)
            } else {
                None
            }
        }
        SimpleAction::Attach {
            attachments,
            is_turn_energy: true,
        } => {
            if let Some((_, _, in_play_idx)) = attachments.first() {
                if *in_play_idx <= 3 {
                    Some(10 + in_play_idx)
                } else {
                    None
                }
            } else {
                None
            }
        }
        SimpleAction::Activate { in_play_idx, .. } => {
            if *in_play_idx >= 1 && *in_play_idx <= 3 {
                Some(14 + in_play_idx - 1)
            } else if *in_play_idx == 0 {
                None
            } else {
                None
            }
        }
        SimpleAction::DiscardFossil { in_play_idx } => {
            if *in_play_idx <= 3 {
                Some(17 + in_play_idx)
            } else {
                None
            }
        }
        SimpleAction::Heal { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(21 + in_play_idx)
            } else {
                None
            }
        }
        SimpleAction::AttachFromDiscard { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(25 + in_play_idx)
            } else {
                None
            }
        }

        // --- 2. Hand Actions (30-129) ---
        // Base: 30 + (HandIdx * 5)

        // Play Trainer (No Target)
        SimpleAction::Play { trainer_card } => {
            maps.card_index.get(&trainer_card.id).and_then(|&idx| {
                if idx < 20 {
                    Some(30 + (idx * 5) + 0)
                } else {
                    None
                }
            })
        }

        // Place Pokemon (Bench)
        SimpleAction::Place(card, in_play_idx) => {
            if let Some(&idx) = maps.card_index.get(&card.get_id()) {
                if idx < 20 && *in_play_idx <= 3 {
                    // Place is targeted to a slot.
                    // 1 + slot.
                    Some(30 + (idx * 5) + 1 + in_play_idx)
                } else {
                    None
                }
            } else {
                // Resolution Place (e.g. from Deck Search)
                if *in_play_idx <= 3 {
                    Some(134 + in_play_idx)
                } else {
                    None
                }
            }
        }

        // Evolve Pokemon
        SimpleAction::Evolve {
            evolution,
            in_play_idx,
            ..
        } => {
            if let Some(&idx) = maps.card_index.get(&evolution.get_id()) {
                if idx < 20 && *in_play_idx <= 3 {
                    // 0 = Play/Evolve
                    Some(30 + (idx * 5) + 0)
                } else {
                    None
                }
            } else {
                // Resolution Evolve (e.g. from Deck Search)
                if *in_play_idx <= 3 {
                    Some(134 + in_play_idx)
                } else {
                    None
                }
            }
        }

        // Attach Tool (Targeted)
        SimpleAction::AttachTool {
            in_play_idx,
            tool_card,
        } => {
            // Try to find the tool in hand (Direct Play)
            if let Some(&idx) = maps.tool_index.get(&tool_card.get_id()) {
                if idx < 20 && *in_play_idx <= 3 {
                    // Hand Action: Play from Hand to Slot
                    Some(30 + (idx * 5) + 1 + in_play_idx)
                } else {
                    None
                }
            } else {
                // If not in hand, it's a Resolution Action (Target Selection)
                if *in_play_idx <= 3 {
                    Some(134 + in_play_idx)
                } else {
                    None
                }
            }
        }

        // --- 3. Resolution Actions (130-168) ---

        // ApplyDamage Targets (Target Selection)
        SimpleAction::ApplyDamage { targets, .. } => {
            if let Some((_, _, in_play_idx)) = targets.first() {
                if *in_play_idx <= 3 {
                    Some(130 + in_play_idx)
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Hand Resolution Actions (Select Hand Card)
        SimpleAction::DiscardOwnCards { cards } => maps
            .card_index
            .get(&cards.first().unwrap().get_id())
            .and_then(|&idx| if idx < 20 { Some(138 + idx) } else { None }),
        SimpleAction::CommunicatePokemon { hand_pokemon: card } => maps
            .card_index
            .get(&card.get_id())
            .and_then(|&idx| if idx < 20 { Some(138 + idx) } else { None }),
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => {
            // For ShufflePokemonIntoDeck, it's a Vec<Card>. Use first.
            if let Some(c) = hand_pokemon.first() {
                maps.card_index.get(&c.get_id()).and_then(|&idx| {
                    if idx < 20 {
                        Some(138 + idx)
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        }

        // Misc
        SimpleAction::DrawCard { .. } => Some(158),
        SimpleAction::Noop => Some(159),
        SimpleAction::ApplyEeveeBagDamageBoost => Some(160),
        SimpleAction::HealAllEeveeEvolutions => Some(161),
        SimpleAction::MoveEnergy { .. } => Some(162),
        SimpleAction::MoveAllDamage { .. } => Some(163),
        SimpleAction::ShuffleOpponentSupporter { .. } => Some(164),
        SimpleAction::DiscardOpponentSupporter { .. } => Some(165),
        SimpleAction::Attach {
            is_turn_energy: false,
            ..
        } => Some(166), // Non-turn attach
        SimpleAction::HealAndDiscardEnergy { .. } => Some(167),
        SimpleAction::ReturnPokemonToHand { .. } => Some(168),
        SimpleAction::UseStadium => Some(169),
        SimpleAction::UseCopiedAttack { attack_index, .. } => {
            if *attack_index <= 1 {
                Some(170 + attack_index)
            } else {
                None
            }
        }
        SimpleAction::ScheduleDelayedSpotDamage {
            target_in_play_idx, ..
        } => {
            if *target_in_play_idx <= 3 {
                Some(172 + target_in_play_idx)
            } else {
                None
            }
        }
    };

    if index.is_none() {
        // eprint to see it in python stdout/stderr
        eprintln!("Warning: Unmapped Action: {:?}", action);
    }

    index
}

/// Generates a boolean action mask for the current state.
pub fn get_action_mask(state: &State) -> Vec<bool> {
    let mut mask = vec![false; ACTION_SPACE_SIZE];

    let (actor, valid_actions) = generate_possible_actions(state);
    let maps = build_hand_maps(&state.hands[actor]);

    for action in &valid_actions {
        if let Some(index) = action_to_index(&action.action, &maps) {
            if index < ACTION_SPACE_SIZE {
                mask[index] = true;
            }
        }
    }

    mask
}

/// Get the list of valid actions with their indices.
pub fn get_indexed_actions(state: &State) -> Vec<(usize, Action)> {
    let (actor, valid_actions) = generate_possible_actions(state);
    let maps = build_hand_maps(&state.hands[actor]);

    valid_actions
        .into_iter()
        .filter_map(|action| action_to_index(&action.action, &maps).map(|idx| (idx, action)))
        .collect()
}
