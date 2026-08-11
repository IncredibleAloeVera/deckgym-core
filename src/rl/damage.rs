//! The pure damage estimator behind the Attack token's threat matrix (§1.2.5, §1.2.9).
//!
//! `estimate_damage(state, attacker, attack, defender) -> (expected, guaranteed_floor)`:
//! side-effect-free and RNG-free, weakness- and modifier-adjusted, resolving coin-flip expectation
//! analytically. It is the observation's heaviest computation — our-attacks × their-Pokémon, on
//! both sides, at every step — so it must never mutate the state or draw RNG.
//!
//! It is not a reimplementation of the damage rules: it reuses the engine's own forecast
//! (`forecast_attack_outcomes`), which carries each branch's damage as *data*, and reads the
//! distribution by inspection. Two deliberate approximations:
//!
//! - the engine forecasts an attack from the **active** spot of the **turn player**, so a benched
//!   attacker (or the off-turn player's board) is evaluated on a *projection* of the state where
//!   that Pokémon sits in the active slot and owns the turn. This is exactly the affordance the
//!   token advertises ("what this attacker threatens"), not a prediction that it will attack this
//!   turn;
//! - an attack whose effect text has no typed mechanic falls back to `fixed_damage` against the
//!   opponent's active (§1.2.9: "higher-order effects it cannot resolve statically fall back to
//!   `fixed_damage`").
//!
//! Zero is returned whenever the attack's energy is unmet or the slot is unreachable.

use crate::actions::{forecast_attack_outcomes, EFFECT_MECHANIC_MAP};
use crate::hooks::{energy_missing, get_attack_cost, modify_damage, DamageModifierContext};
use crate::models::Attack;
use crate::State;

/// Board slots per player.
pub const BOARD_SLOTS: usize = 4;

/// What one attack does to one defender, as a distribution summary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DamageEstimate {
    /// Probability-weighted modified damage (coin flips resolved analytically).
    pub expected: f32,
    /// The least modified damage across every branch — a KO predicted from it is certain.
    pub guaranteed: u32,
}

impl DamageEstimate {
    /// Whether the guaranteed floor alone knocks the defender out.
    pub fn is_lethal_against(&self, remaining_hp: u32) -> bool {
        self.guaranteed > 0 && self.guaranteed >= remaining_hp
    }
}

/// The full affordance of one attack — the dynamic half of the Attack token (§1.2.5): payability
/// and the threat row, computed on **one** projection so they cannot disagree. This is what
/// guarantees the spec's invariant "expected damage is 0 when `can_pay = 0`".
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AttackAffordance {
    /// The (modifier-adjusted) cost is met by the attacker's current effective attachment.
    pub can_pay: bool,
    /// Energies missing toward that cost.
    pub deficit: u32,
    /// Attached energies beyond that cost.
    pub surplus: u32,
    /// What the attack does to each of the four opposing board slots.
    pub threat: [DamageEstimate; BOARD_SLOTS],
}

/// Expected and guaranteed damage of `attack`, used by the Pokémon at `attacker`, against the
/// Pokémon at `defender`. Both refs are `(player, in_play_idx)`.
pub fn estimate_damage(
    state: &State,
    attacker: (usize, usize),
    attack: &Attack,
    defender: (usize, usize),
) -> DamageEstimate {
    let opponent = (attacker.0 + 1) % 2;
    if defender.0 != opponent || defender.1 >= BOARD_SLOTS {
        // Self-damage and cross-side oddities are not part of the threat matrix.
        return DamageEstimate::default();
    }
    estimate_attack_threat(state, attacker, attack)[defender.1]
}

/// The whole row of the threat matrix: what `attack` does to each of the four opposing board
/// slots. Convenience over [`estimate_attack_affordance`], which is what the observation calls.
pub fn estimate_attack_threat(
    state: &State,
    attacker: (usize, usize),
    attack: &Attack,
) -> [DamageEstimate; BOARD_SLOTS] {
    estimate_attack_affordance(state, attacker, attack).threat
}

/// A reusable projection buffer: one owned copy of the state, projected **in place** for each
/// attacker and restored right after.
///
/// This is what keeps the threat matrix affordable. The projection only ever writes two things —
/// `current_player` and a swap of two `in_play_pokemon` slots — so it is exactly reversible, and a
/// full `State::clone` per attack (the observation makes ~6 of them per decision, ≈ 14 µs each
/// because `Card` is an owned enum) buys nothing a swap cannot. Building the scratch once per
/// observation and reusing it across every attacker of both boards turns those N clones into one.
///
/// The buffer must stay an exact copy of the state it was built from; every method restores it
/// before returning, including on panic.
pub struct ProjectionScratch {
    state: State,
}

impl ProjectionScratch {
    /// Take the single copy this scratch will reuse.
    pub fn new(state: &State) -> Self {
        Self {
            state: state.clone(),
        }
    }

    /// [`estimate_attack_affordance`], evaluated on the scratch instead of a fresh clone.
    pub fn attack_affordance(
        &mut self,
        attacker: (usize, usize),
        attack: &Attack,
    ) -> AttackAffordance {
        affordance_with_projection(&mut self.state, attacker, attack)
    }
}

/// Restores a projected state when it goes out of scope, panic included — the scratch is shared
/// across every attacker of an observation, so leaving it swapped would corrupt every later token.
struct Projection<'a> {
    state: &'a mut State,
    previous_player: usize,
    swapped: Option<(usize, usize)>,
}

impl<'a> Projection<'a> {
    /// Put `attacker` in the active slot and give it the turn. Borrowing is not an option here
    /// (the caller may hold the real state), so an already-projected attacker just records no
    /// change to undo.
    fn apply(state: &'a mut State, attacker: (usize, usize)) -> Self {
        let (attacker_player, attacker_idx) = attacker;
        let previous_player = state.current_player;
        state.current_player = attacker_player;
        let swapped = if attacker_idx != 0 {
            state.in_play_pokemon[attacker_player].swap(0, attacker_idx);
            Some((attacker_player, attacker_idx))
        } else {
            None
        };
        Self {
            state,
            previous_player,
            swapped,
        }
    }

    fn get(&self) -> &State {
        self.state
    }
}

impl Drop for Projection<'_> {
    fn drop(&mut self) {
        if let Some((player, idx)) = self.swapped {
            self.state.in_play_pokemon[player].swap(0, idx);
        }
        self.state.current_player = self.previous_player;
    }
}

/// Payability *and* threat of `attack`, used by the Pokémon at `attacker`. Everything — cost
/// modifiers, energy check, damage forecast — is evaluated on the same projection (the attacker
/// in the active slot, owning the turn), never on a mix of real and projected states.
///
/// Clones the state when the attacker is not already the turn player's active. Callers evaluating
/// several attackers over one state — the observation's threat matrix — should build a
/// [`ProjectionScratch`] once instead, which pays that clone a single time.
pub fn estimate_attack_affordance(
    state: &State,
    attacker: (usize, usize),
    attack: &Attack,
) -> AttackAffordance {
    if attacker.1 == 0 && state.current_player == attacker.0 {
        // Already in the shape the forecast wants: no projection, hence no copy.
        return affordance_on_projection(state, attacker.0, attack);
    }
    let mut scratch = state.clone();
    affordance_with_projection(&mut scratch, attacker, attack)
}

/// The projecting half: mutates `state` into the attacker's frame, evaluates, restores.
/// `state` must be an exact copy of (or be) the state being observed.
fn affordance_with_projection(
    state: &mut State,
    attacker: (usize, usize),
    attack: &Attack,
) -> AttackAffordance {
    let projection = Projection::apply(state, attacker);
    affordance_on_projection(projection.get(), attacker.0, attack)
}

/// The pure half: `projected` already has the attacker active and owning the turn.
fn affordance_on_projection(
    projected: &State,
    attacker_player: usize,
    attack: &Attack,
) -> AttackAffordance {
    let empty = [DamageEstimate::default(); BOARD_SLOTS];
    let opponent = (attacker_player + 1) % 2;

    let Some(attacking_pokemon) = projected.in_play_pokemon[attacker_player][0].as_ref() else {
        return AttackAffordance::default();
    };
    if attacking_pokemon.is_fossil() {
        return AttackAffordance::default(); // Fossils never attack.
    }

    // Cost and energy are checked on the projection, so cost modifiers see the attacker as the
    // active — the same frame the forecast below runs in.
    let cost = get_attack_cost(&attack.energy_required, projected, attacker_player);
    let attacking_pokemon = projected.in_play_pokemon[attacker_player][0]
        .as_ref()
        .expect("projection puts the attacker in the active slot");
    let missing = energy_missing(attacking_pokemon, &cost, projected, attacker_player);
    let attached = attacking_pokemon
        .get_effective_attached_energy(projected, attacker_player)
        .len();
    let mut affordance = AttackAffordance {
        can_pay: missing.is_empty(),
        deficit: missing.len() as u32,
        surplus: attached.saturating_sub(cost.len()) as u32,
        threat: empty,
    };

    if !affordance.can_pay {
        return affordance; // §1.2.5: expected damage is 0 when the cost is unmet.
    }
    // Damage modifiers, and several mechanics, read the defending active; without one there is
    // nothing to threaten (and nothing well-defined to forecast).
    if projected.in_play_pokemon[opponent][0].is_none() {
        return affordance;
    }

    let attacking_ref = (attacker_player, 0);
    let context = DamageModifierContext {
        attack_name: Some(&attack.title),
        attack_effect: attack.effect.as_deref(),
    };

    // An effect with no typed mechanic cannot be forecast; fall back to nominal damage on the
    // defending active rather than panicking inside the engine's forecast.
    if let Some(effect) = attack.effect.as_deref() {
        if !EFFECT_MECHANIC_MAP.contains_key(effect) {
            let modified = modify_damage(
                projected,
                attacking_ref,
                (attack.fixed_damage, opponent, 0),
                true,
                context,
            );
            affordance.threat[0] = DamageEstimate {
                expected: modified as f32,
                guaranteed: modified,
            };
            return affordance;
        }
    }

    let outcomes = forecast_attack_outcomes(attacker_player, projected, attack, false);

    for (slot, estimate) in affordance.threat.iter_mut().enumerate() {
        if projected.in_play_pokemon[opponent][slot].is_none() {
            continue;
        }
        *estimate = DamageEstimate {
            expected: outcomes.expected_damage_to(
                projected,
                attacking_ref,
                opponent,
                slot,
                context.attack_name,
                context.attack_effect,
            ) as f32,
            guaranteed: outcomes.min_damage_to(
                projected,
                attacking_ref,
                opponent,
                slot,
                context.attack_name,
                context.attack_effect,
            ),
        };
    }
    affordance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_ids::CardId;
    use crate::models::{EnergyType, PlayedCard};
    use crate::test_support::{get_test_game_with_board, nth_attack};

    /// Charmander's Ember: 30 damage, cost [R][C], no effect. Cyndaquil is Fire (not weak to Fire).
    #[test]
    fn deterministic_attack_is_its_fixed_damage() {
        let attacker = PlayedCard::from_id(CardId::A1033Charmander)
            .with_energy(vec![EnergyType::Fire, EnergyType::Fire]);
        let defender = PlayedCard::from_id(CardId::A1001Bulbasaur);
        let game = get_test_game_with_board(vec![attacker], vec![defender]);
        let state = game.get_state_clone();

        let attack = nth_attack(CardId::A1033Charmander, 0);
        let estimate = estimate_damage(&state, (0, 0), &attack, (1, 0));
        // Bulbasaur is Grass, weak to Fire → 30 + 20.
        assert_eq!(estimate.guaranteed, 50);
        assert_eq!(estimate.expected, 50.0);
        assert!(estimate.is_lethal_against(50));
        assert!(!estimate.is_lethal_against(60));
    }

    #[test]
    fn unmet_energy_cost_estimates_zero() {
        let attacker = PlayedCard::from_id(CardId::A1033Charmander); // no energy attached
        let defender = PlayedCard::from_id(CardId::A1001Bulbasaur);
        let game = get_test_game_with_board(vec![attacker], vec![defender]);
        let state = game.get_state_clone();

        let attack = nth_attack(CardId::A1033Charmander, 0);
        assert_eq!(
            estimate_damage(&state, (0, 0), &attack, (1, 0)),
            DamageEstimate::default()
        );
    }

    /// The affordance couples payability and threat on one projection: an unpayable attack still
    /// reports its deficit, and its threat row is all zeros (§1.2.5).
    #[test]
    fn affordance_reports_deficit_with_zero_threat_when_unpayable() {
        let attacker = PlayedCard::from_id(CardId::A1033Charmander); // Ember costs [R][C]
        let defender = PlayedCard::from_id(CardId::A1001Bulbasaur);
        let game = get_test_game_with_board(vec![attacker], vec![defender]);
        let state = game.get_state_clone();

        let attack = nth_attack(CardId::A1033Charmander, 0);
        let affordance = estimate_attack_affordance(&state, (0, 0), &attack);
        assert!(!affordance.can_pay);
        assert_eq!(affordance.deficit, attack.energy_required.len() as u32);
        assert_eq!(affordance.surplus, 0);
        assert_eq!(affordance.threat, [DamageEstimate::default(); BOARD_SLOTS]);

        // Paying the cost with one spare energy flips can_pay and measures the surplus.
        let spare = 1;
        let attacker = PlayedCard::from_id(CardId::A1033Charmander)
            .with_energy(vec![EnergyType::Fire; attack.energy_required.len() + spare]);
        let defender = PlayedCard::from_id(CardId::A1001Bulbasaur);
        let game = get_test_game_with_board(vec![attacker], vec![defender]);
        let state = game.get_state_clone();
        let affordance = estimate_attack_affordance(&state, (0, 0), &attack);
        assert!(affordance.can_pay);
        assert_eq!(affordance.deficit, 0);
        assert_eq!(affordance.surplus, spare as u32);
        assert_eq!(affordance.threat[0].guaranteed, 50);
    }

    #[test]
    fn a_single_target_attack_does_not_reach_the_bench() {
        let attacker = PlayedCard::from_id(CardId::A1033Charmander)
            .with_energy(vec![EnergyType::Fire, EnergyType::Fire]);
        let game = get_test_game_with_board(
            vec![attacker],
            vec![
                PlayedCard::from_id(CardId::A1001Bulbasaur),
                PlayedCard::from_id(CardId::A1001Bulbasaur),
            ],
        );
        let state = game.get_state_clone();

        let attack = nth_attack(CardId::A1033Charmander, 0);
        let row = estimate_attack_threat(&state, (0, 0), &attack);
        assert_eq!(row[0].guaranteed, 50);
        assert_eq!(row[1], DamageEstimate::default(), "bench is unreachable");
        assert_eq!(row[2], DamageEstimate::default(), "empty slot");
    }

    /// A benched attacker is evaluated on a projection where it is active — the affordance is
    /// reported, and the real state is not touched.
    #[test]
    fn benched_attacker_is_projected_without_mutating_the_state() {
        let bench_attacker = PlayedCard::from_id(CardId::A1033Charmander)
            .with_energy(vec![EnergyType::Fire, EnergyType::Fire]);
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur), bench_attacker],
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        );
        let state = game.get_state_clone();
        let before = state.clone();

        let attack = nth_attack(CardId::A1033Charmander, 0);
        let row = estimate_attack_threat(&state, (0, 1), &attack);
        assert_eq!(row[0].guaranteed, 50);
        assert_eq!(state, before, "estimation is side-effect free");
    }

    /// A coin-flip attack has a fractional expectation but a guaranteed floor at the tails branch.
    /// Exeggutor's Stomp: 30 damage, "Flip a coin. If heads, this attack does 30 more damage."
    #[test]
    fn coin_flip_expectation_is_resolved_analytically() {
        let attacker =
            PlayedCard::from_id(CardId::A1022Exeggutor).with_energy(vec![EnergyType::Grass]);
        let defender = PlayedCard::from_id(CardId::A1033Charmander); // Fire, weak to Water
        let game = get_test_game_with_board(vec![attacker], vec![defender]);
        let state = game.get_state_clone();

        let attack = nth_attack(CardId::A1022Exeggutor, 0);
        let estimate = estimate_damage(&state, (0, 0), &attack, (1, 0));
        assert_eq!(estimate.guaranteed, 30, "tails branch");
        assert_eq!(estimate.expected, 45.0, "0.5 × 30 + 0.5 × 60");
    }

    /// The scratch is reused across every attacker of an observation, so it has to come back
    /// *exactly* to the state it was built from after each projection — and agree with the
    /// cloning path on every one of them. Both boards are populated on and off the bench, and the
    /// off-turn player is included (the case that projects both `current_player` and a swap).
    #[test]
    fn projection_scratch_restores_and_matches_the_cloning_path() {
        let game = get_test_game_with_board(
            vec![
                PlayedCard::from_id(CardId::A1022Exeggutor).with_energy(vec![EnergyType::Grass]),
                PlayedCard::from_id(CardId::A1033Charmander)
                    .with_energy(vec![EnergyType::Fire, EnergyType::Fire]),
            ],
            vec![
                PlayedCard::from_id(CardId::A1001Bulbasaur).with_energy(vec![EnergyType::Grass]),
                PlayedCard::from_id(CardId::A1022Exeggutor).with_energy(vec![EnergyType::Grass]),
            ],
        );
        let state = game.get_state_clone();
        let mut scratch = ProjectionScratch::new(&state);

        for (player, idx, card) in [
            (0, 0, CardId::A1022Exeggutor),
            (0, 1, CardId::A1033Charmander),
            (1, 0, CardId::A1001Bulbasaur),
            (1, 1, CardId::A1022Exeggutor),
        ] {
            let attack = nth_attack(card, 0);
            let cloned = estimate_attack_affordance(&state, (player, idx), &attack);
            let scratched = scratch.attack_affordance((player, idx), &attack);
            assert_eq!(
                cloned, scratched,
                "scratch and cloning path disagree on ({player}, {idx})"
            );
            assert_eq!(
                scratch.state, state,
                "scratch left projected after ({player}, {idx})"
            );
        }
    }
}
