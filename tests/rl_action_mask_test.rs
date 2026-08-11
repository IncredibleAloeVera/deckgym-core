//! Falsifiable tests for the RL action mask (`RL_ARCHITECTURE.md` §1.3), driven through the
//! `Game` public API.
//!
//! §1.3.7 states seven invariants; each gets its own test, checked at **every** decision point of
//! whole random rollouts rather than on hand-built boards:
//!
//! 1. **Bijection** — `unproject(mask)` is `generate_possible_actions(state)` as a set (up to
//!    reprint identity, §1.2.2: two printings of one card share a `card_id` row, hence one bit).
//! 2. **Non-empty** — `|E| ≥ 1` at every frame the engine hands out.
//! 3. **Round-trip** — the action a selected `(head, index)` resolves to is applied without panic.
//! 4. **Regime exclusivity** — exactly one of SETUP / STACK / FREE_PLAY / FORCED.
//! 5. **Perspective** — the observation of a frame is taken from `frame.actor`.
//! 6. **Family consistency** — the `action_type` head is the argument heads' own emptiness.
//! 7. **No free-play demotion** — SETUP / FREE_PLAY candidates never fall back to `CANDIDATE_PTR`.
//!
//! Plus the §1.3.8 shape contract: no head index leaves its self-scoped width, and no head spans
//! both sides of the board.

use std::collections::HashSet;

use deckgym::actions::SimpleAction;
use deckgym::card_ids::CardId;
use deckgym::models::PlayedCard;
use deckgym::rl::action_mask::{
    project, ActionFamily, Head, Regime, ACTION_MASK_DIM, ATTACK_SELF, HEADS, POKEMON_SELF,
    TRAINER_SELF,
};
use deckgym::rl::{canonical_action, get_observation};
use deckgym::state::State;
use deckgym::test_support::{get_test_game_with_board, init_random_players};
use deckgym::Game;

/// A rendering of a `SimpleAction` that ignores which *printing* of a card it names — the only
/// equivalence a pointer head can express, and the one the bijection is stated modulo.
fn canonical_key(action: &SimpleAction) -> String {
    format!("{:?}", canonical_action(action))
}

fn canonical_set<'a>(actions: impl IntoIterator<Item = &'a SimpleAction>) -> HashSet<String> {
    actions.into_iter().map(canonical_key).collect()
}

/// §1.3.7 invariants 1, 2, 4 and 5, over whole games: the mask is exactly the engine's enumeration,
/// re-shaped — never a second implementation of legality.
#[test]
fn the_mask_is_the_engine_enumeration_at_every_decision_point() {
    for seed in 0..8u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let state = game.get_state_clone();
            let (actor, actions) = state.generate_possible_actions();

            // 2. Non-empty.
            assert!(!actions.is_empty(), "seed {seed}: an empty decision point");

            let (mask_actor, observation, mask) = game.get_decision_point();

            // 5. Perspective: the frame's actor, which need not be the turn player.
            assert_eq!(mask_actor, actor, "seed {seed}");
            assert_eq!(observation.perspective, actor, "seed {seed}");
            assert_eq!(mask.actor, actor, "seed {seed}");

            // 4. Regime exclusivity, asserted positively against the state that defines it.
            let expected = if actions.len() <= 1 {
                Regime::Forced
            } else if !state.move_generation_stack.is_empty() {
                Regime::Stack
            } else if state.turn_count == 0 {
                Regime::Setup
            } else {
                Regime::FreePlay
            };
            assert_eq!(mask.regime, expected, "seed {seed}");

            // 1. Bijection, as sets, up to reprint identity.
            let engine = canonical_set(actions.iter().map(|action| &action.action));
            let projected = canonical_set(mask.unproject().iter());
            assert_eq!(
                projected, engine,
                "seed {seed}: mask and engine disagree at turn {}",
                state.turn_count
            );

            game.play_tick();
        }
    }
}

/// §1.3.7 invariant 3: whatever a head points at is an action `apply_action` accepts. Played out on
/// a *clone* of the game so the rollout itself stays on its own trajectory.
#[test]
fn every_set_bit_round_trips_through_apply_action() {
    for seed in 0..4u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() && game.get_state_clone().turn_count < 8 {
            let state = game.get_state_clone();
            let (actor, actions) = state.generate_possible_actions();
            let mask = game.get_action_mask();

            for entry in &mask.entries {
                // `select` reconstructs the whole engine action, `is_stack` included —
                // `apply_action` pops the stack frame iff it is set, so it must round-trip.
                let selected = mask
                    .select(entry.head, entry.index)
                    .expect("a set bit resolves");
                assert_eq!(selected.actor, actor);
                assert!(
                    actions.contains(&selected),
                    "seed {seed}: {:?} does not resolve to an enumerated action",
                    entry.head
                );

                let mut probe = Game::from_state(state.clone(), init_random_players(), seed);
                probe.apply_action(&selected);
            }

            game.play_tick();
        }
    }
}

/// §1.3.8: self-only heads point into the *self-scoped slices* of the Part-2 banks. No index may
/// leave its width, and the two opp-role heads use a 4-slot board index rather than a token bank.
#[test]
fn head_indices_stay_inside_their_egocentric_shapes() {
    for seed in 0..6u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let (_, observation, mask) = game.get_decision_point();

            let self_pokemon = observation.pokemon.iter().filter(|t| t.allied).count();
            let self_trainers = observation.trainers.iter().filter(|t| t.allied).count();
            let self_attacks = observation.attacks.iter().filter(|t| t.allied).count();
            assert!(self_pokemon <= POKEMON_SELF, "seed {seed}: {self_pokemon}");
            assert!(
                self_trainers <= TRAINER_SELF,
                "seed {seed}: {self_trainers}"
            );
            assert!(self_attacks <= ATTACK_SELF, "seed {seed}: {self_attacks}");

            for entry in &mask.entries {
                assert!(
                    entry.index < entry.head.dim(),
                    "seed {seed}: {:?} index {} over {} slots",
                    entry.head,
                    entry.index,
                    entry.head.dim()
                );
                // A pointer head names a *live* token row, never padding.
                match entry.head {
                    Head::Place | Head::Evolve => {
                        assert!(entry.index / 4 < self_pokemon, "seed {seed}")
                    }
                    Head::HandPtr => assert!(entry.index < self_pokemon, "seed {seed}"),
                    Head::Attack => assert!(entry.index < self_attacks, "seed {seed}"),
                    Head::PlayTrainer => assert!(entry.index < self_trainers, "seed {seed}"),
                    _ => {}
                }
            }

            // The wire form tiles the heads without overlap and agrees bit for bit.
            let wire = mask.to_wire();
            assert_eq!(wire.bits.len(), ACTION_MASK_DIM);
            for head in HEADS {
                assert_eq!(wire.head(head), mask.bits(head), "seed {seed}: {head:?}");
            }
            assert_eq!(wire.head(Head::ActionType), mask.family, "seed {seed}");

            game.play_tick();
        }
    }
}

/// §1.3.4: the `action_type` head is masked to families with ≥ 1 legal instantiation, and in
/// free play those families are exactly the free-play actions the engine offers.
#[test]
fn the_family_head_matches_the_free_play_enumeration() {
    for seed in 0..6u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let state = game.get_state_clone();
            let (_, actions) = state.generate_possible_actions();
            let mask = game.get_action_mask();

            if mask.regime == Regime::FreePlay {
                let engine_families = |family: ActionFamily| {
                    actions.iter().any(|action| match &action.action {
                        SimpleAction::EndTurn => family == ActionFamily::EndTurn,
                        SimpleAction::Place(..) => family == ActionFamily::Place,
                        SimpleAction::Evolve { .. } => family == ActionFamily::Evolve,
                        SimpleAction::Attach {
                            is_turn_energy: true,
                            ..
                        } => family == ActionFamily::AttachEnergy,
                        SimpleAction::Retreat(_) => family == ActionFamily::Retreat,
                        SimpleAction::Attack(_) => family == ActionFamily::Attack,
                        SimpleAction::UseAbility { .. } => family == ActionFamily::UseAbility,
                        SimpleAction::Play { .. } => family == ActionFamily::PlayTrainer,
                        SimpleAction::UseStadium => family == ActionFamily::UseStadium,
                        SimpleAction::DiscardFossil { .. } => family == ActionFamily::DiscardFossil,
                        _ => false,
                    })
                };
                for family in ActionFamily::ALL {
                    assert_eq!(
                        mask.family[family.index()],
                        engine_families(family),
                        "seed {seed}: {family:?} at turn {}",
                        state.turn_count
                    );
                }
            }

            game.play_tick();
        }
    }
}

/// The `CANDIDATE_PTR` escape hatch must stay an escape hatch: over ordinary play the free-play
/// families are always addressed by their own typed head, never demoted.
#[test]
fn free_play_never_falls_back_to_the_candidate_pointer() {
    for seed in 0..6u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let mask = game.get_action_mask();
            if mask.regime == Regime::FreePlay || mask.regime == Regime::Setup {
                for entry in &mask.entries {
                    assert_ne!(
                        entry.head,
                        Head::CandidatePtr,
                        "seed {seed}: {} was demoted out of its family head",
                        entry.action
                    );
                }
            }
            game.play_tick();
        }
    }
}

/// §1.3.6.1: a decision point is not aligned with turn ownership. A promotion frame after a KO is
/// owned by the player who lost the Pokémon, and the mask is built from *their* perspective, on
/// *their* board.
#[test]
fn a_reactive_frame_belongs_to_the_off_turn_player() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![
            PlayedCard::from_id(CardId::A1033Charmander),
            PlayedCard::from_id(CardId::A1055Blastoise),
            PlayedCard::from_id(CardId::A1022Exeggutor),
        ],
    );
    let mut state = game.get_state_clone();
    // Player 1 must promote: two bench candidates, decided during player 0's turn.
    state.move_generation_stack.push((
        1,
        vec![
            SimpleAction::Activate {
                player: 1,
                in_play_idx: 1,
            },
            SimpleAction::Activate {
                player: 1,
                in_play_idx: 2,
            },
        ],
    ));
    game.set_state(state);

    let (actor, observation, mask) = game.get_decision_point();
    assert_eq!(actor, 1, "the frame's actor is the KO receiver");
    assert_eq!(observation.perspective, 1);
    assert!(!observation.global.is_my_turn, "and it is not their turn");
    assert_eq!(mask.regime, Regime::Stack);

    // Their own board, so the *self*-role slot pointer — no player-index dimension anywhere.
    assert_eq!(mask.active_heads(), vec![Head::SlotPtrSelf]);
    assert!(mask.is_set(Head::SlotPtrSelf, 1));
    assert!(mask.is_set(Head::SlotPtrSelf, 2));
    assert!(!mask.is_set(Head::SlotPtrSelf, 3));
}

/// §1.3.6.3: `DrawCard` is engine-internal — it only ever reaches the mask as the single entry of
/// a FORCED frame, never as a multi-way choice. (The spot-damage frames, which *are* choices, are
/// covered by `a_spot_damage_frame_points_at_the_opponents_board`.)
#[test]
fn draw_card_only_reaches_the_mask_as_a_forced_frame() {
    for seed in 0..8u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let mask = game.get_action_mask();
            for entry in &mask.entries {
                let internal = matches!(entry.action, SimpleAction::DrawCard { .. });
                if internal {
                    assert_eq!(
                        mask.regime,
                        Regime::Forced,
                        "seed {seed}: DrawCard offered as a choice"
                    );
                }
            }
            game.play_tick();
        }
    }
}

/// §1.3.3: spot damage is a choice, not an internal frame. Single-target `ApplyDamage` and
/// `ScheduleDelayedSpotDamage` candidates ("damage 1 of your opponent's Pokémon") are cross-target
/// decisions on a public board and get the opp-role slot pointer, exactly like Cyrus.
#[test]
fn a_spot_damage_frame_points_at_the_opponents_board() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![
            PlayedCard::from_id(CardId::A1033Charmander),
            PlayedCard::from_id(CardId::A1055Blastoise),
        ],
    );
    let mut state = game.get_state_clone();
    // The attacker picks which opposing Pokémon takes the hit — one immediate, one delayed.
    state.move_generation_stack.push((
        0,
        vec![
            SimpleAction::ApplyDamage {
                attacking_ref: (0, 0),
                targets: vec![(20, 1, 0)],
                is_from_active_attack: false,
            },
            SimpleAction::ScheduleDelayedSpotDamage {
                target_player: 1,
                target_in_play_idx: 1,
                amount: 20,
            },
        ],
    ));
    game.set_state(state);

    let (actor, _, mask) = game.get_decision_point();
    assert_eq!(actor, 0, "the attacker owns the targeting frame");
    assert_eq!(mask.regime, Regime::Stack);
    assert_eq!(mask.active_heads(), vec![Head::SlotPtrOpp]);
    assert!(mask.is_set(Head::SlotPtrOpp, 0));
    assert!(mask.is_set(Head::SlotPtrOpp, 1));
    // The selected bit resolves to the stack action itself, `is_stack` included.
    let selected = mask
        .select(Head::SlotPtrOpp, 1)
        .expect("a set bit resolves");
    assert!(selected.is_stack);
    assert!(matches!(
        selected.action,
        SimpleAction::ScheduleDelayedSpotDamage { .. }
    ));
}

/// A `Forced` frame resolves without a network forward: exactly one entry, and it is the action the
/// engine would have auto-selected.
#[test]
fn forced_frames_carry_exactly_one_entry() {
    for seed in 0..6u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let state = game.get_state_clone();
            let (_, actions) = state.generate_possible_actions();
            let mask = game.get_action_mask();
            if mask.regime == Regime::Forced {
                assert_eq!(mask.entries.len(), 1, "seed {seed}");
                let forced = mask.forced_action().expect("a forced frame has its action");
                // Full `Action` equality: `is_stack` included, or resolving a forced stack frame
                // would leave it on the stack and re-enumerate it forever.
                assert_eq!(forced, actions[0], "seed {seed}");
            } else {
                assert!(mask.forced_action().is_none(), "seed {seed}");
                assert!(
                    mask.entries.len() > 1,
                    "seed {seed}: a non-forced singleton"
                );
            }
            game.play_tick();
        }
    }
}

/// The two default test decks exercise a narrow slice of the pool and almost no stack frames. This
/// sweeps deck pairs across the `example_decks` shelf — Mega Absol's hand-reveal, Professor Sada's
/// energy assignments, coin-flip spot damage, fossils, stadiums — and re-checks the bijection, the
/// egocentric widths and the per-frame pointer caps on all of it.
#[test]
fn the_mask_holds_across_the_deck_shelf() {
    const DECKS: [&str; 12] = [
        "mega-absol-hydreigon.txt",
        "great_tusk_koraidon.txt",
        "arceusdialga.txt",
        "solgaleo_shiinotic.txt",
        "suicune-greninja.txt",
        "giratina-darkrai.txt",
        "venu-serperior.txt",
        "swampert-kanga-ninetails.txt",
        "igglybuff-mega-gengar.txt",
        "coinflip_deck.txt",
        "silvally.txt",
        "mega-garde.txt",
    ];

    let mut widest_candidate = 0;
    let mut widest_revealed = 0;
    let mut widest_self = (0, 0, 0);

    for (index, deck) in DECKS.iter().enumerate() {
        let opponent = DECKS[(index + 1) % DECKS.len()];
        for seed in 0..3u64 {
            let mut game = Game::new(deckgym::test_support::init_decks(deck, opponent), seed);
            while !game.is_game_over() {
                let state = game.get_state_clone();
                let (actor, actions) = state.generate_possible_actions();
                let (mask_actor, observation, mask) = game.get_decision_point();

                assert_eq!(mask_actor, actor, "{deck} vs {opponent} seed {seed}");
                assert_eq!(
                    canonical_set(mask.unproject().iter()),
                    canonical_set(actions.iter().map(|action| &action.action)),
                    "{deck} vs {opponent} seed {seed}: mask and engine disagree"
                );

                widest_self.0 = widest_self
                    .0
                    .max(observation.pokemon.iter().filter(|t| t.allied).count());
                widest_self.1 = widest_self
                    .1
                    .max(observation.trainers.iter().filter(|t| t.allied).count());
                widest_self.2 = widest_self
                    .2
                    .max(observation.attacks.iter().filter(|t| t.allied).count());
                for entry in &mask.entries {
                    match entry.head {
                        Head::CandidatePtr => {
                            widest_candidate = widest_candidate.max(entry.index + 1)
                        }
                        Head::RevealedHandPtr => {
                            widest_revealed = widest_revealed.max(entry.index + 1)
                        }
                        _ => {}
                    }
                }

                // The caps are what `to_wire` asserts on; exercising it is the point.
                mask.to_wire();
                game.play_tick();
            }
        }
    }

    println!(
        "widest: candidate={widest_candidate} revealed={widest_revealed} \
         self=(pokemon {}, trainer {}, attack {})",
        widest_self.0, widest_self.1, widest_self.2
    );
    assert!(widest_self.0 <= POKEMON_SELF);
    assert!(widest_self.1 <= TRAINER_SELF);
    assert!(widest_self.2 <= ATTACK_SELF);
    assert!(
        widest_revealed > 0,
        "the shelf must actually reach a reveal frame"
    );
}

/// Reproducibility: the mask is a pure function of `(state, legal actions, observation)`.
#[test]
fn the_mask_is_deterministic() {
    let mut game = Game::new(init_random_players(), 11);
    for _ in 0..25 {
        if game.is_game_over() {
            break;
        }
        game.play_tick();
    }
    let state: State = game.get_state_clone();
    let (actor, actions) = state.generate_possible_actions();
    let observation = get_observation(&state, actor, &actions, None, None);
    let first = project(&state, &actions, &observation);
    let second = project(&state, &actions, &observation);
    assert_eq!(first, second);
    assert_eq!(first.to_wire(), second.to_wire());
}
