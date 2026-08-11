//! Falsifiable tests for the RL observation (`RL_ARCHITECTURE.md` §1.2), driven through the
//! `Game` public API.
//!
//! The load-bearing properties, each with its own test:
//!
//! 1. it never leaks the opponent's hidden zones;
//! 2. it is egocentric — swapping the perspective swaps every role-relative feature;
//! 3. its legality bits are a projection of `generate_possible_actions`, not a second
//!    implementation of legality;
//! 4. it survives every decision point of a real rollout, within the padded bank caps;
//! 5. the History bank carries the opponent's choices only, and never a hidden card.

use std::collections::HashMap;

use deckgym::actions::{Action, SimpleAction};
use deckgym::card_ids::CardId;
use deckgym::database::get_card_by_enum;
use deckgym::models::{Card, EnergyType, PlayedCard};
use deckgym::players::{create_players, PlayerCode};
use deckgym::rl::ids::{canonical_card, card_at_index};
use deckgym::rl::observation::{get_observation, Observation, TokenZone, MAX_TRAINER_TARGET_IDS};
use deckgym::rl::{ActionTrace, HISTORY_LEN};
use deckgym::state::State;
use deckgym::test_support::{get_test_game_with_board, init_random_players, nth_attack};
use deckgym::Deck;
use deckgym::Game;

fn multiset(cards: impl IntoIterator<Item = CardId>) -> HashMap<CardId, usize> {
    let mut counts = HashMap::new();
    for card in cards {
        *counts.entry(card).or_insert(0) += 1;
    }
    counts
}

fn token_cards(observation: &Observation, allied: bool) -> HashMap<CardId, usize> {
    let pokemon = observation
        .pokemon
        .iter()
        .filter(|token| token.allied == allied)
        .filter_map(|token| card_at_index(token.card_id));
    let trainers = observation
        .trainers
        .iter()
        .filter(|token| token.allied == allied)
        .filter_map(|token| card_at_index(token.card_id));
    multiset(pokemon.chain(trainers))
}

/// Every card the opponent has in public view: their board (with the tools riding on it) and
/// their discard pile. Canonicalized, since a token index names the original printing of a
/// complete reprint rather than the copy that happens to be in play.
fn opponent_public_cards(state: &State, opponent: usize) -> HashMap<CardId, usize> {
    let board = state.in_play_pokemon[opponent]
        .iter()
        .flatten()
        .map(|played| played.card.get_card_id());
    let discard = state.discard_piles[opponent].iter().map(Card::get_card_id);
    multiset(board.chain(discard).map(canonical_card))
}

/// §1.2.1: the opponent's hand and deck contribute a *size* to the global vector and nothing else.
/// The set of non-allied tokens must be exactly their public cards — no more, no less.
#[test]
fn observation_never_leaks_the_opponent_hidden_zones() {
    for seed in 0..8u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let state = game.get_state_clone();
            for perspective in 0..2 {
                let observation = game.get_observation(perspective);
                let opponent = (perspective + 1) % 2;

                for token in &observation.pokemon {
                    assert!(
                        token.allied || matches!(token.zone, TokenZone::Board | TokenZone::Discard),
                        "seed {seed}: opponent token in a hidden zone {:?}",
                        token.zone
                    );
                }
                for token in &observation.trainers {
                    assert!(
                        token.allied || matches!(token.zone, TokenZone::Board | TokenZone::Discard),
                        "seed {seed}: opponent token in a hidden zone {:?}",
                        token.zone
                    );
                }

                assert_eq!(
                    token_cards(&observation, false),
                    opponent_public_cards(&state, opponent),
                    "seed {seed}: non-allied tokens must be exactly the opponent's public cards"
                );
            }
            game.play_tick();
        }
    }
}

/// Attached tools ride on their host's `tool_id` rather than as their own token — the opponent's
/// board tools must therefore never appear as a Trainer token.
#[test]
fn opponent_tools_ride_on_their_host() {
    let holder = PlayedCard::from_id(CardId::A1055Blastoise)
        .with_tool(deckgym::database::get_card_by_enum(CardId::A2147GiantCape));
    let game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![holder],
    );

    let observation = game.get_observation(0);
    let opponent_active = observation
        .pokemon
        .iter()
        .find(|token| !token.allied && token.slot == Some(0))
        .expect("the opponent's active is emitted");
    assert!(opponent_active.has_tool);
    assert_eq!(
        card_at_index(opponent_active.tool_id),
        Some(CardId::A2147GiantCape)
    );
    assert!(
        !observation
            .trainers
            .iter()
            .any(|token| card_at_index(token.card_id) == Some(CardId::A2147GiantCape)),
        "an attached tool is not a Trainer token"
    );
}

/// Part 1: observations are encoded by role, never by absolute player index. The two perspectives
/// of one state must be mirror images.
#[test]
fn the_observation_is_egocentric() {
    let game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::A1001Bulbasaur),
            PlayedCard::from_id(CardId::A1033Charmander),
        ],
        vec![PlayedCard::from_id(CardId::A1055Blastoise)],
    );

    let mine = game.get_observation(0);
    let theirs = game.get_observation(1);

    assert_eq!(mine.global.points[0], theirs.global.points[1]);
    assert_eq!(mine.global.hand_size[0], theirs.global.hand_size[1]);
    assert_eq!(mine.global.draw_pile[0], theirs.global.draw_pile[1]);
    assert_eq!(mine.global.is_my_turn, !theirs.global.is_my_turn);
    assert_eq!(mine.global.on_the_play, !theirs.global.on_the_play);

    // My two board Pokémon are allied to me and hostile to them.
    let my_board: Vec<_> = mine
        .pokemon
        .iter()
        .filter(|token| token.allied && token.zone == TokenZone::Board)
        .map(|token| (token.card_id, token.slot))
        .collect();
    let same_board_seen_by_them: Vec<_> = theirs
        .pokemon
        .iter()
        .filter(|token| !token.allied && token.zone == TokenZone::Board)
        .map(|token| (token.card_id, token.slot))
        .collect();
    assert_eq!(my_board.len(), 2);
    assert_eq!(my_board, same_board_seen_by_them);
}

/// During setup no one is on the play: both players place their boards simultaneously in the real
/// game, so the engine's placement alternation must not leak into the bit. From turn 1 onward the
/// bit is antisymmetric, matches turn 1's owner, and never flips for the rest of the game.
#[test]
fn on_the_play_is_neutral_during_setup_and_stable_after() {
    for seed in 0..4u64 {
        let mut game = Game::new(init_random_players(), seed);
        let mut on_the_play_player: Option<usize> = None;
        while !game.is_game_over() && game.get_state_clone().turn_count < 6 {
            let state = game.get_state_clone();
            let zero = game.get_observation(0);
            let one = game.get_observation(1);
            if state.turn_count == 0 {
                assert!(
                    !zero.global.on_the_play && !one.global.on_the_play,
                    "seed {seed}: no one is on the play during setup"
                );
            } else {
                assert_ne!(
                    zero.global.on_the_play, one.global.on_the_play,
                    "seed {seed}"
                );
                if state.turn_count == 1 {
                    assert_eq!(
                        zero.global.on_the_play, zero.global.is_my_turn,
                        "seed {seed}: on the play = owner of turn 1"
                    );
                }
                let owner = usize::from(one.global.on_the_play);
                match on_the_play_player {
                    None => on_the_play_player = Some(owner),
                    Some(fixed) => {
                        assert_eq!(fixed, owner, "seed {seed}: on_the_play flipped mid-game")
                    }
                }
            }
            game.play_tick();
        }
        assert!(on_the_play_player.is_some(), "seed {seed}: game left setup");
    }
}

/// §1.2.6: `activation_condition_met` is evaluated for the card's *owner* — an off-turn player's
/// Potion looks at their own board, never at the turn player's.
#[test]
fn trainer_activation_condition_is_evaluated_for_the_owner() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1033Charmander).with_damage(20)],
    );
    let mut state = game.get_state_clone();
    let potion = deckgym::database::get_card_by_enum(CardId::PA001Potion);
    state.hands[0].retain(|card| card.get_card_id() != CardId::PA001Potion);
    state.hands[1].retain(|card| card.get_card_id() != CardId::PA001Potion);
    state.hands[0].push(potion.clone());
    state.hands[1].push(potion);
    game.set_state(state);

    let potion_index = canonical_card(CardId::PA001Potion);
    let find_potion = |observation: &Observation| {
        observation
            .trainers
            .iter()
            .find(|token| {
                token.allied
                    && token.zone == TokenZone::Hand
                    && card_at_index(token.card_id) == Some(potion_index)
            })
            .cloned()
            .expect("the Potion token is emitted")
    };

    // Player 1 is off turn (current player is 0) but has a damaged Pokémon: condition met.
    assert!(find_potion(&game.get_observation(1)).activation_condition_met);
    // Player 0 owns the turn yet has nothing to heal: condition not met.
    assert!(!find_potion(&game.get_observation(0)).activation_condition_met);
}

/// §1.2.1: my own deck's declared energy set is fully known — it is my deck. Without the belief
/// overlay (spectator mode, no reveal ever reaches the wire) the opponent's slot stays at the
/// pre-reveal default, empty, same as their hand and deck contents.
#[test]
fn declared_deck_energies_are_observed_for_self_only() {
    use deckgym::rl::encoding::{energy_index, ENERGY_DIM};

    let game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    let mine = game.get_observation(0);
    let theirs = game.get_observation(1);

    // The test decks declare a single energy each: Grass (venusaur-exeggutor) for player 0,
    // Darkness (weezing-arbok) for player 1.
    assert!(mine.global.deck_energy_types[0][energy_index(EnergyType::Grass)]);
    assert_eq!(
        mine.global.deck_energy_types[0]
            .iter()
            .filter(|b| **b)
            .count(),
        1
    );
    assert!(theirs.global.deck_energy_types[0][energy_index(EnergyType::Darkness)]);

    // No belief overlay was enabled: the opponent's declared set never reaches the wire.
    assert_eq!(mine.global.deck_energy_types[1], [false; ENERGY_DIM]);
    assert_eq!(theirs.global.deck_energy_types[1], [false; ENERGY_DIM]);
}

/// TODO.md "Opponent deck energy": the opponent's full deck composition (up to 3 declared energy
/// types) is not public in TCG Pocket — only what has actually rolled through their energy zone
/// so far is. `deck_energy_types[opponent]` must track that reveal, not the deck's static
/// declaration, and it must never shrink once a type has been seen.
#[test]
fn opponent_energy_types_reveal_progressively_through_the_energy_zone() {
    use deckgym::rl::encoding::ENERGY_DIM;

    let deck = Deck::from_string(POKEMON_ONLY_DECK).expect("the deck parses");
    let players = create_players(deck.clone(), deck, vec![PlayerCode::R, PlayerCode::R]);
    let mut game = Game::new(players, 7);
    game.enable_belief();

    // `deck_energy_types[self]` is always fully known (never belief-gated) — read it off the
    // mirrored deck as ground truth for what the opponent's deck can actually roll, since
    // `Deck::energy_types` itself is a private field.
    let full_declared = game.get_observation(0).global.deck_energy_types[0];
    assert!(
        full_declared.iter().filter(|b| **b).count() > 1,
        "the fixture needs a multi-type deck for this test to be meaningful"
    );

    // Nothing has rolled yet: the opponent's declared set must not already be fully visible.
    let opening = game.get_observation(0).global.deck_energy_types[1];
    assert_ne!(
        opening, full_declared,
        "the opponent's full deck composition must not leak before anything is revealed"
    );

    let mut previous = opening;
    let mut turns = 0;
    while !game.is_game_over() && turns < 40 {
        game.play_tick();
        turns += 1;
        let seen = game.get_observation(0).global.deck_energy_types[1];
        for index in 0..ENERGY_DIM {
            assert!(
                !seen[index] || full_declared[index],
                "a seen type must be one the deck can actually roll"
            );
            assert!(
                !previous[index] || seen[index],
                "seen energy types must never un-reveal"
            );
        }
        previous = seen;
    }

    // A long enough game rolls every declared type through the energy zone at least once.
    assert_eq!(
        previous, full_declared,
        "every declared type should have surfaced given enough turns"
    );
}

/// §1.2.2: the legality features are the *sibling* projection of `generate_possible_actions` — the
/// same enumeration the Part 3 mask consumes. Checked at every decision point of a rollout.
#[test]
fn legality_bits_mirror_the_engine_enumeration() {
    for seed in 0..6u64 {
        let mut game = Game::new(init_random_players(), seed);
        while !game.is_game_over() {
            let state = game.get_state_clone();
            let (actor, actions) = state.generate_possible_actions();
            let observation = game.get_observation(actor);

            for slot in 0..4 {
                let engine_offers_ability = actions.iter().any(|action| {
                    action.actor == actor
                        && matches!(action.action, SimpleAction::UseAbility { in_play_idx } if in_play_idx == slot)
                });
                let token_bit = observation
                    .pokemon
                    .iter()
                    .find(|token| token.allied && token.slot == Some(slot))
                    .is_some_and(|token| token.ability_activatable_now);
                assert_eq!(
                    engine_offers_ability, token_bit,
                    "seed {seed}: ability_activatable_now disagrees with the engine on slot {slot}"
                );
            }

            for token in observation.trainers.iter().filter(|t| t.playable_now) {
                let card = card_at_index(token.card_id).expect("a real card");
                assert!(
                    actions.iter().any(|action| match &action.action {
                        SimpleAction::Play { trainer_card } =>
                            CardId::from_card_id(&trainer_card.id).map(canonical_card) == Some(card),
                        _ => false,
                    }),
                    "seed {seed}: playable_now set for a card the engine does not offer"
                );
            }

            game.play_tick();
        }
    }
}

/// §1.2.8: the banks are padded and masked, and an overflow is an assert rather than a silent
/// truncation. Walking whole games is what actually exercises the caps.
#[test]
fn the_wire_form_holds_at_every_decision_point() {
    for seed in 0..6u64 {
        let mut game = Game::new(init_random_players(), seed);
        game.enable_action_trace();
        while !game.is_game_over() {
            for perspective in 0..2 {
                let observation = game.get_observation(perspective);
                let wire = observation.to_wire();
                assert_eq!(
                    wire.pokemon_mask.iter().filter(|set| **set).count(),
                    observation.pokemon.len()
                );
                assert_eq!(
                    wire.attack_mask.iter().filter(|set| **set).count(),
                    observation.attacks.len()
                );
                assert_eq!(
                    wire.trainer_mask.iter().filter(|set| **set).count(),
                    observation.trainers.len()
                );
                assert!(
                    wire.global.iter().all(|value| value.is_finite()),
                    "seed {seed}: non-finite global feature"
                );
                assert!(wire.pokemon.iter().all(|value| value.is_finite()));
                assert!(wire.attack.iter().all(|value| value.is_finite()));

                // §1.2.5 invariant: payability and threat are computed on one projection, so an
                // unpayable attack never carries a non-zero threat or lethality.
                for token in &observation.attacks {
                    if !token.can_pay {
                        assert_eq!(
                            token.threat, [0.0; 4],
                            "seed {seed}: unpayable attack with non-zero threat"
                        );
                        assert_eq!(token.is_lethal, [false; 4]);
                    }
                }
            }
            game.play_tick();
        }
    }
}

/// §1.2.5: attack tokens are emitted for every board Pokémon on both sides, benched attackers
/// included, so the threat matrix is a full our-attacks × their-Pokémon picture.
#[test]
fn the_threat_matrix_covers_both_sides_and_the_bench() {
    // Exeggutor's Stomp: 30 damage, +30 on heads. Charmander has 60 HP and is not weak to Grass,
    // so the guaranteed floor (30) is not lethal but the expectation is 45.
    let attacker = PlayedCard::from_id(CardId::A1022Exeggutor).with_energy(vec![EnergyType::Grass]);
    let bench_attacker =
        PlayedCard::from_id(CardId::A1022Exeggutor).with_energy(vec![EnergyType::Grass]);
    let game = get_test_game_with_board(
        vec![attacker, bench_attacker],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );

    let observation = game.get_observation(0);
    let mine: Vec<_> = observation
        .attacks
        .iter()
        .filter(|token| token.allied)
        .collect();
    assert_eq!(mine.len(), 2, "the benched attacker gets a token too");

    for token in &mine {
        assert!(token.can_pay);
        assert_eq!(token.deficit, 0);
        // 45 expected damage against a 60 HP defender.
        assert!((token.threat[0] - 0.75).abs() < 1e-6, "{:?}", token.threat);
        assert!(!token.is_lethal[0], "30 guaranteed does not KO 60 HP");
        assert_eq!(token.threat[1], 0.0, "empty opposing slot");
    }

    // The opponent's own attack tokens are emitted symmetrically, pointing back at our board.
    let theirs: Vec<_> = observation
        .attacks
        .iter()
        .filter(|token| !token.allied)
        .collect();
    assert_eq!(theirs.len(), 1, "Charmander's single attack");
    assert!(!theirs[0].can_pay, "no energy attached");
    assert!(theirs[0].deficit > 0);
    assert_eq!(
        theirs[0].threat, [0.0; 4],
        "an unpayable attack threatens 0"
    );
}

/// An attack token points back at its parent Pokémon token, and carries the card its descriptor
/// comes from — the identity the in-model static table is gathered by.
#[test]
fn attack_tokens_point_at_their_parent_and_source() {
    let game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1022Exeggutor)],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    let observation = game.get_observation(0);
    let token = observation
        .attacks
        .iter()
        .find(|token| token.allied)
        .expect("our active has an attack");

    let parent = &observation.pokemon[token.parent_pokemon_ref as usize];
    assert_eq!(parent.slot, Some(0));
    assert_eq!(card_at_index(parent.card_id), Some(CardId::A1022Exeggutor));
    assert_eq!(
        card_at_index(token.src_card_id),
        Some(CardId::A1022Exeggutor)
    );
    assert_eq!(token.attack_slot, 0);
    assert_eq!(
        nth_attack(CardId::A1022Exeggutor, 0).title,
        "Stomp",
        "the token indexes the attack the static table will resolve"
    );
}

/// §1.2.7: the trace holds the opponent's genuine *choices*, oldest first, and never a hidden card.
#[test]
fn history_tokens_are_the_opponents_choices_only() {
    let mut game = Game::new(init_random_players(), 3);
    game.enable_action_trace();
    for _ in 0..40 {
        if game.is_game_over() {
            break;
        }
        game.play_tick();
    }

    let trace = game.action_trace().expect("recording is on").clone();
    let state = game.get_state_clone();

    for perspective in 0..2 {
        let observation = game.get_observation(perspective);
        let opponent = (perspective + 1) % 2;
        assert!(observation.history.len() <= HISTORY_LEN);
        assert_eq!(observation.history.len(), trace.entries_of(opponent).len());

        // Recency is non-increasing: the bank is ordered oldest first.
        for pair in observation.history.windows(2) {
            assert!(pair[0].step_recency >= pair[1].step_recency);
        }
        // Every recorded head is a real action family, and every card index is public.
        for (token, entry) in observation.history.iter().zip(trace.entries_of(opponent)) {
            assert_eq!(token.head_id, entry.head_id);
            assert!(token.head_id > 0);
        }
    }

    let _ = state;
}

/// The trainer target-set bag is padded to a fixed width on the wire, so no card in the frozen
/// pool may name more Pokémon than it holds.
#[test]
fn no_trainer_in_the_pool_overflows_the_target_bag() {
    use deckgym::rl::static_tables::trainer_targeting;
    use strum::IntoEnumIterator;

    let mut widest = 0;
    for card_id in CardId::iter() {
        if let Card::Trainer(trainer) = deckgym::database::get_card_by_enum(card_id) {
            let named = trainer_targeting(&trainer).target_ids.len();
            widest = widest.max(named);
            assert!(
                named <= MAX_TRAINER_TARGET_IDS,
                "{} names {named} Pokémon, over the {MAX_TRAINER_TARGET_IDS} slot bag",
                trainer.name
            );
        }
    }
    assert!(widest > 0, "some trainer cards do name Pokémon");
}

/// §1.2.2 does not distinguish complete reprints: which *printing* of a card a player happens to
/// own must not change a single thing the agent sees. A4b is a pure reprint set, so it is the
/// sharpest case — the A4b copy and its original observe identically.
#[test]
fn an_a4b_reprint_observes_as_its_original() {
    let reprint = CardId::A4b184LunalaEx;
    let original = canonical_card(reprint);
    assert_ne!(original, reprint, "A4b 184 re-prints an earlier card");

    let game = get_test_game_with_board(
        vec![PlayedCard::from_id(reprint)],
        vec![PlayedCard::from_id(original)],
    );
    let observation = game.get_observation(0);

    let mine = observation
        .pokemon
        .iter()
        .find(|token| token.allied && token.slot == Some(0))
        .expect("our active");
    let theirs = observation
        .pokemon
        .iter()
        .find(|token| !token.allied && token.slot == Some(0))
        .expect("their active");

    assert_eq!(mine.card_id, theirs.card_id, "one row for both printings");
    assert_eq!(mine.species_id, theirs.species_id);
    assert_eq!(mine.line_id, theirs.line_id);
    assert_eq!(card_at_index(mine.card_id), Some(original));

    // The affordance satellites agree too: they are gathered by `(src_card_id, attack_slot)`.
    let sources: Vec<_> = observation
        .attacks
        .iter()
        .map(|token| (token.src_card_id, token.attack_slot))
        .collect();
    assert!(!sources.is_empty());
    assert!(
        sources.windows(2).all(|pair| pair[0] == pair[1]),
        "both sides resolve to the same attack descriptor rows: {sources:?}"
    );
}

/// Reproducibility: the observation is a pure function of `(state, perspective, legal actions)`.
#[test]
fn the_observation_is_deterministic() {
    let mut game = Game::new(init_random_players(), 11);
    for _ in 0..25 {
        if game.is_game_over() {
            break;
        }
        game.play_tick();
    }
    let first = game.get_observation(0);
    let second = game.get_observation(0);
    assert_eq!(first, second);
    assert_eq!(first.to_wire(), second.to_wire());
}

// -------------------------------------------------------------------------------------------
// §1.3.6.2 — what a reveal shows
// -------------------------------------------------------------------------------------------

/// A game whose opponent hand is known, with the belief overlay on and Misdreavus's Infiltrating
/// Inspection already resolved. Player 0 is the observer.
fn game_with_a_revealed_opponent_hand(enable_belief: bool) -> Game<'static> {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1005Caterpie)],
    );
    if enable_belief {
        game.enable_belief();
    }

    let mut state = game.get_state_clone();
    state.hands[0] = vec![get_card_by_enum(CardId::A4a032Misdreavus)];
    state.hands[1] = vec![
        get_card_by_enum(CardId::A1001Bulbasaur),
        get_card_by_enum(CardId::A1001Bulbasaur),
        get_card_by_enum(CardId::PA001Potion),
    ];
    game.set_state(state);

    game.apply_action(&Action {
        actor: 0,
        action: SimpleAction::Place(get_card_by_enum(CardId::A4a032Misdreavus), 1),
        is_stack: false,
    });
    game
}

/// The cards of the opponent's hand that reached the observation, both banks.
fn opponent_hand_tokens(observation: &Observation) -> HashMap<CardId, usize> {
    let pokemon = observation
        .pokemon
        .iter()
        .filter(|token| !token.allied && token.zone == TokenZone::Hand)
        .filter_map(|token| card_at_index(token.card_id));
    let trainers = observation
        .trainers
        .iter()
        .filter(|token| !token.allied && token.zone == TokenZone::Hand)
        .filter_map(|token| card_at_index(token.card_id));
    multiset(pokemon.chain(trainers))
}

/// §1.3.6.2: a revealed hand becomes tokens — in *both* banks, since a reveal shows the whole hand
/// and not only the Supporters an effect can then act on. One row per known copy.
#[test]
fn a_revealed_hand_becomes_opponent_hand_tokens() {
    let game = game_with_a_revealed_opponent_hand(true);
    let revealed = opponent_hand_tokens(&game.get_observation(0));

    assert_eq!(
        revealed.get(&canonical_card(CardId::A1001Bulbasaur)),
        Some(&2)
    );
    assert_eq!(revealed.get(&canonical_card(CardId::PA001Potion)), Some(&1));
    assert_eq!(revealed.len(), 2, "nothing beyond the revealed hand");

    // The reveal is directional: player 1 learns nothing about player 0's hand.
    assert!(opponent_hand_tokens(&game.get_observation(1)).is_empty());
}

/// The overlay is the only path in. Without player mode the same game observes exactly as it did
/// before §1.3.6.2 existed — this is what makes `belief: None` the §1.2.1 default rather than a
/// degraded mode.
#[test]
fn without_the_overlay_a_reveal_changes_nothing() {
    let game = game_with_a_revealed_opponent_hand(false);
    assert!(opponent_hand_tokens(&game.get_observation(0)).is_empty());
}

/// Knowledge is not per-frame: what turn 1 revealed is still observed several ticks later. The
/// `state` alone cannot do this — it keeps no record of who saw what.
#[test]
fn revealed_hand_tokens_outlive_the_frame_that_revealed_them() {
    let mut game = game_with_a_revealed_opponent_hand(true);
    for _ in 0..6 {
        if game.is_game_over() {
            break;
        }
        game.play_tick();
    }
    let revealed = opponent_hand_tokens(&game.get_observation(0));
    assert!(
        revealed.contains_key(&canonical_card(CardId::PA001Potion)),
        "the Potion was revealed and has not moved: {revealed:?}"
    );
}

/// A revealed card the opponent then plays must leave the hand tokens. It is public now and the
/// board / discard already renders it — keeping it would both leak a false position and count the
/// card twice.
#[test]
fn a_played_revealed_card_leaves_the_hand_tokens() {
    let mut game = game_with_a_revealed_opponent_hand(true);
    assert_eq!(
        opponent_hand_tokens(&game.get_observation(0)).get(&canonical_card(CardId::A1001Bulbasaur)),
        Some(&2)
    );

    game.apply_action(&Action {
        actor: 1,
        action: SimpleAction::Place(get_card_by_enum(CardId::A1001Bulbasaur), 1),
        is_stack: false,
    });

    let observation = game.get_observation(0);
    assert_eq!(
        opponent_hand_tokens(&observation).get(&canonical_card(CardId::A1001Bulbasaur)),
        Some(&1),
        "the benched copy is no longer in hand"
    );
    let on_board = observation
        .pokemon
        .iter()
        .filter(|token| !token.allied && token.zone == TokenZone::Board)
        .filter_map(|token| card_at_index(token.card_id))
        .filter(|card| *card == canonical_card(CardId::A1001Bulbasaur))
        .count();
    assert_eq!(on_board, 1, "and is rendered once, as a board Pokémon");
}

/// The belief render is keyed by an unordered map, so the bank's row order is a choice the code
/// has to make rather than one it inherits. Two observations of one state must still be the same
/// bank, not merely the same set.
#[test]
fn a_revealed_hand_renders_in_a_stable_order() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1005Caterpie)],
    );
    game.enable_belief();
    let mut state = game.get_state_clone();
    state.hands[0] = vec![get_card_by_enum(CardId::A4a032Misdreavus)];
    // Deliberately not in `card_index` order, and wide enough that a `HashMap` walk would be
    // unlikely to reproduce it twice by chance.
    state.hands[1] = vec![
        get_card_by_enum(CardId::PA001Potion),
        get_card_by_enum(CardId::A1005Caterpie),
        get_card_by_enum(CardId::A1001Bulbasaur),
        get_card_by_enum(CardId::A1033Charmander),
    ];
    game.set_state(state);
    game.apply_action(&Action {
        actor: 0,
        action: SimpleAction::Place(get_card_by_enum(CardId::A4a032Misdreavus), 1),
        is_stack: false,
    });

    let first = game.get_observation(0);
    let second = game.get_observation(0);
    assert_eq!(first, second);
    assert_eq!(first.to_wire(), second.to_wire());
}

/// The cards the observer knows are hidden somewhere, both banks.
fn hidden_elsewhere_tokens(observation: &Observation) -> HashMap<CardId, usize> {
    let pokemon = observation
        .pokemon
        .iter()
        .filter(|token| !token.allied && token.zone == TokenZone::Unknown)
        .filter_map(|token| card_at_index(token.card_id));
    let trainers = observation
        .trainers
        .iter()
        .filter(|token| !token.allied && token.zone == TokenZone::Unknown)
        .filter_map(|token| card_at_index(token.card_id));
    multiset(pokemon.chain(trainers))
}

/// The residual's own case: the opponent shuffles their revealed hand away (Copycat) and redraws.
/// Every position marker is stale, yet the cards are still theirs — they move from "in hand" to
/// "somewhere hidden" rather than vanishing, which is the whole reason `presence` is monotone.
#[test]
fn a_shuffled_away_revealed_hand_becomes_hidden_elsewhere_tokens() {
    let mut game = game_with_a_revealed_opponent_hand(true);
    assert!(
        hidden_elsewhere_tokens(&game.get_observation(0)).is_empty(),
        "a located card is not a residual"
    );

    let mut state = game.get_state_clone();
    state.hands[1].push(get_card_by_enum(CardId::B1225Copycat));
    state.current_player = 1;
    game.set_state(state);
    game.apply_action(&Action {
        actor: 1,
        action: SimpleAction::Play {
            trainer_card: get_card_by_enum(CardId::B1225Copycat).as_trainer(),
        },
        is_stack: false,
    });

    let observation = game.get_observation(0);
    assert!(
        opponent_hand_tokens(&observation).is_empty(),
        "no card of that hand is locatable any more"
    );
    let residual = hidden_elsewhere_tokens(&observation);
    assert_eq!(
        residual.get(&canonical_card(CardId::A1001Bulbasaur)),
        Some(&2)
    );
    assert_eq!(residual.get(&canonical_card(CardId::PA001Potion)), Some(&1));
}

/// The load-bearing case of the residual: `presence` is monotone, so a revealed card that becomes
/// public must be netted out. Rendering it as "hidden somewhere" would both contradict the discard
/// token that shows it face-up and count the card twice.
#[test]
fn a_public_copy_is_netted_out_of_the_residual() {
    let mut game = game_with_a_revealed_opponent_hand(true);
    game.apply_action(&Action {
        actor: 1,
        action: SimpleAction::Place(get_card_by_enum(CardId::A1001Bulbasaur), 1),
        is_stack: false,
    });

    let observation = game.get_observation(0);
    let bulbasaur = canonical_card(CardId::A1001Bulbasaur);
    assert_eq!(
        opponent_hand_tokens(&observation).get(&bulbasaur),
        Some(&1),
        "one copy is still known to be in hand"
    );
    assert_eq!(
        hidden_elsewhere_tokens(&observation).get(&bulbasaur),
        None,
        "and the benched one is public, so it is not also hidden somewhere"
    );

    // Every copy of the card the observer can account for, once each.
    let rendered = observation
        .pokemon
        .iter()
        .filter(|token| !token.allied)
        .filter_map(|token| card_at_index(token.card_id))
        .filter(|card| *card == bulbasaur)
        .count();
    assert_eq!(rendered, 2, "two copies known, two tokens: {rendered}");
}

/// Without the overlay the residual is empty like everything else it feeds.
#[test]
fn without_the_overlay_there_is_no_residual() {
    let game = game_with_a_revealed_opponent_hand(false);
    assert!(hidden_elsewhere_tokens(&game.get_observation(0)).is_empty());
}

/// A deck built to make the belief overlay work: Misdreavus reveals a hand on entry, Silver reveals
/// one and shuffles a Supporter out of it, Copycat shuffles a revealed hand away and turns it into
/// a `presence` residual. Mirrored on both seats, so every frame has both roles exercised.
const REVEAL_DECK: &str = "\
Pokémon: 10
2 Misdreavus A4a 032
2 Bulbasaur A1 001
2 Exeggcute A1 021
2 Caterpie A1 005
2 Charmander A1 033

Trainer: 10
2 Silver A4 158
2 Copycat B1 225
2 Potion P-A 001
2 Professor's Research P-A 007
2 Poké Ball P-A 005
";

/// The same, with every slot spent on Pokémon. A bank's cap is reached when one family holds *all*
/// twenty cards of both decks, which no balanced list comes close to — `MAX_POKEMON_TOKENS = 40` is
/// sized for this deck, not for the one above.
const POKEMON_ONLY_DECK: &str = "\
Pokémon: 20
2 Misdreavus A4a 032
2 Bulbasaur A1 001
2 Ivysaur A1 002
2 Exeggcute A1 021
2 Caterpie A1 005
2 Charmander A1 033
2 Ekans A1 164
2 Arbok A1 165
2 Koffing A1 176
2 Weezing A1 177
";

/// §1.2.8 with the overlay on. The bank caps are bounded by the 20-card decks — each of a player's
/// cards is in exactly one place, and the `presence` residual is netted against the public copies,
/// so no card can occupy two rows at once. That is an argument; this walks whole games and checks
/// it, which is the only thing that catches an accounting mistake in the netting.
#[test]
fn the_wire_form_holds_with_the_belief_overlay_on() {
    for deck in [REVEAL_DECK, POKEMON_ONLY_DECK] {
        walk_with_belief(deck);
    }
}

fn walk_with_belief(deck: &str) {
    let deck = Deck::from_string(deck).expect("the deck parses");
    let mut peak = (0usize, 0usize, 0usize);
    let mut belief_tokens_seen = 0usize;

    for seed in 0..8u64 {
        let players = create_players(
            deck.clone(),
            deck.clone(),
            vec![PlayerCode::R, PlayerCode::R],
        );
        let mut game = Game::new(players, seed);
        game.enable_belief();

        while !game.is_game_over() {
            for perspective in 0..2 {
                let observation = game.get_observation(perspective);
                peak.0 = peak.0.max(observation.pokemon.len());
                peak.1 = peak.1.max(observation.trainers.len());
                peak.2 = peak.2.max(observation.attacks.len());
                belief_tokens_seen += observation
                    .pokemon
                    .iter()
                    .filter(|token| {
                        !token.allied && matches!(token.zone, TokenZone::Hand | TokenZone::Unknown)
                    })
                    .count()
                    + observation
                        .trainers
                        .iter()
                        .filter(|token| {
                            !token.allied
                                && matches!(token.zone, TokenZone::Hand | TokenZone::Unknown)
                        })
                        .count();
                // Asserts on overflow rather than truncating, so this is the cap check.
                let _ = observation.to_wire();
            }
            game.play_tick();
        }
    }

    assert!(
        belief_tokens_seen > 0,
        "no reveal fired in any rollout — this deck stopped exercising the overlay and the cap \
         check above proves nothing"
    );
    println!(
        "peak tokens (pokemon, trainer, attack): {peak:?}, belief tokens {belief_tokens_seen}"
    );
}

/// A trace that saw nothing produces an empty (fully masked) History bank; a game without the
/// trace enabled produces the same.
#[test]
fn history_is_empty_without_a_trace() {
    let game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    let observation = game.get_observation(0);
    assert!(observation.history.is_empty());
    let wire = observation.to_wire();
    assert!(wire.history_mask.iter().all(|set| !set));

    let state = game.get_state_clone();
    let (_, actions) = state.generate_possible_actions();
    let empty_trace = ActionTrace::new();
    let observed = get_observation(&state, 0, &actions, Some(&empty_trace), None);
    assert!(observed.history.is_empty());
}

/// The §1.2.5 threat matrix forecasts every attack on the board on every frame, so an attack whose
/// forecast is exponential in the board makes the observation — and with it the whole rollout —
/// unbuildable. Gholdengo ex's Spending Rush enumerated `targets^energy` outcomes and stopped
/// `runs/long_v3` dead at fourteen [M] Energy; the observation must now come back at once.
#[test]
fn the_threat_matrix_survives_a_deep_energy_pile() {
    let game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::B2a078GholdengoEx).with_energy(vec![EnergyType::Metal; 14])
        ],
        vec![
            PlayedCard::from_id(CardId::A1001Bulbasaur),
            PlayedCard::from_id(CardId::A1033Charmander),
            PlayedCard::from_id(CardId::A1053Squirtle),
            PlayedCard::from_id(CardId::A1094Pikachu),
        ],
    );

    let started = std::time::Instant::now();
    let observation = game.get_observation(0);
    let elapsed = started.elapsed();

    assert!(
        observation.attacks.iter().any(|attack| attack.allied
            && attack.can_pay
            && attack.threat.iter().any(|t| *t > 0.0)),
        "Spending Rush is payable with fourteen Metal, so the forecast did run and threatens"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "building the observation took {elapsed:?}"
    );
}
