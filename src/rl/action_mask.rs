//! The action mask (§1.3.1 – §1.3.8): which of the engine's legal actions each head may pick.
//!
//! # The engine is authoritative; this is a projection
//!
//! `mask := project(generate_possible_actions(state))`. Legality is **never** reimplemented here,
//! only bucketed: every set bit carries the exact [`SimpleAction`] it resolves to, so the mask can
//! be un-projected back onto the enumeration it came from. The observation's legality features
//! ([`super::observation`]) are the *sibling* projection of that same enumeration — neither is
//! derived from the other.
//!
//! # Factorize the frequent, point at the rare
//!
//! The ten free-play families get typed argument heads keyed on Part-2 tokens; combinatorial stack
//! frames (energy distributions, card-set choices, nullary "say no") fall through to one generic
//! [`Head::CandidatePtr`] over the engine's enumerated list, so there is no fixed action space to
//! size. [`Head::CandidatePtr`] is also the **escape hatch**: any candidate a typed head cannot
//! address injectively is demoted to it rather than dropped, which is what makes the bijection of
//! §1.3.7 hold unconditionally.
//!
//! # Egocentric by role
//!
//! Every frame is scored from `frame.actor`'s perspective. Self-only heads point into the
//! *self-scoped slices* of the Part-2 token banks — the allied subsequence of the Pokémon /
//! Trainer / Attack banks — never the full mixed banks, which is the concrete halving §1.3.8 buys.
//! The two opp-role heads use a 4-slot board index; no head ever carries a player-index dimension.
//!
//! # Bijection, up to reprint identity
//!
//! §1.2.2 does not distinguish complete reprints, so neither can a pointer head: two printings of
//! one card share a `card_id` row and therefore one bit. [`unproject`](ActionMask::unproject)
//! returns one engine action per set bit, and the set equality of §1.3.7 holds modulo
//! [`canonical_action`] — which is the only sense in which it *can* hold, since the two printings
//! are indistinguishable in every observation the agent ever sees.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::actions::{Action, SimpleAction};
use crate::database::get_card_by_enum;
use crate::models::{Attack, Card, StatusCondition, TrainerCard};
use crate::State;

use super::damage::BOARD_SLOTS;
use super::ids::{canonical_card, card_index};
use super::observation::{available_attacks, Observation, TokenZone};

/// Bench slots a `Retreat` may name (`Retreat(0)` is not a thing).
pub const BENCH_SLOTS: usize = BOARD_SLOTS - 1;

/// Self-scoped bank widths (§1.3.8). A 20-card deck bounds a player's own tokens.
pub const POKEMON_SELF: usize = 20;
pub const TRAINER_SELF: usize = 20;
pub const ATTACK_SELF: usize = 16;

/// The free-play family head (§1.3.4).
pub const ACTION_TYPE_DIM: usize = 10;
/// `[Poisoned, Paralyzed, Asleep, Burned, Confused]` — the Pokémon token's status order.
pub const STATUS_CAT_DIM: usize = 5;

/// Padded widths of the two per-frame pointer heads. Like the Part-2 banks these are asserted at
/// flattening time, not at projection time: [`ActionMask`] itself is unbounded, so a frame wider
/// than the cap is still projected correctly and only [`ActionMask::to_wire`] complains.
pub const MAX_REVEALED_HAND_PTR: usize = 20;
pub const MAX_CANDIDATE_PTR: usize = 512;

// ---------------------------------------------------------------------------------------------
// Regimes
// ---------------------------------------------------------------------------------------------

/// The four mutually exclusive shapes a decision point can take (§1.3.2). [`Regime::of`] is the
/// dispatcher: exactly one applies to any state, by construction of the `match` below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    /// `turn_count == 0`: only `Place` (active first, then bench + `EndTurn`).
    Setup,
    /// `move_generation_stack` non-empty: the top frame's candidate list, for that frame's actor —
    /// which may not be the turn player (§1.3.6.1).
    Stack,
    /// Stack empty, no pending end of turn: the full turn action set.
    FreePlay,
    /// A single candidate (including `end_turn_pending` and the engine-internal frames): the
    /// engine auto-resolves it, no network forward (§1.3.6.3).
    Forced,
}

impl Regime {
    /// Classify a decision point. `Forced` wins over everything else: a one-candidate setup step or
    /// a one-candidate stack frame is auto-resolved, and §1.3.6.3 puts it in this regime. The
    /// remaining order mirrors the engine's own dispatch — a stack frame owns the enumeration
    /// whenever one is open, whatever the turn count says.
    pub fn of(state: &State, legal_actions: &[Action]) -> Self {
        if legal_actions.len() <= 1 {
            Regime::Forced
        } else if !state.move_generation_stack.is_empty() {
            Regime::Stack
        } else if state.turn_count == 0 {
            Regime::Setup
        } else {
            Regime::FreePlay
        }
    }

    /// Whether a learned decision is needed at all. `Forced` frames are resolved without a forward.
    pub fn needs_policy(self) -> bool {
        self != Regime::Forced
    }
}

// ---------------------------------------------------------------------------------------------
// Heads
// ---------------------------------------------------------------------------------------------

/// The head an action's arguments are addressed by (§1.3.3, shapes in §1.3.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Head {
    /// Categorical over the ten free-play families; read in `SETUP` / `FREE_PLAY`.
    ActionType,
    /// Self hand-Pokémon ⊗ empty slot — factorizes exactly.
    Place,
    /// Self hand-evolution → compatible slot. **Bipartite**, not an outer product: evolution X is
    /// legal only on its matching pre-evolution, so most of the rectangle is masked off.
    Evolve,
    /// Self board slot; the energy type is not a choice (it is the zone's `current`).
    AttachEnergy,
    /// Self bench slot.
    Retreat,
    /// Self Attack token.
    Attack,
    /// Self board slot.
    UseAbility,
    /// Self hand-Trainer token.
    PlayTrainer,
    UseStadium,
    EndTurn,
    /// Self board slot.
    DiscardFossil,
    /// Self board slot — the stack frames that point at one of our own Pokémon.
    SlotPtrSelf,
    /// Opponent board slot — the genuine cross-side effects (Cyrus, Field Blower, gust, spot
    /// damage). A 4-slot board index, never an opponent token bank (§1.3.8).
    SlotPtrOpp,
    /// Self board `(from, to)`.
    SlotPair,
    /// Self hand-Pokémon token.
    HandPtr,
    /// One of the five Special Conditions.
    StatusCat,
    /// The opponent's revealed hand subset, in the order the frame enumerated it (§1.3.6.2).
    RevealedHandPtr,
    /// Per-frame candidate list — sets, assignments, nullary choices, and the demotion target for
    /// anything a typed head cannot address injectively.
    CandidatePtr,
}

/// Every head, in wire order.
pub const HEADS: [Head; 18] = [
    Head::ActionType,
    Head::Place,
    Head::Evolve,
    Head::AttachEnergy,
    Head::Retreat,
    Head::Attack,
    Head::UseAbility,
    Head::PlayTrainer,
    Head::UseStadium,
    Head::EndTurn,
    Head::DiscardFossil,
    Head::SlotPtrSelf,
    Head::SlotPtrOpp,
    Head::SlotPair,
    Head::HandPtr,
    Head::StatusCat,
    Head::RevealedHandPtr,
    Head::CandidatePtr,
];

impl Head {
    /// Flat width of the head's argument domain (§1.3.8).
    pub const fn dim(self) -> usize {
        match self {
            Head::ActionType => ACTION_TYPE_DIM,
            Head::Place | Head::Evolve => POKEMON_SELF * BOARD_SLOTS,
            Head::AttachEnergy | Head::UseAbility | Head::DiscardFossil => BOARD_SLOTS,
            Head::Retreat => BENCH_SLOTS,
            Head::Attack => ATTACK_SELF,
            Head::PlayTrainer => TRAINER_SELF,
            Head::UseStadium | Head::EndTurn => 1,
            Head::SlotPtrSelf | Head::SlotPtrOpp => BOARD_SLOTS,
            Head::SlotPair => BOARD_SLOTS * BOARD_SLOTS,
            Head::HandPtr => POKEMON_SELF,
            Head::StatusCat => STATUS_CAT_DIM,
            Head::RevealedHandPtr => MAX_REVEALED_HAND_PTR,
            Head::CandidatePtr => MAX_CANDIDATE_PTR,
        }
    }

    /// Offset of the head's block in the flat wire vector.
    pub const fn offset(self) -> usize {
        let mut offset = 0;
        let mut position = 0;
        while position < HEADS.len() {
            if HEADS[position] as usize == self as usize {
                return offset;
            }
            offset += HEADS[position].dim();
            position += 1;
        }
        offset
    }

    /// Whether the head addresses opponent-role entities (§1.3.6.1 cross-target).
    pub const fn is_opponent_role(self) -> bool {
        matches!(self, Head::SlotPtrOpp | Head::RevealedHandPtr)
    }
}

/// Total width of the flat mask.
pub const ACTION_MASK_DIM: usize = Head::CandidatePtr.offset() + Head::CandidatePtr.dim();

/// The ten free-play families the `action_type` head chooses between (§1.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionFamily {
    EndTurn,
    Place,
    Evolve,
    AttachEnergy,
    Retreat,
    Attack,
    UseAbility,
    PlayTrainer,
    UseStadium,
    DiscardFossil,
}

impl ActionFamily {
    /// Every family, in `action_type` index order.
    pub const ALL: [ActionFamily; ACTION_TYPE_DIM] = [
        ActionFamily::EndTurn,
        ActionFamily::Place,
        ActionFamily::Evolve,
        ActionFamily::AttachEnergy,
        ActionFamily::Retreat,
        ActionFamily::Attack,
        ActionFamily::UseAbility,
        ActionFamily::PlayTrainer,
        ActionFamily::UseStadium,
        ActionFamily::DiscardFossil,
    ];

    /// Position in the `action_type` head.
    pub const fn index(self) -> usize {
        match self {
            ActionFamily::EndTurn => 0,
            ActionFamily::Place => 1,
            ActionFamily::Evolve => 2,
            ActionFamily::AttachEnergy => 3,
            ActionFamily::Retreat => 4,
            ActionFamily::Attack => 5,
            ActionFamily::UseAbility => 6,
            ActionFamily::PlayTrainer => 7,
            ActionFamily::UseStadium => 8,
            ActionFamily::DiscardFossil => 9,
        }
    }

    /// The argument head this family's choice is then made on.
    pub const fn head(self) -> Head {
        match self {
            ActionFamily::EndTurn => Head::EndTurn,
            ActionFamily::Place => Head::Place,
            ActionFamily::Evolve => Head::Evolve,
            ActionFamily::AttachEnergy => Head::AttachEnergy,
            ActionFamily::Retreat => Head::Retreat,
            ActionFamily::Attack => Head::Attack,
            ActionFamily::UseAbility => Head::UseAbility,
            ActionFamily::PlayTrainer => Head::PlayTrainer,
            ActionFamily::UseStadium => Head::UseStadium,
            ActionFamily::DiscardFossil => Head::DiscardFossil,
        }
    }

    /// The family an argument head belongs to, if any. The stack-only heads have none.
    pub const fn of_head(head: Head) -> Option<ActionFamily> {
        match head {
            Head::EndTurn => Some(ActionFamily::EndTurn),
            Head::Place => Some(ActionFamily::Place),
            Head::Evolve => Some(ActionFamily::Evolve),
            Head::AttachEnergy => Some(ActionFamily::AttachEnergy),
            Head::Retreat => Some(ActionFamily::Retreat),
            Head::Attack => Some(ActionFamily::Attack),
            Head::UseAbility => Some(ActionFamily::UseAbility),
            Head::PlayTrainer => Some(ActionFamily::PlayTrainer),
            Head::UseStadium => Some(ActionFamily::UseStadium),
            Head::DiscardFossil => Some(ActionFamily::DiscardFossil),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The mask
// ---------------------------------------------------------------------------------------------

/// One set bit: a head, a position in that head's domain, and the engine action it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskEntry {
    pub head: Head,
    /// Position within [`Head::dim`].
    pub index: usize,
    /// The action `apply_action` will accept if this bit is selected.
    pub action: SimpleAction,
    /// Whether the action came off a stack frame. `apply_action` pops the frame iff this is set,
    /// so it must survive the projection round-trip — resolving a stack frame with it unset would
    /// leave the frame open and re-enumerate it forever.
    pub is_stack: bool,
}

/// The legal-action mask of one decision point, from its actor's perspective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionMask {
    /// `frame.actor` — the player this mask is for, which may not be the turn player (§1.3.6.1).
    pub actor: usize,
    pub regime: Regime,
    /// The `action_type` head: families with ≥ 1 legal instantiation.
    pub family: [bool; ACTION_TYPE_DIM],
    /// Every set bit, in the order the engine enumerated its action.
    pub entries: Vec<MaskEntry>,
}

/// The padded, flat form the model consumes: one bool per head slot, heads concatenated in
/// [`HEADS`] order.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionMaskWire {
    pub regime: Regime,
    pub actor: usize,
    pub bits: Vec<bool>,
}

impl ActionMaskWire {
    /// The block of one head.
    pub fn head(&self, head: Head) -> &[bool] {
        &self.bits[head.offset()..head.offset() + head.dim()]
    }
}

impl ActionMask {
    /// Whether `(head, index)` is legal.
    pub fn is_set(&self, head: Head, index: usize) -> bool {
        self.action_at(head, index).is_some()
    }

    /// The action `(head, index)` resolves to.
    pub fn action_at(&self, head: Head, index: usize) -> Option<&SimpleAction> {
        self.entries
            .iter()
            .find(|entry| entry.head == head && entry.index == index)
            .map(|entry| &entry.action)
    }

    /// The bool vector of one head. [`Head::ActionType`] has no entries of its own — it is the
    /// family mask, i.e. the emptiness of the argument heads — and is returned as such.
    pub fn bits(&self, head: Head) -> Vec<bool> {
        if head == Head::ActionType {
            return self.family.to_vec();
        }
        let mut bits = vec![false; head.dim()];
        for entry in self.entries.iter().filter(|entry| entry.head == head) {
            if entry.index < bits.len() {
                bits[entry.index] = true;
            }
        }
        bits
    }

    /// Heads carrying at least one set bit.
    pub fn active_heads(&self) -> Vec<Head> {
        let mut heads = Vec::new();
        for head in HEADS {
            if self.entries.iter().any(|entry| entry.head == head) {
                heads.push(head);
            }
        }
        heads
    }

    /// The actions the set bits resolve to — the inverse of the projection (§1.3.7 invariant 1).
    /// Equal to `generate_possible_actions`' enumeration **as a set, up to [`canonical_action`]**:
    /// two printings of one card share a pointer row and therefore one bit.
    pub fn unproject(&self) -> Vec<SimpleAction> {
        self.entries
            .iter()
            .map(|entry| entry.action.clone())
            .collect()
    }

    /// The single action of a [`Regime::Forced`] frame — apply it without a network forward.
    pub fn forced_action(&self) -> Option<Action> {
        if self.regime != Regime::Forced {
            return None;
        }
        self.entries.first().map(|entry| self.engine_action(entry))
    }

    /// Turn a chosen `(head, index)` back into the engine action, `is_stack` included — the whole
    /// round-trip of §1.3.7 invariant 3, ready for `apply_action`.
    pub fn select(&self, head: Head, index: usize) -> Option<Action> {
        self.entries
            .iter()
            .find(|entry| entry.head == head && entry.index == index)
            .map(|entry| self.engine_action(entry))
    }

    fn engine_action(&self, entry: &MaskEntry) -> Action {
        Action {
            actor: self.actor,
            action: entry.action.clone(),
            is_stack: entry.is_stack,
        }
    }

    /// Flatten to the wire. Panics if a per-frame head overflows its cap — an assert rather than a
    /// silent truncation, exactly as the Part-2 banks do: dropping a candidate would make the mask
    /// disagree with the engine instead of merely degrading.
    pub fn to_wire(&self) -> ActionMaskWire {
        let mut bits = vec![false; ACTION_MASK_DIM];
        for family in ActionFamily::ALL {
            bits[Head::ActionType.offset() + family.index()] = self.family[family.index()];
        }
        for entry in &self.entries {
            assert!(
                entry.index < entry.head.dim(),
                "{:?} index {} overflows its {} slots",
                entry.head,
                entry.index,
                entry.head.dim()
            );
            bits[entry.head.offset() + entry.index] = true;
        }
        ActionMaskWire {
            regime: self.regime,
            actor: self.actor,
            bits,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------------------------

/// Project `legal_actions` onto the Part-3 heads, from `observation.perspective`'s point of view.
///
/// `legal_actions` is the output of `generate_possible_actions(state)` — the *same* enumeration
/// `get_observation` projected its legality bits from, passed in rather than recomputed so the two
/// projections cannot drift. `observation` supplies the self-scoped token banks the pointer heads
/// index into, so the head indices name the very encoder rows Part 4 reads.
///
/// # Panics
///
/// If `legal_actions` is empty (§1.3.7 invariant 2 — the engine always enumerates at least one
/// candidate), or if `observation.perspective` is not the frame's actor (invariant 5).
pub fn project(state: &State, legal_actions: &[Action], observation: &Observation) -> ActionMask {
    assert!(
        !legal_actions.is_empty(),
        "generate_possible_actions never returns an empty enumeration (§1.3.7 invariant 2)"
    );
    let actor = legal_actions[0].actor;
    assert_eq!(
        observation.perspective, actor,
        "the observation of a decision point is taken from its actor's perspective (§1.3.7)"
    );
    debug_assert!(
        legal_actions.iter().all(|action| action.actor == actor),
        "a frame has a single actor"
    );

    let banks = SelfBanks::of(state, observation, actor);
    let regime = Regime::of(state, legal_actions);

    // 1. Route each candidate to a typed head, or leave it for a per-frame pointer.
    let routings: Vec<Routing> = legal_actions
        .iter()
        .map(|action| route(actor, &action.action, &banks))
        .collect();

    // 2. A typed head that would address two *distinct* actions with one bit cannot be inverted.
    //    Demote the whole head to `CandidatePtr` rather than lose a candidate — this is what makes
    //    the §1.3.7 bijection hold for every frame the engine can produce, not just the tidy ones.
    let mut claimed: HashMap<(Head, usize), SimpleAction> = HashMap::new();
    let mut colliding: HashSet<Head> = HashSet::new();
    for (action, routing) in legal_actions.iter().zip(&routings) {
        let Routing::Bits(head, indices) = routing else {
            continue;
        };
        let canonical = canonical_action(&action.action);
        for index in indices {
            match claimed.get(&(*head, *index)) {
                Some(existing) if *existing != canonical => {
                    colliding.insert(*head);
                }
                Some(_) => {}
                None => {
                    claimed.insert((*head, *index), canonical.clone());
                }
            }
        }
    }

    // 3. Emit the entries, assigning the per-frame pointer heads their positional indices.
    let mut mask = ActionMask {
        actor,
        regime,
        family: [false; ACTION_TYPE_DIM],
        entries: Vec::with_capacity(legal_actions.len()),
    };
    let mut taken: HashSet<(Head, usize)> = HashSet::new();
    let mut revealed_next = 0;
    let mut candidate_next = 0;

    for (action, routing) in legal_actions.iter().zip(&routings) {
        match routing {
            Routing::Bits(head, indices) if !colliding.contains(head) => {
                if let Some(family) = ActionFamily::of_head(*head) {
                    mask.family[family.index()] = true;
                }
                for index in indices {
                    if taken.insert((*head, *index)) {
                        mask.entries.push(MaskEntry {
                            head: *head,
                            index: *index,
                            action: action.action.clone(),
                            is_stack: action.is_stack,
                        });
                    }
                }
            }
            Routing::Revealed => {
                mask.entries.push(MaskEntry {
                    head: Head::RevealedHandPtr,
                    index: revealed_next,
                    action: action.action.clone(),
                    is_stack: action.is_stack,
                });
                revealed_next += 1;
            }
            _ => {
                mask.entries.push(MaskEntry {
                    head: Head::CandidatePtr,
                    index: candidate_next,
                    action: action.action.clone(),
                    is_stack: action.is_stack,
                });
                candidate_next += 1;
            }
        }
    }

    mask
}

/// Where a candidate lands before the per-frame pointers get their positional indices.
enum Routing {
    /// A typed head and the rows it addresses (more than one when identical copies of a card sit
    /// in the same zone — the policy may point at either, they are the same play).
    Bits(Head, Vec<usize>),
    /// [`Head::RevealedHandPtr`].
    Revealed,
    /// [`Head::CandidatePtr`].
    Candidate,
}

impl Routing {
    /// A typed routing, falling back to [`Head::CandidatePtr`] when the head cannot express it:
    /// no matching token row, or a row past the head's self-scoped width. The fallback is the
    /// reason no candidate is ever dropped.
    fn bits(head: Head, indices: impl IntoIterator<Item = usize>) -> Routing {
        let indices: Vec<usize> = indices
            .into_iter()
            .filter(|index| *index < head.dim())
            .collect();
        if indices.is_empty() {
            Routing::Candidate
        } else {
            Routing::Bits(head, indices)
        }
    }
}

/// The §1.3.3 taxonomy, as one exhaustive match over the engine's action enum — so a variant added
/// to `SimpleAction` is a compile error here rather than a silently unmasked action.
fn route(actor: usize, action: &SimpleAction, banks: &SelfBanks) -> Routing {
    // A slot pointer resolves the absolute `player` of a cross-side action to a self/opp *role*
    // (§1.3.6.1): the head never carries a player-index dimension.
    let slot_ptr = |player: usize, in_play_idx: usize| {
        let head = if player == actor {
            Head::SlotPtrSelf
        } else {
            Head::SlotPtrOpp
        };
        Routing::bits(head, [in_play_idx])
    };

    match action {
        // -- Free play, factorized (§1.3.4) ---------------------------------------------------
        SimpleAction::EndTurn => Routing::bits(Head::EndTurn, [0]),
        SimpleAction::UseStadium => Routing::bits(Head::UseStadium, [0]),
        SimpleAction::Place(card, slot) => Routing::bits(
            Head::Place,
            banks
                .pokemon_rows(TokenZone::Hand, card)
                .into_iter()
                .map(|row| row * BOARD_SLOTS + slot),
        ),
        SimpleAction::Evolve {
            evolution,
            in_play_idx,
            from_deck,
        } => {
            // Rare-Candy-style evolutions come off the deck, and the deck is part of the self
            // Pokémon bank — the same head addresses both, only the source zone changes.
            let zone = if *from_deck {
                TokenZone::Deck
            } else {
                TokenZone::Hand
            };
            Routing::bits(
                Head::Evolve,
                banks
                    .pokemon_rows(zone, evolution)
                    .into_iter()
                    .map(|row| row * BOARD_SLOTS + in_play_idx),
            )
        }
        SimpleAction::UseAbility { in_play_idx } => Routing::bits(Head::UseAbility, [*in_play_idx]),
        SimpleAction::DiscardFossil { in_play_idx } => {
            Routing::bits(Head::DiscardFossil, [*in_play_idx])
        }
        SimpleAction::Retreat(in_play_idx) => match in_play_idx.checked_sub(1) {
            Some(bench) => Routing::bits(Head::Retreat, [bench]),
            None => Routing::Candidate,
        },
        SimpleAction::Attack(attack) => Routing::bits(Head::Attack, banks.attack_rows(attack)),
        SimpleAction::Play { trainer_card } => {
            Routing::bits(Head::PlayTrainer, banks.trainer_rows(trainer_card))
        }
        // The turn's energy: the type is the zone's `current`, so only the destination is a choice.
        // Anything else riding on `Attach` is a distribution and belongs to the candidate pointer.
        SimpleAction::Attach {
            attachments,
            is_turn_energy: true,
        } if attachments.len() == 1 => Routing::bits(Head::AttachEnergy, [attachments[0].2]),

        // -- Stack frames, typed arguments (§1.3.5) --------------------------------------------
        SimpleAction::Heal { in_play_idx, .. }
        | SimpleAction::AttachFromDiscard { in_play_idx, .. }
        | SimpleAction::AttachTypedFromDiscard { in_play_idx, .. }
        | SimpleAction::ReturnPokemonToHand { in_play_idx }
        | SimpleAction::ShuffleInPlayPokemonIntoDeck { in_play_idx }
        | SimpleAction::AttachTool { in_play_idx, .. } => slot_ptr(actor, *in_play_idx),
        // Promotion after a KO and Cyrus-style drags are the same action; the `player` field is
        // what decides whether the head is self- or opp-role.
        SimpleAction::Activate {
            player,
            in_play_idx,
        }
        | SimpleAction::DiscardToolFromPokemon {
            player,
            in_play_idx,
        } => slot_ptr(*player, *in_play_idx),
        // Spot-damage target selection: a genuine multi-candidate choice on the opponent's board,
        // not the engine-internal `ApplyDamage` of an attack's own resolution (which is never a
        // frame). A multi-target payload has no slot to point at and falls through.
        SimpleAction::ApplyDamage { targets, .. } if targets.len() == 1 => {
            slot_ptr(targets[0].1, targets[0].2)
        }
        SimpleAction::ScheduleDelayedSpotDamage {
            target_player,
            target_in_play_idx,
            ..
        } => slot_ptr(*target_player, *target_in_play_idx),
        SimpleAction::MoveEnergy {
            from_in_play_idx,
            to_in_play_idx,
            ..
        } => Routing::bits(
            Head::SlotPair,
            [from_in_play_idx * BOARD_SLOTS + to_in_play_idx],
        ),
        SimpleAction::MoveAllDamage { from, to } => {
            Routing::bits(Head::SlotPair, [from * BOARD_SLOTS + to])
        }
        SimpleAction::CommunicatePokemon { hand_pokemon } => Routing::bits(
            Head::HandPtr,
            banks.pokemon_rows(TokenZone::Hand, hand_pokemon),
        ),
        SimpleAction::ApplyStatusToOpponentActive { condition } => {
            Routing::bits(Head::StatusCat, [status_index(*condition)])
        }

        // -- Reveal effects (§1.3.6.2) ----------------------------------------------------------
        SimpleAction::ShuffleOpponentSupporter { .. }
        | SimpleAction::DiscardOpponentSupporter { .. } => Routing::Revealed,

        // -- Everything else: sets, assignments, nullary choices, internal frames ---------------
        SimpleAction::Attach { .. }
        | SimpleAction::SadaAttach { .. }
        | SimpleAction::HealAndDiscardEnergy { .. }
        | SimpleAction::ShufflePokemonIntoDeck { .. }
        | SimpleAction::ShuffleOwnCardsIntoDeck { .. }
        | SimpleAction::DiscardOwnCards { .. }
        | SimpleAction::SwitchHandCardForRandomTool { .. }
        | SimpleAction::ApplyEeveeBagDamageBoost
        | SimpleAction::HealAllEeveeEvolutions
        | SimpleAction::DiscardActiveStadium
        | SimpleAction::DiscardRandomOpponentActiveEnergy
        | SimpleAction::ApplyDamage { .. }
        | SimpleAction::DrawCard { .. }
        | SimpleAction::Noop => Routing::Candidate,
    }
}

/// Position of a Special Condition in the `STATUS_CAT` head — the same order the Pokémon token's
/// `status` block uses.
pub const fn status_index(condition: StatusCondition) -> usize {
    match condition {
        StatusCondition::Poisoned => 0,
        StatusCondition::Paralyzed => 1,
        StatusCondition::Asleep => 2,
        StatusCondition::Burned => 3,
        StatusCondition::Confused => 4,
    }
}

// ---------------------------------------------------------------------------------------------
// Self-scoped token banks
// ---------------------------------------------------------------------------------------------

/// The allied subsequences of the Part-2 banks (§1.3.8). A head index is a position *in these*,
/// not in the full mixed banks — that is the halving the egocentric encoding buys, and it is what
/// makes a head index name an encoder row without a player dimension.
struct SelfBanks {
    /// `(zone, card_id)` of each allied Pokémon token, in bank order.
    pokemon: Vec<(TokenZone, u32)>,
    /// `(zone, card_id)` of each allied Trainer token, in bank order.
    trainer: Vec<(TokenZone, u32)>,
    /// `(board slot, attack)` of each allied Attack token, in bank order.
    attacks: Vec<(usize, Attack)>,
}

impl SelfBanks {
    fn of(state: &State, observation: &Observation, actor: usize) -> Self {
        let pokemon = observation
            .pokemon
            .iter()
            .filter(|token| token.allied)
            .map(|token| (token.zone, token.card_id))
            .collect();
        let trainer = observation
            .trainers
            .iter()
            .filter(|token| token.allied)
            .map(|token| (token.zone, token.card_id))
            .collect();

        // The Attack bank carries no attack payload (the descriptor is gathered in-model from
        // `src_card_id` + `attack_slot`), so the affordance enumeration is walked again here, in
        // the very order `get_observation` emits it.
        let mut attacks = Vec::new();
        for (slot, occupant) in state.in_play_pokemon[actor].iter().enumerate() {
            let Some(played) = occupant.as_ref() else {
                continue;
            };
            for (_, _, attack) in available_attacks(state, actor, played) {
                attacks.push((slot, attack));
            }
        }
        debug_assert_eq!(
            attacks.len(),
            observation
                .attacks
                .iter()
                .filter(|token| token.allied)
                .count(),
            "the self Attack slice and the affordance enumeration must stay in step"
        );

        SelfBanks {
            pokemon,
            trainer,
            attacks,
        }
    }

    /// Rows of the self Pokémon slice holding `card` in `zone`. More than one when the player owns
    /// several copies — they are interchangeable, so every copy's bit is legal.
    fn pokemon_rows(&self, zone: TokenZone, card: &Card) -> Vec<usize> {
        let id = card_index(card.get_card_id());
        self.pokemon
            .iter()
            .enumerate()
            .filter(|(_, (token_zone, card_id))| *token_zone == zone && *card_id == id)
            .map(|(row, _)| row)
            .collect()
    }

    /// Rows of the self Trainer slice holding this card **in hand** — the only zone a `Play` can
    /// come from.
    fn trainer_rows(&self, trainer: &TrainerCard) -> Vec<usize> {
        let Some(id) = crate::card_ids::CardId::from_card_id(&trainer.id).map(card_index) else {
            return Vec::new();
        };
        self.trainer
            .iter()
            .enumerate()
            .filter(|(_, (zone, card_id))| *zone == TokenZone::Hand && *card_id == id)
            .map(|(row, _)| row)
            .collect()
    }

    /// Rows of the self Attack slice offering this attack. Restricted to the **active**: only the
    /// active Pokémon can attack, while the bank also carries the bench's affordances for the
    /// threat matrix (§1.2.5).
    fn attack_rows(&self, attack: &Attack) -> Vec<usize> {
        self.attacks
            .iter()
            .enumerate()
            .filter(|(_, (slot, candidate))| *slot == 0 && candidate == attack)
            .map(|(row, _)| row)
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Reprint identity
// ---------------------------------------------------------------------------------------------

/// An action with every card it names replaced by its canonical printing (§1.2.2 — complete
/// reprints are one card). Two actions that differ only by which printing they touch are the same
/// play, share a `card_id` row, and must therefore share a mask bit; this is the equivalence the
/// §1.3.7 bijection is stated modulo.
pub fn canonical_action(action: &SimpleAction) -> SimpleAction {
    let canonical = |card: &Card| get_card_by_enum(canonical_card(card.get_card_id()));
    let canonical_all = |cards: &[Card]| cards.iter().map(canonical).collect::<Vec<_>>();

    match action {
        SimpleAction::Place(card, slot) => SimpleAction::Place(canonical(card), *slot),
        SimpleAction::Evolve {
            evolution,
            in_play_idx,
            from_deck,
        } => SimpleAction::Evolve {
            evolution: canonical(evolution),
            in_play_idx: *in_play_idx,
            from_deck: *from_deck,
        },
        SimpleAction::Play { trainer_card } => SimpleAction::Play {
            trainer_card: canonical_trainer(trainer_card),
        },
        SimpleAction::AttachTool {
            in_play_idx,
            tool_card,
        } => SimpleAction::AttachTool {
            in_play_idx: *in_play_idx,
            tool_card: canonical(tool_card),
        },
        SimpleAction::CommunicatePokemon { hand_pokemon } => SimpleAction::CommunicatePokemon {
            hand_pokemon: canonical(hand_pokemon),
        },
        SimpleAction::SwitchHandCardForRandomTool { hand_card } => {
            SimpleAction::SwitchHandCardForRandomTool {
                hand_card: canonical(hand_card),
            }
        }
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => {
            SimpleAction::ShufflePokemonIntoDeck {
                hand_pokemon: canonical_all(hand_pokemon),
            }
        }
        SimpleAction::ShuffleOwnCardsIntoDeck { cards } => SimpleAction::ShuffleOwnCardsIntoDeck {
            cards: canonical_all(cards),
        },
        SimpleAction::DiscardOwnCards { cards } => SimpleAction::DiscardOwnCards {
            cards: canonical_all(cards),
        },
        SimpleAction::ShuffleOpponentSupporter { supporter_card } => {
            SimpleAction::ShuffleOpponentSupporter {
                supporter_card: canonical(supporter_card),
            }
        }
        SimpleAction::DiscardOpponentSupporter { supporter_card } => {
            SimpleAction::DiscardOpponentSupporter {
                supporter_card: canonical(supporter_card),
            }
        }
        other => other.clone(),
    }
}

fn canonical_trainer(trainer: &TrainerCard) -> TrainerCard {
    crate::card_ids::CardId::from_card_id(&trainer.id)
        .map(|card_id| get_card_by_enum(canonical_card(card_id)).as_trainer())
        .unwrap_or_else(|| trainer.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_ids::CardId;
    use crate::models::PlayedCard;
    use crate::rl::get_observation;
    use crate::test_support::{get_test_game_with_board, init_random_players};
    use crate::Game;

    fn mask_of(game: &crate::Game<'_>) -> ActionMask {
        let state = game.get_state_clone();
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(&state, actor, &actions, None, None);
        project(&state, &actions, &observation)
    }

    #[test]
    fn the_flat_layout_tiles_the_heads_without_overlap() {
        let mut expected = 0;
        for head in HEADS {
            assert_eq!(head.offset(), expected, "{head:?}");
            expected += head.dim();
        }
        assert_eq!(ACTION_MASK_DIM, expected);
    }

    #[test]
    fn every_family_maps_to_its_own_head_and_index() {
        for (index, family) in ActionFamily::ALL.iter().enumerate() {
            assert_eq!(family.index(), index);
            assert_eq!(ActionFamily::of_head(family.head()), Some(*family));
        }
    }

    #[test]
    fn free_play_families_are_the_engine_enumeration_reshaped() {
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1033Charmander)],
        );
        let mask = mask_of(&game);
        assert_eq!(mask.regime, Regime::FreePlay);
        assert_eq!(mask.actor, 0);

        // Ending the turn is always available in free play.
        assert!(mask.family[ActionFamily::EndTurn.index()]);
        assert!(mask.is_set(Head::EndTurn, 0));
        // Nothing on the bench to retreat to, and no fossil in play.
        assert_eq!(mask.bits(Head::Retreat), vec![false; BENCH_SLOTS]);
        assert!(!mask.family[ActionFamily::DiscardFossil.index()]);

        // A family bit is set exactly when its head carries one: the `action_type` head is the
        // argument heads' own emptiness, not a second opinion on legality.
        for family in ActionFamily::ALL {
            assert_eq!(
                mask.family[family.index()],
                mask.bits(family.head()).iter().any(|set| *set),
                "{family:?}"
            );
        }
    }

    #[test]
    fn a_place_bit_is_the_hand_token_row_crossed_with_the_slot() {
        let game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1033Charmander)],
        );
        let state = game.get_state_clone();
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(&state, actor, &actions, None, None);
        let mask = project(&state, &actions, &observation);

        let self_pokemon: Vec<_> = observation
            .pokemon
            .iter()
            .filter(|token| token.allied)
            .collect();

        for entry in mask.entries.iter().filter(|e| e.head == Head::Place) {
            let (row, slot) = (entry.index / BOARD_SLOTS, entry.index % BOARD_SLOTS);
            let token = self_pokemon[row];
            assert_eq!(token.zone, TokenZone::Hand);
            let SimpleAction::Place(card, action_slot) = &entry.action else {
                panic!("the PLACE head only ever holds Place actions");
            };
            assert_eq!(*action_slot, slot);
            assert_eq!(token.card_id, card_index(card.get_card_id()));
            assert!(
                state.in_play_pokemon[actor][slot].is_none(),
                "an empty slot"
            );
        }
    }

    /// An `ATTACK` bit's index is a row of the allied Attack-token slice, and the token at that
    /// row is the very attack the bit selects — the ordering contract [`SelfBanks`] shares with
    /// `get_observation`, which the count-only `debug_assert` there cannot see.
    #[test]
    fn an_attack_bit_names_the_token_of_its_own_attack() {
        let mut saw_attack = false;
        for seed in 0..8u64 {
            let mut game = Game::new(init_random_players(), seed);
            while !game.is_game_over() {
                let state = game.get_state_clone();
                let (actor, actions) = state.generate_possible_actions();
                let observation = get_observation(&state, actor, &actions, None, None);
                let mask = project(&state, &actions, &observation);

                let allied_attacks: Vec<_> =
                    observation.attacks.iter().filter(|t| t.allied).collect();
                for entry in mask.entries.iter().filter(|e| e.head == Head::Attack) {
                    saw_attack = true;
                    let SimpleAction::Attack(selected) = &entry.action else {
                        panic!("the ATTACK head only ever holds Attack actions");
                    };
                    let token = allied_attacks[entry.index];
                    // Only the active attacks; the row's parent must be it.
                    let parent = &observation.pokemon[token.parent_pokemon_ref as usize];
                    assert!(parent.allied, "seed {seed}");
                    assert_eq!(parent.slot, Some(0), "seed {seed}");
                    // The token's (source, slot) resolve to the selected attack.
                    let active = state.in_play_pokemon[actor][0]
                        .as_ref()
                        .expect("an attack implies an active");
                    let (source, attack_slot, _) = available_attacks(&state, actor, active)
                        .into_iter()
                        .find(|(_, _, attack)| attack == selected)
                        .expect("a legal attack is one of the active's affordances");
                    assert_eq!(
                        token.src_card_id,
                        card_index(source.get_card_id()),
                        "seed {seed}"
                    );
                    assert_eq!(token.attack_slot, attack_slot, "seed {seed}");
                }

                game.play_tick();
            }
        }
        assert!(saw_attack, "the rollouts must actually reach an attack");
    }

    #[test]
    fn a_single_candidate_frame_is_forced() {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1033Charmander)],
        );
        let mut state = game.get_state_clone();
        state.move_generation_stack.push((
            1,
            vec![SimpleAction::Activate {
                player: 1,
                in_play_idx: 0,
            }],
        ));
        game.set_state(state);

        let mask = mask_of(&game);
        assert_eq!(mask.regime, Regime::Forced);
        assert_eq!(mask.actor, 1, "the frame's actor is not the turn player");
        assert_eq!(mask.entries.len(), 1);
        let forced = mask.forced_action().expect("a forced frame has its action");
        assert!(
            forced.is_stack,
            "resolving a stack frame must pop it — apply_action reads is_stack for that"
        );
    }

    #[test]
    fn a_cross_target_frame_uses_the_opponent_role_head() {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![
                PlayedCard::from_id(CardId::A1033Charmander),
                PlayedCard::from_id(CardId::A1055Blastoise),
            ],
        );
        let mut state = game.get_state_clone();
        // Cyrus: the turn player drags one of the *opponent's* benched Pokémon up.
        state.move_generation_stack.push((
            0,
            vec![
                SimpleAction::Activate {
                    player: 1,
                    in_play_idx: 1,
                },
                SimpleAction::Noop,
            ],
        ));
        game.set_state(state);

        let mask = mask_of(&game);
        assert_eq!(mask.regime, Regime::Stack);
        assert_eq!(mask.actor, 0);
        assert!(mask.is_set(Head::SlotPtrOpp, 1));
        assert!(!mask.is_set(Head::SlotPtrSelf, 1));
        // "Say no" stays a candidate of its own (§1.3.6.3).
        assert!(mask.is_set(Head::CandidatePtr, 0));
        assert_eq!(
            mask.action_at(Head::CandidatePtr, 0),
            Some(&SimpleAction::Noop)
        );
    }

    #[test]
    fn a_colliding_typed_head_is_demoted_rather_than_losing_a_candidate() {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1033Charmander)],
        );
        let mut state = game.get_state_clone();
        // Two distinct heals on one slot: the SLOT_PTR head has a single bit for slot 0, so it
        // cannot tell them apart and the whole head steps aside.
        state.move_generation_stack.push((
            0,
            vec![
                SimpleAction::Heal {
                    in_play_idx: 0,
                    amount: 20,
                    cure_status: false,
                },
                SimpleAction::Heal {
                    in_play_idx: 0,
                    amount: 50,
                    cure_status: false,
                },
            ],
        ));
        game.set_state(state);

        let mask = mask_of(&game);
        assert_eq!(mask.active_heads(), vec![Head::CandidatePtr]);
        assert_eq!(mask.entries.len(), 2, "no candidate is lost");
    }

    #[test]
    fn canonicalization_collapses_printings_and_leaves_the_rest_alone() {
        let reprint = CardId::A4b184LunalaEx;
        let original = canonical_card(reprint);
        assert_ne!(reprint, original);

        let place_reprint = SimpleAction::Place(get_card_by_enum(reprint), 1);
        let place_original = SimpleAction::Place(get_card_by_enum(original), 1);
        assert_ne!(place_reprint, place_original);
        assert_eq!(
            canonical_action(&place_reprint),
            canonical_action(&place_original)
        );
        assert_eq!(
            canonical_action(&SimpleAction::EndTurn),
            SimpleAction::EndTurn
        );
    }
}
