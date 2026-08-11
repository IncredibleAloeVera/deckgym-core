//! The three identity granularities of §1.2.2, and the frozen index spaces they map onto.
//!
//! The observation carries *indices*, never payloads: the heavy static descriptor lives in a table
//! gathered in-model by `card_id`. Three embedding tables, kept distinct and concatenated at the
//! Pokémon MLP input:
//!
//! - [`card_index`] — the exact printed card (finest grain).
//! - [`species_index`] — the named Pokémon across all its printings (every "Pikachu" card → one id).
//! - [`line_index`] — the whole evolution lineage (Charmander/Charmeleon/Charizard + variants).
//!
//! "Finest grain" still does **not** distinguish *complete* reprints (§1.2.2): a card re-printed
//! unchanged — a different rarity in the same set, a promo, or the whole A4b reprint set — is the
//! same card to play with, so all its printings share one embedding row via [`canonical_card`].
//! Splitting them would shard the same card's statistics across rows that can never be
//! distinguished by any observation, which is exactly the sample efficiency a closed pool buys.
//! On the current pool this collapses 3520 printings onto **2086 rows** (−40.7%): the whole 379-card
//! A4b set re-prints earlier cards and owns no row at all, and 585 further groups come from
//! alternate rarities and promos (Lunala ex alone is printed seven times).
//!
//! Index `0` is reserved as PAD / none / hidden in **every** space, so a padded token slot and an
//! absent reference share one encoding. Real cards therefore start at `1`.
//!
//! The grouping is derived from the frozen pool itself (`evolves_from` chains + card names) and is
//! deterministic: species ids follow the sorted species key, line ids the sorted representative of
//! each lineage. Freezing the pool is what makes this legitimate (Part 1: closed card pool).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::LazyLock;

use strum::IntoEnumIterator;

use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::models::Card;

/// Reserved index meaning "padding / none / hidden" in all three ID spaces.
pub const PAD_INDEX: u32 = 0;

/// A card's name reduced to the *named Pokémon* it denotes: `Mega Charizard ex` → `Charizard`.
/// Regional forms keep their qualifier (`Alolan Vulpix` is a distinct species with its own line).
pub fn species_key(name: &str) -> &str {
    let name = name.trim();
    let name = name.strip_prefix("Mega ").unwrap_or(name);
    let without_ex = name
        .len()
        .checked_sub(3)
        .filter(|_| name.to_ascii_lowercase().ends_with(" ex"))
        .map(|cut| &name[..cut])
        .unwrap_or(name);
    without_ex.trim()
}

/// Expansions that are pure reprint sets: every card in them re-prints an existing card
/// unchanged. They only ever *lose* the tie-break for which printing represents a group, so a
/// hypothetical original in such a set still gets its own index — this is a preference, not a
/// filter. Verified against the pool by `a4b_is_entirely_reprints`.
const REPRINT_SET_PREFIXES: [&str; 1] = ["A4b"];

/// Everything that defines a card's printed behaviour. Two cards with the same fingerprint are
/// mechanically indistinguishable and therefore share a `card_id` — rarity, art and booster pack
/// are deliberately excluded, being economic rather than mechanical.
///
/// This is the exact set of fields the static descriptors read, which
/// `every_reprint_shares_its_originals_descriptor` checks: if a field the descriptor uses ever
/// escaped this fingerprint, two cards sharing an index would disagree and that test would fail.
fn printing_fingerprint(card: &Card) -> String {
    match card {
        Card::Pokemon(pokemon) => format!(
            "P|{}|{}|{:?}|{}|{:?}|{:?}|{:?}|{:?}|{:?}",
            pokemon.name,
            pokemon.stage,
            pokemon.evolves_from,
            pokemon.hp,
            pokemon.energy_type,
            pokemon.ability,
            pokemon.attacks,
            pokemon.weakness,
            pokemon.retreat_cost,
        ),
        Card::Trainer(trainer) => format!(
            "T|{}|{:?}|{}",
            trainer.name, trainer.trainer_card_type, trainer.effect
        ),
    }
}

/// The `A1` of `"A1 042"`.
fn expansion_prefix(printed_id: &str) -> &str {
    printed_id.split(' ').next().unwrap_or(printed_id)
}

struct IdTables {
    card: HashMap<CardId, u32>,
    /// Every printing → the printing that represents its `card_id`.
    canonical: HashMap<CardId, CardId>,
    /// Canonical printings only, in index order.
    cards_by_index: Vec<CardId>,
    species: HashMap<CardId, u32>,
    line: HashMap<CardId, u32>,
    num_species: usize,
    num_lines: usize,
    /// Species key → `(species_id, line_id)`, restricted to keys backed by a printed *Pokémon*.
    /// Used to resolve the Pokémon a trainer card names in its effect text.
    pokemon_species_lookup: HashMap<String, (u32, u32)>,
    /// Longest species key, in whitespace-separated words ("Alolan Vulpix" → 2).
    max_species_key_words: usize,
}

static ID_TABLES: LazyLock<IdTables> = LazyLock::new(build_id_tables);

fn build_id_tables() -> IdTables {
    // 1. card_id: one index per *distinct printing*, complete reprints collapsed onto the
    //    printing that represents them. Indices are shifted by one for PAD.
    let ordinal_of: HashMap<CardId, usize> =
        CardId::iter().enumerate().map(|(o, c)| (c, o)).collect();

    let mut printings: BTreeMap<String, Vec<CardId>> = BTreeMap::new();
    for card_id in CardId::iter() {
        printings
            .entry(printing_fingerprint(&get_card_by_enum(card_id)))
            .or_default()
            .push(card_id);
    }

    // Representative = the earliest printing outside a reprint set, falling back to plain
    // enumeration order. Enumeration order is release order for main sets (promos sit last).
    let mut representatives: Vec<CardId> = printings
        .values()
        .map(|group| {
            *group
                .iter()
                .min_by_key(|card_id| {
                    let printed_id = get_card_by_enum(**card_id).get_id();
                    let from_reprint_set =
                        REPRINT_SET_PREFIXES.contains(&expansion_prefix(&printed_id));
                    (from_reprint_set, ordinal_of[card_id])
                })
                .expect("a fingerprint group is never empty")
        })
        .collect();
    representatives.sort_unstable_by_key(|card_id| ordinal_of[card_id]);

    let mut card = HashMap::new();
    let mut canonical = HashMap::new();
    let mut cards_by_index = Vec::with_capacity(representatives.len());
    let index_of_representative: HashMap<CardId, u32> = representatives
        .iter()
        .enumerate()
        .map(|(ordinal, card_id)| {
            cards_by_index.push(*card_id);
            (*card_id, ordinal as u32 + 1)
        })
        .collect();
    for group in printings.values() {
        let representative = *group
            .iter()
            .find(|card_id| index_of_representative.contains_key(card_id))
            .expect("every group has its representative");
        let index = index_of_representative[&representative];
        for card_id in group {
            card.insert(*card_id, index);
            canonical.insert(*card_id, representative);
        }
    }

    // 2. species: one id per distinct species key, assigned in sorted key order.
    let mut key_of_card: HashMap<CardId, String> = HashMap::new();
    let mut evolution_edges: Vec<(String, String)> = Vec::new();
    for card_id in CardId::iter() {
        let definition = get_card_by_enum(card_id);
        let key = species_key(&definition.get_name()).to_string();
        if let Card::Pokemon(pokemon) = &definition {
            if let Some(evolves_from) = &pokemon.evolves_from {
                evolution_edges.push((key.clone(), species_key(evolves_from).to_string()));
            }
        }
        key_of_card.insert(card_id, key);
    }

    let species_keys: BTreeSet<&str> = key_of_card.values().map(String::as_str).collect();
    let species_id_of_key: BTreeMap<&str, u32> = species_keys
        .iter()
        .enumerate()
        .map(|(ordinal, key)| (*key, ordinal as u32 + 1))
        .collect();
    let num_species = species_keys.len();

    // 3. line: connected components of the `evolves_from` graph over species keys.
    let mut parent: Vec<usize> = (0..num_species).collect();
    for (child, ancestor) in &evolution_edges {
        // An `evolves_from` may name a Pokémon that has no printed card of its own; such an edge
        // has nothing to merge and is skipped.
        let (Some(&child_id), Some(&ancestor_id)) = (
            species_id_of_key.get(child.as_str()),
            species_id_of_key.get(ancestor.as_str()),
        ) else {
            continue;
        };
        union(&mut parent, child_id as usize - 1, ancestor_id as usize - 1);
    }

    // Name each component by its smallest member (sorted-key order) so line ids are stable.
    let mut representative_of_component: BTreeMap<usize, usize> = BTreeMap::new();
    for member in 0..num_species {
        let root = find(&mut parent, member);
        representative_of_component
            .entry(root)
            .and_modify(|current| *current = (*current).min(member))
            .or_insert(member);
    }
    let mut line_id_of_root: HashMap<usize, u32> = HashMap::new();
    let mut roots: Vec<(usize, usize)> = representative_of_component
        .iter()
        .map(|(root, representative)| (*representative, *root))
        .collect();
    roots.sort_unstable();
    for (ordinal, (_, root)) in roots.iter().enumerate() {
        line_id_of_root.insert(*root, ordinal as u32 + 1);
    }
    let num_lines = line_id_of_root.len();

    let mut species = HashMap::new();
    let mut line = HashMap::new();
    let mut pokemon_species_lookup = HashMap::new();
    let mut max_species_key_words = 1;
    for (card_id, key) in &key_of_card {
        let species_id = species_id_of_key[key.as_str()];
        species.insert(*card_id, species_id);
        let root = find(&mut parent, species_id as usize - 1);
        let line_id = line_id_of_root[&root];
        line.insert(*card_id, line_id);

        if matches!(get_card_by_enum(*card_id), Card::Pokemon(_)) {
            max_species_key_words = max_species_key_words.max(key.split_whitespace().count());
            pokemon_species_lookup.insert(key.clone(), (species_id, line_id));
        }
    }

    IdTables {
        card,
        canonical,
        cards_by_index,
        species,
        line,
        num_species,
        num_lines,
        pokemon_species_lookup,
        max_species_key_words,
    }
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (root_a, root_b) = (find(parent, a), find(parent, b));
    if root_a != root_b {
        // Always hang the larger root under the smaller one: the merge order stops mattering.
        let (keep, merged) = (root_a.min(root_b), root_a.max(root_b));
        parent[merged] = keep;
    }
}

/// Index of the printed card in the `card_id` embedding table. **Complete reprints share one
/// index** (§1.2.2: "do not distinguish complete reprints"), so this is many-to-one: every
/// printing of a mechanically identical card resolves to [`canonical_card`]'s row.
pub fn card_index(card_id: CardId) -> u32 {
    ID_TABLES.card[&card_id]
}

/// The printing that represents `card_id`'s `card_id` row — itself for an original, the original
/// for a reprint. Identity on cards with no twin.
pub fn canonical_card(card_id: CardId) -> CardId {
    ID_TABLES.canonical[&card_id]
}

/// Whether this printing is the one that owns its embedding row.
pub fn is_canonical_printing(card_id: CardId) -> bool {
    canonical_card(card_id) == card_id
}

/// Index of the named Pokémon (all printings collapsed) in the `species_id` table.
pub fn species_index(card_id: CardId) -> u32 {
    ID_TABLES.species[&card_id]
}

/// Index of the whole evolution lineage in the `line_id` table.
pub fn line_index(card_id: CardId) -> u32 {
    ID_TABLES.line[&card_id]
}

/// The **canonical** card an index refers to; `None` for [`PAD_INDEX`] or an out-of-range index.
/// Not a strict inverse of [`card_index`]: a reprint resolves to its original.
pub fn card_at_index(index: u32) -> Option<CardId> {
    if index == PAD_INDEX {
        return None;
    }
    ID_TABLES.cards_by_index.get(index as usize - 1).copied()
}

/// Size of the `card_id` embedding table, PAD row included — distinct printings, not printings.
pub fn card_table_size() -> usize {
    ID_TABLES.cards_by_index.len() + 1
}

/// Every canonical printing, in index order.
pub fn canonical_cards() -> &'static [CardId] {
    &ID_TABLES.cards_by_index
}

/// Size of the `species_id` embedding table, PAD row included.
pub fn species_table_size() -> usize {
    ID_TABLES.num_species + 1
}

/// Size of the `line_id` embedding table, PAD row included.
pub fn line_table_size() -> usize {
    ID_TABLES.num_lines + 1
}

/// `(species_id, line_id)` for a species key that a printed Pokémon backs — the resolution step
/// behind a trainer card's "targets Ninetales, Rapidash, or Magmar" index set.
pub fn ids_for_species_key(key: &str) -> Option<(u32, u32)> {
    ID_TABLES.pokemon_species_lookup.get(key).copied()
}

/// Longest species key in words; the window size a name scanner needs.
pub fn max_species_key_words() -> usize {
    ID_TABLES.max_species_key_words
}

/// Convenience: the three indices of a card, in Pokémon-token order.
pub fn identity_indices(card_id: CardId) -> (u32, u32, u32) {
    (
        card_index(card_id),
        species_index(card_id),
        line_index(card_id),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_index_is_reserved_in_every_space() {
        assert_eq!(card_at_index(PAD_INDEX), None);
        for card_id in CardId::iter() {
            assert_ne!(card_index(card_id), PAD_INDEX);
            assert_ne!(species_index(card_id), PAD_INDEX);
            assert_ne!(line_index(card_id), PAD_INDEX);
        }
    }

    /// An index resolves to the canonical printing, and every printing of a card lands on it.
    #[test]
    fn card_index_round_trips_through_the_canonical_printing() {
        for card_id in CardId::iter() {
            let canonical = canonical_card(card_id);
            assert_eq!(card_at_index(card_index(card_id)), Some(canonical));
            assert_eq!(card_index(canonical), card_index(card_id));
            assert!(is_canonical_printing(canonical));
        }
        for canonical in canonical_cards() {
            assert!(is_canonical_printing(*canonical));
        }
    }

    /// Complete reprints share a row; mechanically different cards never do.
    #[test]
    fn reprints_collapse_and_distinct_cards_do_not() {
        for card_id in CardId::iter() {
            let canonical = canonical_card(card_id);
            assert_eq!(
                printing_fingerprint(&get_card_by_enum(card_id)),
                printing_fingerprint(&get_card_by_enum(canonical)),
                "{card_id:?} shares a row with a mechanically different card"
            );
        }
        // Two Charizards that differ (plain vs ex) keep separate rows.
        assert_ne!(
            card_index(CardId::A1035Charizard),
            card_index(CardId::A1036CharizardEx)
        );
        assert!(
            card_table_size() < CardId::iter().count(),
            "the table shrank"
        );
    }

    /// A4b is a pure reprint set: no card in it is an original, so none of them owns a row.
    #[test]
    fn a4b_is_entirely_reprints() {
        let mut seen = 0;
        for card_id in CardId::iter() {
            let card = get_card_by_enum(card_id);
            let printed_id = card.get_id();
            if expansion_prefix(&printed_id) != "A4b" {
                continue;
            }
            seen += 1;
            let canonical = canonical_card(card_id);
            assert_ne!(
                canonical,
                card_id,
                "{printed_id} ({}) is treated as an original",
                card.get_name()
            );
            let canonical_printed = get_card_by_enum(canonical).get_id();
            assert_ne!(
                expansion_prefix(&canonical_printed),
                "A4b",
                "{printed_id} points at another A4b printing"
            );
        }
        assert_eq!(seen, 379, "the A4b set is the size the pool says it is");
    }

    #[test]
    fn species_collapses_printings_ex_and_mega() {
        // Every Charizard printing — plain, ex, Mega ex — is one species.
        let plain = species_index(CardId::A1035Charizard);
        assert_eq!(species_index(CardId::A1036CharizardEx), plain);
        assert_ne!(species_index(CardId::A1034Charmeleon), plain);
    }

    #[test]
    fn line_groups_the_whole_evolution_chain() {
        let line = line_index(CardId::A1033Charmander);
        assert_eq!(line_index(CardId::A1034Charmeleon), line);
        assert_eq!(line_index(CardId::A1035Charizard), line);
        assert_eq!(line_index(CardId::A1036CharizardEx), line);
        assert_ne!(line_index(CardId::A1001Bulbasaur), line);
    }

    #[test]
    fn regional_forms_are_their_own_species_and_line() {
        let vulpix = line_index(CardId::A1037Vulpix);
        let alolan = line_index(CardId::PB032AlolanVulpix);
        assert_ne!(vulpix, alolan);
    }

    #[test]
    fn tables_are_smaller_the_coarser_the_grain() {
        assert!(line_table_size() < species_table_size());
        assert!(species_table_size() < card_table_size());
    }

    #[test]
    fn species_key_strips_only_the_printing_qualifiers() {
        assert_eq!(species_key("Mega Charizard ex"), "Charizard");
        assert_eq!(species_key("Pikachu ex"), "Pikachu");
        assert_eq!(species_key("Alolan Vulpix"), "Alolan Vulpix");
        assert_eq!(species_key("Exeggutor"), "Exeggutor");
    }
}
