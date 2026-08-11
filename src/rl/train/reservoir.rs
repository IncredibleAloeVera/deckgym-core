//! The magnet's buffer — `RL_ARCHITECTURE.md` §1.5.3, "persistent **reservoir** (magnet clone)".
//!
//! §1.5.1's magnet approximates the NFSP **time-average** policy, and that word is the whole
//! specification of this file: the buffer it clones from has to be a uniform sample of *every*
//! decision the best-response has ever made, not of the recent ones. A ring buffer would make the
//! magnet a lagged copy of the current BR — the same policy a few thousand frames late — and the
//! KL term would then pull the BR toward itself, which is not a proximal step toward anything.
//!
//! So: **Vitter's algorithm R**. The first `capacity` samples are kept; sample `n` (0-based)
//! thereafter replaces a uniformly drawn resident with probability `capacity / (n + 1)`. The
//! invariant is that after any number of offers the buffer is a uniform sample without replacement
//! of the whole stream, which is exactly "the average of every policy the BR has been".
//!
//! Two consequences worth stating, because they are the ones a run notices:
//!
//! - **The buffer is checkpointed on the way out, not on the autosave cadence.** It used to be
//!   dropped entirely, on the argument that the magnet's *weights* are where the average policy
//!   lives and the frames are a few hundred MB — measured wrong, on `long_v3`/`long_v4`: NOTES.md,
//!   "Le magnet : ce qui n'est pas mesuré". A dropped buffer resets `seen` to 0, which makes it a
//!   uniform sample of the stream *since the restart*, so the SL step starts chasing the current
//!   best-response — the lagged copy this file rejects a ring buffer for being.
//!
//!   So [`Reservoir::encode`] rides in §1.5.5's hot checkpoint when the loop is stopping or pausing
//!   — the two exits the user controls — and not on the rolling autosave, whose cadence the write
//!   would dominate. A hard crash still resumes from an empty buffer, and `magnet/reservoir_seen`
//!   is what says which of the two happened.
//! - **Seeding is not offering.** [`Reservoir::seed`] fills the buffer and stops, leaving `seen` at
//!   the fill. Pushing an anchor's frames through [`Reservoir::offer`] instead would leave `seen`
//!   at however many frames the anchor played, and a `seen` far past `capacity` makes the *next*
//!   million BR frames enter at rate `capacity / seen` — the seed would freeze the magnet on the
//!   heuristic rather than starting it there.

use crate::rl::action_mask::ActionMask;
use crate::rl::observation::Observation;

use rand::rngs::StdRng;
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// One behavioral-cloning example: a decision point and the action that was taken at it.
///
/// The same `(observation, mask)` pair the update re-forwards from — the model input is *not*
/// stored, for the reason [`super::rollout::Frame`] gives: the SL step is a gradient step, so the
/// forward is replayed under autodiff whatever is kept, and these are the compact end of it.
#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub observation: Observation,
    pub mask: ActionMask,
    /// Index into the flat `ACTION_MASK_DIM` policy vector — the label of the cloning step.
    pub chosen_bit: usize,
}

/// A uniform sample of a stream, bounded by `capacity`.
///
/// Generic over the item so the sampling law can be tested on values a test can *identify*; the
/// magnet instantiates it at [`Sample`].
pub struct Reservoir<T> {
    capacity: usize,
    items: Vec<T>,
    seen: u64,
}

impl<T> Reservoir<T> {
    pub fn new(capacity: usize) -> Self {
        Reservoir {
            capacity: capacity.max(1),
            items: Vec::new(),
            seen: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Samples offered ever — the denominator of the acceptance rate, and what says how much of the
    /// run's history the buffer is averaging over.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Offer one sample to the stream (algorithm R). Returns whether it was kept.
    pub fn offer(&mut self, item: T, rng: &mut StdRng) -> bool {
        self.offer_with(rng, || item)
    }

    /// [`Reservoir::offer`], with the sample built only if it is kept.
    ///
    /// The acceptance draw does not look at the item, and past the fill it accepts at rate
    /// `capacity / seen` — so by the time a run is a few million frames in, a materializing `offer`
    /// would be deep-copying an [`Observation`] for every frame of every batch to throw ~all of
    /// them away. This is the same law, paid only on the frames that stay.
    pub fn offer_with(&mut self, rng: &mut StdRng, item: impl FnOnce() -> T) -> bool {
        let index = self.seen;
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(item());
            return true;
        }
        // `index` is 0-based, so the (n+1)-th sample is drawn against n+1 slots.
        let draw = rng.gen_range(0..=index);
        if draw < self.capacity as u64 {
            self.items[draw as usize] = item();
            return true;
        }
        false
    }

    /// Fill the buffer without evicting, for the §1.5.1 heuristic seed. Returns whether there was
    /// room — a `false` is the caller's signal to stop playing anchor games.
    ///
    /// See the module docs for why this is not [`Reservoir::offer`].
    pub fn seed(&mut self, item: T) -> bool {
        if self.is_full() {
            return false;
        }
        self.items.push(item);
        self.seen = self.items.len() as u64;
        true
    }

    /// Evicts a uniformly random fraction of residents — a curriculum stage transition's partial
    /// reseed (`RL_ARCHITECTURE.md` §1.5.4): the landscape shifts enough to discard *some* of the
    /// average policy's support, but a full reset would throw away real signal.
    ///
    /// `seen` is left untouched: it counts offers ever, not residents held, and touching it would
    /// distort [`Reservoir::offer_with`]'s future acceptance rate (`capacity / seen`). The freed
    /// slots just make [`Reservoir::is_full`] false, so both [`Reservoir::seed`] and
    /// [`Reservoir::offer_with`]'s fill branch push into them unconditionally until the reservoir
    /// is full again — no other change is needed for a partial reseed to work.
    pub fn evict_fraction(&mut self, fraction: f64, rng: &mut StdRng) -> usize {
        let count = (self.items.len() as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
        let count = count.min(self.items.len());
        for _ in 0..count {
            let index = rng.gen_range(0..self.items.len());
            self.items.swap_remove(index);
        }
        count
    }

    /// `n` distinct residents, uniformly drawn — the magnet's SL minibatch.
    ///
    /// Without replacement: a duplicated frame inside one batch is one frame counted twice by the
    /// mean, which is a silent reweighting of the cloning target rather than a larger batch.
    pub fn draw(&self, n: usize, rng: &mut StdRng) -> Vec<&T> {
        let n = n.min(self.items.len());
        // Fisher-Yates stopped after `n` swaps rather than run to the end: `n` is the SL batch
        // (hundreds) against a capacity of tens of thousands. The index space is still walked once
        // to build it, which is the cheap half — it is the swaps that the prefix saves.
        let mut indices: Vec<usize> = (0..self.items.len()).collect();
        for position in 0..n {
            let pick = rng.gen_range(position..indices.len());
            indices.swap(position, pick);
        }
        indices[..n]
            .iter()
            .map(|index| &self.items[*index])
            .collect()
    }
}

/// The buffer on disk. `seen` is the field the whole exercise is about: restoring residents without
/// it would refill at rate 1 and reset the average the same way an empty buffer does.
#[derive(Serialize, Deserialize)]
struct ReservoirRecord<T> {
    capacity: usize,
    seen: u64,
    items: Vec<T>,
}

impl<T: Serialize + DeserializeOwned> Reservoir<T> {
    /// Encode for §1.5.5's hot checkpoint.
    pub fn encode(&self) -> Result<Vec<u8>, String>
    where
        T: Clone,
    {
        rmp_serde::to_vec(&ReservoirRecord {
            capacity: self.capacity,
            seen: self.seen,
            items: self.items.clone(),
        })
        .map_err(|err| format!("failed to encode the reservoir: {err}"))
    }

    /// Restore a checkpointed buffer, replacing whatever this one holds.
    ///
    /// A `capacity` the run's `.toml` has since shrunk keeps the first `capacity` residents rather
    /// than refusing the record: algorithm R assigns residents to positions uniformly, so positions
    /// are exchangeable and a prefix is itself a uniform sample of the stream. A grown `capacity`
    /// needs nothing — [`Reservoir::offer_with`]'s fill branch tops the free slots up, which is the
    /// same path a curriculum reseed already uses.
    ///
    /// `seen` is taken from the record even when it exceeds what the residents can justify. It is a
    /// count of offers, not of survivors, and lowering it to fit would hand the next batch's frames
    /// an acceptance rate the run never earned.
    pub fn restore(&mut self, encoded: &[u8]) -> Result<(), String> {
        let record: ReservoirRecord<T> = rmp_serde::from_slice(encoded)
            .map_err(|err| format!("failed to decode the reservoir: {err}"))?;
        if record.seen < record.items.len() as u64 {
            return Err(format!(
                "reservoir record holds {} residents against {} offers ever",
                record.items.len(),
                record.seen
            ));
        }
        self.items = record.items;
        self.items.truncate(self.capacity);
        self.seen = record.seen;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::env::env_rng;

    #[test]
    fn it_fills_before_it_evicts() {
        let mut rng = env_rng(1, 0);
        let mut reservoir = Reservoir::new(4);
        for id in 0..4usize {
            assert!(reservoir.offer(id, &mut rng));
        }
        assert!(reservoir.is_full());
        assert_eq!(reservoir.seen(), 4);
        assert_eq!(reservoir.items, vec![0, 1, 2, 3]);
    }

    /// The property the magnet rests on: after a long stream, every sample is equally likely to be
    /// in the buffer. A ring buffer passes every other test in this file and fails this one — the
    /// last `capacity` samples would be present with probability 1 and the rest with probability 0,
    /// which is the difference between an average policy and a lagged copy.
    #[test]
    fn every_sample_of_the_stream_survives_with_equal_probability() {
        const STREAM: usize = 200;
        const CAPACITY: usize = 20;
        const RUNS: usize = 4000;

        let mut counts = vec![0usize; STREAM];
        for run in 0..RUNS {
            let mut rng = env_rng(0x5245_5345_5256, run as u64);
            let mut reservoir = Reservoir::new(CAPACITY);
            for id in 0..STREAM {
                reservoir.offer(id, &mut rng);
            }
            for held in &reservoir.items {
                counts[*held] += 1;
            }
        }

        // Each sample should appear in CAPACITY/STREAM of the runs. The binomial SE at p = 0.1 over
        // 4000 runs is ≈ 0.5 %, so 2 % is a four-sigma band — wide enough not to flake, narrow
        // enough that a recency bias (which puts the tail at p = 1) cannot hide in it.
        let expected = CAPACITY as f64 / STREAM as f64;
        for (id, count) in counts.iter().enumerate() {
            let rate = *count as f64 / RUNS as f64;
            assert!(
                (rate - expected).abs() < 0.02,
                "sample {id} survived at {rate:.3}, expected {expected:.3}"
            );
        }
    }

    /// Seeding leaves `seen` at the fill, so the BR frames that follow enter at the full
    /// `capacity / seen` rate instead of at whatever rate the anchor's frame count implies.
    #[test]
    fn seeding_stops_at_capacity_and_does_not_inflate_seen() {
        let mut reservoir = Reservoir::new(3);
        assert!(reservoir.seed(0));
        assert!(reservoir.seed(1));
        assert!(reservoir.seed(2));
        assert!(!reservoir.seed(3), "a full reservoir refuses a seed");
        assert_eq!(reservoir.seen(), 3);
        assert_eq!(reservoir.len(), 3);
    }

    /// A stage transition (§1.5.4) discards part of the buffer, not all of it — the fraction asked
    /// for is what should go, and `seen` (the acceptance-rate denominator) must not move, or a
    /// partial reseed would silently change how eagerly the reservoir accepts the next BR frames.
    #[test]
    fn evict_fraction_removes_roughly_the_asked_for_share_and_leaves_seen_untouched() {
        let mut rng = env_rng(3, 0);
        let mut reservoir = Reservoir::new(200);
        for id in 0..200usize {
            reservoir.offer(id, &mut rng);
        }
        let seen_before = reservoir.seen();

        let evicted = reservoir.evict_fraction(0.3, &mut rng);

        assert_eq!(evicted, 60, "30% of 200 residents");
        assert_eq!(reservoir.len(), 140);
        assert_eq!(
            reservoir.seen(),
            seen_before,
            "seen counts offers, not residents"
        );
        assert!(!reservoir.is_full());
    }

    /// The whole point of a *partial* reseed: after eviction the reservoir has free capacity again,
    /// and the anchor fill (`Reservoir::seed`) tops it back up rather than being refused outright.
    #[test]
    fn a_partially_evicted_reservoir_refills_at_the_seed_rate() {
        let mut rng = env_rng(4, 0);
        let mut reservoir = Reservoir::new(10);
        for id in 0..10usize {
            assert!(reservoir.seed(id));
        }
        assert!(reservoir.is_full());

        let evicted = reservoir.evict_fraction(0.5, &mut rng);
        assert_eq!(evicted, 5);
        assert!(!reservoir.is_full());

        for id in 100..105usize {
            assert!(reservoir.seed(id), "the freed slots must accept a seed");
        }
        assert!(reservoir.is_full());
        assert!(!reservoir.seed(999), "full again refuses like before");
    }

    /// Fractions outside `[0, 1]` are clamped rather than panicking or under/over-shooting — a
    /// stage's `.toml` typo should not be able to empty or no-op the buffer silently.
    #[test]
    fn evict_fraction_clamps_out_of_range_input() {
        let mut rng = env_rng(5, 0);
        let mut reservoir = Reservoir::new(10);
        for id in 0..10usize {
            reservoir.offer(id, &mut rng);
        }

        assert_eq!(reservoir.evict_fraction(-1.0, &mut rng), 0);
        assert_eq!(reservoir.len(), 10);
        assert_eq!(reservoir.evict_fraction(5.0, &mut rng), 10);
        assert_eq!(reservoir.len(), 0);
    }

    /// The property the checkpoint exists for: a restored buffer accepts the next frames at the
    /// rate the run had reached, not at the rate an empty one would. Without `seen` the resumed
    /// magnet re-averages over the post-resume stream only, which is the restart pathology this
    /// module's docs describe.
    #[test]
    fn a_restored_reservoir_keeps_the_acceptance_rate_it_was_saved_at() {
        let mut rng = env_rng(11, 0);
        let mut saved = Reservoir::new(8);
        for id in 0..1_000usize {
            saved.offer(id, &mut rng);
        }
        assert_eq!(saved.seen(), 1_000);
        let encoded = saved.encode().expect("encode");

        let mut restored = Reservoir::new(8);
        restored.restore(&encoded).expect("restore");
        assert_eq!(restored.seen(), 1_000);
        assert_eq!(restored.items, saved.items);

        // 40 offers against `capacity / seen ≈ 0.008` — a buffer that had reset would take ~all of
        // them through the fill branch instead.
        let mut fresh_rng = env_rng(11, 1);
        let accepted = (0..40)
            .filter(|id| restored.offer(1_000 + id, &mut fresh_rng))
            .count();
        assert!(
            accepted <= 2,
            "{accepted} of 40 offers were kept — the acceptance rate reset"
        );
    }

    /// The generic tests above prove the sampling law; this one proves the payload. [`Sample`] is
    /// an [`Observation`] and an [`ActionMask`], and the mask carries the engine's own
    /// `SimpleAction` per set bit — a decision point that does not survive the round trip byte for
    /// byte would resume the magnet onto frames whose labels no longer mean what they meant.
    #[test]
    fn a_real_decision_point_round_trips() {
        use crate::rl::action_mask::project;
        use crate::rl::observation::get_observation;
        use crate::test_support::init_random_players;
        use crate::Game;

        let game = Game::new(init_random_players(), 9);
        let state = game.get_state_clone();
        let (actor, actions) = state.generate_possible_actions();
        let observation = get_observation(&state, actor, &actions, None, None);
        let mask = project(&state, &actions, &observation);
        let chosen_bit = mask
            .to_wire()
            .bits
            .iter()
            .position(|set| *set)
            .expect("a bit");

        let mut saved = Reservoir::new(2);
        saved.seed(Sample {
            observation: observation.clone(),
            mask: mask.clone(),
            chosen_bit,
        });
        let mut restored = Reservoir::<Sample>::new(2);
        restored
            .restore(&saved.encode().expect("encode"))
            .expect("restore");

        let sample = restored.items.first().expect("one resident");
        assert_eq!(sample.observation, observation);
        assert_eq!(sample.mask, mask);
        assert_eq!(sample.chosen_bit, chosen_bit);
    }

    /// A `capacity` the `.toml` shrank must not make a checkpoint unreadable, and must not inflate
    /// the buffer past the bound the run now asks for.
    #[test]
    fn restoring_into_a_smaller_capacity_truncates_and_keeps_seen() {
        let mut rng = env_rng(12, 0);
        let mut saved = Reservoir::new(100);
        for id in 0..500usize {
            saved.offer(id, &mut rng);
        }
        let encoded = saved.encode().expect("encode");

        let mut restored = Reservoir::<usize>::new(30);
        restored.restore(&encoded).expect("restore");
        assert_eq!(restored.len(), 30);
        assert!(restored.is_full());
        assert_eq!(restored.seen(), 500);
    }

    /// A record claiming more residents than offers is corrupt, not merely surprising: `seen` is the
    /// acceptance denominator, and a buffer that accepted more than it was offered would make every
    /// rate derived from it meaningless.
    #[test]
    fn a_record_with_fewer_offers_than_residents_is_refused() {
        let mut reservoir = Reservoir::new(4);
        for id in 0..4usize {
            reservoir.seed(id);
        }
        let encoded = reservoir.encode().expect("encode");
        let mut record: ReservoirRecord<usize> = rmp_serde::from_slice(&encoded).expect("decode");
        record.seen = 1;
        let corrupt = rmp_serde::to_vec(&record).expect("re-encode");

        assert!(Reservoir::<usize>::new(4).restore(&corrupt).is_err());
    }

    /// A draw is without replacement, and a draw larger than the buffer is the buffer rather than
    /// an error: the SL step runs on what exists.
    #[test]
    fn a_draw_is_distinct_and_clamped_to_what_is_held() {
        let mut rng = env_rng(7, 0);
        let mut reservoir = Reservoir::new(16);
        for id in 0..10usize {
            reservoir.offer(id, &mut rng);
        }

        let mut drawn: Vec<usize> = reservoir.draw(6, &mut rng).into_iter().copied().collect();
        assert_eq!(drawn.len(), 6);
        drawn.sort_unstable();
        drawn.dedup();
        assert_eq!(drawn.len(), 6, "a frame was drawn twice");

        assert_eq!(reservoir.draw(50, &mut rng).len(), 10);
        assert_eq!(Reservoir::<usize>::new(4).draw(2, &mut rng).len(), 0);
    }
}
