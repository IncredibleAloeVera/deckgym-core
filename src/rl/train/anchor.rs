//! The heuristic seed — `RL_ARCHITECTURE.md` §1.1.3, "**heuristic anchor** as the initial magnet
//! (support + progress indicator)", and §1.5.1's "seeded from the heuristic anchor".
//!
//! Without it the magnet's first thousand batches are a clone of an untrained best-response, so
//! `η·KL(π_BR ‖ magnet)` pulls the policy toward the uniform-over-legal it started at — a proximal
//! step toward nothing. Filling the reservoir with a repo heuristic's play instead gives the term a
//! target that is *already* a support: a policy that develops a bench, attaches energy and attacks,
//! which is what "support" means for a magnet.
//!
//! **The seed is a weighted mixture of heuristics, not one of them.** The magnet is an *average*
//! policy, and a search player's behavioral clone is a near-pure strategy: `KL(π_BR ‖ magnet)` reads
//! `log magnet` floored at `1e-9` ≈ −20.7 nats, so a sharp target charges the best-response almost
//! that much per unit of probability it puts anywhere else — a pull toward a single action, at batch
//! 0, against the entropy bonus. Consulting `π_seed = Σ wᵢ·πᵢ` per decision makes the seed one
//! well-defined stochastic policy whose clone is that mixture, so a strong-but-sharp anchor
//! contributes its judgement without its determinism setting the target's entropy.
//!
//! **This runs the anchors at the learner's seat, and records what they chose.** The env yields that
//! seat's frames rather than resolving them ([`crate::rl::env`]), so the anchor is consulted from
//! outside with the state and enumeration the engine would have handed it, and its answer is mapped
//! back to a mask bit before being submitted. Three things follow, all of them deliberate:
//!
//! - **The observation and the mask are the env's own**, not a re-derivation. A seed built from a
//!   second projection of the same frame would be training the magnet on inputs the run never
//!   produces.
//! - **An unmappable answer is skipped, not repaired.** §1.3.7's round-trip is a bijection up to
//!   `canonical_action`, so an anchor choice always has a bit — but the seed counts the ones it
//!   fails to place ([`AnchorStats::unmatched`]) rather than trusting that, because a silent
//!   fallback would clone the *fallback* under the anchor's name. The game still advances on a
//!   legal bit; only the label is dropped.
//! - **It stops at the fill, not at a game count.** The buffer's capacity is the whole budget, so a
//!   seed is over when the reservoir is full — see [`super::reservoir::Reservoir::seed`] for why it
//!   must not go on offering past it.

use crate::players::{create_players, Player, PlayerCode};
use crate::rl::env::{env_rng, split_seed, AgentId, Env, SeatPolicy, SubmitFault, VecEnv};
use crate::rl::recover::catch;

use super::reservoir::{Reservoir, Sample};
use super::rollout::LEARNER_SEAT;
use super::sampler::DeckSampler;

use rand::rngs::StdRng;
use rand::Rng;

/// Stream tags. The seed runs before the loop and draws its own decks, so it must not advance the
/// rollout's generators — a run whose magnet was seeded would otherwise play different games from
/// one whose magnet was not.
const STREAM_ANCHOR_DRAW: u64 = 0x414E_4348_0000_0001;
const STREAM_ANCHOR_GAME: u64 = 0x414E_4348_0000_0002;
const STREAM_ANCHOR_ACTION: u64 = 0x414E_4348_0000_0003;
const STREAM_ANCHOR_MIX: u64 = 0x414E_4348_0000_0004;

/// One component of the mixture: a repo heuristic and its share of the decisions.
#[derive(Debug, Clone)]
pub struct AnchorShare {
    pub player: PlayerCode,
    /// Relative, not required to sum to 1 — normalized at construction, so editing one line of the
    /// `.toml` does not silently rescale the others.
    pub share: f64,
}

#[derive(Debug, Clone)]
pub struct AnchorConfig {
    /// The mixture, in the order the `.toml` wrote it — which is also [`AnchorStats::per_anchor`]'s
    /// order, so the realized shares can be read against the asked-for ones.
    pub anchors: Vec<AnchorShare>,
    /// Games in flight. Nothing is batched here — the anchors are CPU players — so this only trades
    /// memory against how evenly the fill samples game phases.
    pub envs: usize,
    /// SL steps run against the seeded buffer before the training loop starts.
    pub steps: usize,
    /// Engine panics tolerated across the whole seed.
    pub max_crashes: usize,
}

/// What a seed did. Reported rather than assumed: `unmatched` above zero is a projection bug
/// (§1.3.7), and `frames < capacity` means the seed gave up before filling the buffer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorStats {
    pub games: usize,
    pub frames: usize,
    pub unmatched: usize,
    pub crashes: usize,
    /// Frames each anchor contributed, in [`AnchorConfig::anchors`] order. The realized mix rather
    /// than the configured one: a component whose share is a rounding error in practice is a
    /// component that is not in the magnet, whatever the file says.
    pub per_anchor: Vec<usize>,
}

pub struct AnchorSeed {
    /// Cloned off the training sampler, for [`super::eval::Evaluator`]'s reason: the magnet is
    /// seeded on the deck distribution the run will actually train over.
    sampler: DeckSampler,
    config: AnchorConfig,
    /// Normalized cumulative shares, so picking a component is one roll and one scan.
    cumulative: Vec<f64>,
    seed: u64,
}

impl AnchorSeed {
    pub fn new(sampler: DeckSampler, config: AnchorConfig, seed: u64) -> Result<Self, String> {
        if config.envs == 0 {
            return Err("a magnet seed needs at least one env".to_string());
        }
        if config.anchors.is_empty() {
            return Err("a magnet seed needs at least one anchor".to_string());
        }
        for anchor in &config.anchors {
            if !(anchor.share.is_finite() && anchor.share > 0.0) {
                return Err(format!(
                    "anchor {} has share {}, which must be finite and above zero",
                    anchor.player, anchor.share
                ));
            }
        }

        let total: f64 = config.anchors.iter().map(|anchor| anchor.share).sum();
        let mut running = 0.0;
        let cumulative = config
            .anchors
            .iter()
            .map(|anchor| {
                running += anchor.share / total;
                running
            })
            .collect();

        Ok(AnchorSeed {
            sampler,
            config,
            cumulative,
            seed,
        })
    }

    /// The component that answers one decision.
    fn pick(&self, rng: &mut StdRng) -> usize {
        let roll: f64 = rng.gen();
        self.cumulative
            .iter()
            .position(|bound| roll < *bound)
            // The last bound is 1.0 up to floating-point error, so a roll that lands past it is a
            // rounding artifact rather than a case: it belongs to the final component.
            .unwrap_or(self.cumulative.len() - 1)
    }

    pub fn config(&self) -> &AnchorConfig {
        &self.config
    }

    /// Play anchor games until the reservoir is full.
    pub fn fill(&self, reservoir: &mut Reservoir<Sample>) -> Result<AnchorStats, String> {
        let mut stats = AnchorStats {
            per_anchor: vec![0; self.config.anchors.len()],
            ..AnchorStats::default()
        };
        let mut action_rng = env_rng(self.seed, STREAM_ANCHOR_ACTION);
        let mut mix_rng = env_rng(self.seed, STREAM_ANCHOR_MIX);
        let mut dealt = 0u64;

        let parallel = self.config.envs;
        let mut drivers = Vec::with_capacity(parallel);
        let mut envs = Vec::with_capacity(parallel);
        for _ in 0..parallel {
            let (env, driver) = self.spawn(dealt)?;
            dealt += 1;
            envs.push(env);
            drivers.push(driver);
        }
        let mut vec_env = VecEnv::new(envs);

        while !reservoir.is_full() {
            let (pending, finished, crashed) = vec_env.poll();

            for fault in crashed {
                self.charge(&mut stats, &fault.panic.to_string())?;
                self.respawn(&mut vec_env, &mut drivers, fault.env, &mut dealt)?;
            }
            for done in finished {
                stats.games += 1;
                self.respawn(&mut vec_env, &mut drivers, done.env, &mut dealt)?;
            }
            if pending.is_empty() {
                continue;
            }

            for slot in pending {
                if reservoir.is_full() {
                    break;
                }
                let env = vec_env
                    .get(slot.env)
                    .ok_or_else(|| format!("env {} vanished mid-seed", slot.env))?;

                // The enumeration the mask was projected from, re-derived rather than borrowed:
                // `generate_possible_actions` is a pure function of the state, and the env keeps
                // its copy private precisely so nothing outside can answer a frame with a stale one.
                let state = env.state();
                let (_, actions) = state.generate_possible_actions();
                // Per decision, not per game: a component drawn once per game would give a buffer
                // of whole trajectories each cloned from a single heuristic, where the target is
                // meant to be the mixture policy itself.
                let component = self.pick(&mut mix_rng);
                let chosen = match catch(|| {
                    drivers[slot.env][component].decision_fn(&mut action_rng, state, &actions)
                }) {
                    Ok(action) => action,
                    Err(panic) => {
                        // The anchor searching a state the engine cannot describe is the same event
                        // as the engine failing on it, and costs the same: this game.
                        self.charge(&mut stats, &panic.to_string())?;
                        self.respawn(&mut vec_env, &mut drivers, slot.env, &mut dealt)?;
                        continue;
                    }
                };

                let mask = slot.request.mask;
                let entry = mask.entries.iter().find(|entry| {
                    entry.action == chosen.action && entry.is_stack == chosen.is_stack
                });
                let (head, index) = match entry {
                    Some(entry) => {
                        reservoir.seed(Sample {
                            observation: slot.request.observation,
                            mask: mask.clone(),
                            chosen_bit: entry.head.offset() + entry.index,
                        });
                        stats.frames += 1;
                        stats.per_anchor[component] += 1;
                        (entry.head, entry.index)
                    }
                    None => {
                        // Legal, so the game goes on; unrecorded, so the magnet never sees it.
                        stats.unmatched += 1;
                        let fallback = mask.entries.first().expect("a decision frame has bits");
                        (fallback.head, fallback.index)
                    }
                };

                match vec_env.submit(slot.env, head, index) {
                    Ok(()) => {}
                    Err(SubmitFault::Panicked(panic)) => {
                        self.charge(&mut stats, &panic.to_string())?;
                        self.respawn(&mut vec_env, &mut drivers, slot.env, &mut dealt)?;
                    }
                    // §1.3.7 invariant 3: a bit the mask set must resolve. Fatal here as everywhere.
                    Err(SubmitFault::Rejected(err)) => {
                        return Err(format!(
                            "env {} rejected {head:?}[{index}]: {err:?}",
                            slot.env
                        ))
                    }
                }
            }
        }

        Ok(stats)
    }

    fn charge(&self, stats: &mut AnchorStats, panic: &str) -> Result<(), String> {
        stats.crashes += 1;
        if stats.crashes > self.config.max_crashes {
            return Err(format!(
                "magnet seed gave up after {} engine panics: {panic}",
                self.config.max_crashes
            ));
        }
        Ok(())
    }

    fn respawn(
        &self,
        vec_env: &mut VecEnv<'static>,
        drivers: &mut [Vec<Box<dyn Player>>],
        slot: usize,
        dealt: &mut u64,
    ) -> Result<(), String> {
        let (env, driver) = self.spawn(*dealt)?;
        *dealt += 1;
        vec_env.replace(slot, env);
        drivers[slot] = driver;
        Ok(())
    }

    /// Game `index` of the seed, and one driver per mixture component for its learner seat.
    ///
    /// The drivers are *second* instances on the same deck rather than the env's own seat-0 player:
    /// `Game` owns its players, and the seat is an [`SeatPolicy::Agent`] one precisely so its frames
    /// come out instead of being resolved inside. What the engine would have consulted and what this
    /// consults are the same code on the same deck.
    ///
    /// All of them are built up front, per game: a `Player` owns its deck, and constructing one at
    /// the decision that draws it would put deck-building inside the per-frame path.
    fn spawn(&self, index: u64) -> Result<(Env<'static>, Vec<Box<dyn Player>>), String> {
        let mut draw_rng = env_rng(self.seed, split_seed(STREAM_ANCHOR_DRAW, index));
        let [first, second] = self.sampler.sample(&mut draw_rng)?;

        let drivers = self
            .config
            .anchors
            .iter()
            .map(|anchor| {
                create_players(
                    first.deck.clone(),
                    first.deck.clone(),
                    vec![anchor.player.clone(); 2],
                )
                .remove(LEARNER_SEAT)
            })
            .collect();

        // The opponent seat is the mixture's first component. It only has to be a plausible
        // opponent — the seed records the learner seat alone, so this shapes which states are
        // visited, not what is cloned.
        let opponent = self.config.anchors[0].player.clone();
        let mut codes = vec![opponent; 2];
        codes[LEARNER_SEAT] = PlayerCode::ET;
        let mut seats = [SeatPolicy::Scripted, SeatPolicy::Scripted];
        seats[LEARNER_SEAT] = SeatPolicy::Agent(AgentId::LEARNER);

        let players = create_players(first.deck, second.deck, codes);
        let env = Env::from_players(
            players,
            seats,
            split_seed(self.seed, split_seed(STREAM_ANCHOR_GAME, index)),
        );
        Ok((env, drivers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::train::deck_db::DeckDb;
    use crate::rl::train::sampler::SamplerConfig;
    use std::path::Path;

    fn sampler() -> DeckSampler {
        let db = DeckDb::load(Path::new("decks/tutorial")).expect("tutorial db");
        DeckSampler::new(
            db,
            SamplerConfig {
                pure_mirror: 0.05,
                mirror: 0.10,
                archetypes: vec!["beginner".to_string()],
            },
        )
        .expect("sampler")
    }

    fn mixture(anchors: &[(PlayerCode, f64)]) -> AnchorSeed {
        AnchorSeed::new(
            sampler(),
            AnchorConfig {
                anchors: anchors
                    .iter()
                    .map(|(player, share)| AnchorShare {
                        player: player.clone(),
                        share: *share,
                    })
                    .collect(),
                envs: 4,
                steps: 0,
                max_crashes: 8,
            },
            5,
        )
        .expect("seed")
    }

    fn seed(anchor: PlayerCode) -> AnchorSeed {
        mixture(&[(anchor, 1.0)])
    }

    /// The load-bearing property: the anchor's own choice is what lands in the buffer. If the
    /// inverse projection failed, the frames would still arrive — carrying the fallback bit — and
    /// the magnet would clone "always the first legal action", which is a policy no anchor plays.
    #[test]
    fn it_fills_the_reservoir_with_the_anchors_own_choices() {
        let mut reservoir = Reservoir::new(300);
        let stats = seed(PlayerCode::W).fill(&mut reservoir).expect("seed");

        assert!(reservoir.is_full());
        assert_eq!(reservoir.len(), 300);
        assert_eq!(stats.frames, 300);
        assert_eq!(
            stats.unmatched, 0,
            "the §1.3.7 round-trip failed on {} frames",
            stats.unmatched
        );
        assert_eq!(stats.crashes, 0);
        assert_eq!(stats.per_anchor, vec![300]);
    }

    /// Every component answers, in something near its share. The magnet's whole reason for taking a
    /// mixture is that no single anchor's determinism sets the target's entropy — a component that
    /// silently never gets consulted, or one that answers everything, is that property gone.
    #[test]
    fn the_mixture_consults_every_component_in_proportion() {
        let mut reservoir = Reservoir::new(1200);
        let stats = mixture(&[(PlayerCode::W, 3.0), (PlayerCode::R, 1.0)])
            .fill(&mut reservoir)
            .expect("seed");

        assert_eq!(stats.frames, 1200);
        assert_eq!(stats.per_anchor.iter().sum::<usize>(), 1200);
        // Shares are relative, so 3:1 is 0.75/0.25 whatever the numbers were written as. The
        // binomial SE over 1200 frames is ≈ 1.25 %, so 5 % is a four-sigma band.
        let realized = stats.per_anchor[0] as f64 / stats.frames as f64;
        assert!(
            (realized - 0.75).abs() < 0.05,
            "the 3:1 mixture realized at {realized:.3}: {:?}",
            stats.per_anchor
        );
    }

    /// A share that cannot be normalized is a config error, not a component silently dropped: a
    /// zero-share anchor in the file reads as "this heuristic is in the seed" and would not be.
    #[test]
    fn a_mixture_needs_at_least_one_component_with_a_usable_share() {
        let build = |anchors: Vec<AnchorShare>| {
            AnchorSeed::new(
                sampler(),
                AnchorConfig {
                    anchors,
                    envs: 4,
                    steps: 0,
                    max_crashes: 8,
                },
                5,
            )
            .err()
        };

        assert!(build(vec![]).is_some());
        assert!(build(vec![AnchorShare {
            player: PlayerCode::W,
            share: 0.0,
        }])
        .is_some());
        assert!(build(vec![AnchorShare {
            player: PlayerCode::W,
            share: f64::NAN,
        }])
        .is_some());
    }

    /// A seeded sample is a decision point the model can be forwarded on: the recorded bit is one
    /// the mask set, and the observation is the mask's own actor's.
    #[test]
    fn every_seeded_sample_is_a_legal_decision_point() {
        let mut reservoir = Reservoir::new(120);
        seed(PlayerCode::R).fill(&mut reservoir).expect("seed");

        let mut rng = env_rng(1, 0);
        for sample in reservoir.draw(120, &mut rng) {
            assert_eq!(sample.observation.perspective, sample.mask.actor);
            assert!(
                sample
                    .mask
                    .entries
                    .iter()
                    .any(|entry| entry.head.offset() + entry.index == sample.chosen_bit),
                "bit {} is not in the mask",
                sample.chosen_bit
            );
        }
    }

    /// Reproducibility (§1.5.5) reaches the seed too: it is part of the run's initial conditions,
    /// so two runs of one config must start their magnet from the same buffer.
    #[test]
    fn the_same_seed_fills_the_same_buffer() {
        let run = || {
            let mut reservoir = Reservoir::new(80);
            let stats = mixture(&[(PlayerCode::W, 3.0), (PlayerCode::R, 1.0)])
                .fill(&mut reservoir)
                .expect("seed");
            let mut rng = env_rng(2, 0);
            let bits: Vec<usize> = reservoir
                .draw(80, &mut rng)
                .iter()
                .map(|sample| sample.chosen_bit)
                .collect();
            (stats, bits)
        };
        assert_eq!(run(), run());
    }
}
