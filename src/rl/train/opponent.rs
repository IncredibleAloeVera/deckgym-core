//! Who occupies the opponent seat, and which model answers it — the rollout half of §1.5.2.
//!
//! **Env groups, not per-game draws.** Before the pool, every game drew its own opponent from the
//! scripted panel, which cost nothing because a scripted seat is resolved in-process. A *model*
//! opponent is a batched forward, and drawing per game scatters the envs over every pool member at
//! once: with 8 members over 64 envs the opponent forward fragments into batches of ~4, against
//! §1.5.5's measured 4.61 games/s at batch 8 and 23.03 at batch 64. So the envs are cut into
//! contiguous **groups**, each facing one opponent for the length of a collection, and
//! `concurrent_opponents` is how many distinct opponents are in flight at once.
//!
//! **What that knob really trades.** Not batch width — coverage: the learner's own batch halves as
//! soon as any seat is model-driven, whatever the grouping, and raising `envs` is the fix for that,
//! not fewer groups. Full derivation: NOTES.md, "Étape 4 — le chiffre qui a changé le
//! dimensionnement".
//!
//! **Games in flight keep the opponent they started with.** A new assignment applies to the games
//! spawned after it, never to the ones already running — an episode's reward has to be attributable
//! to the opponent that actually played it, and swapping mid-game would silently mix two matchups
//! into one rated result.

use std::collections::HashMap;

use burn::tensor::backend::Backend;

use crate::players::PlayerCode;
use crate::rl::env::AgentId;
use crate::rl::model::RlModel;

use super::rating::OpponentId;

/// How one env's opponent seat is resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum OpponentSeat {
    /// An engine heuristic, played in-process by `play_tick`. Costs no forward.
    Scripted(PlayerCode),
    /// A frozen model, answered out of [`OpponentModels`] under this id.
    Model(AgentId),
}

/// Which opponent each env faces.
#[derive(Debug, Clone, PartialEq)]
pub enum Assignment {
    /// Pre-§1.5.2 behaviour: every game draws uniformly from the scripted panel. Kept as a variant
    /// rather than a special case because it is what `[rollout] opponents` means in a run with no
    /// pool, and because a scripted panel has no reason to pay the grouping's coarser sampling.
    PerGame(Vec<PlayerCode>),
    /// §1.5.2: contiguous env groups, each facing one opponent until the next assignment.
    Grouped(Vec<(OpponentId, OpponentSeat)>),
}

impl Assignment {
    pub fn grouped(groups: Vec<(OpponentId, OpponentSeat)>) -> Result<Self, String> {
        if groups.is_empty() {
            return Err("an assignment needs at least one env group".to_string());
        }
        Ok(Assignment::Grouped(groups))
    }

    /// Every env against the same opponent — `concurrent_opponents = 1`, the default.
    pub fn uniform(id: OpponentId, seat: OpponentSeat) -> Self {
        Assignment::Grouped(vec![(id, seat)])
    }

    /// Distinct opponents in flight. `1` for [`Assignment::PerGame`] is a lie the type cannot tell:
    /// that variant draws per game, so this reports the panel size instead.
    pub fn concurrent(&self) -> usize {
        match self {
            Assignment::PerGame(panel) => panel.len(),
            Assignment::Grouped(groups) => groups.len(),
        }
    }

    /// The group `env` belongs to, out of `envs` slots.
    ///
    /// Contiguous and computed rather than stored: the mapping has to survive a resume, and a
    /// division is one fewer thing to checkpoint. A group count that does not divide `envs` leaves
    /// the last group larger, which costs a few frames of imbalance and no correctness.
    pub fn group_of(&self, env: usize, envs: usize) -> usize {
        match self {
            Assignment::PerGame(_) => 0,
            Assignment::Grouped(groups) => {
                let envs = envs.max(1);
                (env * groups.len() / envs).min(groups.len() - 1)
            }
        }
    }

    /// Every model-driven opponent in the assignment, so the caller can check it holds their
    /// weights before a game is spawned against one.
    pub fn agents(&self) -> Vec<AgentId> {
        match self {
            Assignment::PerGame(_) => Vec::new(),
            Assignment::Grouped(groups) => groups
                .iter()
                .filter_map(|(_, seat)| match seat {
                    OpponentSeat::Model(agent) => Some(*agent),
                    OpponentSeat::Scripted(_) => None,
                })
                .collect(),
        }
    }
}

/// The frozen models answering opponent seats, indexed by [`AgentId`].
///
/// [`AgentId::LEARNER`] is never in here: the learner is the network being trained and the
/// collector holds it directly. Ids are handed out on insertion and stay valid until [`clear`],
/// which is what a pool refresh calls before re-inserting its new slate.
///
/// [`clear`]: OpponentModels::clear
pub struct OpponentModels<B: Backend> {
    /// `AgentId(1 + i)` is `models[i]`. The offset is what keeps `AgentId(0)` meaning the learner
    /// everywhere, including in a `DecisionRequest` the env built without knowing about any of this.
    models: Vec<(OpponentId, RlModel<B>)>,
    by_id: HashMap<OpponentId, AgentId>,
}

impl<B: Backend> Default for OpponentModels<B> {
    fn default() -> Self {
        OpponentModels {
            models: Vec::new(),
            by_id: HashMap::new(),
        }
    }
}

impl<B: Backend> OpponentModels<B> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a model and returns the id the seats will name it by. Re-registering an id
    /// replaces its weights and keeps the id, so a refresh that re-draws the same checkpoint does
    /// not renumber the assignment under it.
    pub fn insert(&mut self, id: OpponentId, model: RlModel<B>) -> AgentId {
        if let Some(agent) = self.by_id.get(&id) {
            self.models[agent.0 as usize - 1] = (id, model);
            return *agent;
        }
        let agent = AgentId(self.models.len() as u16 + 1);
        self.by_id.insert(id.clone(), agent);
        self.models.push((id, model));
        agent
    }

    pub fn get(&self, agent: AgentId) -> Option<&RlModel<B>> {
        if agent == AgentId::LEARNER {
            return None;
        }
        self.models
            .get(agent.0 as usize - 1)
            .map(|(_, model)| model)
    }

    pub fn id_of(&self, agent: AgentId) -> Option<&OpponentId> {
        if agent == AgentId::LEARNER {
            return None;
        }
        self.models.get(agent.0 as usize - 1).map(|(id, _)| id)
    }

    pub fn agent_of(&self, id: &OpponentId) -> Option<AgentId> {
        self.by_id.get(id).copied()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Drops every model. Ids handed out before this are invalid afterwards, so the assignment has
    /// to be rebuilt in the same breath — which is exactly what a pool refresh does.
    pub fn clear(&mut self) {
        self.models.clear();
        self.by_id.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::model::config::ModelConfig;
    use crate::rl::text_embedding::TextEmbeddings;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    fn model() -> RlModel<B> {
        RlModel::<B>::new(
            &ModelConfig::default(),
            &TextEmbeddings::zeros(),
            &Default::default(),
        )
    }

    #[test]
    fn groups_cover_every_env_contiguously() {
        let assignment = Assignment::grouped(vec![
            (OpponentId::Pool(1), OpponentSeat::Model(AgentId(1))),
            (OpponentId::Pool(2), OpponentSeat::Model(AgentId(2))),
        ])
        .expect("assignment");

        assert_eq!(assignment.group_of(0, 64), 0);
        assert_eq!(assignment.group_of(31, 64), 0);
        assert_eq!(assignment.group_of(32, 64), 1);
        assert_eq!(assignment.group_of(63, 64), 1);
    }

    /// A group count that does not divide the env count must not index past the end.
    #[test]
    fn an_uneven_split_stays_in_range() {
        let assignment = Assignment::grouped(vec![
            (OpponentId::Pool(1), OpponentSeat::Model(AgentId(1))),
            (OpponentId::Pool(2), OpponentSeat::Model(AgentId(2))),
            (OpponentId::Pool(3), OpponentSeat::Model(AgentId(3))),
        ])
        .expect("assignment");

        for env in 0..10 {
            assert!(assignment.group_of(env, 10) < 3, "env {env}");
        }
        assert_eq!(assignment.group_of(9, 10), 2);
    }

    #[test]
    fn a_uniform_assignment_puts_every_env_on_one_opponent() {
        let assignment = Assignment::uniform(
            OpponentId::Heuristic(PlayerCode::R),
            OpponentSeat::Scripted(PlayerCode::R),
        );
        assert_eq!(assignment.concurrent(), 1);
        for env in 0..128 {
            assert_eq!(assignment.group_of(env, 128), 0);
        }
        // Scripted seats need no weights, so nothing has to be loaded for this one.
        assert!(assignment.agents().is_empty());
    }

    #[test]
    fn an_empty_assignment_is_refused() {
        assert!(Assignment::grouped(Vec::new()).is_err());
    }

    #[test]
    fn agent_ids_start_after_the_learner() {
        let mut models = OpponentModels::<B>::new();
        let first = models.insert(OpponentId::Pool(10), model());
        let second = models.insert(OpponentId::Baked("proto".to_string()), model());

        assert_ne!(first, AgentId::LEARNER);
        assert_ne!(second, AgentId::LEARNER);
        assert_eq!(first, AgentId(1));
        assert_eq!(second, AgentId(2));
        assert!(models.get(AgentId::LEARNER).is_none());
        assert!(models.get(first).is_some());
        assert_eq!(
            models.id_of(second),
            Some(&OpponentId::Baked("proto".to_string()))
        );
        assert_eq!(models.agent_of(&OpponentId::Pool(10)), Some(first));
    }

    /// A refresh that re-draws the same checkpoint must not renumber it under a live assignment.
    #[test]
    fn re_inserting_an_id_keeps_its_agent() {
        let mut models = OpponentModels::<B>::new();
        let first = models.insert(OpponentId::Pool(10), model());
        models.insert(OpponentId::Pool(20), model());
        let again = models.insert(OpponentId::Pool(10), model());

        assert_eq!(first, again);
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn clear_drops_everything() {
        let mut models = OpponentModels::<B>::new();
        models.insert(OpponentId::Pool(10), model());
        models.clear();

        assert!(models.is_empty());
        assert!(models.get(AgentId(1)).is_none());
        assert!(models.agent_of(&OpponentId::Pool(10)).is_none());
    }

    #[test]
    fn agents_lists_only_the_model_driven_groups() {
        let assignment = Assignment::grouped(vec![
            (OpponentId::Pool(1), OpponentSeat::Model(AgentId(1))),
            (
                OpponentId::Heuristic(PlayerCode::ER),
                OpponentSeat::Scripted(PlayerCode::ER),
            ),
            (OpponentId::Pool(2), OpponentSeat::Model(AgentId(4))),
        ])
        .expect("assignment");

        assert_eq!(assignment.agents(), vec![AgentId(1), AgentId(4)]);
    }
}
