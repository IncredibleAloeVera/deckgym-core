//! Advantage estimation — `RL_ARCHITECTURE.md` §1.5.1.
//!
//! GAE with `γ = 1` (finite horizon, §1.5.1) and `λ = 0.95`, over episodes whose only reward is
//! terminal. That makes every intermediate TD residual a pure value difference:
//!
//! ```text
//! δ_t = V(s_{t+1}) − V(s_t)      for t < T−1
//! δ_{T−1} = R − V(s_{T−1})
//! A_t = δ_t + λ·A_{t+1}
//! ```

/// `λ` of §1.5.1. `γ` is not a parameter: §1.5.1 fixes it at 1.
pub const LAMBDA: f32 = 0.95;

/// One frame's learning targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// GAE advantage, before batch normalization.
    pub advantage: f32,
    /// Value-head regression target, `A_t + V(s_t)`.
    pub value_target: f32,
}

/// Targets for one episode: `values` are the value head's estimates at each of the learner's
/// frames, in order, and `reward` is the terminal outcome for the learner's seat.
pub fn episode_targets(values: &[f32], reward: f32) -> Vec<Target> {
    let mut targets = Vec::with_capacity(values.len());
    let mut carried = 0.0;

    for (index, value) in values.iter().enumerate().rev() {
        // At the last frame the reward replaces the bootstrap, and nothing carries in from past
        // the end of the game.
        let next_value = match values.get(index + 1) {
            Some(next) => *next,
            None => reward,
        };
        carried = (next_value - value) + LAMBDA * carried;
        targets.push(Target {
            advantage: carried,
            value_target: carried + value,
        });
    }

    targets.reverse();
    targets
}

/// Targets for a whole rollout, flattened in episode order.
///
/// Advantages are normalized across the **batch**, not per episode: episode lengths vary by an
/// order of magnitude here, and per-episode normalization would give a 4-frame game the same total
/// gradient weight as a 60-frame one.
pub fn batch_targets<'a>(episodes: impl IntoIterator<Item = (&'a [f32], f32)>) -> Vec<Target> {
    let mut targets: Vec<Target> = episodes
        .into_iter()
        .flat_map(|(values, reward)| episode_targets(values, reward))
        .collect();
    if targets.len() < 2 {
        return targets;
    }

    let count = targets.len() as f32;
    let mean = targets.iter().map(|t| t.advantage).sum::<f32>() / count;
    let variance = targets
        .iter()
        .map(|t| (t.advantage - mean).powi(2))
        .sum::<f32>()
        / count;
    let scale = variance.sqrt().max(1.0e-8);
    for target in &mut targets {
        target.advantage = (target.advantage - mean) / scale;
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one case with a closed form: a value function that already predicts the outcome exactly
    /// makes every residual zero, so every advantage is zero and the value targets reproduce the
    /// predictions. A sign error or a misplaced bootstrap breaks this immediately.
    #[test]
    fn a_perfect_critic_yields_zero_advantage() {
        for target in episode_targets(&[1.0, 1.0, 1.0], 1.0) {
            assert!(target.advantage.abs() < 1e-6, "{target:?}");
            assert!((target.value_target - 1.0).abs() < 1e-6, "{target:?}");
        }
    }

    /// A flat-zero critic that then wins: the terminal residual is the whole reward, and it comes
    /// back through the episode by λ per step and nothing else, since γ = 1.
    #[test]
    fn the_terminal_reward_propagates_by_lambda() {
        let targets = episode_targets(&[0.0, 0.0, 0.0], 1.0);
        assert!((targets[2].advantage - 1.0).abs() < 1e-6, "{targets:?}");
        assert!((targets[1].advantage - LAMBDA).abs() < 1e-6, "{targets:?}");
        assert!(
            (targets[0].advantage - LAMBDA * LAMBDA).abs() < 1e-6,
            "{targets:?}"
        );
    }

    /// A loss has to come back negative everywhere, or the policy gradient would reinforce it.
    #[test]
    fn a_loss_gives_negative_advantage() {
        let targets = episode_targets(&[0.0, 0.0], -1.0);
        assert!(targets.iter().all(|t| t.advantage < 0.0), "{targets:?}");
    }

    /// Nothing may carry across an episode boundary: batching a win in front of a loss must leave
    /// both episodes' advantages exactly where they were on their own.
    #[test]
    fn episodes_do_not_bleed_into_each_other() {
        let win: &[f32] = &[0.0, 0.0, 0.0];
        let loss: &[f32] = &[0.0, 0.0];
        let alone: Vec<_> = episode_targets(win, 1.0)
            .into_iter()
            .chain(episode_targets(loss, -1.0))
            .collect();

        // `batch_targets` standardizes, so compare the ordering-sensitive part: the sign pattern
        // and the ratio between neighbouring frames, both of which a bleed would destroy.
        let batched = batch_targets(vec![(win, 1.0), (loss, -1.0)]);
        assert_eq!(batched.len(), alone.len());
        assert!(
            batched[..3].iter().all(|t| t.advantage > 0.0),
            "{batched:?}"
        );
        assert!(
            batched[3..].iter().all(|t| t.advantage < 0.0),
            "{batched:?}"
        );
        assert!((alone[1].advantage / alone[0].advantage - 1.0 / LAMBDA).abs() < 1e-5);
    }

    #[test]
    fn batch_advantages_are_standardized() {
        let win: &[f32] = &[0.0, 0.0, 0.0];
        let loss: &[f32] = &[0.0, 0.0];
        let targets = batch_targets(vec![(win, 1.0), (loss, -1.0)]);

        let count = targets.len() as f32;
        let mean = targets.iter().map(|t| t.advantage).sum::<f32>() / count;
        let variance = targets
            .iter()
            .map(|t| (t.advantage - mean).powi(2))
            .sum::<f32>()
            / count;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        assert!((variance - 1.0).abs() < 1e-4, "variance {variance}");
    }
}
