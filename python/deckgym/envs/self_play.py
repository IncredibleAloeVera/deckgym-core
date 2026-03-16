#!/usr/bin/env python3
"""
SelfPlayEnv - Gymnasium environment for self-play training.

Wraps the base DeckGymEnv to support training against various opponents:
- Frozen model copies (self-play)
- Rust-side bots (expectiminimax, heuristics)
- ONNX models
"""

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

from deckgym.config import TrainingConfig, DEFAULT_CONFIG
from deckgym.envs.base import DeckGymEnv


class SelfPlayEnv(gym.Env):
    """
    Gymnasium wrapper for training against various opponents.

    Supports:
    - "self": Frozen copy of agent (self-play)
    - "e2", "e3", etc.: Rust expectiminimax bots
    - "random": Random baseline
    - "o*" : ONNX models

    Player 0 (agent) trains against Player 1 (opponent).
    Starting player is randomized by game seed.
    """

    def __init__(
        self,
        deck_loader,
        opponent_model=None,
        opponent_type: str = "self",
        config: TrainingConfig = DEFAULT_CONFIG,
    ):
        """
        Initialize the training environment.

        Args:
            deck_loader: Samples deck pairs
            opponent_model: Frozen model for self-play (used when opponent_type="self")
            opponent_type: "self", "e2", "e3", "v", "random", "o*"
            config: Training configuration
        """
        self.deck_loader = deck_loader
        self.opponent_model = opponent_model
        self.opponent_type = opponent_type
        self.config = config
        self._episode_actions = 0
        self._current_decks = None  # (deck_a, deck_b) strings

        from deckgym.diagnostic_logger import get_logger

        self.diagnostic_logger = get_logger()

        # Initialize with a dummy game to get space dimensions
        self._env = DeckGymEnv(*deck_loader.sample_pair())
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def set_opponent_model(self, model):
        """Update the frozen opponent model."""
        self.opponent_model = model

    def set_opponent_type(self, opponent_type: str):
        """Change opponent type (e.g., 'self', 'e2', 'e3')."""
        self.opponent_type = opponent_type
        if opponent_type != "self":
            self.opponent_model = None  # Clear frozen model when using bots

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment with new random decks.

        The game seed determines starting player (random ~50/50).
        """
        deck_a, deck_b = self.deck_loader.sample_pair()
        self._current_decks = (deck_a, deck_b)

        # Create the base environment
        self._env = DeckGymEnv(deck_a, deck_b, seed=seed)
        obs, info = self._env.reset(seed=seed)

        # For bot opponents, replace the game with one that has the bot
        if self.opponent_type != "self":
            self._replace_game_with_bot(deck_a, deck_b, seed)
            # Re-get observation from the new game
            obs = np.array(self._env.game.get_obs(), dtype=np.float32)

        # Play opponent's turns if they go first
        obs, info, _, _ = self._play_opponent_turns(obs, info)
        self._episode_actions = 0
        self.diagnostic_logger.clear_history(0)  # env_idx 0 for single env

        return obs, info

    def _replace_game_with_bot(self, deck_a: str, deck_b: str, seed=None):
        """Replace _env.game with a PyGame that has bot opponent."""
        import deckgym

        # Write decks to temp files (PyGame requires file paths for bot support)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(deck_a)
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(deck_b)
            path_b = f.name

        try:
            # Create game with: player 0 = random (agent controls), player 1 = bot
            bot_game = deckgym.Game(
                path_a, path_b, players=["r", self.opponent_type], seed=seed
            )
            # Replace the environment's game
            self._env.game = bot_game
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute agent's action, then play opponent's turn(s).

        Returns:
            obs: Game state from agent's perspective
            reward: +1 win, -1 loss, 0 otherwise, plus bonus or penalty based on game score and speed
            terminated: True if game ended normally
            truncated: True if game ended due to limits
            info: Additional information
        """
        # Check turn limit
        turn_count = self._get_turn_count()
        if turn_count > self.config.max_turns:
            return self._end_episode_truncated("max_turns_exceeded")

        # Check action limit
        self._episode_actions += 1
        if self._episode_actions > self.config.max_total_actions:
            return self._end_episode_truncated("max_actions_exceeded")

        # Execute agent's action
        self.diagnostic_logger.record_action(0, action)
        try:
            obs, reward, done, truncated, info = self._env.step(action)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise e
            print(f"WARNING: Game error during agent turn: {e}")
            self.diagnostic_logger.log_error(
                "agent_panic",
                0,
                self._env.game.get_state(),
                {
                    "error": str(e),
                    "deck_a": self._current_decks[0],
                    "deck_b": self._current_decks[1],
                },
            )
            return self._end_episode_error(str(e))

        # Play opponent's response
        if not done and not truncated:
            obs, info, opp_reward, opp_done = self._play_opponent_turns(obs, info)
            if opp_done:
                done = True
                reward = opp_reward

        return obs, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for the agent."""
        try:
            return self._env.action_masks()
        except BaseException as e:
            # Panic recovery - reset and return EndTurn-only
            print(f"WARNING: Panic in SelfPlayEnv.action_masks: {e}")
            self._env.reset()
            mask = np.zeros(self._env.action_space.n, dtype=np.bool_)
            mask[0] = True
            return mask

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _play_opponent_turns(self, obs, info):
        """Play all opponent turns until agent's turn or game over."""
        final_reward = 0.0
        done = False
        turn_actions = 0

        try:
            while (
                self._env.game.current_player() == 1
                and not self._env.game.is_game_over()
            ):
                turn_actions += 1
                self._episode_actions += 1

                # Safety limits
                if turn_actions > self.config.max_actions_per_turn:
                    self._log_freeze_warning("opponent_turn_limit", turn_actions)
                    return obs, info, 0.0, True

                if self._episode_actions > self.config.max_total_actions:
                    self._log_freeze_warning(
                        "episode_action_limit", self._episode_actions
                    )
                    return obs, info, 0.0, True

                if self.opponent_type == "self":
                    # Self-play: select action with frozen model, execute in _env
                    action = self._select_opponent_action(obs)
                    obs, reward, done, truncated, info = self._env.step(action)
                    final_reward = (
                        -reward
                    )  # Invert reward (opponent's loss = agent's gain)
                else:
                    # Bot opponent: use play_tick() on the underlying game
                    # play_tick() both selects AND executes the bot's action
                    self._env.game.play_tick()
                    # Get updated observation after bot's action
                    obs = np.array(self._env.game.get_obs(), dtype=np.float32)
                    done = self._env.game.is_game_over()
                    if done:
                        # Determine turn-based reward (matches vec_game.rs calculate_reward)
                        state = self._env.game.get_state()
                        if state.winner and not state.winner.is_tie:
                            winner = state.winner.winner
                            my_points = float(state.points[0])
                            opp_points = float(state.points[1])
                            point_diff = my_points - opp_points
                            turn_count = float(state.turn_count)

                            # Base reward from point difference
                            base = 1.0 + (point_diff / 6.0)

                            # Speed factor: 1 + (13 - turns) / 13
                            speed_factor = 1.0 + ((13.0 - turn_count) / 13.0)

                            if winner == 0:
                                # Agent won: (1.0 + diff/6) * speed
                                # 3-0: (1 + 3/6) = 1.5 * speed
                                # 3-2: (1 + 1/6) = 1.16 * speed
                                final_reward = (1.0 + (point_diff / 6.0)) * max(
                                    1.0, speed_factor
                                )
                            else:
                                # Opponent won: (-1.0 + diff/6) * speed
                                # 0-3 (diff -3): (-1 + -0.5) = -1.5 * speed (Heavy penalty)
                                # 2-3 (diff -1): (-1 + -0.16) = -1.16 * speed (Lighter penalty)
                                final_reward = (
                                    -1.0 + (point_diff / 6.0)
                                ) * speed_factor
                        else:
                            final_reward = self.config.draw_reward  # Tie

                if done:
                    break
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise e
            print(f"WARNING: Game error during opponent turn: {e}")
            self.diagnostic_logger.log_error(
                "opponent_panic",
                0,
                self._env.game.get_state(),
                {
                    "error": str(e),
                    "deck_a": self._current_decks[0],
                    "deck_b": self._current_decks[1],
                },
            )
            return obs, info, 0.0, True

        return obs, info, final_reward, done

    def _select_opponent_action(self, obs) -> int:
        """Select action for the opponent (self-play only).

        Note: Bot opponents are handled directly via play_tick() in _play_opponent_turns(),
        so this method is only used for self-play with frozen model.
        """
        if self.opponent_type != "self":
            # Bot opponent: action is handled by play_tick() in _play_opponent_turns()
            return -1  # Sentinel: not used for bots

        # Self-play: use frozen model
        action_mask = self._env.action_masks()
        if self.opponent_model is not None:
            action, _ = self.opponent_model.predict(
                obs, action_masks=action_mask, deterministic=False
            )
        else:
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        return action

    def _get_turn_count(self) -> int:
        """Get current turn count from game state."""
        return self._env.game.get_state().turn_count

    def _end_episode_truncated(self, reason: str):
        """End episode due to limit exceeded."""
        self.diagnostic_logger.log_error(reason, 0, self._env.game.get_state())
        obs = np.array(self._env.game.get_obs(), dtype=np.float32)
        return obs, 0.0, False, True, {"truncated_reason": reason}

    def _end_episode_error(self, error: str):
        """End episode due to game error - reset internal game completely."""
        # Reset the internal game to get a fresh state
        self._env.reset()
        obs = np.array(self._env.game.get_obs(), dtype=np.float32)
        return obs, 0.0, True, False, {"error": error}

    def _log_freeze_warning(self, reason: str, count: int):
        """Log warning when safety limit is hit."""
        turn = self._get_turn_count()
        player = self._env.game.current_player()
        print(
            f"WARNING: {reason} triggered (count={count}, turn={turn}, player={player})"
        )
        self.diagnostic_logger.log_error(
            reason, 0, self._env.game.get_state(), {"count": count}
        )
