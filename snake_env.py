from typing import Literal

import numpy as np
from gymnasium import Env, spaces

from snake_game import SnakeGame


class SnakeEnv(Env):
    """Gymnasium 兼容的贪吃蛇环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    FEATURE_DIM = SnakeGame.FEATURE_DIM
    GRID_CHANNELS = 3

    DISTANCE_REWARD_SCALE = 0.05
    DISTANCE_SHAPING_LENGTH_THRESHOLD: int | None = None
    TRAP_PENALTY_SCALE = 2.0
    TRAP_DROP_THRESHOLD = 0.05
    MAX_STEPS_FACTOR = 20
    MILESTONE_INTERVAL = 10
    MILESTONE_REWARD = 5.0

    def __init__(
        self,
        width: int = SnakeGame.DEFAULT_WIDTH,
        height: int = SnakeGame.DEFAULT_HEIGHT,
        max_steps: int | None = None,
        max_steps_factor: int = MAX_STEPS_FACTOR,
        render_mode: Literal["human", "rgb_array"] | None = None,
        obs_mode: Literal["vector", "grid"] = "grid",
        grid_pad_size: int | None = None,
        distance_reward_scale: float = DISTANCE_REWARD_SCALE,
        distance_shaping_length_threshold: int | None = DISTANCE_SHAPING_LENGTH_THRESHOLD,
        trap_penalty_scale: float = TRAP_PENALTY_SCALE,
        trap_drop_threshold: float = TRAP_DROP_THRESHOLD,
        milestone_interval: int = MILESTONE_INTERVAL,
        milestone_reward: float = MILESTONE_REWARD,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps or width * height * max_steps_factor
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.grid_pad_size = grid_pad_size or max(width, height)
        self.distance_reward_scale = distance_reward_scale
        self.distance_shaping_length_threshold = distance_shaping_length_threshold
        self.trap_penalty_scale = trap_penalty_scale
        self.trap_drop_threshold = trap_drop_threshold
        self.milestone_interval = milestone_interval
        self.milestone_reward = milestone_reward
        self.game = SnakeGame(width=width, height=height, render_mode=render_mode)
        self._episode_steps = 0
        self._prev_food_bfs: int | None = None
        self._prev_reachable_ratio = 1.0
        self._last_milestone = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = self._make_observation_space()

    def _make_observation_space(self) -> spaces.Box:
        if self.obs_mode == "grid":
            size = self.grid_pad_size
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.GRID_CHANNELS, size, size),
                dtype=np.float32,
            )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.FEATURE_DIM,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._episode_steps = 0
        self._prev_food_bfs = self.game.food_bfs_distance()
        self._prev_reachable_ratio = self.game.reachable_ratio()
        self._last_milestone = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        prev_bfs = self._prev_food_bfs
        reward, terminated = self.game.step(int(action))
        curr_bfs = self.game.food_bfs_distance()

        if not terminated:
            reward += self._bfs_distance_shaping(prev_bfs, curr_bfs)
            curr_reachable = self.game.reachable_ratio()
            drop = self._prev_reachable_ratio - curr_reachable
            if drop > self.trap_drop_threshold:
                reward -= self.trap_penalty_scale * drop
            self._prev_reachable_ratio = curr_reachable
            reward += self._milestone_reward()

        self._prev_food_bfs = curr_bfs
        self._episode_steps += 1
        truncated = self._episode_steps >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.game.valid_action_mask(), dtype=bool)

    def render(self):
        if self.render_mode is None:
            return None

        self.game.draw()

        if self.render_mode == "rgb_array":
            from snake_game import _get_pygame

            pygame = _get_pygame()
            array = pygame.surfarray.array3d(self.game.surface).transpose(1, 0, 2)
            return array.copy()

        return None

    def close(self):
        self.game.close()

    def _bfs_distance_shaping(
        self, prev_bfs: int | None, curr_bfs: int | None
    ) -> float:
        scale = self._distance_shaping_scale()
        if scale <= 0 or prev_bfs is None or curr_bfs is None:
            return 0.0
        return (prev_bfs - curr_bfs) * scale

    def _distance_shaping_scale(self) -> float:
        if self.distance_reward_scale <= 0:
            return 0.0
        if (
            self.distance_shaping_length_threshold is not None
            and len(self.game.snake) > self.distance_shaping_length_threshold
        ):
            return 0.0
        return self.distance_reward_scale

    def _milestone_reward(self) -> float:
        if self.milestone_interval <= 0 or self.milestone_reward <= 0:
            return 0.0
        milestone = self.game.score // self.milestone_interval
        if milestone <= self._last_milestone:
            return 0.0
        delta = milestone - self._last_milestone
        self._last_milestone = milestone
        return delta * self.milestone_reward

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "grid":
            grid = self.game.rasterize_board(
                pad_width=self.grid_pad_size,
                pad_height=self.grid_pad_size,
            )
            return np.asarray(grid, dtype=np.float32)
        return np.asarray(self.game.extract_features(), dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "score": self.game.score,
            "steps": self.game.steps,
            "game_over": self.game.game_over,
            "won": self.game.won,
            "ate_food": getattr(self.game, "_ate_food", False),
        }
