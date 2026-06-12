"""Gymnasium 贪吃蛇环境：可配置奖励 + 多种观测模式。

观测模式：
- vector    : 24 维手工特征（兼容旧版）
- grid      : 3 通道栅格（蛇头 / 蛇身 / 食物，兼容旧版）
- grid_full : 8 通道栅格，面向满图覆盖任务：
              蛇头、蛇身、食物、蛇身顺序场、棋盘掩码（区分 padding 墙）、
              到食物 BFS 距离场、到蛇尾 BFS 距离场、Hamiltonian index 场

奖励：通过 RewardConfig 完全可配置，内置 default / sparse / coverage / greedy
四种预设，方便消融实验。
"""

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from gymnasium import Env, spaces

from ai_snake.policies.grid_utils import (
    bfs_path,
    blocked_mask_from_coords,
    fill_normalized_distance_channel,
    reachable_ratio_numpy,
)
from ai_snake.snake_game import SnakeGame

if TYPE_CHECKING:
    from ai_snake.profiling import ProfileStats


def default_safety_check_interval(grid_size: int) -> int:
    """按地图尺寸给出安全指标（reachable / tail）默认计算间隔。"""
    if grid_size <= 8:
        return 1
    if grid_size <= 12:
        return 2
    return 4


@dataclass
class RewardConfig:
    """奖励配置。所有惩罚字段填正数，计算时取负。"""

    food_reward: float = 10.0
    step_penalty: float = 0.0
    # 死亡惩罚 = death_penalty + death_penalty_coverage_scale * 当前覆盖率
    # 覆盖率越高死亡越可惜，后期死亡惩罚更大
    death_penalty: float = 50.0
    death_penalty_coverage_scale: float = 0.0
    win_reward: float = 100.0
    # 吃到食物时按覆盖率给的额外奖励：coverage_bonus_scale * coverage
    coverage_bonus_scale: float = 0.0
    # 可达空间骤降惩罚（flood fill 比例下降超过阈值时按降幅惩罚）
    trap_penalty_scale: float = 2.0
    trap_drop_threshold: float = 0.05
    # 吃完食物后蛇头无法到达蛇尾的惩罚（短视自困信号）
    tail_unreachable_penalty: float = 0.0
    # 靠近食物 BFS 距离塑形（谨慎使用，蛇长超过阈值后自动关闭）
    distance_reward_scale: float = 0.05
    distance_shaping_length_threshold: int | None = None
    # 里程碑奖励：每吃 interval 个食物给一次
    milestone_interval: int = 10
    milestone_reward: float = 5.0


REWARD_PRESETS: dict[str, RewardConfig] = {
    # 旧版行为，向后兼容
    "default": RewardConfig(),
    # 几乎无塑形，检验算法本身的信用分配能力
    "sparse": RewardConfig(
        food_reward=1.0,
        death_penalty=1.0,
        win_reward=10.0,
        trap_penalty_scale=0.0,
        distance_reward_scale=0.0,
        milestone_interval=0,
        milestone_reward=0.0,
    ),
    # 面向满图覆盖：后期死亡惩罚大、奖励高覆盖、惩罚自困，弱化靠近食物奖励
    "coverage": RewardConfig(
        food_reward=10.0,
        death_penalty=30.0,
        death_penalty_coverage_scale=120.0,
        win_reward=200.0,
        coverage_bonus_scale=20.0,
        trap_penalty_scale=4.0,
        trap_drop_threshold=0.04,
        tail_unreachable_penalty=5.0,
        distance_reward_scale=0.02,
        distance_shaping_length_threshold=12,
        milestone_interval=10,
        milestone_reward=5.0,
    ),
    # 短视贪吃对照组（用于消融，预期后期表现差）
    "greedy": RewardConfig(
        food_reward=10.0,
        death_penalty=10.0,
        trap_penalty_scale=0.0,
        distance_reward_scale=0.2,
    ),
}


def resolve_reward_config(reward: "RewardConfig | str | None") -> RewardConfig:
    if reward is None:
        return replace(REWARD_PRESETS["default"])
    if isinstance(reward, RewardConfig):
        return reward
    if reward in REWARD_PRESETS:
        return replace(REWARD_PRESETS[reward])
    raise ValueError(f"未知奖励预设: {reward}，可选 {sorted(REWARD_PRESETS)}")


ObsMode = Literal["vector", "grid", "grid_full"]

GRID_CHANNELS = {"grid": 3, "grid_full": 8}


class SnakeEnv(Env):
    """Gymnasium 兼容的贪吃蛇环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    FEATURE_DIM = SnakeGame.FEATURE_DIM
    GRID_CHANNELS = 3  # 旧版兼容
    MAX_STEPS_FACTOR = 20

    def __init__(
        self,
        width: int = SnakeGame.DEFAULT_WIDTH,
        height: int = SnakeGame.DEFAULT_HEIGHT,
        max_steps: int | None = None,
        max_steps_factor: int = MAX_STEPS_FACTOR,
        render_mode: Literal["human", "rgb_array"] | None = None,
        obs_mode: ObsMode = "grid",
        grid_pad_size: int | None = None,
        reward: RewardConfig | str | None = "default",
        safety_check_interval: int | None = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps or width * height * max_steps_factor
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.grid_pad_size = grid_pad_size or max(width, height)
        self.reward_config = resolve_reward_config(reward)
        self.safety_check_interval = safety_check_interval
        self.game = SnakeGame(width=width, height=height, render_mode=render_mode)

        self._episode_steps = 0
        self._prev_food_bfs: int | None = None
        self._prev_reachable_ratio = 1.0
        self._cached_reachable_ratio = 1.0
        self._last_milestone = 0
        self._ham_index_map = self._build_ham_index_map()
        self._profiler: ProfileStats | None = None

        # grid_full 观测预分配缓冲，避免每步重复分配
        self._obs_buffer: np.ndarray | None = None
        self._blocked_mask: np.ndarray | None = None
        self._dist_buf: np.ndarray | None = None
        self._visited_buf: np.ndarray | None = None
        if self.obs_mode == "grid_full":
            size = self.grid_pad_size
            self._obs_buffer = np.zeros((8, size, size), dtype=np.float32)
            self._blocked_mask = np.zeros((self.height, self.width), dtype=np.bool_)
            self._dist_buf = np.zeros((self.height, self.width), dtype=np.int32)
            self._visited_buf = np.zeros((self.height, self.width), dtype=np.bool_)
            self._init_static_obs_channels()

        self.action_space = spaces.Discrete(3)
        self.observation_space = self._make_observation_space()

    # ---- 空间定义 ----

    def _make_observation_space(self) -> spaces.Box:
        if self.obs_mode in GRID_CHANNELS:
            size = self.grid_pad_size
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(GRID_CHANNELS[self.obs_mode], size, size),
                dtype=np.float32,
            )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.FEATURE_DIM,),
            dtype=np.float32,
        )

    def _build_ham_index_map(self) -> np.ndarray | None:
        from ai_snake.policies.hamiltonian import build_hamiltonian_cycle, has_hamiltonian_cycle

        if not has_hamiltonian_cycle(self.width, self.height):
            return None
        cycle = build_hamiltonian_cycle(self.width, self.height)
        index_map = np.zeros((self.height, self.width), dtype=np.float32)
        denom = max(len(cycle) - 1, 1)
        for i, (x, y) in enumerate(cycle):
            index_map[y, x] = i / denom
        return index_map

    def attach_profiler(self, profiler: "ProfileStats") -> None:
        self._profiler = profiler

    def _resolved_safety_interval(self) -> int:
        if self.safety_check_interval is not None:
            return max(1, self.safety_check_interval)
        return default_safety_check_interval(max(self.width, self.height))

    def _need_safety_metrics(self, ate_food: bool) -> bool:
        """是否在奖励阶段计算 flood fill / tail reachable 等重型指标。"""
        cfg = self.reward_config
        needs_trap = cfg.trap_penalty_scale > 0
        needs_tail = cfg.tail_unreachable_penalty > 0 and ate_food
        if not needs_trap and not needs_tail:
            return False
        if ate_food:
            return True
        snake_len = len(self.game.snake)
        if snake_len < 4:
            return False
        interval = self._resolved_safety_interval()
        return self._episode_steps % interval == 0

    def _init_static_obs_channels(self) -> None:
        """棋盘掩码与 Hamiltonian 场在 episode 内不变，初始化一次。"""
        assert self._obs_buffer is not None
        self._obs_buffer[4].fill(0.0)
        self._obs_buffer[4, : self.height, : self.width] = 1.0
        self._obs_buffer[7].fill(0.0)
        if self._ham_index_map is not None:
            self._obs_buffer[7, : self.height, : self.width] = self._ham_index_map

    def _compute_reachable_ratio(self) -> float:
        t0 = time.perf_counter()
        snake = self.game.snake
        head = snake[0]
        blocked = {(seg.x, seg.y) for seg in snake[1:-1]}
        ratio = reachable_ratio_numpy(
            (head.x, head.y),
            blocked,
            self.width,
            self.height,
            blocked_mask=self._blocked_mask,
            visited_buf=self._visited_buf,
        )
        if self._profiler is not None:
            self._profiler.flood_fill_s += time.perf_counter() - t0
        return ratio

    # ---- Gym API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._episode_steps = 0
        self._prev_food_bfs = self.game.food_bfs_distance()
        self._prev_reachable_ratio = self._compute_reachable_ratio()
        self._cached_reachable_ratio = self._prev_reachable_ratio
        self._last_milestone = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action)
        cfg = self.reward_config
        prev_bfs = self._prev_food_bfs
        prev_coverage = self.game.coverage_ratio()

        t_game = time.perf_counter()
        _, terminated = self.game.step(int(action))
        if self._profiler is not None:
            self._profiler.game_step_s += time.perf_counter() - t_game

        ate_food = getattr(self.game, "_ate_food", False)
        t_reward = time.perf_counter()

        reward = -cfg.step_penalty
        if terminated:
            if self.game.won:
                reward += cfg.win_reward + self.game.board_capacity()
            else:
                reward -= (
                    cfg.death_penalty
                    + cfg.death_penalty_coverage_scale * prev_coverage
                )
        else:
            if ate_food:
                reward += cfg.food_reward
                reward += cfg.coverage_bonus_scale * self.game.coverage_ratio()
                reward += self._milestone_reward()
                if cfg.tail_unreachable_penalty > 0 and not self._tail_reachable():
                    reward -= cfg.tail_unreachable_penalty

            if cfg.distance_reward_scale > 0 and self._distance_shaping_active():
                curr_bfs = self.game.food_bfs_distance()
                reward += self._bfs_distance_shaping(prev_bfs, curr_bfs)
                self._prev_food_bfs = curr_bfs
            else:
                self._prev_food_bfs = prev_bfs

            if cfg.trap_penalty_scale > 0:
                if self._need_safety_metrics(ate_food):
                    curr_reachable = self._compute_reachable_ratio()
                    self._cached_reachable_ratio = curr_reachable
                else:
                    curr_reachable = self._cached_reachable_ratio
                drop = self._prev_reachable_ratio - curr_reachable
                if drop > cfg.trap_drop_threshold:
                    reward -= cfg.trap_penalty_scale * drop
                if self._need_safety_metrics(ate_food):
                    self._prev_reachable_ratio = curr_reachable

        if self._profiler is not None:
            self._profiler.reward_s += time.perf_counter() - t_reward

        self._episode_steps += 1
        truncated = not terminated and self._episode_steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _distance_shaping_active(self) -> bool:
        cfg = self.reward_config
        threshold = cfg.distance_shaping_length_threshold
        if threshold is None:
            return True
        return len(self.game.snake) <= threshold

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.game.valid_action_mask(), dtype=bool)

    def render(self):
        if self.render_mode is None:
            return None

        self.game.draw()

        if self.render_mode == "rgb_array":
            from ai_snake.snake_game import _get_pygame

            pygame = _get_pygame()
            array = pygame.surfarray.array3d(self.game.surface).transpose(1, 0, 2)
            return array.copy()

        return None

    def close(self):
        self.game.close()

    # ---- 奖励组件 ----

    def _tail_reachable(self) -> bool:
        t0 = time.perf_counter()
        snake = [(seg.x, seg.y) for seg in self.game.snake]
        if len(snake) < 3:
            ok = True
        else:
            blocked = set(snake[1:-1])
            ok = (
                bfs_path(snake[0], snake[-1], blocked, self.width, self.height)
                is not None
            )
        if self._profiler is not None:
            self._profiler.tail_check_s += time.perf_counter() - t0
        return ok

    def _bfs_distance_shaping(
        self, prev_bfs: int | None, curr_bfs: int | None
    ) -> float:
        cfg = self.reward_config
        if cfg.distance_reward_scale <= 0 or prev_bfs is None or curr_bfs is None:
            return 0.0
        if (
            cfg.distance_shaping_length_threshold is not None
            and len(self.game.snake) > cfg.distance_shaping_length_threshold
        ):
            return 0.0
        return (prev_bfs - curr_bfs) * cfg.distance_reward_scale

    def _milestone_reward(self) -> float:
        cfg = self.reward_config
        if cfg.milestone_interval <= 0 or cfg.milestone_reward <= 0:
            return 0.0
        milestone = self.game.score // cfg.milestone_interval
        if milestone <= self._last_milestone:
            return 0.0
        delta = milestone - self._last_milestone
        self._last_milestone = milestone
        return delta * cfg.milestone_reward

    # ---- 观测 ----

    def _get_obs(self) -> np.ndarray:
        t0 = time.perf_counter()
        if self.obs_mode == "grid":
            grid = self.game.rasterize_board(
                pad_width=self.grid_pad_size,
                pad_height=self.grid_pad_size,
            )
            obs = np.asarray(grid, dtype=np.float32)
        elif self.obs_mode == "grid_full":
            obs = self._full_grid_obs()
        else:
            obs = np.asarray(self.game.extract_features(), dtype=np.float32)
        if self._profiler is not None:
            self._profiler.obs_s += time.perf_counter() - t0
        return obs

    def _full_grid_obs(self) -> np.ndarray:
        assert self._obs_buffer is not None
        assert self._blocked_mask is not None
        assert self._dist_buf is not None

        obs = self._obs_buffer
        w, h = self.width, self.height
        game = self.game
        snake = game.snake
        n = len(snake)
        head = snake[0]
        tail = snake[-1]
        food = game.food

        # 仅清零动态通道 0-3, 5-6；4/7 为静态
        obs[0].fill(0.0)
        obs[1].fill(0.0)
        obs[2].fill(0.0)
        obs[3].fill(0.0)
        obs[5].fill(0.0)
        obs[6].fill(0.0)

        obs[0, head.y, head.x] = 1.0
        if n > 1:
            body_x = np.fromiter((seg.x for seg in snake[1:]), dtype=np.int32, count=n - 1)
            body_y = np.fromiter((seg.y for seg in snake[1:]), dtype=np.int32, count=n - 1)
            obs[1, body_y, body_x] = 1.0
        obs[2, food.y, food.x] = 1.0

        order = np.linspace(1.0, 1.0 / n, n, dtype=np.float32)
        all_x = np.fromiter((seg.x for seg in snake), dtype=np.int32, count=n)
        all_y = np.fromiter((seg.y for seg in snake), dtype=np.int32, count=n)
        obs[3, all_y, all_x] = order

        blocked_mask_from_coords(
            ((seg.x, seg.y) for seg in snake[1:-1]),
            w,
            h,
            out=self._blocked_mask,
        )
        max_dist = w * h
        t_bfs = time.perf_counter()
        fill_normalized_distance_channel(
            obs[5],
            [(food.x, food.y)],
            self._blocked_mask,
            w,
            h,
            max_dist,
            dist_buf=self._dist_buf,
        )
        fill_normalized_distance_channel(
            obs[6],
            [(tail.x, tail.y)],
            self._blocked_mask,
            w,
            h,
            max_dist,
            dist_buf=self._dist_buf,
        )
        if self._profiler is not None:
            self._profiler.bfs_field_s += time.perf_counter() - t_bfs

        return obs

    def _get_info(self) -> dict:
        return {
            "score": self.game.score,
            "steps": self.game.steps,
            "game_over": self.game.game_over,
            "won": self.game.won,
            "ate_food": getattr(self.game, "_ate_food", False),
            "snake_length": len(self.game.snake),
            "coverage": self.game.coverage_ratio(),
            "death_reason": getattr(self.game, "death_reason", None),
        }
