"""轻量训练 profiling：环境 step、奖励、观测、评估与吞吐 FPS。"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ai_snake.snake_env import SnakeEnv


@dataclass
class ProfileStats:
    """累计计时（秒）与计数。"""

    env_step_s: float = 0.0
    game_step_s: float = 0.0
    reward_s: float = 0.0
    obs_s: float = 0.0
    flood_fill_s: float = 0.0
    tail_check_s: float = 0.0
    bfs_field_s: float = 0.0
    n_steps: int = 0
    eval_s: float = 0.0
    n_evals: int = 0
    rollout_fps: list[float] = field(default_factory=list)

    def report(self) -> str:
        n = max(self.n_steps, 1)
        lines = [
            "=== Training profile ===",
            f"  env.step total      : {self.env_step_s:.3f}s ({self.env_step_s / n * 1e3:.3f} ms/step)",
            f"    game.step         : {self.game_step_s / n * 1e3:.3f} ms/step",
            f"    reward compute    : {self.reward_s / n * 1e3:.3f} ms/step",
            f"      flood fill      : {self.flood_fill_s / n * 1e3:.3f} ms/step",
            f"      tail reachable  : {self.tail_check_s / n * 1e3:.3f} ms/step",
            f"    obs construct     : {self.obs_s / n * 1e3:.3f} ms/step",
            f"      bfs distance    : {self.bfs_field_s / n * 1e3:.3f} ms/step",
            f"  evaluation          : {self.eval_s:.3f}s ({self.n_evals} runs)",
        ]
        if self.rollout_fps:
            lines.append(
                f"  model throughput    : {np.mean(self.rollout_fps):.0f} FPS "
                f"(last {len(self.rollout_fps)} rollouts)"
            )
        lines.append("========================")
        return "\n".join(lines)


class ProfiledSnakeEnv(SnakeEnv):
    """SnakeEnv 子类，向 ProfileStats 上报细分耗时。"""

    def __init__(self, *args, profile_stats: ProfileStats | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._profile_stats = profile_stats
        if profile_stats is not None:
            self.attach_profiler(profile_stats)

    def attach_profiler(self, stats: ProfileStats) -> None:
        self._profile_stats = stats
        super().attach_profiler(stats)

    def step(self, action):
        stats = self._profile_stats
        t0 = time.perf_counter()
        result = super().step(action)
        if stats is not None:
            stats.env_step_s += time.perf_counter() - t0
            stats.n_steps += 1
        return result

    def _profile_game_step(self, elapsed: float) -> None:
        if self._profile_stats is not None:
            self._profile_stats.game_step_s += elapsed

    def _profile_reward(self, elapsed: float) -> None:
        if self._profile_stats is not None:
            self._profile_stats.reward_s += elapsed

    def _profile_obs(self, elapsed: float) -> None:
        if self._profile_stats is not None:
            self._profile_stats.obs_s += elapsed

    def _profile_flood_fill(self, elapsed: float) -> None:
        if self._profile_stats is not None:
            self._profile_stats.flood_fill_s += elapsed

    def _profile_tail_check(self, elapsed: float) -> None:
        if self._profile_stats is not None:
            self._profile_stats.tail_check_s += elapsed

    def _profile_bfs_field(self, elapsed: float) -> None:
        if self._profile_stats is not None:
            self._profile_stats.bfs_field_s += elapsed


class TrainingProfileCallback(BaseCallback):
    """训练过程中汇总 SB3 FPS 与自定义 profile 统计。"""

    def __init__(
        self,
        profile_stats: ProfileStats | None = None,
        report_freq: int = 50_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.profile_stats = profile_stats or ProfileStats()
        self.report_freq = report_freq
        self._last_report_step = 0
        self._rollout_start: float | None = None

    def _on_rollout_start(self) -> None:
        self._rollout_start = time.perf_counter()

    def _on_rollout_end(self) -> None:
        if self._rollout_start is not None:
            elapsed = time.perf_counter() - self._rollout_start
            n = self.model.n_steps * self.model.n_envs
            if elapsed > 0:
                self.profile_stats.rollout_fps.append(n / elapsed)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_report_step >= self.report_freq:
            self._last_report_step = self.num_timesteps
            if self.verbose:
                print(f"\n[profile @ {self.num_timesteps:,} steps]\n{self.profile_stats.report()}\n")
        return True

    def _on_training_end(self) -> None:
        if self.verbose:
            print(f"\n[profile final]\n{self.profile_stats.report()}\n")


class EvalTimerCallback(BaseCallback):
    """将 EvalCallback / MaskableEvalCallback 委托并统计 evaluation 耗时。"""

    def __init__(self, eval_callback: BaseCallback, profile_stats: ProfileStats):
        super().__init__(eval_callback.verbose)
        self._eval = eval_callback
        self.profile_stats = profile_stats

    def _init_callback(self) -> None:
        self._eval.init_callback(self.model, self.training_env, self.logger)

    def _on_training_start(self) -> None:
        self._eval.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        self._eval.on_rollout_start()

    def _on_step(self) -> bool:
        self._eval.n_calls = self.n_calls
        eval_freq = getattr(self._eval, "eval_freq", 0)
        will_eval = eval_freq > 0 and self.n_calls % eval_freq == 0
        t0 = time.perf_counter() if will_eval else 0.0
        result = self._eval._on_step()
        if will_eval:
            self.profile_stats.eval_s += time.perf_counter() - t0
            self.profile_stats.n_evals += 1
        return result

    def _on_rollout_end(self) -> None:
        self._eval.on_rollout_end()

    def _on_training_end(self) -> None:
        self._eval.on_training_end()


def _attach_profiler_to_vec_env(vec_env, stats: ProfileStats) -> None:
    """为向量化环境中的每个子环境挂载 profiler。"""
    try:
        envs = vec_env.envs
    except AttributeError:
        envs = [vec_env]
    for env in envs:
        unwrapped = env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "attach_profiler"):
            unwrapped.attach_profiler(stats)
        elif isinstance(unwrapped, SnakeEnv):
            unwrapped._profiler = stats  # noqa: SLF001


def run_env_profile(
    *,
    steps: int = 2000,
    grid_size: int = 10,
    obs_mode: str = "grid_full",
    reward_preset: str = "coverage",
    safety_check_interval: int | None = None,
) -> ProfileStats:
    """独立运行环境采样 profile（不训练神经网络）。"""
    stats = ProfileStats()
    env = ProfiledSnakeEnv(
        width=grid_size,
        height=grid_size,
        obs_mode=obs_mode,  # type: ignore[arg-type]
        grid_pad_size=grid_size,
        reward=reward_preset,
        safety_check_interval=safety_check_interval,
        profile_stats=stats,
    )
    env.reset(seed=0)
    for _ in range(steps):
        masks = env.action_masks()
        valid = np.flatnonzero(masks)
        action = int(valid[0]) if len(valid) else 0
        env.step(action)
    env.close()
    return stats


def _cpu_utilization_hint() -> str:
    try:
        import psutil

        return f"CPU {psutil.cpu_percent(interval=0.1):.0f}% ({psutil.cpu_count()} cores)"
    except ImportError:
        return f"CPU cores: {multiprocessing.cpu_count()} (install psutil for utilization)"


