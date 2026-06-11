import runtime  # noqa: F401

import argparse
import os
from collections import deque
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from algos import available_algos, build_model, load_model, supports_action_masking
from snake_env import REWARD_PRESETS, SnakeEnv
from snake_game import SnakeGame

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

    _HAS_MASKABLE = True
except ImportError:
    MaskableEvalCallback = None  # type: ignore[misc, assignment]
    _HAS_MASKABLE = False


def _curriculum_sizes(final_size: int, custom: str | None) -> list[int]:
    if custom:
        sizes = sorted({int(s) for s in custom.split(",")})
    else:
        sizes = [s for s in (6, 8, 10, 12, 16) if s < final_size]
    if not sizes or sizes[-1] != final_size:
        sizes.append(final_size)
    return sizes


def _curriculum_phases(sizes: list[int], total_timesteps: int) -> list[tuple[int, int]]:
    """按网格尺寸划分课程阶段，最后一个阶段分配双倍步数权重。"""
    weights = [1.0] * (len(sizes) - 1) + [2.0]
    total_weight = sum(weights)
    phases: list[tuple[int, int]] = []
    allocated = 0
    for size, weight in zip(sizes, weights):
        steps = int(total_timesteps * weight / total_weight)
        phases.append((size, steps))
        allocated += steps
    last_size, last_steps = phases[-1]
    phases[-1] = (last_size, last_steps + total_timesteps - allocated)
    return phases


def _env_kwargs(grid_size: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "width": grid_size,
        "height": grid_size,
        "render_mode": None,
        "obs_mode": args.obs_mode,
        "grid_pad_size": args.grid_size,
        "max_steps_factor": args.max_steps_factor,
        "reward": args.reward_preset,
    }


class CurriculumCallback(BaseCallback):
    """按步数或覆盖率推进课程阶段（支持 checkpoint 继承同一模型）。"""

    def __init__(
        self,
        phases: list[tuple[int, int]],
        n_envs: int,
        monitor_dir: str,
        env_builder,
        coverage_threshold: float = 0.0,
        coverage_window: int = 30,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.phases = phases
        self.n_envs = n_envs
        self.monitor_dir = monitor_dir
        self.env_builder = env_builder
        self.coverage_threshold = coverage_threshold
        self.coverage_window = coverage_window
        self._recent_coverage: deque[float] = deque(maxlen=coverage_window)
        self._phase_idx = 0
        self._phase_start = 0

    def _on_training_start(self) -> None:
        self._phase_start = self.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and "coverage" in info:
                self._recent_coverage.append(float(info["coverage"]))
        return True

    def _should_advance(self) -> bool:
        _, phase_steps = self.phases[self._phase_idx]
        elapsed = self.num_timesteps - self._phase_start
        if elapsed >= phase_steps:
            return True
        if (
            self.coverage_threshold > 0
            and len(self._recent_coverage) >= self.coverage_window
            and sum(self._recent_coverage) / len(self._recent_coverage)
            >= self.coverage_threshold
        ):
            return True
        return False

    def _on_rollout_end(self) -> None:
        if self._phase_idx >= len(self.phases) - 1 or not self._should_advance():
            return

        self._phase_idx += 1
        self._phase_start = self.num_timesteps
        self._recent_coverage.clear()
        grid_size, _ = self.phases[self._phase_idx]

        new_env = make_vec_env(
            SnakeEnv,
            n_envs=self.n_envs,
            env_kwargs=self.env_builder(grid_size),
            monitor_dir=os.path.join(self.monitor_dir, f"grid{grid_size}"),
            monitor_kwargs={"info_keywords": ("coverage", "score")},
        )
        old_env = self.model.get_env()
        self.model.set_env(new_env)
        if old_env is not None:
            old_env.close()
        self.model._last_obs = new_env.reset()  # type: ignore[assignment]

        if self.verbose:
            print(
                f"\n[Curriculum] Phase {self._phase_idx + 1}/{len(self.phases)}: "
                f"grid {grid_size}x{grid_size} at {self.num_timesteps} steps\n"
            )


def _main():
    parser = argparse.ArgumentParser(description="Train an RL agent to play Snake.")
    parser.add_argument(
        "--algo",
        choices=available_algos(),
        default="maskable_ppo" if "maskable_ppo" in available_algos() else "ppo",
        help="RL 算法",
    )
    parser.add_argument("--n-envs", type=int, default=16, help="并行环境数")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=SnakeGame.DEFAULT_WIDTH,
        help="最终地图边长（课程学习在此结束）",
    )
    parser.add_argument("--curriculum", action="store_true", help="启用课程学习")
    parser.add_argument(
        "--curriculum-sizes",
        type=str,
        default=None,
        help="自定义课程尺寸，如 '6,8,10'（默认自动推导）",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.0,
        help=">0 时，近期平均覆盖率达到该值即提前进入下一课程阶段",
    )
    parser.add_argument(
        "--obs-mode",
        choices=("grid_full", "grid", "vector"),
        default="grid_full",
        help="观测模式：grid_full（8 通道，推荐）/ grid（3 通道）/ vector",
    )
    parser.add_argument(
        "--reward-preset",
        choices=sorted(REWARD_PRESETS),
        default="coverage",
        help="奖励预设（消融实验用）",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="初始化模型路径（如 BC 预训练模型），用于 RL fine-tuning",
    )
    parser.add_argument("--max-steps-factor", type=int, default=SnakeEnv.MAX_STEPS_FACTOR)
    parser.add_argument("--n-steps", type=int, default=4096, help="PPO rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--buffer-size", type=int, default=200_000, help="DQN 回放池大小")
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--log-dir", type=str, default="tmp/")
    args = parser.parse_args()

    best_dir = os.path.join(args.log_dir, "best")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    sizes = (
        _curriculum_sizes(args.grid_size, args.curriculum_sizes)
        if args.curriculum
        else [args.grid_size]
    )
    phases = _curriculum_phases(sizes, args.total_timesteps)
    start_grid = phases[0][0]
    env_builder = lambda grid_size: _env_kwargs(grid_size, args)  # noqa: E731

    if args.curriculum:
        print("Curriculum phases:")
        for i, (size, steps) in enumerate(phases, 1):
            print(f"  Phase {i}: {size}x{size} for {steps:,} steps")

    vec_env = make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        env_kwargs=env_builder(start_grid),
        monitor_dir=os.path.join(args.log_dir, f"grid{start_grid}"),
        monitor_kwargs={"info_keywords": ("coverage", "score")},
    )

    if args.init_model:
        print(f"Loading init model from {args.init_model}")
        model = load_model(args.init_model, algo=args.algo, env=vec_env)
        model.tensorboard_log = os.path.join(args.log_dir, "tensorboard")
    else:
        model = build_model(
            args.algo,
            vec_env,
            obs_mode=args.obs_mode,
            features_dim=args.features_dim,
            tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            buffer_size=args.buffer_size,
        )

    tb_log_name = f"{args.algo}_{args.obs_mode}_{args.reward_preset}"
    if args.curriculum:
        tb_log_name += "_curriculum"
    if args.init_model:
        tb_log_name += "_finetune"

    callbacks: list = []

    if args.curriculum:
        callbacks.append(
            CurriculumCallback(
                phases=phases,
                n_envs=args.n_envs,
                monitor_dir=args.log_dir,
                env_builder=env_builder,
                coverage_threshold=args.coverage_threshold,
            )
        )

    callbacks.append(
        CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=args.log_dir,
            name_prefix=f"{args.algo}_model",
        )
    )

    eval_env = SnakeEnv(**env_builder(args.grid_size))
    if supports_action_masking(args.algo) and _HAS_MASKABLE:
        eval_callback_cls = MaskableEvalCallback
    else:
        from stable_baselines3.common.callbacks import EvalCallback

        eval_callback_cls = EvalCallback

    callbacks.append(
        eval_callback_cls(
            eval_env,
            best_model_save_path=best_dir,
            log_path=os.path.join(args.log_dir, "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        )
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name,
        reset_num_timesteps=not args.init_model,
    )

    final_path = os.path.join(args.log_dir, f"{args.algo}_model_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")
    print(f"Best eval model saved to {os.path.join(best_dir, 'best_model.zip')}")


if __name__ == "__main__":
    _main()
