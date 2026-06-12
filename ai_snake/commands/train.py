import ai_snake.runtime  # noqa: F401

import argparse
import os
from collections import deque
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from ai_snake.algos import available_algos, build_model, load_model, supports_action_masking
from ai_snake.profiling import (
    EvalTimerCallback,
    ProfileStats,
    TrainingProfileCallback,
    _attach_profiler_to_vec_env,
)
from ai_snake.snake_env import REWARD_PRESETS, SnakeEnv, default_safety_check_interval
from ai_snake.snake_game import SnakeGame
from ai_snake.training_utils import (
    algo_training_defaults,
    apply_torch_compile,
    make_snake_vec_env,
    print_device_info,
    resolve_training_device,
)

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
    safety = args.safety_check_interval
    if safety is None:
        safety = default_safety_check_interval(grid_size)
    return {
        "width": grid_size,
        "height": grid_size,
        "render_mode": None,
        "obs_mode": args.obs_mode,
        "grid_pad_size": args.grid_size,
        "max_steps_factor": args.max_steps_factor,
        "reward": args.reward_preset,
        "safety_check_interval": safety,
    }


class CurriculumCallback(BaseCallback):
    """按步数或覆盖率推进课程阶段（支持 checkpoint 继承同一模型）。"""

    def __init__(
        self,
        phases: list[tuple[int, int]],
        n_envs: int,
        monitor_dir: str,
        env_builder,
        vec_env_type: str,
        seed: int | None,
        coverage_threshold: float = 0.0,
        coverage_window: int = 30,
        profile_stats: ProfileStats | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.phases = phases
        self.n_envs = n_envs
        self.monitor_dir = monitor_dir
        self.env_builder = env_builder
        self.vec_env_type = vec_env_type
        self.seed = seed
        self.coverage_threshold = coverage_threshold
        self.coverage_window = coverage_window
        self.profile_stats = profile_stats
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

        new_env = make_snake_vec_env(
            n_envs=self.n_envs,
            env_kwargs=self.env_builder(grid_size),
            monitor_dir=os.path.join(self.monitor_dir, f"grid{grid_size}"),
            vec_env_type=self.vec_env_type,
            seed=self.seed,
        )
        if self.profile_stats is not None:
            _attach_profiler_to_vec_env(new_env, self.profile_stats)

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


def _parse_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "on")


def main(argv: list[str] | None = None):
    defaults = algo_training_defaults("maskable_ppo", SnakeGame.DEFAULT_WIDTH)

    parser = argparse.ArgumentParser(description="Train an RL agent to play Snake.")
    parser.add_argument(
        "--algo",
        choices=available_algos(),
        default="maskable_ppo" if "maskable_ppo" in available_algos() else "ppo",
        help="RL 算法",
    )
    parser.add_argument("--n-envs", type=int, default=16, help="并行环境数")
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="subproc",
        help="向量化后端：dummy 单进程 / subproc 多进程（n-envs>1 时推荐 subproc）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（多进程环境各自偏移）")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--save-freq", type=int, default=defaults["save_freq"])
    parser.add_argument("--eval-freq", type=int, default=defaults["eval_freq"])
    parser.add_argument("--no-eval", action="store_true", help="训练期间不做评估（完整评估用 snake-ai eval）")
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=defaults["n_eval_episodes"],
        help="训练中每次评估的 episode 数（轻量默认 5）",
    )
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
        "--safety-check-interval",
        type=int,
        default=None,
        help="flood fill / 可达空间惩罚计算间隔（步）；默认按地图尺寸自动",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="初始化模型路径（如 BC 预训练模型），用于 RL fine-tuning",
    )
    parser.add_argument("--max-steps-factor", type=int, default=SnakeEnv.MAX_STEPS_FACTOR)
    parser.add_argument("--n-steps", type=int, default=defaults["n_steps"], help="PPO rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=defaults["n_epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=defaults["ent_coef"])
    parser.add_argument("--buffer-size", type=int, default=200_000, help="DQN 回放池大小")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="DQN 开始学习前的步数")
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--log-dir", type=str, default="tmp/")
    parser.add_argument("--log-interval", type=int, default=defaults["log_interval"], help="TensorBoard / 终端日志间隔（rollout 次数）")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="训练设备：auto/cpu/cuda/mps",
    )
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA 设备编号")
    parser.add_argument(
        "--torch-compile",
        type=_parse_bool,
        default=False,
        help="是否对 policy 启用 torch.compile（PyTorch 2+，实验性）",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用训练 profiling（环境 step / 奖励 / obs / FPS）",
    )
    args = parser.parse_args(argv)

    # 按算法与地图尺寸刷新未显式覆盖的默认超参
    algo_defaults = algo_training_defaults(args.algo, args.grid_size)
    if args.n_steps == defaults["n_steps"]:
        args.n_steps = algo_defaults["n_steps"]
    if args.n_epochs == defaults["n_epochs"]:
        args.n_epochs = algo_defaults["n_epochs"]

    device = resolve_training_device(args.device, args.cuda_device)
    print_device_info(device)

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

    profile_stats = ProfileStats() if args.profile else None
    if args.profile and args.vec_env == "subproc" and args.n_envs > 1:
        print(
            "[profile] 警告: SubprocVecEnv 下环境细分耗时无法跨进程汇总；"
            " 环境瓶颈请用 `snake-ai bench env`，训练吞吐 FPS 仍有效。"
        )

    vec_env = make_snake_vec_env(
        n_envs=args.n_envs,
        env_kwargs=env_builder(start_grid),
        monitor_dir=os.path.join(args.log_dir, f"grid{start_grid}"),
        vec_env_type=args.vec_env,
        seed=args.seed,
    )
    print(
        f"Vec env: {getattr(vec_env, 'vec_env_cls_name', 'unknown')} "
        f"({args.vec_env}, n_envs={args.n_envs})"
    )
    if profile_stats is not None:
        _attach_profiler_to_vec_env(vec_env, profile_stats)

    if args.init_model:
        print(f"Loading init model from {args.init_model}")
        model = load_model(args.init_model, algo=args.algo, env=vec_env, device=device)
        model.tensorboard_log = os.path.join(args.log_dir, "tensorboard")
    else:
        model = build_model(
            args.algo,
            vec_env,
            obs_mode=args.obs_mode,
            features_dim=args.features_dim,
            tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
            device=device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
        )

    apply_torch_compile(model, args.torch_compile)

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
                vec_env_type=args.vec_env,
                seed=args.seed,
                coverage_threshold=args.coverage_threshold,
                profile_stats=profile_stats,
            )
        )

    callbacks.append(
        CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=args.log_dir,
            name_prefix=f"{args.algo}_model",
        )
    )

    if not args.no_eval:
        eval_env = SnakeEnv(**env_builder(args.grid_size))
        if supports_action_masking(args.algo) and _HAS_MASKABLE:
            eval_callback_cls = MaskableEvalCallback
        else:
            from stable_baselines3.common.callbacks import EvalCallback

            eval_callback_cls = EvalCallback

        eval_cb = eval_callback_cls(
            eval_env,
            best_model_save_path=best_dir,
            log_path=os.path.join(args.log_dir, "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        if profile_stats is not None:
            callbacks.append(EvalTimerCallback(eval_cb, profile_stats))
        else:
            callbacks.append(eval_cb)

    if profile_stats is not None:
        callbacks.append(TrainingProfileCallback(profile_stats, report_freq=50_000))

    callback = CallbackList(callbacks) if len(callbacks) > 1 else (callbacks[0] if callbacks else None)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=tb_log_name,
        reset_num_timesteps=not args.init_model,
        log_interval=args.log_interval,
    )

    final_path = os.path.join(args.log_dir, f"{args.algo}_model_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")
    if not args.no_eval:
        print(f"Best eval model saved to {os.path.join(best_dir, 'best_model.zip')}")
    if profile_stats is not None:
        print(profile_stats.report())


if __name__ == "__main__":
    main()
