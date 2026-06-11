import runtime  # noqa: F401

import argparse
import os
from typing import Any

from snake_cnn import SnakeCNN
from snake_env import SnakeEnv
from snake_game import SnakeGame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

    _HAS_MASKABLE = True
except ImportError:
    MaskablePPO = None  # type: ignore[misc, assignment]
    MaskableEvalCallback = None  # type: ignore[misc, assignment]
    _HAS_MASKABLE = False


def _curriculum_phases(final_size: int, total_timesteps: int) -> list[tuple[int, int]]:
    """按网格尺寸划分课程阶段，返回 (grid_size, phase_timesteps) 列表。"""
    if final_size <= 8:
        return [(final_size, total_timesteps)]

    sizes: list[int] = []
    if final_size >= 20:
        sizes = [8, 10, 15, final_size]
    elif final_size >= 15:
        sizes = [8, 10, final_size]
    elif final_size >= 10:
        sizes = [8, final_size]
    else:
        sizes = [final_size]

    fractions = [0.15, 0.15, 0.25, 0.45][-len(sizes) :]
    phases: list[tuple[int, int]] = []
    allocated = 0
    for size, fraction in zip(sizes, fractions):
        steps = int(total_timesteps * fraction)
        phases.append((size, steps))
        allocated += steps
    if phases and allocated < total_timesteps:
        last_size, last_steps = phases[-1]
        phases[-1] = (last_size, last_steps + (total_timesteps - allocated))
    return phases


def _env_kwargs(grid_size: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "width": grid_size,
        "height": grid_size,
        "render_mode": None,
        "obs_mode": args.obs_mode,
        "grid_pad_size": args.grid_size,
        "max_steps_factor": args.max_steps_factor,
    }


def _build_model(vec_env, args: argparse.Namespace):
    if args.obs_mode == "grid":
        policy = "MlpPolicy"
        policy_kwargs = dict(
            features_extractor_class=SnakeCNN,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=dict(pi=[256], vf=[256]),
        )
    else:
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))

    algo_kwargs = dict(
        policy=policy,
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
        device="auto",
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=1.0,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
    )

    use_maskable = args.maskable and _HAS_MASKABLE
    if args.maskable and not _HAS_MASKABLE:
        print("Warning: sb3-contrib not installed, falling back to standard PPO.")

    if use_maskable:
        return MaskablePPO(**algo_kwargs)
    return PPO(**algo_kwargs)


class CurriculumCallback(BaseCallback):
    """在训练过程中按阶段切换网格尺寸。"""

    def __init__(
        self,
        phases: list[tuple[int, int]],
        n_envs: int,
        monitor_dir: str,
        env_builder,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.phases = phases
        self.n_envs = n_envs
        self.monitor_dir = monitor_dir
        self.env_builder = env_builder
        self._phase_idx = 0
        self._phase_start = 0

    def _on_training_start(self) -> None:
        self._phase_start = self.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self._phase_idx >= len(self.phases) - 1:
            return

        _, phase_steps = self.phases[self._phase_idx]
        if self.num_timesteps - self._phase_start < phase_steps:
            return

        self._phase_idx += 1
        self._phase_start = self.num_timesteps
        grid_size, _ = self.phases[self._phase_idx]

        new_env = make_vec_env(
            SnakeEnv,
            n_envs=self.n_envs,
            env_kwargs=self.env_builder(grid_size),
            monitor_dir=os.path.join(self.monitor_dir, f"grid{grid_size}"),
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
    parser = argparse.ArgumentParser(description="Train a PPO agent to play Snake.")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Number of timesteps to train for",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100_000,
        help="Checkpoint frequency in environment steps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Evaluation frequency in environment steps",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=SnakeGame.DEFAULT_WIDTH,
        help="Final grid width and height (curriculum ends here)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum: 8x8 -> 10x10 -> 15x15 -> final grid size",
    )
    parser.add_argument(
        "--obs-mode",
        choices=("grid", "vector"),
        default="grid",
        help="Observation format: grid (CNN) or vector (MLP)",
    )
    parser.add_argument(
        "--maskable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MaskablePPO with action masking (requires sb3-contrib)",
    )
    parser.add_argument(
        "--max-steps-factor",
        type=int,
        default=SnakeEnv.MAX_STEPS_FACTOR,
        help="Episode step limit = grid_cells * factor",
    )
    parser.add_argument("--n-steps", type=int, default=4096, help="PPO rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per rollout")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--features-dim", type=int, default=512, help="CNN feature dimension")
    parser.add_argument("--log-dir", type=str, default="tmp/", help="Logs and checkpoints directory")
    args = parser.parse_args()

    best_dir = os.path.join(args.log_dir, "best")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    phases = _curriculum_phases(args.grid_size, args.total_timesteps)
    start_grid = phases[0][0] if args.curriculum else args.grid_size
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
    )

    model = _build_model(vec_env, args)

    tb_log_name = "ppo"
    if args.n_envs > 0:
        tb_log_name += f"_nenv{args.n_envs}"
    if args.curriculum:
        tb_log_name += "_curriculum"
    if args.obs_mode == "grid":
        tb_log_name += "_cnn"
    if args.maskable and _HAS_MASKABLE:
        tb_log_name += "_maskable"

    callbacks: list = []

    if args.curriculum:
        callbacks.append(
            CurriculumCallback(
                phases=phases,
                n_envs=args.n_envs,
                monitor_dir=args.log_dir,
                env_builder=env_builder,
            )
        )

    callbacks.append(
        CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=args.log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
        )
    )

    eval_env = SnakeEnv(**env_builder(args.grid_size))
    eval_callback_cls = MaskableEvalCallback if args.maskable and _HAS_MASKABLE else None
    if eval_callback_cls is None:
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
    )

    final_path = os.path.join(args.log_dir, "rl_model_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")
    print(f"Best eval model saved to {os.path.join(best_dir, 'best_model.zip')}")


if __name__ == "__main__":
    _main()
