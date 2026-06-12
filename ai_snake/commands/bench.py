"""性能基准：环境 step profiling 与观测构造吞吐。"""

import ai_snake.runtime  # noqa: F401

import argparse
import time

import numpy as np

from ai_snake.profiling import _cpu_utilization_hint, run_env_profile
from ai_snake.snake_env import SnakeEnv


def bench_obs(steps: int, grid_size: int, obs_mode: str = "grid_full") -> float:
    env = SnakeEnv(
        width=grid_size,
        height=grid_size,
        obs_mode=obs_mode,  # type: ignore[arg-type]
        grid_pad_size=grid_size,
        reward="coverage",
    )
    env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(steps):
        masks = env.action_masks()
        valid = np.flatnonzero(masks)
        action = int(valid[0]) if len(valid) else 0
        env.step(action)
    elapsed = time.perf_counter() - t0
    env.close()
    return elapsed


def _run_env(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Profile SnakeEnv step throughput.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--obs-mode", default="grid_full")
    parser.add_argument("--reward-preset", default="coverage")
    parser.add_argument("--safety-check-interval", type=int, default=None)
    args = parser.parse_args(argv)

    print(f"Profiling {args.steps} env steps on {args.grid_size}x{args.grid_size} ...")
    print(_cpu_utilization_hint())

    stats = run_env_profile(
        steps=args.steps,
        grid_size=args.grid_size,
        obs_mode=args.obs_mode,
        reward_preset=args.reward_preset,
        safety_check_interval=args.safety_check_interval,
    )
    print(stats.report())
    n = max(stats.n_steps, 1)
    print(f"  env throughput        : {n / stats.env_step_s:.0f} steps/s")


def _run_obs(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark observation construction.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--obs-mode", default="grid_full")
    args = parser.parse_args(argv)

    elapsed = bench_obs(args.steps, args.grid_size, args.obs_mode)
    ms_per_step = elapsed / args.steps * 1000
    print(f"{args.obs_mode} obs benchmark ({args.grid_size}x{args.grid_size}, {args.steps} steps)")
    print(f"  total : {elapsed:.3f}s")
    print(f"  per step (env.step incl. obs): {ms_per_step:.3f} ms")
    print(f"  throughput: {args.steps / elapsed:.0f} steps/s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark env throughput and observations.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="环境 step 细分 profiling")
    sub.add_parser("obs", help="观测构造吞吐 benchmark")
    args, rest = parser.parse_known_args(argv)

    if args.command == "env":
        _run_env(rest)
    else:
        _run_obs(rest)


if __name__ == "__main__":
    main()
