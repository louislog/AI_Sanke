"""对比 grid_full 观测构造速度（优化后预分配 + numpy BFS）。"""

import argparse
import time

import numpy as np

from snake_env import SnakeEnv


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


def _main():
    parser = argparse.ArgumentParser(description="Benchmark observation construction.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--obs-mode", default="grid_full")
    args = parser.parse_args()

    elapsed = bench_obs(args.steps, args.grid_size, args.obs_mode)
    ms_per_step = elapsed / args.steps * 1000
    print(f"grid_full obs benchmark ({args.grid_size}x{args.grid_size}, {args.steps} steps)")
    print(f"  total : {elapsed:.3f}s")
    print(f"  per step (env.step incl. obs): {ms_per_step:.3f} ms")
    print(f"  throughput: {args.steps / elapsed:.0f} steps/s")


if __name__ == "__main__":
    _main()
