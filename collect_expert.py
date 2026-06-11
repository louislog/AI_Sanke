import runtime  # noqa: F401

"""用规则/搜索专家策略采集 (observation, action) 数据，供行为克隆使用。

示例：
    python collect_expert.py --policy hybrid --grid-size 8 --n-episodes 200 \
        --obs-mode grid_full --out data/expert_hybrid_8.npz
"""

import argparse
import os

import numpy as np

from policies import RULE_POLICIES, make_policy
from snake_env import SnakeEnv


def collect(
    policy_name: str,
    grid_size: int,
    n_episodes: int,
    obs_mode: str,
    grid_pad_size: int | None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    env = SnakeEnv(
        width=grid_size,
        height=grid_size,
        obs_mode=obs_mode,
        grid_pad_size=grid_pad_size or grid_size,
        reward="default",
    )
    policy = make_policy(policy_name)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    episode_scores: list[int] = []
    episode_coverages: list[float] = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=None if seed is None else seed + episode)
        policy.reset(env)
        while True:
            action = policy.select_action(env, obs)
            observations.append(obs)
            actions.append(action)
            masks.append(env.action_masks())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_scores.append(info["score"])
                episode_coverages.append(info["coverage"])
                break
        print(
            f"Episode {episode + 1}/{n_episodes}: score={info['score']} "
            f"coverage={info['coverage']:.2%} won={info['won']}"
        )

    env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "action_masks": np.asarray(masks, dtype=bool),
        "episode_scores": np.asarray(episode_scores, dtype=np.int64),
        "episode_coverages": np.asarray(episode_coverages, dtype=np.float32),
    }


def _main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations.")
    parser.add_argument("--policy", choices=RULE_POLICIES, default="hybrid")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--grid-pad-size", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument(
        "--obs-mode", choices=("grid_full", "grid", "vector"), default="grid_full"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="输出 npz 路径")
    args = parser.parse_args()

    out = args.out or f"data/expert_{args.policy}_{args.grid_size}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    data = collect(
        args.policy,
        args.grid_size,
        args.n_episodes,
        args.obs_mode,
        args.grid_pad_size,
        seed=args.seed,
    )
    np.savez_compressed(out, **data)

    scores = data["episode_scores"]
    coverages = data["episode_coverages"]
    print(
        f"\nSaved {len(data['actions']):,} transitions from {len(scores)} episodes to {out}\n"
        f"expert avg_score={scores.mean():.1f} max_score={scores.max()} "
        f"avg_coverage={coverages.mean():.2%}"
    )


if __name__ == "__main__":
    _main()
