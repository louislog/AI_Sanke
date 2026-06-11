import runtime  # noqa: F401

import argparse
import time

import numpy as np

from snake_env import SnakeEnv


def _main():
    parser = argparse.ArgumentParser(description="Smoke test the Snake Gymnasium environment.")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of random steps")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument(
        "--obs-mode",
        choices=("grid", "vector"),
        default="grid",
        help="Observation format to test",
    )
    parser.add_argument("--grid-size", type=int, default=8, help="Board size for smoke test")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = SnakeEnv(
        width=args.grid_size,
        height=args.grid_size,
        render_mode=render_mode,
        obs_mode=args.obs_mode,
        grid_pad_size=args.grid_size,
    )
    obs, info = env.reset(seed=42)

    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("initial obs shape:", np.asarray(obs).shape, "info:", info)
    print("action_masks:", env.action_masks())

    for step in range(args.n_steps):
        masks = env.action_masks()
        valid_actions = np.flatnonzero(masks)
        action = int(np.random.choice(valid_actions))
        assert masks[action], "sampled invalid action"
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if step < 5 or done:
            print(
                f"step={step + 1} action={action} reward={reward:.2f} "
                f"score={info['score']} won={info['won']} done={done}"
            )

        if args.render:
            env.render()
            time.sleep(0.05)

        if done:
            obs, info = env.reset()
            print("Episode ended, reset.")

    env.close()
    print("Environment test passed.")


if __name__ == "__main__":
    _main()
