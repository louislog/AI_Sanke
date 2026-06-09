import runtime  # noqa: F401

import argparse
import time

from snake_env import SnakeEnv


def _main():
    parser = argparse.ArgumentParser(description="Smoke test the Snake Gymnasium environment.")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of random steps")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = SnakeEnv(render_mode=render_mode)
    obs, info = env.reset(seed=42)

    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("initial obs:", obs, "info:", info)

    for step in range(args.n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"step={step + 1} action={action} reward={reward:.2f} "
            f"score={info['score']} done={done}"
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
