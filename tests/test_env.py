import numpy as np
import pytest

from ai_snake.snake_env import REWARD_PRESETS, RewardConfig, SnakeEnv, resolve_reward_config


@pytest.mark.parametrize("obs_mode,channels", [("grid", 3), ("grid_full", 8)])
def test_grid_obs_shapes(obs_mode, channels):
    env = SnakeEnv(width=8, height=8, obs_mode=obs_mode, grid_pad_size=8)
    obs, info = env.reset(seed=1)
    assert obs.shape == (channels, 8, 8)
    assert env.observation_space.contains(obs)
    env.close()


def test_vector_obs():
    env = SnakeEnv(width=8, height=8, obs_mode="vector")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (SnakeEnv.FEATURE_DIM,)
    env.close()


@pytest.mark.parametrize("size", [6, 8, 11])
def test_step_various_sizes(size):
    env = SnakeEnv(width=size, height=size, obs_mode="grid_full", grid_pad_size=size)
    obs, info = env.reset(seed=2)
    for _ in range(30):
        masks = env.action_masks()
        valid = np.flatnonzero(masks)
        action = int(valid[0]) if len(valid) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert "coverage" in info and "death_reason" in info
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def test_padded_obs():
    env = SnakeEnv(width=6, height=6, obs_mode="grid_full", grid_pad_size=10)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (8, 10, 10)
    # 棋盘掩码通道：实际棋盘内为 1，padding 为 0
    assert obs[4, :6, :6].sum() == 36
    assert obs[4].sum() == 36
    env.close()


@pytest.mark.parametrize("preset", sorted(REWARD_PRESETS))
def test_reward_presets(preset):
    env = SnakeEnv(width=6, height=6, reward=preset)
    env.reset(seed=4)
    _, reward, _, _, _ = env.step(0)
    assert np.isfinite(reward)
    env.close()


def test_resolve_reward_config():
    assert isinstance(resolve_reward_config("coverage"), RewardConfig)
    custom = RewardConfig(food_reward=1.0)
    assert resolve_reward_config(custom) is custom
    with pytest.raises(ValueError):
        resolve_reward_config("nope")


def test_death_reason_wall():
    env = SnakeEnv(width=6, height=6)
    env.reset(seed=5)
    terminated = False
    info = {}
    for _ in range(20):  # 一直直行必撞墙
        _, _, terminated, _, info = env.step(0)
        if terminated:
            break
    assert terminated
    assert info["death_reason"] == "wall"
    env.close()
