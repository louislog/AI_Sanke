import pytest

from policies import make_policy
from snake_env import SnakeEnv


def _run_episode(policy_name: str, size: int, seed: int = 0, max_steps_factor: int = 120):
    env = SnakeEnv(
        width=size,
        height=size,
        obs_mode="grid_full",
        grid_pad_size=size,
        max_steps_factor=max_steps_factor,
    )
    policy = make_policy(policy_name)
    obs, info = env.reset(seed=seed)
    policy.reset(env)
    while True:
        action = policy.select_action(env, obs)
        assert action in (0, 1, 2)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.close()
            return info


@pytest.mark.parametrize("policy_name", ["random", "search", "hamiltonian", "hybrid"])
@pytest.mark.parametrize("size", [6, 8])
def test_policies_run(policy_name, size):
    info = _run_episode(policy_name, size)
    assert info["score"] >= 0


def test_hamiltonian_full_map_6x6():
    wins = 0
    for seed in range(3):
        info = _run_episode("hamiltonian", 6, seed=seed)
        wins += int(info["won"])
    assert wins == 3, "Hamiltonian 策略应在 6x6 稳定满图"


def test_hybrid_full_map_6x6():
    wins = 0
    for seed in range(3):
        info = _run_episode("hybrid", 6, seed=seed)
        wins += int(info["won"])
    assert wins == 3, "Hybrid 策略应在 6x6 稳定满图"


def test_search_policy_odd_board():
    # 奇x奇地图无 Hamiltonian 回路，search/hybrid 仍应正常运行
    info = _run_episode("search", 7, seed=0)
    assert info["score"] > 0
    info = _run_episode("hybrid", 7, seed=0)
    assert info["score"] > 0


def test_hybrid_high_coverage_8x8():
    info = _run_episode("hybrid", 8, seed=0)
    assert info["coverage"] >= 0.9, f"8x8 覆盖率过低: {info['coverage']:.2%}"
