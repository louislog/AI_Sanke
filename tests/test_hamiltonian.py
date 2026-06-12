import pytest

from ai_snake.policies.hamiltonian import build_hamiltonian_cycle, has_hamiltonian_cycle


@pytest.mark.parametrize("width,height", [(6, 6), (8, 8), (6, 9), (9, 6), (10, 12), (2, 2)])
def test_cycle_is_valid(width, height):
    cycle = build_hamiltonian_cycle(width, height)

    # 覆盖每个格子恰好一次
    assert len(cycle) == width * height
    assert len(set(cycle)) == width * height
    assert all(0 <= x < width and 0 <= y < height for x, y in cycle)

    # 相邻格（含首尾闭合）曼哈顿距离为 1
    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_odd_odd_has_no_cycle():
    assert not has_hamiltonian_cycle(7, 7)
    with pytest.raises(ValueError):
        build_hamiltonian_cycle(7, 7)


def test_even_boards_have_cycle():
    assert has_hamiltonian_cycle(6, 6)
    assert has_hamiltonian_cycle(6, 7)
    assert has_hamiltonian_cycle(7, 6)
