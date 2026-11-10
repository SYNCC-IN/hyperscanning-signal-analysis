"""Checks of `src.sim_harness`: seeds and the parallel runner."""

from src.sim_harness import spawn_seeds


def test_spawned_seeds_do_not_depend_on_how_many_are_spawned():
    """Seed i is the same whatever n, so adding replicates or cells on another stream never moves existing seeds."""
    assert spawn_seeds(20260103, 1800)[:200] == spawn_seeds(20260103, 200)
    assert spawn_seeds([20260103, 1], 50)[:10] == spawn_seeds([20260103, 1], 10)
    assert not set(spawn_seeds([20260103, 1], 100)) & set(spawn_seeds(20260103, 100))
