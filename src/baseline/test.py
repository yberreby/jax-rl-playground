import jax.numpy as jnp
from . import update_baseline, compute_advantages, init_baseline


def test_init_baseline():
    state = init_baseline()
    assert state.mean == 0.0
    assert state.n_samples == 0


def test_update_baseline_single_batch():
    state = init_baseline()
    returns = jnp.array([1.0, 2.0, 3.0])

    new_state = update_baseline(state, returns)

    assert jnp.allclose(new_state.mean, 2.0)  # mean of [1, 2, 3]
    assert new_state.n_samples == 3


def test_update_baseline_incremental():
    # Test that incremental updates work correctly
    state = init_baseline()

    # First batch
    returns1 = jnp.array([1.0, 2.0])
    state = update_baseline(state, returns1)
    assert jnp.allclose(state.mean, 1.5)
    assert state.n_samples == 2

    # Second batch
    returns2 = jnp.array([3.0, 4.0, 5.0])
    state = update_baseline(state, returns2)

    # Overall mean should be (1+2+3+4+5)/5 = 3.0
    assert jnp.allclose(state.mean, 3.0)
    assert state.n_samples == 5


def test_compute_advantages():
    returns = jnp.array([1.0, 2.0, 3.0, 4.0])
    baseline = jnp.array(2.5)

    advantages = compute_advantages(returns, baseline)

    expected = jnp.array([-1.5, -0.5, 0.5, 1.5])
    assert jnp.allclose(advantages, expected)


def test_baseline_shapes():
    # Test with different batch sizes
    state = init_baseline()

    for batch_size in [1, 10, 100]:
        returns = jnp.ones(batch_size)
        new_state = update_baseline(state, returns)

        assert new_state.mean.shape == ()

        advantages = compute_advantages(returns, new_state.mean)
        assert advantages.shape == (batch_size,)
