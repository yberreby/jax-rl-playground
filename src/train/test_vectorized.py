import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import pytest
import time
from ..policy import GaussianPolicy
from .vectorized import collect_episode_scan, collect_episodes_vmap
from . import collect_episode  # Original version
from ..pendulum import dynamics, reward, reset, step, reset_env


def test_collect_episode_scan_matches_original():
    """Test that scan version produces similar results to original."""
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(2, 1, hidden_dim=32, use_layernorm=False)

    # Original version
    key1, key2 = jax.random.split(key)
    result_orig = collect_episode(policy, step, reset_env, key1, max_steps=50)

    # Scan version
    result_scan = collect_episode_scan(policy, dynamics, reward, reset, key2, max_steps=50)

    # Check shapes match
    assert result_orig.states.shape == result_scan.states.shape
    assert result_orig.actions.shape == result_scan.actions.shape
    assert result_orig.rewards.shape == result_scan.rewards.shape

    # Check returns calculation
    assert jnp.allclose(
        jnp.sum(result_orig.rewards), result_orig.total_reward, rtol=1e-4
    )
    assert jnp.allclose(
        jnp.sum(result_scan.rewards), result_scan.total_rewards, rtol=1e-4
    )


def test_collect_episodes_vmap_shape():
    """Test vmap version produces correct shapes."""
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(2, 1, hidden_dim=32)

    n_episodes = 5
    max_steps = 100

    results = collect_episodes_vmap(policy, dynamics, reward, reset, key, n_episodes, max_steps)

    # Check flattened shapes
    total_steps = n_episodes * max_steps
    assert results.states.shape == (total_steps, 2)
    assert results.actions.shape == (total_steps, 1)
    assert results.rewards.shape == (total_steps,)
    assert results.returns.shape == (total_steps,)

    # Check per-episode metrics
    assert results.total_rewards.shape == (n_episodes,)
    assert results.episode_lengths.shape == (n_episodes,)


@pytest.mark.slow
def test_vectorized_performance():
    """Compare performance of different collection methods."""
    print("\n=== Episode Collection Performance ===")

    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(2, 1, hidden_dim=64)
    n_episodes = 10

    # Original Python loop version
    start = time.time()
    episodes = []
    for i in range(n_episodes):
        key, subkey = jax.random.split(key)
        ep = collect_episode(policy, step, reset_env, subkey)
        episodes.append(ep)
    # Force computation
    jtu.tree_map(lambda x: x.block_until_ready(), episodes[-1])
    time_orig = time.time() - start
    print(f"Original (Python loop): {time_orig:.3f}s")

    # Scan version with loop
    start = time.time()
    episodes_scan = []
    for i in range(n_episodes):
        key, subkey = jax.random.split(key)
        ep = collect_episode_scan(policy, dynamics, reward, reset, subkey)
        episodes_scan.append(ep)
    jtu.tree_map(lambda x: x.block_until_ready(), episodes_scan[-1])
    time_scan_loop = time.time() - start
    print(f"Scan version (Python loop): {time_scan_loop:.3f}s")

    # Fully vectorized version (with JIT compilation)
    start = time.time()
    results = collect_episodes_vmap(policy, dynamics, reward, reset, key, n_episodes)
    jtu.tree_map(lambda x: x.block_until_ready(), results)
    time_vmap_with_jit = time.time() - start
    print(f"Vectorized (first call with JIT): {time_vmap_with_jit:.3f}s")

    # Fully vectorized version (already compiled)
    start = time.time()
    results = collect_episodes_vmap(policy, dynamics, reward, reset, key, n_episodes)
    jtu.tree_map(lambda x: x.block_until_ready(), results)
    time_vmap = time.time() - start
    print(f"Vectorized (compiled): {time_vmap:.3f}s")

    print(f"\nSpeedup vs original: {time_orig / time_vmap:.1f}x")
    print(f"Speedup vs scan loop: {time_scan_loop / time_vmap:.1f}x")

    # Test with more episodes
    n_episodes_large = 50
    print(f"\nTesting with {n_episodes_large} episodes:")

    start = time.time()
    results = collect_episodes_vmap(policy, dynamics, reward, reset, key, n_episodes_large)
    jtu.tree_map(lambda x: x.block_until_ready(), results)
    time_large = time.time() - start
    print(f"Vectorized {n_episodes_large} episodes: {time_large:.3f}s")
    print(f"Time per episode: {time_large / n_episodes_large * 1000:.1f}ms")
