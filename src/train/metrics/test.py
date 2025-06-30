import jax.numpy as jnp
from . import MetricsTracker
from ..episodes import EpisodeResult


def test_metrics_tracker_initialization():
    tracker = MetricsTracker()
    
    # Check all metric lists are initialized empty
    assert len(tracker.metrics) == 21  # Count actual metrics
    for key, values in tracker.metrics.items():
        assert values == []


def test_metrics_tracker_update():
    tracker = MetricsTracker()
    
    # Create mock episode result
    episode_batch = EpisodeResult(
        states=jnp.ones((100, 2)),
        actions=jnp.ones((100, 1)) * 0.5,
        rewards=jnp.ones(100),
        returns=jnp.ones(100) * 10.0,
        total_reward=jnp.array(1.0),
        log_probs=jnp.ones(100) * -0.5,
    )
    
    # Update metrics
    tracker.update(
        iteration=0,
        episode_batch=episode_batch,
        loss=jnp.array(0.5),
        grad_norm=jnp.array(1.2),
        grad_variance=jnp.array(0.1),
        normalized_advantages=jnp.ones(100),
        raw_advantage_std=jnp.array(2.0),
        baseline_mean=jnp.array(100.0),
        policy_test_mean=jnp.array([[0.1]]),
        policy_test_std=jnp.array([[0.5]]),
        episodes_per_iter=10,
    )
    
    # Check metrics were updated
    assert len(tracker.metrics["iteration"]) == 1
    assert tracker.metrics["iteration"][0] == 0
    assert tracker.metrics["mean_return"][0] == 1.0
    assert tracker.metrics["loss"][0] == 0.5
    assert abs(tracker.metrics["grad_norm"][0] - 1.2) < 1e-6
    assert tracker.metrics["episode_length"][0] == 10.0  # 100 rewards / 10 episodes


def test_metrics_tracker_multiple_updates():
    tracker = MetricsTracker()
    
    # Create mock episode result
    episode_batch = EpisodeResult(
        states=jnp.ones((100, 2)),
        actions=jnp.ones((100, 1)),
        rewards=jnp.ones(100),
        returns=jnp.ones(100),
        total_reward=jnp.array(1.0),
        log_probs=jnp.ones(100),
    )
    
    # Update multiple times
    for i in range(5):
        tracker.update(
            iteration=i,
            episode_batch=episode_batch,
            loss=jnp.array(float(i)),
            grad_norm=jnp.array(1.0),
            grad_variance=jnp.array(0.1),
            normalized_advantages=jnp.ones(100),
            raw_advantage_std=jnp.array(1.0),
            baseline_mean=jnp.array(0.0),
            policy_test_mean=jnp.array([[0.0]]),
            policy_test_std=jnp.array([[1.0]]),
            episodes_per_iter=1,
        )
    
    # Check all metrics have 5 entries
    for key, values in tracker.metrics.items():
        assert len(values) == 5
    
    # Check iterations are correct
    assert tracker.metrics["iteration"] == [0, 1, 2, 3, 4]
    assert tracker.metrics["loss"] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_metrics_tracker_to_dict():
    tracker = MetricsTracker()
    
    # Get dict before any updates
    metrics_dict = tracker.to_dict()
    assert isinstance(metrics_dict, dict)
    assert len(metrics_dict) == 21
    
    # Update and check again
    episode_batch = EpisodeResult(
        states=jnp.ones((10, 2)),
        actions=jnp.ones((10, 1)),
        rewards=jnp.ones(10),
        returns=jnp.ones(10),
        total_reward=jnp.array(1.0),
        log_probs=jnp.ones(10),
    )
    
    tracker.update(
        iteration=0,
        episode_batch=episode_batch,
        loss=jnp.array(0.5),
        grad_norm=jnp.array(1.0),
        grad_variance=jnp.array(0.1),
        normalized_advantages=jnp.ones(10),
        raw_advantage_std=jnp.array(1.0),
        baseline_mean=jnp.array(0.0),
        policy_test_mean=jnp.array([[0.0]]),
        policy_test_std=jnp.array([[1.0]]),
        episodes_per_iter=1,
    )
    
    metrics_dict = tracker.to_dict()
    assert all(len(v) == 1 for v in metrics_dict.values())