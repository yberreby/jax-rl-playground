"""Simple episode-level baseline implementation."""

import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import NamedTuple


class EpisodeBaseline(NamedTuple):
    """Tracks mean episode total return."""

    mean_episode_return: Float[Array, ""]
    n_episodes: int


def update_episode_baseline(
    baseline: EpisodeBaseline,
    episode_totals: Float[Array, "n_episodes"],
) -> EpisodeBaseline:
    """Update baseline with new episode totals."""
    batch_size = episode_totals.shape[0]

    # Running average of episode totals
    total_episodes = baseline.n_episodes + batch_size
    new_mean = (
        baseline.mean_episode_return * baseline.n_episodes + jnp.sum(episode_totals)
    ) / total_episodes

    return EpisodeBaseline(
        mean_episode_return=new_mean,
        n_episodes=total_episodes,
    )


def compute_episode_advantages(
    episode_totals: Float[Array, "n_episodes"],
    baseline_mean: Float[Array, ""],
) -> Float[Array, "n_episodes"]:
    """Compute advantages for each episode."""
    return episode_totals - baseline_mean


def init_episode_baseline() -> EpisodeBaseline:
    """Initialize episode baseline."""
    return EpisodeBaseline(
        mean_episode_return=jnp.array(0.0),
        n_episodes=0,
    )
