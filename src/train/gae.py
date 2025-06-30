"""Generalized Advantage Estimation for REINFORCE."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.jit
def compute_gae(
    rewards: Float[Array, "batch_episodes episode_len"],
    values: Float[Array, "batch_episodes episode_len"] | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Float[Array, "batch_episodes episode_len"], Float[Array, "batch_episodes episode_len"]]:
    """Compute returns and advantages using GAE.
    
    If values is None, uses returns as advantages (standard REINFORCE).
    """
    batch_episodes, episode_len = rewards.shape
    
    if values is None:
        # No baseline - compute returns and use them as advantages
        def compute_returns_for_episode(episode_rewards):
            def scan_fn(future_return, reward):
                return_t = reward + gamma * future_return
                return return_t, return_t
            _, returns = jax.lax.scan(scan_fn, 0.0, episode_rewards, reverse=True)
            return returns
        
        returns = jax.vmap(compute_returns_for_episode)(rewards)
        advantages = returns  # No baseline
    else:
        # With baseline - compute TD residuals and GAE
        # Bootstrap with zero (episode ends)
        next_values = jnp.concatenate([values[:, 1:], jnp.zeros((batch_episodes, 1))], axis=1)
        
        # TD residuals: delta = r + gamma * V(s') - V(s)
        deltas = rewards + gamma * next_values - values
        
        # Compute GAE
        def compute_gae_for_episode(episode_deltas):
            def scan_fn(future_gae, delta):
                gae_t = delta + gamma * gae_lambda * future_gae
                return gae_t, gae_t
            _, advantages = jax.lax.scan(scan_fn, 0.0, episode_deltas, reverse=True)
            return advantages
        
        advantages = jax.vmap(compute_gae_for_episode)(deltas)
        returns = advantages + values
    
    return returns, advantages