import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
import equinox as eqx
from functools import partial
from ...policy import sample_actions


class EpisodeResult(NamedTuple):
    states: Float[Array, "episode_len obs_dim"]
    actions: Float[Array, "episode_len 1"]
    rewards: Float[Array, "episode_len"]
    returns: Float[Array, "episode_len"]
    total_reward: Float[Array, ""]
    log_probs: Float[Array, "episode_len"]


@partial(jax.jit, static_argnames=["gamma"])
def compute_returns(
    rewards: Float[Array, "episode_len"],
    gamma: float = 0.99,
) -> Float[Array, "episode_len"]:
    """Compute discounted returns."""
    def scan_fn(future_return, reward):
        return_t = reward + gamma * future_return
        return return_t, return_t
    
    _, returns = jax.lax.scan(scan_fn, 0.0, rewards, reverse=True)
    return returns


@eqx.filter_jit
def collect_episode(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    key: Array,
    max_steps: int = 400,
    reward_scale: float = 0.1,  # Scale down rewards
) -> EpisodeResult:
    key, reset_key = jax.random.split(key)
    initial_env_state = env_reset(reset_key)

    def step_fn(carry, key):
        env_state, done = carry

        # Sample action from policy
        action, log_prob = sample_actions(policy, env_state.state, key)

        # Step environment
        result = env_step(env_state, action)

        # Mask updates if already done
        new_env_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, old, new), result.env_state, env_state
        )
        new_done = jnp.maximum(done, result.done)

        # Output for this step (current state, not next) with scaled reward
        scaled_reward = result.reward * reward_scale * (1.0 - done)
        output = (env_state.state, action, scaled_reward, log_prob)

        return (new_env_state, new_done), output

    # Generate keys for all steps
    keys = jax.random.split(key, max_steps)

    # Run scan
    (final_env_state, final_done), (states, actions, rewards, log_probs) = jax.lax.scan(
        step_fn, (initial_env_state, jnp.array(0.0)), keys
    )

    # Compute returns with discount
    returns = compute_returns(rewards)

    # Count actual steps
    episode_length = jnp.sum(rewards != 0.0)

    return EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=jnp.sum(rewards) / jnp.maximum(episode_length, 1.0),  # Average per step
        log_probs=log_probs,
    )


@eqx.filter_jit
def collect_episodes(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    key: Array,
    n_episodes: int,
    max_steps: int = 400,
    reward_scale: float = 0.1,
) -> EpisodeResult:
    keys = jax.random.split(key, n_episodes)

    # vmap over episodes
    vmapped_collect = jax.vmap(
        lambda k: collect_episode(policy, env_step, env_reset, k, max_steps, reward_scale), 
        in_axes=0
    )

    results = vmapped_collect(keys)

    # Flatten batch dimension for training
    obs_dim = results.states.shape[-1]
    act_dim = results.actions.shape[-1]
    states = results.states.reshape(-1, obs_dim)
    actions = results.actions.reshape(-1, act_dim)
    rewards = results.rewards.reshape(-1)
    returns = results.returns.reshape(-1)
    log_probs = results.log_probs.reshape(-1)

    # Keep per-episode averages
    # Count non-zero rewards to get episode lengths
    episode_lengths = jax.vmap(lambda r: jnp.sum(r != 0.0))(results.rewards)
    episode_sums = jax.vmap(jnp.sum)(results.rewards)
    episode_averages = episode_sums / jnp.maximum(episode_lengths, 1.0)

    return EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=jnp.mean(episode_averages),  # Mean of per-episode averages
        log_probs=log_probs,
    )