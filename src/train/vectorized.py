import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
import equinox as eqx
# Environment functions passed as parameters instead of importing from specific env


class VectorizedEpisodeState(NamedTuple):
    state: Float[Array, "2"]
    done: Float[Array, ""]
    episode_step: int
    total_reward: Float[Array, ""]


class VectorizedEpisodeResult(NamedTuple):
    states: Float[Array, "T 2"]
    actions: Float[Array, "T 1"]
    rewards: Float[Array, "T"]
    returns: Float[Array, "T"]
    total_rewards: Float[Array, ""]
    episode_lengths: int


@eqx.filter_jit
def collect_episode_scan(
    policy,  # nnx.Module
    env_dynamics,  # Environment dynamics function
    env_reward,  # Environment reward function
    env_reset,  # Environment reset function
    key: Array,
    max_steps: int = 200,
) -> VectorizedEpisodeResult:
    key, reset_key = jax.random.split(key)
    initial_state = env_reset(reset_key)

    # Initialize episode state
    init_episode_state = VectorizedEpisodeState(
        state=initial_state,
        done=jnp.array(0.0),
        episode_step=0,
        total_reward=jnp.array(0.0),
    )

    def step_fn(carry, key):
        episode_state = carry

        # Sample action
        obs_batch = episode_state.state[None, :]
        mean, std = policy(obs_batch)
        eps = jax.random.normal(key, mean.shape)
        action = (mean + std * eps)[0]

        # Step dynamics
        next_state = env_dynamics(episode_state.state, action)

        # Compute reward
        reward = env_reward(episode_state.state, action)

        # Update episode state
        new_episode_step = episode_state.episode_step + 1
        new_done = jnp.where(
            episode_state.done == 1.0,
            1.0,  # Stay done
            jnp.array(new_episode_step >= max_steps, dtype=jnp.float32),
        )

        # Only accumulate reward if not done
        new_total_reward = episode_state.total_reward + reward * (
            1.0 - episode_state.done
        )

        # Only update state if not done
        new_state = jnp.where(
            episode_state.done == 1.0,
            episode_state.state,  # Keep old state if done
            next_state,
        )

        new_episode_state = VectorizedEpisodeState(
            state=new_state,
            done=new_done,
            episode_step=new_episode_step,
            total_reward=new_total_reward,
        )

        # Output for this step (use current state, not next)
        output = (episode_state.state, action, reward, episode_state.done)

        return new_episode_state, output

    # Generate random keys for all steps
    keys = jax.random.split(key, max_steps)

    # Run scan
    final_state, (states, actions, rewards, dones) = jax.lax.scan(
        step_fn, init_episode_state, keys
    )

    # Compute returns (reverse cumsum, masked by done)
    def compute_returns_scan(rewards, dones):
        def backward_fn(future_return, reward_done):
            reward, done = reward_done
            current_return = reward + future_return * (1.0 - done)
            return current_return, current_return

        _, returns = jax.lax.scan(
            backward_fn, jnp.array(0.0), (rewards, dones), reverse=True
        )
        return returns

    returns = compute_returns_scan(rewards, dones)

    return VectorizedEpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_rewards=final_state.total_reward,
        episode_lengths=final_state.episode_step,
    )


@eqx.filter_jit
def collect_episodes_vmap(
    policy,  # nnx.Module
    env_dynamics,  # Environment dynamics function
    env_reward,  # Environment reward function
    env_reset,  # Environment reset function
    key: Array,
    n_episodes: int,
    max_steps: int = 200,
) -> VectorizedEpisodeResult:
    keys = jax.random.split(key, n_episodes)

    # vmap over episodes
    vmapped_collect = jax.vmap(
        lambda k: collect_episode_scan(policy, env_dynamics, env_reward, env_reset, k, max_steps), in_axes=0
    )

    results = vmapped_collect(keys)

    # Results are now batched: (n_episodes, T, ...)
    # Flatten for training
    states = results.states.reshape(-1, 2)
    actions = results.actions.reshape(-1, 1)
    rewards = results.rewards.reshape(-1)
    returns = results.returns.reshape(-1)

    # Keep per-episode metrics separate
    return VectorizedEpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_rewards=results.total_rewards,  # Shape: (n_episodes,)
        episode_lengths=results.episode_lengths,  # Shape: (n_episodes,)
    )
