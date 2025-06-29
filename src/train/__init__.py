import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
from flax import nnx
import optax
import equinox as eqx
from ..baseline import BaselineState, update_baseline, compute_advantages
from ..distributions import gaussian_log_prob


class TrainState(NamedTuple):
    step: int
    key: Array
    baseline: BaselineState


class EpisodeResult(NamedTuple):
    states: Float[Array, "episode_len 2"]
    actions: Float[Array, "episode_len 1"]
    rewards: Float[Array, "episode_len"]
    returns: Float[Array, "episode_len"]
    total_reward: Float[Array, ""]


@jax.jit
def compute_returns(
    rewards: Float[Array, "episode_len"],
) -> Float[Array, "episode_len"]:
    # For undiscounted case, return at time t is sum of rewards from t to end
    returns = jnp.cumsum(rewards[::-1])[::-1]
    return returns


@eqx.filter_jit
def collect_episode(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function  
    key: Array,
    max_steps: int = 200,
) -> EpisodeResult:
    """Collect a single episode using JAX scan (JIT-compiled)."""
    key, reset_key = jax.random.split(key)
    initial_env_state = env_reset(reset_key)
    
    def step_fn(carry, key):
        env_state, done = carry
        
        # Sample action from policy
        obs_batch = env_state.state[None, :]
        mean, std = policy(obs_batch)
        eps = jax.random.normal(key, mean.shape)
        action = (mean + std * eps)[0]
        
        # Step environment
        result = env_step(env_state, action)
        
        # Mask updates if already done
        new_env_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, old, new),
            result.env_state, env_state
        )
        new_done = jnp.maximum(done, result.done)
        
        # Output for this step (current state, not next)
        output = (env_state.state, action, result.reward * (1.0 - done))
        
        return (new_env_state, new_done), output
    
    # Generate keys for all steps
    keys = jax.random.split(key, max_steps)
    
    # Run scan
    (final_env_state, final_done), (states, actions, rewards) = jax.lax.scan(
        step_fn, (initial_env_state, jnp.array(0.0)), keys
    )
    
    # Compute returns
    returns = compute_returns(rewards)
    
    return EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=jnp.sum(rewards),
    )


@eqx.filter_jit
def collect_episodes(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    key: Array,
    n_episodes: int,
    max_steps: int = 200,
) -> EpisodeResult:
    """Collect multiple episodes in parallel using vmap."""
    keys = jax.random.split(key, n_episodes)
    
    # vmap over episodes
    vmapped_collect = jax.vmap(
        lambda k: collect_episode(policy, env_step, env_reset, k, max_steps),
        in_axes=0
    )
    
    results = vmapped_collect(keys)
    
    # Flatten batch dimension for training
    states = results.states.reshape(-1, 2)
    actions = results.actions.reshape(-1, 1)
    rewards = results.rewards.reshape(-1)
    returns = results.returns.reshape(-1)
    
    # Keep per-episode totals
    total_rewards = jax.vmap(jnp.sum)(results.rewards)
    
    return EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=jnp.mean(total_rewards),  # Mean across episodes
    )


@nnx.jit
def train_step(
    policy,  # nnx.Module
    optimizer,  # nnx.Optimizer
    batch_states: Float[Array, "batch obs_dim"],
    batch_actions: Float[Array, "batch act_dim"],
    batch_advantages: Float[Array, "batch"],
) -> Float[Array, ""]:

    def loss_fn(policy):
        mean, std = policy(batch_states)

        log_probs = gaussian_log_prob(batch_actions, mean, std)

        # REINFORCE loss
        return -jnp.mean(log_probs * batch_advantages)

    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(grads)

    return loss


def train(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    n_iterations: int = 100,
    episodes_per_iter: int = 10,
    learning_rate: float = 1e-3,
    use_baseline: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> dict:

    # Initialize
    key = jax.random.PRNGKey(seed)
    optimizer = nnx.Optimizer(policy, optax.adam(learning_rate))
    train_state = TrainState(
        step=0, key=key, baseline=BaselineState(mean=jnp.array(0.0), n_samples=0)
    )

    # Tracking
    metrics = {
        "iteration": [],
        "mean_return": [],
        "std_return": [],
        "loss": [],
        "baseline_value": [],
    }

    for i in range(n_iterations):
        # Collect episodes
        episode_results = []
        for _ in range(episodes_per_iter):
            train_state = train_state._replace(key=jax.random.split(train_state.key)[0])
            episode = collect_episode(policy, env_step, env_reset, train_state.key)
            episode_results.append(episode)

        # Aggregate data
        all_states = jnp.concatenate([ep.states for ep in episode_results])
        all_actions = jnp.concatenate([ep.actions for ep in episode_results])
        all_returns = jnp.concatenate([ep.returns for ep in episode_results])

        # Compute advantages
        if use_baseline:
            advantages = compute_advantages(all_returns, train_state.baseline.mean)
            new_baseline = update_baseline(train_state.baseline, all_returns)
            train_state = train_state._replace(baseline=new_baseline)
        else:
            advantages = all_returns

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Update policy
        loss = train_step(policy, optimizer, all_states, all_actions, advantages)

        # Track metrics
        episode_returns = jnp.array([ep.total_reward for ep in episode_results])
        metrics["iteration"].append(i)
        metrics["mean_return"].append(float(jnp.mean(episode_returns)))
        metrics["std_return"].append(float(jnp.std(episode_returns)))
        metrics["loss"].append(float(loss))
        metrics["baseline_value"].append(float(train_state.baseline.mean))

        if verbose and i % 10 == 0:
            print(
                f"Iter {i:3d} | Return: {metrics['mean_return'][-1]:7.2f} ± {metrics['std_return'][-1]:5.2f} | Loss: {metrics['loss'][-1]:7.4f}"
            )

    return metrics
