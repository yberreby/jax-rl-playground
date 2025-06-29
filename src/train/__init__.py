import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
from flax import nnx
import optax
import equinox as eqx
from ..policy import sample_actions
from ..baseline import BaselineState, update_baseline, compute_advantages


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
    log_probs: Float[Array, "episode_len"]


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
    key, reset_key = jax.random.split(key)
    initial_env_state = env_reset(reset_key)
    
    def step_fn(carry, key):
        env_state, done = carry
        
        # Sample action from policy
        action, log_prob = sample_actions(
            policy, env_state.state[None, :], key
        )
        action = action[0]  # Remove batch dimension
        log_prob = log_prob[0]  # Remove batch dimension
        
        # Step environment
        result = env_step(env_state, action)
        
        # Mask updates if already done
        new_env_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, old, new),
            result.env_state, env_state
        )
        new_done = jnp.maximum(done, result.done)
        
        # Output for this step (current state, not next)
        output = (env_state.state, action, result.reward * (1.0 - done), log_prob)
        
        return (new_env_state, new_done), output
    
    # Generate keys for all steps
    keys = jax.random.split(key, max_steps)
    
    # Run scan
    (final_env_state, final_done), (states, actions, rewards, log_probs) = jax.lax.scan(
        step_fn, (initial_env_state, jnp.array(0.0)), keys
    )
    
    # Compute returns
    returns = compute_returns(rewards)
    
    # Count actual steps
    episode_length = jnp.sum(rewards != 0.0)
    
    return EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=jnp.sum(rewards) / jnp.maximum(episode_length, 1.0),  # Average
        log_probs=log_probs,
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
    keys = jax.random.split(key, n_episodes)
    
    # vmap over episodes
    vmapped_collect = jax.vmap(
        lambda k: collect_episode(policy, env_step, env_reset, k, max_steps),
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


@nnx.jit
def train_step(
    policy,  # nnx.Module
    optimizer,  # nnx.Optimizer
    batch_states: Float[Array, "batch obs_dim"],
    batch_actions: Float[Array, "batch act_dim"],
    batch_advantages: Float[Array, "batch"],
) -> Float[Array, ""]:

    def loss_fn(policy):
        # Use policy's log_prob method
        log_probs = policy.log_prob(batch_states, batch_actions)

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
        "mean_advantage": [],
        "std_advantage": [],
        "mean_log_prob": [],
        "mean_action": [],
        "std_action": [],
    }

    for i in range(n_iterations):
        # Collect episodes in parallel
        key, collect_key = jax.random.split(train_state.key)
        train_state = train_state._replace(key=key)
        
        episode_batch = collect_episodes(
            policy, env_step, env_reset, collect_key, episodes_per_iter
        )
        
        # Data is already aggregated
        all_states = episode_batch.states
        all_actions = episode_batch.actions
        all_returns = episode_batch.returns

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
        
        # Get log probs from the episode data
        log_probs = episode_batch.log_probs

        # Track metrics
        metrics["iteration"].append(i)
        metrics["mean_return"].append(float(episode_batch.total_reward))
        metrics["std_return"].append(0.0)  # TODO: track individual episode returns
        metrics["loss"].append(float(loss))
        metrics["baseline_value"].append(float(train_state.baseline.mean))
        metrics["mean_advantage"].append(float(jnp.mean(advantages)))
        metrics["std_advantage"].append(float(jnp.std(advantages)))
        metrics["mean_log_prob"].append(float(jnp.mean(log_probs)))
        metrics["mean_action"].append(float(jnp.mean(episode_batch.actions)))
        metrics["std_action"].append(float(jnp.std(episode_batch.actions)))

        if verbose and i % 10 == 0:
            print(
                f"Iter {i:3d} | Return: {metrics['mean_return'][-1]:7.2f} | "
                f"Loss: {metrics['loss'][-1]:10.4f} | "
                f"Baseline: {metrics['baseline_value'][-1]:7.2f} | "
                f"Adv: {metrics['mean_advantage'][-1]:6.2f}±{metrics['std_advantage'][-1]:5.2f}"
            )

    return metrics
