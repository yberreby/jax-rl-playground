import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
from flax import nnx
import optax
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


def collect_episode(
    policy,  # nnx.Module but type system doesn't understand callable
    env_step,  # Environment step function
    env_reset,  # Environment reset function  
    key: Array,
    max_steps: int = 200,
) -> EpisodeResult:
    key, reset_key, episode_key = jax.random.split(key, 3)
    env_state = env_reset(reset_key)

    states = []
    actions = []
    rewards = []

    for _ in range(max_steps):
        episode_key, action_key = jax.random.split(episode_key)

        # Sample action from policy
        obs_batch = env_state.state[None, :]
        mean, std = policy(obs_batch)
        eps = jax.random.normal(action_key, mean.shape)
        action = (mean + std * eps)[0]  # Sample and remove batch dim

        # Step environment
        result = env_step(env_state, action)

        states.append(env_state.state)
        actions.append(action)
        rewards.append(result.reward)

        env_state = result.env_state

        if result.done == 1.0:
            break

    states = jnp.stack(states)
    actions = jnp.stack(actions)
    rewards = jnp.stack(rewards)
    returns = compute_returns(rewards)

    return EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=jnp.sum(rewards),
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
