import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
from flax import nnx
import optax
import equinox as eqx
from ..policy import sample_actions
from ..baseline import BaselineState, update_baseline, compute_advantages
from ..advantage_normalizer import (
    RunningStats,
    update_running_stats,
    normalize_advantages,
    init_running_stats,
)
from ..critic import ValueFunction, update_critic, compute_critic_advantages


class TrainState(NamedTuple):
    step: int
    key: Array
    baseline: BaselineState
    advantage_stats: RunningStats


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
    max_steps: int = 400,
) -> EpisodeResult:
    key, reset_key = jax.random.split(key)
    initial_env_state = env_reset(reset_key)

    def step_fn(carry, key):
        env_state, done = carry

        # Sample action from policy (without adding batch dimension)
        # We'll handle the unbatched case in sample_actions
        action, log_prob = sample_actions(policy, env_state.state, key)

        # Step environment
        result = env_step(env_state, action)

        # Mask updates if already done
        new_env_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, old, new), result.env_state, env_state
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
    max_steps: int = 400,
) -> EpisodeResult:
    keys = jax.random.split(key, n_episodes)

    # vmap over episodes
    vmapped_collect = jax.vmap(
        lambda k: collect_episode(policy, env_step, env_reset, k, max_steps), in_axes=0
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
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    def loss_fn(policy):
        # Use policy's log_prob method
        log_probs = policy.log_prob(batch_states, batch_actions)

        # REINFORCE loss
        return -jnp.mean(log_probs * batch_advantages)

    loss, grads = nnx.value_and_grad(loss_fn)(policy)

    # Compute gradient statistics before update
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norms = [
        jnp.linalg.norm(g.value if hasattr(g, "value") else g) for g in grad_leaves
    ]
    total_grad_norm = jnp.sqrt(sum(n**2 for n in grad_norms))
    grad_variance = jnp.var(
        jnp.concatenate(
            [
                g.value.flatten() if hasattr(g, "value") else g.flatten()
                for g in grad_leaves
            ]
        )
    )

    optimizer.update(grads)

    return loss, total_grad_norm, grad_variance


def train(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    n_iterations: int = 100,
    episodes_per_iter: int = 10,
    learning_rate: float = 1e-3,
    use_baseline: bool = True,
    use_critic: bool = False,
    seed: int = 0,
    verbose: bool = True,
    burn_in_iterations: int = 5,
) -> dict:
    # Initialize
    key = jax.random.PRNGKey(seed)
    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
            optax.adam(learning_rate),
        ),
    )

    # Initialize critic if requested
    if use_critic:
        critic = ValueFunction(obs_dim=2, hidden_dim=64)  # Pendulum-specific
        critic_optimizer = nnx.Optimizer(
            critic, optax.adam(learning_rate * 2.0)
        )  # Faster critic learning
    else:
        critic = None
        critic_optimizer = None
    train_state = TrainState(
        step=0,
        key=key,
        baseline=BaselineState(mean=jnp.array(0.0), count=0),
        advantage_stats=init_running_stats(),
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
        "grad_norm": [],
        "grad_variance": [],
        "raw_advantage_std": [],
        "episode_length": [],
        "grad_norm_clipped": [],
        "max_action": [],
        "min_action": [],
        "policy_mean_norm": [],
        "policy_std_mean": [],
        "returns_mean": [],
        "returns_std": [],
    }

    # Burn-in phase: collect data to initialize advantage statistics
    if verbose and burn_in_iterations > 0:
        print(f"=== Burn-in phase: {burn_in_iterations} iterations ===")

    for burn_i in range(burn_in_iterations):
        # Collect episodes
        key, collect_key = jax.random.split(train_state.key)
        train_state = train_state._replace(key=key)

        episode_batch = collect_episodes(
            policy, env_step, env_reset, collect_key, episodes_per_iter
        )

        all_returns = episode_batch.returns

        # Compute advantages
        if use_baseline:
            advantages = compute_advantages(all_returns, train_state.baseline.mean)
            new_baseline = update_baseline(train_state.baseline, all_returns)
            train_state = train_state._replace(baseline=new_baseline)
        else:
            advantages = all_returns

        # Update running statistics only (no learning)
        new_adv_stats = update_running_stats(train_state.advantage_stats, advantages)
        train_state = train_state._replace(advantage_stats=new_adv_stats)

        if verbose:
            print(
                f"Burn-in {burn_i}: mean={float(new_adv_stats.mean):.2f}, "
                f"std={float(jnp.sqrt(new_adv_stats.var)):.2f}, "
                f"count={float(new_adv_stats.count)}"
            )

    if verbose and burn_in_iterations > 0:
        print(f"\n=== Training phase: {n_iterations} iterations ===")

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
        if use_critic:
            # Train critic to predict returns
            assert critic is not None and critic_optimizer is not None
            update_critic(critic, critic_optimizer, all_states, all_returns)
            # Use critic as baseline
            advantages = compute_critic_advantages(critic, all_states, all_returns)
        elif use_baseline:
            advantages = compute_advantages(all_returns, train_state.baseline.mean)
            new_baseline = update_baseline(train_state.baseline, all_returns)
            train_state = train_state._replace(baseline=new_baseline)
        else:
            advantages = all_returns

        # Store raw advantage std before normalization
        raw_adv_std = jnp.std(advantages)

        # Update running statistics
        new_adv_stats = update_running_stats(train_state.advantage_stats, advantages)
        train_state = train_state._replace(advantage_stats=new_adv_stats)

        # Normalize advantages using stable statistics (no centering!)
        advantages = normalize_advantages(advantages, new_adv_stats)

        # Update policy
        loss, grad_norm, grad_var = train_step(
            policy, optimizer, all_states, all_actions, advantages
        )

        # Get log probs from the episode data
        log_probs = episode_batch.log_probs

        # Compute episode length (for debugging)
        episode_length = jnp.sum(episode_batch.rewards != 0.0) / episodes_per_iter

        # Track metrics (batched device transfer for efficiency)
        metric_array = jnp.array([
            episode_batch.total_reward,
            loss,
            train_state.baseline.mean / 400.0,  # Divide by episode length
            jnp.mean(advantages),
            jnp.std(advantages),
            jnp.mean(log_probs),
            jnp.mean(episode_batch.actions),
            jnp.std(episode_batch.actions),
            grad_norm,
            grad_var,
            raw_adv_std,
        ])
        metric_values = [float(x) for x in metric_array]
        
        metrics["iteration"].append(i)
        metrics["mean_return"].append(metric_values[0])  # This is average per step
        metrics["std_return"].append(0.0)  # TODO: track individual episode returns
        metrics["loss"].append(metric_values[1])
        metrics["baseline_value"].append(metric_values[2])
        metrics["mean_advantage"].append(metric_values[3])
        metrics["std_advantage"].append(metric_values[4])
        metrics["mean_log_prob"].append(metric_values[5])
        metrics["mean_action"].append(metric_values[6])
        metrics["std_action"].append(metric_values[7])
        metrics["grad_norm"].append(metric_values[8])
        metrics["grad_variance"].append(metric_values[9])
        metrics["raw_advantage_std"].append(metric_values[10])
        # Additional metrics (batched)
        test_states = jnp.zeros((1, 2))  # Test at origin
        test_mean, test_std = policy(test_states)
        
        additional_metrics = jnp.array([
            episode_length,
            jnp.minimum(grad_norm, 1.0),  # After clipping
            jnp.max(episode_batch.actions),
            jnp.min(episode_batch.actions),
            jnp.linalg.norm(test_mean),
            jnp.mean(test_std),
            jnp.mean(all_returns),
            jnp.std(all_returns),
        ])
        additional_values = [float(x) for x in additional_metrics]
        
        metrics["episode_length"].append(additional_values[0])
        metrics["grad_norm_clipped"].append(additional_values[1])
        metrics["max_action"].append(additional_values[2])
        metrics["min_action"].append(additional_values[3])
        metrics["policy_mean_norm"].append(additional_values[4])
        metrics["policy_std_mean"].append(additional_values[5])
        metrics["returns_mean"].append(additional_values[6])
        metrics["returns_std"].append(additional_values[7])

        if verbose and i % 10 == 0:
            print(
                f"Iter {i:3d} | Return: {metrics['mean_return'][-1]:6.3f} | "
                f"Loss: {metrics['loss'][-1]:10.2f} | "
                f"GradNorm: {metrics['grad_norm'][-1]:8.2f} | "
                f"RawAdvStd: {metrics['raw_advantage_std'][-1]:6.2f}"
            )

    return metrics
