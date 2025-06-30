import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
from flax import nnx
import optax
from ..baseline import BaselineState, update_baseline, compute_advantages
from ..advantage_normalizer import normalize_advantages
from ..critic import ValueFunction, update_critic, compute_critic_advantages
from .episodes import collect_episodes
from .metrics import MetricsTracker
from .lr_schedule import create_lr_schedule
from ..pendulum.features import compute_features
from functools import partial


class TrainState(NamedTuple):
    step: int
    key: Array
    baseline: BaselineState


@partial(nnx.jit, static_argnames=["entropy_coef"])
def train_step(
    policy,  # nnx.Module
    optimizer,  # nnx.Optimizer
    batch_states: Float[Array, "batch obs_dim"],
    batch_actions: Float[Array, "batch act_dim"],
    batch_advantages: Float[Array, "batch"],
    entropy_coef: float = 0.01,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    def loss_fn(policy):
        # Use policy's log_prob method
        log_probs = policy.log_prob(batch_states, batch_actions)
        
        # Get policy distribution for entropy
        mean, std = policy(batch_states)
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2)
        entropy = jnp.mean(0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2))

        # REINFORCE loss with entropy bonus
        pg_loss = -jnp.mean(log_probs * batch_advantages)
        total_loss = pg_loss - entropy_coef * entropy
        return total_loss, (pg_loss, entropy)

    (loss, (pg_loss, entropy)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy)

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

    return loss, total_grad_norm, grad_variance, entropy


def train(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    n_iterations: int = 100,
    episodes_per_iter: int = 10,
    learning_rate: float = 1e-3,
    warmup_steps: int = 100,
    use_baseline: bool = True,
    use_critic: bool = False,
    seed: int = 0,
    verbose: bool = True,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
) -> dict:
    # Initialize
    key = jax.random.PRNGKey(seed)
    
    # Create learning rate schedule
    lr_schedule = create_lr_schedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=n_iterations,
        end_lr_factor=0.1,
    )
    
    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
            optax.adam(lr_schedule, b1=adam_b1, b2=adam_b2),
        ),
    )

    # Initialize critic if requested
    if use_critic:
        critic = ValueFunction(obs_dim=8, hidden_dim=64)  # Uses same 8D features as policy
        critic_optimizer = nnx.Optimizer(
            critic, optax.adam(learning_rate * 2.0)
        )  # Faster critic learning
        
        # Pretrain critic with initial rollouts
        if verbose:
            print("Pretraining critic with initial rollouts...")
        
        pretrain_episodes = min(episodes_per_iter * 5, 500)  # Collect 5x batch or max 500
        key, pretrain_key = jax.random.split(key)
        
        from ..pendulum import MAX_EPISODE_STEPS
        pretrain_batch = collect_episodes(
            policy, env_step, env_reset, pretrain_key, pretrain_episodes,
            max_steps=MAX_EPISODE_STEPS, reward_scale=0.1  # Use same scale as training
        )
        
        pretrain_features = jax.vmap(compute_features)(pretrain_batch.states)
        pretrain_returns = pretrain_batch.returns
        
        # Train critic for several iterations
        for pretrain_iter in range(50):
            loss, mean_pred, std_pred, grad_norm = update_critic(
                critic, critic_optimizer, pretrain_features, pretrain_returns
            )
            
            if verbose and pretrain_iter % 10 == 0:
                # Compute explained variance
                predictions = critic(pretrain_features)
                ss_tot = jnp.sum((pretrain_returns - jnp.mean(pretrain_returns))**2)
                ss_res = jnp.sum((pretrain_returns - predictions)**2)
                explained_var = 1 - ss_res / (ss_tot + 1e-8)
                
                print(f"  Pretrain iter {pretrain_iter}: loss={loss:.4f}, mean_pred={mean_pred:.3f}, explained_var={explained_var:.3f}")
        
        if verbose:
            print("Critic pretraining complete!\n")
    else:
        critic = None
        critic_optimizer = None
        
    train_state = TrainState(
        step=0,
        key=key,
        baseline=BaselineState(mean=jnp.array(0.0), n_samples=0),
    )

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    for i in range(n_iterations):
        # Collect episodes in parallel
        key, collect_key = jax.random.split(train_state.key)
        train_state = train_state._replace(key=key)

        from ..pendulum import MAX_EPISODE_STEPS
        episode_batch = collect_episodes(
            policy, env_step, env_reset, collect_key, episodes_per_iter,
            max_steps=MAX_EPISODE_STEPS, reward_scale=0.1
        )

        # Data is already aggregated
        all_states = episode_batch.states  # Raw 2D states
        all_actions = episode_batch.actions
        
        # Use per-timestep returns for better credit assignment
        all_returns = episode_batch.returns
        
        # Get episode returns for metrics
        returns_per_episode = episode_batch.returns.reshape(episodes_per_iter, -1)
        episode_returns = returns_per_episode[:, 0]  # Get full episode return
        
        # Convert raw states to features for policy
        all_features = jax.vmap(compute_features)(all_states)
        
        # Debug: show policy means vs actual actions (every 50 iterations)
        if i % 50 == 0 and verbose:
            # Get policy means for first few states
            sample_means, _ = policy(all_features[:5])
            sample_actions = all_actions[:5].flatten()
            print(f"  POLICY DEBUG: means={sample_means.flatten()}, actions={sample_actions}")

        # Compute advantages
        critic_loss = None
        critic_mean_pred = None
        critic_std_pred = None
        critic_grad_norm = None
        
        if use_critic:
            # Train critic to predict returns
            assert critic is not None and critic_optimizer is not None
            
            # Debug: show critic predictions vs actual returns (every 10 iterations)
            if verbose and i % 10 == 0:
                critic_preds = critic(all_features)
                print(f"\n  CRITIC DEBUG (iter {i}):")
                print(f"    First 5 returns: {all_returns[:5]}")
                print(f"    First 5 critic preds: {critic_preds[:5]}")
                print(f"    Prediction errors: {(all_returns[:5] - critic_preds[:5])}")
                print(f"    Returns stats: mean={all_returns.mean():.3f}, std={all_returns.std():.3f}")
                print(f"    Critic stats: mean={critic_preds.mean():.3f}, std={critic_preds.std():.3f}")
            
            critic_loss, critic_mean_pred, critic_std_pred, critic_grad_norm = update_critic(
                critic, critic_optimizer, all_features, all_returns
            )
            # Use critic as baseline
            advantages = compute_critic_advantages(critic, all_features, all_returns)
        else:
            # No baseline - just use raw returns
            advantages = all_returns

        # Store raw advantage std before normalization
        raw_adv_std = jnp.std(advantages)
        
        # Compute explained variance (how well baseline predicts returns)
        if use_critic:
            predictions = critic(all_features)
            ss_tot = jnp.sum((all_returns - jnp.mean(all_returns))**2)
            ss_res = jnp.sum((all_returns - predictions)**2)
            explained_var = 1 - ss_res / (ss_tot + 1e-8)
        else:
            explained_var = jnp.array(0.0)

        # Normalize advantages
        normalized_advantages = normalize_advantages(advantages)

        # Update policy
        loss, grad_norm, grad_var, entropy = train_step(
            policy, optimizer, all_features, all_actions, normalized_advantages, 
            entropy_coef=0.01
        )

        # Test policy at origin for tracking
        test_raw_state = jnp.zeros((1, 2))  # Test at origin (raw state)
        test_features = compute_features(test_raw_state[0])[None, :]
        test_mean, test_std = policy(test_features)
        
        # Get current learning rate
        current_lr = float(lr_schedule(i))
        
        # Compute action saturation (fraction near boundaries)
        from ..pendulum import MAX_TORQUE
        action_magnitudes = jnp.abs(all_actions.flatten())
        saturation_threshold = 0.95 * MAX_TORQUE
        action_saturation = jnp.mean(action_magnitudes > saturation_threshold)
        
        # Update metrics
        metrics_tracker.update(
            iteration=i,
            episode_batch=episode_batch,
            loss=loss,
            grad_norm=grad_norm,
            grad_variance=grad_var,
            normalized_advantages=normalized_advantages,
            raw_advantage_std=raw_adv_std,
            baseline_mean=train_state.baseline.mean,
            policy_test_mean=test_mean,
            policy_test_std=test_std,
            episodes_per_iter=episodes_per_iter,
            learning_rate=current_lr,
            entropy=entropy,
            action_saturation=action_saturation,
            explained_variance=explained_var,
            advantages=advantages,
            episode_returns=episode_returns,
            critic_loss=critic_loss,
            critic_mean_pred=critic_mean_pred,
            critic_std_pred=critic_std_pred,
            critic_grad_norm=critic_grad_norm,
        )
        
        # Log progress
        metrics_tracker.log_iteration(i, verbose=verbose)

    return metrics_tracker.to_dict()