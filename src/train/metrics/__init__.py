"""Metrics tracking for training loop."""

import jax.numpy as jnp
from jaxtyping import Array, Float


class MetricsTracker:
    """Track training metrics efficiently."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {
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
            "learning_rate": [],
            "entropy": [],  # Policy entropy
            "kl_divergence": [],  # KL from initial policy
        }
    
    def update(
        self,
        iteration: int,
        episode_batch,  # EpisodeResult
        loss: Float[Array, ""],
        grad_norm: Float[Array, ""],
        grad_variance: Float[Array, ""],
        normalized_advantages: Float[Array, "batch"],
        raw_advantage_std: Float[Array, ""],
        baseline_mean: Float[Array, ""],
        policy_test_mean: Float[Array, "1 act_dim"],
        policy_test_std: Float[Array, "1 act_dim"],
        episodes_per_iter: int,
        learning_rate: float,
        entropy: Float[Array, ""],
    ) -> None:
        """Update metrics with new values."""
        # Compute episode length
        episode_length = jnp.sum(episode_batch.rewards != 0.0) / episodes_per_iter
        
        # Batch metric computation for efficiency
        metric_values = {
            "iteration": float(iteration),
            "mean_return": float(episode_batch.total_reward),
            "std_return": 0.0,  # TODO: track individual episode returns
            "loss": float(loss),
            "baseline_value": float(baseline_mean / 400.0),  # Normalized by max episode length
            "mean_advantage": float(jnp.mean(normalized_advantages)),
            "std_advantage": float(jnp.std(normalized_advantages)),
            "mean_log_prob": float(jnp.mean(episode_batch.log_probs)),
            "mean_action": float(jnp.mean(episode_batch.actions)),
            "std_action": float(jnp.std(episode_batch.actions)),
            "grad_norm": float(grad_norm),
            "grad_variance": float(grad_variance),
            "raw_advantage_std": float(raw_advantage_std),
            "episode_length": float(episode_length),
            "grad_norm_clipped": float(jnp.minimum(grad_norm, 1.0)),
            "max_action": float(jnp.max(episode_batch.actions)),
            "min_action": float(jnp.min(episode_batch.actions)),
            "policy_mean_norm": float(jnp.linalg.norm(policy_test_mean)),
            "policy_std_mean": float(jnp.mean(policy_test_std)),
            "returns_mean": float(jnp.mean(episode_batch.returns)),
            "returns_std": float(jnp.std(episode_batch.returns)),
            "learning_rate": float(learning_rate) if learning_rate is not None else 0.0,
            "entropy": float(entropy) if entropy is not None else 0.0,
            "kl_divergence": 0.0,  # TODO: implement KL tracking
        }
        
        # Update all metrics
        for key, value in metric_values.items():
            self.metrics[key].append(value)
    
    def log_iteration(self, iteration: int, verbose: bool = True) -> None:
        """Log current iteration metrics."""
        if verbose and iteration % 10 == 0:
            print(
                f"Iter {iteration:4d} | "
                f"Return: {self.metrics['mean_return'][-1]:6.3f} | "
                f"Loss: {self.metrics['loss'][-1]:10.2f} | "
                f"GradNorm: {self.metrics['grad_norm'][-1]:6.2f} | "
                f"Entropy: {self.metrics['entropy'][-1]:6.2f} | "
                f"LR: {self.metrics['learning_rate'][-1]:.2e} | "
                f"AdvStd: {self.metrics['raw_advantage_std'][-1]:6.1f}"
            )
            
            # Extra diagnostic every 50 iterations
            if iteration % 50 == 0 and iteration > 0:
                print(
                    f"  └─ Actions: [{self.metrics['min_action'][-1]:5.2f}, {self.metrics['max_action'][-1]:5.2f}] "
                    f"(μ={self.metrics['mean_action'][-1]:5.2f}, σ={self.metrics['std_action'][-1]:5.2f}) | "
                    f"Policy σ: {self.metrics['policy_std_mean'][-1]:5.3f}"
                )
    
    def to_dict(self) -> dict[str, list[float]]:
        """Return metrics as dictionary."""
        return self.metrics