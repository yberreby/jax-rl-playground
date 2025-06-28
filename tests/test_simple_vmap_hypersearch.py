"""Simplified vmap hyperparameter search focusing on continuous parameters."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from src.policy_nnx import GaussianPolicy
from src.hypersearch import create_study
import time
import pandas as pd
import pytest


# Fixed architecture, vary only continuous parameters
FIXED_HIDDEN = 64


@nnx.jit
def train_step(policy, optimizer, obs, targets):
    """Single training step."""

    def loss_fn(policy):
        mean, _ = policy(obs)
        return jnp.mean((mean - targets) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(grads)
    return loss


def train_with_hyperparams(lr: float, sparsity: float, seed: int) -> float:
    """Train with specific hyperparameters."""
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)

    # Create policy
    policy = GaussianPolicy(
        obs_dim=4, action_dim=2, hidden_dim=FIXED_HIDDEN, rngs=nnx.Rngs(key)
    )

    # Override sparsity
    from src.init import sparse_init

    policy.w1 = nnx.Param(sparse_init(key1, (4, FIXED_HIDDEN), sparsity=sparsity))
    policy.w2 = nnx.Param(sparse_init(key2, (FIXED_HIDDEN, 2), sparsity=sparsity))

    optimizer = nnx.Optimizer(policy, optax.adam(lr))

    # Training data
    obs = jnp.ones((32, 4))
    targets = jnp.array([[2.0, -1.0]]) * jnp.ones((32, 2))

    # Train
    n_steps = 100
    for _ in range(n_steps):
        loss = train_step(policy, optimizer, obs, targets)

    return float(loss)


@pytest.mark.slow
def test_simple_vmap_hypersearch():
    """Test vmap hypersearch with continuous parameters only."""
    study = create_study("simple_vmap_test")

    n_trials = 48
    batch_size = 12

    results = []
    total_start = time.time()

    for batch_idx in range(n_trials // batch_size):
        batch_start_time = time.time()

        # Ask for trials
        trials = [study.ask() for _ in range(batch_size)]

        # Get hyperparameters
        hparams = []
        for trial in trials:
            hparams.append(
                {
                    "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                    "sparsity": trial.suggest_float("sparsity", 0.5, 0.95),
                }
            )

        # Run training sequentially but with JIT
        losses = []
        for i, hp in enumerate(hparams):
            loss = train_with_hyperparams(
                hp["lr"], hp["sparsity"], batch_idx * batch_size + i
            )
            losses.append(loss)

        # Record results
        batch_time = time.time() - batch_start_time
        for trial, hp, loss in zip(trials, hparams, losses):
            study.tell(trial, loss)
            results.append(
                {
                    "trial": trial.number,
                    "lr": hp["lr"],
                    "sparsity": hp["sparsity"],
                    "loss": loss,
                    "time_per_trial": batch_time / batch_size,
                }
            )

        print(
            f"Batch {batch_idx + 1}: "
            f"avg loss={sum(losses) / len(losses):.4f}, "
            f"best={min(losses):.4f}, "
            f"time={batch_time:.2f}s ({batch_time / batch_size:.3f}s per trial)"
        )

    total_time = time.time() - total_start

    # Save and analyze
    df = pd.DataFrame(results)
    df.to_csv("tests/outputs/simple_vmap_results.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"Total time: {total_time:.2f}s ({total_time / n_trials:.3f}s per trial)")
    print(f"Best loss: {study.best_value:.6f}")
    print(
        f"Best params: lr={study.best_params['lr']:.6f}, sparsity={study.best_params['sparsity']:.3f}"
    )

    # Analyze by parameter
    print("\n=== PARAMETER ANALYSIS ===")

    # Bin by learning rate
    df["lr_bin"] = pd.cut(df["lr"], bins=5)
    lr_analysis = df.groupby("lr_bin")["loss"].agg(["mean", "std", "min", "count"])
    print("\nLearning rate bins:")
    print(lr_analysis)

    # Bin by sparsity
    df["sparsity_bin"] = pd.cut(df["sparsity"], bins=5)
    sparsity_analysis = df.groupby("sparsity_bin")["loss"].agg(
        ["mean", "std", "min", "count"]
    )
    print("\nSparsity bins:")
    print(sparsity_analysis)

    # Top 10 trials
    print("\n=== TOP 10 TRIALS ===")
    top10 = df.nsmallest(10, "loss")[["trial", "lr", "sparsity", "loss"]]
    print(top10.to_string(index=False))

    # Performance over time
    print("\n=== OPTIMIZATION PROGRESS ===")
    for i in range(0, n_trials, 12):
        batch_df = df[i : i + 12]
        if not batch_df.empty:
            print(
                f"Trials {i}-{i + 11}: "
                f"mean={batch_df['loss'].mean():.4f}, "
                f"best={df[: i + 12]['loss'].min():.4f}"
            )


if __name__ == "__main__":
    test_simple_vmap_hypersearch()
