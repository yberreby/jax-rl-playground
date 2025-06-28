"""Test hyperparameter search for policy learning using ask-and-tell with vmap."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
import pandas as pd
from src.policy_nnx import GaussianPolicy
from src.hypersearch import create_study, suggest_optimizer_params, suggest_architecture_params
import pytest


@nnx.jit
def train_step(policy, optimizer, obs, targets):
    """Single training step."""
    def loss_fn(policy):
        mean, _ = policy(obs)
        return jnp.mean((mean - targets) ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(grads)
    return loss


def train_policy_with_params(
    lr: float, 
    hidden_dim: int,
    optimizer_name: str,
    use_layernorm: bool,
    seed: int
) -> float:
    """Train a policy with given hyperparameters and return final loss."""
    key = jax.random.PRNGKey(seed)
    
    # Create policy
    policy = GaussianPolicy(obs_dim=4, action_dim=2, hidden_dim=hidden_dim, rngs=nnx.Rngs(key))
    
    # Disable LayerNorm if requested (hacky but works)
    if not use_layernorm:
        policy.layer_norm = lambda x: x  # Identity function
    
    # Select optimizer
    if optimizer_name == "adam":
        opt = optax.adam(lr)
    elif optimizer_name == "sgd":
        opt = optax.sgd(lr)
    else:  # rmsprop
        opt = optax.rmsprop(lr)
    
    optimizer = nnx.Optimizer(policy, opt)
    
    # Training task: learn bimodal target distribution
    batch_size = 64
    n_steps = 100
    
    obs = jnp.ones((batch_size, 4))
    # Bimodal targets - half at (2, -1), half at (-2, 1)
    target1 = jnp.array([[2.0, -1.0]])
    target2 = jnp.array([[-2.0, 1.0]])
    targets = jnp.concatenate([
        jnp.tile(target1, (batch_size // 2, 1)),
        jnp.tile(target2, (batch_size // 2, 1))
    ])
    
    final_loss = jnp.inf
    for _ in range(n_steps):
        loss = train_step(policy, optimizer, obs, targets)
        final_loss = float(loss)
    
    return final_loss


@pytest.mark.slow
def test_policy_hypersearch():
    """Test hyperparameter search for policy learning."""
    study = create_study("policy_hypersearch", direction="minimize")
    
    # Run 32 trials using ask-and-tell
    n_trials = 32
    batch_size = 8
    
    all_results = []
    
    for batch_start in range(0, n_trials, batch_size):
        # Ask for batch of trials
        current_batch = min(batch_size, n_trials - batch_start)
        trials = [study.ask() for _ in range(current_batch)]
        
        # Collect hyperparameters
        results_batch = []
        for i, trial in enumerate(trials):
            # Get suggestions
            opt_params = suggest_optimizer_params(trial)
            arch_params = suggest_architecture_params(trial)
            
            # Train
            loss = train_policy_with_params(
                lr=opt_params["learning_rate"],
                hidden_dim=arch_params["hidden_dim"],
                optimizer_name=opt_params["optimizer"],
                use_layernorm=arch_params["use_layernorm"],
                seed=batch_start + i
            )
            
            # Record results
            results_batch.append({
                "trial": trial.number,
                "lr": opt_params["learning_rate"],
                "hidden_dim": arch_params["hidden_dim"],
                "optimizer": opt_params["optimizer"],
                "use_layernorm": arch_params["use_layernorm"],
                "loss": loss
            })
            
            # Tell result
            study.tell(trial, loss)
        
        all_results.extend(results_batch)
        print(f"Completed {batch_start + current_batch}/{n_trials} trials")
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv("tests/outputs/policy_hypersearch_results.csv", index=False)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss vs learning rate by optimizer
    for opt in df["optimizer"].unique():
        opt_df = df[df["optimizer"] == opt]
        axes[0, 0].scatter(opt_df["lr"], opt_df["loss"], label=opt, alpha=0.6)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Learning Rate")
    axes[0, 0].set_ylabel("Final Loss")
    axes[0, 0].set_title("Loss vs Learning Rate by Optimizer")
    axes[0, 0].legend()
    
    # Loss vs hidden dim
    axes[0, 1].scatter(df["hidden_dim"], df["loss"], alpha=0.6)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Hidden Dimension")
    axes[0, 1].set_ylabel("Final Loss")
    axes[0, 1].set_title("Loss vs Hidden Dimension")
    
    # LayerNorm effect
    ln_groups = df.groupby("use_layernorm")["loss"].agg(["mean", "std", "min"])
    x = [0, 1]
    axes[1, 0].bar(x, ln_groups["mean"], yerr=ln_groups["std"], 
                   tick_label=["No LayerNorm", "With LayerNorm"])
    axes[1, 0].set_ylabel("Mean Loss")
    axes[1, 0].set_title("Effect of LayerNorm")
    axes[1, 0].set_yscale("log")
    
    # Best trials over time
    best_so_far = []
    for i in range(len(df)):
        best_so_far.append(df.iloc[:i+1]["loss"].min())
    axes[1, 1].plot(best_so_far)
    axes[1, 1].set_xlabel("Trial Number")
    axes[1, 1].set_ylabel("Best Loss So Far")
    axes[1, 1].set_title("Optimization Progress")
    axes[1, 1].set_yscale("log")
    
    plt.tight_layout()
    plt.savefig("tests/outputs/policy_hypersearch_analysis.png", dpi=150)
    plt.close()
    
    # Print summary
    print("\nBest hyperparameters found:")
    best_trial = df.loc[df["loss"].idxmin()]
    for col in ["lr", "hidden_dim", "optimizer", "use_layernorm", "loss"]:
        print(f"  {col}: {best_trial[col]}")
    
    print("\nLayerNorm comparison:")
    print(ln_groups)


if __name__ == "__main__":
    test_policy_hypersearch()