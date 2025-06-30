#!/usr/bin/env python3
"""Train a pendulum swing-up policy using REINFORCE with features."""

import argparse
import json
from pathlib import Path
from flax import nnx

from src.policy import GaussianPolicy
from src.pendulum import step, reset_env, MAX_EPISODE_STEPS
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train pendulum swing-up")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1024, help="Episodes per iteration")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="pendulum_results", help="Output directory")
    parser.add_argument("--use-critic", action="store_true", default=True, help="Use value function")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("Training pendulum swing-up...")
    print(f"Config: LR={args.lr}, Batch={args.batch_size}, Hidden={args.hidden_dim}x{args.n_hidden}")
    print(f"Episodes are {MAX_EPISODE_STEPS} steps long")
    
    # Initialize policy with features
    policy = GaussianPolicy(
        obs_dim=2,
        action_dim=1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden
    )
    
    # Train
    metrics = train(
        policy,
        step,
        reset_env,
        n_iterations=args.iterations,
        episodes_per_iter=args.batch_size,
        learning_rate=args.lr,
        use_critic=args.use_critic,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save final policy
    policy_state = nnx.state(policy)
    with open(output_dir / "policy.pkl", "wb") as f:
        import pickle
        pickle.dump(policy_state, f)
    
    print(f"\nTraining complete! Results saved to {output_dir}")
    print(f"Final return: {metrics['mean_return'][-1]:.3f}")


if __name__ == "__main__":
    main()