#!/usr/bin/env python3
"""Simple example: train a pendulum swing-up policy using REINFORCE.

For full training with video generation, use tests/train_pendulum.py
"""

import json
from pathlib import Path

from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import train


def main():
    # Simple configuration
    output_dir = Path("pendulum_example")
    output_dir.mkdir(exist_ok=True)
    
    print("Training pendulum swing-up...")
    
    # Initialize policy with 8D features
    policy = GaussianPolicy(
        obs_dim=8,
        action_dim=1,
        hidden_dim=128,
        n_hidden_layers=2,
        use_layernorm=True,
    )
    
    # Train with default settings
    metrics = train(
        policy,
        step,
        reset_env,
        n_iterations=100,
        episodes_per_iter=512,
        learning_rate=3e-4,
        use_baseline=True,
        verbose=True,
    )
    
    # Save results
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Final return: {metrics['mean_return'][-1]:.3f}")
    print(f"Results saved to {output_dir}")
    print("\nFor full training with video generation, use:")
    print("  python tests/train_pendulum.py --iterations 500")


if __name__ == "__main__":
    main()