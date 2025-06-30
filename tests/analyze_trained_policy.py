#!/usr/bin/env python3
"""Analyze a trained policy's behavior."""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_training_results(results_dir: str):
    """Analyze training results and policy behavior."""
    results_dir = Path(results_dir)
    
    # Load metrics
    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)
    
    # Print summary statistics
    print(f"Training Analysis for {results_dir}")
    print("=" * 50)
    
    returns = metrics["mean_return"]
    print(f"Initial return: {returns[0]:.3f}")
    print(f"Final return: {returns[-1]:.3f}")
    print(f"Best return: {max(returns):.3f} (iter {returns.index(max(returns))})")
    print(f"Average return (last 10): {sum(returns[-10:])/10:.3f}")
    
    # Check if learning
    improvement = returns[-1] - returns[0]
    print(f"\nImprovement: {improvement:.3f}")
    
    if improvement > 0.001:
        print("✓ Policy improved!")
    else:
        print("✗ Policy did not improve significantly")
    
    # Plot learning curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Returns
    axes[0, 0].plot(returns)
    axes[0, 0].set_title("Mean Return")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Loss
    axes[0, 1].plot(metrics["loss"])
    axes[0, 1].set_title("Loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_yscale('symlog')
    axes[0, 1].grid(True)
    
    # Gradient norm
    axes[1, 0].plot(metrics["grad_norm"])
    axes[1, 0].set_title("Gradient Norm")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Action statistics
    axes[1, 1].plot(metrics["action_mean"], label="Mean")
    axes[1, 1].fill_between(
        range(len(metrics["action_mean"])),
        [m - s for m, s in zip(metrics["action_mean"], metrics["action_std"])],
        [m + s for m, s in zip(metrics["action_mean"], metrics["action_std"])],
        alpha=0.3
    )
    axes[1, 1].set_title("Action Statistics")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_analysis.png")
    print(f"\nSaved analysis plot to {results_dir}/training_analysis.png")
    
    # Check for specific issues
    print("\nDiagnostics:")
    
    # Check if returns are stuck
    last_20_returns = returns[-20:]
    if max(last_20_returns) - min(last_20_returns) < 0.001:
        print("⚠️  Returns appear stuck (no variation in last 20 iterations)")
    
    # Check if gradients vanished
    if metrics["grad_norm"][-1] < 0.1:
        print("⚠️  Gradient norm very small - possible vanishing gradients")
    
    # Check if actions are saturating
    if abs(metrics["action_mean"][-1]) > 5:
        print("⚠️  Actions may be saturating")
    
    print("\nDone!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_training_results(sys.argv[1])
    else:
        # Default to most recent
        analyze_training_results("../pendulum_results")