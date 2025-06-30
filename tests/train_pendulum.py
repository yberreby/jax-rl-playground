#!/usr/bin/env python3
"""Main entry point for pendulum training with visualization."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from flax import nnx

from src.policy import GaussianPolicy
from src.pendulum import step, reset_env, MAX_EPISODE_STEPS
from src.train import train
from src.viz.pendulum import PendulumVisualizer
from src.pendulum.features import compute_features
import matplotlib.pyplot as plt
import numpy as np


def generate_video(policy, output_path: str, n_episodes: int = 1):
    """Generate video of policy execution."""
    import jax
    import jax.numpy as jnp
    from src.pendulum import EnvState
    
    visualizer = PendulumVisualizer()
    
    all_states = []
    all_actions = []
    all_rewards = []
    
    for ep in range(n_episodes):
        key = jax.random.PRNGKey(ep)
        # For evaluation, always start at bottom
        initial_state = jnp.array([0.0, 0.0])  # theta=0 (down), theta_dot=0
        env_state = EnvState(state=initial_state, step_count=0, key=key)
        
        states = [env_state.state]
        actions = []
        rewards = []
        
        for _ in range(MAX_EPISODE_STEPS):
            # Compute action
            features = compute_features(env_state.state)
            mean, _ = policy(features[None, :])
            action = mean[0]  # Shape [1]
            
            # Step environment
            result = step(env_state, action)
            
            states.append(result.env_state.state)
            actions.append(action)
            rewards.append(result.reward)
            
            env_state = result.env_state
            
            if result.done:
                break
        
        # Add episode separator
        if ep < n_episodes - 1:
            for _ in range(20):  # Pause between episodes
                states.append(states[-1])
                actions.append(actions[-1])
                rewards.append(0.0)
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
    
    # Create animation
    visualizer.create_animation(
        states=all_states,
        actions=all_actions,
        rewards=all_rewards,
        filename=output_path,
        fps=30,
    )
    return True


def generate_plots(metrics: dict, output_dir: Path):
    """Generate comprehensive diagnostic plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Returns over time
    plt.subplot(3, 3, 1)
    plt.plot(metrics['iteration'], metrics['mean_return'], 'b-', linewidth=2)
    plt.title('Mean Return Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Return')
    plt.grid(True, alpha=0.3)
    
    # 2. Loss
    plt.subplot(3, 3, 2)
    plt.plot(metrics['iteration'], metrics['loss'], 'r-', linewidth=2)
    plt.title('Policy Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('symlog')  # Handle negative losses
    
    # 3. Gradient norm
    plt.subplot(3, 3, 3)
    plt.plot(metrics['iteration'], metrics['grad_norm'], 'g-', linewidth=2)
    plt.title('Policy Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    
    # 4. Action statistics
    plt.subplot(3, 3, 4)
    plt.plot(metrics['iteration'], metrics['mean_action'], label='Mean')
    plt.plot(metrics['iteration'], metrics['std_action'], label='Std')
    plt.fill_between(metrics['iteration'], 
                     np.array(metrics['min_action']), 
                     np.array(metrics['max_action']), 
                     alpha=0.3, label='Min/Max')
    plt.title('Action Statistics')
    plt.xlabel('Iteration')
    plt.ylabel('Action Value')
    plt.legend()
    
    # 5. Policy parameters
    plt.subplot(3, 3, 5)
    plt.plot(metrics['iteration'], metrics['policy_mean_norm'], label='Mean Norm')
    plt.plot(metrics['iteration'], metrics['policy_std_mean'], label='Std Mean')
    plt.title('Policy Parameters')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.yscale('log')
    
    # 6. Entropy
    plt.subplot(3, 3, 6)
    plt.plot(metrics['iteration'], metrics['entropy'], 'purple', linewidth=2)
    plt.title('Policy Entropy')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    
    # 7. Learning rate
    plt.subplot(3, 3, 7)
    plt.plot(metrics['iteration'], metrics['learning_rate'], 'orange', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    # 8. Advantage statistics
    plt.subplot(3, 3, 8)
    plt.plot(metrics['iteration'], metrics['raw_advantage_std'], label='Raw Std')
    plt.plot(metrics['iteration'], metrics['baseline_error'], label='Baseline Error')
    plt.title('Advantage Statistics')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    
    # 9. Explained variance & saturation
    plt.subplot(3, 3, 9)
    plt.plot(metrics['iteration'], metrics['explained_variance'], label='Explained Var')
    plt.plot(metrics['iteration'], metrics['action_saturation'], label='Action Saturation')
    plt.title('Baseline & Action Quality')
    plt.xlabel('Iteration')
    plt.ylabel('Fraction')
    plt.legend()
    plt.ylim(-0.5, 1.0)
    
    # Additional critic plot if using critic
    if 'critic_loss' in metrics and metrics['critic_loss'][0] != 0:
        fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        ax1.plot(metrics['iteration'], metrics['critic_loss'], 'darkred', linewidth=2)
        ax1.set_title('Critic Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(metrics['iteration'], metrics['critic_mean_pred'], label='Mean Pred')
        ax2.plot(metrics['iteration'], metrics['critic_std_pred'], label='Std Pred')
        ax2.set_title('Critic Predictions')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(metrics['iteration'], metrics['critic_grad_norm'], 'darkgreen', linewidth=2)
        ax3.set_title('Critic Gradient Norm')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "critic_diagnostics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional focused plots
    # 1. Returns with smoothing
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    returns = np.array(metrics['mean_return'])
    iters = np.array(metrics['iteration'])
    plt.plot(iters, returns, 'b-', alpha=0.3, label='Raw')
    # Moving average
    window = min(50, len(returns) // 10)
    if window > 1:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        plt.plot(iters[window-1:], smoothed, 'b-', linewidth=2, label=f'MA({window})')
    plt.title('Training Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "returns_detailed.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train pendulum swing-up with REINFORCE")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="Episodes per iteration")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-critic", action="store_true", help="Use value function baseline")
    parser.add_argument("--b1", type=float, default=0.0, help="Adam beta1 parameter")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2 parameter")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps for learning rate")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--quiet", action="store_true", help="Suppress training progress output")
    args = parser.parse_args()
    
    # Create output directory
    if args.name:
        exp_name = args.name
    else:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path("tests/outputs/pendulum") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training pendulum swing-up experiment: {exp_name}")
    print(f"Output directory: {output_dir}")
    print(f"Config: LR={args.lr}, Batch={args.batch_size}, Hidden={args.hidden_dim}x{args.n_hidden}")
    print(f"Adam: b1={args.b1}, b2={args.b2}, Warmup={args.warmup}")
    print(f"Episodes are {MAX_EPISODE_STEPS} steps long")
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize policy
    policy = GaussianPolicy(
        obs_dim=8,  # Using pendulum features
        action_dim=1,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
        use_layernorm=True,
    )
    
    # Generate before-training video
    if not args.no_video:
        print("\nGenerating before-training video...")
        before_video_path = output_dir / "policy_before.mp4"
        if generate_video(policy, str(before_video_path), n_episodes=1):
            print(f"Saved before-training video to {before_video_path}")
    
    # Train
    print("\nTraining...")
    metrics = train(
        policy,
        step,
        reset_env,
        n_iterations=args.iterations,
        episodes_per_iter=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        use_baseline=True,
        use_critic=args.use_critic,
        seed=args.seed,
        verbose=not args.quiet,
        adam_b1=args.b1,
        adam_b2=args.b2,
    )
    
    # Save results
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate diagnostic plots
    generate_plots(metrics, output_dir)
    
    # Save final policy
    policy_state = nnx.state(policy)
    with open(output_dir / "policy.pkl", "wb") as f:
        import pickle
        pickle.dump(policy_state, f)
    
    # Generate video
    if not args.no_video:
        print("\nGenerating video...")
        video_path = output_dir / "policy.mp4"
        if generate_video(policy, str(video_path), n_episodes=1):
            print(f"Saved video to {video_path}")
    
    # Print summary
    print("\nTraining complete!")
    print(f"Final return: {metrics['mean_return'][-1]:.3f}")
    print(f"Best return: {max(metrics['mean_return']):.3f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()