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


def generate_video(policy, output_path: str, n_episodes: int = 2):
    """Generate video of policy execution."""
    import jax
    
    visualizer = PendulumVisualizer()
    
    all_states = []
    all_actions = []
    all_rewards = []
    
    for ep in range(n_episodes):
        key = jax.random.PRNGKey(ep)
        env_state = reset_env(key)
        
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


def main():
    parser = argparse.ArgumentParser(description="Train pendulum swing-up with REINFORCE")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1024, help="Episodes per iteration")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-critic", action="store_true", help="Use value function baseline")
    parser.add_argument("--b1", type=float, default=0.0, help="Adam beta1 parameter")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2 parameter")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps for learning rate")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()
    
    # Create output directory
    if args.name:
        exp_name = args.name
    else:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path("tests/outputs") / f"pendulum_{exp_name}"
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
        verbose=args.verbose,
        adam_b1=args.b1,
        adam_b2=args.b2,
    )
    
    # Save results
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save final policy
    policy_state = nnx.state(policy)
    with open(output_dir / "policy.pkl", "wb") as f:
        import pickle
        pickle.dump(policy_state, f)
    
    # Generate video
    if not args.no_video:
        print("\nGenerating video...")
        video_path = output_dir / "policy.mp4"
        if generate_video(policy, str(video_path), n_episodes=2):
            print(f"Saved video to {video_path}")
    
    # Print summary
    print("\nTraining complete!")
    print(f"Final return: {metrics['mean_return'][-1]:.3f}")
    print(f"Best return: {max(metrics['mean_return']):.3f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()