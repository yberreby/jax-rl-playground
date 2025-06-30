# Pendulum Swing-Up with JAX

## Current State

The pendulum swing-up task is implemented with:

1. **Feature computation available** (`src/pendulum/features.py`):
   - Sin/cos encoding of angles (cyclical features)
   - Cartesian coordinates and velocities
   - Kinetic energy normalization
   - 8 features total from 2 raw observations
   - Can be used to preprocess observations before policy

2. **Fixed reward function** (`src/pendulum/__init__.py`):
   - Rewards upright position: `-cos(theta)` gives +1 when up, -1 when down
   - Episodes are 400 steps long (increased from 200)

3. **Fast vectorized training** (`src/train/__init__.py`):
   - Uses `jax.lax.scan` for episode collection
   - `vmap` for parallel episodes
   - JIT compilation throughout

## Quick Start

```bash
# Run tests
uv run pytest tests/test_pendulum.py -xvs

# Train a policy
uv run python examples/train_pendulum.py --iterations 500 --batch-size 2048

# Train with custom settings
uv run python examples/train_pendulum.py \
    --lr 1e-4 \
    --hidden-dim 256 \
    --n-hidden 3 \
    --iterations 1000
```

## File Structure

```
src/
├── pendulum/
│   ├── __init__.py      # Environment dynamics, reward, reset
│   └── features.py      # Feature computation (sin/cos, cartesian, etc.)
├── policy/
│   └── __init__.py      # GaussianPolicy with feature encoding
├── train/
│   └── __init__.py      # Training loop, episode collection
└── viz/
    └── pendulum.py      # Visualization utilities

tests/
└── test_pendulum.py     # Clean unit tests

examples/
└── train_pendulum.py    # Training script
```

## Known Issues

- Training is not yet achieving 100% success rate
- Needs hyperparameter tuning and longer training
- Gradient norms can grow large (controlled with clipping)

## Next Steps

1. **Hyperparameter search**: Learning rate, network size, batch size
2. **Longer training**: Current tests only run 100-500 iterations
3. **Better exploration**: Adjust entropy weight or initial std
4. **Checkpointing**: Save/load training state for long runs