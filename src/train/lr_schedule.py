"""Learning rate schedules for training."""

import optax


def create_lr_schedule(
    base_lr: float,
    warmup_steps: int = 100,
    decay_steps: int = 1000,
    end_lr_factor: float = 0.1,
) -> optax.Schedule:
    """Create learning rate schedule with warmup and cosine decay.
    
    Args:
        base_lr: Base learning rate after warmup
        warmup_steps: Number of steps for linear warmup
        decay_steps: Total number of steps for cosine decay
        end_lr_factor: Final LR as fraction of base_lr
    
    Returns:
        optax schedule function
    """
    # Linear warmup from 0 to base_lr
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    
    # Cosine decay from base_lr to end_lr
    decay_schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=decay_steps - warmup_steps,
        alpha=end_lr_factor,
    )
    
    # Combine schedules
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule],
        boundaries=[warmup_steps],
    )
    
    return schedule