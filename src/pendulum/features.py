"""Feature engineering for pendulum state representation."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.jit
def compute_features(state: Float[Array, "2"]) -> Float[Array, "8"]:
    """Compute rich features from pendulum state.
    
    Args:
        state: [theta, theta_dot] where theta is angle from vertical
        
    Returns:
        features: [sin(theta), cos(theta), theta_dot, x, y, dx/dt, dy/dt, kinetic_energy]
        
    The pendulum hangs down at theta=0 and points up at theta=pi.
    """
    theta, theta_dot = state
    
    # Cyclical encoding of angle
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    
    # Cartesian coordinates (assuming unit length pendulum)
    # x = sin(theta), y = -cos(theta)  (y points down)
    x = sin_theta
    y = -cos_theta
    
    # Cartesian velocities
    dx_dt = cos_theta * theta_dot
    dy_dt = sin_theta * theta_dot
    
    # Kinetic energy (normalized by max possible)
    # KE = 0.5 * L^2 * theta_dot^2, with L=1
    kinetic_energy = 0.5 * theta_dot**2 / 32.0  # Normalize by 0.5 * max_speed^2
    
    return jnp.array([
        sin_theta,
        cos_theta,
        theta_dot / 8.0,  # Normalize by max speed
        x,
        y,
        dx_dt / 8.0,  # Normalize by max cartesian velocity
        dy_dt / 8.0,
        kinetic_energy
    ])