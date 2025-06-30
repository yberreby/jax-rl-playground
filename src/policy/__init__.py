import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
import equinox as eqx
from src.distributions import gaussian_log_prob
from src.constants import INITIAL_LOG_STD, DEFAULT_HIDDEN_DIM
from src.pendulum import MAX_TORQUE


class GaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_hidden_layers: int = 2,
        use_layernorm: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.obs_dim = obs_dim
        
        # Build network layers
        layers = []
        
        # Input layer
        w_init = jax.nn.initializers.orthogonal()
        layers.append(nnx.Linear(obs_dim, hidden_dim, kernel_init=w_init, rngs=rngs))
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            if use_layernorm:
                layers.append(nnx.LayerNorm(hidden_dim, use_bias=False, use_scale=False, rngs=rngs))
            layers.append(nnx.Linear(hidden_dim, hidden_dim, kernel_init=w_init, rngs=rngs))
        
        # Output layer with small initialization
        def output_init(key, shape, dtype=None):
            return jax.random.normal(key, shape, dtype=dtype) * 0.01
        layers.append(nnx.Linear(hidden_dim, action_dim, kernel_init=output_init, rngs=rngs))
        
        self.layers = layers
        self.use_layernorm = use_layernorm

        # Learnable log std
        self.log_std = nnx.Param(jnp.full(action_dim, INITIAL_LOG_STD))

    def __call__(
        self, obs: Float[Array, "batch obs_dim"]
    ) -> tuple[Float[Array, "batch act_dim"], Float[Array, "batch act_dim"]]:
        # Use observations directly - feature computation should be done externally
        h = obs
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            # Apply activation after linear layers (skip LayerNorm)
            if not (self.use_layernorm and isinstance(self.layers[i+1], nnx.LayerNorm)):
                h = jax.nn.relu(h)
        
        # Final layer (no activation)
        mean = self.layers[-1](h)

        std = jnp.exp(self.log_std.value)
        # Broadcast std to match batch dimension
        std = jnp.broadcast_to(std, mean.shape)

        return mean, std

    def log_prob(
        self, obs: Float[Array, "batch obs_dim"], actions: Float[Array, "batch act_dim"]
    ) -> Float[Array, "batch"]:
        """Compute log probability of actions, accounting for tanh squashing."""
        mean, std = self(obs)

        # Inverse tanh to get unbounded actions
        # Clip to avoid numerical issues at boundaries
        actions_clipped = jnp.clip(actions, -MAX_TORQUE + 1e-6, MAX_TORQUE - 1e-6)
        unbounded_actions = MAX_TORQUE * jnp.arctanh(actions_clipped / MAX_TORQUE)

        # Log prob in unbounded space
        log_probs = gaussian_log_prob(unbounded_actions, mean, std)

        # Correction for tanh squashing: subtract log|det J|
        # For a = c*tanh(z/c), we have |det J| = (1 - (a/c)²)
        log_det_jacobian = jnp.sum(
            jnp.log(1.0 - (actions_clipped / MAX_TORQUE) ** 2 + 1e-6), axis=-1
        )

        return log_probs - log_det_jacobian


@eqx.filter_jit
def sample_actions(
    policy: GaussianPolicy, obs: Float[Array, "... obs_dim"], key: Array
) -> tuple[Float[Array, "... act_dim"], Float[Array, "..."]]:
    # Handle both batched and unbatched inputs
    squeeze_output = False
    if obs.ndim == 1:
        obs = obs[None, :]
        squeeze_output = True
    
    # Get unbounded mean and std
    mean, std = policy(obs)

    # Sample in unbounded space
    eps = jax.random.normal(key, mean.shape)
    unbounded_actions = mean + std * eps

    # Apply tanh squashing to bound actions
    actions = MAX_TORQUE * jnp.tanh(unbounded_actions / MAX_TORQUE)

    # Use policy's log_prob method for consistency
    log_probs = policy.log_prob(obs, actions)

    # Remove batch dimension if input was unbatched
    if squeeze_output:
        return actions[0], log_probs[0]
    
    return actions, log_probs
