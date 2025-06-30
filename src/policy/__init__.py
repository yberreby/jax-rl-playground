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
        use_layernorm: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        key1, key2 = jax.random.split(rngs())

        # Initialize weights with orthogonal init (standard)
        self.w1 = nnx.Param(
            jax.nn.initializers.orthogonal()(key1, (obs_dim, hidden_dim))
        )
        self.b1 = nnx.Param(jnp.zeros(hidden_dim))

        # Small initialization for output layer to prevent initial saturation
        self.w2 = nnx.Param(
            jax.random.normal(key2, (hidden_dim, action_dim)) * 0.1
        )
        self.b2 = nnx.Param(jnp.zeros(action_dim))

        self.use_layernorm = use_layernorm

        # LayerNorm WITHOUT learnable parameters
        if use_layernorm:
            self.layer_norm = nnx.LayerNorm(
                num_features=hidden_dim, use_bias=False, use_scale=False, rngs=rngs
            )

        # Fixed log std for now
        self.log_std = nnx.Param(jnp.full(action_dim, INITIAL_LOG_STD))

    def __call__(
        self, obs: Float[Array, "batch obs_dim"]
    ) -> tuple[Float[Array, "batch act_dim"], Float[Array, "batch act_dim"]]:
        h = obs @ self.w1.value + self.b1.value

        if self.use_layernorm:
            # LayerNorm on pre-activations
            h = self.layer_norm(h)

        h = jax.nn.relu(h)

        # Raw unbounded mean
        mean = h @ self.w2.value + self.b2.value

        std = jnp.exp(self.log_std.value)
        # Broadcast std to match batch dimension
        std = jnp.broadcast_to(std, mean.shape)

        return mean, std

    def log_prob(
        self,
        obs: Float[Array, "batch obs_dim"],
        actions: Float[Array, "batch act_dim"]
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
            jnp.log(1.0 - (actions_clipped / MAX_TORQUE)**2 + 1e-6),
            axis=-1
        )

        return log_probs - log_det_jacobian


@eqx.filter_jit
def sample_actions(
    policy: GaussianPolicy, obs: Float[Array, "batch obs_dim"], key: Array
) -> tuple[Float[Array, "batch act_dim"], Float[Array, "batch"]]:
    # Get unbounded mean and std
    mean, std = policy(obs)

    # Sample in unbounded space
    eps = jax.random.normal(key, mean.shape)
    unbounded_actions = mean + std * eps

    # Apply tanh squashing to bound actions
    actions = MAX_TORQUE * jnp.tanh(unbounded_actions / MAX_TORQUE)

    # Use policy's log_prob method for consistency
    log_probs = policy.log_prob(obs, actions)

    return actions, log_probs
