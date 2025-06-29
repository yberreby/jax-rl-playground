import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from src.init import sparse_init
from src.distributions import gaussian_log_prob
from src.constants import DEFAULT_SPARSITY, INITIAL_LOG_STD, DEFAULT_HIDDEN_DIM


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

        # Initialize weights with sparse init (for now)
        self.w1 = nnx.Param(
            sparse_init(key1, (obs_dim, hidden_dim), sparsity=DEFAULT_SPARSITY)
        )
        self.b1 = nnx.Param(jnp.zeros(hidden_dim))
        self.w2 = nnx.Param(
            sparse_init(key2, (hidden_dim, action_dim), sparsity=DEFAULT_SPARSITY)
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

        mean = h @ self.w2.value + self.b2.value
        std = jnp.exp(self.log_std.value)
        # Broadcast std to match batch dimension
        std = jnp.broadcast_to(std, mean.shape)

        return mean, std


@nnx.jit
def sample_actions(
    policy: GaussianPolicy, obs: Float[Array, "batch obs_dim"], key: Array
) -> tuple[Float[Array, "batch act_dim"], Float[Array, "batch"]]:
    mean, std = policy(obs)

    eps = jax.random.normal(key, mean.shape)
    actions = mean + std * eps

    log_probs = gaussian_log_prob(actions, mean, std)

    return actions, log_probs
