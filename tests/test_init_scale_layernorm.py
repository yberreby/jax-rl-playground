import jax
import jax.numpy as jnp
from flax import nnx
from src.init import sparse_init
from tests.constants import (
    DEFAULT_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_OBS_DIM,
    DEFAULT_ACTION_DIM,
    INIT_SCALE_TEST_VALUES,
    SPARSITY_TEST_VALUES,
)


class ScaledGaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        use_layernorm: bool = True,
        init_scale: float = 1.0,
        sparsity: float = 0.5,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        key1, key2 = jax.random.split(rngs())

        # Initialize with scaled sparse init
        self.w1 = nnx.Param(
            init_scale * sparse_init(key1, (obs_dim, hidden_dim), sparsity=sparsity)
        )
        self.b1 = nnx.Param(jnp.zeros(hidden_dim))
        self.w2 = nnx.Param(
            init_scale * sparse_init(key2, (hidden_dim, action_dim), sparsity=sparsity)
        )
        self.b2 = nnx.Param(jnp.zeros(action_dim))

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.layer_norm = nnx.LayerNorm(
                num_features=hidden_dim, use_bias=False, use_scale=False, rngs=rngs
            )

        # Learned log standard deviation
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        hidden = obs @ self.w1.value + self.b1.value
        hidden = jax.nn.relu(hidden)

        if self.use_layernorm:
            hidden = self.layer_norm(hidden)

        mean = hidden @ self.w2.value + self.b2.value
        std = jnp.exp(self.log_std.value)

        return mean, std


def test_init_scales():
    key = jax.random.PRNGKey(DEFAULT_SEED)
    obs = jax.random.normal(key, (DEFAULT_BATCH_SIZE, DEFAULT_OBS_DIM))
    target_values = jnp.array([[1.0, -0.5]])
    targets = jnp.broadcast_to(target_values, (DEFAULT_BATCH_SIZE, DEFAULT_ACTION_DIM))

    scales = INIT_SCALE_TEST_VALUES

    print("=== Initialization Scale vs Initial Loss ===\n")
    print(
        "Scale | No LN Loss | With LN Loss | Ratio | Hidden std (no LN) | Hidden std (LN)"
    )
    print("-" * 80)

    for scale in scales:
        # Without LayerNorm
        policy_no_ln = ScaledGaussianPolicy(
            obs_dim=DEFAULT_OBS_DIM,
            action_dim=DEFAULT_ACTION_DIM,
            hidden_dim=64,
            use_layernorm=False,
            init_scale=scale,
            rngs=nnx.Rngs(key),
        )

        mean_no_ln, _ = policy_no_ln(obs)
        loss_no_ln = float(jnp.mean((mean_no_ln - targets) ** 2))

        # Get hidden activations
        hidden_no_ln = jax.nn.relu(obs @ policy_no_ln.w1.value + policy_no_ln.b1.value)
        hidden_std_no_ln = float(jnp.std(hidden_no_ln))

        # With LayerNorm
        policy_ln = ScaledGaussianPolicy(
            obs_dim=DEFAULT_OBS_DIM,
            action_dim=DEFAULT_ACTION_DIM,
            hidden_dim=64,
            use_layernorm=True,
            init_scale=scale,
            rngs=nnx.Rngs(key),
        )

        mean_ln, _ = policy_ln(obs)
        loss_ln = float(jnp.mean((mean_ln - targets) ** 2))

        # Get hidden activations
        hidden_ln = jax.nn.relu(obs @ policy_ln.w1.value + policy_ln.b1.value)
        hidden_std_ln = float(jnp.std(hidden_ln))

        ratio = loss_ln / loss_no_ln

        print(
            f"{scale:5.1f} | {loss_no_ln:10.4f} | {loss_ln:12.4f} | {ratio:5.2f}x | "
            f"{hidden_std_no_ln:18.4f} | {hidden_std_ln:15.4f}"
        )

    # Test with different sparsity levels
    print("\n=== Sparsity vs Initial Loss (scale=1.0) ===\n")
    print("Sparsity | No LN Loss | With LN Loss | Ratio")
    print("-" * 50)

    sparsities = SPARSITY_TEST_VALUES

    for sparsity in sparsities:
        # Without LayerNorm
        policy_no_ln = ScaledGaussianPolicy(
            obs_dim=DEFAULT_OBS_DIM,
            action_dim=DEFAULT_ACTION_DIM,
            hidden_dim=64,
            use_layernorm=False,
            sparsity=sparsity,
            rngs=nnx.Rngs(key),
        )

        mean_no_ln, _ = policy_no_ln(obs)
        loss_no_ln = float(jnp.mean((mean_no_ln - targets) ** 2))

        # With LayerNorm
        policy_ln = ScaledGaussianPolicy(
            obs_dim=DEFAULT_OBS_DIM,
            action_dim=DEFAULT_ACTION_DIM,
            hidden_dim=64,
            use_layernorm=True,
            sparsity=sparsity,
            rngs=nnx.Rngs(key),
        )

        mean_ln, _ = policy_ln(obs)
        loss_ln = float(jnp.mean((mean_ln - targets) ** 2))

        ratio = loss_ln / loss_no_ln

        print(f"{sparsity:8.2f} | {loss_no_ln:10.4f} | {loss_ln:12.4f} | {ratio:5.2f}x")


if __name__ == "__main__":
    test_init_scales()
