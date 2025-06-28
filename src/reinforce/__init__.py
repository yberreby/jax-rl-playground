import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from functools import partial
from src.distributions import gaussian_log_prob


@partial(jax.jit, static_argnames=["policy_fn"])
def reinforce_loss(
    policy_fn,  # Must be a JIT'd function
    params: PyTree,
    obs: Float[Array, "batch obs_dim"],
    actions: Float[Array, "batch act_dim"],
    rewards: Float[Array, "batch"],
) -> Float[Array, ""]:
    mean, std = policy_fn(params, obs)
    log_probs = gaussian_log_prob(actions, mean, std)

    # REINFORCE: maximize E[log π(a|s) * R]
    return -jnp.mean(log_probs * rewards)
