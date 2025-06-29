import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, ScalarLike
from typing import NamedTuple
import diffrax

# Pendulum constants
GRAVITY = 10.0
MASS = 1.0
LENGTH = 1.0
DT = 0.05
MAX_SPEED = 8.0
MAX_TORQUE = 2.0
MAX_EPISODE_STEPS = 200
ACTION_PENALTY_COEF = 0.001


def pendulum_ode(t: Float[Array, ""], y: Float[Array, "2"], args) -> Float[Array, "2"]:
    theta, theta_dot = y
    torque = args["torque"]
    friction = args["friction"]

    dtheta = theta_dot
    dtheta_dot = (
        -(GRAVITY / LENGTH) * jnp.sin(theta)
        + torque / (MASS * LENGTH**2)
        - friction * theta_dot
    )

    return jnp.stack([dtheta, dtheta_dot])


@jax.jit
def dynamics(
    state: Float[Array, "2"], action: Float[Array, "1"], friction: ScalarLike = 0.0
) -> Float[Array, "2"]:
    theta, theta_dot = state
    torque = jnp.clip(action[0], -MAX_TORQUE, MAX_TORQUE)

    term = diffrax.ODETerm(pendulum_ode)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(t1=True)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=DT,
        dt0=DT / 10,
        y0=state,
        args={"torque": torque, "friction": friction},
        saveat=saveat,
    )

    final_theta, final_theta_dot = solution.ys[0]
    final_theta_dot = jnp.clip(final_theta_dot, -MAX_SPEED, MAX_SPEED)
    final_theta = ((final_theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    return jnp.stack([final_theta, final_theta_dot])


@jax.jit
def reward(state: Float[Array, "2"], action: Float[Array, "1"]) -> Float[Array, ""]:
    theta, theta_dot = state

    # Simple continuous reward based on height
    return jnp.cos(theta)


@jax.jit
def reset(key: PRNGKeyArray) -> Float[Array, "2"]:
    theta = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
    theta_dot = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    return jnp.stack([theta, theta_dot])


class EnvState(NamedTuple):
    state: Float[Array, "2"]
    step_count: int
    key: PRNGKeyArray


class StepResult(NamedTuple):
    state: Float[Array, "2"]
    reward: Float[Array, ""]
    done: Float[Array, ""]
    env_state: EnvState


@jax.jit
def step(env_state: EnvState, action: Float[Array, "1"]) -> StepResult:
    next_state = dynamics(env_state.state, action)
    r = reward(env_state.state, action)
    new_step_count = env_state.step_count + 1
    done = jnp.array(new_step_count >= MAX_EPISODE_STEPS, dtype=jnp.float32)

    new_env_state = EnvState(
        state=next_state, step_count=new_step_count, key=env_state.key
    )

    return StepResult(state=next_state, reward=r, done=done, env_state=new_env_state)


@jax.jit
def reset_env(key: PRNGKeyArray) -> EnvState:
    key, subkey = jax.random.split(key)
    initial_state = reset(subkey)

    return EnvState(state=initial_state, step_count=0, key=key)


@jax.jit
def rollout(
    initial_state: Float[Array, "2"],
    actions: Float[Array, "T 1"],
    friction: ScalarLike = 0.0,
) -> Float[Array, "T+1 2"]:
    """Rollout dynamics for a sequence of actions."""

    def step_fn(state, action):
        next_state = dynamics(state, action, friction)
        return next_state, next_state

    final_state, states = jax.lax.scan(step_fn, initial_state, actions)
    return jnp.concatenate([initial_state[None, :], states], axis=0)
