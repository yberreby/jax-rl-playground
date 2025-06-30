import jax
import jax.numpy as jnp
from . import dynamics, reward, reset, rollout, GRAVITY, LENGTH, MASS


def test_steady_state_is_steady():
    steady_state = jnp.array([0.0, 0.0])
    action = jnp.array([0.0])

    next_state = dynamics(steady_state, action)

    assert jnp.allclose(next_state, steady_state, atol=1e-6), (
        f"Steady state moved: {steady_state} -> {next_state}"
    )


def compute_energy(state):
    theta, theta_dot = state
    kinetic = 0.5 * MASS * (LENGTH * theta_dot) ** 2
    potential = MASS * GRAVITY * LENGTH * (1 - jnp.cos(theta))
    return kinetic + potential


def test_energy_conservation_no_friction():
    initial_state = jnp.array([jnp.pi / 4, 0.0])
    actions = jnp.zeros((100, 1))

    initial_energy = compute_energy(initial_state)
    states = rollout(initial_state, actions, friction=0.0)
    final_energy = compute_energy(states[-1])

    assert jnp.abs(final_energy - initial_energy) < 1e-5, (
        f"Energy not conserved: {initial_energy} -> {final_energy}"
    )


def count_zero_crossings(positions):
    return jnp.sum(jnp.diff(jnp.sign(positions)) != 0)


def test_oscillation_without_friction():
    initial_state = jnp.array([jnp.pi / 6, 0.0])
    actions = jnp.zeros((50, 1))

    states = rollout(initial_state, actions, friction=0.0)
    positions = states[:, 0]

    zero_crossings = count_zero_crossings(positions)
    assert zero_crossings >= 2, (
        f"Pendulum didn't oscillate, only {zero_crossings} zero crossings"
    )


def test_damping_with_friction():
    initial_state = jnp.array([jnp.pi / 2, 0.0])
    actions = jnp.zeros((400, 1))
    friction = 1.0

    states = rollout(initial_state, actions, friction=friction)
    final_state = states[-1]

    assert jnp.abs(final_state[0]) < 0.1, (
        f"Pendulum didn't settle to bottom, theta = {final_state[0]}"
    )
    assert jnp.abs(final_state[1]) < 0.1, (
        f"Pendulum didn't stop moving, theta_dot = {final_state[1]}"
    )


def test_reward_prefers_upright():
    # In pendulum convention: theta=0 is down, theta=pi is up
    downward = jnp.array([0.0, 0.0])  # theta=0 is down
    upright = jnp.array([jnp.pi, 0.0])  # theta=pi is up
    action = jnp.array([0.0])

    r_downward = reward(downward, action)
    r_upright = reward(upright, action)

    assert r_upright > r_downward, (
        f"Upright reward {r_upright} not > downward reward {r_downward}"
    )
    
    # With -cos(theta): upright (theta=pi) gives +1, downward (theta=0) gives -1
    assert jnp.isclose(r_upright, 1.0, atol=1e-3), (
        f"Upright reward should be ~1, got {r_upright}"
    )
    assert jnp.isclose(r_downward, -1.0, atol=1e-3), (
        f"Downward reward should be ~-1, got {r_downward}"
    )


def test_reset_bounds():
    key = jax.random.PRNGKey(42)

    for i in range(10):
        key, subkey = jax.random.split(key)
        state = reset(subkey)
        theta, theta_dot = state

        assert -jnp.pi <= theta <= jnp.pi, f"Initial angle {theta} out of bounds"
        assert -1.0 <= theta_dot <= 1.0, f"Initial velocity {theta_dot} out of bounds"
