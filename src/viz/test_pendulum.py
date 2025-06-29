import jax
import jax.numpy as jnp
import pytest
import os
from .pendulum import PendulumVisualizer
from ..pendulum import reset_env, step


def generate_test_trajectory(n_steps=50):
    key = jax.random.PRNGKey(42)
    env_state = reset_env(key)

    states = [env_state.state]
    actions = []
    rewards = []

    for i in range(n_steps):
        # Simple sinusoidal control
        action = jnp.array([jnp.sin(i * 0.1)])
        result = step(env_state, action)

        states.append(result.state)
        actions.append(action)
        rewards.append(result.reward)
        env_state = result.env_state

    return states, actions, rewards


@pytest.mark.slow
def test_pendulum_animation():
    os.makedirs("tests/outputs", exist_ok=True)

    states, actions, rewards = generate_test_trajectory(100)

    viz = PendulumVisualizer(
        trail_length=30, show_angle_arc=True, show_velocity_arrow=True, dark_mode=False
    )

    output_file = "tests/outputs/pendulum_viz.mp4"
    result = viz.create_animation(
        states=states, actions=actions, rewards=rewards, filename=output_file, fps=30
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0
    assert result == output_file


@pytest.mark.slow
def test_dark_mode_animation():
    os.makedirs("tests/outputs", exist_ok=True)

    states, _, _ = generate_test_trajectory(50)

    viz = PendulumVisualizer(dark_mode=True)
    output_file = "tests/outputs/pendulum_dark.mp4"

    viz.create_animation(states=states, filename=output_file, fps=20)

    assert os.path.exists(output_file)


def test_phase_portrait():
    os.makedirs("tests/outputs", exist_ok=True)

    states, _, _ = generate_test_trajectory(200)

    viz = PendulumVisualizer()
    output_file = "tests/outputs/phase_portrait.png"

    result = viz.create_phase_portrait(
        states=states,
        filename=output_file,
        show_trajectory=True,
        show_vector_field=True,
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0
    assert result == output_file
