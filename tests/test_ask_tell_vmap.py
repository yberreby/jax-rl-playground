import jax
import jax.numpy as jnp
import optuna
from tests.constants import OPTUNA_SEARCH_RANGE, OPTUNA_TRIAL_COUNT


def test_basic_ask_tell_vmap():
    # Simple quadratic objective: (x - 2)^2 + (y - 3)^2
    def objective(x, y):
        return (x - 2.0) ** 2 + (y - 3.0) ** 2

    # Vectorize it
    vmap_objective = jax.vmap(objective)

    study = optuna.create_study(direction="minimize")

    # Ask for 4 trials at once
    trials = [study.ask() for _ in range(OPTUNA_TRIAL_COUNT)]

    # Collect params as arrays
    xs = jnp.array([t.suggest_float("x", *OPTUNA_SEARCH_RANGE) for t in trials])
    ys = jnp.array([t.suggest_float("y", *OPTUNA_SEARCH_RANGE) for t in trials])

    # Evaluate in parallel
    losses = vmap_objective(xs, ys)

    # Tell results
    for trial, loss in zip(trials, losses):
        study.tell(trial, float(loss))

    print(f"Best value after 4 trials: {study.best_value}")
    print(f"Best params: x={study.best_params['x']}, y={study.best_params['y']}")

    assert study.best_value >= 0  # Should be non-negative
    assert len(study.trials) == 4
