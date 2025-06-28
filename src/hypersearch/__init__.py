"""Hyperparameter search utilities using Optuna."""

import optuna
from typing import Any, Optional


def create_study(
    study_name: str,
    direction: str = "minimize",
    storage: Optional[str] = None,
) -> optuna.Study:
    """Create or load an Optuna study."""
    if storage is None:
        storage = f"sqlite:///tests/outputs/{study_name}.db"
    
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True,
    )


def suggest_optimizer_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest optimizer parameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"]),
    }


def suggest_architecture_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest architecture parameters."""
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, step=16),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "use_layernorm": trial.suggest_categorical("use_layernorm", [True, False]),
    }