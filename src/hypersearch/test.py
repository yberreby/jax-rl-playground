import optuna
from src.hypersearch import suggest_optimizer_params


def test_create_study():
    study = optuna.create_study()  # In-memory study without SQLite
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


def test_suggest_optimizer_params():
    study = optuna.create_study()  # In-memory study without SQLite
    trial = study.ask()
    params = suggest_optimizer_params(trial)
    
    assert "learning_rate" in params
    assert "optimizer" in params
    assert 1e-4 <= params["learning_rate"] <= 1e-1
    assert params["optimizer"] in ["adam", "sgd", "rmsprop"]