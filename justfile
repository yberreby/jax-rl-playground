# Create venv and install all dependencies from uv.lock
setup:
    uv sync

fastcheck: setup
    ruff check
    uvx ty check

check: fastcheck
    uv run -m pytest -v
    uv run richbench bench/