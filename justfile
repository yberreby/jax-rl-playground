# Fast check - linting and quick tests only
check: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -n auto -m "not slow"

# Single process check
singlecheck: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -m "not slow"

# Thread-based check - single process with multiple threads
threadcheck: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest --workers 1 --tests-per-worker auto -m "not slow"

# Slow check - all tests including slow ones
slowcheck: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -n auto

# Test specific modules/tests using pattern matching
test pattern: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -n auto -k "{{pattern}}"

# Linting and type checking
fastcheck: setup
    uv run ruff check --fix
    uvx ty check

# Create venv and install all dependencies from uv.lock
setup:
    uv sync

format:
    uv run ruff format
