# Fast check - linting and quick tests only
check: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -n auto -m "not slow"

# Slow check - all tests including slow ones
slowcheck: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -n auto

# Create venv and install all dependencies from uv.lock
setup:
    uv sync

# Linting and type checking
fastcheck: setup
    ruff check --fix
    uvx ty check

format:
    ruff format

# Test specific modules/tests using pattern matching
test pattern: fastcheck
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run -m pytest -n auto -k "{{pattern}}"

# Quick test - same as check
quicktest: check
