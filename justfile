# Fast check - linting and quick tests only
check: fastcheck
    uv run -m pytest -m "not slow"

# Slow check - all tests including slow ones
slowcheck: fastcheck
    uv run -m pytest

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
    uv run -m pytest -k "{{pattern}}"

# Quick test - same as check
quicktest: check
