test: fastcheck

# Create venv and install all dependencies from uv.lock
setup:
    uv sync

# Linting and type checking
fastcheck: setup
    ruff check --fix
    uvx ty check

format:
    ruff format

quicktest: fastcheck
    uv run -m pytest -m "not slow"

# Test specific modules/tests using pattern matching
test pattern: fastcheck
    uv run -m pytest -k "{{pattern}}" -v
