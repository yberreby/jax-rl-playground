fastcheck:
    ruff check
    uvx ty check

check: fastcheck
    uv run -m pytest -v
    uv run richbench bench/
