# jax-ml-template

A JAX-based ML project.

## Conventions

- Type annotations wherever possible.
  - `jaxtyping`-based runtime-checked shape annotations on tensor operations.

## Workflow

- JAX for JIT-compiled tensor operations.
- Notebooks:
  - `jaxtyping` + `beartype` for runtime type checking.
  - `jupyter-lab` + `ipympl` + `%matplotlib widget` + `ipywidgets` + `matplotlib`'s `set_data` for interactive plots.
  - `jupytext` for git-friendly Jupyter notebook storage.
  - `%load_ext autoreload; %autoreload 2` for reloading of local files.
- `uv` for dependency management.
- `ruff` for linting.
- `ty` for static type checking.
- `pytest` for unit testing.
- Claude Code for AI-assisted fast prototyping.

```
# Lint
ruff check

# Typecheck
uvx ty check

# Run tests
uv run -m pytest -v
```
