# jax-ml-template

A JAX-based ML project.

Developed and tested on Arch Linux.

## Git Hooks

This project uses `pre-commit` to enforce code quality. To install the hooks, run:

```bash
uv run pre-commit install
```

## Development

```bash
# Lint
ruff check

# Typecheck
uvx ty check

# Run tests
uv run -m pytest -v

# Run benchmarks
uv run richbench bench/

# You can run all of the above with:
uv run just check

# Formatting
ruff format

# Initial setup: convert .py to .ipynb
uv run jupytext --to notebook nb/interactive_demo.py

# Sync changes from .py to .ipynb
uv run jupytext --sync nb/interactive_demo.py

# Or pair in JupyterLab: Command Palette → "Pair Notebook with percent Script"

# Run headlessly
MPLBACKEND=Agg uv run jupyter execute nb/interactive_demo.ipynb

# Run Jupyter server under correct venv
uv run jupyter-lab .
```

## Benchmarking

Performance benchmarks are located in `bench/` and use `richbench`:

- Create benchmark files named `bench_*.py`
- Define pairs of functions to compare
- Use `block_until_ready()` for JAX operations to be meaningfully timed
- Run with `uv run richbench bench/`

You'll see speedup factors between different implementations.

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
- `richbench` for performance benchmarking.
- Claude Code for AI-assisted fast prototyping.

## License

This project template is licensed under the MIT License.

See `./LICENSE.md` for details.

**Don't forget to update the license if you use this in your own project!**
