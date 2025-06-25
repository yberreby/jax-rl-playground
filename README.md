# jax-ml-template

A JAX-based ML project.

Developed and tested on Arch Linux.

## Development

```bash
# Lint
ruff check

# Typecheck
uvx ty check

# Run tests
uv run -m pytest -v

# If you have the `just` task runner installed, you can run all of the above with:
just check

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

## License

This project template is licensed under the MIT License.

See `./LICENSE.md` for details.

**Don't forget to update the license if you use this in your own project!**
