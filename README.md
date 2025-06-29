# jax-rl-playground

A JAX Reinforcement Learning template, speeding up humans and optimized for LLM-assisted development with [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview).

**Fork this to skip the setup and start building RL algorithms in JAX now.**

Prerequisite: [`uv`](https://github.com/astral-sh/uv). Other tools (`ruff`, `just`, `ty`...) will be installed automatically through `uv`.

If you don't have a recent NVIDIA GPU, see `pyproject.toml` to use `jax[cpu]` instead of `jax[cuda12]`.

## ✅ What's Included

### Infrastructure
- [x] [`uv`](https://github.com/astral-sh/uv)-first setup
- [x] GitHub Actions CI (automated testing on every push)
- [x] [`justfile`](https://github.com/casey/just) commands
- [x] [`ruff`](https://github.com/astral-sh/ruff) linting and formatting, with pre-commit hook
- [x] pytest tests with parallel execution and slow test markers
- [x] Static type checking with [`ty`](https://github.com/astral-sh/ty)
- [x] Runtime shape checking with [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) + [`beartype`](https://github.com/beartype/beartype)

### LLM Optimization
- [x] Comprehensive CLAUDE.md with coding methodology
- [x] Test-driven approach that catches AI mistakes early
- [x] Working JAX examples

## 🚀 Quick Start

```bash
git clone https://github.com/yberreby/jax-rl-playground
cd jax-rl-playground
uv sync
uv run just
```

## 🍴 Fork & Customize

```bash
git clone https://github.com/yberreby/jax-rl-playground my-project-name
cd my-project-name
./customize.py --name "My RL Project" --author "Your Name"

# Review changes
git diff

# Set your remote repository
git remote set-url origin git@github.com:yourusername/my-project-name.git

# Commit only relevant files
git add -u  # stages modified files only
git commit -m "Customize template"
git push -u origin master
```

The customization script updates project name, author, and adds MIT attribution.

## 💡 Key Benefits

- **No setup required** - CI/CD, testing, linting all pre-configured
- **AI-friendly** - Clear patterns and immediate feedback
- **Research-oriented** - Test learning dynamics with `slow` tests

## 📚 Philosophy

- **Correctness first** - All tests pass on `master`
- **Test immediately** - Even for 10 lines of code
- **Minimal and modular** - Small, testable components

Based on [@yberreby/jax-ml-template](https://github.com/yberreby/jax-ml-template).

## License

MIT License. See `./LICENSE` for details.
