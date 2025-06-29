# jax-rl-demo

This is a WIP environment testing [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)'s ability to implement Reinforcement Learning algorithms/components in JAX and semi-autonomously run small-scale experiments in a disciplined manner.

This has extensive linting, type checking, runtime shape checking, and unit testing.
Run `uv run just check` for a test run.

See `./CLAUDE.md` for more.

Tested on Arch Linux with a CUDA environment (RTX 4060).

## Based on `jax-ml-template`

This project was originally initialized from [@yberreby/jax-ml-template](https://github.com/yberreby/jax-ml-template).

## License

This project is licensed under the MIT License.

See `./LICENSE.md` for details.
