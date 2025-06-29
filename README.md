# jax-rl-playground

This is a WIP environment testing [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)'s ability to implement Reinforcement Learning algorithms/components in JAX and semi-autonomously run small-scale experiments in a disciplined manner.

This has extensive linting, type checking, runtime shape checking, and unit testing.
Run `uv run just check` for a test run.

See `./CLAUDE.md` for more.

Tested on Arch Linux with a CUDA environment (RTX 4060).

## Correctness philosophy

At any given point, all tests on the `master` branch should pass.
The point here is largely to streamline the use of LLMs to accelerate experimentation, but with added speed of execution comes a greater risk of errors.
In order to alleviate this issue, the focus here is on fast automated verification.
Don't trust your LLM (nor yourself!) to write correct code; instead, (make it) break down problems into small, well-tested subcomponents.
Even so, conceptual errors might remain; exercise caution.

## Why should I care?

Testing, packaging, sensible defaults for CLAUDE.md have been set up for you.
A tight feedback loop and a coding methodology make the use of Claude Code or similar agentic systems more useful when tackling research questions.
Example code makes it easier for the agent to understand how it should code. This is especially important for newer JAX APIs.

## Based on `jax-ml-template`

This project was originally initialized from [@yberreby/jax-ml-template](https://github.com/yberreby/jax-ml-template).

## License

This project is licensed under the MIT License.

See `./LICENSE.md` for details.
