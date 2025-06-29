# Research Philosophy

Quality > quantity, carefulness over speed, paranoid checking over "it runs, next".

Start simple before adding complexity. Don't jump into advanced methods until something simpler runs, and necessity is established.

# General Methodology

Code like a hacker: concisely, with self-doubt, without fluff, without repeating yourself, keeping code as orthogonal as possible.

## DRY - Apply OBSESSIVELY
- Repeating oneself is unacceptable.
- If your LOCs look suspiciously similar, consider a loop or a lambda.
- If you can refactor to follow a "data-driven" approach (e.g. list of dicts instead of ad-hoc code), consider doing so.
- Don't be afraid of using tiny, local abstractions.
- Apply DRY even in tests - factor out setup, use named constants
- No magic numbers - name everything (TEST_MATRIX_SIZE, TOLERANCE, etc.)
- Domain knowledge stays in ONE place

## The Cost of Code
- Every line of code has a cost.
- Code, by itself, has no value.
- Code that has been verified abundantly, through unit and integration tests, might have some value.
- The fewer LOCs the better - use git instead of duplicating versions

## Breaking Things That Work
- If you're afraid of breaking something that currently works, it's probably too brittle, so you should break it.

## Functions and Variables
- Functions are your friends. Top-level, lambdas. Use and abuse.
- Purity is good. Testability is good.
- A one-line function is better than repeating a complex line twice.

## Documentation Philosophy
- NO docstrings unless code is "200% done, long-validated"
- Code should be self-documenting using:
  - Clear variable and function names
  - Intermediate variables to break up complex expressions
  - Type annotations (always!)
  - Shape annotations with jaxtyping
- Comments are ONLY for domain-specific knowledge or core algorithms (e.g. REINFORCE math)
- If you *need* comments to explain the "what" or "how", that's usually a red flag

## Classes
- Prefer naked pure functions over classes whenever possible.
- Only use stateful constructs if it is clearly the idiomatic, sensible thing to do.

# Testing Philosophy

## Immediate Testing
- Write tests IMMEDIATELY after writing even 10 lines
- Tests can start inline (def test_* right after function)
- If complexity grows without tests, nuke untested parts and simplify
- Ensure changes WORK before moving to next task

## Learning Dynamics Testing
- Write "mini-tests" for LEARNING DYNAMICS, not just correctness
- If you claim something helps optimization, TEST IT:
  - Run 10-1000 gradient steps on test problems
  - Running gradient steps on even 100k-dimensional tensors is cheap
- Example: Test that sparse init reduces interference
- Example: Test that normalization maintains stable gradients
- Tests should produce:
  - Plots for human analysis (save to tests/outputs/ or similar)
  - CSVs for follow-up analysis
  - Don't clutter main directory

## Test Organization
- Many small tests with FACTORED setup
- Well-named test functions
- When tests fail, ask WHY you got it wrong, not just how to fix
- Comments in tests like "should" go in assert messages for reporting

## Test Commands
- Check the `./justfile` for up-to-date information
- Use project-wide test collection with -k, maybe modules, but never point pytest to a .py
- Use `just check` for full test suite
- Use `just test <pattern>` or `uv run -m pytest -k "<pattern>"` for isolated testing
- Use pytest markers for categorization:
  - `@pytest.mark.slow` for expensive tests
  - Run quick tests with `-m "not slow"`

# Code Organization

## Module Structure
- Modules as directories: `src/module/__init__.py` + `src/module/test.py`
- Small files (30-150 lines)
- As many small modules and submodules as needed/appropriate
- tests/ directory for PUBLIC API testing only
- Implementation tests stay within modules

## File Organization
- No example scripts - integration tests serve this purpose
- Throwaway/experimental scripts and outputs should go in dedicated directory, not top level

## Public API
- Separate implementation tests from behavior tests
- Integration tests test the public API

# Development Workflow

## Command Workflow
- ALWAYS run `just check` before any execution
- Chain commands: `just check; uv run myfile.py`
- Get feedback from environment FREQUENTLY
- Don't waste edits on trivial changes (e.g. removing single comments)
- Shortest path to working code following guidelines

## Dependency Management
- Use `uv add <package>` to add dependencies, not editing pyproject.toml
- Use `uv run` to execute code

# Python-specific points
- Use `uv run myfile.py` or `uv run -m` to run code
- Check online documentation before using APIs (especially Flax)

# JAX-specific points

## Performance
- Efficient tensor operations should be thought out in advance
- Use `static_argnames` instead of `static_argnums` whenever possible
- JIT'd functions can take other JIT'd functions as arguments
- Remember that Python control flow = static unrolling
- If code is meant to be fast, it should be profiled/benchmarked
- Use `block_until_ready()` when timing operations to separate JIT compilation from runtime

## JAX Workflow
- Disable GPU preallocation: set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Always check if slowness is from JIT compilation vs runtime
- Use `block_until_ready()` when timing operations
- First call includes JIT compilation, subsequent calls are fast

# Important Reminders

- Quality > Quantity
- Test immediately and frequently
- DRY obsessively
- No docstrings until "200% done"
- Use intermediate variables for clarity
- Always run `just check` before execution, preferably as part of the same command to avoid wasting a tool call
- Don't clutter with examples or throwaway code
- Integration tests ARE the examples

# Current State Notes

## Modules Available
- `src/normalize`: Observation normalization and reward scaling
- `src/init`: Sparse initialization
- `src/reinforce`: REINFORCE loss (takes policy function as argument)
- `src/distributions`: Gaussian log probability calculation
- `src/policy`: Flax NNX policy implementation

## Key Design Decisions
- REINFORCE loss is decoupled from policy implementation - takes a JIT'd policy function
- normalize_obs is for normalizing observations, NOT for layer normalization
- Using Flax NNX for neural network implementations

## Flax NNX Critical Notes
- Use `nnx.Param` for trainable parameters, NOT `nnx.Variable`
- Access parameter values with `.value` (e.g., `self.w1.value`)
- Example pattern:
  ```python
  class Model(nnx.Module):
      def __init__(self, rngs):
          self.w = nnx.Param(jax.random.normal(rngs.params(), shape))

      def __call__(self, x):
          return x @ self.w.value
  ```
- Always use `uv run` when running Python scripts for dependencies

## JAX Function Passing Pattern
- JIT'd functions CAN be passed to other JIT'd functions
- If passing a function as argument to a JIT'd function, mark it as static:
  `@partial(jax.jit, static_argnames=["policy_fn"])`
- The function being passed should itself be JIT'd

## Common Debugging Patterns
- If gradient flow isn't working, check:
  1. Using `nnx.Param` not `nnx.Variable` for trainable params
  2. Accessing with `.value`
  3. Simple isolated test first before complex integration
- Create minimal gradient flow tests to isolate issues
- Test outputs should go to `tests/outputs/` (plots, CSVs)

## Optuna Ask-and-Tell Pattern for JAX vmap
- You can use `study.ask()` to get multiple trials at once
- Vectorize evaluation with `jax.vmap` over hyperparameters
- Use `study.tell()` to report results back
- Example in `tests/test_ask_tell_vmap.py`

# Meta Methodological Points

## Code Writing Discipline
- Write MINIMAL code first, test it, THEN expand
- If something doesn't work after 2-3 attempts, DELETE IT
- Unused functions = immediate red flag, remove them
- Start with POC, extract utilities only after proven useful
- Smaller commits > large commits (split by concern)

## Testing Approach
- Examine CSVs directly with grep/sort for quick insights
- JIT compilation crucial: use @nnx.jit on train_step
- Test IMMEDIATELY after writing even 10 lines
- If tests take >1-5s, add @pytest.mark.slow
- Maintain fast feedback loop with `just check`
- Text output > plots for LLM analysis (save tokens)

## Common Mistakes to Avoid
- Writing utilities before they're needed - write straightforward POC first, get it to work, then refactor
- Keeping overcomplicated/brittle code
- Large monolithic commits
- Not running tests frequently enough
- Generating only plots for agentic debugging, when CSV/text would suffice

## When User Gives Feedback
- Document it in CLAUDE.md
- Look for patterns in repeated feedback
- Adjust approach before continuing
- Ask for clarification if unsure
