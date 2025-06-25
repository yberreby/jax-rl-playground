Check README.md for details.

# General Methodology

Code like a hacker: concisely, with self-doubt, without fluff, without repeating yourself, keeping code as orthogonal as possible.

## DRY
- Repeating oneself is unacceptable.
- If your LOCs look suspiciously similar, consider a loop or a lambda.
- If you can refactor to follow a "data-driven" approach (e.g. list of dicts instead of ad-hoc code), consider doing so.
- Don't be afraid of using tiny, local abstractions.

## The Cost of Code
- Every line of code has a cost.
- Code, by itself, has no value.
- Code that has been verified abundantly, through unit and integration tests, might have some value.

## Breaking Things That Work
- If you're afraid of breaking something that currently works, it's probably too brittle, so you should break it.

## Functions and Variables
- Functions are your friends. Top-level, lambdas. Use and abuse.
- Purity is good. Testability is good.
- A one-line function is better than repeating a complex line twice.

## Comments
- Code should be self-documenting using clear variable names and function names.
- If you *need* comments to explain the "what" or "how", that's usually a red flag.
- Comments are there for "business" logic / field-specific knowledge.

## Classes
- Prefer naked pure functions over classes whenever possible.
- Only use stateful constructs if it is clearly the idiomatic, sensible thing to do.

## Workflow
- Plan out API -> Write a test -> Write implementation -> Lint and run test -> Assess
- Start with something basic, get it to work, refactor early, often, aggressively.

# JAX-specific points

- Efficient tensor operations should be thought out in advance.
- Use `static_argnames` instead of `static_argnums` whenever possible.
- Note that JIT'd functions can take other JIT'd functions as arguments. Remember, also, that `tree_util.Partial` exists.
- Remember that Python control flow = static unrolling. This might be what you want, but often, it isn't.
- If a piece of code is meant to be fast, it should be profiled/benchmarked.
