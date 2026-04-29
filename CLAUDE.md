**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Coding conventions

- Git author: `claude <claude@users.noreply.github.com>`. Commit messages in English.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Tests

**Run `./cx test` before every commit.** If tests fail, fix them before committing — don't skip, don't disable.

For new code components (a module, a standalone function, a class with non-trivial logic), add a free `test()` function in the same module covering the obvious cases and register the module in `claudex/cmd_test.py::MODULES`. Trivial glue code (a one-line wrapper, a subcommand dispatcher) doesn't need tests.

## 6. Code Style

**Imports.** Imports are split into three sections separated by blank lines: plain `import X` first (sorted by line length), then `from X import Y` lines (unrelated modules separated by blank lines, imports sharing a parent package grouped contiguously and sorted by module name length), then local modules last as `import claudex.X as alias`.

**Blank lines.**
- Two blank lines between top-level definitions.
- Inside a function, surround compound statements (`if`/`for`/`while`/`try`/`with`) with one blank line — **except** when the block sits at the very start or end of its enclosing block.
- Separate an assignment from a following block with a blank line.
- Separate logical sections (setup / main / teardown) with blank lines.
- Blank line before a trailing `return` that follows a block.

```python
def f(x):
    if x is None:           # first stmt — no blank before
        return None

    y = compute(x)          # blank line before the next block

    if y > 0:
        handle(y)

    return y                # blank line before trailing return
```
