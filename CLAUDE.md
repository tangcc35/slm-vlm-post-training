# CLAUDE.md

## Writing code: keep it simple

- Write the simplest code that solves the actual problem: fewer lines, fewer branches, fewer abstractions.
- Don't design for hypothetical use cases or "what if later…" scenarios. Build what's needed now.
- Don't chase rare edge cases. If a simple design has a known limitation, accept it and mention it in one line instead of adding code to cover it.
- Don't be over-protective: no defensive checks for states that can't happen, no re-validating values our own code produced, no try/except around code that isn't expected to fail. Validate only at real boundaries (request bodies, external APIs, user-supplied files).
- Keep changes scoped to the task. No drive-by refactors, new config options, helpers, or abstractions unless asked.
- Tests cover the behavior that changed, not every permutation.
