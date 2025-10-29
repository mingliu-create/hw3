## Change ID
add-openspec-cli

## Why
Contributors need a small, local CLI to scaffold, preview, and validate OpenSpec change proposals before opening PRs. A simple `openspec-cli` (thin wrapper) reduces authoring friction and helps maintain consistent proposal structure.

## What Changes
- Add a small command-line helper in `tools/openspec-cli` (or a documented dev script) to:
  - scaffold `changes/<change-id>/` with `proposal.md`, `tasks.md`, and `specs/` folders
  - run local validation (`openspec validate <id> --strict` if `openspec` is available)
  - preview spec deltas and common formatting issues
- Add a minimal tooling spec under `openspec/changes/add-openspec-cli/specs/tooling/spec.md` describing the CLI behaviour.

## Impact
- Affects contributors and reviewer workflow only (no production runtime changes).
- A new dev-tool folder may be added (`tools/openspec-cli` or `scripts/openspec-cli.*`).
- No breaking changes to existing specs.

## Risks
- Duplication with existing project tooling if present. Keep the implementation minimal and optional (not mandatory for reviews).

## Acceptance Criteria
- `openspec/changes/add-openspec-cli/proposal.md`, `tasks.md`, and `specs/tooling/spec.md` exist and explain the feature.
- The scaffolder creates the correct delta format for proposals.
