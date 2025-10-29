## ADDED Requirements

### Requirement: Local OpenSpec scaffolder CLI
The repository SHALL provide a small, optional CLI or script that scaffolds a change proposal layout for a given `change-id`.

#### Scenario: Scaffolding a new change
- **WHEN** a contributor runs `openspec-cli scaffold add-my-feature --preview` (or equivalent)
- **THEN** the CLI prints the proposed files and structure for `openspec/changes/add-my-feature/` including `proposal.md`, `tasks.md`, and `specs/`.

#### Scenario: Idempotent scaffold
- **WHEN** a contributor runs the scaffolder twice with the same `change-id`
- **THEN** it should not overwrite existing files without explicit confirmation and should exit successfully with a non-destructive status.
