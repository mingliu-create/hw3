## 1. Implementation
- [ ] 1.1 Create `tools/openspec-cli` scaffolder script (Node.js or Python) that accepts a `change-id` and minimal metadata.
- [ ] 1.2 Make the script idempotent and validate `change-id` is kebab-case and verb-led.
- [ ] 1.3 Implement a `--preview` mode that prints the scaffolded files to stdout without writing.
- [ ] 1.4 Add `--validate` option that runs `openspec validate <change-id> --strict` when the CLI is present.
- [ ] 1.5 Add README/docs for usage in `openspec/changes/add-openspec-cli/`.

## 2. Tests
- [ ] 2.1 Add unit tests for scaffolder (happy path + invalid change-id).
- [ ] 2.2 Add a smoke test that runs the scaffolder in a temp folder and verifies files created.

## 3. Docs & PR
- [ ] 3.1 Document usage in `openspec/project.md` (done) and the new README for the tool.
- [ ] 3.2 Open a PR linking the proposal directory and request review.
