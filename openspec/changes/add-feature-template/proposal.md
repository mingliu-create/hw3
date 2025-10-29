## Change ID
add-feature-template

## Title
<Short one-line title for the feature>

## Target capability
<e.g. `auth`, `tooling`, `payments` — leave blank if cross-cutting>

## Why
Describe the problem or opportunity this feature addresses in 1–3 sentences. Explain why the current system is insufficient and what user or business value the feature delivers.

Example:
> Users currently cannot toggle email notifications per-project. This leads to noisy inboxes and increased support requests. Adding per-project email notification settings will reduce noise and improve user satisfaction.

## What Changes
- High-level bullets describing what will be added, removed or modified.
- Where possible, mark breaking changes with **BREAKING** and call out migration steps.

Example:
- **ADD** a new capability `notifications/project-preferences` with an API and UI to manage per-project notification preferences.
- **ADD** backend API: `POST /api/projects/{id}/notification-preferences` and `GET /api/projects/{id}/notification-preferences`.
- **MODIFY** the notifications delivery service to consult project preferences before sending emails.

## Impact
- Affected specs: `specs/notifications/spec.md` (new/modified deltas should be added under `openspec/changes/add-feature-template/specs/`).
- Affected code: list high-level subsystems or files (e.g., `services/notifications/*`, `api/routes/projects.js`).
- Affected teams: (frontend, backend, infra, support)

## Risks & Mitigations
- Risk: Increased complexity in notification delivery may affect throughput — Mitigation: Add a cached lookup and monitor latency.
- Risk: Migration of existing preferences — Mitigation: default to global settings and provide a migration job.

## Acceptance Criteria
- A short list of verifiable criteria. Each criterion should map to at least one spec scenario.
- Example:
  - Users can set per-project email notifications via the UI and API.
  - Notification delivery respects project-level preferences for email and in-app messages.
  - Existing users keep their current global settings unless they opt-in to per-project settings.

## Migration / Rollout Plan (if applicable)
- Steps to migrate or roll out safely.
- Example:
  1. Deploy API and UI behind a feature flag.
 2. Run data migration job to create per-project preferences from global defaults where needed.
 3. Flip feature flag after 1 week of monitoring.

## Files to Create / Update
- `openspec/changes/add-feature-template/specs/<capability>/spec.md` — delta(s) using `## ADDED|MODIFIED|REMOVED Requirements` with at least one `#### Scenario:` per requirement.
- `openspec/changes/add-feature-template/tasks.md` — implementation checklist (implementation, tests, docs, CI updates).
- Implementation files (exact paths listed by the implementer when work begins).

## Tasks (starter)
- [ ] Add spec deltas under `openspec/changes/add-feature-template/specs/` (ADDED/MODIFIED as needed).
- [ ] Add `tasks.md` with a small implementation plan and tests.
- [ ] Request validation: run `openspec validate add-feature-template --strict` and fix formatting issues.
- [ ] Open PR linking this proposal and request review / approval.

## Notes for the author
- Use verb-led, kebab-case `change-id` (see `add-feature-template` example).
- Keep deltas small and focused. If multiple capabilities are affected, create one delta file per capability under `specs/`.
- Ensure every requirement includes at least one `#### Scenario:` section.

---

Fill the placeholders above and add deltas to `openspec/changes/add-feature-template/specs/` when ready. Ask the assistant to scaffold the tasks/specs or implement them in your preferred language.
