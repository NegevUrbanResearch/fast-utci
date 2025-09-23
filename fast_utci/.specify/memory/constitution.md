<!--
Sync Impact Report
- Version change: N/A → 1.0.0
- Modified principles: placeholders → concrete principles
- Added sections: Additional Constraints & Performance Standards; Development Workflow & Quality Gates; Governance
- Removed sections: none
- Templates requiring updates:
  ✅ fast_utci/.specify/templates/plan-template.md (footer version updated)
  ✅ fast_utci/.specify/templates/spec-template.md (no changes required)
  ✅ fast_utci/.specify/templates/tasks-template.md (no changes required)
  ✅ fast_utci/.specify/templates/agent-file-template.md (no changes required)
- Follow-up TODOs: none
-->

# fast-utci Constitution

## Core Principles

### I. Code Quality Is Non-Negotiable
All production code MUST be clear, maintainable, and verified by tests.
- Mandatory typing for public functions and module APIs.
- Lint and format checks MUST pass before merge.
- Unit and integration tests MUST cover critical paths; regressions require tests.
- Readability over cleverness; prefer explicit names and straightforward control flow.
Rationale: High code quality reduces defects, speeds iteration, and enables safe performance
work on core algorithms.

### II. Runtime Performance Focused on Ray Casting
Ray casting and MRT-critical paths MUST meet explicit performance budgets.
- Establish and track budgets (e.g., rays/sec throughput, p95 compute time, memory caps).
- Optimize with algorithmic improvements first; then data layout/vectorization.
- Prefer batch operations and spatial indices; avoid per-ray Python loops.
- Profile routinely; changes impacting hot paths MUST include before/after metrics.
Rationale: UTCI maps depend on fast, reliable ray interactions; sustained performance is
essential for usability and scale.

### III. Consistent User Experience
Interfaces and outputs MUST be predictable and consistent across tools.
- Stable CLI flags and config keys; consistent defaults and help text.
- Deterministic output schemas and file formats; consistent units and coordinate frames.
- Visualization consistency: color scales, legends, and labeling MUST match across views.
- Errors use actionable messages; logs are structured and leveled.
Rationale: Consistency reduces user errors, shortens onboarding, and enables automation.

### IV. No Over‑Engineering (YAGNI)
Implement the simplest solution that satisfies current requirements.
- Minimize dependencies; introduce abstractions only when justified by use cases.
- Prefer incremental refactors over speculative architectures.
- Avoid premature generalization; keep modules small and purpose-driven.
Rationale: Simplicity accelerates delivery, improves reliability, and reduces maintenance
burden.

## Additional Constraints & Performance Standards

- Performance targets (initial):
  - Ray casting throughput: define per-model baseline and track p50/p95 deltas in PRs.
  - End-to-end single-hour UTCI run target: record baseline wall time; PRs report delta.
  - Memory ceiling: document peak RSS during representative runs; avoid >10% growth
    without rationale.
- Environment requirements: Radiance and Ladybug Tools versions MUST be documented in
  README and pinned in CI.
- Reproducibility: provide seeds/configs for deterministic runs where applicable.

## Development Workflow & Quality Gates

- PR Checklist (MUST PASS for merge):
  - Lint/format, unit/integration tests, and type checks pass.
  - For changes in ray/MRT hot paths: include profiling notes and before/after metrics.
  - User-facing flags, outputs, or visuals: update help/docs and confirm consistency.
  - Complexity justification for any new dependency or abstraction.
- CI SHOULD run representative performance smoke tests on a small scene.
- Release notes MUST call out performance-impacting changes and UX changes.

## Governance

- This constitution supersedes other process documents where conflicts exist.
- Amendments:
  - Propose via PR including redline and migration/rollout notes if needed.
  - Versioning follows semantic rules: MAJOR for breaking governance changes; MINOR for
    new principles or materially expanded guidance; PATCH for clarifications.
  - Ratification occurs on merge; Last Amended date set to merge date.
- Compliance:
  - Reviewers enforce gates above. Deviations MUST be documented in the PR body with a
    time-bounded plan to return to compliance.

**Version**: 1.0.0 | **Ratified**: 2025-09-23 | **Last Amended**: 2025-09-23