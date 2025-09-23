
# Implementation Plan: Near Real-Time MRT/Raycasting Performance

**Branch**: `001-specify-we-have` | **Date**: 2025-09-23 | **Spec**: specs/main/spec.md
**Input**: Feature specification from `/specs/main/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Primary requirement: Reduce MRT/raycasting time for a 10 m grid city-block scene to approach near real-time while keeping accuracy within ≤ 2.0 °C UTCI RMSE. Provide progressive feedback (coarse-to-fine and/or streamed tiles), support cancelation, and maintain responsiveness on baseline 8‑core CPU integrated-graphics laptops.

Technical approach candidates (see research.md): adopt efficient acceleration structures and batched ray queries (BVH/Embree or GPU where available), reduce variance via importance sampling and stratification, cache sky factors and reuse across frames, pre-tile/voxelize scene for chunked updates, and parallelize across cores with memory-efficient batching.

## Technical Context
**Language/Version**: Python 3.x (repo)  
**Primary Dependencies**: numpy, pandas; scene I/O via glTF/GLB; MRT/UTCI custom modules  
**Storage**: Files (CSV/HTML outputs)  
**Testing**: pytest (assumed), manual validation notebooks/HTML  
**Target Platform**: Windows laptops (baseline), optional GPU acceleration if present  
**Project Type**: single  
**Performance Goals**: ≤ 60 s first full update for typical city-block (≤ 1M tris), progressive feedback faster  
**Constraints**: ≤ 2.0 °C UTCI RMSE vs baseline; responsive UI; cancellable jobs  
**Scale/Scope**: City block scenes (≤ 1M tris); grid spacing typically 10 m (others possible)

## Constitution Check
- No PII/security requirements central to scope.
- Keep design simple, iterative, and testable; avoid premature over-engineering.
- Ensure measurable acceptance: latency, RMSE, scene size recorded in spec.

Gates: PASS (initial).

## Project Structure

### Documentation (this feature)
```
specs/main/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: DEFAULT to Option 1

## Phase 0: Outline & Research
See `research.md` for decisions, rationale, alternatives. Unknowns carried forward: exact throughput target (samples/s), stress-mode latency budget.

**Output**: research.md complete.

## Phase 1: Design & Contracts
- Entities extracted to `data-model.md`.
- Initial contracts documented in `contracts/README.md` (local CLI/UI interactions and validation suite hooks).
- Quickstart path in `quickstart.md` to run baseline vs fast comparison on sample scene.

**Output**: data-model.md, /contracts/*, quickstart.md

## Phase 2: Task Planning Approach
This phase will be executed by /tasks. We will generate tasks from Phase 1 artifacts prioritizing: profiling, acceleration structure integration, sampling strategy, parallelization, caching, progress/cancelation, and validation.

## Phase 3+: Future Implementation
(Out of scope for /plan)

## Complexity Tracking
| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|

## Progress Tracking
**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v1.0.0 - See `/memory/constitution.md`*
