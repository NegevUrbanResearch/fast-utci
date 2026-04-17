# UTCI Parity Final-Mile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate remaining strict parity blockers by restoring Python MRT-component artifacts, improving localized mismatch diagnostics, and tightening parity debug-loop speed/reliability.

**Architecture:** We keep the current proven compute baseline (weather-channel split + current ray epsilon) and focus on observability and artifact contracts. First, make Python reference exports always include MRT component terms so strict compare is meaningful. Then enrich parity diagnostics around the single flip/outlier cell and add fast-loop test defaults (`?parity=1`, low-cost oracle mode). Finally, verify with strict/stats + diagnostics and record a reproducible decision log.

**Tech Stack:** Python (fast_utci/Ladybug pipeline), TypeScript (viewer parity scripts/libs), Playwright, Vitest, Node/tsx.

---

## Scope and Constraints

- No git worktrees.
- Commit once per task.
- Prefer cheap implementer subagents for mechanical edits (`gpt-5.4-mini` low reasoning).
- Minimal tests per task (targeted unit/script checks, not full suite).
- Keep compute physics unchanged unless diagnostics prove a localized fix candidate.

## File Structure Map

- `scripts/export_ben_gurion_intermediates.py`
Role: Python reference artifact generator for `_solar.json`, `_sky.json`, `_mrt.json`, optional `_weather_sample.json`.
- `tests/test_export_ben_gurion_intermediates.py` (new)
Role: fast unit-level contract tests for exporter stage defaults and emitted MRT component schema.
- `viewer/src/lib/parity/buildParityReport.ts`
Role: richer stage diagnostics payload (`hourIndex`, `pointIndex`, margins, contribution metadata).
- `viewer/src/lib/parity/mrtWorstCellDiagnostics.ts`
Role: MRT-delta decomposition and dominant-term attribution.
- `viewer/scripts/diagnose-solar-flips.ts`
Role: localized flip diagnostics with explicit classification/margin fields.
- `viewer/scripts/diagnose-mrt-worst-cell.ts`
Role: report/print enriched MRT worst-cell cause breakdown.
- `viewer/tests/parity/buildParityReport.test.ts`
Role: regression tests for new report fields.
- `viewer/tests/parity/mrtWorstCellDiagnostics.test.ts`
Role: regression tests for term attribution output.
- `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
Role: parity collection harness (already patched to `?parity=1`; keep in sync).
- `viewer/tests/e2e/parity-intermediates.spec.ts`
Role: intermediate parity checks (already patched to `?parity=1`; add memory-safe export handling if needed).
- `viewer/tests/e2e/diagnose-solar-ray-oracle.spec.ts`
Role: ray-oracle diagnostics (already patched with fast defaults/env toggles).
- `docs/plans/2026-03-15-webgpu-python-parity-findings-and-lessons.md`
Role: findings log; append a dated "final-mile" update with new evidence and accepted/rejected changes.

## Task 1: Restore Python MRT Component Contract by Default

**Files:**
- Modify: `scripts/export_ben_gurion_intermediates.py`
- Create: `tests/test_export_ben_gurion_intermediates.py`

- [ ] **Step 1: Write failing Python tests for exporter defaults and MRT schema**

```python
# tests/test_export_ben_gurion_intermediates.py

def test_default_stages_include_mrt_components():
    ...

def test_mrt_json_contains_component_arrays_when_stage_mrt_enabled():
    ...
```

- [ ] **Step 2: Run targeted tests to confirm failure first**
Run: `python -m pytest tests/test_export_ben_gurion_intermediates.py -q`
Expected: FAIL on missing default-stage/MRT-component contract assertions.

- [ ] **Step 3: Implement minimal exporter changes**

```python
# scripts/export_ben_gurion_intermediates.py
# default stages should include mrt (and keep solar/sky)
stages = args.stage if args.stage else ["solar", "sky", "mrt"]
```

Also ensure docstring/usage text reflects default stages accurately.

- [ ] **Step 4: Run targeted tests to verify pass**
Run: `python -m pytest tests/test_export_ben_gurion_intermediates.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_ben_gurion_intermediates.py tests/test_export_ben_gurion_intermediates.py
git commit -m "fix(parity): include MRT component artifacts in default python export"
```

## Task 2: Regenerate Reference Artifacts and Add Contract Sanity Check Command

**Files:**
- Modify: `README.md` (or parity docs section with exact regeneration command)
- Modify: `docs/plans/2026-03-15-webgpu-python-parity-findings-and-lessons.md`

- [ ] **Step 1: Add a one-command artifact regeneration recipe**

```bash
python scripts/export_ben_gurion_intermediates.py \
  --base-path data/analyses/Ben-Gurion/20250815_grid_2m_fullday \
  --model data/3d_models/Ben-Gurion/original_with_layers.glb
```

- [ ] **Step 2: Add a lightweight schema check command in docs**

```bash
node -e "const fs=require('fs');const p='data/analyses/Ben-Gurion/20250815_grid_2m_fullday_mrt.json';const d=JSON.parse(fs.readFileSync(p,'utf8'));for(const k of ['mrt','short_erf','long_erf','short_dmrt','long_dmrt']){if(!Array.isArray(d[k])) throw new Error(k+' missing');}console.log('ok')"
```

- [ ] **Step 3: Verify docs commands locally**
Run both commands once.
Expected: artifact written + schema check prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/plans/2026-03-15-webgpu-python-parity-findings-and-lessons.md
git commit -m "docs(parity): document default MRT-component export and contract check"
```

## Task 3: Enrich Parity Report with Localized Index Context

**Files:**
- Modify: `viewer/src/lib/parity/buildParityReport.ts`
- Modify: `viewer/tests/parity/buildParityReport.test.ts`

- [ ] **Step 1: Write failing tests for enriched worst-index fields**

```ts
// assert worstIndices include hourIndex and pointIndex for pointwise arrays
expect(report.mrt?.detail?.worstIndices?.[0]).toMatchObject({
  index: expect.any(Number),
  hourIndex: expect.any(Number),
  pointIndex: expect.any(Number)
});
```

- [ ] **Step 2: Run targeted test to verify fail**
Run: `cd viewer && npx vitest run tests/parity/buildParityReport.test.ts -v`
Expected: FAIL on missing fields.

- [ ] **Step 3: Implement minimal field additions in report builder**

```ts
// compute from flat index + numHours
const pointIndex = Math.floor(index / numHours);
const hourIndex = index % numHours;
```

- [ ] **Step 4: Re-run targeted tests**
Run: `cd viewer && npx vitest run tests/parity/buildParityReport.test.ts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add viewer/src/lib/parity/buildParityReport.ts viewer/tests/parity/buildParityReport.test.ts
git commit -m "feat(parity): add point/hour context to worst-index diagnostics"
```

## Task 4: Add MRT Dominant-Term Attribution to Worst-Cell Diagnostics

**Files:**
- Modify: `viewer/src/lib/parity/mrtWorstCellDiagnostics.ts`
- Modify: `viewer/scripts/diagnose-mrt-worst-cell.ts`
- Modify: `viewer/tests/parity/mrtWorstCellDiagnostics.test.ts`

- [ ] **Step 1: Write failing tests for dominant term and contribution summary**

```ts
expect(row).toMatchObject({
  dominantTerm: expect.any(String),
  dominantTermDelta: expect.any(Number),
  termAbsSum: expect.any(Number)
});
```

- [ ] **Step 2: Run test to verify fail**
Run: `cd viewer && npx vitest run tests/parity/mrtWorstCellDiagnostics.test.ts -v`
Expected: FAIL.

- [ ] **Step 3: Implement minimal attribution fields and table print columns**

```ts
const dominantTerm = ...; // max abs term delta
const termAbsSum = ...;   // sum abs available term deltas
```

- [ ] **Step 4: Re-run targeted tests**
Run: `cd viewer && npx vitest run tests/parity/mrtWorstCellDiagnostics.test.ts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add viewer/src/lib/parity/mrtWorstCellDiagnostics.ts viewer/scripts/diagnose-mrt-worst-cell.ts viewer/tests/parity/mrtWorstCellDiagnostics.test.ts
git commit -m "feat(parity): add dominant-term attribution for MRT worst cells"
```

## Task 5: Enrich Solar-Flip Diagnostics with Classification Fields

**Files:**
- Modify: `viewer/scripts/diagnose-solar-flips.ts`
- (Optional minimal test) Create: `viewer/tests/parity/diagnoseSolarFlips.classification.test.ts`

- [ ] **Step 1: Add failing test for classification helper (if extracted)**

```ts
expect(classifyFlipCell(...)).toBe('raycast_localized');
```

- [ ] **Step 2: Run targeted test (or script smoke check if no helper extracted)**
Run: `cd viewer && npx vitest run tests/parity/diagnoseSolarFlips.classification.test.ts -v`
Expected: FAIL first (or skip test path if helper not extracted).

- [ ] **Step 3: Implement minimal fields without changing core scoring**
Add per-cell output fields:
- `binaryFlipDirection`
- `marginDelta`
- `shortWaveCompositeDelta`
- `longWaveCompositeDelta`
- `mismatchClass` (heuristic label)

- [ ] **Step 4: Run script smoke check**
Run: `cd viewer && npx tsx scripts/diagnose-solar-flips.ts --top 5`
Expected: JSON report includes new fields.

- [ ] **Step 5: Commit**

```bash
git add viewer/scripts/diagnose-solar-flips.ts viewer/tests/parity/diagnoseSolarFlips.classification.test.ts
git commit -m "feat(parity): classify solar flip cells with margin and term metadata"
```

## Task 6: Stabilize Fast Inner Loop and Memory-Safe E2E Defaults

**Files:**
- Modify: `viewer/tests/e2e/collect-webgpu-parity.spec.ts`
- Modify: `viewer/tests/e2e/parity-intermediates.spec.ts`
- Modify: `viewer/tests/e2e/diagnose-solar-ray-oracle.spec.ts`

- [ ] **Step 1: Confirm fast-loop defaults in all parity e2e entry points**
Ensure all parity diagnostics routes use `?parity=1`.

- [ ] **Step 2: Add memory-safe handling in intermediates test**
Avoid unnecessary large JSON duplication/parsing in test path.

- [ ] **Step 3: Verify with minimal e2e commands**
Run:
- `cd viewer && npm run parity:diagnose-ray-oracle`
- `cd viewer && npx playwright test --config=playwright.collect.config.ts tests/e2e/parity-intermediates.spec.ts --project=chromium --workers=1`
Expected: ray-oracle completes quickly; intermediates test no OOM.

- [ ] **Step 4: Commit**

```bash
git add viewer/tests/e2e/collect-webgpu-parity.spec.ts viewer/tests/e2e/parity-intermediates.spec.ts viewer/tests/e2e/diagnose-solar-ray-oracle.spec.ts
git commit -m "test(parity): enforce fast parity mode and reduce e2e memory pressure"
```

## Task 7: End-to-End Parity Verification and Findings Update

**Files:**
- Modify: `docs/plans/2026-03-15-webgpu-python-parity-findings-and-lessons.md`

- [ ] **Step 1: Recollect and compare with fresh artifacts**
Run:
- `cd viewer && npm run parity:collect-webgpu`
- `cd viewer && npx tsx scripts/compare-parity.ts --mode strict --report parity-report-strict-final-mile.json`
- `cd viewer && npx tsx scripts/compare-parity.ts --mode stats --report parity-report-stats-final-mile.json`
- `cd viewer && npx tsx scripts/diagnose-solar-flips.ts --top 25`
- `cd viewer && npx tsx scripts/diagnose-mrt-worst-cell.ts --top 15`

- [ ] **Step 2: Record concrete outcomes in findings doc**
Append dated section with:
- strict/stats summary
- worst-cell IDs
- whether component strict failures are resolved
- accepted/rejected next experiment list

- [ ] **Step 3: Verify regenerated Python MRT artifact is truly consumed by strict compare**
Run:
- `cd viewer && npx tsx scripts/compare-parity.ts --mode strict --report parity-report-strict-final-mile.json`
Expected:
- strict output should no longer report `short_erf/long_erf/short_dmrt/long_dmrt: FAIL (present in webgpu only)` for the baseline artifact.

- [ ] **Step 4: Minimal regression checks before final commit**
Run:
- `python -m pytest tests/test_export_ben_gurion_intermediates.py -q`
- `cd viewer && npx vitest run tests/parity/buildParityReport.test.ts tests/parity/mrtWorstCellDiagnostics.test.ts -v`

- [ ] **Step 5: Commit**

```bash
git add docs/plans/2026-03-15-webgpu-python-parity-findings-and-lessons.md viewer/parity-report-strict-final-mile.json viewer/parity-report-stats-final-mile.json
git commit -m "docs(parity): record final-mile verification evidence and next decisions"
```

## Subagent Execution Profile (Cost-Aware)

- Implementer per task: `gpt-5.4-mini`, `reasoning_effort=low`.
- Use one implementer at a time (tasks are stateful and overlap files).
- After each implementer task:
1. Spec compliance review pass (cheap explorer/reviewer).
2. Code quality sanity pass (cheap explorer/reviewer).
- Escalate to stronger model only when blocked by design ambiguity or repeated test failure.

## Verification Workflow (Required)

- Keep each task self-contained with one commit.
- Never proceed to next task with failing targeted tests from current task.
- Use only minimal checks listed in each task; avoid full-suite runs unless a task changes shared foundations.

## Exit Criteria

- Python `_mrt.json` for baseline includes `mrt`, `short_erf`, `long_erf`, `short_dmrt`, `long_dmrt` arrays with expected lengths.
- Strict compare failures are no longer due to one-sided component presence.
- Diagnostics outputs include point/hour and dominant-term context for worst cells.
- Ray-oracle loop remains fast for default mode (<30s typical local run).
- Findings doc updated with dated final-mile evidence and explicit next-action decision.
