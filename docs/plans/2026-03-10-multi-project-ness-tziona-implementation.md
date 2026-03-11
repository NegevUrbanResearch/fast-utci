# Multi-Project Analyses + Ness-Tziona Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore Ben-Gurion viewer functionality with the new per-project data layout, then add Ness-Tziona model switching and analysis output support.

**Architecture:** Keep a global `data/analyses/manifest.json` generated from the per-project folder tree. Viewer defaults to Ben-Gurion base analysis and uses a new project/model dropdown to switch between base analyses; scenario comparison remains for Ben-Gurion only. Analysis scripts write outputs to `data/analyses/<Project>/<Category?>` and embed correct model paths in metadata.

**Tech Stack:** Python 3.12 (analysis scripts), SvelteKit/TypeScript (viewer), Vitest (viewer tests).

---

**Git Policy:** Do not commit. The user will handle git manually.

---

### Task 1: Document current data layout + define project config

**Files:**
- Create: `viewer/src/lib/config/projects.ts`
- Modify: `viewer/src/routes/+page.svelte`

**Step 1: Write the failing test**

Create a minimal unit test for project config defaults.

```ts
import { describe, it, expect } from 'vitest';
import { getDefaultAnalysisId, projects } from '$lib/config/projects';

describe('projects config', () => {
  it('has Ben-Gurion as default project', () => {
    expect(getDefaultAnalysisId()).toBe('Ben-Gurion/20250815_grid_2m_fullday');
  });

  it('includes Ness-Tziona variants', () => {
    const nt = projects.find(p => p.id === 'Ness-Tziona');
    expect(nt?.models.length).toBe(2);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer; npm test -- --run tests/projectsConfig.test.ts`
Expected: FAIL (module not found).

**Step 3: Write minimal implementation**

Create `viewer/src/lib/config/projects.ts` with:
- `ProjectConfig` type
- `projects` array with:
  - Ben-Gurion: default analysis `Ben-Gurion/20250815_grid_2m_fullday`
  - Ness-Tziona: models `original` + `exploded` pointing at analysis IDs under `data/analyses/Ness-Tziona/<variant>/...`
- `getDefaultAnalysisId()` helper
- `getProjectById()` helper

**Step 4: Run test to verify it passes**

Run: `cd viewer; npm test -- --run tests/projectsConfig.test.ts`
Expected: PASS.


---

### Task 2: Add analysis/model path normalization helpers

**Files:**
- Create: `viewer/src/lib/utils/analysisPaths.ts`
- Create: `viewer/tests/analysisPaths.test.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { resolveProjectId, resolveModelPath } from '$lib/utils/analysisPaths';

describe('analysisPaths', () => {
  it('resolves project id from analysis id', () => {
    expect(resolveProjectId('Ben-Gurion/20250815_grid_2m_fullday')).toBe('Ben-Gurion');
  });

  it('fixes legacy BG model paths', () => {
    const fixed = resolveModelPath('data/3d_models/original_with_layers.glb', 'Ben-Gurion/20250815_grid_2m_fullday');
    expect(fixed).toBe('data/3d_models/Ben-Gurion/original_with_layers.glb');
  });

  it('keeps already-correct model paths', () => {
    const fixed = resolveModelPath('data/3d_models/Ness-Tziona/nes_tziona_1.gltf', 'Ness-Tziona/original/20250815_grid_2m_fullday');
    expect(fixed).toBe('data/3d_models/Ness-Tziona/nes_tziona_1.gltf');
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer; npm test -- --run tests/analysisPaths.test.ts`
Expected: FAIL (module not found).

**Step 3: Write minimal implementation**

Create `viewer/src/lib/utils/analysisPaths.ts` with:
- `resolveProjectId(analysisId)` -> first path segment or `null`
- `resolveModelPath(modelFile, analysisId)` -> if path starts with `data/3d_models/` and is missing project segment, insert project segment
- Ensure it corrects legacy BG paths like `data/3d_models/scenarios/...` to `data/3d_models/Ben-Gurion/scenarios/...`

**Step 4: Run test to verify it passes**

Run: `cd viewer; npm test -- --run tests/analysisPaths.test.ts`
Expected: PASS.


---

### Task 3: Restore Ben-Gurion default loading in viewer

**Files:**
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`

**Step 1: Write the failing test**

Add a unit test that ensures default analysis id is used when no query param is present. (Mock `window.location.search`.)

**Step 2: Run test to verify it fails**

Run: `cd viewer; npm test -- --run tests/defaultAnalysis.test.ts`
Expected: FAIL (current default lacks project prefix).

**Step 3: Write minimal implementation**

Update `viewer/src/routes/+page.svelte`:
- Set `DEFAULT_ANALYSIS_ID = getDefaultAnalysisId()`
- Use it for initial load and query fallback
- Use `resolveModelPath()` when computing `modelPath`

Update `viewer/src/lib/components/scene/ComparisonRenderer.svelte`:
- Use `resolveModelPath()` for comparison model path

**Step 4: Run test to verify it passes**

Run: `cd viewer; npm test -- --run tests/defaultAnalysis.test.ts`
Expected: PASS.


---

### Task 4: Add project/model dropdown in header

**Files:**
- Create: `viewer/src/lib/components/ui/ProjectSelector.svelte`
- Modify: `viewer/src/routes/+page.svelte`
- Modify: `viewer/src/lib/components/ui/ScenarioSelector.svelte`

**Step 1: Write the failing test**

Add a unit test to verify project selection updates analysis id (mock `loadAnalysis`).

**Step 2: Run test to verify it fails**

Run: `cd viewer; npm test -- --run tests/projectSelector.test.ts`
Expected: FAIL (component missing).

**Step 3: Write minimal implementation**

Create `ProjectSelector.svelte`:
- Dropdown listing `projects` from config
- On change, call a callback passed from `+page.svelte` with selected analysis id

Update `+page.svelte`:
- Render `ProjectSelector` in header-right
- On selection, call `loadAnalysis(id)` and update URL query using `goto` with `?analysis=`
- Derive `currentProjectId` via `resolveProjectId(analysisId)` and pass to `ScenarioSelector`

Update `ScenarioSelector.svelte`:
- Accept `projectId` prop
- If projectId !== `Ben-Gurion`, show a neutral �No scenarios for this project� state and disable list
- When constructing scenario analysis id, prefix with `Ben-Gurion/` (e.g., `Ben-Gurion/existing_buildings/existing_buildings_01`)

**Step 4: Run test to verify it passes**

Run: `cd viewer; npm test -- --run tests/projectSelector.test.ts`
Expected: PASS.


---

### Task 5: Update manifest generator for per-project layout

**Files:**
- Modify: `scripts/generate_manifest.py`
- Create: `data/analyses/manifest.json` (generated)

**Step 1: Write the failing test**

Create a small fixture-based test for manifest generation (Python). Use a temp dir with:
- `data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json`
- `data/analyses/Ben-Gurion/existing_buildings/existing_buildings_01.json`
- `data/analyses/Ness-Tziona/original/20250815_grid_2m_fullday.json`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_manifest.py -v`
Expected: FAIL (project parsing missing).

**Step 3: Write minimal implementation**

Update `scripts/generate_manifest.py`:
- Scan two levels deep: `data/analyses/<project>/*.json` and `data/analyses/<project>/<category>/*.json`
- Set `id` to path relative to analyses dir (e.g., `Ben-Gurion/20250815_grid_2m_fullday`)
- Include `project` and optional `category` fields
- Preserve `model_file`, `analysis_type`, etc.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_manifest.py -v`
Expected: PASS.


---

### Task 6: Update analysis scripts to write into per-project folders

**Files:**
- Modify: `run_analysis.py`
- Modify: `quick_analysis.py`
- Modify: `scripts/export_for_viewer.py`

**Step 1: Write the failing test**

Add a small unit test for `export_utci_for_viewer` output path calculation using a temp output dir and `project='Ben-Gurion'`, `category='existing_buildings'`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_for_viewer_paths.py -v`
Expected: FAIL (project not supported).

**Step 3: Write minimal implementation**

Update `scripts/export_for_viewer.py`:
- Add optional `project` parameter
- Compute output directory as `output_dir / project / category?`

Update `run_analysis.py`:
- Add `project` param (default `Ben-Gurion`)
- Compute `output_dir` as `data/analyses/<project>`
- Ensure `model_file` uses correct project path

Update `quick_analysis.py`:
- Add configs for `Ben-Gurion` and `Ness-Tziona` (original/exploded)
- Include `project` and `category` where relevant

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_export_for_viewer_paths.py -v`
Expected: PASS.


---

### Task 7: Regenerate manifest and verify Ben-Gurion viewer (local + GH Pages)

**Files:**
- Modify: `data/analyses/manifest.json`

**Step 1: Generate manifest**

Run: `python scripts/generate_manifest.py`
Expected: `data/analyses/manifest.json` created with project entries.

**Step 2: Local viewer verification (Ben-Gurion)**

Run: `cd viewer; npm run dev`
Expected: Default loads Ben-Gurion base analysis; scenario selection works; models load from `data/3d_models/Ben-Gurion/...`.

**Step 3: Static build verification (GH Pages)**

Run: `cd viewer; npm run build`
Expected: Build succeeds and the base path logic still resolves `.../data/...` correctly.


---

### Task 8: Run Ness-Tziona analyses for original + exploded

**Files:**
- Modify: `data/analyses/Ness-Tziona/original/*`
- Modify: `data/analyses/Ness-Tziona/exploded/*`

**Step 1: Run original model analysis**

Run:
```
python quick_analysis.py --config ness-tziona-original
```
Expected: New `.bin` + `.json` under `data/analyses/Ness-Tziona/original/` with correct `model_file` path.

**Step 2: Run exploded model analysis**

Run:
```
python quick_analysis.py --config ness-tziona-exploded
```
Expected: New `.bin` + `.json` under `data/analyses/Ness-Tziona/exploded/`.

**Step 3: Regenerate manifest**

Run: `python scripts/generate_manifest.py`
Expected: Ness-Tziona entries appear.

**Step 4: Manual viewer check**

Use project dropdown to switch to Ness-Tziona original/exploded; verify model loads and analysis data renders.


---

### Task 9: Cleanup + documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-03-10-multi-project-ness-tziona-implementation.md`

**Step 1: Update README**

Add a short �Projects + analyses layout� section and the new quick_analysis configs for Ness-Tziona.

**Step 2: Verify docs**

Run: `rg -n "Ness-Tziona|Ben-Gurion" README.md` and confirm references align.


---

