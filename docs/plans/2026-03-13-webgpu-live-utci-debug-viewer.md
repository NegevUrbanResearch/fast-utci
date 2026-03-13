# WebGPU Live UTCI Debug Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated debug viewer route that uses the existing curtain UI to compare current `.bin`-driven UTCI against live WebGPU-computed UTCI for the same project/scenario, reusing the existing scene, model loading, and depth-stable rendering stack.

**Architecture:** Build a thin adapter that runs the existing `ComputeManager` / `UTCIComputePipeline` on the same grid as the `.bin` analysis and wraps the result in an `Analysis`-compatible shape, then compose a new Svelte route that mounts the usual model + `.bin` UTCI overlay on one side and the live-WebGPU UTCI overlay on the other, using a simplified scissor component and the existing `ComparisonCurtain`. All heavy lifting (geometry, depth, shaders, compute) is reused.

**Tech Stack:** SvelteKit (Threlte), Three.js `WebGPURenderer`, TypeScript, existing `compute-manager` + WGSL pipeline, Vitest.

---

### Task 1: Reconfirm compute and analysis contracts

**Files:**
- Read: `viewer/src/lib/compute/compute-manager.ts`
- Read: `viewer/src/lib/compute/gpu-pipeline.ts`
- Read: `viewer/src/lib/compute/grid-generator.ts`
- Read: `viewer/src/lib/services/pointCloudService.ts`
- Read: `viewer/src/lib/types/analysis.ts`

**Step 1: Inspect `ComputeManager` and pipeline outputs**

- Verify what `ComputeManager` currently exposes for:
  - Grid generation (rectangular vs surface).
  - UTCI (and related fields) as buffers (Float32Arrays, etc.).
  - Any existing convenience methods that already look like “analysis”-style outputs.

**Step 2: Inspect `Analysis` and point-cloud/UTCI expectations**

- Confirm the minimal subset of `Analysis` needed by:
  - `createUtciSurfaceMesh` and `updateUtciSurfaceTexture` in `pointCloudService.ts`.
  - `UTCIPointCloud.svelte` (if used for point clouds in addition to the textured plane).
- Identify required fields:
  - `data.positions`, `data.numPositions`.
  - `data.utci` (and/or related metric arrays).
  - `metadata` fields that `createColors` / `resolveUtciRange` depend on (e.g. `utci_range`, `hour_statistics`, `coordinate_system`, `grid_size`).

**Step 3: Capture constraints**

- Note any differences between the Python `.bin` analysis format and the WebGPU compute format that will *not* be bridged in this phase (e.g. exact MRT semantics, sky/solar exposure parity).
- Write short notes in this plan for future reference (no code changes yet).

---

### Task 2: Design a lightweight live-UTCI analysis adapter

**Files:**
- Create (design only in this task): `viewer/src/lib/compute/liveUtciAnalysis.ts`

**Step 1: Define adapter responsibilities**

- The adapter should:
  - Accept: project/scenario identifiers (e.g. `analysisId`), EPW path, model bounds/coordinate system, and grid config.
  - Internally:
    - Use `ComputeManager` to run the full pipeline over the chosen grid (rectangular grid mode for now, matching the `.bin` resolution as closely as practical).
  - Return:
    - An object that satisfies enough of the `Analysis` interface to be treated like a `.bin`-driven analysis by `pointCloudService` (even if some metadata is faked or approximated for now).

**Step 2: Sketch TypeScript interfaces**

- Design (in comments or a small draft interface) something like:

```ts
export interface LiveUtciAnalysis extends Analysis {
  // Optionally mark this as live vs bin-backed for debugging
  __source?: 'webgpu';
}
```

- Decide whether to:
  - Extend `Analysis` directly, or
  - Create a minimal `LiveAnalysisLike` type and then provide a narrow wrapper for `createUtciSurfaceMesh`.

**Step 3: Decide grid strategy for this phase**

- Choose:
  - Either: always use rectangular grid (bbox-based) with spacing from `metadata.grid_size`.
  - Or: use the existing surface grid (raycast), if that is closer to the `.bin` data for your current projects.
- Document the choice in this plan so we don’t forget the intended parity strategy later.

---

### Task 3: Implement the `liveUtciAnalysis` adapter (with focused tests)

**Files:**
- Create: `viewer/src/lib/compute/liveUtciAnalysis.ts`
- Create: `viewer/tests/compute/live-utci-analysis.test.ts`

**Step 1: Implement minimal adapter in `liveUtciAnalysis.ts`**

- Implement a function such as:

```ts
export async function createLiveUtciAnalysisFromCompute(
  params: { analysisId: string; metadata: AnalysisMetadata; /* other needed inputs */ }
): Promise<Analysis> {
  // 1. Build grid for this model/analysis (rectangular or surface).
  // 2. Invoke ComputeManager / UTCIComputePipeline with EPW + grid + sun/sky.
  // 3. Map computed positions and UTCI values into Analysis.data.*
  // 4. Populate minimal Analysis.metadata fields required by pointCloudService.
}
```

- Use existing helpers as much as possible:
  - EPW loading via `epw-parser.ts` / dataLoader.
  - Sun/sky via `sunpath.ts` and `tregenza.ts`.
  - Grid via `grid-generator.ts`.
  - Compute orchestration via `compute-manager.ts`.

**Step 2: Write a small but reusable test in `live-utci-analysis.test.ts`**

- Only add tests that will be useful going forward:
  - Example: verify that for a tiny synthetic model/grid (e.g. 4–9 points) and a simple test EPW, the adapter:
    - Produces `data.numPositions` matching the grid size.
    - Fills `data.positions` with the transformed world coordinates as expected.
    - Produces a `data.utci` array of the right length with finite numbers.
    - Sets `metadata.coordinate_system`, `metadata.grid_size`, and `metadata.utci_range` consistently.
- Structure test like:

```ts
it('produces Analysis-like shape for a simple grid', async () => {
  const analysis = await createLiveUtciAnalysisFromCompute(fakeParams);
  expect(analysis.data.numPositions).toBe(/* expected */);
  expect(analysis.data.positions.length).toBe(analysis.data.numPositions * 3);
  expect(analysis.data.utci.length).toBe(analysis.data.numPositions);
  expect(analysis.metadata.utci_range.min).toBeLessThan(analysis.metadata.utci_range.max);
});
```

**Step 3: Run targeted tests**

- Run:
  - `cd viewer`
  - `npx vitest run tests/compute/live-utci-analysis.test.ts`
- Ensure the test passes and that no existing tests fail.

---

### Task 4: Create a dedicated debug route shell

**Files:**
- Create: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Read: `viewer/src/routes/+page.svelte`

**Step 1: Copy and slim the main viewer scaffold**

- Use the main `+page.svelte` as a template, but:
  - Keep:
    - Header/logo, project selector, time selector, layer controls, analytics panel only as needed for context.
    - `Scene`, `Camera`, `Lights`, `Model`, and base `.bin` `UTCIPointCloud` wiring.
  - Remove:
    - The existing comparison mode wiring that loads comparison GLBs/analyses (`ScenarioSelector` + `ComparisonRenderer` that handles base vs scenario).

**Step 2: Add a non-routed toggle/label indicating this is a debug view**

- At the top of the debug route, clearly mark:
  - “WebGPU UTCI Debug Viewer – `.bin` vs live compute (no parity guaranteed)”
- This helps avoid confusion with the primary viewer route.

**Step 3: Wire project / scenario selection re-use**

- Reuse:
  - `ProjectSelector` to choose BG vs NTZ.
  - `ScenarioSelector` to choose specific scenarios where they already exist (for Ben-Gurion).
- Ensure the debug route:
  - Observes the same `analysisStore` and `viewerStore` patterns as the main route to get the current `.bin` `Analysis` and its metadata.

---

### Task 5: Implement a dedicated scissor renderer for `.bin` vs live UTCI

**Files:**
- Create: `viewer/src/lib/components/scene/DebugUtciScissor.svelte`
- Read: `viewer/src/lib/components/scene/ComparisonRenderer.svelte`

**Step 1: Design `DebugUtciScissor` props**

- The component should take:

```ts
export let baseCamera: PerspectiveCamera | undefined;
export let binUtciMesh: THREE.Mesh | null;
export let liveUtciMesh: THREE.Mesh | null;
```

- Optionally accept references to the shared `Scene` and `Renderer` via `useThrelte()` rather than as props.

**Step 2: Reuse scissor logic from `ComparisonRenderer`**

- In `DebugUtciScissor.svelte`:
  - Use `useThrelte` and `useTask` to:
    - Access `renderer` and `scene`.
    - Sync a secondary `PerspectiveCamera` with `baseCamera` each frame.
    - Compute canvas `width`/`height` via `renderer.getSize`.
    - Implement scissor rectangles using the existing `curtainPosition` store from `comparisonStore`.
  - Render:
    - Left side: full scene with `.bin` UTCI mesh visible; live UTCI hidden.
    - Right side: same scene with live UTCI visible; `.bin` UTCI hidden.
  - Ensure:
    - Visibility toggling only affects the UTCI meshes, not the rest of the scene.
    - Any `setScissorTest` calls are guarded (`canToggleScissorTest`) as in the updated `ComparisonRenderer`.

**Step 3: Integrate `DebugUtciScissor` into the debug route**

- In `debug-webgpu-utci/+page.svelte`:
  - Bind the `.bin` UTCI mesh from `UTCIPointCloud` and the live UTCI mesh from a new `LiveUTCIPointCloud` or direct call to `createUtciSurfaceMesh(liveAnalysis, ...)`.
  - Mount:

```svelte
<Scene ...>
  <Camera bind:cameraRef ... />
  <Lights />
  <Model ... />
  <UTCIPointCloud
    analysis={$analysisStore}
    {model}
    bind:utciSurface={binUtciMesh}
  />
  {#if binUtciMesh && liveUtciMesh}
    <DebugUtciScissor
      {baseCamera}
      binUtciMesh={binUtciMesh}
      liveUtciMesh={liveUtciMesh}
    />
  {/if}
</Scene>
```

- Use the existing `ComparisonCurtain` UI for the screen overlay, but **without** starting the base/scenario comparison logic.

---

### Task 6: Wire live WebGPU UTCI into the debug route

**Files:**
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Modify or Create: `viewer/src/lib/components/scene/LiveUTCIPointCloud.svelte` (optional wrapper)

**Step 1: Trigger live analysis when analysis changes**

- In the debug route:
  - Watch `$analysisStore` (current base analysis) and the current project/scenario selection.
  - On change:
    - Call `createLiveUtciAnalysisFromCompute(...)`.
    - Store the resolved `Analysis`-like object in a local store or component state.

**Step 2: Create live UTCI mesh**

- Either:
  - Directly call `createUtciSurfaceMesh(liveAnalysis, ...)` in the route and keep a `liveUtciMesh: Mesh | null` reference, or
  - Wrap it in a small `LiveUTCIPointCloud.svelte` component similar to `UTCIPointCloud.svelte` for consistency.
- Ensure:
  - The mesh is added to the same scene as the `.bin` mesh.
  - Both use the same normalization and coordinate system.

**Step 3: Keep visibility controlled by viewer state**

- Respect `viewerStore` flags (e.g. `utciVisible`, current hour, color mode) for both `.bin` and live UTCI meshes.
- Keep both meshes’ `userData.utciLayout` populated so tooltips and updates still work if you want them to.

---

### Task 7: Manual testing and visual validation

**Files:**
- Run: no code changes; manual QA

**Step 1: Run the dev server**

- From `viewer`:
  - `npm run dev`

**Step 2: Exercise debug route**

- Navigate to:
  - `/debug-webgpu-utci` (or the chosen route path).
- For each project:
  - Ben-Gurion:
    - Default grid/full-day scenario.
    - Several scenario variants (existing trees, new buildings, etc.).
  - Ness Tziona:
    - At least one available analysis.
- For each:
  - Confirm:
    - Base geometry and UTCI overlays render correctly (no clipping, no regressions).
    - Scissor curtain smoothly transitions between `.bin` (left) and live WebGPU (right).
    - Depth interactions match expectations (buildings occlude UTCI on both sides).

**Step 3: Capture observations**

- Note any obvious numerical mismatches between `.bin` and live WebGPU:
  - E.g., consistent offset, wrong sun direction, drastically different hot/cold zones.
- These will feed future parity tasks but do not block this feature.

---

### Task 8: Review and handoff

**Files:**
- This plan document

**Step 1: Summarize implementation status**

- Once implemented:
  - Summarize how the live vs `.bin` comparison behaves.
  - Note any technical debt or shortcuts in the adapter (e.g., approximate metadata).

**Step 2: Offer execution options**

- After the plan is reviewed:
  - Option 1: Execute tasks now in this repo using subagent-driven development (`superpowers:subagent-driven-development`).
  - Option 2: Use a separate session with `superpowers:executing-plans` to implement this plan with checkpoints.

