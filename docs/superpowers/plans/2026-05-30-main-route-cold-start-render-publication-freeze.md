# Main Route Cold-Start Render Publication Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`. Execute each implementation task with a fresh implementation subagent. After each task that changes files, run a fresh spec-compliance review first; only if that returns APPROVED may a fresh code-quality reviewer run. No commits, no git worktrees.

**Goal:** Reduce the main-route 0.5m cold-path desktop freeze by shrinking the initial compute-buffer UTCI surface publication, while restoring proof that first visible and first post-visible scrub both stay healthy.

**Architecture:** Keep the main route `/` as the proof surface. First repair the stale cold-start evidence path so it reflects current diagnostics and records the first post-visible scrub. Then replace only the compute-buffer render surface geometry with an indexed shared grid that derives the cell lookup in the node material, leaving CPU-uploaded fallback geometry, tooltip/picking, exposure scheduling, and route UX unchanged.

**Tech Stack:** Svelte 5, TypeScript, Vitest, Playwright, WebGPU, Three.js WebGPU/TSL, existing `window.__utciRenderDiagnostics__`, `selectedHourRuntimeContract`, and `main-route-cold-start-waterfall` artifacts.

---

## Hard Constraints

- No commits.
- No git worktrees.
- Preserve unrelated dirty files, especially the existing `.gitignore` change.
- Do not change overlay copy or route UX.
- Do not make lazy exposure the main direction.
- Keep evidence on `/`, not `/debug`, with no `.bin`, Python reference, or parity comparison.
- Every first-visible improvement claim must be paired with first post-visible scrub proof.

## Decision Summary

Explorer consensus:

- Exposure queue wait is still the largest cold bucket, but the prior lazy-exposure proof showed that moving exposure into background can make the first scrub catastrophically worse.
- The strongest desktop-freeze suspect is dense initial render publication: current Ness Tziona 0.5m evidence creates about `49,030,566` render vertices and about `653,741,904` render-owned bytes for the initial compute-buffer surface.
- Current checkout does not contain the newer `main-route-cold-start-guardrails.spec.ts`, and the cold-start markdown is partly stale versus JSON. Repair evidence before trusting any optimization numbers.

## Perspective Ensemble

### Panel A - Council

- **Performance lens:** exposure is bigger overall, but render publication is the clearest main-thread freeze vector -> attack the synchronous non-indexed surface creation first after proof repair.
- **Interaction lens:** first visible alone is not success -> record first post-visible scrub in the cold artifact and fail proof if the initial optimization steals from the first interaction.
- **Maintainability lens:** keep the change inside `gpuUtciRenderBridge.ts` and existing render publication diagnostics -> avoid a new route-level rendering policy.
- **Correctness lens:** preserve compute-buffer selected-hour storage, zero visible readback, and existing cell-to-point mapping -> do not change tooltip/picking or selected-hour compute.

### Panel B - Red Cell

- **Attack target:** replacing the surface geometry could make colors subtly wrong if per-cell lookup still depends on `vertexIndex / 6`.
- **Failure scenario:** an indexed shared grid reuses vertices across cells; if color lookup remains vertex-index based, neighboring cells bleed or sample the wrong point.
- **Mitigation:** write a unit/source-lock test requiring compute-buffer color lookup to derive `cellIndex` from local surface coordinates, not `vertexIndex / SURFACE_VERTICES_PER_CELL`; verify with dense route proof and hover proof.
- **Early warning:** `renderPublicationPointCount` stays right but color bands or tooltip proof drift; `hoverCellLookupProofStatus` stops being `same-point-confirmed`; `renderOwnedSelectedHourBytes` does not drop.
- **Review finding:** `positionLocal` is plausible for this TSL path, but whole-file string checks are not enough. Use scoped Vitest source-lock assertions plus geometry/index proof for the color-index contract; keep Playwright for route-level regression proof.

## File Structure

- Modify `viewer/src/lib/services/gpuUtciRenderBridge.ts`
  - Add indexed shared-grid geometry for compute-buffer UTCI surfaces only.
  - Keep the old non-indexed geometry helper reachable during implementation as an emergency fallback until visual and cold-route proof pass.
  - Keep CPU-uploaded selected-hour geometry unchanged.
  - Change compute-buffer color node to compute cell index from local surface coordinates.
  - Keep `cellToPointStorageAttribute` as the source of point identity.
- Modify `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
  - Report actual compute-buffer mesh geometry position count from `mesh.geometry.getAttribute('position').count`.
  - Add/report index count or draw-index count separately so the artifact distinguishes vertex-buffer memory reduction from draw workload.
- Modify `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
  - Add optional render-publication fields for index/draw count if `UTCIPointCloud.svelte` reports them.
- Modify `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`
  - Include new render-publication index/draw count fields in the controller idempotence comparison.
- Modify `viewer/tests/services/pointCloudService.surface.test.ts`
  - Add small geometry byte/vertex/index assertions for compute-buffer surfaces.
  - Update existing byte expectations.
  - Add compatibility tests for the new compute-buffer vertex-count contract.
- Modify `viewer/tests/compute/live-selected-hour-controller.test.ts`
  - Prove changes to render-publication index/draw count fields are not treated as idempotent diagnostics.
- Modify `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
  - Guard the specific `createUtciColorNode` body against reverting compute-buffer color lookup to `vertexIndex / SURFACE_VERTICES_PER_CELL`.
- Strengthen `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts` and `viewer/tests/services/pointCloudService.surface.test.ts` as the focused color-index proof
  - Avoid a brittle Playwright pixel test for per-cell color; camera framing, antialiasing, color management, and oracle setup make it high-maintenance.
  - Keep Playwright for main-route transport, byte-reduction, no-readback, interaction, and hover regression proof.
- Modify `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`
  - Stop depending on stale `diagnostics.coldStart` fields.
  - Record initial first visible and first post-visible scrub in the same artifact, guarded by a fresh `baseSurfaceRequestId`.
  - Keep bounded waits with last `window.__utciRenderDiagnostics__` dump on timeout.
- Modify `docs/performance/main-route-cold-start-waterfall.md`
  - Regenerate narrative from current JSON only after collection.

## Expected Geometry Impact

Current compute-buffer surface geometry creates non-indexed positions:

```ts
positions = width * height * 6 * 3 * 4 bytes
```

The new compute-buffer geometry uses shared vertices plus an index:

```ts
positions = (width + 1) * (height + 1) * 3 * 4 bytes
indices = width * height * 6 * 4 bytes
```

For Ness Tziona 0.5m (`2237 x 3653` cells), the render geometry portion should drop from roughly `588 MB` of positions to roughly `294 MB` of positions plus indices before the UTCI/cell buffers. This reduces vertex-buffer allocation and CPU fill work, not the indexed draw count. The artifact must report both actual position-vertex count and index/draw count so this is not misread as fewer rasterized triangles. This does not change the selected-hour compute buffer or exposure buffers.

---

### Task 1: Repair Cold Proof Surface

**Files:**
- Modify: `viewer/tests/e2e/main-route-cold-start-waterfall.spec.ts`
- Modify: `docs/performance/main-route-cold-start-waterfall.md`

- [ ] **Step 1: Update collector schema around current diagnostics**

Replace stale required `diagnostics.coldStart` assertions with current live fields:

```ts
type CollectedColdCase = {
	projectLabel: string;
	analysisId: string;
	gridResolutionMeters: 2 | 0.5;
	pointCount: number;
	sourceUrl: string;
	initial: {
		firstVisibleMs: number | null;
		timings: Record<string, number | null>;
		renderPublication: Record<string, unknown> | null;
		proof: Record<string, unknown>;
	};
	firstPostVisibleScrub: {
		selectedHourIndex: 1;
		selectedTimeIndex: 169;
		visibleMs: number | null;
		surfaceRequestId: number | null;
		timings: Record<string, number | null>;
		renderPublication: Record<string, unknown> | null;
		proof: Record<string, unknown>;
	};
	assertions: {
		pythonBinDebugComparisonFieldsAbsent: true;
		initialForbiddenComparisonFieldsPresent: string[];
		scrubForbiddenComparisonFieldsPresent: string[];
		allForbiddenRequestUrls: string[];
		memoryScope: 'utci-owned-webgpu-buffers';
	};
};
```

Keep the top-level artifact path as:

```ts
const ARTIFACT_FILENAME = 'main-route-cold-start-waterfall.json';
```

Replace the fixed `COLLECTED_ON` value with the actual collection date for the run, and update `collectionMethod` so it explicitly says the artifact collects both initial first visible and first post-visible scrub on `/`. Do not leave wording that says "initial selected hour only."

- [ ] **Step 2: Add first post-visible scrub measurement**

After the initial `waitForSelectedHourPublication(page, expectedSelectionKey)` succeeds, focus the main route hour dial and scrub to hour `1`:

```ts
const initialRequestId = initialDiagnostics.baseSurfaceRequestId ?? 0;
const scrubStartedAt = performance.now();
const hourSlider = page.getByRole('slider', { name: /select analysis hour/i });
await expect(hourSlider).toBeVisible();
await hourSlider.focus();
await hourSlider.press('Home');
await hourSlider.press('ArrowRight');
const scrubDiagnostics = await waitForSelectedHourPublication(
	page,
	caseConfig.expectedSelectionKey.replace('|7|0', '|7|1'),
	{ minSurfaceRequestId: initialRequestId }
);
const firstPostVisibleScrubMs = performance.now() - scrubStartedAt;
```

Update `waitForSelectedHourPublication` to accept a `minSurfaceRequestId?: number` option, using the same guard shape as `main-route-performance-0_5m.spec.ts`: the accepted diagnostics must have `baseSurfaceRequestId > minSurfaceRequestId` and `gpuResidentCopyRequestId === baseSurfaceRequestId`.

- [ ] **Step 3: Keep proof assertions identical for both phases**

Assert for initial and scrub diagnostics:

```ts
expect(new URL(sourceUrl, 'http://localhost').pathname).toBe('/');
expect(diagnostics.rendererBackend).toBe('webgpu');
expect(diagnostics.utciRenderResolved).toBe('gpuNative');
expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
expect(diagnostics.baseRenderTransport).toBe('compute-buffer-selected-hour');
expect(diagnostics.dataTextureBuildCount).toBe(0);
expect(diagnostics.selectedHourRuntimeContract?.route).toBe('main');
expect(diagnostics.selectedHourRuntimeContract?.readbackInstrumentation).toBe('instrumented');
expect(diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount).toBe(0);
expect(diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath).toBe(true);
expect(forbiddenComparisonFieldsPresent).toEqual([]);
expect(forbiddenRequestUrls).toEqual([]);
```

Compute forbidden comparison fields for both `initialDiagnostics` and `scrubDiagnostics`, and compute forbidden request URLs only after the scrub phase has completed. This prevents a scrub-time `.bin`, debug, or parity request from slipping through after the initial proof.

- [ ] **Step 4: Run the repaired collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list
```

Expected: pass and write `data/performance-results/main-route-cold-start-waterfall.json`.

- [ ] **Step 5: Refresh the evidence note from JSON**

Update `docs/performance/main-route-cold-start-waterfall.md` so the tables are generated from the current JSON values only. Include:

```md
| Project | Grid m | Points | Initial first visible ms | Exposure queue wait ms | Initial render update ms | Position vertices | Draw indices | Initial render-owned MiB | First post-visible scrub ms |
```

Do not reuse stale values from prior markdown.

---

### Task 2: Add Indexed Compute-Buffer Surface Tests

**Files:**
- Modify: `viewer/tests/services/pointCloudService.surface.test.ts`
- Modify: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
- Modify: `viewer/tests/compute/live-selected-hour-controller.test.ts`

- [ ] **Step 1: Write failing geometry test**

Add a test near `creates compute-buffer surfaces without uploading selected-hour UTCI from CPU readback`:

```ts
it('uses indexed shared-grid geometry for compute-buffer surfaces', () => {
	const mesh = createComputeBufferUtciSurfaceMesh({
		layout: {
			width: 2,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 1,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 1]),
			indexToTexel: new Uint32Array([0, 1]),
			colorBuffer: new Uint8Array(8)
		},
		utciBuffer: {} as GPUBuffer,
		utciRange: { min: 10, max: 40 }
	});

	expect(mesh.geometry.index?.array).toBeInstanceOf(Uint32Array);
	expect(mesh.geometry.index?.count).toBe(12);
	expect(mesh.geometry.getAttribute('position').count).toBe(6);
	expect(mesh.userData.gpuNativeUtciSurfaceState.vertexCount).toBe(6);
	expect(mesh.userData.renderOwnedSelectedHourBytes).toBe(1160);
});
```

- [ ] **Step 2: Write failing compatibility and controller diagnostics tests**

Add/update compatibility coverage in `pointCloudService.surface.test.ts` so compute-buffer compatibility expects `(layout.width + 1) * (layout.height + 1)` position vertices, while CPU-uploaded paths keep their existing `layout.width * layout.height * 6` expectation. Update any existing precomputed compatibility fixture that still hard-codes `layout.width * layout.height * 6` for compute-buffer surfaces.

In `live-selected-hour-controller.test.ts`, add/update a diagnostics idempotence test proving changes to `renderPublicationIndexCount` and `renderPublicationDrawIndexCount` are treated as meaningful render-publication changes.

- [ ] **Step 3: Write failing source guard scoped to the color-node body**

In `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`, add:

```ts
it('derives compute-buffer cell lookup from local surface coordinates', () => {
	const colorNodeBody = extractFunctionBody(renderBridgeSource, 'createUtciColorNode');
	expect(colorNodeBody).toContain('positionLocal.x');
	expect(colorNodeBody).toContain('positionLocal.z');
	expect(colorNodeBody).toContain('.floor()');
	expect(colorNodeBody).toContain('clamp(');
	expect(colorNodeBody).toContain('row.mul(uint(layout.width)).add(column)');
	expect(colorNodeBody).toContain('cellToPointStorage.element(cellIndex)');
	expect(colorNodeBody).toContain('utciStorage.element(pointIndex)');
	expect(colorNodeBody).not.toContain('vertexIndex.div(uint(SURFACE_VERTICES_PER_CELL))');
	expect(renderBridgeSource).toContain('createIndexedGridSurfaceGeometry');
	const computeBufferBody = extractFunctionBody(renderBridgeSource, 'createComputeBufferUtciSurfaceMesh');
	expect(computeBufferBody).toContain('createUtciColorNode(');
	expect(computeBufferBody).toContain('options.layout');
});
```

If the file does not already have a helper for function-body extraction, add a small test-local helper that brackets on the named function instead of whole-file `includes`.

- [ ] **Step 4: Run failing tests**

Run:

```powershell
cd viewer
npx vitest run tests/services/pointCloudService.surface.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/live-selected-hour-controller.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: focused unit/source-lock tests fail before implementation.

---

### Task 3: Implement Indexed Compute-Buffer Surface

**Files:**
- Modify: `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte`
- Modify: `viewer/src/lib/diagnostics/selectedHourRenderPublicationDiagnostics.ts`
- Modify: `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts`

- [ ] **Step 1: Import `positionLocal`**

Change the TSL import to include `positionLocal`:

```ts
import {
	clamp,
	float,
	positionLocal,
	storage,
	texture,
	uint,
	uniform,
	vec2,
	vertexIndex
} from 'three/tsl';
```

Keep `vertexIndex` because CPU-uploaded/synthetic paths still use it.

- [ ] **Step 2: Add vertex-count helper**

Add:

```ts
function getComputeBufferSurfaceVertexCount(layout: UtciGridLayout): number {
	return (layout.width + 1) * (layout.height + 1);
}
```

Use this helper in compute-buffer compatibility instead of `layout.width * layout.height * SURFACE_VERTICES_PER_CELL`.

- [ ] **Step 3: Add indexed grid geometry**

Add below `createGpuNativeSurfaceGeometry`:

```ts
function createIndexedGridSurfaceGeometry(layout: UtciGridLayout): THREE.BufferGeometry {
	const geometry = new THREE.BufferGeometry();
	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;
	const halfWidth = planeWidth / 2;
	const halfHeight = planeHeight / 2;
	const vertexWidth = layout.width + 1;
	const vertexHeight = layout.height + 1;
	const positions = new Float32Array(vertexWidth * vertexHeight * 3);
	const indices = new Uint32Array(layout.width * layout.height * SURFACE_VERTICES_PER_CELL);

	let positionOffset = 0;
	for (let row = 0; row < vertexHeight; row += 1) {
		const z = -halfHeight + row * layout.gridSize;
		for (let col = 0; col < vertexWidth; col += 1) {
			const x = -halfWidth + col * layout.gridSize;
			positions[positionOffset++] = x;
			positions[positionOffset++] = 0;
			positions[positionOffset++] = z;
		}
	}

	let indexOffset = 0;
	for (let row = 0; row < layout.height; row += 1) {
		for (let col = 0; col < layout.width; col += 1) {
			const v00 = row * vertexWidth + col;
			const v10 = v00 + 1;
			const v01 = v00 + vertexWidth;
			const v11 = v01 + 1;
			indices[indexOffset++] = v01;
			indices[indexOffset++] = v11;
			indices[indexOffset++] = v00;
			indices[indexOffset++] = v11;
			indices[indexOffset++] = v10;
			indices[indexOffset++] = v00;
		}
	}

	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geometry.setIndex(new THREE.BufferAttribute(indices, 1));
	geometry.computeBoundingBox();
	geometry.computeBoundingSphere();
	return geometry;
}
```

- [ ] **Step 4: Use indexed geometry only for compute-buffer mesh**

In `createComputeBufferUtciSurfaceMesh`, replace:

```ts
const geometry = createGpuNativeSurfaceGeometry(options.layout);
```

with:

```ts
const geometry = createIndexedGridSurfaceGeometry(options.layout);
```

Do not delete `createGpuNativeSurfaceGeometry` in this task. Keep it available as the emergency fallback until the visual proof and dense-route collector have passed.

- [ ] **Step 5: Report actual geometry and index counts**

In `UTCIPointCloud.svelte`, stop reporting:

```ts
renderPublicationVertexCount: layout.width * layout.height * 6
```

For the compute-buffer path, report actual mesh geometry:

```ts
const positionVertexCount = mesh.geometry.getAttribute('position')?.count ?? null;
const indexCount = mesh.geometry.index?.count ?? null;
```

Then pass:

```ts
renderPublicationVertexCount: positionVertexCount ?? undefined,
renderPublicationIndexCount: indexCount ?? undefined,
renderPublicationDrawIndexCount: indexCount ?? positionVertexCount ?? undefined,
```

Add optional fields to `SelectedHourRenderPublicationDiagnostics` for the new index/draw counts. Existing consumers may leave them absent for CPU-uploaded or older artifacts.

- [ ] **Step 6: Include new diagnostics fields in idempotence checks**

Update `areRenderPublicationEqual` in `liveSelectedHourController.ts` so `renderPublicationIndexCount` and `renderPublicationDrawIndexCount` are compared alongside `renderPublicationVertexCount`. The controller must not treat diagnostics as unchanged when only either new count differs.

- [ ] **Step 7: Derive cell index from local surface coordinates**

Change `createUtciColorNode` to accept `layout`:

```ts
function createUtciColorNode(
	layout: UtciGridLayout,
	utciStorageAttribute: StorageBufferAttribute,
	cellToPointStorageAttribute: StorageBufferAttribute,
	minUniform: ReturnType<typeof uniform>,
	maxUniform: ReturnType<typeof uniform>,
	colorLutTexture: THREE.DataTexture
) {
	const utciStorage = storage(utciStorageAttribute, 'float', utciStorageAttribute.count).toReadOnly();
	const cellToPointStorage = storage(
		cellToPointStorageAttribute,
		'uint',
		cellToPointStorageAttribute.count
	).toReadOnly();
	const halfWidth = float((layout.width * layout.gridSize) / 2);
	const halfHeight = float((layout.height * layout.gridSize) / 2);
	const gridSize = float(layout.gridSize);
	const column = uint(clamp(positionLocal.x.add(halfWidth).div(gridSize).floor(), 0, layout.width - 1));
	const row = uint(clamp(positionLocal.z.add(halfHeight).div(gridSize).floor(), 0, layout.height - 1));
	const cellIndex = row.mul(uint(layout.width)).add(column);
	const pointIndex = cellToPointStorage.element(cellIndex);
	const value = utciStorage.element(pointIndex);
	// keep existing t/lut color logic unchanged
}
```

Update the call site:

```ts
const { colorNode, opacityNode } = createUtciColorNode(
	options.layout,
	utciStorageAttribute,
	cellToPointStorageAttribute,
	minUniform,
	maxUniform,
	colorLutTexture
);
```

- [ ] **Step 8: Run focused tests**

Run:

```powershell
cd viewer
npx vitest run tests/services/pointCloudService.surface.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/live-selected-hour-controller.test.ts --no-file-parallelism --maxWorkers=1
```

Expected: pass after updating intentional byte-count expectations.

---

### Task 4: Verify Dense Main-Route Behavior

**Files:**
- Generated: `data/performance-results/main-route-cold-start-waterfall.json`
- Generated: `docs/performance/main-route-cold-start-waterfall.md`

- [ ] **Step 1: Run static check**

Run:

```powershell
cd viewer
npm run check
```

Expected: pass.

- [ ] **Step 2: Run cold collector**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-cold-start-waterfall.spec.ts --project=chromium --workers=1 --reporter=list
```

Expected: pass. The Ness Tziona 0.5m row should still prove:

```text
rendererBackend=webgpu
utciRenderResolved=gpuNative
utciSurfaceSource=compute-buffer-selected-hour
baseRenderTransport=compute-buffer-selected-hour
dataTextureBuildCount=0
selectedHourRuntimeContract.visibleSelectedHourReadbackCount=0
```

The expected performance direction is lower `renderPublicationVertexCount`, lower `renderPublicationRenderOwnedBytes`, lower `renderOwnedSelectedHourBytes`, and a lower `createComputeBufferSurfaceMeshMs` than the current JSON baseline. Do not claim success unless the artifact shows it.
`renderPublicationDrawIndexCount` should remain comparable to the old cell triangle draw count; this optimization is successful when vertex-buffer memory and mesh creation work drop without changing visible per-cell output.

- [ ] **Step 3: Run 0.5m interaction proof**

Run:

```powershell
cd viewer
npx playwright test tests/e2e/main-route-manual-diagnostics.spec.ts --project=chromium --workers=1 --reporter=list --grep "publishes Ness Tziona 0.5m camera and tooltip interaction diagnostics"
```

Expected: pass, with stable hover/interaction proof and no visible selected-hour readback.

- [ ] **Step 4: Run focused color-index contract proof**

Run the focused Vitest unit/source-lock proof from Task 2. It must prove the indexed compute-buffer geometry shape and the local-position color-index contract without relying on whole-file string matches. The main-route Playwright checks remain the browser-level regression proof.

---

### Task 5: Review Gates

**Files:**
- Review all changed files.

These gates run after each implementation task that changes files, not only once at the end. Use fresh reviewers for each gate. Do not start code-quality review until the spec-compliance reviewer returns APPROVED.

- [ ] **Step 1: Spec compliance review agent**

Dispatch a fresh reviewer. It must verify:

- no commits
- no git worktrees
- no overlay copy/UX change
- cold collector proves first visible and first post-visible scrub
- main-route proof boundary stays `/`, WebGPU, compute-buffer selected-hour, no readback
- lazy exposure remains evidence-only
- protected historical lazy artifact is not overwritten

- [ ] **Step 2: Code quality review agent**

Only after spec compliance is clean, dispatch a fresh reviewer. It must verify:

- compute-buffer geometry is the only rendering path changed
- CPU-uploaded fallback remains unchanged
- indexed geometry keeps the same plane placement and winding
- cell lookup no longer depends on `vertexIndex / 6`
- byte accounting includes geometry index bytes
- compatibility checks use the new compute-buffer vertex-count contract
- render-publication diagnostics report actual position vertices and separate index/draw count
- tests assert the intended memory and vertex-buffer reduction without claiming lower indexed draw count
- scoped source-lock and geometry proof catch regressions back to shared-vertex `vertexIndex / 6` lookup and boundary-prone coordinate math

- [ ] **Step 3: Final verification**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. Treat any `.gitignore` status as a pre-existing unrelated workspace change, outside this implementation scope; do not revert, unstage, or include it in this work.

## Self-Review

- Spec coverage: repairs stale proof, pairs first visible with first post-visible scrub, and targets the strongest desktop-freeze render-publication vector.
- Placeholder scan: no `TBD`, `TODO`, or unspecified tests remain.
- Type consistency: geometry helper, vertex count, byte accounting, and compatibility all use the same indexed compute-buffer shape.
- Constraint check: no commits, no git worktrees, main route only.
