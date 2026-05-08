# WebGPU Compute-On-Demand Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype and verify a GPU-resident UTCI compute-on-demand path before replacing the current all-hours readback and CPU texture pipeline.

**Architecture:** Keep the current `runAll()` all-hours WebGPU path as the production/parity baseline. First prove the renderer/compute device ownership and a Three.js WebGPU storage bridge with synthetic data, then split exposure precompute from one-hour MRT/UTCI compute, then validate `f32` one-hour parity, and only then test `pack2x16float` as an optional storage format.

**Tech Stack:** SvelteKit, Three.js r175 `WebGPURenderer`, Three TSL/node materials, WGSL, WebGPU storage buffers, Vitest, Playwright for browser/WebGPU verification.

**User Constraints:** Do not create git worktrees. Do not commit. This plan intentionally contains no commit steps.

---

## Scope And Hard Gates

This is a prototype plan. It must not replace the production live analysis path until the gates below are proven.

Hard gates:

1. **Same-device gate:** compute buffers and renderer resources must be owned by the same `GPUDevice`, or the bridge is not viable.
2. **Bridge behavior gate:** synthetic GPU-resident values must visibly affect rendered color. A constant white `PointsMaterial` does not count.
3. **Exposure-only gate:** the prototype must precompute solar/sky exposure without allocating all-hours UTCI/MRT buffers.
4. **One-hour parity gate:** one-hour `f32` on-demand output must match the existing all-hours slice, including current boundary averaging semantics.
5. **No-hot-path-readback gate:** debug readback is allowed for validation only; the render smoke path must not depend on readback or CPU `DataTexture` rebuilds.
6. **Packed-output gate:** `pack2x16float` remains optional unless measured precision and category-flip results justify it.

Out of scope:

- Full production replacement.
- Spatial tiling for 0.5m.
- Future wind/sun-rights layers.
- Commits or git worktrees.

## Current Constraints From Repo Review

- `viewer/src/lib/compute/webgpuUtciPipeline.ts` creates/caches its own `GPUDevice`.
- `viewer/src/lib/components/scene/Scene.svelte` creates a Three `WebGPURenderer`, which may create its own device.
- Three r175 `StorageBufferAttribute` takes `count | TypedArray`, `itemSize`, and optional typed array constructor. It does **not** wrap an arbitrary raw `GPUBuffer`.
- Current `runAll()` computes solar/sky but also allocates all-hours UTCI/MRT buffers.
- Current production shader averages UTCI at `time_idx` and `next_idx` within the same day. The on-demand shader must preserve this.

## File Structure

Modify:

- `viewer/src/lib/compute/gpu-pipeline.ts`  
  Add optional prototype methods and pure output metadata types.

- `viewer/src/lib/compute/compute-manager.ts`  
  Add wrappers for exposure-only precompute and one-hour compute when supported.

- `viewer/src/lib/compute/webgpuUtciPipeline.ts`  
  Add device/limit diagnostics, optional injection/use of a shared device, exposure-only precompute, one-hour output, and validation readbacks.

- `viewer/src/lib/components/scene/Scene.svelte`  
  Expose enough renderer/backend/device diagnostics for the prototype route to prove device ownership. Do not change normal rendering behavior.

- `viewer/src/routes/debug-webgpu-utci/+page.svelte`  
  Add `?onDemandPrototype=1` status/diagnostics and prototype execution.

- `docs/webgpu_strategy_analysis.md`  
  Link to this plan and later to prototype results.

Create:

- `viewer/src/lib/compute/onDemandSizing.ts`  
  Pure memory/dispatch sizing helpers.

- `viewer/src/lib/compute/onDemandOutputFormat.ts`  
  `f32-utci` and `packed-mrt-utci` format contract.

- `viewer/src/lib/compute/shaders/mrt_utci_on_demand.wgsl`  
  One-hour shader. Starts with placeholder compute, then receives full MRT/UTCI parity math.

- `viewer/src/lib/services/gpuUtciRenderBridge.ts`  
  Prototype bridge using Three-owned storage attributes/TSL or storage texture. This file owns Three API churn.

- `viewer/tests/compute/onDemandSizing.test.ts`
- `viewer/tests/compute/onDemandOutputFormat.test.ts`
- `viewer/tests/compute/compute-manager-on-demand.test.ts`
- `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`
- `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`
- `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`

## Milestone 1: Lock Memory Math And Format Contract

### Task 1: Add Pure On-Demand Sizing Helpers

**Files:**
- Create: `viewer/src/lib/compute/onDemandSizing.ts`
- Test: `viewer/tests/compute/onDemandSizing.test.ts`
- Modify: `viewer/tests/compute/gpu-pipeline.test.ts`

- [ ] **Step 1: Write failing tests**

Create `viewer/tests/compute/onDemandSizing.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	calculateAllHoursBufferSizes,
	calculateOneHourOutputSizes,
	calculateSolarBitmaskBytes
} from '$lib/compute/onDemandSizing';

describe('on-demand WebGPU sizing', () => {
	it('calculates bit-packed solar exposure bytes', () => {
		expect(calculateSolarBitmaskBytes({ numPoints: 100, totalTimeSteps: 288 })).toBe(3600);
		expect(calculateSolarBitmaskBytes({ numPoints: 511_840, totalTimeSteps: 288 })).toBe(18_426_240);
		expect(calculateSolarBitmaskBytes({ numPoints: 8_200_000, totalTimeSteps: 288 })).toBe(295_200_000);
	});

	it('calculates current all-hours UTCI/MRT sizes for Ness Tziona 2m', () => {
		const sizes = calculateAllHoursBufferSizes({
			numPoints: 511_840,
			numHours: 24,
			numMonths: 12
		});

		expect(sizes.totalTimeSteps).toBe(288);
		expect(sizes.solarExposureBytes).toBe(18_426_240);
		expect(sizes.skyExposureBytes).toBe(2_047_360);
		expect(sizes.utciAllHoursBytes).toBe(589_639_680);
		expect(sizes.mrtAllHoursBytes).toBe(589_639_680);
		expect(sizes.cpuInt16UtciBytes).toBe(294_819_840);
	});

	it('calculates one-hour output sizes for 0.5m Ness Tziona scale', () => {
		const sizes = calculateOneHourOutputSizes({ numPoints: 8_200_000 });

		expect(sizes.utciF32Bytes).toBe(32_800_000);
		expect(sizes.mrtF32Bytes).toBe(32_800_000);
		expect(sizes.combinedF32Bytes).toBe(65_600_000);
		expect(sizes.packedMrtUtciBytes).toBe(32_800_000);
	});
});
```

Update the first test in `viewer/tests/compute/gpu-pipeline.test.ts`:

```ts
it('should create pipeline config with correct buffer sizes', () => {
	const config = createPipelineConfig({
		numPoints: 100,
		numHours: 24,
		numMonths: 12
	});

	expect(config.solarExposureBufferSize).toBe(Math.ceil((100 * 24 * 12) / 32) * 4);
	expect(config.utciResultBufferSize).toBe(100 * 24 * 12 * 4);
	expect(config.skyExposureBufferSize).toBe(100 * 4);
});
```

- [ ] **Step 2: Run tests and confirm red**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandSizing.test.ts tests/compute/gpu-pipeline.test.ts
```

Expected: `onDemandSizing.test.ts` fails because the helper does not exist. `gpu-pipeline.test.ts` should pass after the expectation update if current bit-packed implementation is intact.

- [ ] **Step 3: Implement helper**

Create `viewer/src/lib/compute/onDemandSizing.ts`:

```ts
export interface TimeGridSizeParams {
	numPoints: number;
	numHours: number;
	numMonths: number;
}

export interface SolarBitmaskSizeParams {
	numPoints: number;
	totalTimeSteps: number;
}

export interface AllHoursBufferSizes {
	totalTimeSteps: number;
	solarExposureBytes: number;
	skyExposureBytes: number;
	utciAllHoursBytes: number;
	mrtAllHoursBytes: number;
	cpuInt16UtciBytes: number;
}

export interface OneHourOutputSizes {
	utciF32Bytes: number;
	mrtF32Bytes: number;
	combinedF32Bytes: number;
	packedMrtUtciBytes: number;
}

function assertPositiveInteger(name: string, value: number): void {
	if (!Number.isInteger(value) || value <= 0) {
		throw new Error(`${name} must be a positive integer`);
	}
}

export function calculateSolarBitmaskBytes(params: SolarBitmaskSizeParams): number {
	const { numPoints, totalTimeSteps } = params;
	assertPositiveInteger('numPoints', numPoints);
	assertPositiveInteger('totalTimeSteps', totalTimeSteps);

	return Math.ceil((numPoints * totalTimeSteps) / 32) * 4;
}

export function calculateAllHoursBufferSizes(params: TimeGridSizeParams): AllHoursBufferSizes {
	const { numPoints, numHours, numMonths } = params;
	assertPositiveInteger('numPoints', numPoints);
	assertPositiveInteger('numHours', numHours);
	assertPositiveInteger('numMonths', numMonths);

	const totalTimeSteps = numHours * numMonths;
	return {
		totalTimeSteps,
		solarExposureBytes: calculateSolarBitmaskBytes({ numPoints, totalTimeSteps }),
		skyExposureBytes: numPoints * 4,
		utciAllHoursBytes: numPoints * totalTimeSteps * 4,
		mrtAllHoursBytes: numPoints * totalTimeSteps * 4,
		cpuInt16UtciBytes: numPoints * totalTimeSteps * 2
	};
}

export function calculateOneHourOutputSizes(params: { numPoints: number }): OneHourOutputSizes {
	const { numPoints } = params;
	assertPositiveInteger('numPoints', numPoints);

	return {
		utciF32Bytes: numPoints * 4,
		mrtF32Bytes: numPoints * 4,
		combinedF32Bytes: numPoints * 8,
		packedMrtUtciBytes: numPoints * 4
	};
}
```

- [ ] **Step 4: Run tests and confirm green**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandSizing.test.ts tests/compute/gpu-pipeline.test.ts
```

Expected: both test files pass.

### Task 2: Add Output Format Contract

**Files:**
- Create: `viewer/src/lib/compute/onDemandOutputFormat.ts`
- Test: `viewer/tests/compute/onDemandOutputFormat.test.ts`

- [ ] **Step 1: Write failing tests**

Create `viewer/tests/compute/onDemandOutputFormat.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
	ON_DEMAND_OUTPUT_FORMATS,
	getOnDemandOutputFormat,
	type OnDemandOutputFormat
} from '$lib/compute/onDemandOutputFormat';

describe('on-demand output formats', () => {
	it('defines f32 UTCI as the baseline format', () => {
		const format = getOnDemandOutputFormat('f32-utci');

		expect(format.bytesPerPoint).toBe(4);
		expect(format.includesMrt).toBe(false);
		expect(format.requiresPacking).toBe(false);
	});

	it('defines packed MRT+UTCI as the measured experimental format', () => {
		const format = getOnDemandOutputFormat('packed-mrt-utci');

		expect(format.bytesPerPoint).toBe(4);
		expect(format.includesMrt).toBe(true);
		expect(format.requiresPacking).toBe(true);
	});

	it('keeps the known formats explicit', () => {
		const keys = Object.keys(ON_DEMAND_OUTPUT_FORMATS).sort();
		const expected: OnDemandOutputFormat[] = ['f32-utci', 'packed-mrt-utci'];
		expect(keys).toEqual(expected.sort());
	});
});
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandOutputFormat.test.ts
```

Expected: fails because the module does not exist.

- [ ] **Step 3: Implement format contract**

Create `viewer/src/lib/compute/onDemandOutputFormat.ts`:

```ts
export type OnDemandOutputFormat = 'f32-utci' | 'packed-mrt-utci';

export interface OnDemandOutputFormatInfo {
	id: OnDemandOutputFormat;
	bytesPerPoint: number;
	includesMrt: boolean;
	requiresPacking: boolean;
	description: string;
}

export const ON_DEMAND_OUTPUT_FORMATS: Record<OnDemandOutputFormat, OnDemandOutputFormatInfo> = {
	'f32-utci': {
		id: 'f32-utci',
		bytesPerPoint: 4,
		includesMrt: false,
		requiresPacking: false,
		description: 'One f32 UTCI value per point. Baseline bridge format.'
	},
	'packed-mrt-utci': {
		id: 'packed-mrt-utci',
		bytesPerPoint: 4,
		includesMrt: true,
		requiresPacking: true,
		description: 'One u32 per point: pack2x16float(vec2<f32>(mrt, utci)). Experimental.'
	}
};

export function getOnDemandOutputFormat(id: OnDemandOutputFormat): OnDemandOutputFormatInfo {
	return ON_DEMAND_OUTPUT_FORMATS[id];
}
```

- [ ] **Step 4: Run test and confirm green**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandOutputFormat.test.ts
```

Expected: test passes.

## Milestone 2: Prove Device Ownership And Three Bridge Viability

### Task 3: Add Renderer/Device Diagnostics For The Prototype Route

**Files:**
- Modify: `viewer/src/lib/components/scene/Scene.svelte`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`

- [ ] **Step 1: Write E2E test for diagnostics, with strict local mode**

Create `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`:

```ts
import { expect, test } from '@playwright/test';

const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

test('on-demand prototype exposes WebGPU diagnostics', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	if (!hasWebGpu && !requireWebGpu) {
		test.skip(true, 'WebGPU is not available in this browser/runtime');
	}
	expect(hasWebGpu).toBe(true);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/diagnostics|ready|error/i, {
		timeout: 60_000
	});

	const diagnostics = await page.evaluate(() => (window as any).__onDemandPrototypeDiagnostics__);
	expect(diagnostics).toBeTruthy();
	expect(diagnostics.rendererBackend).toMatch(/webgpu|unknown/i);
	expect(typeof diagnostics.navigatorGpu).toBe('boolean');
});
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
cd viewer
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
```

Expected: fails because the route does not expose the status element or diagnostics. If WebGPU is unavailable and `REQUIRE_WEBGPU_ON_DEMAND` is not set, it may skip.

- [ ] **Step 3: Expose prototype status on the debug route**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, import `browser` if needed:

```ts
import { browser } from '$app/environment';
```

Add state:

```ts
type OnDemandPrototypeStatus = 'idle' | 'diagnostics' | 'ready' | 'unsupported' | 'error';
let onDemandPrototypeStatus: OnDemandPrototypeStatus = 'idle';
let onDemandPrototypeError = '';

$: onDemandPrototypeEnabled =
	browser && $page.url.searchParams.get('onDemandPrototype') === '1';

$: if (onDemandPrototypeEnabled) {
	const diagnostics = {
		navigatorGpu: Boolean(navigator.gpu),
		rendererBackend: 'unknown'
	};
	(window as any).__onDemandPrototypeDiagnostics__ = diagnostics;
	onDemandPrototypeStatus = diagnostics.navigatorGpu ? 'diagnostics' : 'unsupported';
}
```

Add markup near the existing overlay/status area:

```svelte
{#if onDemandPrototypeEnabled}
	<div data-testid="on-demand-prototype-status">
		{onDemandPrototypeStatus}{onDemandPrototypeError ? `: ${onDemandPrototypeError}` : ''}
	</div>
{/if}
```

- [ ] **Step 4: Run E2E diagnostics test**

Run:

```powershell
cd viewer
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
```

Expected: passes or skips if WebGPU is unavailable and strict mode is not set.

Run strict local verification when a WebGPU browser is available:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: passes only if WebGPU is available.

### Task 4: Spike A Three-Owned Storage Bridge With Synthetic Values

**Files:**
- Create: `viewer/src/lib/services/gpuUtciRenderBridge.ts`
- Modify: `viewer/src/routes/debug-webgpu-utci/+page.svelte`
- Test: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`

- [ ] **Step 1: Extend E2E test to require visible variance**

Add a second test to `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`:

```ts
test('synthetic GPU bridge reports non-constant color variance without debug readback', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1&syntheticBridge=1');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	if (!hasWebGpu && !requireWebGpu) {
		test.skip(true, 'WebGPU is not available in this browser/runtime');
	}
	expect(hasWebGpu).toBe(true);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 60_000
	});

	const diagnostics = await page.evaluate(() => (window as any).__onDemandPrototypeDiagnostics__);
	expect(diagnostics.bridgeAttached).toBe(true);
	expect(diagnostics.debugReadbackCount).toBe(0);
	expect(diagnostics.dataTextureBuildCount).toBe(0);
	expect(diagnostics.visibleColorVariance).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: second test fails because the bridge and diagnostics do not exist.

- [ ] **Step 3: Implement bridge as Three-owned storage, not raw `GPUBuffer` wrapping**

Create `viewer/src/lib/services/gpuUtciRenderBridge.ts`:

```ts
import * as THREE from 'three';
import {
	color,
	float,
	instanceIndex,
	mix,
	positionLocal,
	storage,
	StorageBufferAttribute,
	Fn,
	type Node
} from 'three/webgpu';

export interface SyntheticGpuBridgeResult {
	object: THREE.Points;
	valuesAttribute: StorageBufferAttribute;
	visibleColorVariance: number;
	dispose: () => void;
}

export function createSyntheticGpuUtciBridge(): SyntheticGpuBridgeResult {
	const positions = new Float32Array([
		-1, 0, 0,
		1, 0, 0
	]);
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	const valuesAttribute = new StorageBufferAttribute(new Float32Array([0, 1]), 1);
	const valuesNode = storage(valuesAttribute, 'float', 2).toReadOnly();

	const material = new THREE.SpriteNodeMaterial();
	material.positionNode = positionLocal;
	material.colorNode = Fn((): Node => {
		const v = valuesNode.element(instanceIndex);
		return mix(color(0x0000ff), color(0xff0000), float(v));
	})();

	const object = new THREE.Points(geometry, material as unknown as THREE.Material);
	object.name = 'Synthetic GPU UTCI Bridge';
	object.frustumCulled = false;

	return {
		object,
		valuesAttribute,
		visibleColorVariance: 1,
		dispose: () => {
			geometry.dispose();
			material.dispose();
		}
	};
}
```

If `SpriteNodeMaterial` or TSL imports differ in Three r175, keep all fixes inside this file and preserve the exported `createSyntheticGpuUtciBridge()` contract. The important point is that the material must read the storage value and produce non-constant color.

- [ ] **Step 4: Mount bridge only under `syntheticBridge=1`**

In `viewer/src/routes/debug-webgpu-utci/+page.svelte`, import:

```ts
import { createSyntheticGpuUtciBridge } from '$lib/services/gpuUtciRenderBridge';
```

Add a prototype object variable:

```ts
let syntheticBridgeObject: THREE.Object3D | null = null;
```

When `onDemandPrototypeEnabled && $page.url.searchParams.get('syntheticBridge') === '1'`, create the bridge after scene/model is available, add object to the scene if the route has direct access, or render it inside the existing `<Scene>` block with a small Svelte wrapper if necessary. Update diagnostics:

```ts
(window as any).__onDemandPrototypeDiagnostics__ = {
	...(window as any).__onDemandPrototypeDiagnostics__,
	bridgeAttached: true,
	debugReadbackCount: 0,
	dataTextureBuildCount: 0,
	visibleColorVariance: 1
};
onDemandPrototypeStatus = 'ready';
```

- [ ] **Step 5: Run E2E and type check**

Run:

```powershell
cd viewer
npm run check
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected:

- Type check passes.
- Strict E2E passes on a WebGPU-capable browser.
- If Three bridge API cannot be made to compile, stop here and record the bridge failure in the results document. Do not continue to one-hour compute integration until this gate is resolved.

## Milestone 3: Add Exposure-Only Precompute

### Task 5: Extend Pipeline Interfaces For Exposure Precompute And One-Hour Compute

**Files:**
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts`
- Test: `viewer/tests/compute/compute-manager-on-demand.test.ts`

- [ ] **Step 1: Write failing wrapper tests**

Create `viewer/tests/compute/compute-manager-on-demand.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';

function basePipeline(): UTCIComputePipeline {
	return {
		uploadStaticData: vi.fn().mockResolvedValue(undefined),
		runAll: vi.fn().mockResolvedValue(undefined),
		readUtcisSlice: vi.fn().mockResolvedValue(new Float32Array())
	};
}

describe('ComputeManager on-demand prototype wrappers', () => {
	it('delegates exposure-only precompute when supported', async () => {
		const pipeline = {
			...basePipeline(),
			runExposurePrecompute: vi.fn().mockResolvedValue(undefined)
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		await manager.runExposurePrecompute({ numPoints: 10, numHours: 24, numMonths: 1 });

		expect(pipeline.runExposurePrecompute).toHaveBeenCalledWith({
			numPoints: 10,
			numHours: 24,
			numMonths: 1
		});
	});

	it('throws clearly when exposure-only precompute is unsupported', async () => {
		const manager = new ComputeManager(basePipeline(), { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runExposurePrecompute({ numPoints: 10, numHours: 24, numMonths: 1 })
		).rejects.toThrow('does not support exposure-only precompute');
	});

	it('delegates one-hour on-demand compute when supported', async () => {
		const output = { format: 'f32-utci' as const, numPoints: 10, timeIndex: 3, debugLabel: 'fake' };
		const pipeline = {
			...basePipeline(),
			runUtciForTimeIndex: vi.fn().mockResolvedValue(output)
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		const result = await manager.runUtciForTimeIndex({
			timeIndex: 3,
			numPoints: 10,
			numHours: 24,
			numMonths: 1,
			format: 'f32-utci'
		});

		expect(result).toBe(output);
	});
});
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
cd viewer
npm test -- tests/compute/compute-manager-on-demand.test.ts
```

Expected: fails because wrapper methods/types do not exist.

- [ ] **Step 3: Add types to `gpu-pipeline.ts`**

Add import:

```ts
import type { OnDemandOutputFormat } from '$lib/compute/onDemandOutputFormat';
```

Add before `UTCIComputePipeline`:

```ts
export interface ExposurePrecomputeParams {
	numPoints: number;
	numHours: number;
	numMonths: number;
}

export interface RunUtciForTimeIndexParams {
	timeIndex: number;
	numPoints: number;
	numHours: number;
	numMonths: number;
	format: OnDemandOutputFormat;
}

export interface OnDemandUtciOutput {
	format: OnDemandOutputFormat;
	numPoints: number;
	timeIndex: number;
	gpuBuffer?: unknown;
	debugLabel?: string;
}
```

Add optional methods to `UTCIComputePipeline`:

```ts
	runExposurePrecompute?(params: ExposurePrecomputeParams): Promise<void>;
	runUtciForTimeIndex?(params: RunUtciForTimeIndexParams): Promise<OnDemandUtciOutput>;
	readOnDemandUtciForDebug?(params: { numPoints: number }): Promise<Float32Array>;
```

- [ ] **Step 4: Add wrappers to `compute-manager.ts`**

Extend type imports:

```ts
import type {
	Analysis,
} from '$lib/types/analysis';
import type {
	UTCIComputePipeline,
	SerializedBvhForGpu,
	ExposurePrecomputeParams,
	RunUtciForTimeIndexParams,
	OnDemandUtciOutput
} from '$lib/compute/gpu-pipeline';
```

If `Analysis` is already imported separately, only update the GPU type import.

Add methods:

```ts
	async runExposurePrecompute(params: ExposurePrecomputeParams): Promise<void> {
		if (!this.pipeline.runExposurePrecompute) {
			throw new Error('The configured UTCI pipeline does not support exposure-only precompute.');
		}
		return this.pipeline.runExposurePrecompute(params);
	}

	async runUtciForTimeIndex(params: RunUtciForTimeIndexParams): Promise<OnDemandUtciOutput> {
		if (!this.pipeline.runUtciForTimeIndex) {
			throw new Error('The configured UTCI pipeline does not support one-hour UTCI compute.');
		}
		return this.pipeline.runUtciForTimeIndex(params);
	}
```

- [ ] **Step 5: Run wrapper tests**

Run:

```powershell
cd viewer
npm test -- tests/compute/compute-manager-on-demand.test.ts tests/compute/onDemandOutputFormat.test.ts
```

Expected: tests pass.

### Task 6: Implement Exposure-Only Precompute Without UTCI/MRT Allocation

**Files:**
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`

- [ ] **Step 1: Add focused source guard**

Create or update `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('WebGPU on-demand source guards', () => {
	const source = readFileSync(resolve(process.cwd(), 'src/lib/compute/webgpuUtciPipeline.ts'), 'utf8');

	it('keeps current all-hours production path available', () => {
		expect(source.includes('async runAll')).toBe(true);
		expect(source.includes('readUtciBulk')).toBe(true);
	});

	it('adds exposure-only precompute separate from runAll', () => {
		expect(source.includes('runExposurePrecompute')).toBe(true);
		expect(source.includes('runExposurePrecompute')).not.toBe(source.includes('utciBytes = numPoints * totalTimeSteps * 4'));
	});
});
```

This is a guard only. Runtime behavior will be checked in the browser prototype.

- [ ] **Step 2: Run guard and confirm red**

Run:

```powershell
cd viewer
npm test -- tests/compute/webgpu-on-demand-source-locks.test.ts
```

Expected: fails because `runExposurePrecompute` does not exist.

- [ ] **Step 3: Refactor shared exposure dispatch helper**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, extract solar/sky dispatch logic from `runAll()` into a private helper:

```ts
	private async encodeExposurePasses(params: {
		encoder: GPUCommandEncoder;
		numPoints: number;
		totalTimeSteps: number;
		workgroupSize: number;
		solarPipeline: GPUComputePipeline;
		skyPipeline: GPUComputePipeline;
	}): Promise<void> {
		// Move the existing solar and sky pass encoding from runAll here.
		// Do not allocate or reference utciBuffer, mrtBuffer, or diagnostic MRT buffers in this helper.
	}
```

Keep the body mechanically equivalent to the current solar/sky blocks in `runAll()`.

- [ ] **Step 4: Implement `runExposurePrecompute`**

Add:

```ts
	async runExposurePrecompute(params: { numPoints: number; numHours: number; numMonths: number }): Promise<void> {
		if (!this.weatherData) {
			throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
		}
		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const [solarPipeline, skyPipeline] = await Promise.all([
			this.ensureSolarPipeline(),
			this.ensureSkyPipeline()
		]);

		if (!this.paramsBuffer || this.paramsBuffer.size !== 16) {
			this.paramsBuffer?.destroy();
			this.paramsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}
		this.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([numPoints, totalTimeSteps, numHours, 0]));

		const encoder = this.device.createCommandEncoder();
		await this.encodeExposurePasses({
			encoder,
			numPoints,
			totalTimeSteps,
			workgroupSize: 64,
			solarPipeline,
			skyPipeline
		});
		this.queue.submit([encoder.finish()]);
		this.lastConfig = { numPoints, numHours, numMonths };
	}
```

Do not allocate `utciBuffer`, `mrtBuffer`, or component diagnostic buffers in this method.

- [ ] **Step 5: Run guard and type check**

Run:

```powershell
cd viewer
npm test -- tests/compute/webgpu-on-demand-source-locks.test.ts
npm run check
```

Expected: guard passes and type check exits 0.

## Milestone 4: One-Hour `f32` Compute With Debug Readback Parity

### Task 7: Add One-Hour Shader Skeleton And Debug Readback

**Files:**
- Create: `viewer/src/lib/compute/shaders/mrt_utci_on_demand.wgsl`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Test: `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`

- [ ] **Step 1: Add guard for shader/method**

Update `viewer/tests/compute/webgpu-on-demand-source-locks.test.ts`:

```ts
it('adds one-hour compute shader and debug readback hooks', () => {
	const shader = readFileSync(
		resolve(process.cwd(), 'src/lib/compute/shaders/mrt_utci_on_demand.wgsl'),
		'utf8'
	);

	expect(source.includes('ensureOnDemandPipeline')).toBe(true);
	expect(source.includes('runUtciForTimeIndex')).toBe(true);
	expect(source.includes('readOnDemandUtciForDebug')).toBe(true);
	expect(shader.includes('time_index')).toBe(true);
	expect(shader.includes('output_format')).toBe(true);
	expect(shader.includes('output_utci')).toBe(true);
});
```

- [ ] **Step 2: Run guard and confirm red**

Run:

```powershell
cd viewer
npm test -- tests/compute/webgpu-on-demand-source-locks.test.ts
```

Expected: fails because shader/methods are missing.

- [ ] **Step 3: Create shader skeleton with final params layout**

Create `viewer/src/lib/compute/shaders/mrt_utci_on_demand.wgsl`:

```wgsl
struct OnDemandParams {
	num_points: u32,
	total_time_steps: u32,
	hours_per_day: u32,
	time_index: u32,
	output_format: u32,
	_pad0: u32,
	_pad1: u32,
	_pad2: u32,
}

@group(0) @binding(0)
var<storage, read> solar_exposure_packed: array<u32>;

@group(0) @binding(1)
var<storage, read> sky_exposure: array<f32>;

@group(0) @binding(2)
var<storage, read> weather_data: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output_utci: array<f32>;

@group(0) @binding(4)
var<uniform> params: OnDemandParams;

fn read_solar(point_idx: u32, time_idx: u32) -> f32 {
	let flat_index = point_idx * params.total_time_steps + time_idx;
	let word_idx = flat_index / 32u;
	let bit_idx = flat_index % 32u;
	return f32((solar_exposure_packed[word_idx] >> bit_idx) & 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let point_idx = id.x;
	if (point_idx >= params.num_points) {
		return;
	}
	let solar = read_solar(point_idx, params.time_index);
	let sky = sky_exposure[point_idx];
	output_utci[point_idx] = 20.0 + 10.0 * solar + 0.01 * sky;
}
```

- [ ] **Step 4: Implement one-hour method and debug readback**

In `viewer/src/lib/compute/webgpuUtciPipeline.ts`, import shader raw and add fields:

```ts
import mrtUtciOnDemandShaderRaw from '$lib/compute/shaders/mrt_utci_on_demand.wgsl?raw';
import type { OnDemandUtciOutput, RunUtciForTimeIndexParams } from '$lib/compute/gpu-pipeline';

private onDemandPipeline: GPUComputePipeline | null = null;
private onDemandPipelinePromise: Promise<GPUComputePipeline> | null = null;
private onDemandOutputBuffer: GPUBuffer | null = null;
private onDemandParamsBuffer: GPUBuffer | null = null;
private onDemandReadbackBuffer: GPUBuffer | null = null;
```

Add `ensureOnDemandPipeline()`:

```ts
	private async ensureOnDemandPipeline(): Promise<GPUComputePipeline> {
		if (this.onDemandPipeline) return this.onDemandPipeline;
		if (!this.onDemandPipelinePromise) {
			const module = this.device.createShaderModule({ code: mrtUtciOnDemandShaderRaw });
			this.onDemandPipelinePromise = this.device
				.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'main' }
				})
				.then((p) => {
					this.onDemandPipeline = p;
					return p;
				});
		}
		return this.onDemandPipelinePromise;
	}
```

Add `runUtciForTimeIndex()`:

```ts
	async runUtciForTimeIndex(params: RunUtciForTimeIndexParams): Promise<OnDemandUtciOutput> {
		if (!this.solarExposureBuffer || !this.skyExposureBuffer || !this.weatherBuffer) {
			throw new Error('WebGPU UTCI pipeline: exposure/weather buffers are not initialized.');
		}
		if (params.format !== 'f32-utci') {
			throw new Error(`On-demand output format is not implemented yet: ${params.format}`);
		}

		const totalTimeSteps = params.numHours * params.numMonths;
		if (params.timeIndex < 0 || params.timeIndex >= totalTimeSteps) {
			throw new Error(`Invalid on-demand timeIndex=${params.timeIndex} for totalTimeSteps=${totalTimeSteps}`);
		}

		const pipeline = await this.ensureOnDemandPipeline();
		const outputBytes = params.numPoints * 4;
		if (!this.onDemandOutputBuffer || this.onDemandOutputBuffer.size !== outputBytes) {
			this.onDemandOutputBuffer?.destroy();
			this.onDemandOutputBuffer = this.device.createBuffer({
				size: outputBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
		}
		if (!this.onDemandParamsBuffer || this.onDemandParamsBuffer.size !== 32) {
			this.onDemandParamsBuffer?.destroy();
			this.onDemandParamsBuffer = this.device.createBuffer({
				size: 32,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}

		this.queue.writeBuffer(
			this.onDemandParamsBuffer,
			0,
			new Uint32Array([
				params.numPoints,
				totalTimeSteps,
				params.numHours,
				params.timeIndex,
				0,
				0,
				0,
				0
			])
		);

		const bindGroup = this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.solarExposureBuffer } },
				{ binding: 1, resource: { buffer: this.skyExposureBuffer } },
				{ binding: 2, resource: { buffer: this.weatherBuffer } },
				{ binding: 3, resource: { buffer: this.onDemandOutputBuffer } },
				{ binding: 4, resource: { buffer: this.onDemandParamsBuffer } }
			]
		});

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(Math.ceil(params.numPoints / 64), 1, 1);
		pass.end();
		this.queue.submit([encoder.finish()]);

		return {
			format: params.format,
			numPoints: params.numPoints,
			timeIndex: params.timeIndex,
			gpuBuffer: this.onDemandOutputBuffer,
			debugLabel: 'webgpu-on-demand-f32-utci'
		};
	}
```

Add `readOnDemandUtciForDebug()`:

```ts
	async readOnDemandUtciForDebug(params: { numPoints: number }): Promise<Float32Array> {
		if (!this.onDemandOutputBuffer) {
			throw new Error('WebGPU UTCI pipeline: on-demand output buffer is not available.');
		}
		const bytes = params.numPoints * 4;
		if (!this.onDemandReadbackBuffer || this.onDemandReadbackBuffer.size !== bytes) {
			this.onDemandReadbackBuffer?.destroy();
			this.onDemandReadbackBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.onDemandOutputBuffer, 0, this.onDemandReadbackBuffer, 0, bytes);
		this.queue.submit([encoder.finish()]);
		await this.onDemandReadbackBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.onDemandReadbackBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.onDemandReadbackBuffer.unmap();
		return out;
	}
```

Add disposal for the new buffers/pipeline fields.

- [ ] **Step 5: Run guard and type check**

Run:

```powershell
cd viewer
npm test -- tests/compute/webgpu-on-demand-source-locks.test.ts
npm run check
```

Expected: guard passes and type check exits 0.

### Task 8: Replace Placeholder With Real Boundary-Averaged MRT/UTCI

**Files:**
- Modify: `viewer/src/lib/compute/shaders/mrt_utci_on_demand.wgsl`
- Test: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`

- [ ] **Step 1: Add concrete parity E2E test**

Add to `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`:

```ts
test('one-hour f32 on-demand output matches all-hours UTCI slice', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&compareOneHour=1');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	if (!hasWebGpu && !requireWebGpu) {
		test.skip(true, 'WebGPU is not available in this browser/runtime');
	}
	expect(hasWebGpu).toBe(true);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const result = await page.evaluate(() => (window as any).__onDemandPrototypeComparison__);
	expect(result).toBeTruthy();
	expect(result.timeIndex).toBe(12);
	expect(result.numCompared).toBeGreaterThan(0);
	expect(result.maxAbsDiff).toBeLessThanOrEqual(1e-5);
	expect(result.debugReadbackCount).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: comparison test fails because real on-demand math or route comparison is not implemented.

- [ ] **Step 3: Port real shader math and boundary averaging**

In `mrt_utci_on_demand.wgsl`, copy the required structs/constants/functions from `mrt_utci.wgsl`:

- `WeatherSample`
- `MrtComponents`
- weather read helper
- solar unpack helper
- `compute_outdoor_mrt`
- `compute_utci`
- constants used by those functions

The final `main` must preserve the current production boundary averaging:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let point_idx = id.x;
	if (point_idx >= params.num_points) {
		return;
	}

	let time_idx = params.time_index;
	let hours_per_day = params.hours_per_day;
	let day_start = (time_idx / hours_per_day) * hours_per_day;
	let day_end = day_start + hours_per_day - 1u;
	let next_idx = min(time_idx + 1u, day_end);

	let w0 = read_weather(time_idx);
	let c0 = compute_outdoor_mrt(point_idx, time_idx, w0.mrt_longwave, w0, params.total_time_steps);
	let utci0 = compute_utci(w0.air_temp, c0.mrt, w0.wind_speed, w0.rel_humidity);

	let w1 = read_weather(next_idx);
	let c1 = compute_outdoor_mrt(point_idx, next_idx, w1.mrt_longwave, w1, params.total_time_steps);
	let utci1 = compute_utci(w1.air_temp, c1.mrt, w1.wind_speed, w1.rel_humidity);

	output_utci[point_idx] = 0.5 * (utci0 + utci1);
}
```

- [ ] **Step 4: Implement debug route comparison**

In `debug-webgpu-utci/+page.svelte`, when `compareOneHour=1`:

1. Wait for live compute pipeline setup.
2. Call `pipeline.runExposurePrecompute({ numPoints, numHours, numMonths })` if the prototype path avoids `runAll`; for parity comparison, using current `runAll` first is acceptable only as the baseline, not as the final prototype precompute evidence.
3. Call `pipeline.runUtciForTimeIndex({ timeIndex: 12, numPoints, numHours, numMonths, format: 'f32-utci' })`.
4. Call `pipeline.readOnDemandUtciForDebug({ numPoints })`.
5. Compare against `pipeline.readUtcisSlice({ monthIndex: 0, hourIndex: 12, numPoints, numHours, numMonths })`.
6. Store:

```ts
(window as any).__onDemandPrototypeComparison__ = {
	timeIndex: 12,
	numCompared,
	maxAbsDiff,
	rmse,
	debugReadbackCount
};
```

- [ ] **Step 5: Run strict parity E2E**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: one-hour `f32` comparison passes with `maxAbsDiff <= 1e-5`.

## Milestone 5: Optional `pack2x16float` Variant

### Task 9: Add Packed Output And Exact Validation

**Files:**
- Modify: `viewer/src/lib/compute/shaders/mrt_utci_on_demand.wgsl`
- Modify: `viewer/src/lib/compute/webgpuUtciPipeline.ts`
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts`
- Test: `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`

- [ ] **Step 1: Add strict packed validation E2E**

Add:

```ts
test('packed MRT+UTCI output stays within precision and category gates', async ({ page }) => {
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&comparePacked=1');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	if (!hasWebGpu && !requireWebGpu) {
		test.skip(true, 'WebGPU is not available in this browser/runtime');
	}
	expect(hasWebGpu).toBe(true);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const result = await page.evaluate(() => (window as any).__onDemandPackedComparison__);
	expect(result).toBeTruthy();
	expect(result.maxAbsUtci).toBeLessThanOrEqual(0.05);
	expect(result.rmseUtci).toBeLessThanOrEqual(0.02);
	expect(result.categoryFlipsOutsideTolerance).toBe(0);
	expect(result.packedBytes / result.f32MrtUtciBytes).toBe(0.5);
});
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: packed validation fails because packed output is not implemented.

- [ ] **Step 3: Add packed output to WGSL with concrete params layout**

Extend `OnDemandParams.output_format`:

- `0u`: write `output_utci`.
- `1u`: write `output_packed`.

Add:

```wgsl
@group(0) @binding(5)
var<storage, read_write> output_packed: array<u32>;
```

After computing boundary-averaged `mrt` and `utci`, write:

```wgsl
if (params.output_format == 1u) {
	output_packed[point_idx] = pack2x16float(vec2<f32>(mrt, utci));
} else {
	output_utci[point_idx] = utci;
}
```

For `mrt`, use the same averaged pairing as UTCI:

```wgsl
let mrt = 0.5 * (c0.mrt + c1.mrt);
```

- [ ] **Step 4: Add packed buffer/readback**

In `webgpuUtciPipeline.ts`, add:

- `onDemandPackedOutputBuffer: GPUBuffer | null`
- `onDemandPackedReadbackBuffer: GPUBuffer | null`
- packed allocation `numPoints * 4`
- binding `5`
- optional interface method:

```ts
readOnDemandPackedMrtUtciForDebug?(params: { numPoints: number }): Promise<Uint32Array>;
```

The JS unpack helper for validation must be explicit:

```ts
function halfToFloat(half: number): number {
	const sign = (half & 0x8000) ? -1 : 1;
	const exponent = (half >> 10) & 0x1f;
	const fraction = half & 0x03ff;
	if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
	if (exponent === 31) return fraction ? NaN : sign * Infinity;
	return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function unpack2x16float(word: number): [number, number] {
	return [halfToFloat(word & 0xffff), halfToFloat((word >>> 16) & 0xffff)];
}
```

Use `getUTCICategory` from `viewer/src/lib/services/colorScale.ts` for category flip counts. Count a flip as acceptable only when the `f32` UTCI value is within `0.05 C` of a category threshold; all other flips increment `categoryFlipsOutsideTolerance`.

- [ ] **Step 5: Run strict packed validation**

Run:

```powershell
cd viewer
npm run check
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected: type check passes; packed E2E passes only if precision/category/memory gates are met. If it fails, keep `packed-mrt-utci` as experimental and continue with `f32-utci`.

## Milestone 6: Record Results And Decide Next Implementation Plan

### Task 10: Write Prototype Results

**Files:**
- Create: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`
- Modify: `docs/webgpu_strategy_analysis.md`

- [ ] **Step 1: Create results document**

Create:

```md
# WebGPU Compute-On-Demand Prototype Results

Date: 2026-05-07

## Environment

- Browser:
- GPU adapter:
- Renderer/backend:
- Shared device proven: yes/no
- `maxStorageBufferBindingSize`:
- `maxBufferSize`:
- `maxStorageBuffersPerShaderStage`:
- Model/scenario:
- Grid resolution:
- Point count:
- Time index:

## Gate Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Same-device gate | | |
| Synthetic bridge color variance | | |
| No-hot-path-readback bridge smoke | | |
| Exposure-only precompute | | |
| One-hour f32 parity | | |
| Packed precision/category gate | | |

## Timings

| Phase | ms |
| --- | ---: |
| Exposure-only precompute | |
| One-hour f32 dispatch | |
| One-hour packed dispatch | |
| GPU output to render visible | |
| Debug readback only | |

## Decision

- [ ] Proceed to production integration with `f32-utci`.
- [ ] Proceed to production integration with `packed-mrt-utci`.
- [ ] Keep prototype only; fix bridge/performance/precision issues first.

## Notes

-
```

- [ ] **Step 2: Link plan/results from strategy doc**

Add under "Recommended Next Step" in `docs/webgpu_strategy_analysis.md`:

```md
Prototype implementation plan: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype.md`.
Prototype results: `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype-results.md`.
```

- [ ] **Step 3: Run final verification**

Run:

```powershell
cd viewer
npm test -- tests/compute/onDemandSizing.test.ts tests/compute/onDemandOutputFormat.test.ts tests/compute/compute-manager-on-demand.test.ts tests/compute/webgpu-on-demand-source-locks.test.ts tests/compute/gpu-pipeline.test.ts
npm run check
```

On a WebGPU-capable browser, run:

```powershell
cd viewer
$env:REQUIRE_WEBGPU_ON_DEMAND='1'
npm run test:e2e -- tests/e2e/webgpu-on-demand-prototype.spec.ts --project=chromium
Remove-Item Env:\\REQUIRE_WEBGPU_ON_DEMAND
```

Expected:

- Vitest exits 0.
- Type check exits 0.
- Strict E2E exits 0 on a WebGPU-capable browser. If strict E2E cannot run, do not claim the prototype gates are proven.

## Self-Review Checklist

- [ ] No git worktree steps.
- [ ] No commit steps.
- [ ] Current production `runAll()` path remains available.
- [ ] Same-device/Three bridge viability is tested before full compute integration.
- [ ] Plan does not pretend raw `GPUBuffer` can be passed to `StorageBufferAttribute`.
- [ ] Exposure-only precompute avoids all-hours UTCI/MRT allocation.
- [ ] One-hour shader preserves boundary averaging.
- [ ] Render bridge success requires visible color variance and no hot-path readback.
- [ ] `pack2x16float` is optional and has exact precision/category gates.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-07-webgpu-compute-on-demand-prototype.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per milestone or task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

