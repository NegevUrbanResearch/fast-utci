import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import {
	DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS,
	buildInnovationDistrict05mUtciSurfaceBudgetDecision
} from '$lib/services/activeMaskUtciSurfaceBudget';
import {
	INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE,
	createActiveMaskUtciInstancedSurfaceGeometry,
	disposeActiveMaskUtciInstancedSurfaceGeometry,
	estimateActiveMaskUtciInstancedGeometryBytes,
	getActiveMaskUtciCanonicalCellCenter
} from '$lib/services/activeMaskUtciSurfaceGeometry';
import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

function createActiveLayout(params: {
	width: number;
	height: number;
	gridSize?: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	activeCanonicalIndices: number[];
}): ActiveCellsUtciGridLayout {
	const gridSize = params.gridSize ?? 2;
	return {
		renderTopology: 'active-cells',
		width: params.width,
		height: params.height,
		gridSize,
		coordinateSystem: params.coordinateSystem,
		numPositions: params.activeCanonicalIndices.length,
		minX: 10,
		minZ: 20,
		minY: 3,
		maxY: 3,
		centerX: 10 + ((params.width - 1) * gridSize) / 2,
		centerZ: 20 + ((params.height - 1) * gridSize) / 2,
		baseY: 2.95,
		renderCellCount: params.activeCanonicalIndices.length,
		canonicalCellCount: params.width * params.height,
		activeCanonicalIndices: new Uint32Array(params.activeCanonicalIndices),
		activeMaskSignature: 'active-mask-test'
	};
}

function expectAnalyticExtents(
	geometry: THREE.BufferGeometry,
	params: { width: number; height: number; gridSize: number }
): void {
	const halfWidth = (params.width * params.gridSize) / 2;
	const halfHeight = (params.height * params.gridSize) / 2;
	expect(geometry.boundingBox?.min.toArray()).toEqual([-halfWidth, 0, -halfHeight]);
	expect(geometry.boundingBox?.max.toArray()).toEqual([halfWidth, 0, halfHeight]);
	expect(geometry.boundingSphere?.center.toArray()).toEqual([0, 0, 0]);
	expect(geometry.boundingSphere?.radius).toBe(Math.hypot(halfWidth, halfHeight));
}

describe('active mask UTCI surface geometry budget', () => {
	it('estimates the real Innovation District 0.5m shape and selects instancing', () => {
		const decision = buildInnovationDistrict05mUtciSurfaceBudgetDecision();

		expect(decision.input).toEqual(INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE);
		expect(decision.limits.jsLargestTypedArrayBytes).toBe(
			DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS.jsLargestTypedArrayBytes
		);
		expect(decision.selectedStrategy).toBe('active-instanced-quads');
		expect(decision.planRevisionRequired).toBe(false);
		expect(decision.planRevisionText).toBeUndefined();
		expect(decision.limits.source.requested).toEqual({
			maxStorageBufferBindingSize: 512 * 1024 * 1024,
			maxBufferSize: 1024 * 1024 * 1024
		});
		expect(decision.limits.source.device).toBeUndefined();

		const dense = decision.estimates.find(
			(estimate) => estimate.strategy === 'dense-indexed-rect'
		);
		const activeIndexed = decision.estimates.find(
			(estimate) => estimate.strategy === 'active-indexed-quads'
		);
		const activeInstanced = decision.estimates.find(
			(estimate) => estimate.strategy === 'active-instanced-quads'
		);
		const tiledIndexed = decision.estimates.find(
			(estimate) => estimate.strategy === 'active-tiled-indexed-quads'
		);

		expect(dense).toMatchObject({
			totalJsTypedArrayBytes: 1_309_427_460,
			largestSingleJsTypedArrayBytes: 572_904_600,
			vertexBufferBytes: 286_569_792,
			indexBufferBytes: 572_904_600,
			storageBufferBytes: 136_359_292
		});
		expect(activeIndexed).toMatchObject({
			totalJsTypedArrayBytes: 981_004_608,
			largestSingleJsTypedArrayBytes: 490_502_304,
			vertexBufferBytes: 490_502_304,
			indexBufferBytes: 245_251_152,
			storageBufferBytes: 204_375_960
		});
		expect(activeIndexed?.fits.jsLargestTypedArray).toBe(false);
		expect(activeInstanced).toMatchObject({
			totalJsTypedArrayBytes: 81_750_456,
			largestSingleJsTypedArrayBytes: 40_875_192,
			vertexBufferBytes: 48,
			indexBufferBytes: 24,
			storageBufferBytes: 81_750_384
		});
		expect(activeInstanced?.fits).toMatchObject({
			jsLargestTypedArray: true,
			maxBufferSize: true,
			maxStorageBufferBindingSize: true
		});
		expect(tiledIndexed).toMatchObject({
			tileCount: 2,
			maxActiveCellsPerTile: 5_592_405,
			largestSingleJsTypedArrayBytes: 268_435_440
		});
		expect(decision.selectionReason).toContain('active instanced');
	});

	it('sizes tiled instancing against actual storage binding limits', () => {
		const decision = buildInnovationDistrict05mUtciSurfaceBudgetDecision({
			deviceLimits: {
				maxBufferSize: 1024 * 1024 * 1024,
				maxStorageBufferBindingSize: 16 * 1024 * 1024
			}
		});
		const tiledInstanced = decision.estimates.find(
			(estimate) => estimate.strategy === 'active-tiled-instanced-quads'
		);

		expect(decision.selectedStrategy).toBe('active-tiled-instanced-quads');
		expect(decision.planRevisionRequired).toBe(true);
		expect(decision.planRevisionText).toContain('tiling');
		expect(decision.limits.source.requested).toEqual({
			maxStorageBufferBindingSize: 512 * 1024 * 1024,
			maxBufferSize: 1024 * 1024 * 1024
		});
		expect(decision.limits.source.device).toEqual({
			maxBufferSize: 1024 * 1024 * 1024,
			maxStorageBufferBindingSize: 16 * 1024 * 1024
		});
		expect(tiledInstanced?.tileCount).toBe(3);
		expect(tiledInstanced?.maxActiveCellsPerTile).toBe(4_194_304);
		expect(tiledInstanced?.storageBuffers).toEqual([
			{
				name: 'selected-hour-utci-tile',
				bytes: 16_777_216,
				fitsMaxBufferSize: true,
				fitsMaxStorageBufferBindingSize: true
			},
			{
				name: 'active-canonical-indices-tile',
				bytes: 16_777_216,
				fitsMaxBufferSize: true,
				fitsMaxStorageBufferBindingSize: true
			}
		]);
	});

	it('never reselects active indexed even when it comfortably fits', () => {
		const decision = buildInnovationDistrict05mUtciSurfaceBudgetDecision({
			jsLargestTypedArrayBytes: 1024 * 1024 * 1024,
			jsTotalTypedArrayBytes: 2 * 1024 * 1024 * 1024,
			comfortableLimitRatio: 1
		});

		expect(
			decision.estimates.find((estimate) => estimate.strategy === 'active-indexed-quads')?.fits
		).toMatchObject({
			comfortableJsLargestTypedArray: true,
			comfortableJsTotalTypedArray: true,
			maxBufferSize: true,
			maxStorageBufferBindingSize: true
		});
		expect(decision.selectedStrategy).toBe('active-instanced-quads');
		expect(decision.planRevisionRequired).toBe(false);
		expect(decision.selectionReason).toContain('first-slice');
	});
});

describe('active mask UTCI instanced surface geometry', () => {
	it('uses one shared indexed quad and active-sized per-instance canonical indices for an asymmetric 3x2 xz layout', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		const geometry = createActiveMaskUtciInstancedSurfaceGeometry(layout);

		expect(geometry).toBeInstanceOf(THREE.InstancedBufferGeometry);
		expect(geometry.instanceCount).toBe(3);
		expect(geometry.getAttribute('position').count).toBe(4);
		expect(Array.from(geometry.index?.array ?? [])).toEqual([2, 3, 0, 3, 1, 0]);
		expect(geometry.index?.count).toBe(6);
		expect(geometry.getAttribute('activeCanonicalIndex').count).toBe(3);
		expect(geometry.getAttribute('activeCanonicalIndex').array).toBe(
			layout.activeCanonicalIndices
		);
		expect(geometry.getAttribute('position').array.byteLength).toBe(48);
		expect(geometry.index?.array.byteLength).toBe(24);
		expectAnalyticExtents(geometry, layout);
	});

	it('derives non-contiguous xz canonical centers from canonical index without row or column arrays', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		expect([0, 3, 5].map((canonicalIndex) =>
			getActiveMaskUtciCanonicalCellCenter({ layout, canonicalIndex })
		)).toEqual([
			{ x: -2, z: -1 },
			{ x: 0, z: 1 },
			{ x: 2, z: 1 }
		]);
		expect(layout).not.toHaveProperty('indexToRow');
		expect(layout).not.toHaveProperty('indexToColumn');
	});

	it('flips xy rows while preserving canonical ordering for an asymmetric 2x3 layout', () => {
		const layout = createActiveLayout({
			width: 2,
			height: 3,
			gridSize: 1,
			coordinateSystem: 'xy_ground',
			activeCanonicalIndices: [0, 1, 4, 5]
		});

		expect([0, 1, 4, 5].map((canonicalIndex) =>
			getActiveMaskUtciCanonicalCellCenter({ layout, canonicalIndex })
		)).toEqual([
			{ x: -0.5, z: 1 },
			{ x: -0.5, z: 0 },
			{ x: 0.5, z: 0 },
			{ x: 0.5, z: -1 }
		]);
		expectAnalyticExtents(createActiveMaskUtciInstancedSurfaceGeometry(layout), layout);
	});

	it('estimates geometry bytes from active cells instead of canonical cells', () => {
		const sparseLayout = createActiveLayout({
			width: 1_000,
			height: 1_000,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [7, 31, 999_999]
		});
		const smallerLayout = createActiveLayout({
			width: 1_000,
			height: 1_000,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [7]
		});

		expect(estimateActiveMaskUtciInstancedGeometryBytes(sparseLayout)).toMatchObject({
			vertexBufferBytes: 48,
			indexBufferBytes: 24,
			activeCanonicalIndexAttributeBytes: 12,
			totalBytes: 84
		});
		expect(estimateActiveMaskUtciInstancedGeometryBytes(smallerLayout).totalBytes).toBe(76);
		expect(estimateActiveMaskUtciInstancedGeometryBytes(sparseLayout).totalBytes).toBeLessThan(
			sparseLayout.canonicalCellCount * 4
		);
	});

	it('disposes the owned instanced geometry resource', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});
		const geometry = createActiveMaskUtciInstancedSurfaceGeometry(layout);
		const disposeSpy = vi.spyOn(geometry, 'dispose');

		disposeActiveMaskUtciInstancedSurfaceGeometry(geometry);

		expect(disposeSpy).toHaveBeenCalledOnce();
	});
});
