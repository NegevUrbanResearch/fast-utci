import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
	createComputeBufferUtciSurfaceMesh,
	getComputeBufferUtciStorageAttribute,
	getGpuNativeUtciSurfaceSource,
	isComputeBufferUtciSurfaceLayoutCompatible,
	updateComputeBufferUtciSurfaceMesh
} from '$lib/services/gpuUtciRenderBridge';
import type {
	ActiveCellsUtciGridLayout,
	DenseUtciGridLayout
} from '$lib/services/utciGridLayoutTopology';

function createActiveLayout(params?: {
	activeCanonicalIndices?: number[];
	activeMaskSignature?: string;
}): ActiveCellsUtciGridLayout {
	const activeCanonicalIndices = params?.activeCanonicalIndices ?? [0, 3, 5];
	return {
		renderTopology: 'active-cells',
		width: 3,
		height: 2,
		gridSize: 2,
		coordinateSystem: 'xz_ground',
		numPositions: activeCanonicalIndices.length,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: 2,
		centerZ: 1,
		baseY: 0,
		renderCellCount: activeCanonicalIndices.length,
		canonicalCellCount: 6,
		activeCanonicalIndices: new Uint32Array(activeCanonicalIndices),
		activeMaskSignature: params?.activeMaskSignature ?? 'active-mask-a'
	};
}

function createDenseLayout(): DenseUtciGridLayout {
	return {
		renderTopology: 'dense-grid',
		width: 3,
		height: 2,
		gridSize: 2,
		coordinateSystem: 'xz_ground',
		numPositions: 6,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: 2,
		centerZ: 1,
		baseY: 0,
		renderCellCount: 6,
		canonicalCellCount: 6,
		indexToRow: new Uint32Array([0, 0, 0, 1, 1, 1]),
		indexToColumn: new Uint32Array([0, 1, 2, 0, 1, 2]),
		indexToTexel: new Uint32Array([0, 1, 2, 3, 4, 5]),
		cellToPointIndex: new Int32Array([0, 1, 2, 3, 4, 5]),
		colorBuffer: new Uint8Array(24)
	};
}

describe('active compute-buffer UTCI render bridge', () => {
	it('creates an active instanced mesh with compact per-instance canonical data and no dense cell map', () => {
		const layout = createActiveLayout();
		const utciBuffer = {} as GPUBuffer;

		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer,
			utciRange: { min: 10, max: 40 }
		});
		const state = mesh.userData.gpuNativeUtciSurfaceState;
		const activeCanonicalIndexAttribute = mesh.geometry.getAttribute(
			'activeCanonicalIndex'
		) as THREE.InstancedBufferAttribute;

		expect(mesh.geometry).toBeInstanceOf(THREE.InstancedBufferGeometry);
		expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(3);
		expect(mesh.geometry.getAttribute('position').count).toBe(4);
		expect(mesh.geometry.index?.count).toBe(6);
		expect(activeCanonicalIndexAttribute.array).toBe(layout.activeCanonicalIndices);
		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('compute-buffer-selected-hour');
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(utciBuffer);
		expect(state.renderTopology).toBe('active-cells');
		expect(state.activeMaskSignature).toBe('active-mask-a');
		expect(state.utciStorageAttribute.count).toBe(3);
		expect(state.activeCanonicalIndexAttribute.array).toBe(layout.activeCanonicalIndices);
		expect(state).not.toHaveProperty('cellToPointStorageAttribute');
		expect(mesh.raycast).not.toBe(THREE.Mesh.prototype.raycast);
		expect(mesh.userData.raycastDisabledReason).toContain('active-instanced');
		expect(mesh.geometry.boundingBox?.min.toArray()).toEqual([-3, 0, -2]);
		expect(mesh.geometry.boundingBox?.max.toArray()).toEqual([3, 0, 2]);
		expect(mesh.userData.renderOwnedSelectedHourBytes).toBe(1120);
	});

	it('updates active compute-buffer surfaces by swapping the pending source and preserving compact storage', () => {
		const layout = createActiveLayout();
		const initialBuffer = {} as GPUBuffer;
		const nextBuffer = {} as GPUBuffer;
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: initialBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(
			updateComputeBufferUtciSurfaceMesh(mesh, {
				layout: createActiveLayout({ activeMaskSignature: 'active-mask-a' }),
				utciBuffer: nextBuffer,
				utciRange: { min: 5, max: 55 }
			})
		).toBe(true);

		const state = mesh.userData.gpuNativeUtciSurfaceState;
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(nextBuffer);
		expect(state.utciRange).toEqual({ min: 5, max: 55 });
		expect(state.cellToPointStorageAttribute).toBeUndefined();
		expect(state.utciStorageAttribute.count).toBe(3);
		expect(mesh.userData.renderOwnedSelectedHourBytes).toBe(1120);
		expect(getComputeBufferUtciStorageAttribute(mesh)?.count).toBe(3);
	});

	it('rejects active-to-dense surface reuse in the bridge predicate', () => {
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout: createActiveLayout(),
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(isComputeBufferUtciSurfaceLayoutCompatible(mesh, createDenseLayout())).toBe(
			false
		);
	});
});
