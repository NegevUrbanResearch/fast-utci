import { describe, expect, it } from 'vitest';
import {
	createGpuNativeUtciSurfaceMesh,
	createComputeBufferUtciSurfaceMesh,
	getComputeBufferUtciSurfaceLayoutCompatibilityState,
	evaluateComputeBufferUtciSurfaceLayoutCompatibility,
	isComputeBufferUtciSurfaceLayoutCompatible
} from '$lib/services/gpuUtciRenderBridge';
import type { DenseUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

describe('legacy CPU surface transition to compute storage', () => {
	it('rejects CPU color storage without treating it as compute-output storage', () => {
		const layout: DenseUtciGridLayout = {
			renderTopology: 'dense-grid', width: 1, height: 1, gridSize: 2,
			coordinateSystem: 'xz_ground', numPositions: 1, minX: 0, minZ: 0,
			minY: 0, maxY: 0, centerX: 0, centerZ: 0, baseY: 0,
			renderCellCount: 1, canonicalCellCount: 1,
			indexToRow: new Uint32Array([0]), indexToColumn: new Uint32Array([0]),
			indexToTexel: new Uint32Array([0]), cellToPointIndex: new Int32Array([0]),
			colorBuffer: new Uint8Array(4)
		};
		const mesh = createGpuNativeUtciSurfaceMesh({ layout, colors: new Float32Array([1, 0, 0]) });
		mesh.userData.utciLayout = layout;
		const colorStorage = mesh.userData.gpuNativeUtciSurfaceState.colorStorageAttribute;
		const state = getComputeBufferUtciSurfaceLayoutCompatibilityState(mesh);
		expect(evaluateComputeBufferUtciSurfaceLayoutCompatibility({
			state, previousLayout: layout, nextLayout: layout
		})).toMatchObject({ compatible: false, wrongSource: true });
		expect(isComputeBufferUtciSurfaceLayoutCompatible(mesh, layout)).toBe(false);
		expect(mesh.userData.gpuNativeUtciSurfaceState.colorStorageAttribute).toBe(colorStorage);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciStorageAttribute).toBeUndefined();
		const replacement = createComputeBufferUtciSurfaceMesh({
			layout, utciBuffer: {} as GPUBuffer, utciRange: { min: 10, max: 40 }
		});
		replacement.userData.utciLayout = layout;
		expect(isComputeBufferUtciSurfaceLayoutCompatible(replacement, layout)).toBe(true);
	});
});
