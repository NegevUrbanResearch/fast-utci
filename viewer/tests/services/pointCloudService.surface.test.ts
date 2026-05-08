import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
	createColors,
	createUtciSurfaceMesh,
	type UtciSurfaceMeshOptions,
	updateUtciSurfaceMesh
} from '$lib/services/pointCloudService';
import type { Analysis } from '$lib/types/analysis';

function createAnalysis(params?: {
	positions?: number[];
	utciValues?: number[];
	gridSize?: number;
	coordinateSystem?: 'xy_ground' | 'xz_ground';
	bounds?: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number };
	source?: string;
}): Analysis {
	const analysis = {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: 4,
			hours: ['00:00'],
			utci_range: { min: 10, max: 40 },
			grid_size: params?.gridSize ?? 1,
			coordinate_system: params?.coordinateSystem ?? 'xz_ground',
			model_file: 'test.obj',
			bounds: params?.bounds
		},
		data: {
			numPositions: 4,
			numHours: 1 as const,
			positions: new Float32Array(
				params?.positions ?? [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
			),
			utciValues: new Float32Array(params?.utciValues ?? [10, 20, 30, 40])
		}
	} as Analysis & { __source?: string };

	if (params?.source) {
		analysis.__source = params.source;
	}

	return analysis;
}

function getGpuNativeColorArray(mesh: THREE.Mesh): Float32Array {
	return (
		mesh.userData.gpuNativeUtciSurfaceState.colorStorageAttribute.array as Float32Array
	);
}

function toLinearColor(r: number, g: number, b: number): THREE.Color {
	return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}

describe('pointCloudService UTCI surface seam', () => {
	it('paints the same logical cells across DataTexture and gpuNative backends', () => {
		const analysis = createAnalysis();
		const colors = createColors(analysis, 0, 'normalized', 'utci');

		const dataTextureMesh = createUtciSurfaceMesh(analysis);
		const dataTextureLayout = dataTextureMesh.userData.utciLayout;

		expect(Array.from(dataTextureLayout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(dataTextureLayout.indexToColumn)).toEqual([0, 0, 1, 1]);
		expect(Array.from(dataTextureLayout.indexToTexel)).toEqual([2, 0, 3, 1]);

		for (let index = 0; index < analysis.data.numPositions; index += 1) {
			const texelOffset = dataTextureLayout.indexToTexel[index] * 4;
			expect(dataTextureLayout.colorBuffer[texelOffset]).toBe(
				Math.floor(colors[index * 3] * 255)
			);
			expect(dataTextureLayout.colorBuffer[texelOffset + 1]).toBe(
				Math.floor(colors[index * 3 + 1] * 255)
			);
			expect(dataTextureLayout.colorBuffer[texelOffset + 2]).toBe(
				Math.floor(colors[index * 3 + 2] * 255)
			);
			expect(dataTextureLayout.colorBuffer[texelOffset + 3]).toBe(255);
		}

		const gpuNativeMesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});
		const gpuNativeLayout = gpuNativeMesh.userData.utciLayout;
		const gpuNativeColors = getGpuNativeColorArray(gpuNativeMesh);

		expect(Array.from(gpuNativeLayout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(gpuNativeLayout.indexToColumn)).toEqual([0, 0, 1, 1]);

		for (let index = 0; index < analysis.data.numPositions; index += 1) {
			const cellIndex =
				gpuNativeLayout.indexToRow[index] * gpuNativeLayout.width +
				gpuNativeLayout.indexToColumn[index];
			const cellOffset = cellIndex * 6 * 4;
			const expected = toLinearColor(
				colors[index * 3],
				colors[index * 3 + 1],
				colors[index * 3 + 2]
			);

			expect(gpuNativeColors[cellOffset]).toBeCloseTo(expected.r, 6);
			expect(gpuNativeColors[cellOffset + 1]).toBeCloseTo(expected.g, 6);
			expect(gpuNativeColors[cellOffset + 2]).toBeCloseTo(expected.b, 6);
			expect(gpuNativeColors[cellOffset + 3]).toBeCloseTo(0.9, 6);
		}
	});

	it('rebuilds logical fallback cell mapping from metadata.bounds for live webgpu analyses', () => {
		const analysis = createAnalysis({
			positions: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
			coordinateSystem: 'xy_ground',
			bounds: { x_min: 0, x_max: 1, y_min: 0, y_max: 1, z: 2 },
			source: 'webgpu'
		});

		const mesh = createUtciSurfaceMesh(analysis);
		const layout = mesh.userData.utciLayout;

		expect(layout.width).toBe(2);
		expect(layout.height).toBe(2);
		expect(layout.minX).toBe(0);
		expect(layout.minZ).toBe(-1);
		expect(layout.centerX).toBe(0.5);
		expect(layout.centerZ).toBe(-0.5);
		expect(layout.baseY).toBeCloseTo(1.95, 6);
		expect(Array.from(layout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(layout.indexToColumn)).toEqual([0, 0, 1, 1]);
		expect(Array.from(layout.indexToTexel)).toEqual([2, 0, 3, 1]);
	});

	it('returns an observable recreate-required signal from updateUtciSurfaceMesh', () => {
		const analysis = createAnalysis();
		const mesh = createUtciSurfaceMesh(analysis);

		expect(
			updateUtciSurfaceMesh(mesh, {
				analysis,
				backend: 'gpuNative'
			})
		).toBe(false);

		expect(
			updateUtciSurfaceMesh(mesh, {
				analysis
			})
		).toBe(true);
	});

	it('labels gpuNative selected-hour surfaces as cpu-uploaded without a texture map', () => {
		const analysis = createAnalysis();
		const mesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});
		const material = mesh.material as THREE.Material & { map?: THREE.Texture | null };

		expect(mesh.userData.utciSurfaceBackend).toBe('gpuNative');
		expect(mesh.userData.utciSurfaceSource).toBe('cpu-uploaded-selected-hour');
		expect(material.map ?? null).toBeNull();
	});

	it('stores real per-backend surface diagnostics on mesh userData', () => {
		const analysis = createAnalysis();
		const dataTextureMesh = createUtciSurfaceMesh({ analysis, backend: 'dataTexture' });

		expect(dataTextureMesh.userData.dataTextureBuildCount).toBe(1);
		expect(dataTextureMesh.userData.selectedHourTransferCount).toBeUndefined();
		expect(dataTextureMesh.userData.utciSurfaceSource).toBeUndefined();

		expect(
			updateUtciSurfaceMesh(dataTextureMesh, {
				analysis,
				backend: 'dataTexture'
			})
		).toBe(true);
		expect(dataTextureMesh.userData.dataTextureBuildCount).toBe(2);
		expect(dataTextureMesh.userData.selectedHourTransferCount).toBeUndefined();

		const gpuNativeMesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});

		expect(gpuNativeMesh.userData.dataTextureBuildCount).toBe(0);
		expect(gpuNativeMesh.userData.selectedHourTransferCount).toBe(1);
		expect(
			updateUtciSurfaceMesh(gpuNativeMesh, {
				analysis,
				backend: 'gpuNative'
			})
		).toBe(true);
		expect(gpuNativeMesh.userData.dataTextureBuildCount).toBe(0);
		expect(gpuNativeMesh.userData.selectedHourTransferCount).toBe(2);
	});
});
