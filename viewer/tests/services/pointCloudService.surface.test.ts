import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
	createColors,
	createUtciSurfaceMesh,
	type UtciSurfaceMeshOptions,
	updateUtciSurfaceMesh
} from '$lib/services/pointCloudService';
import {
	createVertexToPointIndexArray,
	createComputeBufferUtciSurfaceMesh,
	getGpuNativeUtciSurfaceSource,
	updateComputeBufferUtciSurfaceMesh
} from '$lib/services/gpuUtciRenderBridge';
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
		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('cpu-uploaded-selected-hour');
		expect(getGpuNativeUtciSurfaceSource(mesh)).not.toBe('compute-buffer-selected-hour');
		expect(material.map ?? null).toBeNull();
	});

	it('maps each surface vertex back to the shuffled source point index', () => {
		const vertexToPoint = createVertexToPointIndexArray({
			width: 2,
			height: 2,
			gridSize: 1,
			numPositions: 4,
			centerX: 0.5,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			indexToRow: new Uint32Array([1, 0, 1, 0]),
			indexToColumn: new Uint32Array([1, 0, 0, 1]),
			indexToTexel: new Uint32Array([1, 2, 0, 3]),
			colorBuffer: new Uint8Array(2 * 2 * 4)
		});

		expect(Array.from(vertexToPoint)).toEqual([
			1, 1, 1, 1, 1, 1,
			3, 3, 3, 3, 3, 3,
			2, 2, 2, 2, 2, 2,
			0, 0, 0, 0, 0, 0
		]);
	});

	it('creates compute-buffer surfaces without uploading selected-hour UTCI from CPU readback', () => {
		const computeBuffer = {} as GPUBuffer;
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout: {
				width: 1,
				height: 1,
				gridSize: 1,
				numPositions: 1,
				centerX: 0.5,
				centerZ: 0.5,
				minX: 0,
				minZ: 0,
				baseY: 0,
				indexToRow: new Uint32Array([0]),
				indexToColumn: new Uint32Array([0]),
				indexToTexel: new Uint32Array([0]),
				colorBuffer: new Uint8Array(4)
			},
			utciBuffer: computeBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('compute-buffer-selected-hour');
		expect(mesh.userData.utciSurfaceSource).toBeUndefined();
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(computeBuffer);
		expect(Array.from(mesh.userData.gpuNativeUtciSurfaceState.utciStorageAttribute.array)).toEqual([0]);
		expect(
			Array.from(mesh.userData.gpuNativeUtciSurfaceState.vertexToPointStorageAttribute.array)
		).toEqual([0, 0, 0, 0, 0, 0]);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciRange).toEqual({ min: 10, max: 40 });
	});

	it('updates compute-buffer surfaces by storing pending GPU source and refreshing uniforms only', () => {
		const layout = {
			width: 2,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 1,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 1]),
			indexToTexel: new Uint32Array([0, 1]),
			colorBuffer: new Uint8Array(2 * 4)
		};
		const initialBuffer = {} as GPUBuffer;
		const nextBuffer = {} as GPUBuffer;
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: initialBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(
			updateComputeBufferUtciSurfaceMesh(mesh, {
				layout,
				utciBuffer: nextBuffer,
				utciRange: { min: 5, max: 55 }
			})
		).toBe(true);

		expect(mesh.userData.utciSurfaceSource).toBeUndefined();
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(nextBuffer);
		expect(Array.from(mesh.userData.gpuNativeUtciSurfaceState.utciStorageAttribute.array)).toEqual([
			0, 0
		]);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciRange).toEqual({ min: 5, max: 55 });
		expect(mesh.userData.gpuNativeUtciSurfaceState.minUniform.value).toBe(5);
		expect(mesh.userData.gpuNativeUtciSurfaceState.maxUniform.value).toBe(55);
	});

	it('rejects compute-buffer surface updates when the layout changes', () => {
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout: {
				width: 1,
				height: 1,
				gridSize: 1,
				numPositions: 1,
				centerX: 0.5,
				centerZ: 0.5,
				minX: 0,
				minZ: 0,
				baseY: 0,
				indexToRow: new Uint32Array([0]),
				indexToColumn: new Uint32Array([0]),
				indexToTexel: new Uint32Array([0]),
				colorBuffer: new Uint8Array(4)
			},
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(
			updateComputeBufferUtciSurfaceMesh(mesh, {
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
					indexToRow: new Uint32Array([0, 0]),
					indexToColumn: new Uint32Array([0, 1]),
					indexToTexel: new Uint32Array([0, 1]),
					colorBuffer: new Uint8Array(8)
				},
				utciBuffer: {} as GPUBuffer,
				utciRange: { min: 5, max: 55 }
			})
		).toBe(false);
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
