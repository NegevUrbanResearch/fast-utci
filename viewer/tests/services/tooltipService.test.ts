import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';
import { buildUtciGridLayout } from '$lib/services/pointCloudService';
import { createVertexToPointIndexArray } from '$lib/services/gpuUtciRenderBridge';
import {
	TOOLTIP_SLOW_BUDGET_MS,
	createEmptyTooltipInteractionDiagnostics,
	getTooltipData,
	resolvePositionIndexFromIntersection,
	recordTooltipInteractionMeasurement
} from '$lib/services/tooltipService';

describe('resolvePositionIndexFromIntersection', () => {
	it('returns the expected point index from the hovered UTCI surface cell', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 1, layout.baseY, 1),
			layout,
			analysis
		);

		expect(positionIndex).toBe(3);
	});

	it('falls back to the nearest-point scan when the reverse mapping is invalid', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		layout.cellToPointIndex!.fill(analysis.data.numPositions + 10);
		const mesh = createSurfaceTestMesh(layout);

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 1, layout.baseY, 1),
			layout,
			analysis
		);

		expect(positionIndex).toBe(3);
	});

	it('preserves the legacy distance guard for cell-corner misses', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 0.5, layout.baseY, 0.5),
			layout,
			analysis
		);

		expect(positionIndex).toBeNull();
	});

	it('rejects direct cell hits when the mapped source position is non-finite', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);

		analysis.data.positions[9] = Number.NaN;

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 1, layout.baseY, 1),
			layout,
			analysis
		);

		expect(positionIndex).toBeNull();
	});

	it('falls back conservatively when multiple points map into the same cell', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 0, layout.baseY, 0),
			layout,
			analysis
		);

		expect(positionIndex).toBe(0);
	});
});

describe('gpu-native ambiguous cell mapping', () => {
	it('keeps render-local last-writer mapping for ambiguous cells', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);

		expect(Array.from(layout.cellToPointIndex ?? [])).toEqual([-2]);
		expect(Array.from(createVertexToPointIndexArray(layout))).toEqual([1, 1, 1, 1, 1, 1]);
	});
});

describe('tooltip interaction diagnostics', () => {
	it('resolves a hovered UTCI surface cell through the plane fast path without mesh raycasting', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});
		const intersectSpy = vi.spyOn(THREE.Raycaster.prototype, 'intersectObject');

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'utci',
			0,
			createDomRect()
		);

		expect(result).toEqual({
			value: 23,
			position: { x: 1, y: 0, z: 1 },
			positionIndex: 3
		});
		expect(intersectSpy).not.toHaveBeenCalled();

		intersectSpy.mockRestore();
	});

	it('keeps the conservative ambiguous-cell fallback on the plane fast path', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(0, 5, 0),
			target: new THREE.Vector3(0, layout.baseY, 0)
		});
		const intersectSpy = vi.spyOn(THREE.Raycaster.prototype, 'intersectObject');

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'utci',
			0,
			createDomRect()
		);

		expect(result?.positionIndex).toBe(0);
		expect(intersectSpy).not.toHaveBeenCalled();

		intersectSpy.mockRestore();
	});

	it('starts with conservative hover timing defaults', () => {
		const diagnostics = createEmptyTooltipInteractionDiagnostics(false);

		expect(diagnostics.enabled).toBe(true);
		expect(diagnostics.disabledByQuery).toBe(false);
		expect(diagnostics.slowThresholdMs).toBe(TOOLTIP_SLOW_BUDGET_MS);
		expect(diagnostics.sampleCount).toBe(0);
		expect(diagnostics.hitCount).toBe(0);
		expect(diagnostics.missCount).toBe(0);
		expect(diagnostics.overBudgetCount).toBe(0);
		expect(diagnostics.lastOutcome).toBeNull();
		expect(diagnostics.lastRaycastMs).toBeNull();
		expect(diagnostics.maxRaycastMs).toBe(0);
		expect(diagnostics.lastNearestPointMs).toBeNull();
		expect(diagnostics.maxNearestPointMs).toBe(0);
		expect(diagnostics.lastTotalMs).toBeNull();
		expect(diagnostics.maxTotalMs).toBe(0);
	});

	it('aggregates hit, miss, max, and over-budget timing stats without storing histories', () => {
		const first = recordTooltipInteractionMeasurement(
			createEmptyTooltipInteractionDiagnostics(false),
			{
				hit: true,
				raycastMs: 1.5,
				nearestPointMs: 2.5,
				totalMs: 4,
				overBudget: false
			}
		);
		const second = recordTooltipInteractionMeasurement(first, {
			hit: false,
			raycastMs: 3,
			nearestPointMs: 7,
			totalMs: 10,
			overBudget: true
		});

		expect(second.sampleCount).toBe(2);
		expect(second.hitCount).toBe(1);
		expect(second.missCount).toBe(1);
		expect(second.overBudgetCount).toBe(1);
		expect(second.lastOutcome).toBe('miss');
		expect(second.lastRaycastMs).toBe(3);
		expect(second.maxRaycastMs).toBe(3);
		expect(second.lastNearestPointMs).toBe(7);
		expect(second.maxNearestPointMs).toBe(7);
		expect(second.lastTotalMs).toBe(10);
		expect(second.maxTotalMs).toBe(10);
		expect(second.slowThresholdMs).toBe(TOOLTIP_SLOW_BUDGET_MS);
	});

	it('records one miss sample when tooltip preconditions are missing', () => {
		const samples: Array<{
			hit: boolean;
			raycastMs: number;
			nearestPointMs: number;
			totalMs: number;
			overBudget: boolean;
		}> = [];

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			new THREE.PerspectiveCamera(),
			null,
			null,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result).toBeNull();
		expect(samples).toHaveLength(1);
		expect(samples[0]?.hit).toBe(false);
		expect(samples[0]?.nearestPointMs).toBe(0);
	});

	it('falls back to mesh raycasting when the plane fast path is not applicable', () => {
		const samples: Array<{
			hit: boolean;
			raycastMs: number;
			nearestPointMs: number;
			totalMs: number;
			overBudget: boolean;
		}> = [];
		const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 100);
		camera.position.set(0, 0, 5);
		camera.lookAt(0, 0, 0);
		camera.updateProjectionMatrix();
		camera.updateMatrixWorld(true);

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
		const points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 1 }));
		points.userData.utciLayout = {} as any;
		points.updateMatrixWorld(true);
		const intersectSpy = vi.spyOn(THREE.Raycaster.prototype, 'intersectObject');

		const analysis = {
			data: {
				numPositions: 1,
				numHours: 1,
				positions: new Float32Array([0, 0, 0]),
				utciValues: new Float32Array([26.5])
			},
			metadata: {
				grid_size: 2,
				coordinate_system: 'xy_ground'
			}
		} as any;

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			points as unknown as THREE.Mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result).toEqual({
			value: 26.5,
			position: { x: 0, y: 0, z: 0 },
			positionIndex: 0
		});
		expect(samples).toHaveLength(1);
		expect(samples[0]?.hit).toBe(true);
		expect(samples[0]?.raycastMs).toBeGreaterThanOrEqual(0);
		expect(samples[0]?.nearestPointMs).toBeGreaterThanOrEqual(0);
		expect(samples[0]?.totalMs).toBeGreaterThanOrEqual(0);
		expect(intersectSpy).toHaveBeenCalledOnce();

		intersectSpy.mockRestore();
	});

	it('falls back to mesh raycasting when the plane fast path is invalid for a surface mesh', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const invalidMesh = createSurfaceTestMesh(layout);
		const fallbackMesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});
		const intersectSpy = vi
			.spyOn(THREE.Raycaster.prototype, 'intersectObject')
			.mockReturnValue([createIntersection(fallbackMesh, 1, layout.baseY, 1)]);

		invalidMesh.matrixWorld.elements[0] = Number.NaN;

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			invalidMesh,
			analysis,
			'utci',
			0,
			createDomRect()
		);

		expect(result).toEqual({
			value: 23,
			position: { x: 1, y: 0, z: 1 },
			positionIndex: 3
		});
		expect(intersectSpy).toHaveBeenCalledOnce();

		intersectSpy.mockRestore();
	});

	it('records zero nearestPointMs when the direct cell path resolves the hit', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const performanceSpy = mockPerformanceNow([0, 1, 2, 5]);
		const intersectSpy = vi
			.spyOn(THREE.Raycaster.prototype, 'intersectObject')
			.mockReturnValue([createIntersection(mesh, 1, layout.baseY, 1)]);
		const samples: Array<{
			hit: boolean;
			raycastMs: number;
			nearestPointMs: number;
			totalMs: number;
			overBudget: boolean;
		}> = [];

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			new THREE.PerspectiveCamera(),
			mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result?.positionIndex).toBe(3);
		expect(samples).toHaveLength(1);
		expect(samples[0]?.nearestPointMs).toBe(0);

		intersectSpy.mockRestore();
		performanceSpy.mockRestore();
	});

	it('records nearestPointMs only when the conservative fallback scan runs', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const performanceSpy = mockPerformanceNow([0, 1, 2, 3, 7, 9]);
		const intersectSpy = vi
			.spyOn(THREE.Raycaster.prototype, 'intersectObject')
			.mockReturnValue([createIntersection(mesh, 0, layout.baseY, 0)]);
		const samples: Array<{
			hit: boolean;
			raycastMs: number;
			nearestPointMs: number;
			totalMs: number;
			overBudget: boolean;
		}> = [];

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			new THREE.PerspectiveCamera(),
			mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result?.positionIndex).toBe(0);
		expect(samples).toHaveLength(1);
		expect(samples[0]?.nearestPointMs).toBe(4);

		intersectSpy.mockRestore();
		performanceSpy.mockRestore();
	});
});

function createDomRect(): DOMRect {
	return {
		left: 0,
		top: 0,
		width: 100,
		height: 100,
		x: 0,
		y: 0,
		bottom: 100,
		right: 100,
		toJSON: () => ({})
	};
}

function createGridAnalysis() {
	return {
		data: {
			numPositions: 4,
			numHours: 1,
			positions: new Float32Array([
				0, 0, 0,
				1, 0, 0,
				0, 0, 1,
				1, 0, 1
			]),
			utciValues: new Float32Array([20, 21, 22, 23])
		},
		metadata: {
			grid_size: 1,
			coordinate_system: 'xz_ground'
		}
	} as any;
}

function createAmbiguousCellAnalysis() {
	return {
		data: {
			numPositions: 2,
			numHours: 1,
			positions: new Float32Array([
				0, 0, 0,
				0.2, 0, 0.2
			]),
			utciValues: new Float32Array([20, 21])
		},
		metadata: {
			grid_size: 1,
			coordinate_system: 'xz_ground'
		}
	} as any;
}

function createSurfaceTestMesh(layout: ReturnType<typeof buildUtciGridLayout>): THREE.Mesh {
	const geometry = new THREE.PlaneGeometry(layout.width * layout.gridSize, layout.height * layout.gridSize);
	geometry.rotateX(-Math.PI / 2);

	const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
	mesh.position.set(layout.centerX, layout.baseY, layout.centerZ);
	mesh.userData.utciLayout = layout;
	mesh.updateMatrixWorld(true);
	return mesh;
}

function createIntersection(
	object: THREE.Object3D,
	x: number,
	y: number,
	z: number
): THREE.Intersection {
	return {
		distance: 0,
		point: new THREE.Vector3(x, y, z),
		object
	} as THREE.Intersection;
}

function createHoverCamera(params: {
	position: THREE.Vector3;
	target: THREE.Vector3;
}): THREE.PerspectiveCamera {
	const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 100);
	camera.position.copy(params.position);
	camera.lookAt(params.target);
	camera.updateProjectionMatrix();
	camera.updateMatrixWorld(true);
	return camera;
}

function mockPerformanceNow(values: number[]) {
	let index = 0;
	return vi.spyOn(performance, 'now').mockImplementation(() => {
		const value = values[Math.min(index, values.length - 1)];
		index += 1;
		return value ?? 0;
	});
}
