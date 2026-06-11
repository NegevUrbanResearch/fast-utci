import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	buildUtciGridLayout,
	buildUtciGridLayoutReuseProofDiagnostics
} from '$lib/services/pointCloudService';
import { clearMetricPointReadbackCache } from '$lib/compute/gpu/metricPointReadback';
import { createVertexToPointIndexArray } from '$lib/services/gpuUtciRenderBridge';
import {
	TOOLTIP_SLOW_BUDGET_MS,
	createEmptyTooltipInteractionDiagnostics,
	getTooltipData,
	getTooltipDataAsync,
	getTooltipProbeData,
	resolvePositionIndexFromIntersection,
	recordTooltipInteractionMeasurement
} from '$lib/services/tooltipService';

afterEach(() => {
	vi.unstubAllGlobals();
	clearMetricPointReadbackCache();
});

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

	it('uses the render-consistent last-writer mapping when multiple points map into the same cell', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 0, layout.baseY, 0),
			layout,
			analysis
		);

		expect(positionIndex).toBe(1);
	});

	it('uses a deterministic layout fallback for ambiguous cells without scanning nearest points', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const samples: Array<{
			hit: boolean;
			raycastMs: number;
			nearestPointMs: number;
			totalMs: number;
			overBudget: boolean;
		}> = [];

		const positionIndex = resolvePositionIndexFromIntersection(
			createIntersection(mesh, 0, layout.baseY, 0),
			layout,
			analysis
		);
		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			createHoverCamera({
				position: new THREE.Vector3(0, 5, 0),
				target: new THREE.Vector3(0, layout.baseY, 0)
			}),
			mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(positionIndex).toBe(1);
		expect(result?.positionIndex).toBe(1);
		expect(samples).toHaveLength(1);
		expect(samples[0]?.nearestPointMs).toEqual(expect.any(Number));
		expect(samples[0]?.nearestPointMs).toBeGreaterThanOrEqual(0);
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

describe('layout reuse hover lookup proof', () => {
	it('resolves representative lookup coordinates to the same point when proof says layouts are compatible', () => {
		const analysis = createGridAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = buildUtciGridLayout(analysis);
		const previousMesh = createSurfaceTestMesh(previousLayout);
		const nextMesh = createSurfaceTestMesh(nextLayout);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: true
		});

		expect(proof.decision).toBe('reuse-safe');

		const previousIndex = resolvePositionIndexFromIntersection(
			createIntersection(previousMesh, 1, previousLayout.baseY, 1),
			previousLayout,
			analysis
		);
		const nextIndex = resolvePositionIndexFromIntersection(
			createIntersection(nextMesh, 1, nextLayout.baseY, 1),
			nextLayout,
			analysis
		);

		expect(previousIndex).toBe(3);
		expect(nextIndex).toBe(previousIndex);
	});

	it('does not claim lookup safety for ambiguous same-count layouts with different mapping', () => {
		const analysis = createAmbiguousCellAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = {
			...buildUtciGridLayout(analysis),
			indexToColumn: new Uint32Array([0, 1])
		};
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: false
		});

		expect(proof.decision).toBe('rebuild-required');
		expect(proof.cellToPointMappingMatch).toBe(false);
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
		const samples: Array<Parameters<typeof recordTooltipInteractionMeasurement>[1]> = [];

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result).toEqual({
			value: 23,
			position: { x: 1, y: 0, z: 1 },
			positionIndex: 3
		});
		expect(intersectSpy).not.toHaveBeenCalled();
		expect(samples).toHaveLength(1);
		expect(samples[0]).toMatchObject({
			hit: true,
			resolutionPath: 'plane-cell',
			directCellHit: true,
			nearestScanUsed: false,
			directCellMissCount: 0
		});
		const diagnostics = recordTooltipInteractionMeasurement(
			createEmptyTooltipInteractionDiagnostics(false),
			samples[0]!
		);
		expect(diagnostics.planeCellPathCount).toBe(1);
		expect(diagnostics.meshRaycastPathCount).toBe(0);
		expect(diagnostics.directCellHitCount).toBe(1);
		expect(diagnostics.nearestScanFallbackCount).toBe(0);

		intersectSpy.mockRestore();
	});

	it('keeps the ambiguous-cell plane fast path off the mesh raycast fallback', () => {
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

		expect(result?.positionIndex).toBe(1);
		expect(intersectSpy).not.toHaveBeenCalled();

		intersectSpy.mockRestore();
	});

	it('starts with conservative hover timing defaults', () => {
		const diagnostics = createEmptyTooltipInteractionDiagnostics(false);

		expect(diagnostics.enabled).toBe(true);
		expect(diagnostics.disabledByQuery).toBe(false);
		expect(diagnostics.slowThresholdMs).toBe(TOOLTIP_SLOW_BUDGET_MS);
		expect(diagnostics.hoverAttemptCount).toBe(0);
		expect(diagnostics.suppressedHoverCount).toBe(0);
		expect(diagnostics.throttledHoverCount).toBe(0);
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
		expect(diagnostics.lastResolutionPath).toBeNull();
		expect(diagnostics.planeCellPathCount).toBe(0);
		expect(diagnostics.meshRaycastPathCount).toBe(0);
		expect(diagnostics.directCellHitCount).toBe(0);
		expect(diagnostics.directCellMissCount).toBe(0);
		expect(diagnostics.nearestScanFallbackCount).toBe(0);
		expect(diagnostics.metricPointReadbackCount).toBe(0);
		expect(diagnostics.metricPointReadbackBytes).toBe(0);
		expect(diagnostics.metricPointReadbackLastBytes).toBeNull();
		expect(diagnostics.metricPointReadbackCacheEntries).toBe(0);
		expect(diagnostics.metricPointReadbackCacheHitCount).toBe(0);
		expect(diagnostics.metricPointReadbackCacheMissCount).toBe(0);
		expect(diagnostics.metricPointReadbackLastLatencyMs).toBeNull();
		expect(diagnostics.metricPointReadbackMaxLatencyMs).toBe(0);
	});

	it('aggregates hit, miss, max, and over-budget timing stats without storing histories', () => {
		const first = recordTooltipInteractionMeasurement(
			createEmptyTooltipInteractionDiagnostics(false),
			{
				hit: true,
				raycastMs: 1.5,
				nearestPointMs: 2.5,
				totalMs: 4,
				overBudget: false,
				resolutionPath: 'plane-cell',
				directCellHit: true,
				nearestScanUsed: false,
				directCellMissCount: 0
			}
		);
		const second = recordTooltipInteractionMeasurement(first, {
			hit: false,
			raycastMs: 3,
			nearestPointMs: 7,
			totalMs: 10,
			overBudget: true,
			resolutionPath: 'mesh-raycast',
			directCellHit: false,
			nearestScanUsed: true,
			directCellMissCount: 1
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
		expect(second.lastResolutionPath).toBe('mesh-raycast');
		expect(second.planeCellPathCount).toBe(1);
		expect(second.meshRaycastPathCount).toBe(1);
		expect(second.directCellHitCount).toBe(1);
		expect(second.directCellMissCount).toBe(1);
		expect(second.nearestScanFallbackCount).toBe(1);
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
		const samples: Array<Parameters<typeof recordTooltipInteractionMeasurement>[1]> = [];
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
		expect(samples[0]?.resolutionPath).toBe('mesh-raycast');
		expect(intersectSpy).toHaveBeenCalledOnce();
		const diagnostics = recordTooltipInteractionMeasurement(
			createEmptyTooltipInteractionDiagnostics(false),
			samples[0]!
		);
		expect(diagnostics.meshRaycastPathCount).toBe(1);
		expect(diagnostics.planeCellPathCount).toBe(0);

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

	it('records direct cell mapping time when the direct cell path resolves the hit', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});
		const performanceSpy = mockPerformanceNow([0, 1, 3, 5]);
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
			camera,
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
		expect(samples[0]?.nearestPointMs).toBe(2);

		intersectSpy.mockRestore();
		performanceSpy.mockRestore();
	});

	it('records mapping time when ambiguous cells resolve through the layout fallback', () => {
		const analysis = createAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(0, 5, 0),
			target: new THREE.Vector3(0, layout.baseY, 0)
		});
		const performanceSpy = mockPerformanceNow([0, 1, 4, 6]);
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
			camera,
			mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result?.positionIndex).toBe(1);
		expect(samples).toHaveLength(1);
		expect(samples[0]?.nearestPointMs).toBe(3);

		intersectSpy.mockRestore();
		performanceSpy.mockRestore();
	});

	it('keeps 0.5m near-corner ambiguous cells on the render-consistent direct path', () => {
		const analysis = createNearCornerAmbiguousCellAnalysis();
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(0.49, 5, 0.49),
			target: new THREE.Vector3(0.49, layout.baseY, 0.49)
		});
		const intersectSpy = vi
			.spyOn(THREE.Raycaster.prototype, 'intersectObject')
			.mockReturnValue([createIntersection(mesh, 0.49, layout.baseY, 0.49)]);
		const samples: Array<{
			hit: boolean;
			raycastMs: number;
			nearestPointMs: number;
			totalMs: number;
			overBudget: boolean;
		}> = [];

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'utci',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement)
			}
		);

		expect(result?.positionIndex).toBe(1);
		expect(samples).toHaveLength(1);
		expect(intersectSpy).not.toHaveBeenCalled();

		intersectSpy.mockRestore();
	});

	it('records nearest-scan fallback diagnostics when direct cell mapping fails', () => {
		const analysis = createGridAnalysis();
		const layout = buildUtciGridLayout(analysis);
		layout.cellToPointIndex!.fill(analysis.data.numPositions + 10);
		(layout as any).indexToRow = new Uint32Array([]);
		(layout as any).indexToColumn = new Uint32Array([]);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});
		const samples: Array<Parameters<typeof recordTooltipInteractionMeasurement>[1]> = [];

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
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
		expect(samples[0]).toMatchObject({
			hit: true,
			resolutionPath: 'plane-cell',
			directCellHit: false,
			nearestScanUsed: true,
			directCellMissCount: 1
		});
		const diagnostics = recordTooltipInteractionMeasurement(
			createEmptyTooltipInteractionDiagnostics(false),
			samples[0]!
		);
		expect(diagnostics.nearestScanFallbackCount).toBe(1);
		expect(diagnostics.directCellMissCount).toBe(1);
	});

	it('reads Shading Index directly from CPU fallback/debug values when present', () => {
		const analysis = {
			...createGridAnalysis(),
			data: {
				...createGridAnalysis().data,
				shadingIndex: new Float32Array([0.1, 0.2, 0.3, 0.85])
			},
			metadata: {
				...createGridAnalysis().metadata,
				has_shading_index: true,
				shading_index_range: { min: 0, max: 1 }
			}
		} as any;
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});

		const result = getTooltipData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'shading_index',
			0,
			createDomRect()
		);

		expect(result?.positionIndex).toBe(3);
		expect(result?.value).toBeCloseTo(0.85);
	});

	it('uses the GPU point-value path for live Shading Index when CPU values are absent', async () => {
		const analysis = {
			...createGridAnalysis(),
			data: {
				...createGridAnalysis().data,
				liveShadingIndexOutput: {
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'shading-output',
					metricType: 'shading_index',
					valueLayout: 'one-f32-per-point',
					period: { kind: 'month-index', index: 7, startTimeIndex: 168, timeCount: 24 },
					outputBytes: 16
				}
			},
			metadata: {
				...createGridAnalysis().metadata,
				has_shading_index: true,
				shading_index_range: { min: 0, max: 1 }
			}
		} as any;
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});
		const readMetricPointValue = vi.fn(async () => 0.9);
		const readbackSamples: any[] = [];

		const first = await getTooltipDataAsync(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'shading_index',
			0,
			createDomRect(),
			{
				onMetricPointReadbackSample: (measurement) => readbackSamples.push(measurement),
				metricPointValueReader: {
					monthIndex: 7,
					requestId: 12,
					ownerId: 'shading-output',
					readMetricPointValue
				}
			}
		);
		const second = await getTooltipDataAsync(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'shading_index',
			0,
			createDomRect(),
			{
				onMetricPointReadbackSample: (measurement) => readbackSamples.push(measurement),
				metricPointValueReader: {
					monthIndex: 7,
					requestId: 12,
					ownerId: 'shading-output',
					readMetricPointValue
				}
			}
		);

		expect(first?.value).toBeCloseTo(0.9);
		expect(second?.value).toBeCloseTo(0.9);
		expect(readMetricPointValue).toHaveBeenCalledTimes(1);
		expect(readbackSamples).toHaveLength(2);
		expect(readbackSamples[0]).toMatchObject({
			metricType: 'shading_index',
			cacheHit: false,
			byteLength: 4,
			success: true
		});
		expect(readbackSamples[0]?.latencyMs).toEqual(expect.any(Number));
		expect(readbackSamples[0]?.latencyMs).toBeGreaterThanOrEqual(0);
		expect(readbackSamples[1]).toMatchObject({
			metricType: 'shading_index',
			cacheHit: true,
			byteLength: 0,
			success: true
		});
		expect(readbackSamples[1]?.latencyMs).toEqual(expect.any(Number));
		expect(readbackSamples[1]?.latencyMs).toBeGreaterThanOrEqual(0);
		expect(readMetricPointValue).toHaveBeenCalledWith({
			metricType: 'shading_index',
			monthIndex: 7,
			positionIndex: 3,
			requestId: 12,
			ownerId: 'shading-output'
		});
	});

	it('retries the GPU point-value path after a transient readback failure', async () => {
		const analysis = {
			...createGridAnalysis(),
			data: {
				...createGridAnalysis().data,
				liveShadingIndexOutput: {
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'shading-output',
					metricType: 'shading_index',
					valueLayout: 'one-f32-per-point',
					period: { kind: 'month-index', index: 7, startTimeIndex: 168, timeCount: 24 },
					outputBytes: 16
				}
			},
			metadata: {
				...createGridAnalysis().metadata,
				has_shading_index: true,
				shading_index_range: { min: 0, max: 1 }
			}
		} as any;
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});
		const readMetricPointValue = vi
			.fn<() => Promise<number>>()
			.mockRejectedValueOnce(new Error('transient gpu readback failure'))
			.mockResolvedValueOnce(0.91);

		const first = await getTooltipDataAsync(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'shading_index',
			0,
			createDomRect(),
			{
				metricPointValueReader: {
					monthIndex: 7,
					requestId: 12,
					ownerId: 'shading-output',
					readMetricPointValue
				}
			}
		);
		const second = await getTooltipDataAsync(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'shading_index',
			0,
			createDomRect(),
			{
				metricPointValueReader: {
					monthIndex: 7,
					requestId: 12,
					ownerId: 'shading-output',
					readMetricPointValue
				}
			}
		);
		const third = await getTooltipDataAsync(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			'shading_index',
			0,
			createDomRect(),
			{
				metricPointValueReader: {
					monthIndex: 7,
					requestId: 12,
					ownerId: 'shading-output',
					readMetricPointValue
				}
			}
		);

		expect(first).toBeNull();
		expect(second?.value).toBeCloseTo(0.91);
		expect(third?.value).toBeCloseTo(0.91);
		expect(readMetricPointValue).toHaveBeenCalledTimes(2);
	});

	it('resolves GPU-only Shading Index hits once without synthetic full-size UTCI values', async () => {
		const originalFloat32Array = globalThis.Float32Array;
		const float32Allocations: unknown[] = [];
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
		const points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 1 }));
		points.userData.utciLayout = {} as any;
		points.updateMatrixWorld(true);
		const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 100);
		camera.position.set(0, 0, 5);
		camera.lookAt(0, 0, 0);
		camera.updateProjectionMatrix();
		camera.updateMatrixWorld(true);
		const analysis = {
			data: {
				numPositions: 1,
				numHours: 0,
				positions: new originalFloat32Array([0, 0, 0]),
				liveShadingIndexOutput: {
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'shading-output',
					metricType: 'shading_index',
					valueLayout: 'one-f32-per-point',
					period: { kind: 'month-index', index: 7, startTimeIndex: 168, timeCount: 24 },
					outputBytes: 4
				}
			},
			metadata: {
				grid_size: 2,
				coordinate_system: 'xy_ground',
				has_shading_index: true,
				shading_index_range: { min: 0, max: 1 }
			}
		} as any;
		const intersectSpy = vi.spyOn(THREE.Raycaster.prototype, 'intersectObject');
		const samples: Array<Parameters<typeof recordTooltipInteractionMeasurement>[1]> = [];
		const readMetricPointValue = vi.fn(async () => 0.6);
		const Float32ArraySpy = vi.fn(function (
			input?: number | ArrayLike<number> | ArrayBufferLike
		) {
			float32Allocations.push(input);
			return new originalFloat32Array(input as any);
		});
		vi.stubGlobal('Float32Array', Float32ArraySpy);

		const result = await getTooltipDataAsync(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			points as unknown as THREE.Mesh,
			analysis,
			'shading_index',
			0,
			createDomRect(),
			{
				onDiagnosticsSample: (measurement) => samples.push(measurement),
				metricPointValueReader: {
					monthIndex: 7,
					requestId: 13,
					ownerId: 'shading-output',
					readMetricPointValue
				}
			}
		);

		expect(result).toEqual({
			value: 0.6,
			position: { x: 0, y: 0, z: 0 },
			positionIndex: 0
		});
		expect(intersectSpy).toHaveBeenCalledOnce();
		expect(samples).toHaveLength(1);
		expect(readMetricPointValue).toHaveBeenCalledOnce();
		expect(float32Allocations).not.toContain(analysis.data.numPositions);

		intersectSpy.mockRestore();
	});

	it('resolves current-surface probe coordinates without a metric value', () => {
		const analysis = {
			...createGridAnalysis(),
			data: {
				...createGridAnalysis().data,
				utciByHour: []
			}
		};
		const layout = buildUtciGridLayout(analysis);
		const mesh = createSurfaceTestMesh(layout);
		const camera = createHoverCamera({
			position: new THREE.Vector3(1, 5, 1),
			target: new THREE.Vector3(1, layout.baseY, 1)
		});

		const result = getTooltipProbeData(
			{ clientX: 50, clientY: 50 } as MouseEvent,
			camera,
			mesh,
			analysis,
			createDomRect()
		);

		expect(result).toEqual({
			position: { x: 1, y: 0, z: 1 },
			positionIndex: 3
		});
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

function createNearCornerAmbiguousCellAnalysis() {
	return {
		data: {
			numPositions: 2,
			numHours: 1,
			positions: new Float32Array([
				0, 0, 0,
				0.49, 0, 0.49
			]),
			utciValues: new Float32Array([20, 21])
		},
		metadata: {
			grid_size: 0.5,
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
