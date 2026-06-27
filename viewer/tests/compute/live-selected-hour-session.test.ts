import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import {
	SURFACE_FLAGS,
	type Analysis,
	type ClassifiedAnalysisActiveMask
} from '$lib/types/analysis';
import {
	disposeSelectedHourGpuResidentOutput,
	prepareSelectedHourLiveSession
} from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import { createSelectedHourOutputHandle } from '$lib/compute/gpu/selectedHourOutputHandle';
import {
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker
} from '$lib/compute/gpu/mergeAndBvhWorkerClient';
import { updateViewerConfig } from '$lib/config/viewerConfig';

const mockState = vi.hoisted(() => ({
	pipeline: null as any,
	rendererDevice: {} as GPUDevice,
	gpuBuffer: null as any,
	shadingBuffer: null as any,
	outputOverride: null as any,
	shadingOutputOverride: null as any,
	initResult: null as any,
	runtimeDiagnostics: null as any,
	constructors: [] as any[]
}));

vi.mock('$lib/compute/gpu/mergeAndBvhWorkerClient', () => ({
	MAX_GRID_POINTS_GUARD: 100000,
	prepareMeshPayloadForWorkerAsync: vi.fn(async () => ({
		meshes: [],
		totalTriangles: 0,
		preflight: {
			estimatedGridPoints: 2,
			estimatedBytes: 128
		}
	})),
	runMergeAndBvhInWorker: vi.fn(async () => ({
		gridPoints: new Float32Array([0, 0, 0, 1, 0, 0]),
		serializedBvh: {
			nodes: new Float32Array(),
			triangles: new Float32Array(),
			triangleCount: 0
		}
	}))
}));

vi.mock('$lib/compute/gpu/webgpuUtciPipeline', () => ({
	createWebgpuUtciPipeline: vi.fn(async () => mockState.pipeline)
}));

vi.mock('$lib/compute/telemetry', () => ({
	emitComputeTelemetry: vi.fn()
}));

vi.mock('$lib/compute/compute-manager', () => ({
	ComputeManager: class {
		pipeline: any;

		constructor(pipeline: any) {
			this.pipeline = pipeline;
			mockState.constructors.push(this);
		}

		initFromModelAndWeather = vi.fn(async () => mockState.initResult ?? {
			numPoints: 2,
			gridPoints: new Float32Array([0, 0.9, 0, 1, 0.9, 0])
		});
		runExposurePrecompute = vi.fn(async () => undefined);
		runUtciForTimeIndex = vi.fn(async () => mockState.outputOverride ?? { gpuBuffer: mockState.gpuBuffer });
		runShadingIndex = vi.fn(async (params) =>
			mockState.shadingOutputOverride ?? this.pipeline.runShadingIndex(params)
		);
		runUtciRangeSummaryForTimeIndex = vi.fn(async (params) =>
			this.pipeline.runUtciRangeSummaryForTimeIndex(params)
		);
		runUtciRangeSummaryForOutput = vi.fn(async (params) =>
			this.pipeline.runUtciRangeSummaryForOutput(params)
		);
		getOnDemandDiagnostics = vi.fn(() => mockState.runtimeDiagnostics);
		getDeviceForDebug = vi.fn(() => mockState.rendererDevice);
	}
}));

function createBaseAnalysis(): Analysis {
	return {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: 2,
			hours: ['12:00'],
			utci_range: { min: 10, max: 30 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'base.glb',
			bounds: {
				x_min: 0,
				x_max: 1,
				y_min: 0,
				y_max: 1,
				z: 0
			}
		},
		data: {
			numPositions: 2,
			numHours: 1,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciValues: new Float32Array([10, 30])
		}
	};
}

function createFullDayBaseAnalysis(): Analysis {
	return {
		...createBaseAnalysis(),
		metadata: {
			...createBaseAnalysis().metadata,
			analysis_type: 'full_day',
			hours: Array.from({ length: 24 }, (_, hour) => `${hour.toString().padStart(2, '0')}:00`)
		},
		data: {
			...createBaseAnalysis().data,
			numHours: 24,
			utciByHour: Array.from({ length: 24 }, () => new Float32Array([10, 30]))
		}
	};
}

function expectClassifiedActiveMask(
	activeMask: Analysis['metadata']['activeMask'] | undefined
): asserts activeMask is ClassifiedAnalysisActiveMask {
	expect(activeMask).toBeDefined();
	expect(activeMask).toHaveProperty('surfaceFlagsByActiveCell');
}

function createActiveMaskBaseAnalysis(): Analysis {
	return {
		...createBaseAnalysis(),
		metadata: {
			...createBaseAnalysis().metadata,
			num_positions: 9,
			grid_size: 1,
			coordinate_system: 'xz_ground',
			bounds: {
				x_min: 0,
				x_max: 2,
				y_min: 0,
				y_max: 2,
				z: 0
			}
		},
		data: {
			numPositions: 9,
			numHours: 1,
			positions: new Float32Array(9 * 3),
			utciValues: new Float32Array(9)
		}
	};
}

function createLayerTriangleMesh(params: {
	layerType: string;
	layerName: string;
	points: [number, number, number][];
}): THREE.Mesh {
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute(
		'position',
		new THREE.Float32BufferAttribute(params.points.flat(), 3)
	);
	const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
	mesh.userData.layerType = params.layerType;
	mesh.userData.layerName = params.layerName;
	mesh.updateMatrixWorld(true);
	return mesh;
}

function createActiveMaskModel(): THREE.Group {
	const model = new THREE.Group();
	model.add(
		createLayerTriangleMesh({
			layerType: 'base',
			layerName: 'ground',
			points: [
				[0, 0, 0],
				[1, 0, 0],
				[0, 0, 1]
			]
		})
	);
	model.add(
		createLayerTriangleMesh({
			layerType: 'road',
			layerName: 'street',
			points: [
				[2, 0, 2],
				[2, 0, 0],
				[0, 0, 2]
			]
		})
	);
	model.updateMatrixWorld(true);
	return model;
}

function createBaseWithNonSamplingLayersActiveMaskModel(): THREE.Group {
	const model = new THREE.Group();
	model.add(
		createLayerTriangleMesh({
			layerType: 'base',
			layerName: 'ground',
			points: [
				[0, 0, 0],
				[1, 0, 0],
				[0, 0, 1]
			]
		})
	);
	model.add(
		createLayerTriangleMesh({
			layerType: 'building',
			layerName: 'existing_buildings',
			points: [
				[2, 0, 2],
				[2, 0, 0],
				[0, 0, 2]
			]
		})
	);
	model.add(
		createLayerTriangleMesh({
			layerType: 'vegetation',
			layerName: 'trees_canopy',
			points: [
				[2, 0, 2],
				[2, 0, 0],
				[0, 0, 2]
			]
		})
	);
	model.add(
		createLayerTriangleMesh({
			layerType: 'rail',
			layerName: 'train_tracks',
			points: [
				[2, 0, 2],
				[2, 0, 0],
				[0, 0, 2]
			]
		})
	);
	model.updateMatrixWorld(true);
	return model;
}

function createSidewalkBuildingOverlapActiveMaskModel(): THREE.Group {
	const model = new THREE.Group();
	model.add(
		createLayerTriangleMesh({
			layerType: 'base',
			layerName: 'ground',
			points: [
				[0, 0, 0],
				[2, 0, 0],
				[0, 0, 2],
				[2, 0, 0],
				[2, 0, 2],
				[0, 0, 2]
			]
		})
	);
	model.add(
		createLayerTriangleMesh({
			layerType: 'surface',
			layerName: 'sidewalks',
			points: [
				[0, 0, 0],
				[1, 0, 0],
				[0, 0, 1]
			]
		})
	);
	model.add(
		createLayerTriangleMesh({
			layerType: 'building',
			layerName: 'existing_buildings',
			points: [
				[0, 0, 0],
				[1, 0, 0],
				[0, 0, 1]
			]
		})
	);
	model.updateMatrixWorld(true);
	return model;
}

function createExcludedOnlyActiveMaskModel(layerType: string, layerName: string): THREE.Group {
	const model = new THREE.Group();
	model.add(
		createLayerTriangleMesh({
			layerType,
			layerName,
			points: [
				[0, 0, 0],
				[2, 0, 0],
				[0, 0, 2]
			]
		})
	);
	model.updateMatrixWorld(true);
	return model;
}

function createOutOfBoundsSampledActiveMaskModel(): THREE.Group {
	const model = new THREE.Group();
	model.add(
		createLayerTriangleMesh({
			layerType: 'base',
			layerName: 'ground',
			points: [
				[10, 0, 10],
				[12, 0, 10],
				[10, 0, 12]
			]
		})
	);
	model.updateMatrixWorld(true);
	return model;
}

function createNormalizationTranslatedActiveMaskModel(): THREE.Group {
	const model = createActiveMaskModel();
	model.position.set(-1, -1, 0);
	model.updateMatrixWorld(true);
	return model;
}

function createEmptyModel(): THREE.Group {
	return new THREE.Group();
}

describe('selected-hour live session', () => {
	beforeEach(() => {
		vi.restoreAllMocks();
		updateViewerConfig({ anchorOffset: new THREE.Vector3(0, 0, 0), enableNormalization: true });
		mockState.rendererDevice = {} as GPUDevice;
		mockState.gpuBuffer = { destroy: vi.fn() } as unknown as GPUBuffer;
		mockState.shadingBuffer = { destroy: vi.fn() } as unknown as GPUBuffer;
		mockState.outputOverride = null;
		mockState.shadingOutputOverride = null;
		mockState.initResult = null;
		mockState.runtimeDiagnostics = {
			timings: {
				exposurePrecomputeMs: 12.5,
				oneHourDispatchMs: 3.75,
				shadingIndexDispatchMs: 4.5,
				shadingIndexQueueWaitMs: 1.25,
				shadingIndexOutputBytes: 8,
				shadingIndexSnapshotBytes: 8
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 128,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 8,
				selectedHourOutputBytesHighWatermark: 8,
				renderOwnedSelectedHourBytes: 0,
				renderOwnedSelectedHourBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		};
		mockState.constructors.length = 0;
		mockState.pipeline = {
			uploadStaticData: vi.fn(async () => undefined),
			runAll: vi.fn(async () => undefined),
			readUtcisSlice: vi.fn(async () => new Float32Array([10, 30])),
			readOnDemandUtciForDebug: vi.fn(async () => new Float32Array([11, 29])),
			runUtciRangeSummaryForTimeIndex: vi.fn(async (params: { timeIndex: number }) => ({
				timeIndex: params.timeIndex,
				range: { min: 10, max: 30 },
				validCount: 2,
				readbackBytes: 16,
				reductionPassCount: 1,
				debugLabel: 'webgpu-on-demand-f32-utci-range-summary' as const
			})),
			runUtciRangeSummaryForOutput: vi.fn(async (params: { timeIndex: number }) => ({
				timeIndex: params.timeIndex,
				range: { min: 11, max: 29 },
				validCount: 2,
				readbackBytes: 16,
				reductionPassCount: 1,
				debugLabel: 'webgpu-on-demand-f32-utci-range-summary' as const
			})),
			runShadingIndex: vi.fn(async (params: { monthIndex: number; startTimeIndex: number; timeCount: number }) => {
				const ownerId = `webgpu-shading-index:${params.monthIndex}:${params.startTimeIndex}:${params.timeCount}:test`;
				const period = {
					kind: 'month-index' as const,
					index: params.monthIndex,
					startTimeIndex: params.startTimeIndex,
					timeCount: params.timeCount
				};
				const gpuOutputHandle = createSelectedHourOutputHandle({
					buffer: mockState.shadingBuffer,
					byteLength: 8,
					source: 'webgpu-on-demand-snapshot',
					ownerId,
					metricType: 'shading_index',
					valueLayout: 'one-f32-per-point',
					period
				});
				return {
					source: 'webgpu-on-demand-snapshot' as const,
					ownerId,
					metricType: 'shading_index' as const,
					valueLayout: 'one-f32-per-point' as const,
					period,
					numPoints: 2,
					gpuBuffer: mockState.shadingBuffer,
					gpuOutputHandle,
					outputBytes: 8,
					debugLabel: 'webgpu-shading-index'
				};
			}),
			getDeviceForDebug: vi.fn(() => mockState.rendererDevice),
			dispose: vi.fn()
		};
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => ({
				ok: true,
				text: async () => 'EPW'
			}))
		);
		vi.stubGlobal('Worker', vi.fn());
	});

	it('propagates live session lifecycle timings onto selected-hour runtime diagnostics', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(result.diagnostics.timings).toMatchObject({
			exposurePrecomputeMs: 12.5,
			oneHourDispatchMs: 3.75
		});
		expect(result.diagnostics.timings.payloadPrepareMs).toEqual(expect.any(Number));
		expect(result.diagnostics.timings.workerBvhMs).toEqual(expect.any(Number));
		expect(result.diagnostics.timings.pipelineUploadMs).toEqual(expect.any(Number));
		expect(result.diagnostics.timings.firstSelectedHourReadyMs).toEqual(expect.any(Number));
		expect(result.diagnostics.timings.firstSelectedHourReadyMs ?? -1).toBeGreaterThanOrEqual(0);
	});

	it('includes street geometry in the active mask and reports active-count diagnostics', async () => {
		updateViewerConfig({ enableNormalization: false });
		mockState.initResult = {
			numPoints: 9,
			canonicalPointCount: 9,
			gridPoints: new Float32Array([
				0, 0.9, 0,
				0, 0.9, 1,
				0, 0.9, 2,
				1, 0.9, 0,
				1, 0.9, 1,
				1, 0.9, 2,
				2, 0.9, 0,
				2, 0.9, 1,
				2, 0.9, 2
			])
		};
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(
			async () => new Float32Array([11, 13, 15, 17, 19, 21, 23, 25, 29])
		);
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createActiveMaskBaseAnalysis(),
			model: createActiveMaskModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 0,
			timeIndex: 0,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(mockState.constructors[0].initFromModelAndWeather).toHaveBeenCalledWith(
			expect.objectContaining({
				activeCanonicalIndices: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8])
			})
		);
		expect(mockState.constructors[1].runExposurePrecompute).toHaveBeenCalledWith(
			expect.objectContaining({ numPoints: 9 })
		);
		expect(mockState.constructors[1].runUtciForTimeIndex).toHaveBeenCalledWith(
			expect.objectContaining({ numPoints: 9 })
		);
		expect(result.diagnostics).toMatchObject({
			activeMaskSource: 'base+road',
			canonicalPointCount: 9,
			activePointCount: 9,
			inactivePointCount: 0,
			activePointRatio: 1,
			activeMaskChecksum: expect.stringMatching(/^[0-9a-f]{8}$/)
		});
		expect(result.analysis?.metadata.activeMask).toMatchObject({
			source: 'base+road',
			canonicalPointCount: 9,
			activePointCount: 9,
			activeCanonicalIndices: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8])
		});
		const activeMask = result.analysis?.metadata.activeMask;
		expectClassifiedActiveMask(activeMask);
		const surfaceFlags = activeMask.surfaceFlagsByActiveCell;
		expect(surfaceFlags).toBeInstanceOf(Uint8Array);
		expect(surfaceFlags).toHaveLength(9);
		expect(Array.from(surfaceFlags).some((flags) => (flags & SURFACE_FLAGS.streetSurface) !== 0)).toBe(true);
		expect(
			Array.from(surfaceFlags).every(
				(flags) => (flags & (SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface)) !== 0
			)
		).toBe(true);
		expect(result.gpuResidentOutput?.tooltipUtciValues).toHaveLength(9);
	});

	it('does not expand the active mask from building, vegetation, or train-track geometry', async () => {
		updateViewerConfig({ enableNormalization: false });
		mockState.initResult = {
			numPoints: 3,
			canonicalPointCount: 9,
			gridPoints: new Float32Array([
				0, 0.9, 0,
				0, 0.9, 1,
				1, 0.9, 0
			])
		};
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(async () => new Float32Array([11, 21, 29]));
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createActiveMaskBaseAnalysis(),
			model: createBaseWithNonSamplingLayersActiveMaskModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 0,
			timeIndex: 0,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(mockState.constructors[0].initFromModelAndWeather).toHaveBeenCalledWith(
			expect.objectContaining({
				activeCanonicalIndices: new Uint32Array([0, 1, 3])
			})
		);
		expect(result.diagnostics).toMatchObject({
			activeMaskSource: 'base+road',
			canonicalPointCount: 9,
			activePointCount: 3,
			inactivePointCount: 6,
			activePointRatio: 3 / 9,
			activeMaskChecksum: expect.stringMatching(/^[0-9a-f]{8}$/)
		});
		const activeMask = result.analysis?.metadata.activeMask;
		expectClassifiedActiveMask(activeMask);
		const surfaceFlags = Array.from(activeMask.surfaceFlagsByActiveCell);
		expect(surfaceFlags).toHaveLength(3);
		expect(
			surfaceFlags.every(
				(flags) => (flags & (SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface)) !== 0
			)
		).toBe(true);
	});

	it('preserves sidewalk-family and building-footprint overlap flags on active rows', async () => {
		updateViewerConfig({ enableNormalization: false });
		mockState.initResult = {
			numPoints: 9,
			canonicalPointCount: 9,
			gridPoints: new Float32Array([
				0, 0.9, 0,
				0, 0.9, 1,
				0, 0.9, 2,
				1, 0.9, 0,
				1, 0.9, 1,
				1, 0.9, 2,
				2, 0.9, 0,
				2, 0.9, 1,
				2, 0.9, 2
			])
		};
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(
			async () => new Float32Array([11, 13, 15, 17, 19, 21, 23, 25, 29])
		);

		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createActiveMaskBaseAnalysis(),
			model: createSidewalkBuildingOverlapActiveMaskModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 0,
			timeIndex: 0,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		const activeMask = result.analysis?.metadata.activeMask;
		expectClassifiedActiveMask(activeMask);
		const surfaceFlags = Array.from(activeMask.surfaceFlagsByActiveCell);
		expect(surfaceFlags).toHaveLength(9);
		expect(surfaceFlags[0]).toBe(
			SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface | SURFACE_FLAGS.buildingFootprint
		);
		expect(surfaceFlags[1]).toBe(
			SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface | SURFACE_FLAGS.buildingFootprint
		);
		const includeInPublicRealmStats = surfaceFlags.map((flags) => {
			return (
				(flags & SURFACE_FLAGS.streetSurface) !== 0 &&
				(flags & SURFACE_FLAGS.buildingFootprint) === 0
			);
		});
		expect(includeInPublicRealmStats[0]).toBe(false);
		expect(includeInPublicRealmStats[1]).toBe(false);
	});

	it('passes an explicit empty active mask when sampled surfaces produce zero active cells', async () => {
		updateViewerConfig({ enableNormalization: false });
		mockState.initResult = {
			numPoints: 0,
			canonicalPointCount: 9,
			gridPoints: new Float32Array()
		};

		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createActiveMaskBaseAnalysis(),
			model: createOutOfBoundsSampledActiveMaskModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		expect(mockState.constructors[0].initFromModelAndWeather).toHaveBeenCalledWith(
			expect.objectContaining({
				activeCanonicalIndices: new Uint32Array()
			})
		);
		expect(session.base.metadata.activeMask).toMatchObject({
			source: 'base+road',
			canonicalPointCount: 9,
			activePointCount: 0,
			inactivePointCount: 9,
			activePointRatio: 0,
			activeCanonicalIndices: new Uint32Array()
		});
		const activeMask = session.base.metadata.activeMask;
		expectClassifiedActiveMask(activeMask);
		expect(activeMask.surfaceFlagsByActiveCell).toEqual(new Uint8Array());
	});

	it.each([
		['building-only', 'building', 'existing_buildings'],
		['train-track-only', 'rail', 'train_tracks']
	])('rejects %s active-mask classification instead of falling through to the full grid', async (_label, layerType, layerName) => {
		updateViewerConfig({ enableNormalization: false });

		await expect(
			prepareSelectedHourLiveSession({
				analysisId: 'Innovation-District/innovation_district_webgpu',
				base: createActiveMaskBaseAnalysis(),
				model: createExcludedOnlyActiveMaskModel(layerType, layerName),
				epwUrl: '/weather.epw',
				signal: new AbortController().signal,
				preferredDevice: mockState.rendererDevice
			})
		).rejects.toThrow(/sampled ground\/street surface/i);
		expect(mockState.constructors).toHaveLength(0);
	});

	it('keeps rectangular analyses on the full grid when model geometry is not surface-classified', async () => {
		updateViewerConfig({ enableNormalization: false });

		const session = await prepareSelectedHourLiveSession({
			analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
			base: createBaseAnalysis(),
			model: createExcludedOnlyActiveMaskModel('building', 'existing_buildings'),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		expect(mockState.constructors[0].initFromModelAndWeather).toHaveBeenCalledWith(
			expect.objectContaining({
				activeCanonicalIndices: undefined
			})
		);
		expect(session.base.metadata.activeMask).toBeUndefined();
	});

	it('extracts the normalized base and street footprint in canonical metadata coordinates', async () => {
		mockState.initResult = {
			numPoints: 9,
			canonicalPointCount: 9,
			gridPoints: new Float32Array([
				-1, -0.1, 0,
				-1, -0.1, 1,
				-1, -0.1, 2,
				0, -0.1, 0,
				0, -0.1, 1,
				0, -0.1, 2,
				1, -0.1, 0,
				1, -0.1, 1,
				1, -0.1, 2
			])
		};
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(
			async () => new Float32Array([11, 13, 15, 17, 19, 21, 23, 25, 29])
		);

		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createActiveMaskBaseAnalysis(),
			model: createNormalizationTranslatedActiveMaskModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 0,
			timeIndex: 0,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(mockState.constructors[0].initFromModelAndWeather).toHaveBeenCalledWith(
			expect.objectContaining({
				gridOriginOffset: { x: -1, y: -1, z: 0 },
				activeCanonicalIndices: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8])
			})
		);
		expect(result.diagnostics).toMatchObject({
			activeMaskSource: 'base+road',
			canonicalPointCount: 9,
			activePointCount: 9,
			inactivePointCount: 0,
			activeMaskChecksum: expect.stringMatching(/^[0-9a-f]{8}$/)
		});
		expect(result.analysis?.metadata.activeMask?.activeCanonicalIndices).toEqual(
			new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8])
		);
	});

	it('passes exposure scheduler options and session abort signal to exposure precompute', async () => {
		const abortController = new AbortController();
		const exposureScheduling = {
			mode: 'chunked' as const,
			maxWorkgroupsPerSlice: 8192,
			yieldBetweenSlices: true
		};
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: abortController.signal,
			preferredDevice: mockState.rendererDevice,
			exposureScheduling
		});

		await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(mockState.constructors[1].runExposurePrecompute).toHaveBeenCalledWith({
			numPoints: 2,
			numHours: 1,
			numMonths: 12,
			exposureScheduling,
			diagnosticsEnabled: false,
			signal: abortController.signal
		});
	});

	it('attaches live selected-hour values for same-device GPU-resident range and tooltip data', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(result).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			analysis: {
				metadata: {
					analysis_type: 'single_hour',
					utci_range: { min: 11, max: 29 }
				}
			},
			gpuResidentOutput: {
				output: { gpuBuffer: mockState.gpuBuffer },
				utciRange: { min: 11, max: 29 },
				tooltipUtciValues: new Float32Array([11, 29])
			}
		});
		expect(result.loadCpuFallback).toEqual(expect.any(Function));
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
		expect(mockState.pipeline.runUtciRangeSummaryForOutput).toHaveBeenCalledTimes(1);
	});

	it('preserves explicit UTCI metric behavior on the GPU-resident path', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			metricType: 'utci',
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(result.analysis?.metadata.has_shading_index).not.toBe(true);
		expect(result.gpuResidentOutput?.metricType).toBe('utci');
		expect(mockState.constructors[1].runUtciForTimeIndex).toHaveBeenCalledTimes(1);
		expect(mockState.constructors[1].runShadingIndex).not.toHaveBeenCalled();
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
		expect(mockState.pipeline.runUtciRangeSummaryForOutput).toHaveBeenCalledTimes(1);
	});

	it('publishes Shading Index as a same-device GPU-resident metric output', async () => {
		const destroy = vi.fn();
		mockState.shadingBuffer = { destroy } as unknown as GPUBuffer;
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			numMonths: 12
		});

		const result = await session.runSelectedHour({
			metricType: 'shading_index',
			monthIndex: 1,
			hourIndex: 3,
			timeIndex: 27,
			colorMode: 'normalized',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});
		const timeline = result.diagnostics.timings.renderPublication?.renderPublicationTimeline;

		expect(result).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			analysis: {
				metadata: {
					analysis_type: 'single_hour',
					has_shading_index: true,
					shading_index_range: { min: 0, max: 1 }
				}
			}
		});
		expect(result.gpuResidentOutput).toMatchObject({
			metricType: 'shading_index',
			utciRange: { min: 0, max: 1 },
			shadingIndexRange: { min: 0, max: 1 },
			output: {
				gpuBuffer: mockState.shadingBuffer,
				metricType: 'shading_index'
			}
		});
		expect(result.gpuResidentOutput?.gpuOutputHandle?.disposed).toBe(false);
		expect((result.analysis?.data as any).liveShadingIndexOutput).toMatchObject({
			source: 'webgpu-on-demand-snapshot',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			outputBytes: 8
		});
		expect((result.analysis?.data as any).liveShadingIndexOutput.gpuOutputHandle).toBe(
			result.gpuResidentOutput?.gpuOutputHandle
		);
		expect(mockState.constructors[1].runExposurePrecompute).toHaveBeenCalledTimes(1);
		expect(mockState.constructors[1].runShadingIndex).toHaveBeenCalledWith({
			numPoints: 2,
			numHours: 24,
			numMonths: 12,
			monthIndex: 1,
			startTimeIndex: 24,
			timeCount: 24,
			signal: expect.any(AbortSignal)
		});
		expect(mockState.constructors[1].runUtciForTimeIndex).not.toHaveBeenCalled();
		expect(mockState.pipeline.readOnDemandUtciForDebug).not.toHaveBeenCalled();
		expect(mockState.pipeline.runUtciRangeSummaryForOutput).not.toHaveBeenCalled();
		expect(mockState.pipeline.runUtciRangeSummaryForTimeIndex).not.toHaveBeenCalled();
		expect(destroy).not.toHaveBeenCalled();
		expect(timeline).toMatchObject({
			sessionMetricType: 'shading_index',
			sessionOutputBytes: 8,
			sessionCompactSummaryBytes: 0,
			sessionShadingIndexDispatchMs: 4.5,
			sessionShadingIndexQueueWaitMs: 1.25,
			sessionShadingIndexOutputBytes: 8,
			sessionShadingIndexSnapshotBytes: 8,
			sessionShadingIndexSource: 'fresh-dispatch',
			sessionShadingIndexMonthCacheHit: false,
			sessionFullSolarReadbackCount: 0,
			sessionTooltipPointReadbackCount: 0,
			sessionTooltipPointReadbackBytes: 0
		});
		expect(result.diagnostics.timings).toMatchObject({
			shadingIndexDispatchMs: 4.5,
			shadingIndexQueueWaitMs: 1.25,
			shadingIndexOutputBytes: 8,
			shadingIndexSnapshotBytes: 8
		});
		expect(result.diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(8);
		expect(timeline?.sessionSelectedHourRangeResolutionPath).toBeUndefined();
		expect(timeline?.sessionSelectedDayRangeResolutionPath).toBeUndefined();
		expect(result.diagnostics.selectedHourReadbackReasons ?? []).not.toContain('tooltip');
		expect(result.diagnostics.selectedHourReadbackReasons ?? []).not.toContain('range');

		disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);
		expect(destroy).toHaveBeenCalledTimes(1);
	});

	it('uses fixed Shading Index range for shading publication when base UTCI metadata range is invalid', async () => {
		const base = createFullDayBaseAnalysis();
		base.metadata.utci_range = { min: Number.NaN, max: Number.NaN };
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base,
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			numMonths: 12
		});

		const result = await session.runSelectedHour({
			metricType: 'shading_index',
			monthIndex: 1,
			hourIndex: 3,
			timeIndex: 27,
			colorMode: 'normalized',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(result.renderTransport).toBe('compute-buffer-selected-hour');
		expect(result.gpuResidentOutput?.metricType).toBe('shading_index');
		expect(result.gpuResidentOutput?.utciRange).toEqual({ min: 0, max: 1 });
		expect(result.gpuResidentOutput?.shadingIndexRange).toEqual({ min: 0, max: 1 });
		expect(result.analysis?.metadata.shading_index_range).toEqual({ min: 0, max: 1 });

		disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);
	});

	it('disposes shading GPU output when aborted after shading output creation', async () => {
		const abort = new AbortController();
		const destroy = vi.fn();
		mockState.shadingBuffer = { destroy } as unknown as GPUBuffer;
		const originalRunShadingIndex = mockState.pipeline.runShadingIndex;
		mockState.pipeline.runShadingIndex = vi.fn(async (params) => {
			const output = await originalRunShadingIndex(params);
			abort.abort();
			return output;
		});
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: abort.signal,
			preferredDevice: mockState.rendererDevice,
			numMonths: 12
		});

		await expect(
			session.runSelectedHour({
				metricType: 'shading_index',
				monthIndex: 1,
				hourIndex: 3,
				timeIndex: 27,
				colorMode: 'normalized',
				preferGpuResident: true,
				rendererDevice: mockState.rendererDevice
			})
		).rejects.toThrow(/aborted/i);

		expect(destroy).toHaveBeenCalledTimes(1);
	});

	it('disposes shading GPU output when publication errors after shading output creation', async () => {
		const destroy = vi.fn();
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 8,
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-shading-index:test',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'month-index', index: 1, startTimeIndex: 24, timeCount: 24 }
		});
		mockState.shadingOutputOverride = {
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-shading-index:mismatch',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'month-index', index: 1, startTimeIndex: 24, timeCount: 24 },
			numPoints: 2,
			gpuBuffer: handle.buffer,
			gpuOutputHandle: handle,
			outputBytes: 8,
			debugLabel: 'webgpu-shading-index'
		};
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			numMonths: 12
		});

		await expect(
			session.runSelectedHour({
				metricType: 'shading_index',
				monthIndex: 1,
				hourIndex: 3,
				timeIndex: 27,
				colorMode: 'normalized',
				preferGpuResident: true,
				rendererDevice: mockState.rendererDevice
			})
		).rejects.toThrow(/authoritative shading index output/i);

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(handle.disposed).toBe(true);
	});

	it('uses compact selected-hour output summary for discrete GPU-resident range', async () => {
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(async () => new Float32Array([100, 120]));
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});
		const timeline = result.diagnostics.timings.renderPublication?.renderPublicationTimeline;

		expect(result.analysis?.metadata.utci_range).toEqual({ min: 11, max: 29 });
		expect(result.gpuResidentOutput?.utciRange).toEqual({ min: 11, max: 29 });
		expect(result.gpuResidentOutput?.tooltipUtciValues).toEqual(new Float32Array([100, 120]));
		expect(timeline).toMatchObject({
			sessionSelectedHourRangeScanStartedAtMs: expect.any(Number),
			sessionSelectedHourRangeScanCompletedAtMs: expect.any(Number),
			sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
			sessionSelectedHourRangeReadbackCount: 0,
			sessionSelectedHourRangeCpuScanCount: 0,
			sessionSelectedHourRangeSummaryReadbackCount: 1,
			sessionSelectedHourRangeSummaryReadbackBytes: 16,
			sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
			sessionSelectedHourRangeSummaryReductionPassCount: 1,
			sessionGpuResidentRangeResolveStartedAtMs: expect.any(Number),
			sessionGpuResidentRangeResolveCompletedAtMs: expect.any(Number)
		});
		expect(result.diagnostics.timings).toMatchObject({
			selectedHourRangeSummaryMs: expect.any(Number),
			selectedHourRangeSummaryReadbackBytes: 16,
			selectedHourRangeSummaryReadbackCount: 1,
			selectedHourRangeSummaryReductionPassCount: 1,
			selectedHourRangeFullReadbackAvoidedCount: 1
		});
		expect(
			(timeline?.sessionGpuResidentRangeResolveCompletedAtMs ?? 0) -
				(timeline?.sessionGpuResidentRangeResolveStartedAtMs ?? 0)
		).toBeGreaterThanOrEqual(0);
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
		expect(mockState.pipeline.runUtciRangeSummaryForOutput).toHaveBeenCalledWith(
			expect.objectContaining({
				timeIndex: 12,
				numPoints: 2,
				format: 'f32-utci',
				output: expect.objectContaining({
					gpuOutputHandle: expect.objectContaining({
						source: 'webgpu-on-demand-snapshot'
					})
				})
			})
		);
	});

	it('falls back to scanning existing tooltip values when compact selected-hour summary is unavailable', async () => {
		mockState.pipeline.runUtciRangeSummaryForOutput = undefined;
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});
		const timeline = result.diagnostics.timings.renderPublication?.renderPublicationTimeline;

		expect(result.analysis?.metadata.utci_range).toEqual({ min: 11, max: 29 });
		expect(result.gpuResidentOutput?.utciRange).toEqual({ min: 11, max: 29 });
		expect(timeline).toMatchObject({
			sessionSelectedHourRangeResolutionPath: 'cpu-scan-existing-values',
			sessionSelectedHourRangeReadbackCount: 0,
			sessionSelectedHourRangeCpuScanCount: 1,
			sessionSelectedHourRangeSummaryReadbackCount: 0,
			sessionSelectedHourRangeSummaryReadbackBytes: 0,
			sessionSelectedHourRangeFullReadbackAvoidedCount: 0,
			sessionSelectedHourRangeSummaryReductionPassCount: 0
		});
		expect(result.diagnostics.selectedHourReadbackReasons).toEqual(['tooltip']);
		expect(result.diagnostics.selectedHourReadbackReasons ?? []).not.toContain('range');
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
	});

	it('marks selected-hour range unavailable when compact summary and tooltip values are unavailable', async () => {
		mockState.pipeline.runUtciRangeSummaryForOutput = undefined;
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(async () => undefined);
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});
		const timeline = result.diagnostics.timings.renderPublication?.renderPublicationTimeline;

		expect(result.analysis).toBeNull();
		expect(result.gpuResidentOutput?.utciRange).toEqual({ min: -20, max: 60 });
		expect(timeline).toMatchObject({
			sessionSelectedHourRangeResolutionPath: 'unavailable',
			sessionSelectedHourRangeReadbackCount: 0,
			sessionSelectedHourRangeCpuScanCount: 0,
			sessionSelectedHourRangeSummaryReadbackCount: 0,
			sessionSelectedHourRangeSummaryReadbackBytes: 0,
			sessionSelectedHourRangeFullReadbackAvoidedCount: 0,
			sessionSelectedHourRangeSummaryReductionPassCount: 0
		});
		expect(result.diagnostics.selectedHourReadbackReasons ?? []).not.toContain('range');
	});

	it('records selected-day range cache diagnostics for cold and warm normalized months', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const cold = await session.runSelectedHour({
			monthIndex: 1,
			hourIndex: 3,
			timeIndex: 27,
			colorMode: 'normalized',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
		const warm = await session.runSelectedHour({
			monthIndex: 1,
			hourIndex: 4,
			timeIndex: 28,
			colorMode: 'normalized',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(cold.diagnostics.timings.renderPublication?.renderPublicationTimeline).toMatchObject({
			sessionSelectedDayRangeCacheKey: '1:24',
			sessionSelectedDayRangeCacheHit: false,
			sessionSelectedDayRangeCacheSizeBefore: 0,
			sessionSelectedDayRangeCacheSizeAfter: 1,
			sessionSelectedDayRangeReadbackCount: 0,
			sessionSelectedDayRangeComputedHourCount: 23,
			sessionSelectedDayRangeResolutionPath: 'compact-gpu-summary',
			sessionSelectedDayRangeSummaryReadbackCount: 23,
			sessionSelectedDayRangeSummaryReadbackBytes: 23 * 16,
			sessionSelectedDayRangeFullReadbackAvoidedCount: 23
		});
		expect(warm.diagnostics.timings.renderPublication?.renderPublicationTimeline).toMatchObject({
			sessionSelectedDayRangeCacheKey: '1:24',
			sessionSelectedDayRangeCacheHit: true,
			sessionSelectedDayRangeCacheSizeBefore: 1,
			sessionSelectedDayRangeCacheSizeAfter: 1,
			sessionSelectedDayRangeReadbackCount: 0,
			sessionSelectedDayRangeComputedHourCount: 0,
			sessionSelectedDayRangeResolutionPath: 'cache-hit',
			sessionSelectedDayRangeSummaryReadbackCount: 0,
			sessionSelectedDayRangeSummaryReadbackBytes: 0
		});
		expect(mockState.pipeline.runUtciRangeSummaryForTimeIndex).toHaveBeenCalledTimes(23);
	});

	it('uses an explicitly requested 0.5m grid instead of clamping to base 2m metadata', async () => {
		vi.mocked(prepareMeshPayloadForWorkerAsync).mockClear();
		mockState.initResult = {
			numPoints: 3,
			gridPoints: new Float32Array([10, 1.8, 20, 11, 1.8, 20, 12, 1.8, 20])
		};

		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			gridResolution: 0.5
		});

		expect(prepareMeshPayloadForWorkerAsync).toHaveBeenCalledWith(
			expect.anything(),
			expect.objectContaining({ gridResolution: 0.5 })
		);
		expect(mockState.constructors[0].initFromModelAndWeather).toHaveBeenCalledWith(
			expect.objectContaining({ gridResolution: 0.5 })
		);
		expect(session.base.metadata.grid_size).toBe(0.5);
		expect(session.base.metadata.num_positions).toBe(3);
		expect(session.base.data.numPositions).toBe(3);
		expect(session.base.data.positions).toHaveLength(9);
		expect(session.base.data.positions).not.toEqual(createFullDayBaseAnalysis().data.positions);
	});

	it('does not apply the default 600k grid cap to selected-hour 0.5m preflight or BVH generation', async () => {
		const densePointCount = 1_896_487;
		vi.mocked(prepareMeshPayloadForWorkerAsync).mockImplementationOnce(
			async (_model, options) => {
				const liveOptions = options as
					| { gridResolution?: number; maxGridPoints?: number; maxEstimatedBytes?: number }
					| undefined;
				if ((liveOptions?.maxGridPoints ?? 100_000) <= densePointCount) {
					throw new Error(
						'Estimated grid too dense (1,896,487 points) exceeds safety cap (600,000). Increase grid size.'
					);
				}
				if ((liveOptions?.maxEstimatedBytes ?? 0) < Number.MAX_SAFE_INTEGER) {
					throw new Error('Selected-hour dense preflight still uses all-hours byte budget');
				}
				return {
					meshes: [],
					totalTriangles: 0,
					preflight: {
						estimatedGridPoints: densePointCount,
						estimatedBytes: 0,
						totalTriangles: 0,
						meshCount: 0,
						bounds: { min: [0, 0, 0], max: [1, 1, 1] }
					}
				};
			}
		);

		await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			gridResolution: 0.5
		});

		expect(prepareMeshPayloadForWorkerAsync).toHaveBeenCalledWith(
			expect.anything(),
			expect.objectContaining({
				gridResolution: 0.5,
				maxGridPoints: expect.any(Number),
				maxEstimatedBytes: Number.POSITIVE_INFINITY
			})
		);
		expect(
			(
				vi.mocked(prepareMeshPayloadForWorkerAsync).mock.calls[0]?.[1] as
					| { maxGridPoints?: number }
					| undefined
			)?.maxGridPoints
		).toBeGreaterThan(densePointCount);
		expect(runMergeAndBvhInWorker).toHaveBeenCalledWith(
			expect.objectContaining({
				gridResolution: 0.5,
				maxGridPoints: expect.any(Number),
				bvhOnly: true
			})
		);
		expect(
			(
				vi.mocked(runMergeAndBvhInWorker).mock.calls[0]?.[0] as
					| { maxGridPoints?: number }
					| undefined
			)?.maxGridPoints
		).toBeGreaterThan(densePointCount);
	});

	it.each([
		['missing', undefined],
		['wrong-length', new Float32Array([10, 1.8, 20])]
	])('rejects %s generated grid points instead of publishing stale base positions', async (_label, gridPoints) => {
		mockState.initResult = {
			numPoints: 3,
			gridPoints
		};

		await expect(
			prepareSelectedHourLiveSession({
				analysisId: 'analysis-a',
				base: createFullDayBaseAnalysis(),
				model: createEmptyModel(),
				epwUrl: '/weather.epw',
				signal: new AbortController().signal,
				preferredDevice: mockState.rendererDevice,
				gridResolution: 0.5
			})
		).rejects.toThrow(
			'Live selected-hour generated grid point length mismatch'
		);
	});

	it('falls back only to coarser grid resolutions when preflight exceeds budget', async () => {
		vi.mocked(prepareMeshPayloadForWorkerAsync).mockClear();
		vi.mocked(prepareMeshPayloadForWorkerAsync).mockImplementation(
			async (_model, options) => {
				const gridResolution = options?.gridResolution ?? 0;
				if ([4, 6, 8].includes(gridResolution)) {
					throw new Error('grid preflight exceeds budget');
				}
				return {
					meshes: [],
					totalTriangles: 0,
					preflight: {
						estimatedGridPoints: 2,
						estimatedBytes: 128,
						totalTriangles: 0,
						meshCount: 0,
						bounds: { min: [0, 0, 0], max: [1, 1, 1] }
					}
				};
			}
		);

		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			gridResolution: 4
		});

		expect(
			vi.mocked(prepareMeshPayloadForWorkerAsync).mock.calls.map(
				([, options]) => options?.gridResolution
			)
		).toEqual([4, 6, 8, 10]);
		expect(session.base.metadata.grid_size).toBe(10);
	});

	it('keeps an accepted GPU output handle alive until explicit selected-hour disposal', async () => {
		const destroy = vi.fn();
		mockState.gpuBuffer = { destroy } as unknown as GPUBuffer;
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});
		const acceptedOutput = result.gpuResidentOutput;

		expect(acceptedOutput?.gpuOutputHandle).toMatchObject({
			buffer: mockState.gpuBuffer,
			byteLength: 8,
			requestId: 1,
			timeIndex: 12,
			source: 'webgpu-on-demand-snapshot',
			disposed: false
		});
		expect(acceptedOutput?.output.gpuOutputHandle).toBe(acceptedOutput?.gpuOutputHandle);
		expect(destroy).not.toHaveBeenCalled();

		disposeSelectedHourGpuResidentOutput(acceptedOutput);

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(acceptedOutput?.gpuOutputHandle?.disposed).toBe(true);
	});

	it('attaches session request identity to an existing GPU output handle and disposes through it', async () => {
		const destroy = vi.fn();
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 8,
			source: 'webgpu-on-demand-snapshot'
		});
		mockState.outputOverride = {
			format: 'f32-utci',
			numPoints: 2,
			timeIndex: 12,
			gpuOutputHandle: handle,
			outputBytes: 8
		};
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(result.gpuResidentOutput?.gpuOutputHandle).toBe(handle);
		expect(handle.requestId).toBe(1);
		expect(handle.timeIndex).toBe(12);
		expect(result.gpuResidentOutput?.output.gpuBuffer).toBe(handle.buffer);
		expect(destroy).not.toHaveBeenCalled();

		disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(handle.disposed).toBe(true);
	});

	it('normalizes mismatched gpuBuffer compatibility output to the canonical handle buffer', async () => {
		const canonicalDestroy = vi.fn();
		const legacyDestroy = vi.fn();
		const legacyBuffer = { destroy: legacyDestroy } as unknown as GPUBuffer;
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: canonicalDestroy } as unknown as GPUBuffer,
			byteLength: 8,
			source: 'webgpu-on-demand-snapshot'
		});
		mockState.outputOverride = {
			format: 'f32-utci',
			numPoints: 2,
			timeIndex: 12,
			gpuBuffer: legacyBuffer,
			gpuOutputHandle: handle,
			outputBytes: 8
		};
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(legacyDestroy).toHaveBeenCalledTimes(1);
		expect(result.gpuResidentOutput?.output.gpuBuffer).toBe(handle.buffer);
		expect(result.gpuResidentOutput?.gpuOutputHandle).toBe(handle);
		expect(canonicalDestroy).not.toHaveBeenCalled();

		disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);

		expect(canonicalDestroy).toHaveBeenCalledTimes(1);
		expect(legacyDestroy).toHaveBeenCalledTimes(1);
		expect(handle.disposed).toBe(true);
	});

	it('disposes a superseded GPU output handle only after the next output is recorded', async () => {
		const firstDestroy = vi.fn();
		const secondDestroy = vi.fn();
		mockState.gpuBuffer = { destroy: firstDestroy } as unknown as GPUBuffer;
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const first = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		mockState.gpuBuffer = { destroy: secondDestroy } as unknown as GPUBuffer;
		const second = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 13,
			timeIndex: 13,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(first.gpuResidentOutput?.gpuOutputHandle?.disposed).toBe(false);
		expect(second.gpuResidentOutput?.gpuOutputHandle?.disposed).toBe(false);
		expect(firstDestroy).not.toHaveBeenCalled();

		disposeSelectedHourGpuResidentOutput(first.gpuResidentOutput);

		expect(firstDestroy).toHaveBeenCalledTimes(1);
		expect(first.gpuResidentOutput?.gpuOutputHandle?.disposed).toBe(true);
		expect(secondDestroy).not.toHaveBeenCalled();
	});

	it('uses selected-day range for normalized analysis and GPU-resident range', async () => {
		const readbacks = [
			new Float32Array([100, 120]),
			...Array.from({ length: 24 }, (_, hour) => new Float32Array([hour, hour + 20])).filter(
				(_, hour) => hour !== 12
			)
		];
		mockState.pipeline.readOnDemandUtciForDebug = vi.fn(async () => {
			return readbacks.shift() ?? new Float32Array([0, 1]);
		});
		mockState.pipeline.runUtciRangeSummaryForTimeIndex = undefined;
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice,
			numMonths: 12
		});

		const first = await session.runSelectedHour({
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 7 * 24 + 12,
			colorMode: 'normalized',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(first.analysis?.metadata.utci_range).toEqual({ min: 0, max: 120 });
		expect(first.gpuResidentOutput?.utciRange).toEqual({ min: 0, max: 120 });
		expect(first.diagnostics.selectedHourReadbackReasons).toContain('range');
		expect(first.diagnostics.selectedHourReadbackReasonCounts?.range).toBeGreaterThan(0);
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(24);

		await session.runSelectedHour({
			monthIndex: 7,
			hourIndex: 13,
			timeIndex: 7 * 24 + 13,
			colorMode: 'normalized',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(25);
	});

	it('records comparison selected-hour CPU readbacks separately from visible render transport', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice,
			selectedHourReadbackReason: 'comparison'
		});

		expect(result.renderTransport).toBe('compute-buffer-selected-hour');
		expect(result.diagnostics.selectedHourReadbackReasons).toContain('comparison');
		expect(result.diagnostics.selectedHourReadbackReasonCounts?.comparison).toBeGreaterThan(0);
		expect(result.diagnostics.timings.renderPublication?.renderPublicationTimeline).toMatchObject({
			sessionSelectedHourAnalysisBuildStartedAtMs: expect.any(Number),
			sessionSelectedHourAnalysisBuildCompletedAtMs: expect.any(Number),
			sessionCpuFallbackSetupStartedAtMs: expect.any(Number),
			sessionCpuFallbackSetupCompletedAtMs: expect.any(Number),
			sessionGpuResidentRangeResolveStartedAtMs: expect.any(Number),
			sessionGpuResidentRangeResolveCompletedAtMs: expect.any(Number),
			sessionTooltipValuesHandoffStartedAtMs: expect.any(Number),
			sessionTooltipValuesHandoffCompletedAtMs: expect.any(Number),
			sessionGpuResidentResultAssemblyStartedAtMs: expect.any(Number),
			sessionGpuResidentResultAssemblyCompletedAtMs: expect.any(Number),
			sessionResultReadyAtMs: expect.any(Number)
		});
	});

	it('records visible fallback CPU readbacks when GPU-resident rendering is unavailable', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: false,
			rendererDevice: mockState.rendererDevice
		});

		expect(result.renderTransport).toBe('cpu-uploaded-selected-hour');
		expect(result.diagnostics.timings.renderPublication?.renderPublicationPath).toBe(
			'cpu-uploaded-selected-hour'
		);
		expect(result.diagnostics.selectedHourReadbackReasons).toContain('visible-fallback');
		expect(result.diagnostics.selectedHourReadbackReasonCounts?.['visible-fallback']).toBeGreaterThan(0);
	});

	it('keeps deferred visible fallback readback reasons observable on returned diagnostics', async () => {
		const deferredFallbackValues = new Float32Array([13, 27]);
		mockState.pipeline.readOnDemandUtciForDebug = vi
			.fn()
			.mockResolvedValueOnce(undefined)
			.mockResolvedValueOnce(deferredFallbackValues);
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: createEmptyModel(),
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		const result = await session.runSelectedHour({
			monthIndex: 0,
			hourIndex: 12,
			timeIndex: 12,
			colorMode: 'discrete',
			preferGpuResident: true,
			rendererDevice: mockState.rendererDevice
		});

		expect(result.renderTransport).toBe('compute-buffer-selected-hour');
		expect(result.diagnostics.selectedHourReadbackReasons ?? []).not.toContain('visible-fallback');

		const fallback = await result.loadCpuFallback?.();

		expect(fallback?.cpuFallbackValues).toBe(deferredFallbackValues);
		expect(result.diagnostics.selectedHourReadbackReasons).toContain('visible-fallback');
		expect(result.diagnostics.selectedHourReadbackReasonCounts?.['visible-fallback']).toBeGreaterThan(0);
	});
});

