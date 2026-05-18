import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Group } from 'three';
import type { Analysis } from '$lib/types/analysis';
import {
	disposeSelectedHourGpuResidentOutput,
	prepareSelectedHourLiveSession
} from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import { createSelectedHourOutputHandle } from '$lib/compute/gpu/selectedHourOutputHandle';
import {
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker
} from '$lib/compute/gpu/mergeAndBvhWorkerClient';

const mockState = vi.hoisted(() => ({
	pipeline: null as any,
	rendererDevice: {} as GPUDevice,
	gpuBuffer: null as any,
	outputOverride: null as any,
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

describe('selected-hour live session', () => {
	beforeEach(() => {
		vi.restoreAllMocks();
		mockState.rendererDevice = {} as GPUDevice;
		mockState.gpuBuffer = { destroy: vi.fn() } as unknown as GPUBuffer;
		mockState.outputOverride = null;
		mockState.initResult = null;
		mockState.runtimeDiagnostics = {
			timings: {
				exposurePrecomputeMs: 12.5,
				oneHourDispatchMs: 3.75
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
			model: {} as Group,
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

	it('attaches live selected-hour values for same-device GPU-resident range and tooltip data', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: {} as Group,
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
	});

	it('records one selected-hour range scan and reuses it for discrete GPU-resident range', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: {} as Group,
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
			sessionSelectedHourRangeScanStartedAtMs: expect.any(Number),
			sessionSelectedHourRangeScanCompletedAtMs: expect.any(Number),
			sessionGpuResidentRangeResolveStartedAtMs: expect.any(Number),
			sessionGpuResidentRangeResolveCompletedAtMs: expect.any(Number)
		});
		expect(
			(timeline?.sessionGpuResidentRangeResolveCompletedAtMs ?? 0) -
				(timeline?.sessionGpuResidentRangeResolveStartedAtMs ?? 0)
		).toBeGreaterThanOrEqual(0);
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
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
			model: {} as Group,
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
			model: {} as Group,
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
				model: {} as Group,
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
			model: {} as Group,
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
			model: {} as Group,
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
			model: {} as Group,
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
			model: {} as Group,
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
			model: {} as Group,
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
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createFullDayBaseAnalysis(),
			model: {} as Group,
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
			model: {} as Group,
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
			model: {} as Group,
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
			model: {} as Group,
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
