import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Group } from 'three';
import type { Analysis } from '$lib/types/analysis';
import { prepareSelectedHourLiveSession } from '$lib/compute/liveUtciSelectedHourSession';

const mockState = vi.hoisted(() => ({
	pipeline: null as any,
	rendererDevice: {} as GPUDevice,
	gpuBuffer: null as any,
	constructors: [] as any[]
}));

vi.mock('$lib/compute/mergeAndBvhWorkerClient', () => ({
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

vi.mock('$lib/compute/webgpuUtciPipeline', () => ({
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

		initFromModelAndWeather = vi.fn(async () => ({ numPoints: 2 }));
		runExposurePrecompute = vi.fn(async () => undefined);
		runUtciForTimeIndex = vi.fn(async () => ({ gpuBuffer: mockState.gpuBuffer }));
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

	it('attaches live selected-hour values for same-device GPU-resident range and tooltip data', async () => {
		const session = await prepareSelectedHourLiveSession({
			analysisId: 'analysis-a',
			base: createBaseAnalysis(),
			model: {} as Group,
			epwUrl: '/weather.epw',
			signal: new AbortController().signal,
			preferredDevice: mockState.rendererDevice
		});

		await expect(
			session.runSelectedHour({
				monthIndex: 0,
				hourIndex: 12,
				timeIndex: 12,
				colorMode: 'discrete',
				preferGpuResident: true,
				rendererDevice: mockState.rendererDevice
			})
		).resolves.toMatchObject({
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
		expect(mockState.pipeline.readOnDemandUtciForDebug).toHaveBeenCalledTimes(1);
	});

	it('uses the selected month WebGPU day range for normalized GPU-resident coloring', async () => {
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

		expect(first.gpuResidentOutput?.utciRange).toEqual({ min: 0, max: 120 });
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
});
