import { describe, expect, it, vi } from 'vitest';
import type { Group } from 'three';
import type { Analysis } from '$lib/types/analysis';
import type {
	SelectedHourGpuResidentOutput,
	SelectedHourLiveResult,
	SelectedHourLiveSession
} from '$lib/compute/liveUtciSelectedHourSession';
import {
	createLiveSelectedHourController,
	type LiveSelectedHourController,
	type LiveSelectedHourControllerSurfaceDiagnostics
} from '$lib/compute/liveSelectedHourController';

function decomposeTimeIndex(timeIndex: number) {
	return {
		monthIndex: Math.floor(timeIndex / 24),
		hourIndex: timeIndex % 24
	};
}

function createBaseAnalysis(label: string): Analysis {
	return {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: 2,
			hours: ['12:00'],
			utci_range: { min: 12, max: 24 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: `${label}.glb`
		},
		data: {
			numPositions: 2,
			numHours: 1,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciValues: new Float32Array([12, 24])
		}
	};
}

function createSelectionAnalysis(label: string, values: number[]): Analysis {
	return {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: values.length,
			hours: ['12:00'],
			utci_range: { min: Math.min(...values), max: Math.max(...values) },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: `${label}.glb`,
			num_months: 1
		},
		data: {
			numPositions: values.length,
			numHours: 1,
			positions: new Float32Array(values.length * 3),
			utciValues: new Float32Array(values)
		}
	};
}

function createGpuResidentOutput(
	requestId: number,
	timeIndex: number
): {
	accepted: SelectedHourGpuResidentOutput;
	destroy: ReturnType<typeof vi.fn>;
} {
	const destroy = vi.fn();
	const gpuBuffer = { destroy } as unknown as GPUBuffer;
	const { monthIndex, hourIndex } = decomposeTimeIndex(timeIndex);

	return {
		accepted: {
			requestId,
			monthIndex,
			hourIndex,
			timeIndex,
			output: { gpuBuffer } as SelectedHourGpuResidentOutput['output'],
			utciRange: { min: 10, max: 30 },
			tooltipUtciValues: new Float32Array([10, 20])
		},
		destroy
	};
}

function createLiveResult(params: {
	requestId: number;
	timeIndex: number;
	analysis: Analysis | null;
	gpuResidentOutput?: SelectedHourGpuResidentOutput | null;
	loadCpuFallback?: () => Promise<{
		analysis: Analysis;
		cpuFallbackValues: Float32Array;
	}>;
	renderTransport?: 'cpu-uploaded-selected-hour' | 'compute-buffer-selected-hour';
	sameDeviceForComputeAndRender?: boolean | null;
	pendingRenderUpdateStartedAt?: number;
}): SelectedHourLiveResult {
	const { monthIndex, hourIndex } = decomposeTimeIndex(params.timeIndex);
	return {
		requestId: params.requestId,
		monthIndex,
		hourIndex,
		timeIndex: params.timeIndex,
		analysis: params.analysis,
		gpuResidentOutput: params.gpuResidentOutput ?? null,
		cpuFallbackValues: new Float32Array([10, 20]),
		loadCpuFallback: params.loadCpuFallback,
		pendingRenderUpdateStartedAt: params.pendingRenderUpdateStartedAt ?? 123,
		renderTransport: params.renderTransport ?? 'cpu-uploaded-selected-hour',
		sameDeviceForComputeAndRender: params.sameDeviceForComputeAndRender ?? null
	};
}

function createCpuSurfaceDiagnostics(params: {
	requestId: number;
	timeIndex: number;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
}): LiveSelectedHourControllerSurfaceDiagnostics {
	const { monthIndex, hourIndex } = decomposeTimeIndex(params.timeIndex);
	return {
		utciSurfaceSource: 'cpu-uploaded-selected-hour',
		selectedHourTransferCount: params.selectedHourTransferCount,
		dataTextureBuildCount: params.dataTextureBuildCount,
		cpuPublishRequestId: params.requestId,
		cpuPublishMonthIndex: monthIndex,
		cpuPublishHourIndex: hourIndex,
		cpuPublishTimeIndex: params.timeIndex,
		cpuPublishSelectionKey: `${params.requestId}:${monthIndex}:${hourIndex}:${params.timeIndex}`
	};
}

function getCurrentSurfaceRequestId(controller: LiveSelectedHourController): number {
	const requestId = controller.getState().surfaceIdentity?.requestId;
	expect(requestId).toBeDefined();
	return requestId!;
}

function createCurrentGpuCopyDiagnostics(
	controller: LiveSelectedHourController,
	status: 'complete' | 'failed',
	error?: string
): LiveSelectedHourControllerSurfaceDiagnostics {
	return {
		gpuResidentCopyStatus: status,
		gpuResidentCopyError: error,
		gpuResidentCopyRequestId: getCurrentSurfaceRequestId(controller),
		utciSurfaceSource: status === 'complete' ? 'compute-buffer-selected-hour' : undefined
	};
}

function createCurrentCpuSurfaceDiagnostics(
	controller: LiveSelectedHourController,
	params: {
		selectedHourTransferCount?: number;
		dataTextureBuildCount?: number;
	}
): LiveSelectedHourControllerSurfaceDiagnostics {
	const state = controller.getState();
	const surfaceIdentity = state.surfaceIdentity;
	expect(surfaceIdentity).not.toBeNull();
	return {
		utciSurfaceSource: 'cpu-uploaded-selected-hour',
		selectedHourTransferCount: params.selectedHourTransferCount,
		dataTextureBuildCount: params.dataTextureBuildCount,
		cpuPublishRequestId: surfaceIdentity!.requestId,
		cpuPublishMonthIndex: surfaceIdentity!.monthIndex,
		cpuPublishHourIndex: surfaceIdentity!.hourIndex,
		cpuPublishTimeIndex: surfaceIdentity!.timeIndex,
		cpuPublishSelectionKey: surfaceIdentity!.selectionKey
	};
}

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason?: unknown) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

function createSessionMock(
	implementations: Array<(params: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		colorMode: 'normalized' | 'discrete';
		preferGpuResident: boolean;
		rendererDevice?: GPUDevice;
	}) => Promise<SelectedHourLiveResult>>
) {
	const runSelectedHour = vi.fn();
	for (const implementation of implementations) {
		runSelectedHour.mockImplementationOnce(implementation);
	}
	const dispose = vi.fn();

	return {
		session: {
			base: createBaseAnalysis('session-base'),
			numPoints: 2,
			numHours: 24,
			numMonths: 12,
			deviceSource: 'renderer',
			runSelectedHour,
			dispose
		} satisfies SelectedHourLiveSession,
		runSelectedHour,
		dispose
	};
}

function createRequestParams(timeIndex: number) {
	const { monthIndex, hourIndex } = decomposeTimeIndex(timeIndex);
	return {
		sessionKey: 'analysis-a|model-a|renderer',
		sessionConfig: {
			analysisId: 'analysis-a',
			base: createBaseAnalysis('base-analysis'),
			model: {} as Group,
			epwUrl: '/epw'
		},
		monthIndex,
		hourIndex,
		timeIndex,
		colorMode: 'discrete' as const,
		preferGpuResident: true
	};
}

function createRequestParamsForSession(sessionKey: string, timeIndex: number) {
	return {
		...createRequestParams(timeIndex),
		sessionKey,
		sessionConfig: {
			...createRequestParams(timeIndex).sessionConfig,
			analysisId: sessionKey,
			base: createBaseAnalysis(sessionKey)
		}
	};
}

describe('liveSelectedHourController', () => {
	it('reuses the session for repeated requests and rejects stale results when a newer request wins', async () => {
		const sessionReady = deferred<SelectedHourLiveSession>();
		const first = deferred<SelectedHourLiveResult>();
		const second = deferred<SelectedHourLiveResult>();
		const staleGpu = createGpuResidentOutput(1, 0);
		const latestAnalysis = createSelectionAnalysis('latest', [22, 24]);
		const sessionMock = createSessionMock([
			async () => first.promise,
			async () => second.promise
		]);
		const prepareSession = vi.fn(async () => sessionReady.promise);
		const controller = createLiveSelectedHourController({ prepareSession });

		const firstRequest = controller.requestSelection(createRequestParams(0));
		sessionReady.resolve(sessionMock.session);
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(1);
		});
		const secondRequest = controller.requestSelection(createRequestParams(1));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
		});

		second.resolve(
			createLiveResult({
				requestId: 2,
				timeIndex: 1,
				analysis: latestAnalysis,
				renderTransport: 'cpu-uploaded-selected-hour'
			})
		);
		await expect(secondRequest).resolves.toMatchObject({ accepted: true });

		first.resolve(
			createLiveResult({
				requestId: 1,
				timeIndex: 0,
				analysis: createSelectionAnalysis('stale', [12, 14]),
				gpuResidentOutput: staleGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(firstRequest).resolves.toMatchObject({ accepted: false, reason: 'stale' });

		expect(prepareSession).toHaveBeenCalledTimes(1);
		expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
		expect(staleGpu.destroy).toHaveBeenCalledTimes(1);
		expect(controller.getState().analysis).toBe(latestAnalysis);
		expect(controller.getState().renderTransport).toBe('cpu-uploaded-selected-hour');
	});

	it('disposes the previously accepted GPU buffer when a new accepted result supersedes it', async () => {
		const firstGpu = createGpuResidentOutput(7, 7);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 7,
					timeIndex: 7,
					analysis: createSelectionAnalysis('gpu-first', [19, 21]),
					gpuResidentOutput: firstGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 700
				}),
			async () =>
				createLiveResult({
					requestId: 8,
					timeIndex: 8,
					analysis: createSelectionAnalysis('cpu-next', [25, 27]),
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender: false
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(7));
		await controller.requestSelection(createRequestParams(8));

		expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
		expect(controller.getState().acceptedGpuResidentOutput).toBeNull();
		expect(controller.getState().analysis?.metadata.model_file).toBe('cpu-next.glb');
		expect(controller.getState().renderTransport).toBe('cpu-uploaded-selected-hour');
	});

	it('activates the deferred CPU fallback when render-surface diagnostics report a GPU copy failure', async () => {
		const gpu = createGpuResidentOutput(11, 11);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 11,
					timeIndex: 11,
					analysis: createSelectionAnalysis('gpu-fallback', [30, 31]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 1100
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(11));

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'failed', 'copy failed')
		);

		expect(gpu.destroy).toHaveBeenCalledTimes(1);
		expect(controller.getState()).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			awaitingGpuSurface: false,
			ready: true,
			renderReady: false
		});
		expect(controller.getState().acceptedGpuResidentOutput).toBeNull();
		expect(controller.getState().analysis?.metadata.model_file).toBe('gpu-fallback.glb');
		expect(controller.getState().renderSurfaceDiagnostics.gpuResidentCopyError).toBe('copy failed');

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 1
			})
		);

		expect(controller.getState()).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			awaitingGpuSurface: false,
			ready: true,
			renderReady: true
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 1,
			cpuPublishRequestId: getCurrentSurfaceRequestId(controller),
			cpuPublishTimeIndex: 11
		});
	});

	it('accepts current CPU publication diagnostics with idle GPU copy status after GPU fallback', async () => {
		const gpu = createGpuResidentOutput(12, 12);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 12,
					timeIndex: 12,
					analysis: createSelectionAnalysis('gpu-fallback-with-idle-status', [28, 29]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 1200
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(12));
		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'failed', 'copy failed')
		);
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 1
			}),
			gpuResidentCopyStatus: 'idle'
		});

		expect(controller.getState()).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			awaitingGpuSurface: false,
			ready: true,
			renderReady: true
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 1,
			gpuResidentCopyStatus: 'idle',
			cpuPublishRequestId: getCurrentSurfaceRequestId(controller),
			cpuPublishTimeIndex: 12
		});
		expect(controller.getState().renderSurfaceDiagnostics.gpuResidentCopyError).toBeUndefined();
		expect(
			controller.getState().renderSurfaceDiagnostics.gpuResidentCopyRequestId
		).toBeUndefined();
	});

	it('builds the deferred CPU fallback only after a GPU copy failure when the GPU result has no selected-hour analysis', async () => {
		const gpu = createGpuResidentOutput(13, 13);
		const fallbackAnalysis = createSelectionAnalysis('gpu-lazy-fallback', [32, 33]);
		const loadCpuFallback = vi.fn(async () => ({
			analysis: fallbackAnalysis,
			cpuFallbackValues: new Float32Array([32, 33])
		}));
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 13,
					timeIndex: 13,
					analysis: null,
					gpuResidentOutput: gpu.accepted,
					loadCpuFallback,
					pendingRenderUpdateStartedAt: 1300,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(13));

		expect(loadCpuFallback).not.toHaveBeenCalled();
		expect(controller.getState()).toMatchObject({
			analysis: null,
			renderTransport: 'compute-buffer-selected-hour',
			ready: true,
			renderReady: false,
			awaitingGpuSurface: true
		});

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'failed', 'copy failed')
		);

		expect(loadCpuFallback).toHaveBeenCalledTimes(1);
		expect(gpu.destroy).toHaveBeenCalledTimes(1);
		expect(controller.getState()).toMatchObject({
			analysis: fallbackAnalysis,
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			awaitingGpuSurface: false,
			ready: true,
			renderReady: false
		});
	});

	it('does not publish a stale lazy CPU fallback after a newer request starts', async () => {
		const staleGpu = createGpuResidentOutput(81, 81);
		const staleFallbackAnalysis = createSelectionAnalysis('stale-lazy-fallback', [9, 10]);
		const latestAnalysis = createSelectionAnalysis('latest-after-stale-fallback', [29, 30]);
		const loadCpuFallback = vi.fn(async () => ({
			analysis: staleFallbackAnalysis,
			cpuFallbackValues: new Float32Array([9, 10])
		}));
		const second = deferred<SelectedHourLiveResult>();
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 81,
					timeIndex: 81,
					analysis: null,
					gpuResidentOutput: staleGpu.accepted,
					loadCpuFallback,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 8100
				}),
			async () => second.promise
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(81));
		const staleRequestId = getCurrentSurfaceRequestId(controller);
		const secondRequest = controller.requestSelection(createRequestParams(82));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
		});

		await controller.handleRenderSurfaceDiagnostics({
			gpuResidentCopyStatus: 'failed',
			gpuResidentCopyError: 'old copy failed',
			gpuResidentCopyRequestId: staleRequestId
		});

		expect(loadCpuFallback).not.toHaveBeenCalled();
		expect(controller.getState()).toMatchObject({
			analysis: null,
			renderTransport: 'compute-buffer-selected-hour',
			loading: true
		});
		expect(controller.getState().acceptedGpuResidentOutput?.requestId).toBe(staleRequestId);

		second.resolve(
			createLiveResult({
				requestId: 82,
				timeIndex: 82,
				analysis: latestAnalysis,
				renderTransport: 'cpu-uploaded-selected-hour',
				sameDeviceForComputeAndRender: false
			})
		);
		await expect(secondRequest).resolves.toMatchObject({ accepted: true });
		expect(controller.getState().analysis).toBe(latestAnalysis);
		expect(staleGpu.destroy).toHaveBeenCalledTimes(1);
	});

	it('does not let a stale lazy CPU fallback rejection overwrite a newer accepted request', async () => {
		const staleGpu = createGpuResidentOutput(83, 83);
		const latestAnalysis = createSelectionAnalysis('latest-after-rejected-fallback', [31, 32]);
		const fallback = deferred<{
			analysis: Analysis;
			cpuFallbackValues: Float32Array;
		}>();
		const loadCpuFallback = vi.fn(async () => fallback.promise);
		const second = deferred<SelectedHourLiveResult>();
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 83,
					timeIndex: 83,
					analysis: null,
					gpuResidentOutput: staleGpu.accepted,
					loadCpuFallback,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 8300
				}),
			async () => second.promise
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(83));
		const fallbackHandling = controller.handleRenderSurfaceDiagnostics({
			gpuResidentCopyStatus: 'failed',
			gpuResidentCopyError: 'old copy failed',
			gpuResidentCopyRequestId: getCurrentSurfaceRequestId(controller)
		});
		await vi.waitFor(() => {
			expect(loadCpuFallback).toHaveBeenCalledTimes(1);
		});

		const secondRequest = controller.requestSelection(createRequestParams(84));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
		});
		second.resolve(
			createLiveResult({
				requestId: 84,
				timeIndex: 84,
				analysis: latestAnalysis,
				renderTransport: 'cpu-uploaded-selected-hour',
				sameDeviceForComputeAndRender: false
			})
		);
		await expect(secondRequest).resolves.toMatchObject({ accepted: true });

		fallback.reject(new Error('lazy fallback unavailable'));
		await fallbackHandling;

		expect(controller.getState()).toMatchObject({
			analysis: latestAnalysis,
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			error: null
		});
		expect(controller.getState().renderSurfaceDiagnostics).toEqual({});
		expect(staleGpu.destroy).toHaveBeenCalledTimes(1);
	});

	it('ignores late stale GPU diagnostics after fallback has moved the controller back to CPU transport', async () => {
		const gpu = createGpuResidentOutput(12, 12);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 12,
					timeIndex: 12,
					analysis: createSelectionAnalysis('gpu-fallback-stale', [28, 29]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 1200
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(12));
		const failedRequestId = getCurrentSurfaceRequestId(controller);
		await controller.handleRenderSurfaceDiagnostics({
			gpuResidentCopyStatus: 'failed',
			gpuResidentCopyError: 'copy failed',
			gpuResidentCopyRequestId: failedRequestId
		});

		const stateAfterFallback = controller.getState();
		expect(stateAfterFallback).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			acceptedGpuResidentOutput: null
		});
		expect(stateAfterFallback.renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'failed',
			gpuResidentCopyError: 'copy failed',
			gpuResidentCopyRequestId: failedRequestId
		});

		await controller.handleRenderSurfaceDiagnostics({
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: failedRequestId,
			utciSurfaceSource: 'compute-buffer-selected-hour'
		});

		expect(controller.getState()).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			acceptedGpuResidentOutput: null
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'failed',
			gpuResidentCopyError: 'copy failed',
			gpuResidentCopyRequestId: failedRequestId
		});
		expect(controller.getState().renderSurfaceDiagnostics.utciSurfaceSource).toBeUndefined();
	});

	it('keeps current GPU pending diagnostics when an empty diagnostics payload arrives', async () => {
		const gpu = createGpuResidentOutput(34, 34);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 34,
					timeIndex: 34,
					analysis: createSelectionAnalysis('gpu-empty-diagnostics', [12, 13]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3400
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(34));
		const pendingRequestId = getCurrentSurfaceRequestId(controller);
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'pending',
			gpuResidentCopyRequestId: pendingRequestId
		});

		await controller.handleRenderSurfaceDiagnostics({});

		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			loading: true,
			ready: true,
			renderReady: false,
			awaitingGpuSurface: true,
			pendingRenderUpdateStartedAt: 3400
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'pending',
			gpuResidentCopyRequestId: pendingRequestId
		});
		expect(controller.getState().renderSurfaceDiagnostics.utciSurfaceSource).toBeUndefined();
	});

	it('ignores stale CPU publication diagnostics while the current request is using GPU transport', async () => {
		const gpu = createGpuResidentOutput(35, 35);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 35,
					timeIndex: 35,
					analysis: createSelectionAnalysis('gpu-ignore-stale-cpu-publication', [14, 15]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3500
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(35));
		await controller.handleRenderSurfaceDiagnostics(
			createCpuSurfaceDiagnostics({
				requestId: 34,
				timeIndex: 34,
				selectedHourTransferCount: 7,
				dataTextureBuildCount: 3
			})
		);

		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			loading: true,
			ready: true,
			renderReady: false,
			awaitingGpuSurface: true,
			pendingRenderUpdateStartedAt: 3500
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'pending',
			gpuResidentCopyRequestId: getCurrentSurfaceRequestId(controller)
		});
		expect(controller.getState().renderSurfaceDiagnostics.utciSurfaceSource).toBeUndefined();
		expect(controller.getState().renderSurfaceDiagnostics.cpuPublishRequestId).toBeUndefined();
		expect(controller.getState().renderSurfaceDiagnostics.selectedHourTransferCount).toBeUndefined();
		expect(controller.getState().renderSurfaceDiagnostics.dataTextureBuildCount).toBeUndefined();
	});

	it('ignores stale GPU completion diagnostics from an older request while a newer GPU request is still awaiting the surface', async () => {
		const firstGpu = createGpuResidentOutput(31, 31);
		const secondGpu = createGpuResidentOutput(32, 32);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 31,
					timeIndex: 31,
					analysis: createSelectionAnalysis('gpu-first-awaiting', [11, 13]),
					gpuResidentOutput: firstGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3100
				}),
			async () =>
				createLiveResult({
					requestId: 32,
					timeIndex: 32,
					analysis: createSelectionAnalysis('gpu-second-awaiting', [21, 23]),
					gpuResidentOutput: secondGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3200
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(31));
		const staleRequestId = getCurrentSurfaceRequestId(controller);
		await controller.requestSelection(createRequestParams(32));
		const currentRequestId = getCurrentSurfaceRequestId(controller);

		await controller.handleRenderSurfaceDiagnostics({
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: staleRequestId,
			utciSurfaceSource: 'compute-buffer-selected-hour'
		});

		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			loading: true,
			ready: true,
			renderReady: false,
			awaitingGpuSurface: true,
			pendingRenderUpdateStartedAt: 3200
		});
		expect(controller.getState().acceptedGpuResidentOutput?.requestId).toBe(currentRequestId);
		expect(controller.getState().analysis?.metadata.model_file).toBe('gpu-second-awaiting.glb');
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'pending',
			gpuResidentCopyRequestId: currentRequestId
		});
	});

	it('ignores stale GPU completion and failure diagnostics when a replaced session reuses a request id', async () => {
		for (const staleStatus of ['complete', 'failed'] as const) {
			const firstGpu = createGpuResidentOutput(1, 31);
			const secondGpu = createGpuResidentOutput(1, 32);
			const firstSession = createSessionMock([
				async () =>
					createLiveResult({
						requestId: 1,
						timeIndex: 31,
						analysis: createSelectionAnalysis(`gpu-first-${staleStatus}`, [11, 13]),
						gpuResidentOutput: firstGpu.accepted,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender: true,
						pendingRenderUpdateStartedAt: 3100
					})
			]);
			const secondSession = createSessionMock([
				async () =>
					createLiveResult({
						requestId: 1,
						timeIndex: 32,
						analysis: createSelectionAnalysis(`gpu-second-${staleStatus}`, [21, 23]),
						gpuResidentOutput: secondGpu.accepted,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender: true,
						pendingRenderUpdateStartedAt: 3200
					})
			]);
			const prepareSession = vi
				.fn()
				.mockResolvedValueOnce(firstSession.session)
				.mockResolvedValueOnce(secondSession.session);
			const controller = createLiveSelectedHourController({ prepareSession });

			await controller.requestSelection(createRequestParamsForSession('session-a', 31));
			const staleRequestId = getCurrentSurfaceRequestId(controller);
			await controller.requestSelection(createRequestParamsForSession('session-b', 32));
			const currentRequestId = getCurrentSurfaceRequestId(controller);

			await controller.handleRenderSurfaceDiagnostics({
				gpuResidentCopyStatus: staleStatus,
				gpuResidentCopyError: staleStatus === 'failed' ? 'stale copy failed' : undefined,
				gpuResidentCopyRequestId: staleRequestId,
				utciSurfaceSource:
					staleStatus === 'complete' ? 'compute-buffer-selected-hour' : undefined
			});

			expect(controller.getState()).toMatchObject({
				renderTransport: 'compute-buffer-selected-hour',
				loading: true,
				ready: true,
				renderReady: false,
				awaitingGpuSurface: true,
				pendingRenderUpdateStartedAt: 3200
			});
			expect(controller.getState().acceptedGpuResidentOutput).toMatchObject({
				...secondGpu.accepted,
				requestId: currentRequestId
			});
			expect(controller.getState().analysis?.metadata.model_file).toBe(
				`gpu-second-${staleStatus}.glb`
			);
			expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
				gpuResidentCopyStatus: 'pending',
				gpuResidentCopyRequestId: currentRequestId
			});
			expect(secondGpu.destroy).not.toHaveBeenCalled();

			controller.dispose();
		}
	});

	it('preserves cpu-uploaded surface diagnostics when no GPU request id is active', async () => {
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 33,
					timeIndex: 33,
					analysis: createSelectionAnalysis('cpu-surface-diagnostics', [14, 16]),
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender: false
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(33));
		await controller.handleRenderSurfaceDiagnostics(
			createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 2
			})
		);

		expect(controller.getState()).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			ready: true,
			renderReady: true
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 2,
			cpuPublishRequestId: getCurrentSurfaceRequestId(controller),
			cpuPublishTimeIndex: 33
		});
	});

	it('treats repeated current CPU publication diagnostics as idempotent', async () => {
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 36,
					timeIndex: 36,
					analysis: createSelectionAnalysis('cpu-idempotent-diagnostics', [16, 18]),
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender: false
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(36));
		const currentCpuDiagnostics = createCurrentCpuSurfaceDiagnostics(controller, {
			selectedHourTransferCount: 2,
			dataTextureBuildCount: 1
		});
		await controller.handleRenderSurfaceDiagnostics(currentCpuDiagnostics);
		expect(controller.getState()).toMatchObject({
			renderTransport: 'cpu-uploaded-selected-hour',
			renderReady: true
		});

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({ ...currentCpuDiagnostics });

		unsubscribe();
		expect(emittedStates).toHaveLength(0);
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 2,
			dataTextureBuildCount: 1,
			cpuPublishRequestId: getCurrentSurfaceRequestId(controller),
			cpuPublishTimeIndex: 36
		});
		expect(controller.getState().renderReady).toBe(true);
	});

	it('treats repeated current pending GPU diagnostics as idempotent', async () => {
		const gpu = createGpuResidentOutput(37, 37);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 37,
					timeIndex: 37,
					analysis: createSelectionAnalysis('gpu-idempotent-pending-diagnostics', [17, 19]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3700
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(37));
		const pendingDiagnostics = {
			gpuResidentCopyStatus: 'pending' as const,
			gpuResidentCopyRequestId: getCurrentSurfaceRequestId(controller)
		};
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject(pendingDiagnostics);

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({ ...pendingDiagnostics });

		unsubscribe();
		expect(emittedStates).toHaveLength(0);
		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			loading: true,
			ready: true,
			renderReady: false,
			awaitingGpuSurface: true,
			pendingRenderUpdateStartedAt: 3700
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject(pendingDiagnostics);
	});

	it('does not accept compute-buffer transport without same-device proof', async () => {
		const gpu = createGpuResidentOutput(61, 13);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 61,
					timeIndex: 13,
					analysis: createSelectionAnalysis('gpu-without-same-device-proof', [15, 17]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: false,
					pendingRenderUpdateStartedAt: 6100
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(13));
		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'complete')
		);

		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: false,
			loading: true,
			renderReady: false,
			awaitingGpuSurface: true,
			pendingRenderUpdateStartedAt: 6100
		});
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			gpuResidentCopyStatus: 'pending',
			gpuResidentCopyRequestId: getCurrentSurfaceRequestId(controller)
		});
		expect(controller.getState().renderSurfaceDiagnostics.utciSurfaceSource).toBeUndefined();
	});

	it('ignores stale cpu publication from a previous request after a newer cpu request wins', async () => {
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 71,
					timeIndex: 14,
					analysis: createSelectionAnalysis('cpu-first-publication', [18, 20]),
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender: false
				}),
			async () =>
				createLiveResult({
					requestId: 72,
					timeIndex: 15,
					analysis: createSelectionAnalysis('cpu-second-publication', [28, 30]),
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender: false
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(14));
		const firstCpuDiagnostics = createCurrentCpuSurfaceDiagnostics(controller, {
			selectedHourTransferCount: 1
		});
		await controller.handleRenderSurfaceDiagnostics(
			firstCpuDiagnostics
		);
		expect(controller.getState().renderSurfaceDiagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 1,
			cpuPublishRequestId: firstCpuDiagnostics.cpuPublishRequestId,
			cpuPublishTimeIndex: 14
		});

		await controller.requestSelection(createRequestParams(15));
		expect(controller.getState().analysis?.metadata.model_file).toBe('cpu-second-publication.glb');
		expect(controller.getState().renderTransport).toBe('cpu-uploaded-selected-hour');
		expect(controller.getState().renderSurfaceDiagnostics).toEqual({});

		await controller.handleRenderSurfaceDiagnostics(
			firstCpuDiagnostics
		);

		expect(controller.getState().analysis?.metadata.model_file).toBe('cpu-second-publication.glb');
		expect(controller.getState().renderTransport).toBe('cpu-uploaded-selected-hour');
		expect(controller.getState().renderSurfaceDiagnostics).toEqual({});
	});

	it('emits atomic state updates for GPU acceptance and fallback transitions', async () => {
		const gpu = createGpuResidentOutput(41, 41);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 41,
					timeIndex: 41,
					analysis: createSelectionAnalysis('gpu-atomic', [17, 19]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 4100
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});
		const seenStates: Array<{
			acceptedRequestId: number | null;
			analysisFile: string | null;
			renderTransport: string;
			loading: boolean;
			pendingRenderUpdateStartedAt: number | undefined;
		}> = [];
		controller.subscribe((state) => {
			seenStates.push({
				acceptedRequestId: state.acceptedGpuResidentOutput?.requestId ?? null,
				analysisFile: state.analysis?.metadata.model_file ?? null,
				renderTransport: state.renderTransport,
				loading: state.loading,
				pendingRenderUpdateStartedAt: state.pendingRenderUpdateStartedAt
			});
		});

		await controller.requestSelection(createRequestParams(41));

		expect(
			seenStates.some(
				(state) =>
					state.acceptedRequestId === 41 &&
					(state.analysisFile == null ||
						state.renderTransport !== 'compute-buffer-selected-hour' ||
						state.pendingRenderUpdateStartedAt === undefined)
			)
		).toBe(false);

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'failed', 'copy failed')
		);

		expect(
			seenStates.some(
				(state) =>
					state.acceptedRequestId == null &&
					state.renderTransport === 'compute-buffer-selected-hour'
			)
		).toBe(false);
		expect(controller.getState()).toMatchObject({
			acceptedGpuResidentOutput: null,
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			renderReady: false
		});
		expect(controller.getState().analysis?.metadata.model_file).toBe('gpu-atomic.glb');
		expect(gpu.destroy).toHaveBeenCalledTimes(1);

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 1
			})
		);

		expect(controller.getState()).toMatchObject({
			acceptedGpuResidentOutput: null,
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			renderReady: true
		});
	});

	it('tracks route-neutral loading and ready state through GPU completion diagnostics and disposes accepted output on controller disposal', async () => {
		const gpu = createGpuResidentOutput(21, 21);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 21,
					timeIndex: 21,
					analysis: null,
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 2100
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});
		const seenStates: Array<{
			loading: boolean;
			ready: boolean;
			renderReady: boolean;
			awaitingGpuSurface: boolean;
		}> = [];
		const unsubscribe = controller.subscribe((state) => {
			seenStates.push({
				loading: state.loading,
				ready: state.ready,
				renderReady: state.renderReady,
				awaitingGpuSurface: state.awaitingGpuSurface
			});
		});

		const requestPromise = controller.requestSelection(createRequestParams(21));
		expect(controller.getState()).toMatchObject({
			loading: true,
			ready: false,
			renderReady: false,
			awaitingGpuSurface: false
		});

		await requestPromise;
		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			loading: true,
			ready: true,
			renderReady: false,
			awaitingGpuSurface: true,
			sameDeviceForComputeAndRender: true,
			pendingRenderUpdateStartedAt: 2100
		});

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'complete')
		);
		expect(controller.getState()).toMatchObject({
			analysis: null,
			renderTransport: 'compute-buffer-selected-hour',
			loading: false,
			ready: true,
			renderReady: true,
			awaitingGpuSurface: false
		});
		expect(
			seenStates.some((state) => state.loading && !state.ready && !state.renderReady)
		).toBe(true);
		expect(
			seenStates.some((state) => state.loading && state.ready && state.awaitingGpuSurface)
		).toBe(true);

		unsubscribe();
		controller.dispose();

		expect(gpu.destroy).toHaveBeenCalledTimes(1);
		expect(sessionMock.dispose).toHaveBeenCalledTimes(1);
		expect(controller.getState()).toMatchObject({
			analysis: null,
			acceptedGpuResidentOutput: null,
			loading: false,
			renderTransport: 'idle',
			ready: false,
			renderReady: false
		});
	});

	it('does not emit a render-ready while still loading intermediate state when the accepted GPU copy completes', async () => {
		const gpu = createGpuResidentOutput(51, 51);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 51,
					timeIndex: 51,
					analysis: createSelectionAnalysis('gpu-complete-atomic', [18, 20]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 5100
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});
		const seenStates: Array<{
			loading: boolean;
			renderReady: boolean;
			awaitingGpuSurface: boolean;
			pendingRenderUpdateStartedAt: number | undefined;
			gpuResidentCopyStatus: string | undefined;
		}> = [];
		controller.subscribe((state) => {
			seenStates.push({
				loading: state.loading,
				renderReady: state.renderReady,
				awaitingGpuSurface: state.awaitingGpuSurface,
				pendingRenderUpdateStartedAt: state.pendingRenderUpdateStartedAt,
				gpuResidentCopyStatus: state.renderSurfaceDiagnostics.gpuResidentCopyStatus
			});
		});

		await controller.requestSelection(createRequestParams(51));
		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'complete')
		);

		expect(
			seenStates.some(
				(state) =>
					state.gpuResidentCopyStatus === 'complete' &&
					state.renderReady &&
					state.loading
			)
		).toBe(false);
		expect(
			seenStates.some(
				(state) =>
					state.gpuResidentCopyStatus === 'complete' &&
					state.awaitingGpuSurface === false &&
					state.pendingRenderUpdateStartedAt !== undefined
			)
		).toBe(false);
		expect(controller.getState()).toMatchObject({
			loading: false,
			renderReady: true,
			awaitingGpuSurface: false,
			pendingRenderUpdateStartedAt: undefined
		});
	});
});
