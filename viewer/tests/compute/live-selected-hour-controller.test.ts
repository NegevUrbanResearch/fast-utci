import { describe, expect, it, vi } from 'vitest';
import type { Group } from 'three';
import type { Analysis } from '$lib/types/analysis';
import type {
	SelectedHourGpuResidentOutput,
	SelectedHourLiveResult,
	SelectedHourLiveSession
} from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import type { SelectedHourOutputHandle } from '$lib/compute/gpu/selectedHourOutputHandle';
import {
	createLiveSelectedHourController,
	type LiveSelectedHourController,
	type LiveSelectedHourControllerSurfaceDiagnostics
} from '$lib/compute/selected-hour/liveSelectedHourController';
import { createEmptyOnDemandDiagnostics } from '$lib/compute/on-demand/onDemandDiagnostics';
import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';
import { createRenderPublicationDiagnostics } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

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
	diagnostics?: SelectedHourLiveResult['diagnostics'];
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
		sameDeviceForComputeAndRender: params.sameDeviceForComputeAndRender ?? null,
		diagnostics: params.diagnostics ?? createEmptyOnDemandDiagnostics()
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
		metricType: 'utci' | 'shading_index';
		colorMode: 'normalized' | 'discrete';
		preferGpuResident: boolean;
		rendererDevice?: GPUDevice;
		selectedHourReadbackReason?: SelectedHourReadbackReason;
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
		metricType: 'utci' as const,
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
		expect(controller.getState().acceptedVisibleSurface).toBeNull();
		expect(controller.getState().acceptedRequestId).toBeUndefined();
		expect(controller.getState().acceptedSelectionKey).toBeUndefined();
		expect(controller.getState().acceptedVisibleAtMs).toBeUndefined();
		await controller.handleRenderSurfaceDiagnostics(
			createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 1
			})
		);
		const acceptedVisibleAtMs = controller.getState().acceptedVisibleAtMs;

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
		expect(controller.getState().acceptedRequestId).toBe(2);
		expect(controller.getState().acceptedSelectionKey).toBe('2:0:1:1');
		expect(controller.getState().acceptedVisibleAtMs).toBe(acceptedVisibleAtMs);
		expect(controller.getState().acceptedVisibleSurface).toEqual({
			requestId: 2,
			selectionKey: '2:0:1:1',
			visibleAtMs: acceptedVisibleAtMs
		});
	});

	it('retires a superseded accepted GPU output until the scene releases it', async () => {
		const firstGpu = createGpuResidentOutput(7, 7);
		const secondGpu = createGpuResidentOutput(8, 8);
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
					analysis: createSelectionAnalysis('gpu-second', [25, 27]),
					gpuResidentOutput: secondGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 800
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(7));
		const firstAcceptedRequestId = controller.getState().acceptedGpuResidentOutput?.requestId;
		expect(firstAcceptedRequestId).toBeDefined();
		await controller.requestSelection(createRequestParams(8));

		expect(firstGpu.destroy).not.toHaveBeenCalled();
		expect(secondGpu.destroy).not.toHaveBeenCalled();

		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: firstAcceptedRequestId!,
			monthIndex: 0,
			timeIndex: 7,
			reason: 'superseded'
		});

		expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
		expect(secondGpu.destroy).not.toHaveBeenCalled();
		expect(controller.getState().acceptedGpuResidentOutput?.requestId).toBe(2);
	});

	it('marks the current accepted GPU output releasable and disposes it after replacement', async () => {
		const firstGpu = createGpuResidentOutput(11, 11);
		const secondGpu = createGpuResidentOutput(12, 12);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 11,
					timeIndex: 11,
					analysis: createSelectionAnalysis('gpu-first', [19, 21]),
					gpuResidentOutput: firstGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				}),
			async () =>
				createLiveResult({
					requestId: 12,
					timeIndex: 12,
					analysis: createSelectionAnalysis('gpu-second', [25, 27]),
					gpuResidentOutput: secondGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(11));
		const firstAcceptedRequestId = controller.getState().acceptedGpuResidentOutput?.requestId;
		expect(firstAcceptedRequestId).toBeDefined();
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: firstAcceptedRequestId!,
			monthIndex: 0,
			timeIndex: 11,
			reason: 'copy-complete'
		});

		expect(firstGpu.destroy).not.toHaveBeenCalled();

		await controller.requestSelection(createRequestParams(12));

		expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
		expect(secondGpu.destroy).not.toHaveBeenCalled();
	});

	it('does not let a reused GPU output object inherit releasable state across controller requests', async () => {
		const destroy = vi.fn();
		const sharedOutput = {
			gpuBuffer: { destroy } as unknown as GPUBuffer
		} as SelectedHourGpuResidentOutput['output'];
		const thirdGpu = createGpuResidentOutput(23, 23);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 21,
					timeIndex: 21,
					analysis: createSelectionAnalysis('gpu-first-reused-output', [19, 21]),
					gpuResidentOutput: {
						...createGpuResidentOutput(21, 21).accepted,
						output: sharedOutput
					},
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				}),
			async () =>
				createLiveResult({
					requestId: 22,
					timeIndex: 22,
					analysis: createSelectionAnalysis('gpu-second-reused-output', [25, 27]),
					gpuResidentOutput: {
						...createGpuResidentOutput(22, 22).accepted,
						output: sharedOutput
					},
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				}),
			async () =>
				createLiveResult({
					requestId: 23,
					timeIndex: 23,
					analysis: createSelectionAnalysis('gpu-third-replacement', [29, 31]),
					gpuResidentOutput: thirdGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(21));
		const firstAcceptedRequestId = controller.getState().acceptedGpuResidentOutput?.requestId;
		expect(firstAcceptedRequestId).toBeDefined();
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: firstAcceptedRequestId!,
			monthIndex: 0,
			timeIndex: 21,
			reason: 'copy-complete'
		});

		await controller.requestSelection(createRequestParams(22));
		expect(destroy).not.toHaveBeenCalled();
		const secondAcceptedRequestId = controller.getState().acceptedGpuResidentOutput?.requestId;
		expect(secondAcceptedRequestId).toBeDefined();
		expect(secondAcceptedRequestId).not.toBe(firstAcceptedRequestId);

		await controller.requestSelection(createRequestParams(23));

		expect(destroy).not.toHaveBeenCalled();
		expect(thirdGpu.destroy).not.toHaveBeenCalled();

		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: secondAcceptedRequestId!,
			monthIndex: 0,
			timeIndex: 22,
			reason: 'superseded'
		});

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(thirdGpu.destroy).not.toHaveBeenCalled();
	});

	it('does not destroy a stale rejected GPU output when it shares the current accepted output handle', async () => {
		const destroy = vi.fn();
		const sharedHandle: SelectedHourOutputHandle = {
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 4,
			requestId: 31,
			timeIndex: 31,
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-on-demand-snapshot',
			metricType: 'utci',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'time-index', index: 31 },
			disposed: false,
			dispose() {
				if (sharedHandle.disposed) return;
				sharedHandle.disposed = true;
				destroy();
			}
		};
		const currentOutput = {
			gpuOutputHandle: sharedHandle,
			gpuBuffer: sharedHandle.buffer
		} as unknown as SelectedHourGpuResidentOutput['output'];
		const staleOutput = {
			gpuOutputHandle: sharedHandle,
			gpuBuffer: sharedHandle.buffer
		} as unknown as SelectedHourGpuResidentOutput['output'];
		const first = deferred<SelectedHourLiveResult>();
		const second = deferred<SelectedHourLiveResult>();
		const third = deferred<SelectedHourLiveResult>();
		const replacementGpu = createGpuResidentOutput(32, 32);
		const sessionMock = createSessionMock([
			async () => first.promise,
			async () => second.promise,
			async () => third.promise
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		const firstRequest = controller.requestSelection(createRequestParams(30));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(1);
		});
		const secondRequest = controller.requestSelection(createRequestParams(31));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
		});

		second.resolve(
			createLiveResult({
				requestId: 31,
				timeIndex: 31,
				analysis: createSelectionAnalysis('current-shared-output', [21, 23]),
				gpuResidentOutput: {
					...createGpuResidentOutput(31, 31).accepted,
					output: currentOutput,
					gpuOutputHandle: sharedHandle
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(secondRequest).resolves.toMatchObject({ accepted: true });
		const currentSurfaceIdentity = controller.getState().surfaceIdentity;
		expect(currentSurfaceIdentity).not.toBeNull();

		first.resolve(
			createLiveResult({
				requestId: 30,
				timeIndex: 30,
				analysis: createSelectionAnalysis('stale-shared-output', [17, 19]),
				gpuResidentOutput: {
					...createGpuResidentOutput(30, 30).accepted,
					output: staleOutput,
					gpuOutputHandle: sharedHandle
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(firstRequest).resolves.toMatchObject({ accepted: false, reason: 'stale' });

		expect(destroy).not.toHaveBeenCalled();

		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: currentSurfaceIdentity!.controllerIdentity,
			controllerInstanceId: currentSurfaceIdentity!.controllerInstanceId,
			requestId: currentSurfaceIdentity!.requestId,
			monthIndex: currentSurfaceIdentity!.monthIndex,
			timeIndex: currentSurfaceIdentity!.timeIndex,
			reason: 'copy-complete'
		});
		const thirdRequest = controller.requestSelection(createRequestParams(32));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(3);
		});
		third.resolve(
			createLiveResult({
				requestId: 32,
				timeIndex: 32,
				analysis: createSelectionAnalysis('replacement-output', [25, 27]),
				gpuResidentOutput: replacementGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(thirdRequest).resolves.toMatchObject({ accepted: true });

		expect(destroy).toHaveBeenCalledTimes(1);
	});

	it('does not destroy a stale rejected GPU output when it shares a retired accepted output handle', async () => {
		const destroy = vi.fn();
		const sharedHandle: SelectedHourOutputHandle = {
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 4,
			requestId: 41,
			timeIndex: 41,
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-on-demand-snapshot',
			metricType: 'utci',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'time-index', index: 41 },
			disposed: false,
			dispose() {
				if (sharedHandle.disposed) return;
				sharedHandle.disposed = true;
				destroy();
			}
		};
		const retiredOutput = {
			gpuOutputHandle: sharedHandle,
			gpuBuffer: sharedHandle.buffer
		} as unknown as SelectedHourGpuResidentOutput['output'];
		const staleOutput = {
			gpuOutputHandle: sharedHandle,
			gpuBuffer: sharedHandle.buffer
		} as unknown as SelectedHourGpuResidentOutput['output'];
		const first = deferred<SelectedHourLiveResult>();
		const second = deferred<SelectedHourLiveResult>();
		const third = deferred<SelectedHourLiveResult>();
		const replacementGpu = createGpuResidentOutput(42, 42);
		const sessionMock = createSessionMock([
			async () => first.promise,
			async () => second.promise,
			async () => third.promise
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		const firstRequest = controller.requestSelection(createRequestParams(40));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(1);
		});
		const secondRequest = controller.requestSelection(createRequestParams(41));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(2);
		});

		second.resolve(
			createLiveResult({
				requestId: 41,
				timeIndex: 41,
				analysis: createSelectionAnalysis('retired-shared-output', [21, 23]),
				gpuResidentOutput: {
					...createGpuResidentOutput(41, 41).accepted,
					output: retiredOutput,
					gpuOutputHandle: sharedHandle
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(secondRequest).resolves.toMatchObject({ accepted: true });
		const retiredSurfaceIdentity = controller.getState().surfaceIdentity;
		expect(retiredSurfaceIdentity).not.toBeNull();

		const thirdRequest = controller.requestSelection(createRequestParams(42));
		await vi.waitFor(() => {
			expect(sessionMock.runSelectedHour).toHaveBeenCalledTimes(3);
		});
		third.resolve(
			createLiveResult({
				requestId: 42,
				timeIndex: 42,
				analysis: createSelectionAnalysis('replacement-output', [25, 27]),
				gpuResidentOutput: replacementGpu.accepted,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(thirdRequest).resolves.toMatchObject({ accepted: true });

		first.resolve(
			createLiveResult({
				requestId: 40,
				timeIndex: 40,
				analysis: createSelectionAnalysis('stale-retired-shared-output', [17, 19]),
				gpuResidentOutput: {
					...createGpuResidentOutput(40, 40).accepted,
					output: staleOutput,
					gpuOutputHandle: sharedHandle
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true
			})
		);
		await expect(firstRequest).resolves.toMatchObject({ accepted: false, reason: 'stale' });

		expect(destroy).not.toHaveBeenCalled();

		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: retiredSurfaceIdentity!.controllerIdentity,
			controllerInstanceId: retiredSurfaceIdentity!.controllerInstanceId,
			requestId: retiredSurfaceIdentity!.requestId,
			monthIndex: retiredSurfaceIdentity!.monthIndex,
			timeIndex: retiredSurfaceIdentity!.timeIndex,
			reason: 'superseded'
		});

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(replacementGpu.destroy).not.toHaveBeenCalled();
	});

	it('disposes retired GPU outputs on controller disposal even without scene release', async () => {
		const firstGpu = createGpuResidentOutput(13, 13);
		const secondGpu = createGpuResidentOutput(14, 14);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 13,
					timeIndex: 13,
					analysis: createSelectionAnalysis('gpu-first', [19, 21]),
					gpuResidentOutput: firstGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				}),
			async () =>
				createLiveResult({
					requestId: 14,
					timeIndex: 14,
					analysis: createSelectionAnalysis('gpu-second', [25, 27]),
					gpuResidentOutput: secondGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(13));
		await controller.requestSelection(createRequestParams(14));
		controller.dispose();

		expect(firstGpu.destroy).toHaveBeenCalledTimes(1);
		expect(secondGpu.destroy).toHaveBeenCalledTimes(1);
	});

	it('forwards the selected-hour readback reason to the session', async () => {
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 9,
					timeIndex: 9,
					analysis: createSelectionAnalysis('comparison-readback', [21, 23]),
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection({
			...createRequestParams(9),
			selectedHourReadbackReason: 'comparison'
		});

		expect(sessionMock.runSelectedHour).toHaveBeenCalledWith(
			expect.objectContaining({
				timeIndex: 9,
				selectedHourReadbackReason: 'comparison'
			})
		);
	});

	it('publishes selected-hour readback reasons from the accepted session result', async () => {
		const diagnostics = createEmptyOnDemandDiagnostics();
		diagnostics.selectedHourReadbackReasons = ['range', 'tooltip'];
		diagnostics.selectedHourReadbackReasonCounts = { range: 1, tooltip: 1 };
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 10,
					timeIndex: 10,
					analysis: createSelectionAnalysis('readback-diagnostics', [21, 23]),
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(10));

		expect(controller.getState().selectedHourReadbackReasons).toEqual([
			'range',
			'tooltip'
		]);
		expect(controller.getState().selectedHourReadbackReasonCounts).toEqual({
			range: 1,
			tooltip: 1
		});
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
		const failedRequestId = getCurrentSurfaceRequestId(controller);

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'failed', 'copy failed')
		);

		expect(gpu.destroy).not.toHaveBeenCalled();
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
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: failedRequestId,
			monthIndex: 0,
			timeIndex: 11,
			reason: 'copy-failed'
		});
		expect(gpu.destroy).toHaveBeenCalledTimes(1);

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
		const failedRequestId = getCurrentSurfaceRequestId(controller);

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
		expect(gpu.destroy).not.toHaveBeenCalled();
		expect(controller.getState()).toMatchObject({
			analysis: fallbackAnalysis,
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			awaitingGpuSurface: false,
			ready: true,
			renderReady: false
		});
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: failedRequestId,
			monthIndex: 0,
			timeIndex: 13,
			reason: 'copy-failed'
		});
		expect(gpu.destroy).toHaveBeenCalledTimes(1);
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
		expect(staleGpu.destroy).not.toHaveBeenCalled();
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: staleRequestId,
			monthIndex: 3,
			timeIndex: 81,
			reason: 'superseded'
		});
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
		expect(staleGpu.destroy).not.toHaveBeenCalled();
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: 1,
			monthIndex: 3,
			timeIndex: 83,
			reason: 'superseded'
		});
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

	it('does not treat changed render publication detail bytes as an idempotent diagnostics update', async () => {
		const gpu = createGpuResidentOutput(38, 38);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 38,
					timeIndex: 38,
					analysis: createSelectionAnalysis('gpu-render-publication-detail-change', [18, 20]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3800
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(38));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationVertexCount: 49030566,
				renderPublicationGridWidth: 2861,
				renderPublicationGridHeight: 2856,
				renderPublicationGridSize: 0.5,
				renderPublicationSourceByteLength: 32687044,
				renderPublicationTargetByteLength: 32687044,
				renderPublicationRenderOwnedBytes: 32687044
			}
		});

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationVertexCount: 49030566,
				renderPublicationGridWidth: 2861,
				renderPublicationGridHeight: 2856,
				renderPublicationGridSize: 0.5,
				renderPublicationSourceByteLength: 32687044,
				renderPublicationTargetByteLength: 32687044,
				renderPublicationRenderOwnedBytes: 65374088
			}
		});

		unsubscribe();
		expect(emittedStates).toHaveLength(1);
		expect(controller.getState().renderSurfaceDiagnostics.renderPublication).toMatchObject({
			renderPublicationRenderOwnedBytes: 65374088
		});
		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationRenderOwnedBytes: 65374088
		});
	});

	it('does not treat changed render publication index counts as idempotent diagnostics updates', async () => {
		const gpu = createGpuResidentOutput(40, 40);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 40,
					timeIndex: 40,
					analysis: createSelectionAnalysis('gpu-render-publication-index-change', [18, 20]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 4000
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(40));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationVertexCount: 8177472,
				renderPublicationIndexCount: 49030566,
				renderPublicationDrawIndexCount: 49030566,
				renderPublicationGridWidth: 2861,
				renderPublicationGridHeight: 2856,
				renderPublicationGridSize: 0.5,
				renderPublicationSourceByteLength: 32687044,
				renderPublicationTargetByteLength: 32687044,
				renderPublicationRenderOwnedBytes: 32687044
			}
		});

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationVertexCount: 8177472,
				renderPublicationIndexCount: 49030560,
				renderPublicationDrawIndexCount: 49030566,
				renderPublicationGridWidth: 2861,
				renderPublicationGridHeight: 2856,
				renderPublicationGridSize: 0.5,
				renderPublicationSourceByteLength: 32687044,
				renderPublicationTargetByteLength: 32687044,
				renderPublicationRenderOwnedBytes: 32687044
			}
		});

		expect(emittedStates).toHaveLength(1);
		expect(controller.getState().renderSurfaceDiagnostics.renderPublication).toMatchObject({
			renderPublicationIndexCount: 49030560,
			renderPublicationDrawIndexCount: 49030566
		});
		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationIndexCount: 49030560,
			renderPublicationDrawIndexCount: 49030566
		});

		await controller.handleRenderSurfaceDiagnostics({
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationVertexCount: 8177472,
				renderPublicationIndexCount: 49030560,
				renderPublicationDrawIndexCount: 49030554,
				renderPublicationGridWidth: 2861,
				renderPublicationGridHeight: 2856,
				renderPublicationGridSize: 0.5,
				renderPublicationSourceByteLength: 32687044,
				renderPublicationTargetByteLength: 32687044,
				renderPublicationRenderOwnedBytes: 32687044
			}
		});

		unsubscribe();
		expect(emittedStates).toHaveLength(2);
		expect(controller.getState().renderSurfaceDiagnostics.renderPublication).toMatchObject({
			renderPublicationIndexCount: 49030560,
			renderPublicationDrawIndexCount: 49030554
		});
		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationIndexCount: 49030560,
			renderPublicationDrawIndexCount: 49030554
		});
	});

	it('does not treat changed active-window reset history as an idempotent diagnostics update', async () => {
		const gpu = createGpuResidentOutput(39, 39);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 39,
					timeIndex: 39,
					analysis: createSelectionAnalysis('gpu-active-reset-window-change', [18, 20]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 3900
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(39));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					scenePendingSurfaceObservedAtMs: 101,
					sceneSyncAttemptStartedAtMs: 105,
					sceneSyncActiveWindowResetHistory: []
				}
			})
		});

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					scenePendingSurfaceObservedAtMs: 101,
					sceneSyncAttemptStartedAtMs: 105,
					sceneSyncActiveWindowResetHistory: [
						{
							resetAtMs: 103,
							resetReason: 'fallback-cpu-surface',
							invalidateActiveRun: false,
							previousCopyRunToken: 3,
							nextCopyRunToken: 3
						}
					]
				}
			})
		});

		unsubscribe();
		expect(emittedStates).toHaveLength(1);
		expect(
			controller.getState().runtimeDiagnostics?.timings.renderPublication
				?.renderPublicationTimeline?.sceneSyncActiveWindowResetHistory
		).toEqual([
			{
				resetAtMs: 103,
				resetReason: 'fallback-cpu-surface',
				invalidateActiveRun: false,
				previousCopyRunToken: 3,
				nextCopyRunToken: 3
			}
		]);
	});

	it('stamps selected-hour publication start and visible acknowledgment into render publication timeline', async () => {
		const gpu = createGpuResidentOutput(62, 14);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 62,
					timeIndex: 14,
					analysis: createSelectionAnalysis('gpu-publication-window-split', [15, 17]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 6200
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(62));

		expect(
			controller.getState().runtimeDiagnostics?.timings.renderPublication
				?.renderPublicationTimeline
		).toMatchObject({
			selectedHourValuePublicationStartedAtMs: 6200,
			controllerAcceptedAtMs: expect.any(Number)
		});

		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					sceneSurfaceReceivedAtMs: 6210,
					sceneSyncCompletedAtMs: 6220
				}
			})
		});

		const timeline =
			controller.getState().runtimeDiagnostics?.timings.renderPublication
				?.renderPublicationTimeline;
		expect(timeline?.selectedHourValuePublicationStartedAtMs).toBe(6200);
		expect(timeline?.controllerVisibleAcknowledgedAtMs).toEqual(expect.any(Number));
		expect(
			timeline?.controllerVisibleAcknowledgedAtMs ?? 0
		).toBeGreaterThanOrEqual(timeline?.controllerAcceptedAtMs ?? 0);
	});

	it('does not advance controllerVisibleAcknowledgedAtMs on later CPU fallback visibility without a renderPublication payload', async () => {
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(103)
			.mockReturnValueOnce(107)
			.mockReturnValueOnce(150)
			.mockReturnValueOnce(220);
		const gpu = createGpuResidentOutput(63, 15);
		const fallbackAnalysis = createSelectionAnalysis('gpu-then-cpu-visible-ack-fallback', [
			15,
			17
		]);
		const loadCpuFallback = vi.fn(async () => ({
			analysis: fallbackAnalysis,
			cpuFallbackValues: new Float32Array([15, 17])
		}));
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 63,
					timeIndex: 15,
					analysis: null,
					gpuResidentOutput: gpu.accepted,
					loadCpuFallback,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 6300
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(63));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'failed', 'copy failed'),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					sceneSurfaceReceivedAtMs: 6310,
					sceneSyncCompletedAtMs: 6320
				}
			})
		});

		expect(loadCpuFallback).toHaveBeenCalledTimes(1);
		expect(controller.getState().renderTransport).toBe('cpu-uploaded-selected-hour');
		const beforeCpuVisible =
			controller.getState().runtimeDiagnostics?.timings.renderPublication
				?.renderPublicationTimeline?.controllerVisibleAcknowledgedAtMs;
		expect(beforeCpuVisible).toBeUndefined();

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 1
			})
		);

		expect(controller.getState().acceptedVisibleSurface).toMatchObject({
			requestId: getCurrentSurfaceRequestId(controller),
			selectionKey: '1:0:15:15',
			visibleAtMs: expect.any(Number)
		});
		expect(controller.getState().visibleSelectedHourReadbackCount).toBe(1);
		const timeline =
			controller.getState().runtimeDiagnostics?.timings.renderPublication
				?.renderPublicationTimeline;
		expect(timeline?.controllerVisibleAcknowledgedAtMs).toBeUndefined();
		nowSpy.mockRestore();
	});

	it('does not treat changed layout-reuse timeline diagnostics as an idempotent diagnostics update', async () => {
		const gpu = createGpuResidentOutput(40, 40);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 40,
					timeIndex: 40,
					analysis: createSelectionAnalysis('gpu-layout-reuse-telemetry-change', [18, 20]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 4000
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(40));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					renderLayoutBuildTrace: {
						totalMs: 5,
						arrayAllocationMs: 0.5,
						transformBoundsPassMs: 2,
						coordinateAssignmentMs: 1,
						indexToTexelFillMs: 0.5,
						cellToPointIndexBuildMs: 0.75,
						colorBufferAllocationMs: 0.25
					},
					renderLayoutReuseProofTrace: {
						decision: 'reuse-safe',
						hoverCellLookupProofStatus: 'same-point-confirmed',
						previousLayoutPresent: true,
						canonicalRuntimeCompatibilityWouldReuse: true,
						proofMatchesCanonicalRuntimeCompatibility: true,
						positionsReferenceMatch: true,
						pointCountMatch: true,
						gridSizeMatch: true,
						coordinateSystemMatch: true,
						normalizationSignature: {
							enabled: true,
							offset: { x: 0.5, y: 0, z: -0.5 },
							provenance: 'anchor-offset-minus-origin'
						},
						previousNormalizationSignature: {
							enabled: true,
							offset: { x: 0.5, y: 0, z: -0.5 },
							provenance: 'anchor-offset-minus-origin'
						},
						normalizationSignatureMatch: true,
						constructionMode: 'world-positions',
						previousConstructionMode: 'world-positions',
						constructionModeMatch: true,
						dimensionsMatch: true,
						placementMatch: true,
						cellToPointMappingMatch: true,
						proofCostMs: 1.25,
						estimatedRetainedCpuLayoutBytes: 32687044
					},
					renderLayoutReuseAction: 'build-required',
					renderLayoutReuseReason: 'layout-key-mismatch',
					renderLayoutReuseDecisionMs: 0.5,
					renderLayoutReuseKeyMs: 0.75,
					renderLayoutReuseSourceSignatureMs: 0.2,
					renderLayoutReusePositionsSignatureMs: 0.15,
					renderLayoutReusePositionsSignatureCacheHit: false,
					renderLayoutReuseFrameCacheLookupMs: 0.1,
					renderLayoutReuseFrameDerivationMs: 0.25,
					renderLayoutReuseFrameCacheHit: false,
					renderLayoutReuseFrameCacheKind: 'structural',
					renderLayoutReuseKeyMatch: false,
					renderLayoutReuseProofSource: 'fresh-build-proof',
					renderLayoutReusePreviousKey: 'analysis|old-key',
					renderLayoutReusePreviousRequestId: 39,
					renderLayoutReusePreviousSelectionKey: 'old-selection',
					activeLayoutCandidateCount: 1
				}
			})
		});

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					renderLayoutBuildTrace: {
						totalMs: 5,
						arrayAllocationMs: 0.5,
						transformBoundsPassMs: 2,
						coordinateAssignmentMs: 1,
						indexToTexelFillMs: 0.5,
						cellToPointIndexBuildMs: 0.75,
						colorBufferAllocationMs: 0.25
					},
					renderLayoutReuseProofTrace: {
						decision: 'reuse-safe',
						hoverCellLookupProofStatus: 'same-point-confirmed',
						previousLayoutPresent: true,
						canonicalRuntimeCompatibilityWouldReuse: true,
						proofMatchesCanonicalRuntimeCompatibility: true,
						positionsReferenceMatch: true,
						pointCountMatch: true,
						gridSizeMatch: true,
						coordinateSystemMatch: true,
						normalizationSignature: {
							enabled: true,
							offset: { x: 0.5, y: 0, z: -0.5 },
							provenance: 'anchor-offset-minus-origin'
						},
						previousNormalizationSignature: {
							enabled: true,
							offset: { x: 0.5, y: 0, z: -0.5 },
							provenance: 'anchor-offset-minus-origin'
						},
						normalizationSignatureMatch: true,
						constructionMode: 'world-positions',
						previousConstructionMode: 'world-positions',
						constructionModeMatch: true,
						dimensionsMatch: true,
						placementMatch: true,
						cellToPointMappingMatch: true,
						proofCostMs: 1.25,
						estimatedRetainedCpuLayoutBytes: 32687044
					},
					renderLayoutReuseAction: 'build-required',
					renderLayoutReuseReason: 'layout-key-mismatch',
					renderLayoutReuseDecisionMs: 0.5,
					renderLayoutReuseKeyMs: 1.25,
					renderLayoutReuseSourceSignatureMs: 0.21,
					renderLayoutReusePositionsSignatureMs: 0.05,
					renderLayoutReusePositionsSignatureCacheHit: true,
					renderLayoutReuseFrameCacheLookupMs: 0.08,
					renderLayoutReuseFrameDerivationMs: 0,
					renderLayoutReuseFrameCacheHit: true,
					renderLayoutReuseFrameCacheKind: 'structural',
					renderLayoutReuseKeyMatch: false,
					renderLayoutReuseProofSource: 'previous-publication-proof',
					renderLayoutReusePreviousKey: 'analysis|new-key',
					renderLayoutReusePreviousRequestId: 40,
					renderLayoutReusePreviousSelectionKey: 'new-selection',
					activeLayoutCandidateCount: 2
				}
			})
		});

		unsubscribe();
		expect(emittedStates).toHaveLength(1);
		expect(
			controller.getState().renderSurfaceDiagnostics.renderPublication?.renderPublicationTimeline
		).toMatchObject({
			renderLayoutReuseDecisionMs: 0.5,
			renderLayoutReuseKeyMs: 1.25,
			renderLayoutReuseSourceSignatureMs: 0.21,
			renderLayoutReusePositionsSignatureMs: 0.05,
			renderLayoutReusePositionsSignatureCacheHit: true,
			renderLayoutReuseFrameCacheLookupMs: 0.08,
			renderLayoutReuseFrameDerivationMs: 0,
			renderLayoutReuseFrameCacheHit: true,
			renderLayoutReuseFrameCacheKind: 'structural',
			renderLayoutReuseProofSource: 'previous-publication-proof',
			renderLayoutReusePreviousKey: 'analysis|new-key',
			renderLayoutReusePreviousRequestId: 40,
			renderLayoutReusePreviousSelectionKey: 'new-selection',
			activeLayoutCandidateCount: 2,
			renderLayoutBuildTrace: {
				totalMs: 5
			},
			renderLayoutReuseProofTrace: {
				decision: 'reuse-safe',
				proofCostMs: 1.25
			}
		});
		expect(
			controller.getState().runtimeDiagnostics?.timings.renderPublication?.renderPublicationTimeline
		).toMatchObject({
			renderLayoutReuseKeyMs: 1.25,
			renderLayoutReuseSourceSignatureMs: 0.21,
			renderLayoutReusePositionsSignatureMs: 0.05,
			renderLayoutReusePositionsSignatureCacheHit: true,
			renderLayoutReuseFrameCacheLookupMs: 0.08,
			renderLayoutReuseFrameDerivationMs: 0,
			renderLayoutReuseFrameCacheHit: true,
			renderLayoutReuseProofSource: 'previous-publication-proof',
			activeLayoutCandidateCount: 2
		});
	});

	it('does not treat changed selected-day range cache timeline diagnostics as idempotent', async () => {
		const gpu = createGpuResidentOutput(64, 16);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 64,
					timeIndex: 16,
					analysis: createSelectionAnalysis('gpu-range-cache-diagnostics', [18, 20]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 6400
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(16));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					sessionRangeResolveStartedAtMs: 100,
					sessionRangeResolveCompletedAtMs: 220
				}
			})
		});

		const emittedStates: LiveSelectedHourControllerSurfaceDiagnostics[] = [];
		const unsubscribe = controller.subscribe((state) => {
			emittedStates.push(state.renderSurfaceDiagnostics);
		});

		await controller.handleRenderSurfaceDiagnostics({
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationTimeline: {
					sessionRangeResolveStartedAtMs: 100,
					sessionRangeResolveCompletedAtMs: 220,
					sessionSelectedDayRangeCacheKey: '8:24',
					sessionSelectedDayRangeCacheHit: false,
					sessionSelectedDayRangeCacheSizeBefore: 1,
					sessionSelectedDayRangeCacheSizeAfter: 2,
					sessionSelectedDayRangeReadbackCount: 23,
					sessionSelectedDayRangeComputedHourCount: 23,
					sceneReactiveToSyncQueuedMs: 0.1,
					sceneSyncQueuedToStartMs: 0.2
				}
			})
		});

		unsubscribe();

		expect(emittedStates).toHaveLength(1);
		expect(
			controller.getState().runtimeDiagnostics?.timings.renderPublication
				?.renderPublicationTimeline
		).toMatchObject({
			sessionSelectedDayRangeCacheKey: '8:24',
			sessionSelectedDayRangeCacheHit: false,
			sessionSelectedDayRangeCacheSizeBefore: 1,
			sessionSelectedDayRangeCacheSizeAfter: 2,
			sessionSelectedDayRangeReadbackCount: 23,
			sessionSelectedDayRangeComputedHourCount: 23,
			sceneReactiveToSyncQueuedMs: 0.1,
			sceneSyncQueuedToStartMs: 0.2
		});
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
		expect(controller.getState().acceptedVisibleSurface).toBeNull();
		expect(controller.getState().acceptedRequestId).toBeUndefined();
		expect(controller.getState().acceptedSelectionKey).toBeUndefined();
		expect(controller.getState().acceptedVisibleAtMs).toBeUndefined();

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
			acceptedVisibleSurface: null,
			acceptedRequestId: undefined,
			acceptedSelectionKey: undefined,
			acceptedVisibleAtMs: undefined,
			renderTransport: 'cpu-uploaded-selected-hour',
			loading: false,
			renderReady: false
		});
		expect(controller.getState().analysis?.metadata.model_file).toBe('gpu-atomic.glb');
		controller.releaseAcceptedGpuResidentOutput({
			controllerIdentity: 'controller',
			controllerInstanceId: 0,
			requestId: getCurrentSurfaceRequestId(controller),
			monthIndex: 1,
			timeIndex: 41,
			reason: 'copy-failed'
		});
		expect(gpu.destroy).toHaveBeenCalledTimes(1);

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentCpuSurfaceDiagnostics(controller, {
				selectedHourTransferCount: 1
			})
		);

		expect(controller.getState()).toMatchObject({
			acceptedGpuResidentOutput: null,
			acceptedVisibleSurface: {
				requestId: 1,
				selectionKey: '1:1:17:41',
				visibleAtMs: expect.any(Number)
			},
			acceptedRequestId: 1,
			acceptedSelectionKey: '1:1:17:41',
			acceptedVisibleAtMs: expect.any(Number),
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
			acceptedVisibleSurface: null,
			acceptedRequestId: undefined,
			acceptedSelectionKey: undefined,
			acceptedVisibleAtMs: undefined,
			visibleSelectedHourReadbackCount: undefined,
			readbackInstrumentation: 'not-instrumented',
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
			acceptedVisibleSurface: {
				requestId: 1,
				selectionKey: '1:0:21:21',
				visibleAtMs: expect.any(Number)
			},
			acceptedRequestId: 1,
			acceptedSelectionKey: '1:0:21:21',
			acceptedVisibleAtMs: expect.any(Number),
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
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

	it('tracks explicit zero visible readbacks for compute-buffer visible surfaces', async () => {
		const gpu = createGpuResidentOutput(91, 91);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 91,
					timeIndex: 91,
					analysis: createSelectionAnalysis('gpu-visible-readback-proof', [18, 22]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9100
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(91));

		expect(controller.getState()).toMatchObject({
			visibleSelectedHourReadbackCount: undefined,
			readbackInstrumentation: 'not-instrumented'
		});

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'complete')
		);

		expect(controller.getState()).toMatchObject({
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});
	});

	it('preserves existing render-update timings when later non-visibility diagnostics arrive', async () => {
		const gpu = createGpuResidentOutput(94, 94);
		const diagnostics = createEmptyOnDemandDiagnostics();
		diagnostics.timings.oneHourDispatchMs = 12.5;
		diagnostics.trackedGpuAllocationBytes.persistentExposureBytes = 128;
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes = 64;
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark = 64;
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 94,
					timeIndex: 94,
					analysis: createSelectionAnalysis('gpu-timing-preserve', [18, 22]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9400,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(94));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderSceneSyncTotalMs: 5.5
		});

		const afterVisible = controller.getState().runtimeDiagnostics;
		expect(afterVisible?.timings.oneHourDispatchMs).toBe(12.5);
		expect(afterVisible?.timings.renderUpdateMs).toBeGreaterThanOrEqual(0);
		expect(afterVisible?.timings.gpuSurfaceUpdateMs).toBe(afterVisible?.timings.renderUpdateMs);

		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderSceneSyncTotalMs: 8.25
		});

		const afterFollowUp = controller.getState().runtimeDiagnostics;
		expect(afterFollowUp?.timings.oneHourDispatchMs).toBe(12.5);
		expect(afterFollowUp?.timings.renderUpdateMs).toBe(afterVisible?.timings.renderUpdateMs);
		expect(afterFollowUp?.timings.gpuSurfaceUpdateMs).toBe(
			afterVisible?.timings.gpuSurfaceUpdateMs
		);
		expect(afterFollowUp?.timings.renderSceneSyncTotalMs).toBe(8.25);
	});

	it('preserves detailed render publication diagnostics after visible surface acknowledgement', async () => {
		const gpu = createGpuResidentOutput(96, 96);
		const diagnostics = createEmptyOnDemandDiagnostics();
		diagnostics.timings.oneHourDispatchMs = 12.5;
		diagnostics.trackedGpuAllocationBytes.persistentExposureBytes = 128;
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes = 64;
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark = 64;
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 96,
					timeIndex: 96,
					analysis: createSelectionAnalysis(
						'gpu-render-publication-diagnostics',
						[18, 22]
					),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9600,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(96));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderSceneSyncTotalMs: 8.25,
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationTargetByteLength: 32687044
			}
		});

		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationVersion: 1,
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044
		});
	});

	it('isolates render publication diagnostics from caller and returned state mutations', async () => {
		const gpu = createGpuResidentOutput(97, 97);
		const diagnostics = createEmptyOnDemandDiagnostics();
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 97,
					timeIndex: 97,
					analysis: createSelectionAnalysis('gpu-render-publication-isolation', [18, 22]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9700,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});
		const renderPublication = createRenderPublicationDiagnostics({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044
		});

		await controller.requestSelection(createRequestParams(97));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication
		});

		renderPublication.renderPublicationPhase = 'unknown';
		renderPublication.renderPublicationPointCount = 12;

		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationVersion: 1,
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044
		});

		const snapshot = controller.getState();
		expect(snapshot.runtimeDiagnostics?.timings.renderPublication).toBeDefined();
		snapshot.runtimeDiagnostics!.timings.renderPublication!.renderPublicationPhase = 'unknown';
		snapshot.runtimeDiagnostics!.timings.renderPublication!.renderPublicationPointCount = 24;

		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationVersion: 1,
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044
		});
	});

	it('stamps compute and controller acceptance timeline values and preserves them after scene diagnostics merge', async () => {
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(103)
			.mockReturnValueOnce(107)
			.mockReturnValueOnce(109)
			.mockReturnValueOnce(111);
		const gpu = createGpuResidentOutput(98, 98);
		const diagnostics = createEmptyOnDemandDiagnostics();
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 98,
					timeIndex: 98,
					analysis: createSelectionAnalysis('gpu-render-publication-timeline', [18, 22]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9800,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(98));

		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'initial',
			renderPublicationMeshAction: 'skipped',
			renderPublicationTimeline: {
				controllerSessionRunStartedAtMs: 101,
				computeCompletedAtMs: 103,
				controllerAcceptedAtMs: 107,
				controllerDiagnosticsMergedAtMs: 109,
				controllerStatePublishedAtMs: 111
			}
		});

		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused',
				renderPublicationPointCount: 8171761,
				renderPublicationTargetByteLength: 32687044,
				renderPublicationTimeline: {
					scenePendingSurfaceObservedAtMs: 205,
					sceneSyncAttemptStartedAtMs: 209,
					sceneSyncAttemptToken: 17,
					sceneSurfaceReceivedAtMs: 211,
					publicationEffectStartedAtMs: 223,
					renderSurfaceMeshTrace: {
						action: 'updated',
						totalMs: 12,
						recreateDecision: {
							missingSurface: false,
							notComputeBufferSurface: false,
							analysisIdentityChanged: false,
							layoutCompatible: true
						},
						updateComputeBufferSurfaceMeshMs: 9,
						fallbackDecisionMs: 1,
						applySurfaceMeshStateMs: 2
					}
				}
			})
		});

		expect(controller.getState().runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044,
			renderPublicationTimeline: {
				computeCompletedAtMs: 103,
				controllerAcceptedAtMs: 107,
				controllerDiagnosticsMergedAtMs: 109,
				controllerStatePublishedAtMs: 111,
				scenePendingSurfaceObservedAtMs: 205,
				sceneSyncAttemptStartedAtMs: 209,
				sceneSyncAttemptToken: 17,
				sceneSurfaceReceivedAtMs: 211,
				publicationEffectStartedAtMs: 223,
				renderSurfaceMeshTrace: {
					action: 'updated',
					totalMs: 12,
					recreateDecision: {
						missingSurface: false,
						notComputeBufferSurface: false,
						analysisIdentityChanged: false,
						layoutCompatible: true
					},
					updateComputeBufferSurfaceMeshMs: 9,
					fallbackDecisionMs: 1,
					applySurfaceMeshStateMs: 2
				}
			}
		});

		nowSpy.mockRestore();
	});

	it('keeps active instance-proof preflight failures graceful without falling back to CPU dense publication', async () => {
		const gpu = createGpuResidentOutput(99, 99);
		const loadCpuFallback = vi.fn(async () => ({
			analysis: createSelectionAnalysis('cpu-fallback-should-not-load', [18, 22]),
			cpuFallbackValues: new Float32Array([18, 22])
		}));
		const diagnostics = createEmptyOnDemandDiagnostics();
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 99,
					timeIndex: 99,
					analysis: createSelectionAnalysis('active-preflight-failure', [18, 22]),
					gpuResidentOutput: gpu.accepted,
					loadCpuFallback,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9900,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(99));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(
				controller,
				'failed',
				'Active UTCI render allocation preflight failed: active instanced rendering requires Three TSL instanceIndex support: Three TSL instanceIndex is unavailable or not an instance uint node.'
			),
			renderPublication: createRenderPublicationDiagnostics({
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'initial',
				renderPublicationMeshAction: 'skipped',
				renderPublicationPointCount: 3,
				renderAllocationPreflight: {
					status: 'failed',
					renderTopology: 'active-cells',
					renderCellCount: 3,
					canonicalCellCount: 6,
					activePointCount: 3,
					estimatedRenderGeometryBytes: 84,
					estimatedLargestSingleRenderAllocationBytes: 48,
					estimatedDenseRectGeometryBytes: 168,
					estimatedLargestJsTypedArrayBytes: 48,
					jsLargestTypedArrayByteLimit: 268_435_456,
					rendererMaxBufferSize: 1024,
					rendererMaxStorageBufferBindingSize: 1024,
					activeRenderStrategy: 'active-instanced-quads',
					activeRenderInstanceCount: 3,
					activeRenderSharedVertexCount: 4,
					activeRenderSharedIndexCount: 6,
					activeCanonicalIndexBufferBytes: 12,
					failureReasons: [
						'active instanced rendering requires Three TSL instanceIndex support: Three TSL instanceIndex is unavailable or not an instance uint node.'
					],
					forbiddenDenseAllocationProof: {
						noDenseCellToPointStorageAttribute: true,
						noDenseColorBuffer: true,
						noWidthHeightRenderGeometry: true,
						noPerActiveCellDuplicatedVertexBuffer: true,
						noPerActiveCellDuplicatedIndexBuffer: true,
						sharedQuadVertexIndexBuffersConstantSize: true,
						instanceCountEqualsActivePointCount: true,
						noFullDenseTooltipReverseMapWithoutExplicitApprovalAndByteAccounting: true
					}
				}
			})
		});

		const state = controller.getState();
		expect(loadCpuFallback).not.toHaveBeenCalled();
		expect(state.loading).toBe(false);
		expect(state.renderTransport).toBe('compute-buffer-selected-hour');
		expect(state.error).toMatch(/allocation preflight failed/i);
		expect(state.runtimeDiagnostics?.timings.renderPublication).toMatchObject({
			renderAllocationPreflight: {
				status: 'failed',
				renderTopology: 'active-cells',
				activeRenderStrategy: 'active-instanced-quads'
			}
		});
		expect(state.acceptedVisibleSurface).toBeNull();
	});

	it('clears render-owned GPU memory diagnostics when the surface is disposed', async () => {
		const gpu = createGpuResidentOutput(95, 95);
		const diagnostics = createEmptyOnDemandDiagnostics();
		diagnostics.trackedGpuAllocationBytes.persistentExposureBytes = 128;
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes = 64;
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark = 64;
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 95,
					timeIndex: 95,
					analysis: createSelectionAnalysis('gpu-memory-clear', [18, 22]),
					gpuResidentOutput: gpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9500,
					diagnostics
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(95));
		await controller.handleRenderSurfaceDiagnostics({
			...createCurrentGpuCopyDiagnostics(controller, 'complete'),
			renderOwnedSelectedHourBytes: 512
		});

		expect(
			controller.getState().runtimeDiagnostics?.trackedGpuAllocationBytes
				.renderOwnedSelectedHourBytes
		).toBe(512);
		expect(
			controller.getState().runtimeDiagnostics?.trackedGpuAllocationBytes
				.renderOwnedSelectedHourBytesHighWatermark
		).toBe(512);

		await controller.handleRenderSurfaceDiagnostics({
			renderOwnedSelectedHourBytes: 0
		});

		expect(controller.getState().renderSurfaceDiagnostics.renderOwnedSelectedHourBytes).toBe(0);
		expect(
			controller.getState().runtimeDiagnostics?.trackedGpuAllocationBytes
				.renderOwnedSelectedHourBytes
		).toBe(0);
		expect(
			controller.getState().runtimeDiagnostics?.trackedGpuAllocationBytes
				.renderOwnedSelectedHourBytesHighWatermark
		).toBe(0);
	});

	it('resets visible-readback proof for a replacement GPU request until the new visible surface completes', async () => {
		const firstGpu = createGpuResidentOutput(92, 92);
		const secondGpu = createGpuResidentOutput(93, 93);
		const sessionMock = createSessionMock([
			async () =>
				createLiveResult({
					requestId: 92,
					timeIndex: 92,
					analysis: createSelectionAnalysis('gpu-proof-first', [18, 22]),
					gpuResidentOutput: firstGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9200
				}),
			async () =>
				createLiveResult({
					requestId: 93,
					timeIndex: 93,
					analysis: createSelectionAnalysis('gpu-proof-second', [19, 23]),
					gpuResidentOutput: secondGpu.accepted,
					renderTransport: 'compute-buffer-selected-hour',
					sameDeviceForComputeAndRender: true,
					pendingRenderUpdateStartedAt: 9300
				})
		]);
		const controller = createLiveSelectedHourController({
			prepareSession: vi.fn(async () => sessionMock.session)
		});

		await controller.requestSelection(createRequestParams(92));
		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'complete')
		);

		expect(controller.getState()).toMatchObject({
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});

		await controller.requestSelection(createRequestParams(93));

		expect(controller.getState()).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			visibleSelectedHourReadbackCount: undefined,
			readbackInstrumentation: 'not-instrumented'
		});

		await controller.handleRenderSurfaceDiagnostics(
			createCurrentGpuCopyDiagnostics(controller, 'complete')
		);

		expect(controller.getState()).toMatchObject({
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});
	});
});
