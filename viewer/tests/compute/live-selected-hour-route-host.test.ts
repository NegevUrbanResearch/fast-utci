import { describe, expect, it, vi, type Mock } from 'vitest';
import type { Group } from 'three';
import type { Analysis } from '$lib/types/analysis';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';
import {
	createLiveSelectedHourRouteHost,
	type LiveSelectedHourRouteHostDeps,
	type LiveSelectedHourRouteInputs
} from '$lib/compute/liveSelectedHourRouteHost';
import type { UtciRenderMode } from '$lib/utciRenderMode';
import type {
	LiveSelectedHourController,
	LiveSelectedHourControllerRequest,
	LiveSelectedHourControllerState,
	LiveSelectedHourControllerSurfaceDiagnostics,
	LiveSelectedHourRenderTransport
} from '$lib/compute/liveSelectedHourController';
import { createFullDayAnalysis } from './live-selected-hour-route-host.test-support';

type TestLiveSelectedHourRouteInputs = LiveSelectedHourRouteInputs & {
	utciRenderMode: UtciRenderMode;
};

type TestLiveSelectedHourRouteInputOverrides = Omit<
	Partial<TestLiveSelectedHourRouteInputs>,
	'selection' | 'comparison'
> & {
	selection?: Partial<LiveSelectedHourRouteInputs['selection']>;
	comparison?: Partial<LiveSelectedHourRouteInputs['comparison']>;
};

function createSingleHourAnalysis(label: string): Analysis {
	return {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: 2,
			hours: ['12:00'],
			utci_range: { min: 15, max: 25 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: `${label}.glb`,
			source_analysis_id: label
		},
		data: {
			numPositions: 2,
			numHours: 1,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciValues: new Float32Array([15, 25])
		}
	};
}

function createInitialControllerState(): LiveSelectedHourControllerState {
	return {
		analysis: null,
		acceptedGpuResidentOutput: null,
		surfaceIdentity: null,
		acceptedVisibleSurface: null,
		acceptedRequestId: undefined,
		acceptedSelectionKey: undefined,
		acceptedVisibleAtMs: undefined,
		visibleSelectedHourReadbackCount: undefined,
		readbackInstrumentation: 'not-instrumented',
		selectedHourReadbackReasons: [],
		selectedHourReadbackReasonCounts: {},
		loading: false,
		error: null,
		renderTransport: 'idle',
		sameDeviceForComputeAndRender: null,
		pendingRenderUpdateStartedAt: undefined,
		renderSurfaceDiagnostics: {},
		ready: false,
		renderReady: false,
		awaitingGpuSurface: false
	};
}

function createFakeGpuResidentOutput(requestId: number): SelectedHourGpuResidentOutput {
	return {
		requestId,
		monthIndex: 7,
		hourIndex: 12,
		timeIndex: 180,
		output: {} as SelectedHourGpuResidentOutput['output'],
		utciRange: { min: 18, max: 30 }
	};
}

function cloneControllerState(state: LiveSelectedHourControllerState): LiveSelectedHourControllerState {
	return {
		...state,
		acceptedVisibleSurface: state.acceptedVisibleSurface
			? { ...state.acceptedVisibleSurface }
			: null,
		selectedHourReadbackReasons: [...state.selectedHourReadbackReasons],
		selectedHourReadbackReasonCounts: { ...state.selectedHourReadbackReasonCounts },
		surfaceIdentity: state.surfaceIdentity ? { ...state.surfaceIdentity } : null,
		renderSurfaceDiagnostics: { ...state.renderSurfaceDiagnostics }
	};
}

type ControllerRecord = {
	requests: LiveSelectedHourControllerRequest[];
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics[];
	dispose: Mock<() => void>;
};

function createControllerFactory() {
	const records: ControllerRecord[] = [];

	const createController = (): LiveSelectedHourController => {
		let state = createInitialControllerState();
		const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
		const record: ControllerRecord = {
			requests: [],
			diagnostics: [],
			dispose: vi.fn()
		};
		records.push(record);

		function emit(): void {
			const snapshot = cloneControllerState(state);
			for (const listener of listeners) {
				listener(snapshot);
			}
		}

		function createSurfaceIdentity(
			request: LiveSelectedHourControllerRequest,
			requestId: number
		): LiveSelectedHourSurfaceIdentity {
			return {
				requestId,
				monthIndex: request.monthIndex,
				hourIndex: request.hourIndex,
				timeIndex: request.timeIndex,
				selectionKey:
					request.selectionKey ??
					`${requestId}:${request.monthIndex}:${request.hourIndex}:${request.timeIndex}`,
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			};
		}

		function resolveRenderTransport(
			request: LiveSelectedHourControllerRequest
		): LiveSelectedHourRenderTransport {
			return request.preferGpuResident
				? 'compute-buffer-selected-hour'
				: 'cpu-uploaded-selected-hour';
		}

		return {
			getState() {
				return cloneControllerState(state);
			},

			subscribe(listener) {
				listeners.add(listener);
				return () => {
					listeners.delete(listener);
				};
			},

			async requestSelection(request) {
				record.requests.push(request);
				const requestId = record.requests.length;
				const renderTransport = resolveRenderTransport(request);
				const gpuPending = renderTransport === 'compute-buffer-selected-hour';
				const surfaceIdentity = createSurfaceIdentity(request, requestId);
				const acceptedVisibleSurface = gpuPending
					? null
					: {
							requestId,
							selectionKey: surfaceIdentity.selectionKey,
							visibleAtMs: requestId * 1000
						};
				state = {
					...state,
					analysis: request.sessionConfig.base,
					surfaceIdentity,
					acceptedVisibleSurface,
					acceptedRequestId: acceptedVisibleSurface?.requestId,
					acceptedSelectionKey: acceptedVisibleSurface?.selectionKey,
					acceptedVisibleAtMs: acceptedVisibleSurface?.visibleAtMs,
					visibleSelectedHourReadbackCount: undefined,
					readbackInstrumentation: 'not-instrumented',
					selectedHourReadbackReasons: request.selectedHourReadbackReason
						? [request.selectedHourReadbackReason]
						: [],
					selectedHourReadbackReasonCounts: request.selectedHourReadbackReason
						? { [request.selectedHourReadbackReason]: 1 }
						: {},
					loading: gpuPending,
					error: null,
					renderTransport,
					sameDeviceForComputeAndRender: request.preferGpuResident ? true : false,
					pendingRenderUpdateStartedAt: gpuPending ? requestId * 100 : undefined,
					renderSurfaceDiagnostics: gpuPending
						? {
								gpuResidentCopyStatus: 'pending',
								gpuResidentCopyRequestId: requestId
							}
						: {},
					ready: true,
					renderReady: !gpuPending,
					awaitingGpuSurface: gpuPending
				};
				emit();
				return {
					accepted: true,
					state: cloneControllerState(state)
				};
			},

			async handleRenderSurfaceDiagnostics(diagnostics) {
				record.diagnostics.push(diagnostics);
				const renderSurfaceDiagnostics = {
					...state.renderSurfaceDiagnostics,
					...diagnostics
				};
				const gpuRenderReady =
					state.renderTransport === 'compute-buffer-selected-hour' &&
					renderSurfaceDiagnostics.gpuResidentCopyStatus === 'complete' &&
					renderSurfaceDiagnostics.utciSurfaceSource === 'compute-buffer-selected-hour';
				const cpuPublicationAccepted =
					state.renderTransport === 'cpu-uploaded-selected-hour' &&
					renderSurfaceDiagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour' &&
					renderSurfaceDiagnostics.cpuPublishRequestId === state.surfaceIdentity?.requestId &&
					renderSurfaceDiagnostics.cpuPublishMonthIndex === state.surfaceIdentity?.monthIndex &&
					renderSurfaceDiagnostics.cpuPublishHourIndex === state.surfaceIdentity?.hourIndex &&
					renderSurfaceDiagnostics.cpuPublishTimeIndex === state.surfaceIdentity?.timeIndex &&
					renderSurfaceDiagnostics.cpuPublishSelectionKey === state.surfaceIdentity?.selectionKey;
				const acceptedVisibleSurface = gpuRenderReady
					? {
							requestId: state.surfaceIdentity?.requestId ?? 0,
							selectionKey: state.surfaceIdentity?.selectionKey ?? 'selection',
							visibleAtMs: (state.surfaceIdentity?.requestId ?? 0) * 1000
						}
					: cpuPublicationAccepted
						? (state.acceptedVisibleSurface ?? {
								requestId: state.surfaceIdentity?.requestId ?? 0,
								selectionKey: state.surfaceIdentity?.selectionKey ?? 'selection',
								visibleAtMs: (state.surfaceIdentity?.requestId ?? 0) * 1000
							})
						: state.acceptedVisibleSurface;
				const visibleSelectedHourReadbackCount = gpuRenderReady
					? 0
					: cpuPublicationAccepted
						? 1
					: state.visibleSelectedHourReadbackCount;
				const readbackInstrumentation = gpuRenderReady
					? 'instrumented'
					: cpuPublicationAccepted
						? 'instrumented'
					: state.readbackInstrumentation;
				state = {
					...state,
					renderSurfaceDiagnostics,
					acceptedVisibleSurface,
					acceptedRequestId: acceptedVisibleSurface?.requestId,
					acceptedSelectionKey: acceptedVisibleSurface?.selectionKey,
					acceptedVisibleAtMs: acceptedVisibleSurface?.visibleAtMs,
					visibleSelectedHourReadbackCount,
					readbackInstrumentation,
					selectedHourReadbackReasons: state.selectedHourReadbackReasons,
					selectedHourReadbackReasonCounts: state.selectedHourReadbackReasonCounts,
					loading: gpuRenderReady ? false : state.loading,
					renderReady: gpuRenderReady ? true : state.renderReady,
					awaitingGpuSurface: gpuRenderReady ? false : state.awaitingGpuSurface,
					pendingRenderUpdateStartedAt: gpuRenderReady
						? undefined
						: state.pendingRenderUpdateStartedAt
				};
				emit();
			},

			dispose() {
				record.dispose();
				listeners.clear();
				state = createInitialControllerState();
			}
		};
	};

	return { createController, records };
}

function deferred<T>() {
	let resolve!: (value: T | PromiseLike<T>) => void;
	let reject!: (reason?: unknown) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

function makeHostDeps(factory = createControllerFactory()) {
	return {
		createController: factory.createController,
		resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
	} satisfies LiveSelectedHourRouteHostDeps;
}

function makeBaseInputs(
	overrides: TestLiveSelectedHourRouteInputOverrides = {}
): TestLiveSelectedHourRouteInputs {
	const { selection: selectionOverrides, comparison: comparisonOverrides, ...rootOverrides } =
		overrides;
	const monthIndex = selectionOverrides?.monthIndex ?? 7;
	const hourIndex = selectionOverrides?.hourIndex ?? 12;
	const timeIndex = selectionOverrides?.timeIndex ?? 180;
	return {
		enabled: true,
		analysisId: 'Ben-Gurion/base',
		baseAnalysis: createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		}),
		baseModel: {} as Group,
		selection: {
			monthIndex,
			hourIndex,
			timeIndex,
			selectionKey:
				selectionOverrides?.selectionKey ??
				`Ben-Gurion/base|${monthIndex}|${hourIndex}|${timeIndex}`,
			...selectionOverrides
		},
		colorMode: 'discrete',
		utciRenderMode: 'data',
		rendererBackend: 'webgpu',
		rendererDevice: { label: 'base-renderer' } as unknown as GPUDevice,
		utciSurfaceBackend: 'dataTexture',
		comparison: {
			active: false,
			analysisId: null,
			sourceAnalysis: null,
			model: null,
			...comparisonOverrides
		},
		...rootOverrides
	};
}

function makeComparisonInputs(
	overrides: TestLiveSelectedHourRouteInputOverrides = {}
): TestLiveSelectedHourRouteInputs {
	const comparisonAnalysisId =
		overrides.comparison?.analysisId ?? 'Ben-Gurion/comparison/winter';
	const comparisonSourceAnalysis =
		overrides.comparison?.sourceAnalysis ??
		createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: comparisonAnalysisId,
			baseMin: 5,
			baseMax: 22
		});

	return makeBaseInputs({
		...overrides,
		comparison: {
			active: true,
			analysisId: comparisonAnalysisId,
			sourceAnalysis: comparisonSourceAnalysis,
			model: (overrides.comparison?.model ?? ({} as Group)) as Group | null,
			rendererDevice:
				overrides.comparison?.rendererDevice ??
				({ label: 'comparison-renderer' } as unknown as GPUDevice),
			...overrides.comparison
		}
	});
}

describe('liveSelectedHourRouteHost', () => {
	it('keeps base and comparison controller orchestration out of the route', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'data',
				rendererBackend: 'unknown',
				rendererDevice: undefined,
				utciSurfaceBackend: 'dataTexture'
			})
		);
		await host.flush();

		expect(host.getState().baseDisplayAnalysis?.metadata.model_file).toBe('base.glb');
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().baseReady).toBe(true);
		expect(factory.records).toHaveLength(2);
		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.selectionKey).toBe('Ben-Gurion/base|7|12|180');
	});

	it('defers auto-mode base startup until the route surface is GPU-native', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const rendererDevice = { label: 'base-renderer-ready' } as unknown as GPUDevice;

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'auto',
				rendererBackend: 'webgpu',
				rendererDevice,
				utciSurfaceBackend: 'dataTexture',
				baseAnalysis,
				baseModel
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(0);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseDisplayAnalysis).toBeNull();
		expect(host.getState().baseSceneSurfaceIdentity).toBeNull();
		expect(host.getState().baseSurfaceIdentity).toBeNull();

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'auto',
				rendererBackend: 'webgpu',
				rendererDevice,
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.preferGpuResident).toBe(true);
		expect(factory.records[0].requests[0]?.rendererDevice).toBe(rendererDevice);
	});

	it('defers auto-mode base startup until renderer WebGPU readiness is available', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const rendererDevice = { label: 'base-renderer-ready' } as unknown as GPUDevice;

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'auto',
				rendererBackend: 'unknown',
				rendererDevice: undefined,
				utciSurfaceBackend: 'dataTexture',
				baseAnalysis,
				baseModel
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(0);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseDisplayAnalysis).toBeNull();
		expect(host.getState().baseSceneSurfaceIdentity).toBeNull();
		expect(host.getState().baseSurfaceIdentity).toBeNull();

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'auto',
				rendererBackend: 'webgpu',
				rendererDevice,
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.preferGpuResident).toBe(true);
		expect(factory.records[0].requests[0]?.rendererDevice).toBe(rendererDevice);
	});

	it('defers requested gpu-mode base startup until renderer WebGPU readiness is available', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const rendererDevice = { label: 'base-renderer-ready' } as unknown as GPUDevice;

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				rendererBackend: 'unknown',
				rendererDevice: undefined,
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(0);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseDisplayAnalysis).toBeNull();
		expect(host.getState().baseSceneSurfaceIdentity).toBeNull();
		expect(host.getState().baseSurfaceIdentity).toBeNull();

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				rendererBackend: 'webgpu',
				rendererDevice,
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.preferGpuResident).toBe(true);
		expect(factory.records[0].requests[0]?.rendererDevice).toBe(rendererDevice);
	});

	it('allows explicit data-mode base startup before renderer readiness resolves', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'data',
				rendererBackend: 'unknown',
				rendererDevice: undefined,
				utciSurfaceBackend: 'dataTexture'
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.preferGpuResident).toBe(false);
		expect(factory.records[0].requests[0]?.rendererDevice).toBeUndefined();
		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
	});

	it('forwards visible-readback instrumentation to route state', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				utciSurfaceBackend: 'gpuNative'
			})
		);
		await host.flush();

		expect(host.getState().base).toMatchObject({
			visibleSelectedHourReadbackCount: undefined,
			readbackInstrumentation: 'not-instrumented'
		});

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1,
			dataTextureBuildCount: 0
		});
		await host.flush();

		expect(host.getState().base).toMatchObject({
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});
	});

	it('resets forwarded visible-readback proof while a replacement base GPU surface is pending', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				utciSurfaceBackend: 'gpuNative',
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1,
			dataTextureBuildCount: 0
		});
		await host.flush();

		expect(host.getState().base).toMatchObject({
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				utciSurfaceBackend: 'gpuNative',
				selection: {
					monthIndex: 7,
					hourIndex: 13,
					timeIndex: 181,
					selectionKey: 'Ben-Gurion/base|7|13|181'
				}
			})
		);
		await host.flush();

		expect(host.getState().base).toMatchObject({
			renderTransport: 'compute-buffer-selected-hour',
			visibleSelectedHourReadbackCount: undefined,
			readbackInstrumentation: 'not-instrumented'
		});

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 2,
			dataTextureBuildCount: 0
		});
		await host.flush();

		expect(host.getState().base).toMatchObject({
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});
	});

	it('clears comparison source ownership immediately when the selected comparison analysis becomes stale or ineligible', async () => {
		const host = createLiveSelectedHourRouteHost(makeHostDeps(createControllerFactory()));

		host.setRouteInputs(
			makeComparisonInputs({
				comparison: {
					analysisId: 'Ben-Gurion/comparison/winter',
					sourceAnalysis: createFullDayAnalysis({
						label: 'comparison-winter',
						sourceAnalysisId: 'Ben-Gurion/comparison/winter'
					})
				}
			})
		);
		await host.flush();
		expect(host.getState().comparisonSourceAnalysisId).toBe('Ben-Gurion/comparison/winter');
		expect(host.getState().comparisonDisplayAnalysis?.metadata.model_file).toBe(
			'comparison-winter.glb'
		);
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().comparisonReady).toBe(true);

		host.setRouteInputs(
			makeComparisonInputs({
				comparison: {
					analysisId: 'Ben-Gurion/comparison/summer',
					sourceAnalysis: createFullDayAnalysis({
						label: 'comparison-winter-stale',
						sourceAnalysisId: 'Ben-Gurion/comparison/winter'
					})
				}
			})
		);
		expect(host.getState().comparisonSourceAnalysisId).toBeNull();
		expect(host.getState().comparisonDisplayAnalysis).toBeNull();
		expect(host.getState().comparisonSurfaceIdentity).toBeNull();
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().comparison.analysis).toBeNull();
		expect(host.getState().comparison.renderReady).toBe(false);

		host.setRouteInputs(
			makeComparisonInputs({
				comparison: {
					analysisId: 'Ben-Gurion/comparison/summer',
					sourceAnalysis: createSingleHourAnalysis('comparison-single-hour')
				}
			})
		);
		expect(host.getState().comparisonSourceAnalysisId).toBeNull();
		expect(host.getState().comparisonDisplayAnalysis).toBeNull();
		expect(host.getState().comparisonSurfaceIdentity).toBeNull();
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().comparison.analysis).toBeNull();
		expect(host.getState().comparison.renderReady).toBe(false);

		host.setRouteInputs(
			makeComparisonInputs({
				comparison: {
					model: null
				}
			})
		);
		expect(host.getState().comparisonSourceAnalysisId).toBeNull();
		expect(host.getState().comparisonDisplayAnalysis).toBeNull();
		expect(host.getState().comparisonSurfaceIdentity).toBeNull();
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().comparison.analysis).toBeNull();
		expect(host.getState().comparison.renderReady).toBe(false);
	});

	it('drops stale comparison controller state when the comparison analysis identity changes', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;
		const winterAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter'
		});
		const summerAnalysis = createFullDayAnalysis({
			label: 'comparison-summer',
			sourceAnalysisId: 'Ben-Gurion/comparison/summer'
		});

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				comparison: {
					analysisId: 'Ben-Gurion/comparison/winter',
					sourceAnalysis: winterAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();
		expect(host.getState().comparisonSourceAnalysisId).toBe('Ben-Gurion/comparison/winter');
		expect(factory.records[1].requests).toHaveLength(1);

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				comparison: {
					analysisId: 'Ben-Gurion/comparison/summer',
					sourceAnalysis: summerAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		expect(host.getState().comparisonSourceAnalysisId).toBe('Ben-Gurion/comparison/summer');
		expect(factory.records).toHaveLength(3);
		expect(factory.records[1].dispose).toHaveBeenCalledTimes(1);
		expect(factory.records[2].requests).toHaveLength(1);
		expect(factory.records[2].requests[0]?.sessionKey).toContain(
			'Ben-Gurion/comparison/summer'
		);
	});

	it('recomputes comparison state when month changes after comparison activation', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter',
			baseMin: 5,
			baseMax: 22
		});

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();
		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		await host.flush();
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 12,
					timeIndex: 36,
					selectionKey: 'Ben-Gurion/base|1|12|36'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();
		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 2
		});
		await host.flush();

		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|1|12|36'
		);
		expect(factory.records[1].requests).toHaveLength(2);
		expect(factory.records[1].requests[1]?.selectionKey).toBe('Ben-Gurion/base|1|12|36');
	});

	it('keeps the previous published base surface visible before reconcile catches up to a same-session selection change', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();
		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 12,
					timeIndex: 36,
					selectionKey: 'Ben-Gurion/base|1|12|36'
				}
			})
		);

		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().baseDisplayAnalysis).toBe(baseAnalysis);
	});

	it('seeds the base scene contract from the current gpu-native request before the first publish completes', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const selectedHourAnalysis = createSingleHourAnalysis('base-selected-hour');
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const pendingGpuResidentOutput = createFakeGpuResidentOutput(1);

		function createPendingBaseController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					state = {
						...state,
						analysis: selectedHourAnalysis,
						acceptedGpuResidentOutput: pendingGpuResidentOutput,
						surfaceIdentity: {
							requestId: pendingGpuResidentOutput.requestId,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: 1234,
							acceptedGpuResidentOutput: pendingGpuResidentOutput
						},
						loading: true,
						error: null,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender: true,
						pendingRenderUpdateStartedAt: 1234,
						renderSurfaceDiagnostics: {
							gpuResidentCopyStatus: 'pending',
							gpuResidentCopyRequestId: pendingGpuResidentOutput.requestId
						},
						ready: true,
						renderReady: false,
						awaitingGpuSurface: true
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		let createdControllers = 0;
		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 1 ? createPendingBaseController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity).toBeNull();
		expect(host.getState().baseDisplayAnalysis).toBe(selectedHourAnalysis);
		expect(host.getState().baseRenderContext).toMatchObject({
			analysis: selectedHourAnalysis,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			colorMode: 'discrete'
		});
		expect(host.getState().baseSceneSurfaceIdentity).toMatchObject({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			pendingRenderUpdateStartedAt: 1234
		});
		expect(host.getState().baseSceneSurfaceIdentity?.acceptedGpuResidentOutput).toBe(
			pendingGpuResidentOutput
		);
	});

	it('treats comparison as a first-class controller path and forwards diagnostics', async () => {
		const factory = createControllerFactory();
		const comparisonRendererDevice = { label: 'comparison-renderer' } as unknown as GPUDevice;
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				comparison: {
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		expect(factory.records[1].requests[0]?.preferGpuResident).toBe(true);
		expect(factory.records[1].requests[0]?.rendererDevice).toBe(comparisonRendererDevice);
		expect(factory.records[1].requests[0]?.selectedHourReadbackReason).toBe('comparison');
		expect(factory.records[1].requests[0]?.sessionConfig.preferredDevice).toBe(
			comparisonRendererDevice
		);
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().comparison.awaitingGpuSurface).toBe(true);

		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		await host.flush();

		expect(host.getState().comparison.renderSurfaceDiagnostics).toMatchObject({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		expect(host.getState().comparisonReady).toBe(true);
		expect(host.getState().comparison.awaitingGpuSurface).toBe(false);
		expect(host.getState().comparisonAcceptedVisibleSurface).toEqual({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			visibleAtMs: 1000
		});
	});

	it('seeds the comparison scene contract from the current gpu-native request before the first comparison publish completes', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter',
			baseMin: 5,
			baseMax: 22
		});
		const selectedHourAnalysis = createSingleHourAnalysis('comparison-selected-hour');
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;
		const pendingGpuResidentOutput = createFakeGpuResidentOutput(1);

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		function createPendingComparisonController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					state = {
						...state,
						analysis: selectedHourAnalysis,
						acceptedGpuResidentOutput: pendingGpuResidentOutput,
						surfaceIdentity: {
							requestId: pendingGpuResidentOutput.requestId,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: 4321,
							acceptedGpuResidentOutput: pendingGpuResidentOutput
						},
						loading: true,
						error: null,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender: true,
						pendingRenderUpdateStartedAt: 4321,
						renderSurfaceDiagnostics: {
							gpuResidentCopyStatus: 'pending',
							gpuResidentCopyRequestId: pendingGpuResidentOutput.requestId
						},
						ready: true,
						renderReady: false,
						awaitingGpuSurface: true
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		let createdControllers = 0;
		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 2
					? createPendingComparisonController()
					: createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().comparisonSurfaceIdentity).toBeNull();
		expect(host.getState().comparisonDisplayAnalysis).toBe(selectedHourAnalysis);
		expect(host.getState().comparisonRenderContext).toMatchObject({
			analysis: selectedHourAnalysis,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			colorMode: 'discrete'
		});
		expect(host.getState().comparisonSceneSurfaceIdentity).toMatchObject({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			pendingRenderUpdateStartedAt: 4321
		});
		expect(host.getState().comparisonSceneSurfaceIdentity?.acceptedGpuResidentOutput).toBe(
			pendingGpuResidentOutput
		);
	});

	it('replaces the base controller when the base model instance changes', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const initialModel = {} as Group;
		const replacementModel = {} as Group;

		host.setRouteInputs(
			makeBaseInputs({
				baseModel: initialModel
			})
		);
		await host.flush();
		expect(factory.records).toHaveLength(2);
		expect(factory.records[0].requests).toHaveLength(1);

		host.setRouteInputs(
			makeBaseInputs({
				baseModel: replacementModel
			})
		);
		await host.flush();

		expect(factory.records).toHaveLength(3);
		expect(factory.records[0].dispose).toHaveBeenCalledTimes(1);
		expect(factory.records[2].requests).toHaveLength(1);
	});

	it('replaces the base controller when the base analysis object changes under the same route identity', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const initialAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const replacementAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 12,
			baseMax: 24
		});

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis: initialAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice
			})
		);
		await host.flush();
		expect(factory.records).toHaveLength(2);
		expect(factory.records[0].requests).toHaveLength(1);

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis: replacementAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice
			})
		);
		await host.flush();

		expect(factory.records).toHaveLength(3);
		expect(factory.records[0].dispose).toHaveBeenCalledTimes(1);
		expect(factory.records[2].requests).toHaveLength(1);
		expect(host.getState().baseDisplayAnalysis).toBe(replacementAnalysis);
	});

	it('publishes the first current gpu-native base surface after the renderer device becomes available', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const rendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const baseControllers: Array<{
			completeCurrentSurface: () => void;
			dispose: ReturnType<typeof vi.fn>;
			requests: LiveSelectedHourControllerRequest[];
		}> = [];

		function createPendingGpuController(
			requestId: number,
			sameDeviceForComputeAndRender: boolean | null
		): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			const dispose = vi.fn();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			function completeCurrentSurface(): void {
				state = {
					...state,
					renderSurfaceDiagnostics: {
						...state.renderSurfaceDiagnostics,
						utciSurfaceSource: 'compute-buffer-selected-hour',
						gpuResidentCopyStatus: 'complete',
						gpuResidentCopyRequestId: requestId
					},
					loading: false,
					renderReady: true,
					awaitingGpuSurface: false,
					pendingRenderUpdateStartedAt: undefined
				};
				emit();
			}

			const requests: LiveSelectedHourControllerRequest[] = [];
			baseControllers.push({ completeCurrentSurface, dispose, requests });

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					requests.push(request);
					state = {
						...state,
						analysis: request.sessionConfig.base,
						surfaceIdentity: {
							requestId,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? `selection-${requestId}`,
							pendingRenderUpdateStartedAt: requestId * 100,
							acceptedGpuResidentOutput: createFakeGpuResidentOutput(requestId)
						},
						acceptedGpuResidentOutput: createFakeGpuResidentOutput(requestId),
						loading: true,
						error: null,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender,
						pendingRenderUpdateStartedAt: requestId * 100,
						renderSurfaceDiagnostics: {
							gpuResidentCopyStatus: 'pending',
							gpuResidentCopyRequestId: requestId
						},
						ready: true,
						renderReady: false,
						awaitingGpuSurface: true
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					dispose();
					listeners.clear();
				}
			};
		}

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		let controllerCount = 0;
		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				controllerCount += 1;
				if (controllerCount === 1) {
					return createPendingGpuController(1, true);
				}
				if (controllerCount === 2) {
					return createIdleController();
				}
				return createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererBackend: 'webgpu',
				rendererDevice: undefined,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(baseControllers[0]?.requests).toHaveLength(0);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity).toBeNull();
		expect(host.getState().base.sameDeviceForComputeAndRender).toBeNull();

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererBackend: 'webgpu',
				rendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(baseControllers).toHaveLength(1);
		expect(baseControllers[0]?.dispose).not.toHaveBeenCalled();
		expect(baseControllers[0]?.requests).toHaveLength(1);
		expect(host.getState().base.sameDeviceForComputeAndRender).toBe(true);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity).toBeNull();

		baseControllers[0]?.completeCurrentSurface();
		await host.flush();

		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity).toMatchObject({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180'
		});
		expect(host.getState().baseSceneSurfaceIdentity).toMatchObject({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180'
		});
	});

	it('publishes a current gpu-native base surface even when the controller carries no CPU analysis', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const rendererDevice = { label: 'base-renderer-ready' } as unknown as GPUDevice;
		const controllerAcceptedVisibleAtMs = 12345;

		function createGpuOnlyController(requestId: number): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					state = {
						...state,
						analysis: null,
						surfaceIdentity: {
							requestId,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? `selection-${requestId}`,
							pendingRenderUpdateStartedAt: requestId * 100,
							acceptedGpuResidentOutput: createFakeGpuResidentOutput(requestId)
						},
						acceptedGpuResidentOutput: createFakeGpuResidentOutput(requestId),
						loading: true,
						error: null,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender: true,
						pendingRenderUpdateStartedAt: requestId * 100,
						renderSurfaceDiagnostics: {
							gpuResidentCopyStatus: 'pending',
							gpuResidentCopyRequestId: requestId
						},
						ready: true,
						renderReady: false,
						awaitingGpuSurface: true
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							utciSurfaceSource: 'compute-buffer-selected-hour',
							gpuResidentCopyStatus: 'complete',
							gpuResidentCopyRequestId: requestId
						},
						acceptedVisibleSurface: {
							requestId,
							selectionKey:
								state.surfaceIdentity?.selectionKey ?? `selection-${requestId}`,
							visibleAtMs: controllerAcceptedVisibleAtMs
						},
						acceptedRequestId: requestId,
						acceptedSelectionKey:
							state.surfaceIdentity?.selectionKey ?? `selection-${requestId}`,
						acceptedVisibleAtMs: controllerAcceptedVisibleAtMs,
						loading: false,
						renderReady: true,
						awaitingGpuSurface: false,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		const host = createLiveSelectedHourRouteHost({
			createController: () => createGpuOnlyController(1),
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				utciRenderMode: 'gpu',
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererBackend: 'webgpu',
				rendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		await host.flush();

		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity).toMatchObject({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180'
		});
		expect(host.getState().acceptedRequestId).toBe(1);
		expect(host.getState().acceptedSelectionKey).toBe('Ben-Gurion/base|7|12|180');
		expect(host.getState().acceptedVisibleAtMs).toBe(controllerAcceptedVisibleAtMs);
		expect(host.getState().primaryAcceptedVisibleSurface).toEqual({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			visibleAtMs: controllerAcceptedVisibleAtMs
		});
		expect(host.getState().baseAcceptedVisibleSurface).toEqual(
			host.getState().primaryAcceptedVisibleSurface
		);
		expect(host.getState().comparisonAcceptedVisibleSurface).toBeNull();
		expect(host.getState().baseDisplayAnalysis).toBe(baseAnalysis);
		expect(host.getState().baseRenderContext?.analysis).toBe(baseAnalysis);
	});

	it('replaces the comparison controller when the comparison renderer device changes', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonModel = {} as Group;
		const initialRendererDevice = { label: 'comparison-renderer-a' } as unknown as GPUDevice;
		const replacementRendererDevice = {
			label: 'comparison-renderer-b'
		} as unknown as GPUDevice;
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter',
			baseMin: 5,
			baseMax: 22
		});

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: initialRendererDevice
				}
			})
		);
		await host.flush();
		expect(factory.records).toHaveLength(2);
		expect(factory.records[1].requests[0]?.rendererDevice).toBe(initialRendererDevice);

		host.setRouteInputs(
			makeComparisonInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: replacementRendererDevice
				}
			})
		);
		await host.flush();

		expect(factory.records).toHaveLength(3);
		expect(factory.records[1].dispose).toHaveBeenCalledTimes(1);
		expect(factory.records[2].requests[0]?.rendererDevice).toBe(
			replacementRendererDevice
		);
	});

	it('drops late comparison diagnostics after a comparison controller replacement', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;

		host.setRouteInputs(
			makeComparisonInputs({
				baseModel,
				rendererDevice: baseRendererDevice,
				comparison: {
					analysisId: 'Ben-Gurion/comparison/winter',
					sourceAnalysis: createFullDayAnalysis({
						label: 'comparison-winter',
						sourceAnalysisId: 'Ben-Gurion/comparison/winter'
					}),
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		host.setRouteInputs(
			makeComparisonInputs({
				baseModel,
				rendererDevice: baseRendererDevice,
				comparison: {
					analysisId: 'Ben-Gurion/comparison/summer',
					sourceAnalysis: createFullDayAnalysis({
						label: 'comparison-summer',
						sourceAnalysisId: 'Ben-Gurion/comparison/summer'
					}),
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		await host.flush();

		expect(host.getState().comparisonSourceAnalysisId).toBe('Ben-Gurion/comparison/summer');
		expect(host.getState().comparison.renderSurfaceDiagnostics).toEqual({});
	});

	it('drops late base diagnostics after a base controller replacement', async () => {
		const factory = createControllerFactory();
		const host = createLiveSelectedHourRouteHost(makeHostDeps(factory));
		const initialBaseAnalysis = createFullDayAnalysis({
			label: 'base-winter',
			sourceAnalysisId: 'Ben-Gurion/base'
		});
		const replacementBaseAnalysis = createFullDayAnalysis({
			label: 'base-summer',
			sourceAnalysisId: 'Ben-Gurion/base'
		});
		const initialBaseModel = {} as Group;
		const replacementBaseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis: initialBaseAnalysis,
				baseModel: initialBaseModel,
				rendererDevice: baseRendererDevice
			})
		);
		await host.flush();

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis: replacementBaseAnalysis,
				baseModel: replacementBaseModel,
				rendererDevice: baseRendererDevice
			})
		);
		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		await host.flush();

		expect(factory.records).toHaveLength(3);
		expect(host.getState().base.analysis).toBe(replacementBaseAnalysis);
		expect(host.getState().base.renderSurfaceDiagnostics).toEqual({});
	});

	it('continues reconciling after an EPW resolution failure instead of wedging the queue', async () => {
		const factory = createControllerFactory();
		let shouldThrow = true;
		const host = createLiveSelectedHourRouteHost({
			createController: factory.createController,
			resolveEpwUrl: ({ analysisId }) => {
				if (shouldThrow) {
					shouldThrow = false;
					throw new Error(`Missing EPW for ${analysisId ?? 'unknown'}`);
				}
				return `/weather/${analysisId ?? 'default'}.epw`;
			}
		});
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await expect(host.flush()).rejects.toThrow('Missing EPW for Ben-Gurion/base');
		expect(factory.records[0].requests).toHaveLength(0);

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 12,
					timeIndex: 36,
					selectionKey: 'Ben-Gurion/base|1|12|36'
				}
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.selectionKey).toBe('Ben-Gurion/base|1|12|36');
	});

	it('retries the same selection after an EPW resolution failure instead of treating it as already consumed', async () => {
		const factory = createControllerFactory();
		let shouldThrow = true;
		const host = createLiveSelectedHourRouteHost({
			createController: factory.createController,
			resolveEpwUrl: ({ analysisId }) => {
				if (shouldThrow) {
					shouldThrow = false;
					throw new Error(`Missing EPW for ${analysisId ?? 'unknown'}`);
				}
				return `/weather/${analysisId ?? 'default'}.epw`;
			}
		});
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const stableSelection = {
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'Ben-Gurion/base|7|12|180'
		};

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: stableSelection
			})
		);
		await expect(host.flush()).rejects.toThrow('Missing EPW for Ben-Gurion/base');
		expect(factory.records[0].requests).toHaveLength(0);

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: stableSelection
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
	});

	it('keeps a stale same-session surface visible while a GPU pending selection advances controller identity before render publication', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		let baseRequestCount = 0;
		const acceptedVisibleAtMsByRequest = new Map<number, number>();

		function createBaseController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},

				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},

				async requestSelection(request) {
					baseRequestCount += 1;
					state = {
						...state,
						analysis: request.sessionConfig.base,
						surfaceIdentity: {
							requestId: baseRequestCount,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'updated-selection',
							pendingRenderUpdateStartedAt: baseRequestCount * 100,
							acceptedGpuResidentOutput: null
						},
						loading: true,
						error: null,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender: true,
						pendingRenderUpdateStartedAt: baseRequestCount * 100,
						renderSurfaceDiagnostics: {
							gpuResidentCopyStatus: 'pending',
							gpuResidentCopyRequestId: baseRequestCount
						},
						ready: true,
						renderReady: false,
						awaitingGpuSurface: true
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},

				async handleRenderSurfaceDiagnostics(diagnostics) {
					const renderSurfaceDiagnostics = {
						...state.renderSurfaceDiagnostics,
						...diagnostics
					};
					const gpuRenderReady =
						state.renderTransport === 'compute-buffer-selected-hour' &&
						renderSurfaceDiagnostics.gpuResidentCopyStatus === 'complete' &&
						renderSurfaceDiagnostics.utciSurfaceSource === 'compute-buffer-selected-hour';
					const gpuAcceptedVisibleAtMs = 6000 + baseRequestCount;
					const acceptedVisibleAtMs = gpuRenderReady
						? gpuAcceptedVisibleAtMs
						: state.acceptedVisibleAtMs;
					if (gpuRenderReady) {
						acceptedVisibleAtMsByRequest.set(baseRequestCount, gpuAcceptedVisibleAtMs);
					}
					state = {
						...state,
						renderSurfaceDiagnostics,
						acceptedVisibleSurface: gpuRenderReady
							? {
									requestId: baseRequestCount,
									selectionKey: state.surfaceIdentity?.selectionKey ?? 'selection',
									visibleAtMs: gpuAcceptedVisibleAtMs
								}
							: state.acceptedVisibleSurface,
						acceptedRequestId: gpuRenderReady
							? baseRequestCount
							: state.acceptedRequestId,
						acceptedSelectionKey: gpuRenderReady
							? state.surfaceIdentity?.selectionKey
							: state.acceptedSelectionKey,
						acceptedVisibleAtMs,
						loading: gpuRenderReady ? false : state.loading,
						renderReady: gpuRenderReady ? true : state.renderReady,
						awaitingGpuSurface: gpuRenderReady ? false : state.awaitingGpuSurface,
						pendingRenderUpdateStartedAt: gpuRenderReady
							? undefined
							: state.pendingRenderUpdateStartedAt
					};
					emit();
					return;
				},

				dispose() {
					listeners.clear();
				}
			};
		}

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		let createdControllers = 0;
		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 1 ? createBaseController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();
		expect(host.getState().base.loading).toBe(true);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity).toBeNull();

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 1
		});
		await host.flush();

		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().acceptedRequestId).toBe(1);
		expect(host.getState().acceptedSelectionKey).toBe('Ben-Gurion/base|7|12|180');
		expect(host.getState().acceptedVisibleAtMs).toBe(acceptedVisibleAtMsByRequest.get(1));
		expect(host.getState().primaryAcceptedVisibleSurface).toEqual({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			visibleAtMs: acceptedVisibleAtMsByRequest.get(1)
		});
		expect(host.getState().baseAcceptedVisibleSurface).toEqual(
			host.getState().primaryAcceptedVisibleSurface
		);

		host.setRouteInputs(
			makeBaseInputs({
				utciSurfaceBackend: 'gpuNative',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 12,
					timeIndex: 36,
					selectionKey: 'Ben-Gurion/base|1|12|36'
				}
			})
		);

		await host.flush();

		expect(host.getState().base.loading).toBe(true);
		expect(host.getState().base.surfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|1|12|36'
		);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().acceptedRequestId).toBe(1);
		expect(host.getState().acceptedSelectionKey).toBe('Ben-Gurion/base|7|12|180');
		expect(host.getState().acceptedVisibleAtMs).toBe(acceptedVisibleAtMsByRequest.get(1));
		expect(host.getState().baseAcceptedVisibleSurface).toEqual({
			requestId: 1,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			visibleAtMs: acceptedVisibleAtMsByRequest.get(1)
		});
		expect(host.getState().baseDisplayAnalysis).toBe(baseAnalysis);

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'compute-buffer-selected-hour',
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 2
		});
		await host.flush();

		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|1|12|36'
		);
		expect(host.getState().acceptedRequestId).toBe(2);
		expect(host.getState().acceptedSelectionKey).toBe('Ben-Gurion/base|1|12|36');
		expect(host.getState().acceptedVisibleAtMs).toBe(acceptedVisibleAtMsByRequest.get(2));
		expect(host.getState().primaryAcceptedVisibleSurface).toEqual({
			requestId: 2,
			selectionKey: 'Ben-Gurion/base|1|12|36',
			visibleAtMs: acceptedVisibleAtMsByRequest.get(2)
		});
		expect(host.getState().baseAcceptedVisibleSurface).toEqual(
			host.getState().primaryAcceptedVisibleSurface
		);
	});

	it('keeps stale visible output while exposing the pending base render context for a color-mode replacement', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		let baseRequestCount = 0;

		function createBaseController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},

				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},

				async requestSelection(request) {
					baseRequestCount += 1;
					state = {
						...state,
						analysis: request.sessionConfig.base,
						surfaceIdentity: {
							requestId: baseRequestCount,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: baseRequestCount * 100,
							acceptedGpuResidentOutput: null
						},
						loading: true,
						error: null,
						renderTransport: 'cpu-uploaded-selected-hour',
						sameDeviceForComputeAndRender: false,
						pendingRenderUpdateStartedAt: baseRequestCount * 100,
						renderSurfaceDiagnostics: {},
						ready: true,
						renderReady: true,
						awaitingGpuSurface: false
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},

				async handleRenderSurfaceDiagnostics(diagnostics) {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							...diagnostics
						},
						loading: false,
						renderReady: true,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},

				dispose() {
					listeners.clear();
				}
			};
		}

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		let createdControllers = 0;
		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 1 ? createBaseController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'discrete',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();
		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'normalized',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(host.getState().base.loading).toBe(true);
		expect(host.getState().base.surfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().baseDisplayAnalysis).toBe(baseAnalysis);
		expect(host.getState().baseRenderContext?.colorMode).toBe('normalized');
		expect(host.getState().baseRenderContext?.monthIndex).toBe(7);
		expect(host.getState().baseRenderContext?.hourIndex).toBe(12);

		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 2,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().baseReady).toBe(true);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(baseRequestCount).toBe(2);
	});

	it('keeps the published base display while exposing the pending base render context before replacement publish completes', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const publishedSelectedHourAnalysis = createSingleHourAnalysis(
			'base-selected-hour-published'
		);
		const pendingSelectedHourAnalysis = createSingleHourAnalysis(
			'base-selected-hour-pending'
		);
		let createdControllers = 0;
		let baseRequestCount = 0;

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		function createBaseController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					baseRequestCount += 1;
					state = {
						...state,
						analysis:
							baseRequestCount === 1
								? publishedSelectedHourAnalysis
								: pendingSelectedHourAnalysis,
						surfaceIdentity: {
							requestId: baseRequestCount,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: baseRequestCount * 100,
							acceptedGpuResidentOutput: null
						},
						loading: true,
						error: null,
						renderTransport: 'cpu-uploaded-selected-hour',
						sameDeviceForComputeAndRender: false,
						pendingRenderUpdateStartedAt: baseRequestCount * 100,
						renderSurfaceDiagnostics: {},
						ready: true,
						renderReady: true,
						awaitingGpuSurface: false
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics(diagnostics) {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							...diagnostics
						},
						loading: false,
						renderReady: true,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 1 ? createBaseController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'discrete',
				baseAnalysis,
				baseModel,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();
		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().base.analysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().baseDisplayAnalysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().baseRenderContext?.analysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().baseRenderContext?.colorMode).toBe('discrete');

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'normalized',
				baseAnalysis,
				baseModel,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(host.getState().base.analysis).toBe(pendingSelectedHourAnalysis);
		expect(host.getState().baseDisplayAnalysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().baseRenderContext?.analysis).toBe(pendingSelectedHourAnalysis);
		expect(host.getState().baseRenderContext?.colorMode).toBe('normalized');
		expect(host.getState().baseRenderContext?.monthIndex).toBe(7);
		expect(host.getState().baseRenderContext?.hourIndex).toBe(12);
	});

	it('keeps stale comparison output visible while exposing the pending comparison render context for a color-mode replacement', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter',
			baseMin: 5,
			baseMax: 22
		});
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;
		const publishedSelectedHourAnalysis = createSingleHourAnalysis(
			'comparison-selected-hour-published'
		);
		const pendingSelectedHourAnalysis = createSingleHourAnalysis(
			'comparison-selected-hour-pending'
		);
		let createdControllers = 0;
		let comparisonRequestCount = 0;

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		function createComparisonController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},

				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},

				async requestSelection(request) {
					comparisonRequestCount += 1;
					state = {
						...state,
						analysis:
							comparisonRequestCount === 1
								? publishedSelectedHourAnalysis
								: pendingSelectedHourAnalysis,
						surfaceIdentity: {
							requestId: comparisonRequestCount,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: comparisonRequestCount * 100,
							acceptedGpuResidentOutput: null
						},
						loading: true,
						error: null,
						renderTransport: 'cpu-uploaded-selected-hour',
						sameDeviceForComputeAndRender: false,
						pendingRenderUpdateStartedAt: comparisonRequestCount * 100,
						renderSurfaceDiagnostics: {},
						ready: true,
						renderReady: true,
						awaitingGpuSurface: false
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},

				async handleRenderSurfaceDiagnostics(diagnostics) {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							...diagnostics
						},
						loading: false,
						renderReady: true,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},

				dispose() {
					listeners.clear();
				}
			};
		}

		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 2 ? createComparisonController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeComparisonInputs({
				colorMode: 'discrete',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();
		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().comparisonReady).toBe(true);
		expect(host.getState().comparison.analysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().comparisonDisplayAnalysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().comparisonRenderContext?.analysis).toBe(
			publishedSelectedHourAnalysis
		);
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);

		host.setRouteInputs(
			makeComparisonInputs({
				colorMode: 'normalized',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		expect(host.getState().comparison.loading).toBe(true);
		expect(host.getState().comparison.analysis).toBe(pendingSelectedHourAnalysis);
		expect(host.getState().comparison.surfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().comparisonDisplayAnalysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().comparisonRenderContext?.analysis).toBe(
			pendingSelectedHourAnalysis
		);
		expect(host.getState().comparisonRenderContext?.colorMode).toBe('normalized');
		expect(host.getState().comparisonRenderContext?.monthIndex).toBe(7);
		expect(host.getState().comparisonRenderContext?.hourIndex).toBe(12);

		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 2,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().comparisonReady).toBe(true);
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(comparisonRequestCount).toBe(2);
	});

	it('keeps pending scene identities coherent with pending render contexts while published surfaces remain visible', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter',
			baseMin: 5,
			baseMax: 22
		});
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;
		let createdControllers = 0;
		let baseRequestCount = 0;
		let comparisonRequestCount = 0;

		function createPendingController(params: {
			getRequestCount(): number;
			setRequestCount(value: number): void;
		}): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					const nextRequestCount = params.getRequestCount() + 1;
					params.setRequestCount(nextRequestCount);
					state = {
						...state,
						analysis: request.sessionConfig.base,
						surfaceIdentity: {
							requestId: nextRequestCount,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: nextRequestCount * 100,
							acceptedGpuResidentOutput: null
						},
						loading: true,
						error: null,
						renderTransport: 'cpu-uploaded-selected-hour',
						sameDeviceForComputeAndRender: false,
						pendingRenderUpdateStartedAt: nextRequestCount * 100,
						renderSurfaceDiagnostics: {},
						ready: true,
						renderReady: true,
						awaitingGpuSurface: false
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics(diagnostics) {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							...diagnostics
						},
						loading: false,
						renderReady: true,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				if (createdControllers === 1) {
					return createPendingController({
						getRequestCount: () => baseRequestCount,
						setRequestCount: (value) => {
							baseRequestCount = value;
						}
					});
				}
				if (createdControllers === 2) {
					return createPendingController({
						getRequestCount: () => comparisonRequestCount,
						setRequestCount: (value) => {
							comparisonRequestCount = value;
						}
					});
				}
				return createPendingController({
					getRequestCount: () => 0,
					setRequestCount: () => undefined
				});
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeComparisonInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();
		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);

		host.setRouteInputs(
			makeComparisonInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 9,
					timeIndex: 33,
					selectionKey: 'Ben-Gurion/base|1|9|33'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().comparisonSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|7|12|180'
		);
		expect(host.getState().baseSceneSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|1|9|33'
		);
		expect(host.getState().comparisonSceneSurfaceIdentity?.selectionKey).toBe(
			'Ben-Gurion/base|1|9|33'
		);
		expect(host.getState().baseRenderContext).toMatchObject({
			selectionKey: 'Ben-Gurion/base|1|9|33',
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 33
		});
		expect(host.getState().comparisonRenderContext).toMatchObject({
			selectionKey: 'Ben-Gurion/base|1|9|33',
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 33
		});
	});

	it('retries the same replacement selection after a current-selection rejection without an explicit reason', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const publishedSelectedHourAnalysis = createSingleHourAnalysis(
			'base-selected-hour-published'
		);
		const baseModel = {} as Group;
		let createdControllers = 0;
		let baseRequestCount = 0;

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		function createBaseController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					baseRequestCount += 1;
					if (baseRequestCount === 1) {
						state = {
							...state,
							analysis: publishedSelectedHourAnalysis,
							surfaceIdentity: {
								requestId: 1,
								monthIndex: request.monthIndex,
								hourIndex: request.hourIndex,
								timeIndex: request.timeIndex,
								selectionKey: request.selectionKey ?? 'selection',
								pendingRenderUpdateStartedAt: 100,
								acceptedGpuResidentOutput: null
							},
							loading: true,
							error: null,
							renderTransport: 'cpu-uploaded-selected-hour',
							sameDeviceForComputeAndRender: false,
							pendingRenderUpdateStartedAt: 100,
							renderSurfaceDiagnostics: {},
							ready: true,
							renderReady: true,
							awaitingGpuSurface: false
						};
						emit();
						return { accepted: true, state: cloneControllerState(state) };
					}

					return { accepted: false, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics(diagnostics) {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							...diagnostics
						},
						loading: false,
						renderReady: true,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 1 ? createBaseController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'discrete',
				baseAnalysis,
				baseModel,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();
		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'normalized',
				baseAnalysis,
				baseModel,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseRenderContext?.analysis).toBe(publishedSelectedHourAnalysis);
		expect(host.getState().baseRenderContext?.colorMode).toBe('discrete');

		host.setRouteInputs(
			makeBaseInputs({
				colorMode: 'normalized',
				baseAnalysis,
				baseModel,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();

		expect(baseRequestCount).toBe(3);
	});

	it('keeps the published unified range frozen while same-selection replacement render contexts advance', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison-winter',
			sourceAnalysisId: 'Ben-Gurion/comparison/winter',
			baseMin: 5,
			baseMax: 22
		});
		const comparisonModel = {} as Group;
		const comparisonRendererDevice = {
			label: 'comparison-renderer-stable'
		} as unknown as GPUDevice;
		let createdControllers = 0;
		let baseRequestCount = 0;
		let comparisonRequestCount = 0;

		function createPendingController(params: {
			getRequestCount(): number;
			setRequestCount(value: number): void;
		}): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					const nextRequestCount = params.getRequestCount() + 1;
					params.setRequestCount(nextRequestCount);
					state = {
						...state,
						analysis: request.sessionConfig.base,
						surfaceIdentity: {
							requestId: nextRequestCount,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey: request.selectionKey ?? 'selection',
							pendingRenderUpdateStartedAt: nextRequestCount * 100,
							acceptedGpuResidentOutput: null
						},
						loading: true,
						error: null,
						renderTransport: 'cpu-uploaded-selected-hour',
						sameDeviceForComputeAndRender: false,
						pendingRenderUpdateStartedAt: nextRequestCount * 100,
						renderSurfaceDiagnostics: {},
						ready: true,
						renderReady: true,
						awaitingGpuSurface: false
					};
					emit();
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics(diagnostics) {
					state = {
						...state,
						renderSurfaceDiagnostics: {
							...state.renderSurfaceDiagnostics,
							...diagnostics
						},
						loading: false,
						renderReady: true,
						pendingRenderUpdateStartedAt: undefined
					};
					emit();
				},
				dispose() {
					listeners.clear();
				}
			};
		}

		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				if (createdControllers === 1) {
					return createPendingController({
						getRequestCount: () => baseRequestCount,
						setRequestCount: (value) => {
							baseRequestCount = value;
						}
					});
				}
				if (createdControllers === 2) {
					return createPendingController({
						getRequestCount: () => comparisonRequestCount,
						setRequestCount: (value) => {
							comparisonRequestCount = value;
						}
					});
				}
				return {
					getState() {
						return createInitialControllerState();
					},
					subscribe() {
						return () => undefined;
					},
					async requestSelection() {
						return { accepted: true, state: createInitialControllerState() };
					},
					async handleRenderSurfaceDiagnostics() {
						return;
					},
					dispose() {
						return;
					}
				};
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeComparisonInputs({
				colorMode: 'discrete',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();
		host.handleBaseSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		host.handleComparisonSurfaceDiagnostics({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 1,
			cpuPublishMonthIndex: 7,
			cpuPublishHourIndex: 12,
			cpuPublishTimeIndex: 180,
			cpuPublishSelectionKey: 'Ben-Gurion/base|7|12|180'
		});
		await host.flush();

		expect(host.getState().liveUnifiedRange).toEqual({
			utciMin: 17,
			utciMax: 42
		});
		expect(host.getState().baseRenderContext?.colorMode).toBe('discrete');
		expect(host.getState().comparisonRenderContext?.colorMode).toBe('discrete');

		host.setRouteInputs(
			makeComparisonInputs({
				colorMode: 'normalized',
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				},
				comparison: {
					sourceAnalysis: comparisonAnalysis,
					model: comparisonModel,
					rendererDevice: comparisonRendererDevice
				}
			})
		);
		await host.flush();

		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().comparisonReady).toBe(false);
		expect(host.getState().liveUnifiedRange).toEqual({
			utciMin: 17,
			utciMax: 42
		});
		expect(host.getState().baseRenderContext).toMatchObject({
			colorMode: 'normalized',
			monthIndex: 7,
			hourIndex: 12
		});
		expect(host.getState().comparisonRenderContext).toMatchObject({
			colorMode: 'normalized',
			monthIndex: 7,
			hourIndex: 12
		});
	});

	it('retries the same selection after a controller-level failure and keeps the stale surface hidden while exposing the error state', async () => {
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const baseRendererDevice = { label: 'base-renderer-stable' } as unknown as GPUDevice;
		let baseRequestCount = 0;

		function createBaseController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();

			function emit(): void {
				const snapshot = cloneControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneControllerState(state);
				},

				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},

				async requestSelection(request) {
					baseRequestCount += 1;
					if (baseRequestCount === 1) {
						state = {
							...state,
							analysis: request.sessionConfig.base,
							surfaceIdentity: {
								requestId: 1,
								monthIndex: request.monthIndex,
								hourIndex: request.hourIndex,
								timeIndex: request.timeIndex,
								selectionKey: request.selectionKey ?? 'initial-selection',
								pendingRenderUpdateStartedAt: undefined,
								acceptedGpuResidentOutput: null
							},
							loading: false,
							error: null,
							renderTransport: 'cpu-uploaded-selected-hour',
							sameDeviceForComputeAndRender: false,
							pendingRenderUpdateStartedAt: undefined,
							renderSurfaceDiagnostics: {},
							ready: true,
							renderReady: true,
							awaitingGpuSurface: false
						};
						emit();
						return { accepted: true, state: cloneControllerState(state) };
					}

					state = {
						...state,
						loading: true,
						error: null
					};
					emit();

					state = {
						...state,
						loading: false,
						error: 'Compute failed for current selection'
					};
					emit();

					return { accepted: false, state: cloneControllerState(state) };
				},

				async handleRenderSurfaceDiagnostics() {
					return;
				},

				dispose() {
					listeners.clear();
				}
			};
		}

		function createIdleController(): LiveSelectedHourController {
			let state = createInitialControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			return {
				getState() {
					return cloneControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection() {
					return { accepted: true, state: cloneControllerState(state) };
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialControllerState();
				}
			};
		}

		let createdControllers = 0;
		const host = createLiveSelectedHourRouteHost({
			createController: () => {
				createdControllers += 1;
				return createdControllers === 1 ? createBaseController() : createIdleController();
			},
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 7,
					hourIndex: 12,
					timeIndex: 180,
					selectionKey: 'Ben-Gurion/base|7|12|180'
				}
			})
		);
		await host.flush();
		expect(host.getState().baseReady).toBe(true);

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 12,
					timeIndex: 36,
					selectionKey: 'Ben-Gurion/base|1|12|36'
				}
			})
		);
		await host.flush();

		expect(host.getState().base.loading).toBe(false);
		expect(host.getState().base.error).toBe('Compute failed for current selection');
		expect(host.getState().baseReady).toBe(false);
		expect(host.getState().baseSurfaceIdentity).toBeNull();
		expect(baseRequestCount).toBe(2);

		host.setRouteInputs(
			makeBaseInputs({
				baseAnalysis,
				baseModel,
				rendererDevice: baseRendererDevice,
				selection: {
					monthIndex: 1,
					hourIndex: 12,
					timeIndex: 36,
					selectionKey: 'Ben-Gurion/base|1|12|36'
				}
			})
		);
		await host.flush();

		expect(baseRequestCount).toBe(3);
	});
});
