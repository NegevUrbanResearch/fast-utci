import { describe, expect, it, vi } from 'vitest';
import type { Group } from 'three';

import {
	buildMainRouteLiveSelectedHourDiagnostics,
	createMainRouteRenderPublicationProjectionTracker,
	resolveMainRouteLiveMetricSelection,
	releaseBaseAcceptedGpuResidentOutput,
	releaseComparisonAcceptedGpuResidentOutput,
	type MainRouteAcceptedGpuResidentOutputReleaseParams,
	type MainRouteLiveSelectedHourDiagnosticsParams
} from '../../src/routes/main/liveSelectedHour';
import type { Analysis } from '$lib/types/analysis';
import {
	createLiveSelectedHourRouteHost,
	type LiveSelectedHourRouteHost
} from '$lib/compute/selected-hour/liveSelectedHourRouteHost';
import {
	createFullDayAnalysis,
	createLiveRouteInputs,
	createMetricRecordingControllerFactory
} from '../compute/live-selected-hour-route-host.test-support';

function createReleaseParams(): MainRouteAcceptedGpuResidentOutputReleaseParams {
	return {
		controllerIdentity: 'controller-a',
		controllerInstanceId: 7,
		requestId: 11,
		monthIndex: 2,
		timeIndex: 51,
		reason: 'copy-complete'
	};
}

function createDiagnosticsParams(): MainRouteLiveSelectedHourDiagnosticsParams {
	return {
		enabled: true,
		utciOnDemand: 'f32',
		utciRenderRequested: 'gpu',
		utciRenderResolved: 'gpuNative',
		rendererBackend: 'webgpu',
		rendererRequiredLimits: undefined,
		rendererDeviceLimits: undefined,
		liveRouteState: {
			base: {
				renderSurfaceDiagnostics: {
					utciSurfaceSource: 'compute-buffer-selected-hour',
					selectedHourTransferCount: 0,
					dataTextureBuildCount: 0,
					gpuResidentCopyStatus: 'complete',
					gpuResidentCopyRequestId: 11
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				visibleSelectedHourReadbackCount: 0,
				readbackInstrumentation: 'not-instrumented',
				selectedHourReadbackReasons: [],
				selectedHourReadbackReasonCounts: {}
			},
			comparison: {
				renderSurfaceDiagnostics: {
					utciSurfaceSource: 'compute-buffer-selected-hour',
					selectedHourTransferCount: 0,
					dataTextureBuildCount: 0,
					gpuResidentCopyStatus: 'complete',
					gpuResidentCopyRequestId: 13
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				selectedHourReadbackReasons: [],
				selectedHourReadbackReasonCounts: {}
			},
			baseSurfaceIdentity: {
				controllerIdentity: 'controller-a',
				controllerInstanceId: 1,
				requestId: 11,
				monthIndex: 7,
				hourIndex: 12,
				timeIndex: 180,
				selectionKey: 'analysis|7|12',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			},
			baseSceneSurfaceIdentity: {
				controllerIdentity: 'controller-a',
				controllerInstanceId: 1,
				requestId: 11,
				monthIndex: 7,
				hourIndex: 12,
				timeIndex: 180,
				selectionKey: 'analysis|7|12',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			},
			comparisonSurfaceIdentity: {
				controllerIdentity: 'controller-b',
				controllerInstanceId: 1,
				requestId: 13,
				monthIndex: 7,
				hourIndex: 12,
				timeIndex: 180,
				selectionKey: 'comparison|7|12',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			}
		} as unknown as MainRouteLiveSelectedHourDiagnosticsParams['liveRouteState'],
		lastBaseGpuResidentCopyFailure: undefined,
		baseLiveReady: true,
		comparisonLiveReady: true,
		selectedMonthIndex: 7,
		selectedHourIndex: 12,
		selectedTimeIndex: 180,
		baseSceneRenderContextTimeIndex: 180,
		baseAcceptedUtciRange: { min: -10, max: 42 },
		tooltipInteraction: {
			enabled: true,
			disabledByQuery: false,
			slowThresholdMs: 8,
			hoverAttemptCount: 3,
			suppressedHoverCount: 0,
			throttledHoverCount: 0,
			hoverSampleCount: 3,
			sampleCount: 1,
			hitCount: 1,
			missCount: 0,
			overBudgetCount: 0,
			lastOutcome: 'hit',
			lastRaycastMs: 0,
			maxRaycastMs: 0,
			lastNearestPointMs: 0,
			maxNearestPointMs: 0,
			lastTotalMs: 1,
			maxTotalMs: 1,
			lastResolutionPath: 'plane-cell',
			planeCellPathCount: 1,
			meshRaycastPathCount: 0,
			directCellHitCount: 1,
			directCellMissCount: 0,
			nearestScanFallbackCount: 0,
			metricPointReadbackCount: 0,
			metricPointReadbackBytes: 0,
			metricPointReadbackLastBytes: null,
			metricPointReadbackCacheEntries: 0,
			metricPointReadbackCacheHitCount: 0,
			metricPointReadbackCacheMissCount: 0,
			metricPointReadbackLastLatencyMs: null,
			metricPointReadbackMaxLatencyMs: 0
		},
		cameraInteraction: {
			slowThresholdMs: 20,
			sampleCount: 2,
			wheelEventCount: 1,
			overBudgetCount: 0,
			lastFrameMs: 16,
			maxFrameMs: 17,
			p95FrameMs: 17
		}
	};
}

describe('main route live selected-hour helper', () => {
	it('builds a month-only live Shading Index selection for the main route', () => {
		const analysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base'
		});
		const selection = resolveMainRouteLiveMetricSelection({
			analysis,
			analysisId: 'Ben-Gurion/base',
			metricType: 'shading_index',
			currentMonth: 7,
			currentHour: 12,
			rendererBackend: 'webgpu',
			rendererDevice: { label: 'renderer' } as unknown as GPUDevice,
			utciSurfaceBackend: 'gpuNative'
		});

		expect(selection).toMatchObject({
			useLiveMetricOnMainRoute: true,
			liveRouteEnabled: true,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			selectedTimeIndex: 168,
			selectionKey: 'Ben-Gurion/base|shading_index|7',
			fixedTimePickerMode: 'month',
			liveMetricUnavailableError: null
		});

		const selectionAfterHourChange = resolveMainRouteLiveMetricSelection({
			analysis,
			analysisId: 'Ben-Gurion/base',
			metricType: 'shading_index',
			currentMonth: 7,
			currentHour: 17,
			rendererBackend: 'webgpu',
			rendererDevice: { label: 'renderer' } as unknown as GPUDevice,
			utciSurfaceBackend: 'gpuNative'
		});

		expect(selectionAfterHourChange).toMatchObject({
			selectedHourIndex: 0,
			selectedTimeIndex: 168,
			selectionKey: 'Ben-Gurion/base|shading_index|7',
			fixedTimePickerMode: 'month'
		});
	});

	it('keeps Shading Index on the live route with an explicit unavailable state instead of a data fallback', () => {
		const selection = resolveMainRouteLiveMetricSelection({
			analysis: createFullDayAnalysis({
				label: 'base',
				sourceAnalysisId: 'Ben-Gurion/base'
			}),
			analysisId: 'Ben-Gurion/base',
			metricType: 'shading_index',
			currentMonth: 7,
			currentHour: 12,
			rendererBackend: 'unknown',
			rendererDevice: undefined,
			utciSurfaceBackend: 'dataTexture'
		});

		expect(selection.useLiveMetricOnMainRoute).toBe(true);
		expect(selection.liveRouteEnabled).toBe(false);
		expect(selection.liveMetricUnavailableError).toMatch(/requires WebGPU/i);
	});

	it('advertises live shading metric availability on the main route without pretending baked shading metadata exists', () => {
		const analysis = createFullDayAnalysis({
			label: 'innovation-district',
			sourceAnalysisId: 'Innovation-District/innovation_district_webgpu'
		});
		analysis.metadata.has_shading_index = false;
		analysis.metadata.shading_index_range = undefined;

		const selection = resolveMainRouteLiveMetricSelection({
			analysis,
			analysisId: 'Innovation-District/innovation_district_webgpu',
			metricType: 'utci',
			currentMonth: 7,
			currentHour: 12,
			rendererBackend: 'webgpu',
			rendererDevice: { label: 'renderer' } as unknown as GPUDevice,
			utciSurfaceBackend: 'gpuNative'
		});

		expect(selection.liveShadingMetricAvailable).toBe(true);
		expect(analysis.metadata.has_shading_index).toBe(false);
	});

	it('forwards metric type into live route controller requests and invalidates same selection on metric switch', async () => {
		const factory = createMetricRecordingControllerFactory();
		const host = createLiveSelectedHourRouteHost({
			createController: factory.createController,
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const baseModel = {} as Group;
		const rendererDevice = { label: 'renderer' } as unknown as GPUDevice;
		const selectionKey = 'Ben-Gurion/base|7';

		host.setRouteInputs(
			createLiveRouteInputs({
				metricType: 'utci',
				baseAnalysis,
				baseModel,
				rendererDevice,
				selectionKey
			})
		);
		await host.flush();
		host.setRouteInputs(
			createLiveRouteInputs({
				metricType: 'shading_index',
				baseAnalysis,
				baseModel,
				rendererDevice,
				selectionKey
			})
		);
		await host.flush();

		expect(factory.records[0].requests).toHaveLength(2);
		expect(factory.records[0].requests.map((request) => request.metricType)).toEqual([
			'utci',
			'shading_index'
		]);
	});

	it('does not compute or apply UTCI unified range overrides for live Shading Index', async () => {
		const factory = createMetricRecordingControllerFactory();
		const host = createLiveSelectedHourRouteHost({
			createController: factory.createController,
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const comparisonAnalysis = createFullDayAnalysis({
			label: 'comparison',
			sourceAnalysisId: 'Ben-Gurion/comparison',
			baseMin: 5,
			baseMax: 20
		});

		host.setRouteInputs(
			createLiveRouteInputs({
				metricType: 'shading_index',
				baseAnalysis,
				comparisonAnalysis,
				selectionKey: 'Ben-Gurion/base|shading_index|7',
				hourIndex: 0,
				timeIndex: 168
			})
		);
		await host.flush();

		const state = host.getState();
		expect(state.liveUnifiedRange).toBeNull();
		expect(state.baseRenderContext?.metricType).toBe('shading_index');
		expect(state.comparisonRenderContext?.metricType).toBe('shading_index');
		expect(state.baseRenderContext?.rangeOverride).toBeNull();
		expect(state.comparisonRenderContext?.rangeOverride).toBeNull();
	});

	it('publishes live Shading Index route/context once the metric-aware render bridge is available', async () => {
		const factory = createMetricRecordingControllerFactory();
		const host = createLiveSelectedHourRouteHost({
			createController: factory.createController,
			resolveEpwUrl: ({ analysisId }) => `/weather/${analysisId ?? 'default'}.epw`
		});
		const baseAnalysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});

		host.setRouteInputs(
			createLiveRouteInputs({
				metricType: 'shading_index',
				baseAnalysis,
				selectionKey: 'Ben-Gurion/base|shading_index|7',
				hourIndex: 0,
				timeIndex: 168
			})
		);
		await host.flush();

		const state = host.getState();
		expect(factory.records[0].requests).toHaveLength(1);
		expect(factory.records[0].requests[0]).toMatchObject({
			metricType: 'shading_index',
			selectionKey: 'Ben-Gurion/base|shading_index|7'
		});
		expect(state.base.renderTransport).toBe('compute-buffer-selected-hour');
		expect(state.baseRenderContext?.metricType).toBe('shading_index');
		expect(state.baseDisplayAnalysis).toBe(baseAnalysis);
		expect(state.baseSceneSurfaceIdentity?.selectionKey).toBe('Ben-Gurion/base|shading_index|7');
		expect(state.baseSurfaceIdentity?.selectionKey).toBe('Ben-Gurion/base|shading_index|7');
		expect(state.baseHasVisibleLiveSurface).toBe(true);
		expect(state.baseReady).toBe(true);
	});

	it('forwards base accepted GPU resident output releases with controller instance ids intact', () => {
		const host = {
			releaseBaseAcceptedGpuResidentOutput: vi.fn(),
			releaseComparisonAcceptedGpuResidentOutput: vi.fn()
		} as unknown as LiveSelectedHourRouteHost;
		const params = createReleaseParams();

		releaseBaseAcceptedGpuResidentOutput(host, params);

		expect(host.releaseBaseAcceptedGpuResidentOutput).toHaveBeenCalledWith(params);
	});

	it('forwards comparison accepted GPU resident output releases with controller instance ids intact', () => {
		const host = {
			releaseBaseAcceptedGpuResidentOutput: vi.fn(),
			releaseComparisonAcceptedGpuResidentOutput: vi.fn()
		} as unknown as LiveSelectedHourRouteHost;
		const params = createReleaseParams();

		releaseComparisonAcceptedGpuResidentOutput(host, params);

		expect(host.releaseComparisonAcceptedGpuResidentOutput).toHaveBeenCalledWith(params);
	});

	it('builds diagnostics without debug-only fields', () => {
		const diagnostics = buildMainRouteLiveSelectedHourDiagnostics(createDiagnosticsParams());

		expect(diagnostics).toMatchObject({
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'compute-buffer-selected-hour',
			baseSelectionKey: 'analysis|7|12'
		});
		expect(diagnostics?.selectedHourRuntimeContract.acceptedRequestId).toBe(11);
		expect(diagnostics?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics?.selectedHourRuntimeContract).toMatchObject({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour'
		});
		const diagnosticsRecord = diagnostics as Record<string, unknown>;
		for (const debugField of [
			'binComparisonEnabled',
			'binComparisonValid',
			'parityMode',
			'pythonReferencePath',
			'loadReferenceFromFs',
			'__onDemandPrototypeDiagnostics__',
			'LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID'
		]) {
			expect(diagnosticsRecord).not.toHaveProperty(debugField);
		}
	});

	it('stamps routeProjectedAtMs once per published request and selection', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(602);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

		const first = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const second = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});
		expect(second?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});

		nowSpy.mockRestore();
	});

	it('stamps a fresh routeProjectedAtMs after controller recreation even when request and selection repeat', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(777)
			.mockReturnValueOnce(877);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;
		params.liveRouteState.baseSurfaceIdentity = {
			controllerIdentity: 'controller-a',
			controllerInstanceId: 1,
			requestId: 11,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'analysis|7|12',
			pendingRenderUpdateStartedAt: undefined,
			acceptedGpuResidentOutput: null
		};

		const first = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const second = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: {
				...params.liveRouteState.baseSceneSurfaceIdentity!,
				controllerInstanceId: 2
			},
			publishedSurfaceIdentity: {
				...params.liveRouteState.baseSurfaceIdentity,
				controllerInstanceId: 2
			},
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});
		expect(second?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 777,
			routeProjectedAtMs: 777
		});

		nowSpy.mockRestore();
	});

	it('preserves routeProjectedAtMs across transient gating loss for the same published surface key', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(602)
			.mockReturnValueOnce(103)
			.mockReturnValueOnce(603);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

		const first = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const transientLoss = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: undefined,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const recovered = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});
		expect(
			transientLoss?.renderPublication?.renderPublicationTimeline
		).toMatchObject({
			routePendingSurfaceExposedAtMs: 501
		});
		expect(
			transientLoss?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBeUndefined();
		expect(recovered?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});

		nowSpy.mockRestore();
	});

	it('stamps routePendingSurfaceExposedAtMs before routeProjectedAtMs for a bootstrap candidate', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(602);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

		const bootstrapOnly = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: null,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const published = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: {
				controllerIdentity: 'controller-a',
				controllerInstanceId: 1,
				requestId: 11,
				monthIndex: 7,
				hourIndex: 12,
				timeIndex: 180,
				selectionKey: 'analysis|7|12',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			},
			publishedSurfaceIdentity: {
				controllerIdentity: 'controller-a',
				controllerInstanceId: 1,
				requestId: 11,
				monthIndex: 7,
				hourIndex: 12,
				timeIndex: 180,
				selectionKey: 'analysis|7|12',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			},
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(bootstrapOnly?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501
		});
		expect(
			bootstrapOnly?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBe(undefined);
		expect(published?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501
		});
		expect(
			published?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBeGreaterThanOrEqual(501);

		nowSpy.mockRestore();
	});

	it('stamps a fresh routePendingSurfaceExposedAtMs when the projected scene surface key changes before publish visibility', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(777)
			.mockReturnValueOnce(877);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

		const first = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: null,
			sceneRenderContextTimeIndex: undefined,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const second = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: {
				...params.liveRouteState.baseSceneSurfaceIdentity!,
				requestId: 12
			},
			publishedSurfaceIdentity: null,
			sceneRenderContextTimeIndex: undefined,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501
		});
		expect(
			first?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBeUndefined();
		expect(
			second?.renderPublication?.renderPublicationTimeline?.routePendingSurfaceExposedAtMs
		).toBeGreaterThan(501);
		expect(
			second?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBeUndefined();

		nowSpy.mockRestore();
	});

	it('does not pair a new pending surface exposure with an old visible surface projection', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(777)
			.mockReturnValueOnce(877)
			.mockReturnValueOnce(103)
			.mockReturnValueOnce(999)
			.mockReturnValueOnce(1099);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

		const visibleSurfaceA = {
			...params.liveRouteState.baseSurfaceIdentity!
		};
		const pendingSurfaceB = {
			...params.liveRouteState.baseSceneSurfaceIdentity!,
			requestId: 12
		};

		const visibleA = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: visibleSurfaceA,
			publishedSurfaceIdentity: visibleSurfaceA,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const pendingBWhileVisibleA = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: pendingSurfaceB,
			publishedSurfaceIdentity: visibleSurfaceA,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const publishedB = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: pendingSurfaceB,
			publishedSurfaceIdentity: pendingSurfaceB,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(visibleA?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});
		expect(
			pendingBWhileVisibleA?.renderPublication?.renderPublicationTimeline
		).toMatchObject({
			routePendingSurfaceExposedAtMs: 777
		});
		expect(
			pendingBWhileVisibleA?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBeUndefined();
		expect(publishedB?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 777,
			routeProjectedAtMs: 999
		});

		nowSpy.mockRestore();
	});

	it('resets pending and projected timeline state across disable and teardown boundaries', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(102)
			.mockReturnValueOnce(103)
			.mockReturnValueOnce(104)
			.mockReturnValueOnce(105)
			.mockReturnValueOnce(777)
			.mockReturnValueOnce(877);
		const params = createDiagnosticsParams();
		params.liveRouteState.base.runtimeDiagnostics = {
			timings: {
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused'
				}
			},
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 0,
				allHoursOutputBytes: 0,
				selectedHourOutputBytes: 0,
				selectedHourOutputBytesHighWatermark: 0,
				trackingScope: 'utci-owned-webgpu-buffers'
			}
		} as NonNullable<typeof params.liveRouteState.base.runtimeDiagnostics>;

		const first = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const disabled = tracker.apply({
			enabled: false,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const tornDown = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: null,
			publishedSurfaceIdentity: null,
			sceneRenderContextTimeIndex: undefined,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const noTimings = tracker.apply({
			enabled: true,
			timings: undefined,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const rebuilt = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			projectedSceneSurfaceIdentity: params.liveRouteState.baseSceneSurfaceIdentity,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 501,
			routeProjectedAtMs: 501
		});
		expect(disabled?.renderPublication?.renderPublicationTimeline).toBeUndefined();
		expect(tornDown?.renderPublication?.renderPublicationTimeline).toBeUndefined();
		expect(noTimings).toBeUndefined();
		expect(rebuilt?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routePendingSurfaceExposedAtMs: 103,
			routeProjectedAtMs: 103
		});

		nowSpy.mockRestore();
	});
});
