import { describe, expect, it, vi } from 'vitest';

import {
	buildMainRouteLiveSelectedHourDiagnostics,
	createMainRouteRenderPublicationProjectionTracker,
	releaseBaseAcceptedGpuResidentOutput,
	releaseComparisonAcceptedGpuResidentOutput,
	type MainRouteAcceptedGpuResidentOutputReleaseParams,
	type MainRouteLiveSelectedHourDiagnosticsParams
} from '../../src/routes/main/liveSelectedHour';
import type { LiveSelectedHourRouteHost } from '$lib/compute/selected-hour/liveSelectedHourRouteHost';

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
		tooltipHoverSampleCount: 3,
		cameraWheelEventCount: 1
	};
}

describe('main route live selected-hour helper', () => {
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
		expect(JSON.stringify(diagnostics)).not.toMatch(
			/\.bin|parity|Python|loadReferenceFromFs|__onDemandPrototypeDiagnostics__|LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/i
		);
	});

	it('stamps routeProjectedAtMs once per published request and selection', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(777);
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
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const second = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 501
		});
		expect(second?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 501
		});

		nowSpy.mockRestore();
	});

	it('stamps a fresh routeProjectedAtMs after controller recreation even when request and selection repeat', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(777);
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
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const second = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			publishedSurfaceIdentity: {
				...params.liveRouteState.baseSurfaceIdentity,
				controllerInstanceId: 2
			},
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 501
		});
		expect(second?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 777
		});

		nowSpy.mockRestore();
	});

	it('preserves routeProjectedAtMs across transient gating loss for the same published surface key', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(777);
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
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const transientLoss = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: undefined,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const recovered = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
			publishedSurfaceIdentity: params.liveRouteState.baseSurfaceIdentity,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});

		expect(first?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 501
		});
		expect(
			transientLoss?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs
		).toBeUndefined();
		expect(recovered?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 501
		});

		nowSpy.mockRestore();
	});

	it('does not stamp routeProjectedAtMs for a bootstrap candidate before the published visible surface exists', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const nowSpy = vi.spyOn(performance, 'now').mockReturnValueOnce(501);
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
			publishedSurfaceIdentity: null,
			sceneRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
			selectedTimeIndex: params.selectedTimeIndex
		});
		const published = tracker.apply({
			enabled: true,
			timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
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

		expect(bootstrapOnly?.renderPublication?.renderPublicationTimeline?.routeProjectedAtMs).toBe(
			undefined
		);
		expect(published?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 501
		});

		nowSpy.mockRestore();
	});
});
