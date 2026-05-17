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
			maxTotalMs: 1
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
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(777)
			.mockReturnValueOnce(999);
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
			routePendingSurfaceExposedAtMs: 777,
			routeProjectedAtMs: 777
		});

		nowSpy.mockRestore();
	});
});
