import { describe, expect, it } from 'vitest';
import { buildMainRouteUtciDiagnostics } from '$lib/diagnostics/mainRouteUtciDiagnostics';
import { createRenderPublicationDiagnostics } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

describe('buildMainRouteUtciDiagnostics', () => {
	it('returns undefined when diagnostics are disabled', () => {
		expect(
			buildMainRouteUtciDiagnostics({
				enabled: false,
				utciOnDemand: 'f32',
				utciRenderRequested: 'auto',
				utciRenderResolved: 'gpuNative',
				rendererBackend: 'webgpu',
				baseSurfaceDiagnostics: {},
				comparisonSurfaceDiagnostics: {},
				baseRenderTransport: 'idle',
				comparisonRenderTransport: 'idle',
				baseLiveReady: false,
				comparisonLiveReady: true,
				baseSameDeviceForComputeAndRender: null,
				comparisonSameDeviceForComputeAndRender: null,
				baseSelectedMonthIndex: 7,
				baseSelectedHourIndex: 12,
				baseSelectedTimeIndex: 180
			})
		).toBeUndefined();
	});

	it('builds a gpu-native selected-hour payload without debug parity fields', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			rendererRequiredLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
			rendererDeviceLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0,
				gpuResidentCopyStatus: 'complete',
				gpuResidentCopyRequestId: 3
			},
			comparisonSurfaceDiagnostics: {},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'idle',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'analysis|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			baseColorMode: 'normalized',
			basePointCount: 1234,
			baseMetadataGridSize: 0.5,
			baseRenderContextTimeIndex: 180,
			baseAcceptedUtciRange: { min: 20, max: 41 },
			comparisonSameDeviceForComputeAndRender: null,
			selectedHourReadbackReasons: ['range', 'tooltip'],
			selectedHourReadbackReasonCounts: {
				range: 1,
				tooltip: 1
			}
		});

		expect(diagnostics).toMatchObject({
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			baseRenderTransport: 'compute-buffer-selected-hour',
			baseLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			baseColorMode: 'normalized',
			basePointCount: 1234,
			baseMetadataGridSize: 0.5,
			baseAcceptedUtciRange: { min: 20, max: 41 }
		});
		expect(diagnostics?.selectedHourRuntimeContract.readbackInstrumentation).toBe('not-instrumented');
		expect(diagnostics?.selectedHourRuntimeContract.acceptedRequestId).toBe(3);
		expect(diagnostics?.selectedHourRuntimeContract.sceneRequestId).toBe(3);
		expect(diagnostics?.selectedHourRuntimeContract.requestMatchesScene).toBe(true);
		expect(diagnostics?.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasons).toEqual([
			'range',
			'tooltip'
		]);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasonCounts).toEqual({
			range: 1,
			tooltip: 1
		});
		expect(diagnostics?.selectedHourRuntimeContract.totalSelectedHourReadbackReasonCount).toBe(2);
		expect(JSON.stringify(diagnostics)).not.toMatch(/\.bin|parity|Python|loadReferenceFromFs/i);
	});

	it('includes comparison readback reasons and counts in the main route contract', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0
			},
			comparisonSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0
			},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'compute-buffer-selected-hour',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'base|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'base|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			comparisonSurfaceRequestId: 4,
			comparisonSelectionKey: 'comparison|7|12',
			comparisonSameDeviceForComputeAndRender: true,
			selectedHourReadbackReasons: ['range'],
			selectedHourReadbackReasonCounts: {
				range: 1
			},
			comparisonSelectedHourReadbackReasons: ['comparison', 'tooltip'],
			comparisonSelectedHourReadbackReasonCounts: {
				comparison: 1,
				tooltip: 2
			}
		});

		expect(diagnostics?.selectedHourReadbackReasons).toEqual([
			'range',
			'comparison',
			'tooltip'
		]);
		expect(diagnostics?.selectedHourReadbackReasonCounts).toEqual({
			range: 1,
			comparison: 1,
			tooltip: 2
		});
		expect(diagnostics?.selectedHourRuntimeContract.readbackInstrumentation).toBe('not-instrumented');
		expect(diagnostics?.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasons).toEqual([
			'range',
			'comparison',
			'tooltip'
		]);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasonCounts).toEqual({
			range: 1,
			comparison: 1,
			tooltip: 2
		});
		expect(diagnostics?.selectedHourRuntimeContract.totalSelectedHourReadbackReasonCount).toBe(4);
		expect(JSON.stringify(diagnostics)).not.toMatch(/\.bin|parity|Python|loadReferenceFromFs/i);
	});

	it('publishes a strong visible GPU path when visible readbacks are explicitly instrumented', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0,
				gpuResidentCopyStatus: 'complete',
				gpuResidentCopyRequestId: 3
			},
			comparisonSurfaceDiagnostics: {},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'idle',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'analysis|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			comparisonSameDeviceForComputeAndRender: null,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});

		expect(diagnostics?.selectedHourRuntimeContract).toMatchObject({
			readbackInstrumentation: 'instrumented',
			visibleSelectedHourReadbackCount: 0,
			visibleSelectedHourReadbackCountInstrumented: true,
			strongVisibleGpuPath: true
		});
	});

	it('exposes render publication timings without changing proof fields', () => {
		const renderPublication = createRenderPublicationDiagnostics({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044,
			renderPublicationTimeline: {
				computeCompletedAtMs: 101,
				controllerAcceptedAtMs: 103,
				routePendingSurfaceExposedAtMs: 105,
				routePublishedAtMs: 107,
				routeProjectedAtMs: 109,
				scenePendingSurfaceObservedAtMs: 111,
				sceneSyncAttemptStartedAtMs: 112,
				sceneSyncAttemptToken: 7,
				sceneSurfaceReceivedAtMs: 113,
				publicationEffectStartedAtMs: 127,
				renderSurfaceMeshTrace: {
					action: 'updated',
					totalMs: 9,
					recreateDecision: {
						missingSurface: false,
						notComputeBufferSurface: false,
						analysisIdentityChanged: false,
						layoutCompatible: true
					},
					updateComputeBufferSurfaceMeshMs: 4,
					fallbackDecisionMs: 1,
					applySurfaceMeshStateMs: 2,
					setPostSurfacePendingStorageInitMs: 0.5
				},
				renderStorageReadyAtMs: 131,
				renderStorageWaitTrace: {
					waitStartedAtMs: 127,
					waitFinishedAtMs: 131,
					waitMs: 4,
					readAttemptCount: 2,
					frameWaitCount: 1,
					deviceAvailableCount: 2,
					backendEntryAvailableCount: 1,
					bufferAvailableCount: 1,
					firstDeviceAtMs: 127.5,
					firstBackendEntryAtMs: 130.5,
					firstBufferAtMs: 130.5,
					lastReadState: {
						deviceAvailable: true,
						backendEntryAvailable: true,
						bufferAvailable: true
					},
					samples: [
						{
							atMs: 127.5,
							deviceAvailable: true,
							backendEntryAvailable: false,
							bufferAvailable: false
						},
						{
							atMs: 130.5,
							deviceAvailable: true,
							backendEntryAvailable: true,
							bufferAvailable: true
						}
					]
				},
				sceneSyncCompletedAtMs: 137,
				sceneSyncResetHistory: [
					{
						resetAtMs: 108,
						resetReason: 'fallback-cpu-surface',
						invalidateActiveRun: false,
						previousCopyRunToken: 4,
						nextCopyRunToken: 4,
						previousSyncRunKey: 'old-sync'
					}
				],
				sceneSyncActiveWindowResetHistory: [
					{
						resetAtMs: 111.5,
						resetReason: 'fallback-cpu-surface',
						invalidateActiveRun: false,
						previousCopyRunToken: 6,
						nextCopyRunToken: 6,
						previousSyncRunKey: 'current-sync'
					}
				]
			}
		});
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0,
				gpuResidentCopyStatus: 'complete',
				gpuResidentCopyRequestId: 3
			},
			comparisonSurfaceDiagnostics: {},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'idle',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'analysis|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			comparisonSameDeviceForComputeAndRender: null,
			timings: {
				oneHourDispatchMs: 12.5,
				renderPublication
			}
		});

		renderPublication.renderPublicationPhase = 'unknown';
		renderPublication.renderPublicationPointCount = 12;
		renderPublication.renderPublicationTimeline!.sceneSyncCompletedAtMs = 999;
		renderPublication.renderPublicationTimeline!.renderSurfaceMeshTrace!.totalMs = 999;
		renderPublication.renderPublicationTimeline!.renderSurfaceMeshTrace!.recreateDecision!.layoutCompatible =
			false;
		renderPublication.renderPublicationTimeline!.renderStorageWaitTrace!.lastReadState.bufferAvailable =
			false;
		renderPublication.renderPublicationTimeline!.renderStorageWaitTrace!.samples[0].bufferAvailable =
			true;
		renderPublication.renderPublicationTimeline!.sceneSyncResetHistory![0].resetReason =
			'mutated';
		renderPublication.renderPublicationTimeline!.sceneSyncActiveWindowResetHistory![0].resetReason =
			'mutated';

		expect(diagnostics?.timings?.renderPublication).toMatchObject({
			renderPublicationVersion: 1,
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044,
			renderPublicationTimeline: {
				computeCompletedAtMs: 101,
				controllerAcceptedAtMs: 103,
				routePendingSurfaceExposedAtMs: 105,
				routePublishedAtMs: 107,
				routeProjectedAtMs: 109,
				scenePendingSurfaceObservedAtMs: 111,
				sceneSyncAttemptStartedAtMs: 112,
				sceneSyncAttemptToken: 7,
				sceneSurfaceReceivedAtMs: 113,
				publicationEffectStartedAtMs: 127,
				renderSurfaceMeshTrace: {
					action: 'updated',
					totalMs: 9,
					recreateDecision: {
						missingSurface: false,
						notComputeBufferSurface: false,
						analysisIdentityChanged: false,
						layoutCompatible: true
					},
					updateComputeBufferSurfaceMeshMs: 4,
					fallbackDecisionMs: 1,
					applySurfaceMeshStateMs: 2,
					setPostSurfacePendingStorageInitMs: 0.5
				},
				renderStorageReadyAtMs: 131,
				renderStorageWaitTrace: {
					waitStartedAtMs: 127,
					waitFinishedAtMs: 131,
					waitMs: 4,
					readAttemptCount: 2,
					frameWaitCount: 1,
					deviceAvailableCount: 2,
					backendEntryAvailableCount: 1,
					bufferAvailableCount: 1,
					firstDeviceAtMs: 127.5,
					firstBackendEntryAtMs: 130.5,
					firstBufferAtMs: 130.5,
					lastReadState: {
						deviceAvailable: true,
						backendEntryAvailable: true,
						bufferAvailable: true
					},
					samples: [
						{
							atMs: 127.5,
							deviceAvailable: true,
							backendEntryAvailable: false,
							bufferAvailable: false
						},
						{
							atMs: 130.5,
							deviceAvailable: true,
							backendEntryAvailable: true,
							bufferAvailable: true
						}
					]
				},
				sceneSyncCompletedAtMs: 137,
				sceneSyncResetHistory: [
					{
						resetAtMs: 108,
						resetReason: 'fallback-cpu-surface',
						invalidateActiveRun: false,
						previousCopyRunToken: 4,
						nextCopyRunToken: 4,
						previousSyncRunKey: 'old-sync'
					}
				],
				sceneSyncActiveWindowResetHistory: [
					{
						resetAtMs: 111.5,
						resetReason: 'fallback-cpu-surface',
						invalidateActiveRun: false,
						previousCopyRunToken: 6,
						nextCopyRunToken: 6,
						previousSyncRunKey: 'current-sync'
					}
				]
			}
		});
		expect(diagnostics?.selectedHourRuntimeContract).toMatchObject({
			acceptedRequestId: 3,
			sceneRequestId: 3,
			requestMatchesScene: true,
			strongVisibleGpuPath: false,
			visibleRenderPathAvoidsCpuReadback: false
		});

		if (diagnostics?.timings?.renderPublication?.renderPublicationTimeline) {
			diagnostics.timings.renderPublication.renderPublicationTimeline.routeProjectedAtMs = 777;
			diagnostics.timings.renderPublication.renderPublicationTimeline.sceneSyncResetHistory![0].resetReason =
				'output-mutated';
			diagnostics.timings.renderPublication.renderPublicationTimeline.sceneSyncActiveWindowResetHistory![0].resetReason =
				'output-mutated';
		}
		expect(diagnostics?.timings?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 777,
			sceneSyncResetHistory: [
				{
					resetReason: 'output-mutated'
				}
			],
			sceneSyncActiveWindowResetHistory: [
				{
					resetReason: 'output-mutated'
				}
			]
		});
		expect(
			diagnostics?.timings?.renderPublication?.renderPublicationTimeline?.renderSurfaceMeshTrace
				?.recreateDecision
		).toEqual({
			missingSurface: false,
			notComputeBufferSurface: false,
			analysisIdentityChanged: false,
			layoutCompatible: true
		});
	});
});
