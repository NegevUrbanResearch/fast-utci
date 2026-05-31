import { describe, expect, it } from 'vitest';
import { buildMainRouteUtciDiagnostics } from '$lib/diagnostics/mainRouteUtciDiagnostics';
import {
	createRenderPublicationDiagnostics,
	mergeRenderPublicationTimeline
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

describe('buildMainRouteUtciDiagnostics', () => {
	it('does not erase existing render publication fields with undefined partial timeline values', () => {
		const merged = mergeRenderPublicationTimeline(
			{
				sessionSelectedDayRangeCacheKey: '8:24',
				sessionSelectedDayRangeCacheHit: false,
				sessionSelectedDayRangeCacheSizeBefore: 1,
				sessionSelectedDayRangeCacheSizeAfter: 2,
				sessionSelectedDayRangeReadbackCount: 0,
				sessionSelectedDayRangeComputedHourCount: 23,
				sessionSelectedDayRangeResolutionPath: 'compact-gpu-summary',
				sessionSelectedDayRangeSummaryReadbackCount: 23,
				sessionSelectedDayRangeSummaryReadbackBytes: 23 * 16,
				sessionSelectedDayRangeFullReadbackAvoidedCount: 23,
				sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
				sessionSelectedHourRangeReadbackCount: 0,
				sessionSelectedHourRangeCpuScanCount: 0,
				sessionSelectedHourRangeSummaryReadbackCount: 1,
				sessionSelectedHourRangeSummaryReadbackBytes: 16,
				sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
				sessionSelectedHourRangeSummaryReductionPassCount: 1,
				sceneReactiveToSyncQueuedMs: 0.1,
				sceneSyncQueuedToStartMs: 0.2
			},
			{
				sessionSelectedDayRangeCacheKey: undefined,
				sessionSelectedDayRangeCacheHit: undefined,
				sessionSelectedDayRangeCacheSizeBefore: undefined,
				sessionSelectedDayRangeCacheSizeAfter: undefined,
				sessionSelectedDayRangeReadbackCount: undefined,
				sessionSelectedDayRangeComputedHourCount: undefined,
				sessionSelectedDayRangeResolutionPath: undefined,
				sessionSelectedDayRangeSummaryReadbackCount: undefined,
				sessionSelectedDayRangeSummaryReadbackBytes: undefined,
				sessionSelectedDayRangeFullReadbackAvoidedCount: undefined,
				sessionSelectedHourRangeResolutionPath: undefined,
				sessionSelectedHourRangeReadbackCount: undefined,
				sessionSelectedHourRangeCpuScanCount: undefined,
				sessionSelectedHourRangeSummaryReadbackCount: undefined,
				sessionSelectedHourRangeSummaryReadbackBytes: undefined,
				sessionSelectedHourRangeFullReadbackAvoidedCount: undefined,
				sessionSelectedHourRangeSummaryReductionPassCount: undefined,
				sceneReactiveToSyncQueuedMs: undefined,
				sceneSyncQueuedToStartMs: undefined,
				routeProjectedAtMs: 120
			}
		);

		expect(merged).toMatchObject({
			sessionSelectedDayRangeCacheKey: '8:24',
			sessionSelectedDayRangeCacheHit: false,
			sessionSelectedDayRangeCacheSizeBefore: 1,
			sessionSelectedDayRangeCacheSizeAfter: 2,
			sessionSelectedDayRangeReadbackCount: 0,
			sessionSelectedDayRangeComputedHourCount: 23,
			sessionSelectedDayRangeResolutionPath: 'compact-gpu-summary',
			sessionSelectedDayRangeSummaryReadbackCount: 23,
			sessionSelectedDayRangeSummaryReadbackBytes: 23 * 16,
			sessionSelectedDayRangeFullReadbackAvoidedCount: 23,
			sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
			sessionSelectedHourRangeReadbackCount: 0,
			sessionSelectedHourRangeCpuScanCount: 0,
			sessionSelectedHourRangeSummaryReadbackCount: 1,
			sessionSelectedHourRangeSummaryReadbackBytes: 16,
			sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
			sessionSelectedHourRangeSummaryReductionPassCount: 1,
			sceneReactiveToSyncQueuedMs: 0.1,
			sceneSyncQueuedToStartMs: 0.2,
			routeProjectedAtMs: 120
		});
	});

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
				selectedHourValuePublicationStartedAtMs: 102,
				controllerAcceptedAtMs: 103,
				routePendingSurfaceExposedAtMs: 105,
				routePublishedAtMs: 107,
				routeProjectedAtMs: 109,
				scenePendingSurfaceObservedAtMs: 111,
				sceneSyncAttemptStartedAtMs: 112,
				sceneSyncAttemptToken: 7,
				sceneSurfaceReceivedAtMs: 113,
				controllerVisibleAcknowledgedAtMs: 126,
				publicationEffectStartedAtMs: 127,
				sceneLayoutKeyStartedAtMs: 112.25,
				sceneLayoutKeyCompletedAtMs: 113.25,
				scenePublicationPlanReadyAtMs: 119,
				sessionSelectedDayRangeCacheKey: '8:24',
				sessionSelectedDayRangeCacheHit: false,
				sessionSelectedDayRangeCacheSizeBefore: 1,
				sessionSelectedDayRangeCacheSizeAfter: 2,
				sessionSelectedDayRangeReadbackCount: 0,
				sessionSelectedDayRangeComputedHourCount: 23,
				sessionSelectedDayRangeResolutionPath: 'compact-gpu-summary',
				sessionSelectedDayRangeSummaryReadbackCount: 23,
				sessionSelectedDayRangeSummaryReadbackBytes: 23 * 16,
				sessionSelectedDayRangeFullReadbackAvoidedCount: 23,
				sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
				sessionSelectedHourRangeReadbackCount: 0,
				sessionSelectedHourRangeCpuScanCount: 0,
				sessionSelectedHourRangeSummaryReadbackCount: 1,
				sessionSelectedHourRangeSummaryReadbackBytes: 16,
				sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
				sessionSelectedHourRangeSummaryReductionPassCount: 1,
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
				renderLayoutReuseAction: 'reuse-candidate',
				renderLayoutReuseReason: 'reuse-safe',
				renderLayoutReuseDecisionMs: 0.5,
				renderLayoutReuseKeyMs: 0.75,
				renderLayoutReuseSourceSignatureMs: 0.2,
				renderLayoutReusePositionsSignatureMs: 0.15,
				renderLayoutReusePositionsSignatureCacheHit: false,
				renderLayoutReuseFrameCacheLookupMs: 0.1,
				renderLayoutReuseFrameDerivationMs: 0.25,
				renderLayoutReuseFrameCacheHit: false,
				renderLayoutReuseFrameCacheKind: 'structural',
				renderLayoutReuseKeyMatch: true,
				renderLayoutReuseProofSource: 'fresh-build-proof',
				renderLayoutReusePreviousKey: null,
				renderLayoutReusePreviousRequestId: null,
				renderLayoutReusePreviousSelectionKey: null,
				activeLayoutCandidateCount: 1,
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
				renderStorageWaitStartedAtMs: 127,
				renderStoragePreWaitMs: 15,
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
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeCacheKey =
			'mutated-cache-key';
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeCacheHit =
			true;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeCacheSizeBefore =
			99;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeCacheSizeAfter =
			100;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeReadbackCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeComputedHourCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeResolutionPath =
			'full-readback';
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeSummaryReadbackCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeSummaryReadbackBytes =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedDayRangeFullReadbackAvoidedCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeResolutionPath =
			'cpu-scan-existing-values';
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeReadbackCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeCpuScanCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeSummaryReadbackCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeSummaryReadbackBytes =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeFullReadbackAvoidedCount =
			999;
		renderPublication.renderPublicationTimeline!.sessionSelectedHourRangeSummaryReductionPassCount =
			999;
		renderPublication.renderPublicationTimeline!.renderLayoutBuildTrace!.totalMs = 999;
		renderPublication.renderPublicationTimeline!.renderLayoutBuildTrace!.arrayAllocationMs = 999;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseProofTrace!.decision =
			'rebuild-required';
		renderPublication.renderPublicationTimeline!.renderLayoutReuseProofTrace!.hoverCellLookupProofStatus =
			'not-compatible';
		renderPublication.renderPublicationTimeline!.renderLayoutReuseProofTrace!.proofCostMs = 999;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseAction = 'build-required';
		renderPublication.renderPublicationTimeline!.renderLayoutReuseReason =
			'layout-key-mismatch';
		renderPublication.renderPublicationTimeline!.renderLayoutReuseDecisionMs = 777;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseKeyMs = 888;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseSourceSignatureMs = 444;
		renderPublication.renderPublicationTimeline!.renderLayoutReusePositionsSignatureMs = 333;
		renderPublication.renderPublicationTimeline!.renderLayoutReusePositionsSignatureCacheHit =
			true;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseFrameCacheLookupMs = 222;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseFrameDerivationMs =
			999;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseFrameCacheHit = true;
		renderPublication.renderPublicationTimeline!.renderLayoutReuseKeyMatch = false;
		renderPublication.renderPublicationTimeline!.renderLayoutReusePreviousKey =
			'mutated-key';
		renderPublication.renderPublicationTimeline!.renderLayoutReusePreviousRequestId = 99;
		renderPublication.renderPublicationTimeline!.renderLayoutReusePreviousSelectionKey =
			'mutated-selection';
		renderPublication.renderPublicationTimeline!.activeLayoutCandidateCount = 9;
		renderPublication.renderPublicationTimeline!.renderSurfaceMeshTrace!.totalMs = 999;
		renderPublication.renderPublicationTimeline!.renderSurfaceMeshTrace!.recreateDecision!.layoutCompatible =
			false;
		renderPublication.renderPublicationTimeline!.renderStorageWaitStartedAtMs = 999;
		renderPublication.renderPublicationTimeline!.renderStoragePreWaitMs = 999;
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
				selectedHourValuePublicationStartedAtMs: 102,
				controllerAcceptedAtMs: 103,
				routePendingSurfaceExposedAtMs: 105,
				routePublishedAtMs: 107,
				routeProjectedAtMs: 109,
				scenePendingSurfaceObservedAtMs: 111,
				sceneSyncAttemptStartedAtMs: 112,
				sceneSyncAttemptToken: 7,
				sceneSurfaceReceivedAtMs: 113,
				controllerVisibleAcknowledgedAtMs: 126,
				publicationEffectStartedAtMs: 127,
				sceneLayoutKeyStartedAtMs: 112.25,
				sceneLayoutKeyCompletedAtMs: 113.25,
				scenePublicationPlanReadyAtMs: 119,
				sessionSelectedDayRangeCacheKey: '8:24',
				sessionSelectedDayRangeCacheHit: false,
				sessionSelectedDayRangeCacheSizeBefore: 1,
				sessionSelectedDayRangeCacheSizeAfter: 2,
				sessionSelectedDayRangeReadbackCount: 0,
				sessionSelectedDayRangeComputedHourCount: 23,
				sessionSelectedDayRangeResolutionPath: 'compact-gpu-summary',
				sessionSelectedDayRangeSummaryReadbackCount: 23,
				sessionSelectedDayRangeSummaryReadbackBytes: 23 * 16,
				sessionSelectedDayRangeFullReadbackAvoidedCount: 23,
				sessionSelectedHourRangeResolutionPath: 'compact-gpu-summary',
				sessionSelectedHourRangeReadbackCount: 0,
				sessionSelectedHourRangeCpuScanCount: 0,
				sessionSelectedHourRangeSummaryReadbackCount: 1,
				sessionSelectedHourRangeSummaryReadbackBytes: 16,
				sessionSelectedHourRangeFullReadbackAvoidedCount: 1,
				sessionSelectedHourRangeSummaryReductionPassCount: 1,
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
				renderLayoutReuseAction: 'reuse-candidate',
				renderLayoutReuseReason: 'reuse-safe',
				renderLayoutReuseDecisionMs: 0.5,
				renderLayoutReuseKeyMs: 0.75,
				renderLayoutReuseSourceSignatureMs: 0.2,
				renderLayoutReusePositionsSignatureMs: 0.15,
				renderLayoutReusePositionsSignatureCacheHit: false,
				renderLayoutReuseFrameCacheLookupMs: 0.1,
				renderLayoutReuseFrameDerivationMs: 0.25,
				renderLayoutReuseFrameCacheHit: false,
				renderLayoutReuseFrameCacheKind: 'structural',
				renderLayoutReuseKeyMatch: true,
				renderLayoutReuseProofSource: 'fresh-build-proof',
				renderLayoutReusePreviousKey: null,
				renderLayoutReusePreviousRequestId: null,
				renderLayoutReusePreviousSelectionKey: null,
				activeLayoutCandidateCount: 1,
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
				renderStorageWaitStartedAtMs: 127,
				renderStoragePreWaitMs: 15,
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
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutBuildTrace!.totalMs =
				555;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseProofTrace!.proofCostMs =
				222;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseAction =
				'build-required';
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseReason =
				'canonical-mismatch';
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseDecisionMs =
				333;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseKeyMs =
				444;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseFrameDerivationMs =
				555;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseFrameCacheHit =
				true;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReuseKeyMatch =
				false;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReusePreviousKey =
				'output-mutated-key';
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReusePreviousRequestId =
				123;
			diagnostics.timings.renderPublication.renderPublicationTimeline.renderLayoutReusePreviousSelectionKey =
				'output-mutated-selection';
			diagnostics.timings.renderPublication.renderPublicationTimeline.activeLayoutCandidateCount =
				5;
			diagnostics.timings.renderPublication.renderPublicationTimeline.sceneSyncResetHistory![0].resetReason =
				'output-mutated';
			diagnostics.timings.renderPublication.renderPublicationTimeline.sceneSyncActiveWindowResetHistory![0].resetReason =
				'output-mutated';
		}
		expect(diagnostics?.timings?.renderPublication?.renderPublicationTimeline).toMatchObject({
			routeProjectedAtMs: 777,
			renderLayoutBuildTrace: {
				totalMs: 555,
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
				proofCostMs: 222,
				estimatedRetainedCpuLayoutBytes: 32687044
			},
			renderLayoutReuseAction: 'build-required',
			renderLayoutReuseReason: 'canonical-mismatch',
			renderLayoutReuseDecisionMs: 333,
			renderLayoutReuseKeyMs: 444,
			renderLayoutReuseFrameDerivationMs: 555,
			renderLayoutReuseFrameCacheHit: true,
			renderLayoutReuseFrameCacheKind: 'structural',
			renderLayoutReuseKeyMatch: false,
			renderLayoutReuseProofSource: 'fresh-build-proof',
			renderLayoutReusePreviousKey: 'output-mutated-key',
			renderLayoutReusePreviousRequestId: 123,
			renderLayoutReusePreviousSelectionKey: 'output-mutated-selection',
			activeLayoutCandidateCount: 5,
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
