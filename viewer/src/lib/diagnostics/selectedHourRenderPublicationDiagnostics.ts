import type {
	RenderStorageWaitDiagnostics
} from '$lib/components/scene/utciComputeBufferRenderBridge';

export type SelectedHourRenderPublicationPath =
	| 'compute-buffer-selected-hour'
	| 'cpu-uploaded-selected-hour'
	| 'none';

export type SelectedHourRenderPublicationPhase = 'initial' | 'scrub' | 'unknown';

export type SelectedHourRenderPublicationMeshAction = 'created' | 'reused' | 'skipped';

export type SelectedHourRenderSurfaceMeshTraceAction =
	| 'created'
	| 'updated'
	| 'update-failed-created';

export type SelectedHourRenderLayoutBuildTrace = {
	totalMs: number;
	arrayAllocationMs?: number;
	transformBoundsPassMs?: number;
	coordinateAssignmentMs?: number;
	indexToTexelFillMs?: number;
	cellToPointIndexBuildMs?: number;
	colorBufferAllocationMs?: number;
};

export type SelectedHourRenderLayoutNormalizationSignature = {
	enabled: boolean;
	offset: {
		x: number;
		y: number;
		z: number;
	};
	provenance: 'normalization-disabled' | 'anchor-offset-minus-origin';
};

export type SelectedHourRenderLayoutConstructionMode =
	| 'world-positions'
	| 'metadata-bounds-fallback';

export type SelectedHourRenderLayoutReuseProofTrace = {
	decision: 'reuse-safe' | 'rebuild-required' | 'proof-inconclusive';
	hoverCellLookupProofStatus:
		| 'same-point-confirmed'
		| 'not-compatible'
		| 'proof-inconclusive';
	previousLayoutPresent: boolean;
	canonicalRuntimeCompatibilityWouldReuse: boolean | null;
	proofMatchesCanonicalRuntimeCompatibility: boolean | null;
	positionsReferenceMatch: boolean | null;
	pointCountMatch: boolean | null;
	gridSizeMatch: boolean | null;
	coordinateSystemMatch: boolean | null;
	normalizationSignature: SelectedHourRenderLayoutNormalizationSignature;
	previousNormalizationSignature: SelectedHourRenderLayoutNormalizationSignature | null;
	normalizationSignatureMatch: boolean | null;
	constructionMode: SelectedHourRenderLayoutConstructionMode;
	previousConstructionMode: SelectedHourRenderLayoutConstructionMode | null;
	constructionModeMatch: boolean | null;
	dimensionsMatch: boolean | null;
	placementMatch: boolean | null;
	cellToPointMappingMatch: boolean | null;
	proofCostMs: number | null;
	estimatedRetainedCpuLayoutBytes: number | null;
};

export type SelectedHourRenderLayoutReuseAction =
	| 'reuse-candidate'
	| 'build-required'
	| 'reused';
export type SelectedHourRenderLayoutReuseProofSource =
	| 'fresh-build-proof'
	| 'previous-publication-proof'
	| 'refreshed-runtime-proof';

export type SelectedHourRenderSurfaceMeshRecreateDecision = {
	missingSurface: boolean;
	notComputeBufferSurface: boolean;
	analysisIdentityChanged: boolean;
	layoutCompatible: boolean;
};

export type SelectedHourRenderSurfaceMeshTrace = {
	action: SelectedHourRenderSurfaceMeshTraceAction;
	totalMs: number;
	recreateDecision?: SelectedHourRenderSurfaceMeshRecreateDecision;
	disposeResetMeshRemovalMs?: number;
	createComputeBufferSurfaceMeshMs?: number;
	createComputeBufferSurfacePositionArrayAllocMs?: number;
	createComputeBufferSurfacePositionArrayFillMs?: number;
	createComputeBufferSurfaceIndexArrayAllocMs?: number;
	createComputeBufferSurfaceIndexArrayFillMs?: number;
	createComputeBufferSurfaceGeometryAttributeAttachMs?: number;
	createComputeBufferSurfaceBoundsMs?: number;
	createComputeBufferSurfaceUtciStorageAllocMs?: number;
	createComputeBufferSurfaceCellToPointAllocFillMs?: number;
	createComputeBufferSurfaceColorLutSetupMs?: number;
	createComputeBufferSurfaceMaterialSetupMs?: number;
	createComputeBufferSurfaceMeshConstructMs?: number;
	createComputeBufferSurfaceByteAccountingMs?: number;
	createComputeBufferSurfaceGeometryBytes?: number;
	createComputeBufferSurfaceUtciStorageBytes?: number;
	createComputeBufferSurfaceCellToPointBytes?: number;
	createComputeBufferSurfaceColorLutBytes?: number;
	updateComputeBufferSurfaceMeshMs?: number;
	updateComputeBufferSurfaceRangeUniformMs?: number;
	updateComputeBufferSurfacePendingSourceMs?: number;
	updateComputeBufferSurfaceLayoutUserDataMs?: number;
	updateComputeBufferSurfaceByteAccountingMs?: number;
	fallbackDecisionMs?: number;
	applySurfaceMeshStateMs?: number;
	setCreatedSurfacePendingStorageInitMs?: number;
	setPostSurfacePendingStorageInitMs?: number;
	sceneAddMs?: number;
	publishUtciSurfaceDiagnosticsMs?: number;
};

export type SelectedHourRenderPublicationSceneSyncResetEvent = {
	resetAtMs: number;
	resetReason: string;
	invalidateActiveRun: boolean;
	previousCopyRunToken: number;
	nextCopyRunToken: number;
	previousSyncRunKey?: string;
};

export type SelectedHourRenderPublicationTimeline = {
	controllerSessionRunStartedAtMs?: number;
	controllerSessionRunCompletedAtMs?: number;
	controllerAcceptStartedAtMs?: number;
	controllerDiagnosticsMergedAtMs?: number;
	controllerStatePublishedAtMs?: number;
	sessionComputeOutputReturnedAtMs?: number;
	sessionDiagnosticsAppliedAtMs?: number;
	sessionGpuOutputHandleReadyAtMs?: number;
	sessionPreferGpuResidentResolvedAtMs?: number;
	sessionDebugReadbackStartedAtMs?: number;
	sessionDebugReadbackCompletedAtMs?: number;
	sessionSelectedHourRangeScanStartedAtMs?: number;
	sessionSelectedHourRangeScanCompletedAtMs?: number;
	sessionSelectedHourRangeResolutionPath?:
		| 'compact-gpu-summary'
		| 'cpu-scan-existing-values'
		| 'unavailable'
		| 'not-needed';
	sessionSelectedHourRangeReadbackCount?: number;
	sessionSelectedHourRangeCpuScanCount?: number;
	sessionSelectedHourRangeSummaryReadbackCount?: number;
	sessionSelectedHourRangeSummaryReadbackBytes?: number;
	sessionSelectedHourRangeFullReadbackAvoidedCount?: number;
	sessionSelectedHourRangeSummaryReductionPassCount?: number;
	sessionSelectedHourAnalysisBuildStartedAtMs?: number;
	sessionSelectedHourAnalysisBuildCompletedAtMs?: number;
	sessionRangeResolveStartedAtMs?: number;
	sessionRangeResolveCompletedAtMs?: number;
	sessionSelectedDayRangeCacheKey?: string;
	sessionSelectedDayRangeCacheHit?: boolean;
	sessionSelectedDayRangeCacheSizeBefore?: number;
	sessionSelectedDayRangeCacheSizeAfter?: number;
	sessionSelectedDayRangeReadbackCount?: number;
	sessionSelectedDayRangeComputedHourCount?: number;
	sessionSelectedDayRangeResolutionPath?:
		| 'full-readback'
		| 'compact-gpu-summary'
		| 'cache-hit'
		| 'unavailable';
	sessionSelectedDayRangeSummaryReadbackCount?: number;
	sessionSelectedDayRangeSummaryReadbackBytes?: number;
	sessionSelectedDayRangeFullReadbackAvoidedCount?: number;
	sessionCpuFallbackSetupStartedAtMs?: number;
	sessionCpuFallbackSetupCompletedAtMs?: number;
	sessionGpuResidentRangeResolveStartedAtMs?: number;
	sessionGpuResidentRangeResolveCompletedAtMs?: number;
	sessionTooltipValuesHandoffStartedAtMs?: number;
	sessionTooltipValuesHandoffCompletedAtMs?: number;
	sessionGpuResidentResultAssemblyStartedAtMs?: number;
	sessionGpuResidentResultAssemblyCompletedAtMs?: number;
	sessionResultReadyAtMs?: number;
	sessionResultReturnedAtMs?: number;
	computeCompletedAtMs?: number;
	selectedHourValuePublicationStartedAtMs?: number;
	controllerAcceptedAtMs?: number;
	routePendingSurfaceExposedAtMs?: number;
	routePublishedAtMs?: number;
	routeProjectedAtMs?: number;
	routeProjectionEvaluationStartedAtMs?: number;
	routeProjectionEvaluationCompletedAtMs?: number;
	scenePendingSurfaceObservedAtMs?: number;
	sceneReactiveBlockEnteredAtMs?: number;
	sceneRenderStateResolvedAtMs?: number;
	sceneAcceptedKeyResolvedAtMs?: number;
	sceneSyncInvocationQueuedAtMs?: number;
	sceneReactiveToSyncQueuedMs?: number;
	sceneSyncQueuedToStartMs?: number;
	sceneStartSyncEnteredAtMs?: number;
	sceneStartSyncReturnedAtMs?: number;
	sceneSyncAttemptStartedAtMs?: number;
	sceneSyncAttemptToken?: number;
	sceneSurfaceReceivedAtMs?: number;
	controllerVisibleAcknowledgedAtMs?: number;
	publicationEffectStartedAtMs?: number;
	sceneLayoutKeyStartedAtMs?: number;
	sceneLayoutKeyCompletedAtMs?: number;
	scenePublicationPlanReadyAtMs?: number;
	renderLayoutBuildTrace?: SelectedHourRenderLayoutBuildTrace | null;
	renderLayoutReuseProofTrace?: SelectedHourRenderLayoutReuseProofTrace;
	renderLayoutReuseAction?: SelectedHourRenderLayoutReuseAction;
	renderLayoutReuseReason?: string;
	renderLayoutReuseDecisionMs?: number;
	renderLayoutReuseKeyMs?: number;
	renderLayoutReuseSourceSignatureMs?: number;
	renderLayoutReusePositionsSignatureMs?: number;
	renderLayoutReusePositionsSignatureCacheHit?: boolean;
	renderLayoutReuseFrameCacheLookupMs?: number;
	renderLayoutReuseFrameDerivationMs?: number;
	renderLayoutReuseFrameCacheHit?: boolean;
	renderLayoutReuseFrameCacheKind?: 'analysis-object' | 'structural' | 'miss';
	renderLayoutPublicationPlanMs?: number;
	renderLayoutCompatibilityMs?: number;
	renderLayoutCompatibilityRequiredExpensiveMappingComparison?: boolean;
	renderLayoutCompatibilityPerformedExpensiveMappingComparison?: boolean;
	renderLayoutReuseProofMs?: number;
	renderLayoutReuseKeyMatch?: boolean;
	renderLayoutReuseProofSource?: SelectedHourRenderLayoutReuseProofSource;
	renderLayoutReusePreviousKey?: string | null;
	renderLayoutReusePreviousRequestId?: number | null;
	renderLayoutReusePreviousSelectionKey?: string | null;
	activeLayoutCandidateCount?: number;
	renderSurfaceMeshTrace?: SelectedHourRenderSurfaceMeshTrace;
	sceneSurfacePendingStorageInitAtMs?: number;
	renderStorageWaitStartedAtMs?: number;
	renderStoragePreWaitMs?: number;
	renderStorageReadyAtMs?: number;
	renderStorageWaitTrace?: RenderStorageWaitDiagnostics;
	renderBufferCopyEncoderCreateMs?: number;
	renderBufferCopyCommandRecordMs?: number;
	renderBufferCopySubmitMs?: number;
	sceneSyncCompletedAtMs?: number;
	sceneSyncResetHistory?: SelectedHourRenderPublicationSceneSyncResetEvent[];
	sceneSyncActiveWindowResetHistory?: SelectedHourRenderPublicationSceneSyncResetEvent[];
};

export type SelectedHourRenderPublicationDiagnostics = {
	renderPublicationVersion: 1;
	renderPublicationPath: SelectedHourRenderPublicationPath;
	renderPublicationPhase: SelectedHourRenderPublicationPhase;
	renderPublicationMeshAction: SelectedHourRenderPublicationMeshAction;
	renderPublicationPointCount?: number;
	renderPublicationVertexCount?: number;
	renderPublicationIndexCount?: number;
	renderPublicationDrawIndexCount?: number;
	renderPublicationGridWidth?: number;
	renderPublicationGridHeight?: number;
	renderPublicationGridSize?: number;
	renderPublicationSourceByteLength?: number;
	renderPublicationTargetByteLength?: number;
	renderPublicationRenderOwnedBytes?: number;
	renderPublicationTimeline?: SelectedHourRenderPublicationTimeline;
};

export function copyRenderPublicationTimeline(
	timeline: SelectedHourRenderPublicationTimeline | undefined
): SelectedHourRenderPublicationTimeline | undefined {
	return timeline
		? {
				...timeline,
				renderLayoutBuildTrace: timeline.renderLayoutBuildTrace
					? {
							...timeline.renderLayoutBuildTrace
						}
					: timeline.renderLayoutBuildTrace,
				renderLayoutReuseProofTrace: timeline.renderLayoutReuseProofTrace
					? {
							...timeline.renderLayoutReuseProofTrace,
							normalizationSignature: {
								...timeline.renderLayoutReuseProofTrace.normalizationSignature,
								offset: {
									...timeline.renderLayoutReuseProofTrace.normalizationSignature.offset
								}
							},
							previousNormalizationSignature:
								timeline.renderLayoutReuseProofTrace.previousNormalizationSignature
									? {
											...timeline.renderLayoutReuseProofTrace.previousNormalizationSignature,
											offset: {
												...timeline.renderLayoutReuseProofTrace
													.previousNormalizationSignature.offset
											}
										}
									: timeline.renderLayoutReuseProofTrace.previousNormalizationSignature
						}
					: timeline.renderLayoutReuseProofTrace,
				sceneReactiveToSyncQueuedMs: timeline.sceneReactiveToSyncQueuedMs,
				sceneSyncQueuedToStartMs: timeline.sceneSyncQueuedToStartMs,
				sessionSelectedDayRangeCacheKey: timeline.sessionSelectedDayRangeCacheKey,
				sessionSelectedDayRangeCacheHit: timeline.sessionSelectedDayRangeCacheHit,
				sessionSelectedDayRangeCacheSizeBefore:
					timeline.sessionSelectedDayRangeCacheSizeBefore,
				sessionSelectedDayRangeCacheSizeAfter:
					timeline.sessionSelectedDayRangeCacheSizeAfter,
				sessionSelectedDayRangeReadbackCount:
					timeline.sessionSelectedDayRangeReadbackCount,
				sessionSelectedDayRangeComputedHourCount:
					timeline.sessionSelectedDayRangeComputedHourCount,
				sessionSelectedDayRangeResolutionPath:
					timeline.sessionSelectedDayRangeResolutionPath,
				sessionSelectedDayRangeSummaryReadbackCount:
					timeline.sessionSelectedDayRangeSummaryReadbackCount,
				sessionSelectedDayRangeSummaryReadbackBytes:
					timeline.sessionSelectedDayRangeSummaryReadbackBytes,
				sessionSelectedDayRangeFullReadbackAvoidedCount:
					timeline.sessionSelectedDayRangeFullReadbackAvoidedCount,
				sessionSelectedHourRangeResolutionPath:
					timeline.sessionSelectedHourRangeResolutionPath,
				sessionSelectedHourRangeReadbackCount:
					timeline.sessionSelectedHourRangeReadbackCount,
				sessionSelectedHourRangeCpuScanCount:
					timeline.sessionSelectedHourRangeCpuScanCount,
				sessionSelectedHourRangeSummaryReadbackCount:
					timeline.sessionSelectedHourRangeSummaryReadbackCount,
				sessionSelectedHourRangeSummaryReadbackBytes:
					timeline.sessionSelectedHourRangeSummaryReadbackBytes,
				sessionSelectedHourRangeFullReadbackAvoidedCount:
					timeline.sessionSelectedHourRangeFullReadbackAvoidedCount,
				sessionSelectedHourRangeSummaryReductionPassCount:
					timeline.sessionSelectedHourRangeSummaryReductionPassCount,
				renderLayoutReuseAction: timeline.renderLayoutReuseAction,
				renderLayoutReuseReason: timeline.renderLayoutReuseReason,
				renderLayoutReuseDecisionMs: timeline.renderLayoutReuseDecisionMs,
				renderLayoutReuseKeyMs: timeline.renderLayoutReuseKeyMs,
				renderLayoutReuseSourceSignatureMs:
					timeline.renderLayoutReuseSourceSignatureMs,
				renderLayoutReusePositionsSignatureMs:
					timeline.renderLayoutReusePositionsSignatureMs,
				renderLayoutReusePositionsSignatureCacheHit:
					timeline.renderLayoutReusePositionsSignatureCacheHit,
				renderLayoutReuseFrameCacheLookupMs:
					timeline.renderLayoutReuseFrameCacheLookupMs,
				renderLayoutReuseFrameDerivationMs:
					timeline.renderLayoutReuseFrameDerivationMs,
				renderLayoutReuseFrameCacheHit:
					timeline.renderLayoutReuseFrameCacheHit,
				renderLayoutReuseFrameCacheKind:
					timeline.renderLayoutReuseFrameCacheKind,
				renderLayoutReuseKeyMatch: timeline.renderLayoutReuseKeyMatch,
				renderLayoutReuseProofSource: timeline.renderLayoutReuseProofSource,
				renderLayoutReusePreviousKey: timeline.renderLayoutReusePreviousKey,
				renderLayoutReusePreviousRequestId:
					timeline.renderLayoutReusePreviousRequestId,
				renderLayoutReusePreviousSelectionKey:
					timeline.renderLayoutReusePreviousSelectionKey,
				activeLayoutCandidateCount: timeline.activeLayoutCandidateCount,
				renderSurfaceMeshTrace: timeline.renderSurfaceMeshTrace
					? {
							...timeline.renderSurfaceMeshTrace,
							recreateDecision: timeline.renderSurfaceMeshTrace.recreateDecision
								? {
										...timeline.renderSurfaceMeshTrace.recreateDecision
									}
								: timeline.renderSurfaceMeshTrace.recreateDecision
						}
					: timeline.renderSurfaceMeshTrace,
				sceneSurfacePendingStorageInitAtMs:
					timeline.sceneSurfacePendingStorageInitAtMs,
				renderStorageWaitStartedAtMs: timeline.renderStorageWaitStartedAtMs,
				renderStoragePreWaitMs: timeline.renderStoragePreWaitMs,
				renderStorageWaitTrace: timeline.renderStorageWaitTrace
					? {
							...timeline.renderStorageWaitTrace,
							lastReadState: {
								...timeline.renderStorageWaitTrace.lastReadState
							},
							samples: timeline.renderStorageWaitTrace.samples.map((sample) => ({
								...sample
							}))
						}
					: timeline.renderStorageWaitTrace,
				sceneSyncResetHistory: timeline.sceneSyncResetHistory?.map((event) => ({
					...event
				})),
				sceneSyncActiveWindowResetHistory: timeline.sceneSyncActiveWindowResetHistory?.map((event) => ({
					...event
				}))
			}
		: timeline;
}

export function copyRenderPublicationDiagnostics(
	diagnostics: SelectedHourRenderPublicationDiagnostics | undefined
): SelectedHourRenderPublicationDiagnostics | undefined {
	if (!diagnostics) return diagnostics;
	return {
		...diagnostics,
		renderPublicationTimeline: copyRenderPublicationTimeline(
			diagnostics.renderPublicationTimeline
		)
	};
}

export function mergeRenderPublicationTimeline(
	current: SelectedHourRenderPublicationTimeline | undefined,
	next: SelectedHourRenderPublicationTimeline | undefined
): SelectedHourRenderPublicationTimeline | undefined {
	if (!current) return copyRenderPublicationTimeline(next);
	if (!next) return copyRenderPublicationTimeline(current);
	const merged = { ...current };
	for (const [key, value] of Object.entries(next) as [
		keyof SelectedHourRenderPublicationTimeline,
		SelectedHourRenderPublicationTimeline[keyof SelectedHourRenderPublicationTimeline]
	][]) {
		if (value !== undefined) {
			merged[key] = value as never;
		}
	}
	return copyRenderPublicationTimeline(merged);
}

export function mergeRenderPublicationDiagnostics(
	current: SelectedHourRenderPublicationDiagnostics | undefined,
	next: SelectedHourRenderPublicationDiagnostics | undefined
): SelectedHourRenderPublicationDiagnostics | undefined {
	if (!current) return copyRenderPublicationDiagnostics(next);
	if (!next) return copyRenderPublicationDiagnostics(current);
	return {
		...current,
		...next,
		renderPublicationTimeline: mergeRenderPublicationTimeline(
			current.renderPublicationTimeline,
			next.renderPublicationTimeline
		)
	};
}

export function stampRenderPublicationTimeline(params: {
	current: SelectedHourRenderPublicationDiagnostics | undefined;
	timeline: SelectedHourRenderPublicationTimeline;
	fallback: Omit<
		SelectedHourRenderPublicationDiagnostics,
		'renderPublicationVersion' | 'renderPublicationTimeline'
	>;
}): SelectedHourRenderPublicationDiagnostics {
	const base =
		copyRenderPublicationDiagnostics(params.current) ??
		createRenderPublicationDiagnostics(params.fallback);
	return {
		...base,
		renderPublicationTimeline: mergeRenderPublicationTimeline(
			base.renderPublicationTimeline,
			params.timeline
		)
	};
}

export function createRenderPublicationDiagnostics(
	diagnostics: Omit<SelectedHourRenderPublicationDiagnostics, 'renderPublicationVersion'>
): SelectedHourRenderPublicationDiagnostics {
	return copyRenderPublicationDiagnostics({
		renderPublicationVersion: 1,
		...diagnostics
	}) as SelectedHourRenderPublicationDiagnostics;
}
