import type { Analysis } from '$lib/types/analysis';
import {
	disposeSelectedHourGpuResidentOutput,
	prepareSelectedHourLiveSession,
	type SelectedHourCpuFallbackOutput,
	type SelectedHourGpuResidentOutput,
	type SelectedHourLiveMetricType,
	type SelectedHourLiveSession
} from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import {
	mergeSelectedHourRenderTimings,
	type OnDemandRuntimeDiagnostics,
	type TrackedGpuAllocationBytes,
	type SelectedHourRenderTimingSubsteps
} from '$lib/compute/on-demand/onDemandDiagnostics';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import type {
	SelectedHourReadbackInstrumentation,
	SelectedHourReadbackReason
} from '$lib/diagnostics/selectedHourRuntimeContract';
import {
	copyRenderPublicationDiagnostics,
	mergeRenderPublicationDiagnostics,
	stampRenderPublicationTimeline
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

export type LiveSelectedHourRenderTransport =
	| 'idle'
	| 'cpu-uploaded-selected-hour'
	| 'compute-buffer-selected-hour'
	| 'live-render-pending';

export type LiveSelectedHourControllerSurfaceDiagnostics = {
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	renderOwnedSelectedHourBytes?: number;
	cpuPublishRequestId?: number;
	cpuPublishMonthIndex?: number;
	cpuPublishHourIndex?: number;
	cpuPublishTimeIndex?: number;
	cpuPublishSelectionKey?: string;
	gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
} & SelectedHourRenderTimingSubsteps;

export type LiveSelectedHourAcceptedVisibleSurface = {
	requestId: number;
	selectionKey: string;
	visibleAtMs: number;
	visibleStartedAtMs?: number;
};

export type LiveSelectedHourRuntimeDiagnostics = Pick<
	OnDemandRuntimeDiagnostics,
	| 'timings'
	| 'trackedGpuAllocationBytes'
	| 'activeMaskSource'
	| 'canonicalPointCount'
	| 'activePointCount'
	| 'inactivePointCount'
	| 'activePointRatio'
>;

export type LiveSelectedHourControllerState = {
	analysis: Analysis | null;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	surfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	// Last request that became visible, not the current pending request.
	acceptedVisibleSurface: LiveSelectedHourAcceptedVisibleSurface | null;
	acceptedRequestId: number | undefined;
	acceptedSelectionKey: string | undefined;
	acceptedVisibleAtMs: number | undefined;
	visibleSelectedHourReadbackCount: number | undefined;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
	selectedHourReadbackReasons: SelectedHourReadbackReason[];
	selectedHourReadbackReasonCounts: Partial<Record<SelectedHourReadbackReason, number>>;
	loading: boolean;
	error: string | null;
	renderTransport: LiveSelectedHourRenderTransport;
	sameDeviceForComputeAndRender: boolean | null;
	runtimeDiagnostics?: LiveSelectedHourRuntimeDiagnostics;
	pendingRenderUpdateStartedAt: number | undefined;
	renderSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	ready: boolean;
	renderReady: boolean;
	awaitingGpuSurface: boolean;
};

type LiveSelectedHourControllerMutableState = Omit<
	LiveSelectedHourControllerState,
	'ready' | 'renderReady' | 'awaitingGpuSurface'
>;

export type LiveSelectedHourSessionConfig = Omit<
	Parameters<typeof prepareSelectedHourLiveSession>[0],
	'signal'
>;

export type LiveSelectedHourControllerRequest = {
	sessionKey: string;
	sessionConfig: LiveSelectedHourSessionConfig;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey?: string;
	metricType: SelectedHourLiveMetricType;
	colorMode: 'normalized' | 'discrete';
	preferGpuResident: boolean;
	rendererDevice?: GPUDevice;
	selectedHourReadbackReason?: SelectedHourReadbackReason;
};

export type LiveSelectedHourControllerRequestResult = {
	accepted: boolean;
	reason?: 'stale' | 'disposed';
	state: LiveSelectedHourControllerState;
};

export type LiveSelectedHourGpuResidentReleaseReason =
	| 'copy-complete'
	| 'copy-failed'
	| 'superseded';

export interface LiveSelectedHourGpuResidentRelease {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: LiveSelectedHourGpuResidentReleaseReason;
}

export type LiveSelectedHourController = {
	getState(): LiveSelectedHourControllerState;
	subscribe(
		listener: (state: LiveSelectedHourControllerState) => void
	): () => void;
	requestSelection(
		request: LiveSelectedHourControllerRequest
	): Promise<LiveSelectedHourControllerRequestResult>;
	releaseAcceptedGpuResidentOutput(
		release: LiveSelectedHourGpuResidentRelease
	): void;
	handleRenderSurfaceDiagnostics(
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	): Promise<void>;
	dispose(): void;
};

type DeferredCpuFallbackState = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	analysis: Analysis | null;
	loadCpuFallback?: () => Promise<SelectedHourCpuFallbackOutput>;
};

type AcceptedCpuPublication = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
};

type ManagedAcceptedGpuResidentOutput = {
	value: SelectedHourGpuResidentOutput;
	releasable: boolean;
};

type CreateLiveSelectedHourControllerOptions = {
	prepareSession?: (
		params: Parameters<typeof prepareSelectedHourLiveSession>[0]
	) => Promise<SelectedHourLiveSession>;
};

const EMPTY_SURFACE_DIAGNOSTICS: LiveSelectedHourControllerSurfaceDiagnostics = {};

export function copyRenderPublication(
	renderPublication:
		| LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		| LiveSelectedHourRuntimeDiagnostics['timings']['renderPublication']
): typeof renderPublication {
	return copyRenderPublicationDiagnostics(renderPublication);
}

export function copyRuntimeDiagnosticsTimings(
	timings: LiveSelectedHourRuntimeDiagnostics['timings']
): LiveSelectedHourRuntimeDiagnostics['timings'] {
	return {
		...timings,
		renderPublication: copyRenderPublication(timings.renderPublication)
	};
}

export function copyRenderSurfaceDiagnostics(
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
): LiveSelectedHourControllerSurfaceDiagnostics {
	const next: LiveSelectedHourControllerSurfaceDiagnostics = {
		...diagnostics,
		renderPublication: copyRenderPublication(diagnostics.renderPublication)
	};
	if (next.renderPublication === undefined) {
		delete next.renderPublication;
	}
	return next;
}

function areRenderPublicationEqual(
	left:
		| LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		| undefined,
	right:
		| LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.renderPublicationVersion === right.renderPublicationVersion &&
		left.renderPublicationPath === right.renderPublicationPath &&
		left.renderPublicationPhase === right.renderPublicationPhase &&
		left.renderPublicationMeshAction === right.renderPublicationMeshAction &&
		left.renderPublicationPointCount === right.renderPublicationPointCount &&
		left.renderPublicationVertexCount === right.renderPublicationVertexCount &&
		left.renderPublicationIndexCount === right.renderPublicationIndexCount &&
		left.renderPublicationDrawIndexCount ===
			right.renderPublicationDrawIndexCount &&
		left.renderPublicationGridWidth === right.renderPublicationGridWidth &&
		left.renderPublicationGridHeight === right.renderPublicationGridHeight &&
		left.renderPublicationGridSize === right.renderPublicationGridSize &&
		left.renderPublicationSourceByteLength === right.renderPublicationSourceByteLength &&
		left.renderPublicationTargetByteLength === right.renderPublicationTargetByteLength &&
		left.renderPublicationRenderOwnedBytes ===
			right.renderPublicationRenderOwnedBytes &&
		areRenderAllocationPreflightsEqual(
			left.renderAllocationPreflight,
			right.renderAllocationPreflight
		) &&
		areRenderStorageCopyPreflightsEqual(
			left.renderStorageCopyPreflight,
			right.renderStorageCopyPreflight
		) &&
		left.renderPublicationTimeline?.computeCompletedAtMs ===
			right.renderPublicationTimeline?.computeCompletedAtMs &&
		left.renderPublicationTimeline?.controllerSessionRunStartedAtMs ===
			right.renderPublicationTimeline?.controllerSessionRunStartedAtMs &&
		left.renderPublicationTimeline?.controllerSessionRunCompletedAtMs ===
			right.renderPublicationTimeline?.controllerSessionRunCompletedAtMs &&
		left.renderPublicationTimeline?.controllerAcceptStartedAtMs ===
			right.renderPublicationTimeline?.controllerAcceptStartedAtMs &&
		left.renderPublicationTimeline?.controllerDiagnosticsMergedAtMs ===
			right.renderPublicationTimeline?.controllerDiagnosticsMergedAtMs &&
		left.renderPublicationTimeline?.controllerStatePublishedAtMs ===
			right.renderPublicationTimeline?.controllerStatePublishedAtMs &&
		left.renderPublicationTimeline?.sessionComputeOutputReturnedAtMs ===
			right.renderPublicationTimeline?.sessionComputeOutputReturnedAtMs &&
		left.renderPublicationTimeline?.sessionDiagnosticsAppliedAtMs ===
			right.renderPublicationTimeline?.sessionDiagnosticsAppliedAtMs &&
		left.renderPublicationTimeline?.sessionGpuOutputHandleReadyAtMs ===
			right.renderPublicationTimeline?.sessionGpuOutputHandleReadyAtMs &&
		left.renderPublicationTimeline?.sessionPreferGpuResidentResolvedAtMs ===
			right.renderPublicationTimeline?.sessionPreferGpuResidentResolvedAtMs &&
		left.renderPublicationTimeline?.sessionDebugReadbackStartedAtMs ===
			right.renderPublicationTimeline?.sessionDebugReadbackStartedAtMs &&
		left.renderPublicationTimeline?.sessionDebugReadbackCompletedAtMs ===
			right.renderPublicationTimeline?.sessionDebugReadbackCompletedAtMs &&
		left.renderPublicationTimeline?.sessionSelectedHourRangeScanStartedAtMs ===
			right.renderPublicationTimeline?.sessionSelectedHourRangeScanStartedAtMs &&
		left.renderPublicationTimeline?.sessionSelectedHourRangeScanCompletedAtMs ===
			right.renderPublicationTimeline
				?.sessionSelectedHourRangeScanCompletedAtMs &&
		left.renderPublicationTimeline?.sessionSelectedHourAnalysisBuildStartedAtMs ===
			right.renderPublicationTimeline?.sessionSelectedHourAnalysisBuildStartedAtMs &&
		left.renderPublicationTimeline?.sessionSelectedHourAnalysisBuildCompletedAtMs ===
			right.renderPublicationTimeline
				?.sessionSelectedHourAnalysisBuildCompletedAtMs &&
		left.renderPublicationTimeline?.sessionRangeResolveStartedAtMs ===
			right.renderPublicationTimeline?.sessionRangeResolveStartedAtMs &&
		left.renderPublicationTimeline?.sessionRangeResolveCompletedAtMs ===
			right.renderPublicationTimeline?.sessionRangeResolveCompletedAtMs &&
		left.renderPublicationTimeline?.sessionSelectedDayRangeCacheKey ===
			right.renderPublicationTimeline?.sessionSelectedDayRangeCacheKey &&
		left.renderPublicationTimeline?.sessionSelectedDayRangeCacheHit ===
			right.renderPublicationTimeline?.sessionSelectedDayRangeCacheHit &&
		left.renderPublicationTimeline?.sessionSelectedDayRangeCacheSizeBefore ===
			right.renderPublicationTimeline?.sessionSelectedDayRangeCacheSizeBefore &&
		left.renderPublicationTimeline?.sessionSelectedDayRangeCacheSizeAfter ===
			right.renderPublicationTimeline?.sessionSelectedDayRangeCacheSizeAfter &&
		left.renderPublicationTimeline?.sessionSelectedDayRangeReadbackCount ===
			right.renderPublicationTimeline?.sessionSelectedDayRangeReadbackCount &&
		left.renderPublicationTimeline?.sessionSelectedDayRangeComputedHourCount ===
			right.renderPublicationTimeline?.sessionSelectedDayRangeComputedHourCount &&
		left.renderPublicationTimeline?.sessionCpuFallbackSetupStartedAtMs ===
			right.renderPublicationTimeline?.sessionCpuFallbackSetupStartedAtMs &&
		left.renderPublicationTimeline?.sessionCpuFallbackSetupCompletedAtMs ===
			right.renderPublicationTimeline?.sessionCpuFallbackSetupCompletedAtMs &&
		left.renderPublicationTimeline?.sessionGpuResidentRangeResolveStartedAtMs ===
			right.renderPublicationTimeline
				?.sessionGpuResidentRangeResolveStartedAtMs &&
		left.renderPublicationTimeline?.sessionGpuResidentRangeResolveCompletedAtMs ===
			right.renderPublicationTimeline
				?.sessionGpuResidentRangeResolveCompletedAtMs &&
		left.renderPublicationTimeline?.sessionTooltipValuesHandoffStartedAtMs ===
			right.renderPublicationTimeline
				?.sessionTooltipValuesHandoffStartedAtMs &&
		left.renderPublicationTimeline?.sessionTooltipValuesHandoffCompletedAtMs ===
			right.renderPublicationTimeline
				?.sessionTooltipValuesHandoffCompletedAtMs &&
		left.renderPublicationTimeline?.sessionGpuResidentResultAssemblyStartedAtMs ===
			right.renderPublicationTimeline
				?.sessionGpuResidentResultAssemblyStartedAtMs &&
		left.renderPublicationTimeline?.sessionGpuResidentResultAssemblyCompletedAtMs ===
			right.renderPublicationTimeline
				?.sessionGpuResidentResultAssemblyCompletedAtMs &&
		left.renderPublicationTimeline?.sessionResultReadyAtMs ===
			right.renderPublicationTimeline?.sessionResultReadyAtMs &&
		left.renderPublicationTimeline?.sessionResultReturnedAtMs ===
			right.renderPublicationTimeline?.sessionResultReturnedAtMs &&
		left.renderPublicationTimeline?.selectedHourValuePublicationStartedAtMs ===
			right.renderPublicationTimeline?.selectedHourValuePublicationStartedAtMs &&
		left.renderPublicationTimeline?.controllerAcceptedAtMs ===
			right.renderPublicationTimeline?.controllerAcceptedAtMs &&
		left.renderPublicationTimeline?.routePendingSurfaceExposedAtMs ===
			right.renderPublicationTimeline?.routePendingSurfaceExposedAtMs &&
		left.renderPublicationTimeline?.routePublishedAtMs ===
			right.renderPublicationTimeline?.routePublishedAtMs &&
		left.renderPublicationTimeline?.routeProjectedAtMs ===
			right.renderPublicationTimeline?.routeProjectedAtMs &&
		left.renderPublicationTimeline?.routeProjectionEvaluationStartedAtMs ===
			right.renderPublicationTimeline?.routeProjectionEvaluationStartedAtMs &&
		left.renderPublicationTimeline?.routeProjectionEvaluationCompletedAtMs ===
			right.renderPublicationTimeline?.routeProjectionEvaluationCompletedAtMs &&
		left.renderPublicationTimeline?.scenePendingSurfaceObservedAtMs ===
			right.renderPublicationTimeline?.scenePendingSurfaceObservedAtMs &&
		left.renderPublicationTimeline?.sceneReactiveBlockEnteredAtMs ===
			right.renderPublicationTimeline?.sceneReactiveBlockEnteredAtMs &&
		left.renderPublicationTimeline?.sceneRenderStateResolvedAtMs ===
			right.renderPublicationTimeline?.sceneRenderStateResolvedAtMs &&
		left.renderPublicationTimeline?.sceneAcceptedKeyResolvedAtMs ===
			right.renderPublicationTimeline?.sceneAcceptedKeyResolvedAtMs &&
		left.renderPublicationTimeline?.sceneSyncInvocationQueuedAtMs ===
			right.renderPublicationTimeline?.sceneSyncInvocationQueuedAtMs &&
		left.renderPublicationTimeline?.sceneReactiveToSyncQueuedMs ===
			right.renderPublicationTimeline?.sceneReactiveToSyncQueuedMs &&
		left.renderPublicationTimeline?.sceneSyncQueuedToStartMs ===
			right.renderPublicationTimeline?.sceneSyncQueuedToStartMs &&
		left.renderPublicationTimeline?.sceneStartSyncEnteredAtMs ===
			right.renderPublicationTimeline?.sceneStartSyncEnteredAtMs &&
		left.renderPublicationTimeline?.sceneStartSyncReturnedAtMs ===
			right.renderPublicationTimeline?.sceneStartSyncReturnedAtMs &&
		left.renderPublicationTimeline?.sceneSyncAttemptStartedAtMs ===
			right.renderPublicationTimeline?.sceneSyncAttemptStartedAtMs &&
		left.renderPublicationTimeline?.sceneSyncAttemptToken ===
			right.renderPublicationTimeline?.sceneSyncAttemptToken &&
		left.renderPublicationTimeline?.sceneSurfaceReceivedAtMs ===
			right.renderPublicationTimeline?.sceneSurfaceReceivedAtMs &&
		left.renderPublicationTimeline?.controllerVisibleAcknowledgedAtMs ===
			right.renderPublicationTimeline?.controllerVisibleAcknowledgedAtMs &&
		left.renderPublicationTimeline?.publicationEffectStartedAtMs ===
			right.renderPublicationTimeline?.publicationEffectStartedAtMs &&
		left.renderPublicationTimeline?.sceneLayoutKeyStartedAtMs ===
			right.renderPublicationTimeline?.sceneLayoutKeyStartedAtMs &&
		left.renderPublicationTimeline?.sceneLayoutKeyCompletedAtMs ===
			right.renderPublicationTimeline?.sceneLayoutKeyCompletedAtMs &&
		left.renderPublicationTimeline?.scenePublicationPlanReadyAtMs ===
			right.renderPublicationTimeline?.scenePublicationPlanReadyAtMs &&
		areRenderLayoutBuildTracesEqual(
			left.renderPublicationTimeline?.renderLayoutBuildTrace,
			right.renderPublicationTimeline?.renderLayoutBuildTrace
		) &&
		areRenderLayoutReuseProofTracesEqual(
			left.renderPublicationTimeline?.renderLayoutReuseProofTrace,
			right.renderPublicationTimeline?.renderLayoutReuseProofTrace
		) &&
		left.renderPublicationTimeline?.renderLayoutReuseAction ===
			right.renderPublicationTimeline?.renderLayoutReuseAction &&
		left.renderPublicationTimeline?.renderLayoutReuseReason ===
			right.renderPublicationTimeline?.renderLayoutReuseReason &&
		left.renderPublicationTimeline?.renderLayoutReuseDecisionMs ===
			right.renderPublicationTimeline?.renderLayoutReuseDecisionMs &&
		left.renderPublicationTimeline?.renderLayoutReuseKeyMs ===
			right.renderPublicationTimeline?.renderLayoutReuseKeyMs &&
		left.renderPublicationTimeline?.renderLayoutReuseSourceSignatureMs ===
			right.renderPublicationTimeline?.renderLayoutReuseSourceSignatureMs &&
		left.renderPublicationTimeline?.renderLayoutReusePositionsSignatureMs ===
			right.renderPublicationTimeline?.renderLayoutReusePositionsSignatureMs &&
		left.renderPublicationTimeline?.renderLayoutReusePositionsSignatureCacheHit ===
			right.renderPublicationTimeline?.renderLayoutReusePositionsSignatureCacheHit &&
		left.renderPublicationTimeline?.renderLayoutReuseFrameCacheLookupMs ===
			right.renderPublicationTimeline?.renderLayoutReuseFrameCacheLookupMs &&
		left.renderPublicationTimeline?.renderLayoutReuseFrameDerivationMs ===
			right.renderPublicationTimeline?.renderLayoutReuseFrameDerivationMs &&
		left.renderPublicationTimeline?.renderLayoutReuseFrameCacheHit ===
			right.renderPublicationTimeline?.renderLayoutReuseFrameCacheHit &&
		left.renderPublicationTimeline?.renderLayoutReuseFrameCacheKind ===
			right.renderPublicationTimeline?.renderLayoutReuseFrameCacheKind &&
		left.renderPublicationTimeline?.renderLayoutPublicationPlanMs ===
			right.renderPublicationTimeline?.renderLayoutPublicationPlanMs &&
		left.renderPublicationTimeline?.renderLayoutCompatibilityMs ===
			right.renderPublicationTimeline?.renderLayoutCompatibilityMs &&
		left.renderPublicationTimeline
			?.renderLayoutCompatibilityRequiredExpensiveMappingComparison ===
			right.renderPublicationTimeline
				?.renderLayoutCompatibilityRequiredExpensiveMappingComparison &&
		left.renderPublicationTimeline
			?.renderLayoutCompatibilityPerformedExpensiveMappingComparison ===
			right.renderPublicationTimeline
				?.renderLayoutCompatibilityPerformedExpensiveMappingComparison &&
		left.renderPublicationTimeline?.renderLayoutReuseProofMs ===
			right.renderPublicationTimeline?.renderLayoutReuseProofMs &&
		left.renderPublicationTimeline?.renderLayoutReuseKeyMatch ===
			right.renderPublicationTimeline?.renderLayoutReuseKeyMatch &&
		left.renderPublicationTimeline?.renderLayoutReuseProofSource ===
			right.renderPublicationTimeline?.renderLayoutReuseProofSource &&
		left.renderPublicationTimeline?.renderLayoutReusePreviousKey ===
			right.renderPublicationTimeline?.renderLayoutReusePreviousKey &&
		left.renderPublicationTimeline?.renderLayoutReusePreviousRequestId ===
			right.renderPublicationTimeline?.renderLayoutReusePreviousRequestId &&
		left.renderPublicationTimeline?.renderLayoutReusePreviousSelectionKey ===
			right.renderPublicationTimeline?.renderLayoutReusePreviousSelectionKey &&
		left.renderPublicationTimeline?.activeLayoutCandidateCount ===
			right.renderPublicationTimeline?.activeLayoutCandidateCount &&
		areRenderSurfaceMeshTracesEqual(
			left.renderPublicationTimeline?.renderSurfaceMeshTrace,
			right.renderPublicationTimeline?.renderSurfaceMeshTrace
		) &&
		left.renderPublicationTimeline?.sceneSurfacePendingStorageInitAtMs ===
			right.renderPublicationTimeline?.sceneSurfacePendingStorageInitAtMs &&
		left.renderPublicationTimeline?.renderStorageWaitStartedAtMs ===
			right.renderPublicationTimeline?.renderStorageWaitStartedAtMs &&
		left.renderPublicationTimeline?.renderStoragePreWaitMs ===
			right.renderPublicationTimeline?.renderStoragePreWaitMs &&
		left.renderPublicationTimeline?.renderStorageReadyAtMs ===
			right.renderPublicationTimeline?.renderStorageReadyAtMs &&
		left.renderPublicationTimeline?.renderPublicationPreStorageStartedAtMs ===
			right.renderPublicationTimeline?.renderPublicationPreStorageStartedAtMs &&
		left.renderPublicationTimeline?.renderPublicationPreStorageCompletedAtMs ===
			right.renderPublicationTimeline?.renderPublicationPreStorageCompletedAtMs &&
		left.renderPublicationTimeline?.renderPublicationPreStorageMs ===
			right.renderPublicationTimeline?.renderPublicationPreStorageMs &&
		left.renderPublicationTimeline?.renderStoragePendingFlagStartedAtMs ===
			right.renderPublicationTimeline?.renderStoragePendingFlagStartedAtMs &&
		left.renderPublicationTimeline?.renderStoragePendingFlagCompletedAtMs ===
			right.renderPublicationTimeline?.renderStoragePendingFlagCompletedAtMs &&
		left.renderPublicationTimeline?.renderStorageInvalidateRequestedAtMs ===
			right.renderPublicationTimeline?.renderStorageInvalidateRequestedAtMs &&
		left.renderPublicationTimeline?.renderStorageFirstWaitFrameRequestedAtMs ===
			right.renderPublicationTimeline?.renderStorageFirstWaitFrameRequestedAtMs &&
		left.renderPublicationTimeline?.renderStorageFirstWaitFrameCompletedAtMs ===
			right.renderPublicationTimeline?.renderStorageFirstWaitFrameCompletedAtMs &&
		left.renderPublicationTimeline?.renderCopyQueueDrainStartedAtMs ===
			right.renderPublicationTimeline?.renderCopyQueueDrainStartedAtMs &&
		left.renderPublicationTimeline?.renderCopyQueueDrainCompletedAtMs ===
			right.renderPublicationTimeline?.renderCopyQueueDrainCompletedAtMs &&
		left.renderPublicationTimeline?.renderCopyQueueDrainMs ===
			right.renderPublicationTimeline?.renderCopyQueueDrainMs &&
		areRenderStorageWaitTracesEqual(
			left.renderPublicationTimeline?.renderStorageWaitTrace,
			right.renderPublicationTimeline?.renderStorageWaitTrace
		) &&
		left.renderPublicationTimeline?.renderBufferCopyEncoderCreateMs ===
			right.renderPublicationTimeline?.renderBufferCopyEncoderCreateMs &&
		left.renderPublicationTimeline?.renderBufferCopyCommandRecordMs ===
			right.renderPublicationTimeline?.renderBufferCopyCommandRecordMs &&
		left.renderPublicationTimeline?.renderBufferCopySubmitMs ===
			right.renderPublicationTimeline?.renderBufferCopySubmitMs &&
		left.renderPublicationTimeline?.sceneSyncCompletedAtMs ===
			right.renderPublicationTimeline?.sceneSyncCompletedAtMs &&
		areSceneSyncResetHistoriesEqual(
			left.renderPublicationTimeline?.sceneSyncResetHistory,
			right.renderPublicationTimeline?.sceneSyncResetHistory
		) &&
		areSceneSyncResetHistoriesEqual(
			left.renderPublicationTimeline?.sceneSyncActiveWindowResetHistory,
			right.renderPublicationTimeline?.sceneSyncActiveWindowResetHistory
		)
	);
}

function areRenderAllocationPreflightsEqual(
	left:
		| NonNullable<
				LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		  >['renderAllocationPreflight']
		| undefined,
	right:
		| NonNullable<
				LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		  >['renderAllocationPreflight']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.status === right.status &&
		left.renderTopology === right.renderTopology &&
		left.renderCellCount === right.renderCellCount &&
		left.canonicalCellCount === right.canonicalCellCount &&
		left.activePointCount === right.activePointCount &&
		left.estimatedRenderGeometryBytes === right.estimatedRenderGeometryBytes &&
		left.estimatedLargestSingleRenderAllocationBytes ===
			right.estimatedLargestSingleRenderAllocationBytes &&
		left.estimatedDenseRectGeometryBytes === right.estimatedDenseRectGeometryBytes &&
		left.estimatedLargestJsTypedArrayBytes ===
			right.estimatedLargestJsTypedArrayBytes &&
		left.jsLargestTypedArrayByteLimit === right.jsLargestTypedArrayByteLimit &&
		left.rendererMaxBufferSize === right.rendererMaxBufferSize &&
		left.rendererMaxStorageBufferBindingSize ===
			right.rendererMaxStorageBufferBindingSize &&
		left.activeRenderStrategy === right.activeRenderStrategy &&
		left.activeRenderInstanceCount === right.activeRenderInstanceCount &&
		left.activeRenderSharedVertexCount === right.activeRenderSharedVertexCount &&
		left.activeRenderSharedIndexCount === right.activeRenderSharedIndexCount &&
		left.activeCanonicalIndexBufferBytes === right.activeCanonicalIndexBufferBytes &&
		areStringArraysEqual(left.failureReasons, right.failureReasons) &&
		left.forbiddenDenseAllocationProof?.noDenseCellToPointStorageAttribute ===
			right.forbiddenDenseAllocationProof?.noDenseCellToPointStorageAttribute &&
		left.forbiddenDenseAllocationProof?.noDenseColorBuffer ===
			right.forbiddenDenseAllocationProof?.noDenseColorBuffer &&
		left.forbiddenDenseAllocationProof?.noWidthHeightRenderGeometry ===
			right.forbiddenDenseAllocationProof?.noWidthHeightRenderGeometry &&
		left.forbiddenDenseAllocationProof?.noPerActiveCellDuplicatedVertexBuffer ===
			right.forbiddenDenseAllocationProof?.noPerActiveCellDuplicatedVertexBuffer &&
		left.forbiddenDenseAllocationProof?.noPerActiveCellDuplicatedIndexBuffer ===
			right.forbiddenDenseAllocationProof?.noPerActiveCellDuplicatedIndexBuffer &&
		left.forbiddenDenseAllocationProof?.sharedQuadVertexIndexBuffersConstantSize ===
			right.forbiddenDenseAllocationProof?.sharedQuadVertexIndexBuffersConstantSize &&
		left.forbiddenDenseAllocationProof?.instanceCountEqualsActivePointCount ===
			right.forbiddenDenseAllocationProof?.instanceCountEqualsActivePointCount &&
		left.forbiddenDenseAllocationProof
			?.noFullDenseTooltipReverseMapWithoutExplicitApprovalAndByteAccounting ===
			right.forbiddenDenseAllocationProof
				?.noFullDenseTooltipReverseMapWithoutExplicitApprovalAndByteAccounting
	);
}

function areRenderStorageCopyPreflightsEqual(
	left:
		| NonNullable<
				LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		  >['renderStorageCopyPreflight']
		| undefined,
	right:
		| NonNullable<
				LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
		  >['renderStorageCopyPreflight']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.status === right.status &&
		left.sourceByteLength === right.sourceByteLength &&
		left.targetByteLength === right.targetByteLength &&
		left.requestedByteLength === right.requestedByteLength &&
		left.byteLengthsMatch === right.byteLengthsMatch &&
		left.sourceUsage === right.sourceUsage &&
		left.targetUsage === right.targetUsage &&
		left.sourceHasCopySrcUsage === right.sourceHasCopySrcUsage &&
		left.targetHasCopyDstUsage === right.targetHasCopyDstUsage &&
		left.targetHasStorageUsage === right.targetHasStorageUsage &&
		areStringArraysEqual(left.failureReasons, right.failureReasons)
	);
}

function areStringArraysEqual(left: string[] | undefined, right: string[] | undefined): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return left.length === right.length && left.every((value, index) => value === right[index]);
}

function areRenderLayoutBuildTracesEqual(
	left:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderLayoutBuildTrace']
		| undefined,
	right:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderLayoutBuildTrace']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.totalMs === right.totalMs &&
		left.arrayAllocationMs === right.arrayAllocationMs &&
		left.transformBoundsPassMs === right.transformBoundsPassMs &&
		left.coordinateAssignmentMs === right.coordinateAssignmentMs &&
		left.indexToTexelFillMs === right.indexToTexelFillMs &&
		left.cellToPointIndexBuildMs === right.cellToPointIndexBuildMs &&
		left.colorBufferAllocationMs === right.colorBufferAllocationMs
	);
}

function areRenderLayoutReuseProofTracesEqual(
	left:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderLayoutReuseProofTrace']
		| undefined,
	right:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderLayoutReuseProofTrace']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.decision === right.decision &&
		left.hoverCellLookupProofStatus === right.hoverCellLookupProofStatus &&
		left.previousLayoutPresent === right.previousLayoutPresent &&
		left.canonicalRuntimeCompatibilityWouldReuse ===
			right.canonicalRuntimeCompatibilityWouldReuse &&
		left.proofMatchesCanonicalRuntimeCompatibility ===
			right.proofMatchesCanonicalRuntimeCompatibility &&
		left.positionsReferenceMatch === right.positionsReferenceMatch &&
		left.pointCountMatch === right.pointCountMatch &&
		left.gridSizeMatch === right.gridSizeMatch &&
		left.coordinateSystemMatch === right.coordinateSystemMatch &&
		left.normalizationSignature.enabled === right.normalizationSignature.enabled &&
		left.normalizationSignature.offset.x === right.normalizationSignature.offset.x &&
		left.normalizationSignature.offset.y === right.normalizationSignature.offset.y &&
		left.normalizationSignature.offset.z === right.normalizationSignature.offset.z &&
		left.normalizationSignature.provenance ===
			right.normalizationSignature.provenance &&
		left.previousNormalizationSignature?.enabled ===
			right.previousNormalizationSignature?.enabled &&
		left.previousNormalizationSignature?.offset.x ===
			right.previousNormalizationSignature?.offset.x &&
		left.previousNormalizationSignature?.offset.y ===
			right.previousNormalizationSignature?.offset.y &&
		left.previousNormalizationSignature?.offset.z ===
			right.previousNormalizationSignature?.offset.z &&
		left.previousNormalizationSignature?.provenance ===
			right.previousNormalizationSignature?.provenance &&
		left.normalizationSignatureMatch === right.normalizationSignatureMatch &&
		left.constructionMode === right.constructionMode &&
		left.previousConstructionMode === right.previousConstructionMode &&
		left.constructionModeMatch === right.constructionModeMatch &&
		left.dimensionsMatch === right.dimensionsMatch &&
		left.placementMatch === right.placementMatch &&
		left.cellToPointMappingMatch === right.cellToPointMappingMatch &&
		left.proofCostMs === right.proofCostMs &&
		left.estimatedRetainedCpuLayoutBytes ===
			right.estimatedRetainedCpuLayoutBytes
	);
}

function areRenderSurfaceMeshTracesEqual(
	left:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderSurfaceMeshTrace']
		| undefined,
	right:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderSurfaceMeshTrace']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.action === right.action &&
		left.totalMs === right.totalMs &&
		left.recreateDecision?.missingSurface === right.recreateDecision?.missingSurface &&
		left.recreateDecision?.notComputeBufferSurface ===
			right.recreateDecision?.notComputeBufferSurface &&
		left.recreateDecision?.analysisIdentityChanged ===
			right.recreateDecision?.analysisIdentityChanged &&
		left.recreateDecision?.layoutCompatible ===
			right.recreateDecision?.layoutCompatible &&
		left.disposeResetMeshRemovalMs === right.disposeResetMeshRemovalMs &&
		left.createComputeBufferSurfaceMeshMs === right.createComputeBufferSurfaceMeshMs &&
		left.createComputeBufferSurfacePositionArrayAllocMs ===
			right.createComputeBufferSurfacePositionArrayAllocMs &&
		left.createComputeBufferSurfacePositionArrayFillMs ===
			right.createComputeBufferSurfacePositionArrayFillMs &&
		left.createComputeBufferSurfaceIndexArrayAllocMs ===
			right.createComputeBufferSurfaceIndexArrayAllocMs &&
		left.createComputeBufferSurfaceIndexArrayFillMs ===
			right.createComputeBufferSurfaceIndexArrayFillMs &&
		left.createComputeBufferSurfaceGeometryAttributeAttachMs ===
			right.createComputeBufferSurfaceGeometryAttributeAttachMs &&
		left.createComputeBufferSurfaceBoundsMs ===
			right.createComputeBufferSurfaceBoundsMs &&
		left.createComputeBufferSurfaceUtciStorageAllocMs ===
			right.createComputeBufferSurfaceUtciStorageAllocMs &&
		left.createComputeBufferSurfaceCellToPointAllocFillMs ===
			right.createComputeBufferSurfaceCellToPointAllocFillMs &&
		left.createComputeBufferSurfaceColorLutSetupMs ===
			right.createComputeBufferSurfaceColorLutSetupMs &&
		left.createComputeBufferSurfaceMaterialSetupMs ===
			right.createComputeBufferSurfaceMaterialSetupMs &&
		left.createComputeBufferSurfaceMeshConstructMs ===
			right.createComputeBufferSurfaceMeshConstructMs &&
		left.createComputeBufferSurfaceByteAccountingMs ===
			right.createComputeBufferSurfaceByteAccountingMs &&
		left.createComputeBufferSurfaceGeometryBytes ===
			right.createComputeBufferSurfaceGeometryBytes &&
		left.createComputeBufferSurfaceUtciStorageBytes ===
			right.createComputeBufferSurfaceUtciStorageBytes &&
		left.createComputeBufferSurfaceCellToPointBytes ===
			right.createComputeBufferSurfaceCellToPointBytes &&
		left.createComputeBufferSurfaceColorLutBytes ===
			right.createComputeBufferSurfaceColorLutBytes &&
		left.updateComputeBufferSurfaceMeshMs === right.updateComputeBufferSurfaceMeshMs &&
		left.updateComputeBufferSurfaceRangeUniformMs ===
			right.updateComputeBufferSurfaceRangeUniformMs &&
		left.updateComputeBufferSurfacePendingSourceMs ===
			right.updateComputeBufferSurfacePendingSourceMs &&
		left.updateComputeBufferSurfaceLayoutUserDataMs ===
			right.updateComputeBufferSurfaceLayoutUserDataMs &&
		left.updateComputeBufferSurfaceByteAccountingMs ===
			right.updateComputeBufferSurfaceByteAccountingMs &&
		left.fallbackDecisionMs === right.fallbackDecisionMs &&
		left.applySurfaceMeshStateMs === right.applySurfaceMeshStateMs &&
		left.setCreatedSurfacePendingStorageInitMs ===
			right.setCreatedSurfacePendingStorageInitMs &&
		left.setPostSurfacePendingStorageInitMs ===
			right.setPostSurfacePendingStorageInitMs &&
		left.sceneAddMs === right.sceneAddMs &&
		left.publishUtciSurfaceDiagnosticsMs === right.publishUtciSurfaceDiagnosticsMs
	);
}

function areRenderStorageWaitTracesEqual(
	left:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderStorageWaitTrace']
		| undefined,
	right:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['renderStorageWaitTrace']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	return (
		left.waitStartedAtMs === right.waitStartedAtMs &&
		left.waitFinishedAtMs === right.waitFinishedAtMs &&
		left.waitMs === right.waitMs &&
		left.readAttemptCount === right.readAttemptCount &&
		left.frameWaitCount === right.frameWaitCount &&
		left.deviceAvailableCount === right.deviceAvailableCount &&
		left.backendEntryAvailableCount === right.backendEntryAvailableCount &&
		left.bufferAvailableCount === right.bufferAvailableCount &&
		left.firstDeviceAtMs === right.firstDeviceAtMs &&
		left.firstBackendEntryAtMs === right.firstBackendEntryAtMs &&
		left.firstBufferAtMs === right.firstBufferAtMs &&
		left.lastReadState.deviceAvailable === right.lastReadState.deviceAvailable &&
		left.lastReadState.backendEntryAvailable ===
			right.lastReadState.backendEntryAvailable &&
		left.lastReadState.bufferAvailable === right.lastReadState.bufferAvailable &&
		left.samples.length === right.samples.length &&
		left.samples.every((leftSample, index) => {
			const rightSample = right.samples[index];
			return (
				leftSample.atMs === rightSample.atMs &&
				leftSample.deviceAvailable === rightSample.deviceAvailable &&
				leftSample.backendEntryAvailable === rightSample.backendEntryAvailable &&
				leftSample.bufferAvailable === rightSample.bufferAvailable
			);
		})
	);
}

function areSceneSyncResetHistoriesEqual(
	left:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['sceneSyncResetHistory']
		| undefined,
	right:
		| NonNullable<
				NonNullable<
					LiveSelectedHourControllerSurfaceDiagnostics['renderPublication']
				>['renderPublicationTimeline']
		  >['sceneSyncResetHistory']
		| undefined
): boolean {
	if (left === right) return true;
	if (left == null || right == null) return left === right;
	if (left.length !== right.length) return false;
	return left.every((leftEvent, index) => {
		const rightEvent = right[index];
		return (
			leftEvent.resetAtMs === rightEvent.resetAtMs &&
			leftEvent.resetReason === rightEvent.resetReason &&
			leftEvent.invalidateActiveRun === rightEvent.invalidateActiveRun &&
			leftEvent.previousCopyRunToken === rightEvent.previousCopyRunToken &&
			leftEvent.nextCopyRunToken === rightEvent.nextCopyRunToken &&
			leftEvent.previousSyncRunKey === rightEvent.previousSyncRunKey
		);
	});
}

function resolveRenderPublicationPath(
	renderTransport: LiveSelectedHourRenderTransport
): 'compute-buffer-selected-hour' | 'cpu-uploaded-selected-hour' | 'none' {
	if (renderTransport === 'compute-buffer-selected-hour') {
		return 'compute-buffer-selected-hour';
	}
	if (renderTransport === 'cpu-uploaded-selected-hour') {
		return 'cpu-uploaded-selected-hour';
	}
	return 'none';
}

function resolveRenderPublicationPhase(requestId: number): 'initial' | 'scrub' {
	return requestId <= 1 ? 'initial' : 'scrub';
}

function cloneState(state: LiveSelectedHourControllerState): LiveSelectedHourControllerState {
	return {
		...state,
		acceptedVisibleSurface: state.acceptedVisibleSurface
			? { ...state.acceptedVisibleSurface }
			: null,
		surfaceIdentity: state.surfaceIdentity ? { ...state.surfaceIdentity } : null,
		selectedHourReadbackReasons: [...state.selectedHourReadbackReasons],
		selectedHourReadbackReasonCounts: { ...state.selectedHourReadbackReasonCounts },
		runtimeDiagnostics: state.runtimeDiagnostics
			? {
					timings: copyRuntimeDiagnosticsTimings(state.runtimeDiagnostics.timings),
					trackedGpuAllocationBytes: {
						...state.runtimeDiagnostics.trackedGpuAllocationBytes
					},
					activeMaskSource: state.runtimeDiagnostics.activeMaskSource,
					canonicalPointCount: state.runtimeDiagnostics.canonicalPointCount,
					activePointCount: state.runtimeDiagnostics.activePointCount,
					inactivePointCount: state.runtimeDiagnostics.inactivePointCount,
					activePointRatio: state.runtimeDiagnostics.activePointRatio
				}
			: state.runtimeDiagnostics,
		renderSurfaceDiagnostics: copyRenderSurfaceDiagnostics(state.renderSurfaceDiagnostics)
	};
}

function createSelectionKey(params: {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
}): string {
	return `${params.requestId}:${params.monthIndex}:${params.hourIndex}:${params.timeIndex}`;
}

function createSurfaceIdentity(params: {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	selectedHourVisibleStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
}): LiveSelectedHourSurfaceIdentity {
	return {
		controllerIdentity: 'controller',
		controllerInstanceId: 0,
		requestId: params.requestId,
		monthIndex: params.monthIndex,
		hourIndex: params.hourIndex,
		timeIndex: params.timeIndex,
		selectionKey: params.selectionKey,
		pendingRenderUpdateStartedAt: params.pendingRenderUpdateStartedAt,
		selectedHourVisibleStartedAt: params.selectedHourVisibleStartedAt,
		acceptedGpuResidentOutput: params.acceptedGpuResidentOutput
	};
}

function hasAcceptedCpuRenderSurface(params: {
	renderTransport: LiveSelectedHourRenderTransport;
	renderSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	acceptedCpuPublication: AcceptedCpuPublication | null;
}): boolean {
	const { renderTransport, renderSurfaceDiagnostics, acceptedCpuPublication } = params;
	if (renderTransport !== 'cpu-uploaded-selected-hour' || acceptedCpuPublication == null) {
		return false;
	}

	return (
		renderSurfaceDiagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour' &&
		renderSurfaceDiagnostics.cpuPublishRequestId === acceptedCpuPublication.requestId &&
		renderSurfaceDiagnostics.cpuPublishMonthIndex === acceptedCpuPublication.monthIndex &&
		renderSurfaceDiagnostics.cpuPublishHourIndex === acceptedCpuPublication.hourIndex &&
		renderSurfaceDiagnostics.cpuPublishTimeIndex === acceptedCpuPublication.timeIndex &&
		renderSurfaceDiagnostics.cpuPublishSelectionKey === acceptedCpuPublication.selectionKey
	);
}

function hasAcceptedGpuRenderSurface(
	state: LiveSelectedHourControllerMutableState
): boolean {
	const acceptedRequestId =
		state.acceptedGpuResidentOutput != null ? state.surfaceIdentity?.requestId : undefined;
	return (
		acceptedRequestId !== undefined &&
		state.sameDeviceForComputeAndRender === true &&
		state.renderSurfaceDiagnostics.gpuResidentCopyRequestId === acceptedRequestId &&
		state.renderSurfaceDiagnostics.gpuResidentCopyStatus === 'complete' &&
		state.renderSurfaceDiagnostics.utciSurfaceSource === 'compute-buffer-selected-hour'
	);
}

function deriveState(
	state: LiveSelectedHourControllerMutableState,
	acceptedCpuPublication: AcceptedCpuPublication | null
): LiveSelectedHourControllerState {
	const awaitingGpuSurface =
		state.acceptedGpuResidentOutput != null &&
		state.renderTransport === 'compute-buffer-selected-hour' &&
		!hasAcceptedGpuRenderSurface(state);
	const ready = state.analysis != null || state.acceptedGpuResidentOutput != null;
	const cpuRenderReady =
		state.renderTransport !== 'cpu-uploaded-selected-hour' ||
		acceptedCpuPublication == null ||
		hasAcceptedCpuRenderSurface({
			renderTransport: state.renderTransport,
			renderSurfaceDiagnostics: state.renderSurfaceDiagnostics,
			acceptedCpuPublication
		});
	return {
		...state,
		awaitingGpuSurface,
		ready,
		renderReady: ready && !awaitingGpuSurface && cpuRenderReady
	};
}

function createInitialState(): LiveSelectedHourControllerState {
	return deriveState({
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
		runtimeDiagnostics: undefined,
		pendingRenderUpdateStartedAt: undefined,
		renderSurfaceDiagnostics: EMPTY_SURFACE_DIAGNOSTICS
	}, null);
}

function copyRuntimeDiagnostics(
	diagnostics: LiveSelectedHourRuntimeDiagnostics
): LiveSelectedHourRuntimeDiagnostics {
	return {
		timings: copyRuntimeDiagnosticsTimings(diagnostics.timings),
		trackedGpuAllocationBytes: { ...diagnostics.trackedGpuAllocationBytes },
		activeMaskSource: diagnostics.activeMaskSource,
		canonicalPointCount: diagnostics.canonicalPointCount,
		activePointCount: diagnostics.activePointCount,
		inactivePointCount: diagnostics.inactivePointCount,
		activePointRatio: diagnostics.activePointRatio
	};
}

function mergeTrackedGpuAllocationBytesWithRenderSurface(params: {
	tracked: TrackedGpuAllocationBytes;
	renderSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
}): TrackedGpuAllocationBytes {
	const hasRenderOwnedBytes = Object.prototype.hasOwnProperty.call(
		params.renderSurfaceDiagnostics,
		'renderOwnedSelectedHourBytes'
	);
	const renderOwnedSelectedHourBytes =
		hasRenderOwnedBytes
			? (params.renderSurfaceDiagnostics.renderOwnedSelectedHourBytes ?? 0)
			: (params.tracked.renderOwnedSelectedHourBytes ?? 0);
	return {
		...params.tracked,
		renderOwnedSelectedHourBytes,
		renderOwnedSelectedHourBytesHighWatermark: hasRenderOwnedBytes
			? renderOwnedSelectedHourBytes
			: Math.max(
					params.tracked.renderOwnedSelectedHourBytesHighWatermark ?? 0,
					renderOwnedSelectedHourBytes
				),
		trackingScope: 'utci-owned-webgpu-buffers'
	};
}

function mergeRuntimeDiagnosticsWithRenderSurface(params: {
	runtimeDiagnostics: LiveSelectedHourRuntimeDiagnostics | undefined;
	renderSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	pendingRenderUpdateStartedAt: number | undefined;
	selectedHourVisibleStartedAt: number | undefined;
	visibleAtMs: number | undefined;
	visibleAcknowledgementEligible?: boolean;
}): LiveSelectedHourRuntimeDiagnostics | undefined {
	if (!params.runtimeDiagnostics) return params.runtimeDiagnostics;
	const surfaceUpdateMs =
		typeof params.pendingRenderUpdateStartedAt === 'number' &&
		typeof params.visibleAtMs === 'number'
			? Math.max(0, params.visibleAtMs - params.pendingRenderUpdateStartedAt)
			: undefined;
	const selectedHourVisibleMs =
		typeof params.selectedHourVisibleStartedAt === 'number' &&
		typeof params.visibleAtMs === 'number'
			? Math.max(0, params.visibleAtMs - params.selectedHourVisibleStartedAt)
			: undefined;
	const trackedGpuAllocationBytes = mergeTrackedGpuAllocationBytesWithRenderSurface({
		tracked: params.runtimeDiagnostics.trackedGpuAllocationBytes,
		renderSurfaceDiagnostics: params.renderSurfaceDiagnostics
	});
	const renderPublication =
		mergeRenderPublicationDiagnostics(
			params.runtimeDiagnostics.timings.renderPublication,
			params.renderSurfaceDiagnostics.renderPublication
		);
	const shouldStampVisibleAcknowledgement =
		typeof params.visibleAtMs === 'number' &&
		params.visibleAcknowledgementEligible === true;
	const renderPublicationWithVisibleAcknowledgement =
		shouldStampVisibleAcknowledgement
			? stampRenderPublicationTimeline({
					current: renderPublication,
					timeline: {
						controllerVisibleAcknowledgedAtMs: params.visibleAtMs
					},
					fallback: {
						renderPublicationPath:
							params.renderSurfaceDiagnostics.renderPublication?.renderPublicationPath ??
							params.runtimeDiagnostics.timings.renderPublication
								?.renderPublicationPath ??
							'none',
						renderPublicationPhase:
							params.renderSurfaceDiagnostics.renderPublication?.renderPublicationPhase ??
							params.runtimeDiagnostics.timings.renderPublication
								?.renderPublicationPhase ??
							'unknown',
						renderPublicationMeshAction:
							params.renderSurfaceDiagnostics.renderPublication
								?.renderPublicationMeshAction ??
							params.runtimeDiagnostics.timings.renderPublication
								?.renderPublicationMeshAction ??
							'skipped'
					}
				})
			: renderPublication;
	if (surfaceUpdateMs === undefined) {
		return {
			trackedGpuAllocationBytes,
			timings: {
				...params.runtimeDiagnostics.timings,
				firstSelectedHourVisibleMs:
					selectedHourVisibleMs ??
					params.runtimeDiagnostics.timings.firstSelectedHourVisibleMs,
				renderSceneSyncStartDelayMs:
					params.renderSurfaceDiagnostics.renderSceneSyncStartDelayMs ??
					params.runtimeDiagnostics.timings.renderSceneSyncStartDelayMs,
				renderSceneSyncTotalMs:
					params.renderSurfaceDiagnostics.renderSceneSyncTotalMs ??
					params.runtimeDiagnostics.timings.renderSceneSyncTotalMs,
				renderLayoutBuildMs:
					params.renderSurfaceDiagnostics.renderLayoutBuildMs ??
					params.runtimeDiagnostics.timings.renderLayoutBuildMs,
				renderSurfaceMeshMs:
					params.renderSurfaceDiagnostics.renderSurfaceMeshMs ??
					params.runtimeDiagnostics.timings.renderSurfaceMeshMs,
				renderStorageInitWaitMs:
					params.renderSurfaceDiagnostics.renderStorageInitWaitMs ??
					params.runtimeDiagnostics.timings.renderStorageInitWaitMs,
				renderBufferCopyMs:
					params.renderSurfaceDiagnostics.renderBufferCopyMs ??
					params.runtimeDiagnostics.timings.renderBufferCopyMs,
				renderQueueDrainMs:
					params.renderSurfaceDiagnostics.renderQueueDrainMs ??
					params.runtimeDiagnostics.timings.renderQueueDrainMs,
				renderPublication: copyRenderPublication(
					renderPublicationWithVisibleAcknowledgement
				)
			}
		};
	}
	return {
		trackedGpuAllocationBytes,
		timings: mergeSelectedHourRenderTimings({
			existingTimings: params.runtimeDiagnostics.timings,
			renderUpdateMs: surfaceUpdateMs,
			gpuSurfaceUpdateMs: surfaceUpdateMs,
			firstSelectedHourVisibleMs:
				selectedHourVisibleMs ??
				params.runtimeDiagnostics.timings.firstSelectedHourVisibleMs ??
				surfaceUpdateMs,
			renderSubsteps: {
				renderSceneSyncStartDelayMs:
					params.renderSurfaceDiagnostics.renderSceneSyncStartDelayMs,
				renderSceneSyncTotalMs: params.renderSurfaceDiagnostics.renderSceneSyncTotalMs,
				renderLayoutBuildMs: params.renderSurfaceDiagnostics.renderLayoutBuildMs,
				renderSurfaceMeshMs: params.renderSurfaceDiagnostics.renderSurfaceMeshMs,
				renderStorageInitWaitMs:
					params.renderSurfaceDiagnostics.renderStorageInitWaitMs,
				renderBufferCopyMs: params.renderSurfaceDiagnostics.renderBufferCopyMs,
				renderQueueDrainMs: params.renderSurfaceDiagnostics.renderQueueDrainMs,
				renderPublication: copyRenderPublication(
					renderPublicationWithVisibleAcknowledgement
				)
			}
		})
	};
}

function isAbortError(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

function withControllerRequestId(
	gpuResidentOutput: SelectedHourGpuResidentOutput | null,
	requestId: number
): SelectedHourGpuResidentOutput | null {
	return gpuResidentOutput ? { ...gpuResidentOutput, requestId } : null;
}

function getGpuResidentOutputKey(params: {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
}): string {
	return `${params.requestId}:${params.monthIndex}:${params.timeIndex}`;
}

function mergeRenderSurfaceDiagnostics(
	current: LiveSelectedHourControllerSurfaceDiagnostics,
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics,
	trackedGpuRequestId: number | undefined,
	acceptedCpuPublication: AcceptedCpuPublication | null
): LiveSelectedHourControllerSurfaceDiagnostics {
	if (Object.keys(diagnostics).length === 0) {
		return current;
	}

	const hasGpuRequestScopedUpdate =
		diagnostics.utciSurfaceSource === 'compute-buffer-selected-hour' ||
		(diagnostics.gpuResidentCopyStatus !== undefined &&
			diagnostics.gpuResidentCopyStatus !== 'idle') ||
		diagnostics.gpuResidentCopyError !== undefined ||
		diagnostics.gpuResidentCopyRequestId !== undefined;
	if (
		hasGpuRequestScopedUpdate &&
		(trackedGpuRequestId === undefined ||
			diagnostics.gpuResidentCopyRequestId !== trackedGpuRequestId)
	) {
		return {
			...current,
			selectedHourTransferCount:
				diagnostics.selectedHourTransferCount ?? current.selectedHourTransferCount,
			dataTextureBuildCount:
				diagnostics.dataTextureBuildCount ?? current.dataTextureBuildCount,
			renderOwnedSelectedHourBytes: current.renderOwnedSelectedHourBytes
		};
	}

	const hasCpuRequestScopedUpdate =
		diagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour' ||
		diagnostics.cpuPublishRequestId !== undefined ||
		diagnostics.cpuPublishMonthIndex !== undefined ||
		diagnostics.cpuPublishHourIndex !== undefined ||
		diagnostics.cpuPublishTimeIndex !== undefined ||
		diagnostics.cpuPublishSelectionKey !== undefined;
	if (hasCpuRequestScopedUpdate && acceptedCpuPublication != null) {
		const matchesAcceptedCpuPublication =
			diagnostics.cpuPublishRequestId === acceptedCpuPublication.requestId &&
			diagnostics.cpuPublishMonthIndex === acceptedCpuPublication.monthIndex &&
			diagnostics.cpuPublishHourIndex === acceptedCpuPublication.hourIndex &&
			diagnostics.cpuPublishTimeIndex === acceptedCpuPublication.timeIndex &&
			diagnostics.cpuPublishSelectionKey === acceptedCpuPublication.selectionKey;
		if (!matchesAcceptedCpuPublication) {
			return current;
		}
	}
	if (hasCpuRequestScopedUpdate && acceptedCpuPublication == null) {
		return current;
	}

	const next = {
		...current,
		...diagnostics
	};
	if (current.renderPublication || diagnostics.renderPublication) {
		next.renderPublication = mergeRenderPublicationDiagnostics(
			current.renderPublication,
			diagnostics.renderPublication
		);
	} else {
		delete next.renderPublication;
	}
	const isAcceptedIdleCpuPublication =
		hasCpuRequestScopedUpdate &&
		diagnostics.gpuResidentCopyStatus === 'idle' &&
		diagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour';
	if (isAcceptedIdleCpuPublication) {
		if (!Object.prototype.hasOwnProperty.call(diagnostics, 'gpuResidentCopyError')) {
			delete next.gpuResidentCopyError;
		}
		if (!Object.prototype.hasOwnProperty.call(diagnostics, 'gpuResidentCopyRequestId')) {
			delete next.gpuResidentCopyRequestId;
		}
	}

	return next;
}

function areDiagnosticsEqual(
	left: LiveSelectedHourControllerSurfaceDiagnostics,
	right: LiveSelectedHourControllerSurfaceDiagnostics
): boolean {
	const leftKeys = Object.keys(left);
	const rightKeys = Object.keys(right);
	if (leftKeys.length !== rightKeys.length) {
		return false;
	}
	for (const key of leftKeys) {
		const typedKey = key as keyof LiveSelectedHourControllerSurfaceDiagnostics;
		if (typedKey === 'renderPublication') {
			if (!areRenderPublicationEqual(left.renderPublication, right.renderPublication)) {
				return false;
			}
			continue;
		}
		if (left[typedKey] !== right[typedKey]) {
			return false;
		}
	}
	return true;
}

export function createLiveSelectedHourController(
	options: CreateLiveSelectedHourControllerOptions = {}
): LiveSelectedHourController {
	const prepareSession = options.prepareSession ?? prepareSelectedHourLiveSession;
	const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
	let disposed = false;
	let state = createInitialState();
	let activeRequestToken = 0;
	let sessionEpoch = 0;
	let currentSessionKey: string | null = null;
	let currentSession: SelectedHourLiveSession | null = null;
	let currentSessionPromise: Promise<SelectedHourLiveSession> | null = null;
	let currentSessionAbortController: AbortController | null = null;
	let deferredCpuFallback: DeferredCpuFallbackState | null = null;
	let acceptedCpuPublication: AcceptedCpuPublication | null = null;
	let acceptedGpuResidentOutputEntry: ManagedAcceptedGpuResidentOutput | null = null;
	const retiredGpuResidentOutputs = new Map<string, ManagedAcceptedGpuResidentOutput>();

	function emit(): void {
		const snapshot = cloneState(state);
		for (const listener of listeners) {
			listener(snapshot);
		}
	}

	function setState(
		updater:
			| Partial<LiveSelectedHourControllerMutableState>
			| ((
					current: LiveSelectedHourControllerState
			  ) => Partial<LiveSelectedHourControllerMutableState>)
	): void {
		const patch = typeof updater === 'function' ? updater(state) : updater;
		const acceptedVisibleSurface =
			'acceptedVisibleSurface' in patch
				? patch.acceptedVisibleSurface ?? null
				: state.acceptedVisibleSurface;
		state = deriveState({
			analysis: 'analysis' in patch ? patch.analysis ?? null : state.analysis,
			acceptedGpuResidentOutput:
				'acceptedGpuResidentOutput' in patch
					? patch.acceptedGpuResidentOutput ?? null
					: state.acceptedGpuResidentOutput,
			surfaceIdentity:
				'surfaceIdentity' in patch ? patch.surfaceIdentity ?? null : state.surfaceIdentity,
			acceptedVisibleSurface,
			acceptedRequestId: acceptedVisibleSurface?.requestId,
			acceptedSelectionKey: acceptedVisibleSurface?.selectionKey,
			acceptedVisibleAtMs: acceptedVisibleSurface?.visibleAtMs,
			visibleSelectedHourReadbackCount:
				'visibleSelectedHourReadbackCount' in patch
					? patch.visibleSelectedHourReadbackCount
					: state.visibleSelectedHourReadbackCount,
			readbackInstrumentation:
				patch.readbackInstrumentation ?? state.readbackInstrumentation,
			selectedHourReadbackReasons:
				'selectedHourReadbackReasons' in patch
					? [...(patch.selectedHourReadbackReasons ?? [])]
					: state.selectedHourReadbackReasons,
			selectedHourReadbackReasonCounts:
				'selectedHourReadbackReasonCounts' in patch
					? { ...(patch.selectedHourReadbackReasonCounts ?? {}) }
					: state.selectedHourReadbackReasonCounts,
			loading: 'loading' in patch ? patch.loading ?? false : state.loading,
			error: 'error' in patch ? patch.error ?? null : state.error,
			renderTransport: patch.renderTransport ?? state.renderTransport,
			sameDeviceForComputeAndRender:
				'sameDeviceForComputeAndRender' in patch
					? patch.sameDeviceForComputeAndRender ?? null
					: state.sameDeviceForComputeAndRender,
			runtimeDiagnostics:
				'runtimeDiagnostics' in patch
					? patch.runtimeDiagnostics
						? copyRuntimeDiagnostics(patch.runtimeDiagnostics)
						: undefined
					: state.runtimeDiagnostics,
			pendingRenderUpdateStartedAt:
				'pendingRenderUpdateStartedAt' in patch
					? patch.pendingRenderUpdateStartedAt
					: state.pendingRenderUpdateStartedAt,
			renderSurfaceDiagnostics:
				patch.renderSurfaceDiagnostics ?? state.renderSurfaceDiagnostics
		}, acceptedCpuPublication);
		emit();
	}

	function destroyManagedAcceptedGpuResidentOutput(
		entry: ManagedAcceptedGpuResidentOutput | null
	): void {
		disposeSelectedHourGpuResidentOutput(entry?.value ?? null);
	}

	function getGpuResidentOwnershipHandle(
		output: SelectedHourGpuResidentOutput | null
	): unknown {
		return (
			output?.gpuOutputHandle ??
			output?.output.gpuOutputHandle ??
			output?.output.gpuBuffer ??
			output?.output ??
			null
		);
	}

	function gpuResidentOutputsShareOwnership(
		left: SelectedHourGpuResidentOutput | null,
		right: SelectedHourGpuResidentOutput | null
	): boolean {
		const leftHandle = getGpuResidentOwnershipHandle(left);
		const rightHandle = getGpuResidentOwnershipHandle(right);
		return leftHandle != null && leftHandle === rightHandle;
	}

	function disposeStaleGpuResidentOutput(output: SelectedHourGpuResidentOutput | null): void {
		if (!output) return;
		if (gpuResidentOutputsShareOwnership(acceptedGpuResidentOutputEntry?.value ?? null, output)) {
			return;
		}
		for (const retired of retiredGpuResidentOutputs.values()) {
			if (gpuResidentOutputsShareOwnership(retired.value, output)) {
				return;
			}
		}
		disposeSelectedHourGpuResidentOutput(output);
	}

	function maybeDestroyManagedAcceptedGpuResidentOutput(
		entry: ManagedAcceptedGpuResidentOutput
	): boolean {
		if (!entry.releasable) return false;
		destroyManagedAcceptedGpuResidentOutput(entry);
		return true;
	}

	function retireAcceptedGpuResidentOutput(
		entry: ManagedAcceptedGpuResidentOutput | null
	): void {
		if (!entry) return;
		if (maybeDestroyManagedAcceptedGpuResidentOutput(entry)) {
			return;
		}
		retiredGpuResidentOutputs.set(
			getGpuResidentOutputKey(entry.value),
			entry
		);
	}

	function replaceAcceptedGpuResidentOutput(
		next: SelectedHourGpuResidentOutput | null,
		patch: Partial<LiveSelectedHourControllerMutableState>
	): void {
		const previous = acceptedGpuResidentOutputEntry;
		const previousKey = previous ? getGpuResidentOutputKey(previous.value) : null;
		const nextKey = next ? getGpuResidentOutputKey(next) : null;
		const sameManagedIdentity = previousKey != null && previousKey === nextKey;
		if (previous && !sameManagedIdentity && previous.value.output !== next?.output) {
			retireAcceptedGpuResidentOutput(previous);
			acceptedGpuResidentOutputEntry = null;
		}
		if (next == null) {
			acceptedGpuResidentOutputEntry = null;
		} else if (sameManagedIdentity && previous) {
			acceptedGpuResidentOutputEntry = {
				value: next,
				releasable: previous.releasable
			};
		} else {
			acceptedGpuResidentOutputEntry = {
				value: next,
				releasable: false
			};
		}
		const runtimeDiagnostics =
			patch.runtimeDiagnostics && patch.renderTransport
				? copyRuntimeDiagnostics({
						...patch.runtimeDiagnostics,
						timings: {
							...patch.runtimeDiagnostics.timings,
							renderPublication: stampRenderPublicationTimeline({
								current: patch.runtimeDiagnostics.timings.renderPublication,
								timeline: {
									controllerStatePublishedAtMs: performance.now()
								},
								fallback: {
									renderPublicationPath: resolveRenderPublicationPath(
										patch.renderTransport
									),
									renderPublicationPhase: resolveRenderPublicationPhase(
										patch.surfaceIdentity?.requestId ?? 0
									),
									renderPublicationMeshAction: 'skipped'
								}
							})
						}
					})
				: patch.runtimeDiagnostics;
		setState({
			...patch,
			runtimeDiagnostics,
			acceptedGpuResidentOutput: acceptedGpuResidentOutputEntry?.value ?? null
		});
	}

	function resetControllerState(): void {
		deferredCpuFallback = null;
		acceptedCpuPublication = null;
		for (const retired of retiredGpuResidentOutputs.values()) {
			destroyManagedAcceptedGpuResidentOutput(retired);
		}
		retiredGpuResidentOutputs.clear();
		destroyManagedAcceptedGpuResidentOutput(acceptedGpuResidentOutputEntry);
		acceptedGpuResidentOutputEntry = null;
		state = createInitialState();
		emit();
	}

	function disposeCurrentSession(): void {
		currentSessionAbortController?.abort();
		currentSessionAbortController = null;
		currentSessionPromise = null;
		currentSession?.dispose();
		currentSession = null;
		currentSessionKey = null;
		sessionEpoch += 1;
	}

	async function ensureSession(
		request: LiveSelectedHourControllerRequest
	): Promise<SelectedHourLiveSession> {
		if (disposed) {
			throw new DOMException('Aborted', 'AbortError');
		}

		if (currentSessionKey === request.sessionKey) {
			if (currentSession) return currentSession;
			if (currentSessionPromise) return currentSessionPromise;
		}

		disposeCurrentSession();
		resetControllerState();
		setState({ loading: true });

		const abortController = new AbortController();
		const requestedEpoch = sessionEpoch;
		currentSessionAbortController = abortController;
		currentSessionKey = request.sessionKey;
		const sessionPromise = prepareSession({
			...request.sessionConfig,
			signal: abortController.signal
		})
			.then((session) => {
				if (
					disposed ||
					abortController.signal.aborted ||
					requestedEpoch !== sessionEpoch ||
					currentSessionKey !== request.sessionKey
				) {
					session.dispose();
					throw new DOMException('Aborted', 'AbortError');
				}
				currentSession = session;
				return session;
			})
			.finally(() => {
				if (currentSessionPromise === sessionPromise) {
					currentSessionPromise = null;
				}
			});
		currentSessionPromise = sessionPromise;
		return sessionPromise;
	}

	function ownsRequest(requestToken: number): boolean {
		return !disposed && requestToken === activeRequestToken;
	}

	return {
		getState() {
			return cloneState(state);
		},

		subscribe(listener) {
			listeners.add(listener);
			return () => {
				listeners.delete(listener);
			};
		},

		async requestSelection(request) {
			const requestToken = ++activeRequestToken;
			deferredCpuFallback = null;
			setState({ loading: true, error: null });

			try {
				const session = await ensureSession(request);
				if (!ownsRequest(requestToken)) {
					return {
						accepted: false,
						reason: disposed ? 'disposed' : 'stale',
						state: cloneState(state)
					};
				}

				const controllerSessionRunStartedAtMs = performance.now();
				const result = await session.runSelectedHour({
					monthIndex: request.monthIndex,
					hourIndex: request.hourIndex,
					timeIndex: request.timeIndex,
					metricType: request.metricType,
					colorMode: request.colorMode,
					preferGpuResident: request.preferGpuResident,
					rendererDevice: request.rendererDevice,
					selectedHourReadbackReason: request.selectedHourReadbackReason
				});
				if (!ownsRequest(requestToken)) {
					disposeStaleGpuResidentOutput(result.gpuResidentOutput);
					return {
						accepted: false,
						reason: disposed ? 'disposed' : 'stale',
						state: cloneState(state)
					};
				}

				const computeCompletedAtMs = performance.now();
				const controllerSessionRunCompletedAtMs = computeCompletedAtMs;
				const controllerAcceptStartedAtMs = computeCompletedAtMs;
				const controllerRequestId = requestToken;
				const acceptedGpuResidentOutput = withControllerRequestId(
					result.gpuResidentOutput,
					controllerRequestId
				);
				const acceptedSelectionKey =
					request.selectionKey ??
					createSelectionKey({
						requestId: controllerRequestId,
						monthIndex: result.monthIndex,
						hourIndex: result.hourIndex,
						timeIndex: result.timeIndex
					});
				deferredCpuFallback = acceptedGpuResidentOutput
					? {
							requestId: controllerRequestId,
							monthIndex: result.monthIndex,
							hourIndex: result.hourIndex,
							timeIndex: result.timeIndex,
							analysis: result.analysis,
							loadCpuFallback: result.loadCpuFallback
						}
					: null;
				acceptedCpuPublication =
					result.renderTransport === 'cpu-uploaded-selected-hour'
						? {
								requestId: controllerRequestId,
								monthIndex: result.monthIndex,
								hourIndex: result.hourIndex,
								timeIndex: result.timeIndex,
								selectionKey: acceptedSelectionKey
							}
						: null;
				const controllerAcceptedAtMs = performance.now();
				const acceptedRuntimeDiagnosticsBeforeMergeStamp = copyRuntimeDiagnostics({
					...result.diagnostics,
					timings: {
						...result.diagnostics.timings,
						renderPublication: stampRenderPublicationTimeline({
							current: result.diagnostics.timings.renderPublication,
							timeline: {
								controllerSessionRunStartedAtMs,
								controllerSessionRunCompletedAtMs,
								controllerAcceptStartedAtMs,
								computeCompletedAtMs,
								selectedHourValuePublicationStartedAtMs:
									result.pendingRenderUpdateStartedAt,
								controllerAcceptedAtMs
							},
							fallback: {
								renderPublicationPath: resolveRenderPublicationPath(
									result.renderTransport
								),
								renderPublicationPhase: resolveRenderPublicationPhase(
									controllerRequestId
								),
								renderPublicationMeshAction: 'skipped'
							}
						})
					}
				});
				const controllerDiagnosticsMergedAtMs = performance.now();
				const acceptedRuntimeDiagnostics = copyRuntimeDiagnostics({
					...acceptedRuntimeDiagnosticsBeforeMergeStamp,
					timings: {
						...acceptedRuntimeDiagnosticsBeforeMergeStamp.timings,
						renderPublication: stampRenderPublicationTimeline({
							current:
								acceptedRuntimeDiagnosticsBeforeMergeStamp.timings
									.renderPublication,
							timeline: {
								controllerDiagnosticsMergedAtMs
							},
							fallback: {
								renderPublicationPath: resolveRenderPublicationPath(
									result.renderTransport
								),
								renderPublicationPhase: resolveRenderPublicationPhase(
									controllerRequestId
								),
								renderPublicationMeshAction: 'skipped'
							}
						})
					}
				});

				replaceAcceptedGpuResidentOutput(acceptedGpuResidentOutput, {
					analysis: result.analysis,
					surfaceIdentity:
						result.renderTransport === 'live-render-pending'
							? null
							: createSurfaceIdentity({
									requestId: controllerRequestId,
									monthIndex: result.monthIndex,
									hourIndex: result.hourIndex,
									timeIndex: result.timeIndex,
									selectionKey: acceptedSelectionKey,
									pendingRenderUpdateStartedAt:
										result.renderTransport === 'compute-buffer-selected-hour'
											? result.pendingRenderUpdateStartedAt
											: undefined,
									selectedHourVisibleStartedAt: result.selectedHourVisibleStartedAt,
									acceptedGpuResidentOutput: acceptedGpuResidentOutput
								}),
					loading: result.renderTransport === 'compute-buffer-selected-hour',
					error: null,
					renderTransport: result.renderTransport,
					sameDeviceForComputeAndRender: result.sameDeviceForComputeAndRender,
					runtimeDiagnostics: acceptedRuntimeDiagnostics,
					visibleSelectedHourReadbackCount: undefined,
					readbackInstrumentation: 'not-instrumented',
					selectedHourReadbackReasons: result.diagnostics.selectedHourReadbackReasons ?? [],
					selectedHourReadbackReasonCounts:
						result.diagnostics.selectedHourReadbackReasonCounts ?? {},
					pendingRenderUpdateStartedAt:
						result.renderTransport === 'compute-buffer-selected-hour'
							? result.pendingRenderUpdateStartedAt
							: undefined,
					renderSurfaceDiagnostics:
						result.renderTransport === 'compute-buffer-selected-hour'
							? {
									gpuResidentCopyStatus: 'pending',
									gpuResidentCopyRequestId: controllerRequestId
								}
							: EMPTY_SURFACE_DIAGNOSTICS
				});

				return { accepted: true, state: cloneState(state) };
			} catch (error) {
				if (isAbortError(error) || !ownsRequest(requestToken)) {
					return {
						accepted: false,
						reason: disposed ? 'disposed' : 'stale',
						state: cloneState(state)
					};
				}

				deferredCpuFallback = null;
				setState({
					loading: false,
					error: error instanceof Error ? error.message : 'Failed to compute live UTCI.',
					pendingRenderUpdateStartedAt: undefined,
					renderSurfaceDiagnostics:
						state.acceptedGpuResidentOutput == null
							? EMPTY_SURFACE_DIAGNOSTICS
							: state.renderSurfaceDiagnostics
				});
				return { accepted: false, state: cloneState(state) };
			}
		},

		releaseAcceptedGpuResidentOutput(release) {
			if (disposed) return;

			const releasedKey = getGpuResidentOutputKey(release);
			const currentKey = acceptedGpuResidentOutputEntry
				? getGpuResidentOutputKey(acceptedGpuResidentOutputEntry.value)
				: null;
			if (currentKey === releasedKey && acceptedGpuResidentOutputEntry) {
				acceptedGpuResidentOutputEntry = {
					...acceptedGpuResidentOutputEntry,
					releasable: true
				};
				return;
			}

			const retired = retiredGpuResidentOutputs.get(releasedKey);
			if (!retired) return;
			retiredGpuResidentOutputs.delete(releasedKey);
			destroyManagedAcceptedGpuResidentOutput(retired);
		},

		async handleRenderSurfaceDiagnostics(diagnostics) {
			if (disposed) return;

			const acceptedRequestId =
				state.acceptedGpuResidentOutput != null ? state.surfaceIdentity?.requestId : undefined;
			const requestId = diagnostics.gpuResidentCopyRequestId;
			const nextDiagnostics = mergeRenderSurfaceDiagnostics(
				state.renderSurfaceDiagnostics,
				diagnostics,
				acceptedRequestId,
				acceptedCpuPublication
			);
			const acceptsGpuCompletion =
				diagnostics.gpuResidentCopyStatus === 'complete' &&
				requestId !== undefined &&
				acceptedRequestId === requestId &&
				state.sameDeviceForComputeAndRender === true &&
				state.renderTransport === 'compute-buffer-selected-hour' &&
				diagnostics.utciSurfaceSource === 'compute-buffer-selected-hour';
			if (
				diagnostics.gpuResidentCopyStatus === 'complete' &&
				!acceptsGpuCompletion
			) {
				return;
			}
			if (acceptsGpuCompletion) {
				const visibleAtMs = performance.now();
				deferredCpuFallback = null;
				acceptedCpuPublication = null;
				setState({
					acceptedVisibleSurface: state.surfaceIdentity
						? {
								requestId,
								selectionKey: state.surfaceIdentity.selectionKey,
								visibleAtMs,
								visibleStartedAtMs:
									state.surfaceIdentity.selectedHourVisibleStartedAt
							}
						: null,
					surfaceIdentity: state.surfaceIdentity
						? {
								...state.surfaceIdentity,
								pendingRenderUpdateStartedAt: undefined,
								acceptedGpuResidentOutput: state.acceptedGpuResidentOutput
							}
						: null,
						runtimeDiagnostics: mergeRuntimeDiagnosticsWithRenderSurface({
							runtimeDiagnostics: state.runtimeDiagnostics,
							renderSurfaceDiagnostics: nextDiagnostics,
							pendingRenderUpdateStartedAt: state.pendingRenderUpdateStartedAt,
							selectedHourVisibleStartedAt:
								state.surfaceIdentity?.selectedHourVisibleStartedAt,
							visibleAtMs,
							visibleAcknowledgementEligible:
								diagnostics.utciSurfaceSource ===
									'compute-buffer-selected-hour' ||
								diagnostics.renderPublication?.renderPublicationPath ===
									'compute-buffer-selected-hour'
						}),
					renderSurfaceDiagnostics: nextDiagnostics,
					loading: false,
					renderTransport: 'compute-buffer-selected-hour',
					visibleSelectedHourReadbackCount: 0,
					readbackInstrumentation: 'instrumented',
					pendingRenderUpdateStartedAt: undefined
				});
				return;
			}

			const shouldHandleGpuFallback =
				diagnostics.gpuResidentCopyStatus === 'failed' &&
				requestId !== undefined &&
				deferredCpuFallback?.requestId === requestId;
			const shouldHandleActivePreflightFailure =
				shouldHandleGpuFallback &&
				diagnostics.renderPublication?.renderAllocationPreflight?.status ===
					'failed' &&
				diagnostics.renderPublication.renderAllocationPreflight.renderTopology ===
					'active-cells';
			const acceptsCpuPublication =
				hasAcceptedCpuRenderSurface({
					renderTransport: state.renderTransport,
					renderSurfaceDiagnostics: nextDiagnostics,
					acceptedCpuPublication
				}) && acceptedCpuPublication != null;
			const cpuPublicationAlreadyVisible =
				acceptsCpuPublication &&
				acceptedCpuPublication != null &&
				state.acceptedVisibleSurface?.requestId === acceptedCpuPublication.requestId &&
				state.acceptedVisibleSurface.selectionKey === acceptedCpuPublication.selectionKey &&
				state.acceptedVisibleSurface.visibleAtMs !== undefined;
			if (
				areDiagnosticsEqual(nextDiagnostics, state.renderSurfaceDiagnostics) &&
				!shouldHandleActivePreflightFailure &&
				!shouldHandleGpuFallback &&
				(!acceptsCpuPublication || cpuPublicationAlreadyVisible)
			) {
				return;
			}

			if (shouldHandleActivePreflightFailure) {
				deferredCpuFallback = null;
				setState({
					runtimeDiagnostics: mergeRuntimeDiagnosticsWithRenderSurface({
						runtimeDiagnostics: state.runtimeDiagnostics,
						renderSurfaceDiagnostics: nextDiagnostics,
						pendingRenderUpdateStartedAt: state.pendingRenderUpdateStartedAt,
						selectedHourVisibleStartedAt: undefined,
						visibleAtMs: undefined,
						visibleAcknowledgementEligible: false
					}),
					renderSurfaceDiagnostics: nextDiagnostics,
					loading: false,
					error:
						diagnostics.gpuResidentCopyError ??
						'Active UTCI render allocation preflight failed.',
					renderTransport: 'compute-buffer-selected-hour',
					pendingRenderUpdateStartedAt: undefined
				});
				return;
			}

			const visibleAtMs = acceptsCpuPublication ? performance.now() : undefined;
			setState(
				acceptsCpuPublication && acceptedCpuPublication && !cpuPublicationAlreadyVisible
					? {
							runtimeDiagnostics: mergeRuntimeDiagnosticsWithRenderSurface({
								runtimeDiagnostics: state.runtimeDiagnostics,
								renderSurfaceDiagnostics: nextDiagnostics,
								pendingRenderUpdateStartedAt: state.pendingRenderUpdateStartedAt,
								selectedHourVisibleStartedAt:
								state.surfaceIdentity?.selectedHourVisibleStartedAt,
								visibleAtMs,
								visibleAcknowledgementEligible: false
							}),
							acceptedVisibleSurface: {
								requestId: acceptedCpuPublication.requestId,
								selectionKey: acceptedCpuPublication.selectionKey,
								visibleAtMs: visibleAtMs ?? performance.now(),
								visibleStartedAtMs:
									state.surfaceIdentity?.selectedHourVisibleStartedAt
							},
							renderSurfaceDiagnostics: nextDiagnostics,
							visibleSelectedHourReadbackCount: 1,
							readbackInstrumentation: 'instrumented'
					  }
					: {
							runtimeDiagnostics: mergeRuntimeDiagnosticsWithRenderSurface({
								runtimeDiagnostics: state.runtimeDiagnostics,
								renderSurfaceDiagnostics: nextDiagnostics,
							pendingRenderUpdateStartedAt: state.pendingRenderUpdateStartedAt,
							selectedHourVisibleStartedAt: undefined,
							visibleAtMs: undefined,
							visibleAcknowledgementEligible: false
						}),
							renderSurfaceDiagnostics: nextDiagnostics
					  }
			);

			if (shouldHandleGpuFallback) {
				const fallbackRequest = deferredCpuFallback;
				if (fallbackRequest == null) {
					return;
				}
				const ownsDeferredFallback = () =>
					!disposed &&
					(state.acceptedGpuResidentOutput != null
						? state.surfaceIdentity?.requestId === requestId
						: false) &&
					deferredCpuFallback?.requestId === requestId;
				let fallbackAnalysis = fallbackRequest.analysis;
				if (!fallbackAnalysis && fallbackRequest.loadCpuFallback) {
					try {
						const fallback = await fallbackRequest.loadCpuFallback();
						if (!ownsDeferredFallback()) {
							return;
						}
						fallbackAnalysis = fallback.analysis;
					} catch (error) {
						if (!ownsDeferredFallback()) {
							return;
						}
						deferredCpuFallback = null;
						replaceAcceptedGpuResidentOutput(null, {
							analysis: null,
							loading: false,
							error:
								error instanceof Error
									? `GPU copy failed and CPU fallback failed: ${error.message}`
									: 'GPU copy failed and CPU fallback failed.',
							renderTransport: 'cpu-uploaded-selected-hour',
							pendingRenderUpdateStartedAt: undefined
						});
						return;
					}
				}
				if (!fallbackAnalysis) {
					deferredCpuFallback = null;
					replaceAcceptedGpuResidentOutput(null, {
						analysis: null,
						loading: false,
						error: 'GPU copy failed and no CPU fallback analysis was available.',
						renderTransport: 'cpu-uploaded-selected-hour',
						pendingRenderUpdateStartedAt: undefined
					});
					return;
				}
				acceptedCpuPublication = {
					requestId: fallbackRequest.requestId,
					monthIndex: fallbackRequest.monthIndex,
					hourIndex: fallbackRequest.hourIndex,
					timeIndex: fallbackRequest.timeIndex,
					selectionKey:
						state.surfaceIdentity?.selectionKey ??
						createSelectionKey({
							requestId: fallbackRequest.requestId,
							monthIndex: fallbackRequest.monthIndex,
							hourIndex: fallbackRequest.hourIndex,
							timeIndex: fallbackRequest.timeIndex
						})
				};
				deferredCpuFallback = null;
				replaceAcceptedGpuResidentOutput(null, {
					analysis: fallbackAnalysis,
					surfaceIdentity: createSurfaceIdentity({
						requestId: acceptedCpuPublication.requestId,
						monthIndex: acceptedCpuPublication.monthIndex,
						hourIndex: acceptedCpuPublication.hourIndex,
						timeIndex: acceptedCpuPublication.timeIndex,
						selectionKey: acceptedCpuPublication.selectionKey,
						pendingRenderUpdateStartedAt: undefined,
						selectedHourVisibleStartedAt:
							state.surfaceIdentity?.selectedHourVisibleStartedAt,
						acceptedGpuResidentOutput: null
					}),
					loading: false,
					error: null,
					renderTransport: 'cpu-uploaded-selected-hour',
					pendingRenderUpdateStartedAt: undefined
				});
			}
		},

		dispose() {
			if (disposed) return;
			disposed = true;
			disposeCurrentSession();
			resetControllerState();
			listeners.clear();
		}
	};
}
