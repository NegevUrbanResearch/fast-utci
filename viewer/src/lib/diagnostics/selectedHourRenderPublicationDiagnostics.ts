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
	updateComputeBufferSurfaceMeshMs?: number;
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
	computeCompletedAtMs?: number;
	controllerAcceptedAtMs?: number;
	routePendingSurfaceExposedAtMs?: number;
	routePublishedAtMs?: number;
	routeProjectedAtMs?: number;
	scenePendingSurfaceObservedAtMs?: number;
	sceneSyncAttemptStartedAtMs?: number;
	sceneSyncAttemptToken?: number;
	sceneSurfaceReceivedAtMs?: number;
	publicationEffectStartedAtMs?: number;
	renderSurfaceMeshTrace?: SelectedHourRenderSurfaceMeshTrace;
	renderStorageReadyAtMs?: number;
	renderStorageWaitTrace?: RenderStorageWaitDiagnostics;
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
	return copyRenderPublicationTimeline({
		...current,
		...next
	});
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
