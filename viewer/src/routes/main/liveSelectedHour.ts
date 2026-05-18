import {
	buildMainRouteUtciDiagnostics,
	type MainRouteUtciDiagnosticsInputs,
	type MainRouteUtciDiagnosticsPayload,
} from '$lib/diagnostics/mainRouteUtciDiagnostics';
import type {
	WebgpuLargeBufferDeviceLimits,
	WebgpuLargeBufferRequiredLimits,
} from '$lib/compute/gpu/webgpuDeviceLimits';
import type { LiveSelectedHourGpuResidentRelease } from '$lib/compute/selected-hour/liveSelectedHourController';
import type {
	LiveSelectedHourRouteHost,
	LiveSelectedHourRouteState,
} from '$lib/compute/selected-hour/liveSelectedHourRouteHost';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import {
	copyRenderPublicationDiagnostics,
	stampRenderPublicationTimeline
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import type { UtciRendererBackend, UtciRenderMode } from '$lib/utciRenderMode';
import type { ColorMode } from '$lib/types/viewer';
import type { TooltipInteractionDiagnostics } from '$lib/services/tooltipService';
import type { CameraInteractionDiagnostics } from '$lib/services/cameraInteractionTelemetry';

export type MainRouteWindow = Window & {
	__utciRenderDiagnostics__?: MainRouteUtciDiagnosticsPayload;
};

export type MainRouteAcceptedGpuResidentOutputReleaseParams =
	LiveSelectedHourGpuResidentRelease;

export type MainRouteLiveSelectedHourDiagnosticsParams = {
	enabled: boolean;
	utciOnDemand: 'f32';
	utciRenderRequested: UtciRenderMode;
	utciRenderResolved: 'dataTexture' | 'gpuNative';
	rendererBackend: UtciRendererBackend;
	rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
	rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
	liveRouteState: Pick<
		LiveSelectedHourRouteState,
		| 'base'
		| 'comparison'
		| 'baseSurfaceIdentity'
		| 'comparisonSurfaceIdentity'
		| 'baseSceneSurfaceIdentity'
	>;
	lastBaseGpuResidentCopyFailure?: { error?: string; requestId?: number };
	baseLiveReady: boolean;
	comparisonLiveReady: boolean;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	baseColorMode?: ColorMode;
	basePointCount?: number | null;
	baseMetadataGridSize?: number | null;
	baseSceneRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	tooltipInteraction: TooltipInteractionDiagnostics & { hoverSampleCount: number };
	cameraInteraction: CameraInteractionDiagnostics;
	timingsOverride?: MainRouteUtciDiagnosticsInputs['timings'];
};

function resolveRenderPublicationPhase(requestId: number): 'initial' | 'scrub' {
	return requestId <= 1 ? 'initial' : 'scrub';
}

export function createMainRouteRenderPublicationProjectionTracker() {
	let pendingSurfaceProjectedKey: string | null = null;
	let routePendingSurfaceExposedAtMs: number | undefined;
	let projectedKey: string | null = null;
	let routeProjectedAtMs: number | undefined;

	function resetProjectionState(): void {
		pendingSurfaceProjectedKey = null;
		routePendingSurfaceExposedAtMs = undefined;
		projectedKey = null;
		routeProjectedAtMs = undefined;
	}

	return {
		apply(params: {
			enabled: boolean;
			timings: MainRouteUtciDiagnosticsInputs['timings'] | undefined;
			projectedSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
			publishedSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
			sceneRenderContextTimeIndex: number | undefined;
			selectedTimeIndex: number;
		}): MainRouteUtciDiagnosticsInputs['timings'] | undefined {
			const routeProjectionEvaluationStartedAtMs = performance.now();
			const timings = params.timings
				? {
						...params.timings,
						renderPublication: copyRenderPublicationDiagnostics(
							params.timings.renderPublication
						)
					}
				: undefined;
			if (!params.enabled || params.projectedSceneSurfaceIdentity == null || timings == null) {
				resetProjectionState();
				return timings;
			}
			const projectedSceneControllerIdentity =
				params.projectedSceneSurfaceIdentity?.controllerIdentity;
			const projectedSceneControllerInstanceId =
				params.projectedSceneSurfaceIdentity?.controllerInstanceId;
			const projectedSceneRequestId =
				params.projectedSceneSurfaceIdentity?.requestId;
			const projectedSceneSelectionKey =
				params.projectedSceneSurfaceIdentity?.selectionKey;
			let pendingSurfaceExposureStampedThisPass = false;
			const shouldStampPendingSurfaceExposure =
				projectedSceneControllerIdentity !== undefined &&
				projectedSceneControllerInstanceId !== undefined &&
				projectedSceneRequestId !== undefined &&
				projectedSceneSelectionKey !== undefined;
			if (shouldStampPendingSurfaceExposure) {
				const nextPendingSurfaceProjectedKey = `${projectedSceneControllerIdentity}|${projectedSceneControllerInstanceId}|${projectedSceneRequestId}|${projectedSceneSelectionKey}`;
				if (pendingSurfaceProjectedKey !== nextPendingSurfaceProjectedKey) {
					pendingSurfaceProjectedKey = nextPendingSurfaceProjectedKey;
					routePendingSurfaceExposedAtMs = performance.now();
					pendingSurfaceExposureStampedThisPass = true;
				}
			}
			const controllerIdentity = params.publishedSurfaceIdentity?.controllerIdentity;
			const controllerInstanceId =
				params.publishedSurfaceIdentity?.controllerInstanceId;
			const requestId = params.publishedSurfaceIdentity?.requestId;
			const selectionKey = params.publishedSurfaceIdentity?.selectionKey;
			const shouldStamp =
				controllerIdentity !== undefined &&
				controllerInstanceId !== undefined &&
				requestId !== undefined &&
				selectionKey !== undefined &&
				params.sceneRenderContextTimeIndex === params.selectedTimeIndex;
			if (!shouldStamp && routePendingSurfaceExposedAtMs === undefined) {
				return timings;
			}

			let nextProjectedKey: string | null = null;
			if (shouldStamp) {
				nextProjectedKey = `${controllerIdentity}|${controllerInstanceId}|${requestId}|${selectionKey}`;
				if (projectedKey !== nextProjectedKey) {
					projectedKey = nextProjectedKey;
					routeProjectedAtMs =
						pendingSurfaceExposureStampedThisPass &&
						nextProjectedKey === pendingSurfaceProjectedKey
							? routePendingSurfaceExposedAtMs
							: performance.now();
				}
			}
			const projectedAtMsForCurrentSurface =
				shouldStamp && nextProjectedKey === pendingSurfaceProjectedKey
					? routeProjectedAtMs
					: undefined;
			const routeProjectionEvaluationCompletedAtMs = performance.now();

			return {
				...timings,
				renderPublication: stampRenderPublicationTimeline({
					current: timings?.renderPublication,
					timeline: {
						routeProjectionEvaluationStartedAtMs,
						routePendingSurfaceExposedAtMs,
						routeProjectedAtMs: projectedAtMsForCurrentSurface,
						routeProjectionEvaluationCompletedAtMs
					},
					fallback: {
						renderPublicationPath:
							timings?.renderPublication?.renderPublicationPath ?? 'none',
						renderPublicationPhase:
							timings?.renderPublication?.renderPublicationPhase ??
							resolveRenderPublicationPhase(
								requestId ?? projectedSceneRequestId ?? 0
							),
						renderPublicationMeshAction:
							timings?.renderPublication?.renderPublicationMeshAction ?? 'skipped'
					}
				})
			};
		}
	};
}

export function releaseBaseAcceptedGpuResidentOutput(
	host: Pick<LiveSelectedHourRouteHost, 'releaseBaseAcceptedGpuResidentOutput'>,
	params: MainRouteAcceptedGpuResidentOutputReleaseParams,
): void {
	host.releaseBaseAcceptedGpuResidentOutput(params);
}

export function releaseComparisonAcceptedGpuResidentOutput(
	host: Pick<
		LiveSelectedHourRouteHost,
		'releaseComparisonAcceptedGpuResidentOutput'
	>,
	params: MainRouteAcceptedGpuResidentOutputReleaseParams,
): void {
	host.releaseComparisonAcceptedGpuResidentOutput(params);
}

export function buildMainRouteLiveSelectedHourDiagnosticsInputs(
	params: MainRouteLiveSelectedHourDiagnosticsParams,
): MainRouteUtciDiagnosticsInputs {
	return {
		enabled: params.enabled,
		utciOnDemand: params.utciOnDemand,
		utciRenderRequested: params.utciRenderRequested,
		utciRenderResolved: params.utciRenderResolved,
		rendererBackend: params.rendererBackend,
		rendererRequiredLimits: params.rendererRequiredLimits,
		rendererDeviceLimits: params.rendererDeviceLimits,
		baseSurfaceDiagnostics: params.liveRouteState.base.renderSurfaceDiagnostics,
		comparisonSurfaceDiagnostics:
			params.liveRouteState.comparison.renderSurfaceDiagnostics,
		lastBaseGpuResidentCopyFailure: params.lastBaseGpuResidentCopyFailure,
		baseRenderTransport: params.liveRouteState.base.renderTransport,
		comparisonRenderTransport: params.liveRouteState.comparison.renderTransport,
		baseLiveReady: params.baseLiveReady,
		comparisonLiveReady: params.comparisonLiveReady,
		baseSurfaceRequestId: params.liveRouteState.baseSurfaceIdentity?.requestId,
		baseSelectionKey: params.liveRouteState.baseSurfaceIdentity?.selectionKey,
		baseSceneSurfaceRequestId:
			params.liveRouteState.baseSceneSurfaceIdentity?.requestId,
		baseSceneSelectionKey:
			params.liveRouteState.baseSceneSurfaceIdentity?.selectionKey,
		baseSameDeviceForComputeAndRender:
			params.liveRouteState.base.sameDeviceForComputeAndRender,
		baseSelectedMonthIndex: params.selectedMonthIndex,
		baseSelectedHourIndex: params.selectedHourIndex,
		baseSelectedTimeIndex: params.selectedTimeIndex,
		baseColorMode: params.baseColorMode,
		basePointCount: params.basePointCount,
		baseMetadataGridSize: params.baseMetadataGridSize,
		baseRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
		baseAcceptedUtciRange: params.baseAcceptedUtciRange,
		comparisonSurfaceRequestId:
			params.liveRouteState.comparisonSurfaceIdentity?.requestId,
		comparisonSelectionKey:
			params.liveRouteState.comparisonSurfaceIdentity?.selectionKey,
		comparisonSameDeviceForComputeAndRender:
			params.liveRouteState.comparison.sameDeviceForComputeAndRender,
		tooltipInteraction: params.tooltipInteraction,
		cameraInteraction: params.cameraInteraction,
		timings:
			params.timingsOverride ?? params.liveRouteState.base.runtimeDiagnostics?.timings,
		trackedGpuAllocationBytes:
			params.liveRouteState.base.runtimeDiagnostics?.trackedGpuAllocationBytes,
		visibleSelectedHourReadbackCount:
			params.liveRouteState.base.visibleSelectedHourReadbackCount,
		readbackInstrumentation: params.liveRouteState.base.readbackInstrumentation,
		selectedHourReadbackReasons:
			params.liveRouteState.base.selectedHourReadbackReasons,
		selectedHourReadbackReasonCounts:
			params.liveRouteState.base.selectedHourReadbackReasonCounts,
		comparisonSelectedHourReadbackReasons:
			params.liveRouteState.comparison.selectedHourReadbackReasons,
		comparisonSelectedHourReadbackReasonCounts:
			params.liveRouteState.comparison.selectedHourReadbackReasonCounts,
	};
}

export function buildMainRouteLiveSelectedHourDiagnostics(
	params: MainRouteLiveSelectedHourDiagnosticsParams,
): MainRouteUtciDiagnosticsPayload | undefined {
	return buildMainRouteUtciDiagnostics(
		buildMainRouteLiveSelectedHourDiagnosticsInputs(params),
	);
}

export function publishMainRouteUtciDiagnostics(
	win: MainRouteWindow,
	params: MainRouteLiveSelectedHourDiagnosticsParams,
): void {
	win.__utciRenderDiagnostics__ =
		buildMainRouteLiveSelectedHourDiagnostics(params);
}
