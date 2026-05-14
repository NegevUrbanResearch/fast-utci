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
import type { UtciRendererBackend, UtciRenderMode } from '$lib/utciRenderMode';

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
	baseSceneRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	tooltipHoverSampleCount: number;
	cameraWheelEventCount: number;
};

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
		baseRenderContextTimeIndex: params.baseSceneRenderContextTimeIndex,
		baseAcceptedUtciRange: params.baseAcceptedUtciRange,
		comparisonSurfaceRequestId:
			params.liveRouteState.comparisonSurfaceIdentity?.requestId,
		comparisonSelectionKey:
			params.liveRouteState.comparisonSurfaceIdentity?.selectionKey,
		comparisonSameDeviceForComputeAndRender:
			params.liveRouteState.comparison.sameDeviceForComputeAndRender,
		tooltipInteraction: {
			hoverSampleCount: params.tooltipHoverSampleCount,
		},
		cameraInteraction: {
			wheelEventCount: params.cameraWheelEventCount,
		},
		timings: params.liveRouteState.base.runtimeDiagnostics?.timings,
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
