import {
	createLiveSelectedHourRouteHost,
	type LiveSelectedHourRouteHost,
	type LiveSelectedHourRouteInputs,
	type LiveSelectedHourRouteState
} from '$lib/compute/selected-hour/liveSelectedHourRouteHost';
import {
	projectMainRouteLiveSceneState,
	type MainRouteLiveSceneProjection
} from '$lib/compute/selected-hour/liveSelectedHourRouteProjection';
import type { LiveSelectedHourControllerSurfaceDiagnostics } from '$lib/compute/selected-hour/liveSelectedHourController';
import type { Analysis } from '$lib/types/analysis';
import type { Group } from 'three';
import type { LiveSelectedHourPublishedRenderContext } from '$lib/compute/selected-hour/liveSelectedHourRenderContext';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';
import type { UtciRenderMode, UtciRendererBackend } from '$lib/utciRenderMode';
import type { DebugRouteAcceptedGpuResidentOutputRelease } from './legacySelectedHourWiring';
import type { MetricType } from '$lib/types/viewer';

export function createDebugSharedRouteHost(dataBasePath: string): LiveSelectedHourRouteHost {
	return createLiveSelectedHourRouteHost({ dataBasePath });
}

export function buildDebugSharedRouteHostInputs(params: {
	enabled: boolean;
	analysisId: string;
	baseAnalysis: Analysis | null;
	baseModel: Group | null;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	metricType?: MetricType;
	colorMode: 'normalized' | 'discrete';
	utciRenderMode: UtciRenderMode;
	rendererBackend: UtciRendererBackend;
	rendererDevice?: GPUDevice;
	utciSurfaceBackend: 'dataTexture' | 'gpuNative';
}): LiveSelectedHourRouteInputs {
	const metricType = params.metricType ?? 'utci';
	const selectionKey =
		metricType === 'shading_index'
			? [params.analysisId, metricType, params.monthIndex].join('|')
			: [params.analysisId, params.monthIndex, params.hourIndex].join('|');
	return {
		enabled: params.enabled,
		analysisId: params.analysisId,
		baseAnalysis: params.baseAnalysis,
		baseModel: params.baseModel,
		metricType,
		selection: {
			monthIndex: params.monthIndex,
			hourIndex: params.hourIndex,
			timeIndex: params.timeIndex,
			selectionKey
		},
		colorMode: params.colorMode,
		utciRenderMode: params.utciRenderMode,
		rendererBackend: params.rendererBackend,
		rendererDevice: params.rendererDevice,
		utciSurfaceBackend: params.utciSurfaceBackend,
		comparison: {
			active: false,
			analysisId: null,
			sourceAnalysis: null,
			model: null,
			rendererDevice: params.rendererDevice
		}
	};
}

export function projectDebugSharedRouteSceneState(params: {
	enabled: boolean;
	baseAnalysis: Analysis | null;
	liveRouteState: LiveSelectedHourRouteState;
}): MainRouteLiveSceneProjection {
	return projectMainRouteLiveSceneState({
		useLiveUtciOnMainRoute: params.enabled,
		isComparing: false,
		baseAnalysis: params.baseAnalysis,
		comparisonAnalysis: null,
		liveRouteState: params.liveRouteState
	});
}

export function forwardDebugSharedBaseSurfaceDiagnostics(
	host: LiveSelectedHourRouteHost,
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
): void {
	host.handleBaseSurfaceDiagnostics(diagnostics);
}

export function releaseDebugSharedAcceptedGpuResidentOutput(
	host: LiveSelectedHourRouteHost,
	params: DebugRouteAcceptedGpuResidentOutputRelease
): void {
	host.releaseBaseAcceptedGpuResidentOutput(params);
}

export function forwardDebugRouteAcceptedGpuResidentOutputRelease(params: {
	release: DebugRouteAcceptedGpuResidentOutputRelease;
	releaseLegacyAcceptedOutput: (params: DebugRouteAcceptedGpuResidentOutputRelease) => void;
	releaseSharedAcceptedOutput: (params: DebugRouteAcceptedGpuResidentOutputRelease) => void;
	isLegacyRelease: (params: DebugRouteAcceptedGpuResidentOutputRelease) => boolean;
}): void {
	if (params.isLegacyRelease(params.release)) {
		params.releaseLegacyAcceptedOutput(params.release);
		return;
	}

	params.releaseSharedAcceptedOutput(params.release);
}

export function buildDebugSharedDiagnosticsPatch(params: {
	debugSharedRouteState: LiveSelectedHourRouteState;
	debugSharedBaseRenderContext: LiveSelectedHourPublishedRenderContext | null;
	debugSharedBasePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	debugOnDemandSelection: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
	};
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
}): {
	selectedHourEngine: 'shared-host';
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	surfaceRequestId: number | undefined;
	selectionKey: string | undefined;
	sceneSurfaceRequestId: number | undefined;
	sceneSelectionKey: string | undefined;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	renderContextTimeIndex: number | undefined;
	acceptedUtciRange: { min: number; max: number } | undefined;
	renderTransport: 'none' | 'compute-buffer-selected-hour' | 'cpu-uploaded-selected-hour';
	sameDeviceForComputeAndRender: boolean | null;
	utciSurfaceSource: string | undefined;
	selectedHourReadbackCount: undefined;
	visibleSelectedHourReadbackCount: number | undefined;
	selectedHourTransferCount: number | undefined;
	dataTextureBuildCount: number | undefined;
	gpuResidentCopyStatus: 'idle' | 'pending' | 'complete' | 'failed' | undefined;
	gpuResidentCopyError: string | undefined;
	gpuResidentCopyRequestId: number | undefined;
	selectedHourReadbackReasons: SelectedHourReadbackReason[];
	selectedHourReadbackReasonCounts: Partial<Record<SelectedHourReadbackReason, number>>;
} {
	const renderSurfaceDiagnostics = params.debugSharedRouteState.base.renderSurfaceDiagnostics;
	return {
		selectedHourEngine: 'shared-host',
		legacySelectedHourDispatchCount: params.legacySelectedHourDispatchCount,
		legacyScrubScheduleCount: params.legacyScrubScheduleCount,
		surfaceRequestId: params.debugSharedRouteState.baseSurfaceIdentity?.requestId,
		selectionKey: params.debugSharedRouteState.baseSurfaceIdentity?.selectionKey,
		sceneSurfaceRequestId: params.debugSharedRouteState.baseSceneSurfaceIdentity?.requestId,
		sceneSelectionKey: params.debugSharedRouteState.baseSceneSurfaceIdentity?.selectionKey,
		selectedMonthIndex: params.debugOnDemandSelection.monthIndex,
		selectedHourIndex: params.debugOnDemandSelection.hourIndex,
		selectedTimeIndex: params.debugOnDemandSelection.timeIndex,
		renderContextTimeIndex: params.debugSharedBaseRenderContext?.timeIndex,
		acceptedUtciRange: params.debugSharedBasePendingGpuResidentOutput?.utciRange,
		renderTransport:
			params.debugSharedRouteState.base.renderTransport === 'idle' ||
			params.debugSharedRouteState.base.renderTransport === 'live-render-pending'
				? 'none'
				: params.debugSharedRouteState.base.renderTransport,
		sameDeviceForComputeAndRender:
			params.debugSharedRouteState.base.sameDeviceForComputeAndRender,
		utciSurfaceSource: renderSurfaceDiagnostics.utciSurfaceSource,
		selectedHourReadbackCount: undefined,
		visibleSelectedHourReadbackCount:
			params.debugSharedRouteState.base.visibleSelectedHourReadbackCount,
		selectedHourTransferCount: renderSurfaceDiagnostics.selectedHourTransferCount,
		dataTextureBuildCount: renderSurfaceDiagnostics.dataTextureBuildCount,
		gpuResidentCopyStatus: renderSurfaceDiagnostics.gpuResidentCopyStatus,
		gpuResidentCopyError: renderSurfaceDiagnostics.gpuResidentCopyError,
		gpuResidentCopyRequestId: renderSurfaceDiagnostics.gpuResidentCopyRequestId,
		selectedHourReadbackReasons:
			params.debugSharedRouteState.base.selectedHourReadbackReasons,
		selectedHourReadbackReasonCounts:
			params.debugSharedRouteState.base.selectedHourReadbackReasonCounts
	};
}
