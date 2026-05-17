import type {
	LiveSelectedHourControllerSurfaceDiagnostics,
	LiveSelectedHourRenderTransport
} from '$lib/compute/selected-hour/liveSelectedHourController';
import {
	buildSelectedHourRuntimeContract,
	type SelectedHourReadbackInstrumentation,
	type SelectedHourReadbackReason,
	type SelectedHourRenderTransport,
	type SelectedHourRuntimeContract
} from '$lib/diagnostics/selectedHourRuntimeContract';
import type {
	WebgpuLargeBufferDeviceLimits,
	WebgpuLargeBufferRequiredLimits
} from '$lib/compute/gpu/webgpuDeviceLimits';
import type {
	OnDemandTimings,
	TrackedGpuAllocationBytes
} from '$lib/compute/on-demand/onDemandDiagnostics';
import {
	copyRenderPublicationDiagnostics as copySelectedHourRenderPublicationDiagnostics,
	type SelectedHourRenderPublicationDiagnostics
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import type { UtciRendererBackend, UtciRenderMode } from '$lib/utciRenderMode';
import type { ColorMode } from '$lib/types/viewer';
import type { TooltipInteractionDiagnostics } from '$lib/services/tooltipService';
import type { CameraInteractionDiagnostics } from '$lib/services/cameraInteractionTelemetry';

type MainRouteUtciDiagnosticsTimings = Omit<OnDemandTimings, 'renderPublication'> & {
	renderPublication?: SelectedHourRenderPublicationDiagnostics;
};

export type MainRouteUtciDiagnosticsPayload = {
	utciOnDemand: 'f32';
	utciRenderRequested: UtciRenderMode;
	utciRenderResolved: 'dataTexture' | 'gpuNative';
	rendererBackend: UtciRendererBackend;
	rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
	rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
	lastGpuResidentCopyFailureError?: string;
	lastGpuResidentCopyFailureRequestId?: number;
	baseRenderTransport: LiveSelectedHourRenderTransport;
	comparisonRenderTransport: LiveSelectedHourRenderTransport;
	baseLiveReady: boolean;
	comparisonLiveReady: boolean;
	baseSurfaceRequestId?: number;
	baseSelectionKey?: string;
	baseSceneSurfaceRequestId?: number;
	baseSceneSelectionKey?: string;
	baseSameDeviceForComputeAndRender: boolean | null;
	baseSelectedMonthIndex: number;
	baseSelectedHourIndex: number;
	baseSelectedTimeIndex: number;
	baseColorMode?: ColorMode;
	basePointCount?: number | null;
	baseMetadataGridSize?: number | null;
	baseRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	comparisonSurfaceRequestId?: number;
	comparisonSelectionKey?: string;
	comparisonSameDeviceForComputeAndRender: boolean | null;
	comparisonUtciSurfaceSource?: string;
	comparisonSelectedHourTransferCount?: number;
	comparisonDataTextureBuildCount?: number;
	comparisonGpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	comparisonGpuResidentCopyError?: string;
	comparisonGpuResidentCopyRequestId?: number;
	tooltipInteraction?: TooltipInteractionDiagnostics & {
		hoverSampleCount: number;
	};
	cameraInteraction?: CameraInteractionDiagnostics;
	timings?: MainRouteUtciDiagnosticsTimings;
	trackedGpuAllocationBytes?: TrackedGpuAllocationBytes;
	visibleSelectedHourReadbackCount?: number;
	readbackInstrumentation?: SelectedHourReadbackInstrumentation;
	selectedHourReadbackReasons?: SelectedHourReadbackReason[];
	selectedHourReadbackReasonCounts?: Partial<Record<SelectedHourReadbackReason, number>>;
	comparisonSelectedHourReadbackReasons?: SelectedHourReadbackReason[];
	comparisonSelectedHourReadbackReasonCounts?: Partial<
		Record<SelectedHourReadbackReason, number>
	>;
	selectedHourRuntimeContract: SelectedHourRuntimeContract;
};

export type MainRouteUtciDiagnosticsInputs = {
	enabled: boolean;
	utciOnDemand: 'f32';
	utciRenderRequested: UtciRenderMode;
	utciRenderResolved: 'dataTexture' | 'gpuNative';
	rendererBackend: UtciRendererBackend;
	rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
	rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
	baseSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	comparisonSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	lastBaseGpuResidentCopyFailure?: { error?: string; requestId?: number };
	baseRenderTransport: LiveSelectedHourRenderTransport;
	comparisonRenderTransport: LiveSelectedHourRenderTransport;
	baseLiveReady: boolean;
	comparisonLiveReady: boolean;
	baseSurfaceRequestId?: number;
	baseSelectionKey?: string;
	baseSceneSurfaceRequestId?: number;
	baseSceneSelectionKey?: string;
	baseSameDeviceForComputeAndRender: boolean | null;
	baseSelectedMonthIndex: number;
	baseSelectedHourIndex: number;
	baseSelectedTimeIndex: number;
	baseColorMode?: ColorMode;
	basePointCount?: number | null;
	baseMetadataGridSize?: number | null;
	baseRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	comparisonSurfaceRequestId?: number;
	comparisonSelectionKey?: string;
	comparisonSameDeviceForComputeAndRender: boolean | null;
	tooltipInteraction?: TooltipInteractionDiagnostics & {
		hoverSampleCount: number;
	};
	cameraInteraction?: CameraInteractionDiagnostics;
	timings?: MainRouteUtciDiagnosticsTimings;
	trackedGpuAllocationBytes?: TrackedGpuAllocationBytes;
	visibleSelectedHourReadbackCount?: number;
	readbackInstrumentation?: SelectedHourReadbackInstrumentation;
	selectedHourReadbackReasons?: SelectedHourReadbackReason[];
	selectedHourReadbackReasonCounts?: Partial<Record<SelectedHourReadbackReason, number>>;
	comparisonSelectedHourReadbackReasons?: SelectedHourReadbackReason[];
	comparisonSelectedHourReadbackReasonCounts?: Partial<
		Record<SelectedHourReadbackReason, number>
	>;
};

function toSelectedHourRenderTransport(
	transport: LiveSelectedHourRenderTransport | string | undefined
): SelectedHourRenderTransport {
	if (
		transport === 'cpu-uploaded-selected-hour' ||
		transport === 'compute-buffer-selected-hour'
	) {
		return transport;
	}
	return 'none';
}

function mergeSelectedHourReadbackReasons(
	baseReasons: SelectedHourReadbackReason[] | undefined,
	comparisonReasons: SelectedHourReadbackReason[] | undefined
): SelectedHourReadbackReason[] | undefined {
	const merged = [...(baseReasons ?? []), ...(comparisonReasons ?? [])];
	return merged.length > 0 ? merged : undefined;
}

function mergeSelectedHourReadbackReasonCounts(
	baseCounts: Partial<Record<SelectedHourReadbackReason, number>> | undefined,
	comparisonCounts: Partial<Record<SelectedHourReadbackReason, number>> | undefined
): Partial<Record<SelectedHourReadbackReason, number>> | undefined {
	const merged: Partial<Record<SelectedHourReadbackReason, number>> = {};
	for (const counts of [baseCounts, comparisonCounts]) {
		for (const [reason, count] of Object.entries(counts ?? {}) as [
			SelectedHourReadbackReason,
			number | undefined
		][]) {
			if (typeof count !== 'number') continue;
			merged[reason] = (merged[reason] ?? 0) + count;
		}
	}
	return Object.keys(merged).length > 0 ? merged : undefined;
}

function copyRenderPublicationDiagnostics(
	renderPublication: MainRouteUtciDiagnosticsTimings['renderPublication']
): MainRouteUtciDiagnosticsTimings['renderPublication'] {
	return copySelectedHourRenderPublicationDiagnostics(renderPublication);
}

function copyDiagnosticsTimings(
	timings: MainRouteUtciDiagnosticsTimings | undefined
): MainRouteUtciDiagnosticsTimings | undefined {
	if (!timings) return undefined;
	return {
		...timings,
		renderPublication: copyRenderPublicationDiagnostics(timings.renderPublication)
	};
}

export function buildMainRouteUtciDiagnostics(
	inputs: MainRouteUtciDiagnosticsInputs
): MainRouteUtciDiagnosticsPayload | undefined {
	if (!inputs.enabled) return undefined;

	const selectedHourReadbackReasons = mergeSelectedHourReadbackReasons(
		inputs.selectedHourReadbackReasons,
		inputs.comparisonSelectedHourReadbackReasons
	);
	const selectedHourReadbackReasonCounts = mergeSelectedHourReadbackReasonCounts(
		inputs.selectedHourReadbackReasonCounts,
		inputs.comparisonSelectedHourReadbackReasonCounts
	);

	return {
		utciOnDemand: inputs.utciOnDemand,
		utciRenderRequested: inputs.utciRenderRequested,
		utciRenderResolved: inputs.utciRenderResolved,
		rendererBackend: inputs.rendererBackend,
		rendererRequiredLimits: inputs.rendererRequiredLimits,
		rendererDeviceLimits: inputs.rendererDeviceLimits,
		utciSurfaceSource: inputs.baseSurfaceDiagnostics.utciSurfaceSource,
		selectedHourTransferCount: inputs.baseSurfaceDiagnostics.selectedHourTransferCount,
		dataTextureBuildCount: inputs.baseSurfaceDiagnostics.dataTextureBuildCount,
		gpuResidentCopyStatus: inputs.baseSurfaceDiagnostics.gpuResidentCopyStatus,
		gpuResidentCopyError: inputs.baseSurfaceDiagnostics.gpuResidentCopyError,
		gpuResidentCopyRequestId: inputs.baseSurfaceDiagnostics.gpuResidentCopyRequestId,
		lastGpuResidentCopyFailureError: inputs.lastBaseGpuResidentCopyFailure?.error,
		lastGpuResidentCopyFailureRequestId: inputs.lastBaseGpuResidentCopyFailure?.requestId,
		baseRenderTransport: inputs.baseRenderTransport,
		comparisonRenderTransport: inputs.comparisonRenderTransport,
		baseLiveReady: inputs.baseLiveReady,
		comparisonLiveReady: inputs.comparisonLiveReady,
		baseSurfaceRequestId: inputs.baseSurfaceRequestId,
		baseSelectionKey: inputs.baseSelectionKey,
		baseSceneSurfaceRequestId: inputs.baseSceneSurfaceRequestId,
		baseSceneSelectionKey: inputs.baseSceneSelectionKey,
		baseSameDeviceForComputeAndRender: inputs.baseSameDeviceForComputeAndRender,
		baseSelectedMonthIndex: inputs.baseSelectedMonthIndex,
		baseSelectedHourIndex: inputs.baseSelectedHourIndex,
		baseSelectedTimeIndex: inputs.baseSelectedTimeIndex,
		baseColorMode: inputs.baseColorMode,
		basePointCount: inputs.basePointCount,
		baseMetadataGridSize: inputs.baseMetadataGridSize,
		baseRenderContextTimeIndex: inputs.baseRenderContextTimeIndex,
		baseAcceptedUtciRange: inputs.baseAcceptedUtciRange,
		comparisonSurfaceRequestId: inputs.comparisonSurfaceRequestId,
		comparisonSelectionKey: inputs.comparisonSelectionKey,
		comparisonSameDeviceForComputeAndRender:
			inputs.comparisonSameDeviceForComputeAndRender,
		comparisonUtciSurfaceSource: inputs.comparisonSurfaceDiagnostics.utciSurfaceSource,
		comparisonSelectedHourTransferCount:
			inputs.comparisonSurfaceDiagnostics.selectedHourTransferCount,
		comparisonDataTextureBuildCount:
			inputs.comparisonSurfaceDiagnostics.dataTextureBuildCount,
		comparisonGpuResidentCopyStatus:
			inputs.comparisonSurfaceDiagnostics.gpuResidentCopyStatus,
		comparisonGpuResidentCopyError:
			inputs.comparisonSurfaceDiagnostics.gpuResidentCopyError,
		comparisonGpuResidentCopyRequestId:
			inputs.comparisonSurfaceDiagnostics.gpuResidentCopyRequestId,
		tooltipInteraction: inputs.tooltipInteraction,
		cameraInteraction: inputs.cameraInteraction,
		timings: copyDiagnosticsTimings(inputs.timings),
		trackedGpuAllocationBytes: inputs.trackedGpuAllocationBytes
			? { ...inputs.trackedGpuAllocationBytes }
			: undefined,
		selectedHourReadbackReasons,
		selectedHourReadbackReasonCounts,
		comparisonSelectedHourReadbackReasons: inputs.comparisonSelectedHourReadbackReasons,
		comparisonSelectedHourReadbackReasonCounts:
			inputs.comparisonSelectedHourReadbackReasonCounts,
		selectedHourRuntimeContract: buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: toSelectedHourRenderTransport(inputs.baseRenderTransport),
			utciSurfaceSource: toSelectedHourRenderTransport(
				inputs.baseSurfaceDiagnostics.utciSurfaceSource
			),
			sameDeviceForComputeAndRender: inputs.baseSameDeviceForComputeAndRender === true,
			dataTextureBuildCount: inputs.baseSurfaceDiagnostics.dataTextureBuildCount,
			visibleSelectedHourReadbackCount: inputs.visibleSelectedHourReadbackCount,
			readbackInstrumentation: inputs.readbackInstrumentation ?? 'not-instrumented',
			requestId: inputs.baseSurfaceRequestId,
			sceneRequestId: inputs.baseSceneSurfaceRequestId,
			selectionKey: inputs.baseSelectionKey,
			sceneSelectionKey: inputs.baseSceneSelectionKey,
			readbackReasons: selectedHourReadbackReasons,
			readbackReasonCounts: selectedHourReadbackReasonCounts
		})
	};
}
