import type {
	LiveSelectedHourControllerSurfaceDiagnostics,
	LiveSelectedHourRenderTransport
} from '$lib/compute/liveSelectedHourController';
import type {
	WebgpuLargeBufferDeviceLimits,
	WebgpuLargeBufferRequiredLimits
} from '$lib/compute/webgpuDeviceLimits';
import type { UtciRendererBackend, UtciRenderMode } from '$lib/utciRenderMode';

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
	baseRenderContextTimeIndex?: number;
	baseAcceptedUtciRange?: { min: number; max: number };
	comparisonSurfaceRequestId?: number;
	comparisonSelectionKey?: string;
	comparisonSameDeviceForComputeAndRender: boolean | null;
};

export function buildMainRouteUtciDiagnostics(
	inputs: MainRouteUtciDiagnosticsInputs
): MainRouteUtciDiagnosticsPayload | undefined {
	if (!inputs.enabled) return undefined;

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
			inputs.comparisonSurfaceDiagnostics.gpuResidentCopyRequestId
	};
}
