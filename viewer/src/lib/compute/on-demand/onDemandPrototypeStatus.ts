import type { OnDemandRuntimeDiagnostics } from '$lib/compute/on-demand/onDemandDiagnostics';

export type OnDemandPrototypeStatus =
	| 'idle'
	| 'diagnostics'
	| 'ready'
	| 'unsupported'
	| 'error';

type UtciRenderResolved = 'dataTexture' | 'gpuNative';

export type OnDemandPrototypeStatusDiagnostics = Pick<
	Partial<OnDemandRuntimeDiagnostics>,
	| 'error'
	| 'navigatorGpu'
	| 'oneHourOutputBytes'
	| 'path'
	| 'renderTransport'
	| 'rendererBackend'
	| 'usedExposureOnlyPrecompute'
	| 'usedRunAllForSelectedHour'
> & {
	bridgeAttached?: boolean;
	liveAnalysisConstructedForSelectedHour?: boolean;
	utciRenderResolved?: UtciRenderResolved;
	utciSurfaceSource?: string;
	visibleColorVariance?: number;
};

export type OnDemandPrototypeStatusInputs = {
	diagnostics: OnDemandPrototypeStatusDiagnostics;
	syntheticBridgeEnabled: boolean;
	strictExposureOnlyEnabled: boolean;
	compareOneHourEnabled: boolean;
	hasOnDemandPrototypeComparison: boolean;
	compareHoursEnabled: boolean;
	hasOnDemandMultiHourComparison: boolean;
	compareMonthHoursEnabled: boolean;
	hasCompletedOnDemandMonthHourComparison: boolean;
};

function hasStrongRuntimeWebgpuProof(diagnostics: OnDemandPrototypeStatusDiagnostics): boolean {
	return (
		diagnostics.rendererBackend === 'webgpu' ||
		diagnostics.renderTransport === 'compute-buffer-selected-hour' ||
		diagnostics.utciSurfaceSource === 'compute-buffer-selected-hour'
	);
}

export function deriveOnDemandPrototypeStatus({
	diagnostics,
	syntheticBridgeEnabled,
	strictExposureOnlyEnabled,
	compareOneHourEnabled,
	hasOnDemandPrototypeComparison,
	compareHoursEnabled,
	hasOnDemandMultiHourComparison,
	compareMonthHoursEnabled,
	hasCompletedOnDemandMonthHourComparison
}: OnDemandPrototypeStatusInputs): OnDemandPrototypeStatus {
	if (diagnostics.error) {
		return 'error';
	}

	if (!diagnostics.navigatorGpu && !hasStrongRuntimeWebgpuProof(diagnostics)) {
		return syntheticBridgeEnabled ? 'error' : 'unsupported';
	}

	if (syntheticBridgeEnabled) {
		return diagnostics.rendererBackend === 'webgpu' &&
			diagnostics.bridgeAttached === true &&
			(diagnostics.visibleColorVariance ?? 0) > 0
			? 'ready'
			: 'diagnostics';
	}

	if (strictExposureOnlyEnabled) {
		return diagnostics.path === 'exposure-only-f32' &&
			diagnostics.usedExposureOnlyPrecompute === true &&
			diagnostics.usedRunAllForSelectedHour === false &&
			diagnostics.liveAnalysisConstructedForSelectedHour === false &&
			(diagnostics.oneHourOutputBytes ?? 0) > 0 &&
			(!compareHoursEnabled || hasOnDemandMultiHourComparison) &&
			(!compareMonthHoursEnabled || hasCompletedOnDemandMonthHourComparison)
			? 'ready'
			: 'diagnostics';
	}

	return diagnostics.rendererBackend === 'webgpu' &&
		(!compareOneHourEnabled || hasOnDemandPrototypeComparison)
		? 'ready'
		: 'diagnostics';
}
