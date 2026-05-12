import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';

export const LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID = 'debug-legacy-selected-hour';

export type DebugRouteAcceptedGpuResidentOutputRelease = {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: 'copy-complete' | 'copy-failed' | 'superseded';
};

export function buildLegacyDebugSurfaceIdentity(
	params: {
		useDebugSharedSelectedHourHost: boolean;
		acceptedOutput: SelectedHourGpuResidentOutput | null;
		analysisId: string;
		selectedMonthIndex: number;
		selectedHourIndex: number;
		pendingRenderUpdateStartedAt?: number;
	}
): LiveSelectedHourSurfaceIdentity | null {
	const { useDebugSharedSelectedHourHost, acceptedOutput } = params;
	if (useDebugSharedSelectedHourHost || !acceptedOutput) {
		return null;
	}

	const selectionKey = [
		params.analysisId,
		acceptedOutput.monthIndex ?? params.selectedMonthIndex,
		acceptedOutput.hourIndex ?? params.selectedHourIndex
	].join('|');

	return {
		controllerIdentity: LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID,
		controllerInstanceId: 0,
		requestId: acceptedOutput.requestId,
		monthIndex: acceptedOutput.monthIndex,
		hourIndex: acceptedOutput.hourIndex,
		timeIndex: acceptedOutput.timeIndex,
		selectionKey,
		pendingRenderUpdateStartedAt: params.pendingRenderUpdateStartedAt,
		acceptedGpuResidentOutput: acceptedOutput
	};
}

export function isLegacyDebugAcceptedGpuResidentOutputRelease(
	params: DebugRouteAcceptedGpuResidentOutputRelease
): boolean {
	return params.controllerIdentity === LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID;
}

export function handleLegacyAcceptedGpuResidentOutputRelease(
	params: DebugRouteAcceptedGpuResidentOutputRelease,
	releaseAcceptedOutput: (key: {
		requestId: number;
		monthIndex: number;
		timeIndex: number;
	}) => void
): boolean {
	if (!isLegacyDebugAcceptedGpuResidentOutputRelease(params)) {
		return false;
	}

	releaseAcceptedOutput({
		requestId: params.requestId,
		monthIndex: params.monthIndex,
		timeIndex: params.timeIndex
	});
	return true;
}
