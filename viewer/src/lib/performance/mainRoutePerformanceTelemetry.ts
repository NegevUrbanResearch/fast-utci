import type {
	OnDemandTimings,
	TrackedGpuAllocationBytes
} from '$lib/compute/on-demand/onDemandDiagnostics';
import {
	EMPTY_PERFORMANCE_SNAPSHOT,
	type UserFacingPerformanceSnapshot
} from '$lib/stores/performanceStore';

export interface MainRoutePerformanceDiagnosticsLike {
	baseLiveReady?: boolean;
	timings?: OnDemandTimings;
	trackedGpuAllocationBytes?: TrackedGpuAllocationBytes;
	error?: string;
}

export interface BuildMainRoutePerformanceSnapshotParams {
	analysisId: string | null;
	projectLabel: string | null;
	pointCount: number | null;
	gridSizeMeters: number | null;
	selectedMonthIndex: number | null;
	selectedHourIndex: number | null;
	diagnostics: MainRoutePerformanceDiagnosticsLike | null | undefined;
	now: number;
}

export function getOwnedGpuMemoryBytes(
	tracked: TrackedGpuAllocationBytes | undefined
): number | null {
	if (!tracked) return null;
	return (
		tracked.persistentExposureBytes +
		tracked.allHoursOutputBytes +
		tracked.selectedHourOutputBytes +
		(tracked.renderOwnedSelectedHourBytes ?? 0)
	);
}

function getActiveRenderPreflightError(
	diagnostics: MainRoutePerformanceDiagnosticsLike
): string | null {
	const preflight =
		diagnostics.timings?.renderPublication?.renderAllocationPreflight;
	if (
		preflight?.status !== 'failed' ||
		preflight.renderTopology !== 'active-cells'
	) {
		return null;
	}
	const reasonText = preflight.failureReasons?.join('; ');
	return reasonText
		? `Active UTCI render allocation preflight failed: ${reasonText}.`
		: 'Active UTCI render allocation preflight failed.';
}

export function buildMainRoutePerformanceSnapshot(
	params: BuildMainRoutePerformanceSnapshotParams
): UserFacingPerformanceSnapshot {
	const diagnostics = params.diagnostics;
	if (!diagnostics) {
		return {
			...EMPTY_PERFORMANCE_SNAPSHOT,
			status: 'loading',
			analysisId: params.analysisId,
			projectLabel: params.projectLabel,
			pointCount: params.pointCount,
			gridSizeMeters: params.gridSizeMeters,
			selectedMonthIndex: params.selectedMonthIndex,
			selectedHourIndex: params.selectedHourIndex,
			measuredAt: params.now
		};
	}

	const error = diagnostics.error ?? getActiveRenderPreflightError(diagnostics);
	const status = error
		? 'error'
		: diagnostics.baseLiveReady
			? 'ready'
			: 'fallback';

	return {
		status,
		analysisId: params.analysisId,
		projectLabel: params.projectLabel,
		pointCount: params.pointCount,
		gridSizeMeters: params.gridSizeMeters,
		selectedMonthIndex: params.selectedMonthIndex,
		selectedHourIndex: params.selectedHourIndex,
		totalToVisibleMs: diagnostics.timings?.firstSelectedHourVisibleMs ?? null,
		utciComputeMs: diagnostics.timings?.oneHourDispatchMs ?? null,
		ownedGpuMemoryBytes: getOwnedGpuMemoryBytes(diagnostics.trackedGpuAllocationBytes),
		memoryScope: diagnostics.trackedGpuAllocationBytes?.trackingScope ?? null,
		measuredAt: params.now,
		error
	};
}

export function formatDuration(valueMs: number | null): string {
	if (valueMs === null || !Number.isFinite(valueMs)) return 'Measuring';
	if (valueMs < 1000) return `${Math.round(valueMs)} ms`;
	return `${(valueMs / 1000).toFixed(1)} s`;
}

export function formatMemory(valueBytes: number | null): string {
	if (valueBytes === null || !Number.isFinite(valueBytes)) return 'Measuring';
	const mib = valueBytes / (1024 * 1024);
	return `${mib.toFixed(1)} MiB`;
}
