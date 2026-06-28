import { writable } from 'svelte/store';

export type PerformanceStatus = 'idle' | 'loading' | 'ready' | 'fallback' | 'error';

export interface UserFacingPerformanceSnapshot {
	status: PerformanceStatus;
	analysisId: string | null;
	projectLabel: string | null;
	pointCount: number | null;
	gridSizeMeters: number | null;
	selectedMonthIndex: number | null;
	selectedHourIndex: number | null;
	totalToVisibleMs: number | null;
	utciComputeMs: number | null;
	ownedGpuMemoryBytes: number | null;
	memoryScope: 'utci-owned-webgpu-buffers' | null;
	measuredAt: number | null;
	error: string | null;
}

export const EMPTY_PERFORMANCE_SNAPSHOT: UserFacingPerformanceSnapshot = {
	status: 'idle',
	analysisId: null,
	projectLabel: null,
	pointCount: null,
	gridSizeMeters: null,
	selectedMonthIndex: null,
	selectedHourIndex: null,
	totalToVisibleMs: null,
	utciComputeMs: null,
	ownedGpuMemoryBytes: null,
	memoryScope: null,
	measuredAt: null,
	error: null
};

export const performanceStore = writable<UserFacingPerformanceSnapshot>(
	EMPTY_PERFORMANCE_SNAPSHOT
);
