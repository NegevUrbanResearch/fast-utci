import { describe, expect, it } from 'vitest';
import {
	buildMainRoutePerformanceSnapshot,
	formatDuration,
	formatMemory
} from '$lib/performance/mainRoutePerformanceTelemetry';

describe('mainRoutePerformanceTelemetry', () => {
	it('builds an end-user snapshot from cheap route diagnostics', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: {
				baseLiveReady: true,
				timings: {
					firstSelectedHourVisibleMs: 4200,
					oneHourDispatchMs: 86.5,
					renderSceneSyncTotalMs: 154.7
				},
				trackedGpuAllocationBytes: {
					persistentExposureBytes: 4_194_304,
					allHoursOutputBytes: 1_048_576,
					selectedHourOutputBytes: 131_072,
					selectedHourOutputBytesHighWatermark: 262_144,
					renderOwnedSelectedHourBytes: 524_288,
					renderOwnedSelectedHourBytesHighWatermark: 524_288,
					trackingScope: 'utci-owned-webgpu-buffers'
				}
			},
			now: 10000
		});

		expect(snapshot).toMatchObject({
			status: 'ready',
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			totalToVisibleMs: 4200,
			utciComputeMs: 86.5,
			ownedGpuMemoryBytes: 5_898_240,
			memoryScope: 'utci-owned-webgpu-buffers',
			measuredAt: 10000,
			error: null
		});
	});

	it('sums current tracked app-owned GPU memory without treating high-watermarks as current VRAM', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: {
				baseLiveReady: true,
				trackedGpuAllocationBytes: {
					persistentExposureBytes: 100,
					allHoursOutputBytes: 200,
					selectedHourOutputBytes: 300,
					selectedHourOutputBytesHighWatermark: 900,
					renderOwnedSelectedHourBytes: 400,
					renderOwnedSelectedHourBytesHighWatermark: 800,
					trackingScope: 'utci-owned-webgpu-buffers'
				}
			},
			now: 10000
		});

		expect(snapshot.ownedGpuMemoryBytes).toBe(1000);
	});

	it('marks fallback when the visible result is not the live ready path', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: {
				baseLiveReady: false,
				timings: {},
				trackedGpuAllocationBytes: undefined
			},
			now: 10000
		});

		expect(snapshot.status).toBe('fallback');
		expect(snapshot.totalToVisibleMs).toBeNull();
		expect(snapshot.ownedGpuMemoryBytes).toBeNull();
	});

	it('builds a loading snapshot when diagnostics are not available yet', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: null,
			now: 10000
		});

		expect(snapshot).toMatchObject({
			status: 'loading',
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			totalToVisibleMs: null,
			utciComputeMs: null,
			ownedGpuMemoryBytes: null,
			memoryScope: null,
			measuredAt: 10000,
			error: null
		});
	});

	it('builds an error snapshot when diagnostics report an error', () => {
		const snapshot = buildMainRoutePerformanceSnapshot({
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			projectLabel: 'Ben-Gurion',
			pointCount: 32123,
			gridSizeMeters: 2,
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			diagnostics: {
				baseLiveReady: false,
				error: 'device lost',
				timings: {
					firstSelectedHourVisibleMs: 4200,
					oneHourDispatchMs: 86.5,
					renderSceneSyncTotalMs: 154.7
				}
			},
			now: 10000
		});

		expect(snapshot).toMatchObject({
			status: 'error',
			totalToVisibleMs: 4200,
			utciComputeMs: 86.5,
			error: 'device lost'
		});
	});

	it('formats user-facing durations and memory without jargon', () => {
		expect(formatDuration(86.49)).toBe('86 ms');
		expect(formatDuration(86.5)).toBe('87 ms');
		expect(formatDuration(4200)).toBe('4.2 s');
		expect(formatDuration(null)).toBe('Measuring');
		expect(formatMemory(4_456_448)).toBe('4.3 MiB');
		expect(formatMemory(null)).toBe('Measuring');
	});
});
