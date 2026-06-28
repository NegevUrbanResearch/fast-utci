import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const viewerRoot = resolve(__dirname, '../..');
const collectorPath = 'tests/e2e/main-route-transition-scrub-diagnostics.spec.ts';

function readCollector(): string {
	return readFileSync(resolve(viewerRoot, collectorPath), 'utf8');
}

describe('main-route transition scrub diagnostics collector source lock', () => {
	it('locks the artifact name and required transition scrub cases', () => {
		const source = readCollector();

		expect(source).toContain('main-route-transition-scrub-diagnostics.json');
		expect(source).toContain('main-route-transition-scrub-diagnostics-progress.json');
		expect(source).toContain('direct-nz-0_5m-default-chunked-2048');
		expect(source).toContain('bg-then-nz-0_5m-default-chunked-2048');
		expect(source).toContain('direct-nz-0_5m-chunked-2048');
		expect(source).toContain('bg-then-nz-0_5m-chunked-2048');
	});

	it('keeps main-route default and explicit chunked-2048 coverage', () => {
		const source = readCollector();

		expect(source).toContain("SOURCE_ROUTE = '/'");
		expect(source).toContain("utciExposureSchedule: 'chunked'");
		expect(source).toContain("utciExposureMaxWorkgroupsPerSlice: '2048'");
		expect(source).toContain("gridResolutionMeters: 0.5");
		expect(source).toContain('BG_ENTRY_GRID_RESOLUTION_METERS = 2');
		expect(source).toContain('__mainRouteDiagnosticsSetGridResolution');
		expect(source).toContain("analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2'");
	});

	it('locks comparable timing and render publication diagnostics', () => {
		const source = readCollector();

		expect(source).toContain('exposurePrecomputeMs');
		expect(source).toContain('exposureQueueWaitMs');
		expect(source).toContain('exposurePointChunks');
		expect(source).toContain('firstSelectedHourVisibleMs');
		expect(source).toContain('renderSceneSyncStartDelayMs');
		expect(source).toContain('renderSceneSyncTotalMs');
		expect(source).toContain('renderPublicationTimeline');
		expect(source).toContain('sceneReactiveToSyncQueuedMs');
		expect(source).toContain('sceneSyncQueuedToStartMs');
		expect(source).toContain('sessionSelectedDayRangeCacheKey');
		expect(source).toContain('sessionSelectedDayRangeCacheHit');
		expect(source).toContain('sessionSelectedDayRangeCacheSizeBefore');
		expect(source).toContain('sessionSelectedDayRangeCacheSizeAfter');
		expect(source).toContain('sessionSelectedDayRangeReadbackCount');
		expect(source).toContain('sessionSelectedDayRangeComputedHourCount');
		expect(source).toContain('sessionSelectedDayRangeResolutionPath');
		expect(source).toContain('sessionSelectedDayRangeSummaryReadbackCount');
		expect(source).toContain('sessionSelectedDayRangeSummaryReadbackBytes');
		expect(source).toContain('sessionSelectedDayRangeFullReadbackAvoidedCount');
		expect(source).toContain('sessionSelectedHourRangeResolutionPath');
		expect(source).toContain('sessionSelectedHourRangeReadbackCount');
		expect(source).toContain('sessionSelectedHourRangeCpuScanCount');
		expect(source).toContain('sessionSelectedHourRangeSummaryReadbackCount');
		expect(source).toContain('sessionSelectedHourRangeSummaryReadbackBytes');
		expect(source).toContain('sessionSelectedHourRangeFullReadbackAvoidedCount');
		expect(source).toContain('assertPerHourRangeResolutionProof');
		expect(source).toContain('setColorScaleMode');
		expect(source).toContain('colorMode');
		expect(source).toContain('per-hour-mode');
		expect(source).toContain('per-hour-hour-1');
		expect(source).toContain('compactPerHourSamples.length');
		expect(source).toContain('compact-gpu-summary');
		expect(source).toContain('cache-hit');
		expect(source).toContain('renderLayoutReuseAction');
		expect(source).toContain('renderLayoutReuseProofSource');
		expect(source).toContain('entryUrl: entry.entryUrl');
		expect(source).toContain('targetUrl: entry.targetUrl');
		expect(source).toContain('proofRouteUrl: sourceUrl');
	});
});
