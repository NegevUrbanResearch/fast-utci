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
		expect(source).toContain('direct-nz-0_5m-single-submit');
		expect(source).toContain('bg-then-nz-0_5m-single-submit');
		expect(source).toContain('direct-nz-0_5m-chunked-2048');
		expect(source).toContain('bg-then-nz-0_5m-chunked-2048');
	});

	it('keeps chunked-2048 mode query-gated on the main route', () => {
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
		expect(source).toContain('renderLayoutReuseAction');
		expect(source).toContain('renderLayoutReuseProofSource');
		expect(source).toContain('entryUrl: entry.entryUrl');
		expect(source).toContain('targetUrl: entry.targetUrl');
		expect(source).toContain('proofRouteUrl: sourceUrl');
	});
});
