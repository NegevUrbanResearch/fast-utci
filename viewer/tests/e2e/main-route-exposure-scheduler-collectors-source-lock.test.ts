import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const viewerRoot = resolve(__dirname, '../..');

const collectorFiles = [
	{
		label: 'visual freeze map',
		relativePath: 'tests/e2e/main-route-visual-freeze-map.spec.ts'
	},
	{
		label: 'cold-start waterfall',
		relativePath: 'tests/e2e/main-route-cold-start-waterfall.spec.ts'
	}
];

function readCollector(relativePath: string): string {
	return readFileSync(resolve(viewerRoot, relativePath), 'utf8');
}

function countMatches(source: string, pattern: RegExp): number {
	return [...source.matchAll(pattern)].length;
}

describe('main-route exposure scheduler collector source lock', () => {
	for (const collector of collectorFiles) {
		it(`${collector.label} uses the promoted default scheduler instead of stale slice comparisons`, () => {
			const source = readCollector(collector.relativePath);

			expect(source).not.toContain('chunked-8192');
			expect(source).not.toContain('chunked-4096');
			expect(source).not.toContain('chunked-2048');
			expect(source).not.toContain('utciExposureMaxWorkgroupsPerSlice');
			expect(source).not.toContain("utciExposureSchedule: 'chunked'");
			expect(source).not.toContain("utciExposureSchedule: 'single-submit'");
			expect(
				countMatches(source, /analysisId:\s*'Ness-Tziona\/exploded\/nes_tziona_unblock_2'/g)
			).toBeGreaterThanOrEqual(2);
			expect(countMatches(source, /gridResolutionMeters:\s*0\.5/g)).toBeGreaterThanOrEqual(2);
		});

		it(`${collector.label} preserves summarized diagnostics fields needed for exposure evidence`, () => {
			const source = readCollector(collector.relativePath);

			expect(source).toContain('exposureSchedulerBreathingTrace');
			expect(source).toContain('renderPublicationPreStorageMs');
			expect(source).toContain('renderCopyQueueDrainMs');
			expect(source).toContain('visibleSelectedHourReadbackCount');
			expect(source).toContain('baseSameDeviceForComputeAndRender');
		});
	}

	it('visual freeze map preserves browser gap fields needed for raf evidence', () => {
		const source = readCollector('tests/e2e/main-route-visual-freeze-map.spec.ts');

		expect(source).toContain('topRafGaps');
		expect(source).toContain('longTasks');
	});
});
