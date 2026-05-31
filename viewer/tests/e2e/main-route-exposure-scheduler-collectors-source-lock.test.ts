import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const viewerRoot = resolve(__dirname, '../..');

const collectorFiles = [
	{
		label: 'visual freeze map',
		relativePath: 'tests/e2e/main-route-visual-freeze-map.spec.ts',
		caseIdPrefix: 'ness-tziona-0_5m-chunked',
		hasLiteralCaseIds: true
	},
	{
		label: 'cold-start waterfall',
		relativePath: 'tests/e2e/main-route-cold-start-waterfall.spec.ts',
		caseIdPrefix: 'chunked',
		hasLiteralCaseIds: false
	}
];

const schedulerSlices = ['8192', '4096', '2048'] as const;

function readCollector(relativePath: string): string {
	return readFileSync(resolve(viewerRoot, relativePath), 'utf8');
}

function countMatches(source: string, pattern: RegExp): number {
	return [...source.matchAll(pattern)].length;
}

describe('main-route exposure scheduler collector source lock', () => {
	for (const collector of collectorFiles) {
		it(`${collector.label} includes distinct query-gated Ness-Tziona 0.5m chunked scheduler cases`, () => {
			const source = readCollector(collector.relativePath);

			for (const slice of schedulerSlices) {
				if (collector.hasLiteralCaseIds) {
					expect(
						source,
						`${collector.relativePath} should include a ${slice} slice case id`
					).toContain(`${collector.caseIdPrefix}-${slice}`);
				} else {
					expect(
						source,
						`${collector.relativePath} should derive chunked case suffixes from max workgroups`
					).toContain('`${schedule}-${maxWorkgroups}`');
				}
				expect(
					source,
					`${collector.relativePath} should include a ${slice} slice query param`
				).toContain(`utciExposureMaxWorkgroupsPerSlice: '${slice}'`);
			}

			expect(countMatches(source, /utciExposureSchedule:\s*'chunked'/g)).toBeGreaterThanOrEqual(3);
			expect(countMatches(source, /analysisId:\s*'Ness-Tziona\/exploded\/nes_tziona_unblock_2'/g)).toBeGreaterThanOrEqual(3);
			expect(countMatches(source, /gridResolutionMeters:\s*0\.5/g)).toBeGreaterThanOrEqual(3);
		});
	}
});
