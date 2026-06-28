import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const strictProtectedFiles = [
	'src/routes/+page.svelte',
	'src/lib/diagnostics/mainRouteUtciDiagnostics.ts',
	'src/lib/performance/mainRoutePerformanceTelemetry.ts'
];

const sharedProtectedFiles = [
	'src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts',
	'src/lib/compute/selected-hour/liveSelectedHourRouteProjection.ts',
	'src/lib/components/scene/utciSurfaceSync.ts'
];

const strictForbiddenPatterns = [
	/\.bin\b/i,
	/['"]\$lib\/debug/,
	/debugWebgpuUtci/i,
	/loadReferenceFromFs/,
	/loadValidationData/,
	/compareWithValidation/,
	/calculateAvgMeanDiffAllHours/,
	/readbackForComparison/,
	/\bparity\b/i,
	/pythonBin/i,
	/binComparison/i,
	/__onDemandPrototypeDiagnostics__/,
	/parityMode/i,
	/Python/i,
	/\brunAll\b/,
	/performance\.memory/
];

const sharedForbiddenPatterns = [
	/\.bin\b/i,
	/['"]\$lib\/debug/,
	/debugWebgpuUtci/i,
	/loadReferenceFromFs/,
	/readbackForComparison/,
	/\bparity\b/i,
	/\brunAll\b/
];

describe('main route debug boundary source lock', () => {
	for (const relativePath of strictProtectedFiles) {
		it(`${relativePath} stays free of debug-only bin, parity, and Python behavior`, () => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			for (const pattern of strictForbiddenPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		});
	}

	it('PerformancePanel source stays free of validation, bin, and memory polling hooks when present', () => {
		const relativePath = 'src/lib/components/ui/PerformancePanel.svelte';
		const absolutePath = resolve(viewerRoot, relativePath);
		if (!existsSync(absolutePath)) {
			expect(existsSync(resolve(viewerRoot, 'src/routes/+page.svelte'))).toBe(true);
			return;
		}
		const source = readFileSync(absolutePath, 'utf8');
		for (const pattern of strictForbiddenPatterns) {
			expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
		}
	});

	for (const relativePath of sharedProtectedFiles) {
		it(`${relativePath} stays free of debug-only imports and runtime baseline hooks`, () => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			for (const pattern of sharedForbiddenPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		});
	}
});
