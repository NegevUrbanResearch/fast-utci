import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const strictProtectedFiles = [
	'src/routes/+page.svelte',
	'src/lib/diagnostics/mainRouteUtciDiagnostics.ts'
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
	/readbackForComparison/,
	/\bparity\b/i,
	/pythonBin/i,
	/binComparison/i,
	/__onDemandPrototypeDiagnostics__/,
	/parityMode/i,
	/Python/i,
	/\brunAll\b/
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

	for (const relativePath of sharedProtectedFiles) {
		it(`${relativePath} stays free of debug-only imports and runtime baseline hooks`, () => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			for (const pattern of sharedForbiddenPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		});
	}
});
