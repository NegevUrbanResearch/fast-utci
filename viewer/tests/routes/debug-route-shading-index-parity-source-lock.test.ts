import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const mainRouteFiles = [
	'src/routes/+page.svelte',
	'src/routes/main/liveSelectedHour.ts',
	'src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts'
];

const debugOnlyParityPatterns = [
	/__debugShadingIndexParity__/,
	/compareShadingIndex/,
	/DebugShadingIndexParitySnapshot/,
	/compare-shading-index-parity/,
	/shading-index-parity/
];

describe('debug route Shading Index parity source lock', () => {
	it('keeps the E2E parity collector on the debug route', () => {
		const source = readFileSync(
			resolve(viewerRoot, 'tests/e2e/debug-route-shading-index-parity.spec.ts'),
			'utf8'
		);
		expect(source).toMatch(/\/debug\?parity=0/);
		expect(source).toMatch(/shadingIndexParity=1/);
		expect(source).not.toMatch(/page\.goto\(\s*`\/\?/);
		expect(source).toMatch(/metric=shading_index/);
	});

	it('exposes the shading parity snapshot only from the debug route', () => {
		const debugSource = readFileSync(resolve(viewerRoot, 'src/routes/debug/+page.svelte'), 'utf8');
		expect(debugSource).toMatch(/__debugShadingIndexParity__/);
		expect(debugSource).toMatch(/source: "debug-shared-host"/);
		expect(debugSource).toMatch(/metricType: "shading_index"/);
		expect(debugSource).toMatch(/shadingIndexParity=1/);
	});

	it.each(mainRouteFiles)('%s stays free of debug shading parity symbols', (relativePath) => {
		const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
		for (const pattern of debugOnlyParityPatterns) {
			expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
		}
	});
});
