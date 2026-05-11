import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const requiredMainRoutePaths = [
	'src/routes/+page.svelte',
	'src/routes/main/liveSelectedHour.ts',
	'src/routes/main/modelSelection.ts',
	'src/routes/main/tooltip.ts',
	'src/routes/main/MainRouteViewport.svelte',
	'src/routes/main/modelLifecycle.ts',
	'src/routes/main/MainRouteTooltipLayer.svelte',
	'src/routes/main/MainRouteOverlays.svelte'
];

const optionalMainRoutePaths: string[] = [];

const debugOnlyPatterns = [
	/\.bin/i,
	/\bparity\b/i,
	/Python/i,
	/loadReferenceFromFs/i,
	/__onDemandPrototypeDiagnostics__/i,
	/LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/i
];

describe('main route live selected-hour source lock', () => {
	it.each(requiredMainRoutePaths)('%s exists for the decomposition slice', (relativePath) => {
		expect(existsSync(resolve(viewerRoot, relativePath))).toBe(true);
	});

	it.each(requiredMainRoutePaths)('%s stays free of debug-only selected-hour wiring', (relativePath) => {
		const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
		for (const pattern of debugOnlyPatterns) {
			expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
		}
	});

	it.each(optionalMainRoutePaths)(
		'%s stays free of debug-only selected-hour wiring when present',
		(relativePath) => {
			const absolutePath = resolve(viewerRoot, relativePath);
			if (!existsSync(absolutePath)) return;
			const source = readFileSync(absolutePath, 'utf8');
			for (const pattern of debugOnlyPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		}
	);
});
