import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const protectedFiles = [
	'src/routes/+page.svelte',
	'src/routes/main/liveSelectedHour.ts',
	'src/routes/main/modelSelection.ts',
	'src/routes/main/tooltip.ts'
];

const forbiddenPatterns = [
	/\.bin\b/i,
	/\bparity\b/i,
	/Python/i,
	/loadReferenceFromFs/,
	/__onDemandPrototypeDiagnostics__/,
	/LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/
];

describe('main route live selected-hour source lock', () => {
	it.each(protectedFiles)('%s exists for the decomposition slice', (relativePath) => {
		expect(existsSync(resolve(viewerRoot, relativePath))).toBe(true);
	});

	it.each(protectedFiles)('%s stays free of debug-only selected-hour wiring', (relativePath) => {
		const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
		for (const pattern of forbiddenPatterns) {
			expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
		}
	});
});
