import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const viewerRoot = resolve(__dirname, '../..');

const mainRouteProtectedFiles = [
	'src/routes/+page.svelte',
	'src/routes/main/liveSelectedHour.ts',
	'src/routes/main/modelSelection.ts',
	'src/routes/main/tooltip.ts',
	'src/lib/compute/selected-hour/liveSelectedHourRouteHost.ts',
	'src/lib/compute/selected-hour/liveSelectedHourRouteProjection.ts'
];

const debugRouteHelperFiles = [
	'src/routes/debug/+page.svelte',
	'src/routes/debug/queryState.ts',
	'src/routes/debug/selectedHourMode.ts',
	'src/routes/debug/sharedHostWiring.ts',
	'src/routes/debug/legacySelectedHourWiring.ts'
];

const optionalFutureDebugRouteHelperFiles = [
	'src/routes/debug/parityRuntime.ts'
];

const debugRouteHelperImportPatterns = [
	/routes\/debug\/queryState/,
	/routes\/debug\/selectedHourMode/,
	/routes\/debug\/sharedHostWiring/,
	/routes\/debug\/legacySelectedHourWiring/,
	/routes\/debug\/parityRuntime/,
	/debugRouteQueryState/,
	/debugRouteSelectedHourMode/,
	/debugRouteSharedHostWiring/,
	/debugRouteLegacySelectedHourWiring/,
	/debugRouteParityRuntime/
];

const routeOnlyPatternAllowlist = [
	{
		pattern: /LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/,
		allowedFiles: [
			'src/routes/debug/legacySelectedHourWiring.ts'
		]
	},
	{
		pattern: /\.bin\b/i,
		allowedFiles: ['src/routes/debug/+page.svelte']
	},
	{
		pattern: /\bparity\b/i,
		allowedFiles: ['src/routes/debug/+page.svelte']
	},
	{
		pattern: /Python/i,
		allowedFiles: ['src/routes/debug/+page.svelte']
	}
];

describe('debug route decomposition source lock', () => {
	it.each(debugRouteHelperFiles)(
		'%s exists for the debug-route decomposition slice',
		(relativePath) => {
			expect(existsSync(resolve(viewerRoot, relativePath))).toBe(true);
		}
	);

	it.each(optionalFutureDebugRouteHelperFiles)(
		'%s is optional until parity runtime extraction happens',
		(relativePath) => {
			expect(typeof existsSync(resolve(viewerRoot, relativePath))).toBe('boolean');
		}
	);

	it.each(mainRouteProtectedFiles)(
		'%s does not import debug route helper modules',
		(relativePath) => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			for (const pattern of debugRouteHelperImportPatterns) {
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		}
	);

	it.each(mainRouteProtectedFiles)(
		'%s stays free of legacy debug selected-hour route constants',
		(relativePath) => {
			const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
			expect(source).not.toMatch(/LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/);
		}
	);

	for (const { pattern, allowedFiles } of routeOnlyPatternAllowlist) {
		it(`keeps ${pattern} confined to the explicit debug-route allowlist`, () => {
			const scannedFiles = [...mainRouteProtectedFiles, ...debugRouteHelperFiles];
			for (const relativePath of scannedFiles) {
				if (allowedFiles.includes(relativePath)) continue;
				const source = readFileSync(resolve(viewerRoot, relativePath), 'utf8');
				expect(source, `${relativePath} should not match ${pattern}`).not.toMatch(pattern);
			}
		});
	}
});
