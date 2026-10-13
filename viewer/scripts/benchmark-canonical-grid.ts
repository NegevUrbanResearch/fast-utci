import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import type { CanonicalGridParams } from '../src/lib/compute/core/canonicalGrid';

const script = fileURLToPath(import.meta.url);
const viewerRoot = resolve(dirname(script), '..');
const cases = [
	{
		name: 'ben-gurion',
		counts: [104445, 1662657],
		bounds: {
			x_min: -3636.616943359375,
			x_max: -2648.616943359375,
			y_min: -613.2862548828125,
			y_max: -193.2862548828125,
			z: 1.5000000001117586
		}
	},
	{
		name: 'ness-tziona',
		counts: [511840, 8171761],
		bounds: {
			x_min: -1886.1756591796875,
			x_max: -768.1756591796875,
			y_min: -1036.610595703125,
			y_max: 789.389404296875,
			z: 1.5
		}
	}
].flatMap(area => [2, 0.5].map((gridSize, index) => ({
	id: `${area.name}-${gridSize}m`,
	expectedPointCount: area.counts[index],
	params: {
		bounds: area.bounds,
		gridSize,
		coordinateSystem: 'xy_ground'
	} satisfies CanonicalGridParams
})));

function positiveInteger(value: string, flag: string): number {
	const parsed = Number(value);
	if (!Number.isSafeInteger(parsed) || parsed <= 0)
		throw new Error(`${flag} requires a positive integer`);
	return parsed;
}

function parseFlags() {
	const flags = new Map<string, string>();
	const allowed = new Set(['--module', '--output', '--samples', '--case']);
	const args = process.argv.slice(2);
	for (let index = 0; index < args.length; index += 2) {
		const flag = args[index];
		const value = args[index + 1];
		if (!allowed.has(flag) || !value || value.startsWith('--') || flags.has(flag)) {
			throw new Error(`Invalid or duplicate flag: ${flag}`);
		}
		flags.set(flag, value);
	}
	return flags;
}

async function main() {
	const flags = parseFlags();
	const modulePath = resolve(flags.get('--module') ?? resolve(viewerRoot, 'src/lib/compute/core/canonicalGrid.ts'));
	const sourceHash = createHash('sha256').update(readFileSync(modulePath)).digest('hex');
	const samples = positiveInteger(flags.get('--samples') ?? '3', '--samples');
	if (flags.has('--case')) {
		if (flags.has('--output') || flags.has('--samples'))
			throw new Error('Worker --case cannot use --output or --samples');
		const selected = cases.find(value => value.id === flags.get('--case'));
		if (!selected)
			throw new Error(`Unknown case: ${flags.get('--case')}`);
		const module = await import(pathToFileURL(modulePath).href);
		if (typeof module.canonicalGridPoints !== 'function')
			throw new Error('Module must export canonicalGridPoints');
		if (!global.gc)
			throw new Error('Worker requires --expose-gc');
		global.gc();
		const before = process.memoryUsage();
		const start = performance.now();
		const result = module.canonicalGridPoints(selected.params);
		const durationMs = performance.now() - start;
		const after = process.memoryUsage();
		if (!(result.points instanceof Float32Array) || result.numPoints !== selected.expectedPointCount || result.points.length !== result.numPoints * 3) {
			throw new Error('Module returned an invalid grid');
		}
		const sha256 = createHash('sha256').update(new Uint8Array(result.points.buffer, result.points.byteOffset, result.points.byteLength)).digest('hex');
		console.log(JSON.stringify({
			caseId: selected.id,
			input: selected.params,
			durationMs,
			pointCount: result.numPoints,
			outputBytes: result.points.byteLength,
			sha256,
			memoryBefore: {
				heapUsed: before.heapUsed,
				arrayBuffers: before.arrayBuffers
			},
			memoryAfter: {
				heapUsed: after.heapUsed,
				arrayBuffers: after.arrayBuffers
			},
			maxRSSKiB: process.resourceUsage().maxRSS,
			nodeVersion: process.version,
			modulePath,
			sourceHash
		}));
		return;
	}
	const measurements = [];
	for (const selected of cases) {
		for (let sample = 1; sample <= samples; sample++) {
			const child = spawnSync(process.execPath, ['--expose-gc', '--import', 'tsx', script,
				'--module', modulePath, '--case', selected.id], {
				cwd: viewerRoot,
				encoding: 'utf8',
				maxBuffer: 1024 * 1024
			});
			if (child.error || child.status !== 0)
				throw new Error(child.error?.message ?? child.stderr ?? `Worker exited ${child.status}`);
			measurements.push({
				sample,
				...JSON.parse(child.stdout)
			});
		}
	}
	const output = JSON.stringify({
		samplesPerCase: samples,
		measurements
	}, null, 2) + '\n';
	if (flags.has('--output'))
		writeFileSync(resolve(flags.get('--output')!), output);
	else
		process.stdout.write(output);
}
main().catch(error => {
	console.error(error);
	process.exitCode = 1;
});
