/**
 * Inspect WebGPU intermediates (solar/sky exposure) without asserting.
 * Run: cd viewer && npx playwright test tests/e2e/inspect-intermediates.spec.ts
 * Requires dev server (Playwright starts it if not running). Prints stats, probe check, sun samples, and distribution.
 */

import { test } from '@playwright/test';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
/** WebGPU loads in under 10s; wait a bit longer for full intermediates + file write. */
const INTERMEDIATES_WAIT_MS = 15_000;
const POLL_INTERVAL_MS = 500;

/** Solar shader probe: when PROBE_FORCE_WRITE is true, (0,0) is forced to 0.5 to verify compute→readback path. */
const PROBE_EXPECTED_VALUE = 0.5;

function stats(arr: number[]): { mean: number; min: number; max: number; std: number; n: number } {
	const n = arr.length;
	if (n === 0) return { mean: 0, min: 0, max: 0, std: 0, n: 0 };
	let sum = 0;
	let min = arr[0];
	let max = arr[0];
	for (let i = 0; i < n; i++) {
		const v = arr[i];
		sum += v;
		if (v < min) min = v;
		if (v > max) max = v;
	}
	const mean = sum / n;
	let sumSq = 0;
	for (let i = 0; i < n; i++) {
		const d = arr[i] - mean;
		sumSq += d * d;
	}
	const std = n > 1 ? Math.sqrt(sumSq / (n - 1)) : 0;
	return { mean, min, max, std, n };
}

function countNonZero(arr: number[]): number {
	let c = 0;
	for (let i = 0; i < arr.length; i++) if (arr[i] !== 0) c++;
	return c;
}

function firstNonZeroIndices(arr: number[], maxCount: number): number[] {
	const out: number[] = [];
	for (let i = 0; i < arr.length && out.length < maxCount; i++) if (arr[i] !== 0) out.push(i);
	return out;
}

/** Sample at 0, 25%, 50%, 75%, 99% of length so open vs shaded areas are both represented. */
function sampledValues(arr: number[], label: string): { index: number; value: number }[] {
	const n = arr.length;
	if (n === 0) return [];
	const indices = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), Math.floor(n * 0.99)];
	return indices.map((index) => ({ index, value: arr[index] ?? 0 }));
}

test('inspect WebGPU intermediates', async ({ page }) => {
	test.setTimeout(INTERMEDIATES_WAIT_MS + 10_000);

	await page.goto(
		`/debug?analysis=${encodeURIComponent('Ben-Gurion/20250815_grid_2m_fullday')}`
	);

	await page.waitForFunction(
		() => {
			const w = window as unknown as { __parityIntermediates__?: unknown; __parityIntermediatesError__?: string };
			return w.__parityIntermediates__ != null || w.__parityIntermediatesError__ != null;
		},
		{ timeout: INTERMEDIATES_WAIT_MS, polling: POLL_INTERVAL_MS }
	);

	const { intermediates: raw, error: readbackError, debug: rawDebug } = (await page.evaluate(() => {
		const w = window as unknown as {
			__parityIntermediates__?: {
				solarExposure: number[];
				skyExposure: number[];
				mrt?: number[];
				numPoints: number;
				numHours: number;
			};
			__parityIntermediatesError__?: string;
			__parityDebug__?: { sunVectorSamples: number[] | null; mrt?: number[]; weatherSample?: unknown[] };
		};
		return {
			intermediates: w.__parityIntermediates__ ?? null,
			error: w.__parityIntermediatesError__ ?? null,
			debug: w.__parityDebug__ ?? null
		};
	})) as {
		intermediates: {
			solarExposure: number[];
			skyExposure: number[];
			mrt?: number[];
			numPoints: number;
			numHours: number;
		} | null;
		error: string | null;
		debug: { sunVectorSamples: number[] | null; mrt?: number[]; weatherSample?: unknown[] } | null;
	};

	if (readbackError) {
		console.error('Readback error:', readbackError);
		return;
	}

	const data = raw;
	if (!data) {
		console.error('No __parityIntermediates__');
		return;
	}

	const { solarExposure, skyExposure, mrt: mrtArray, numPoints, numHours } = data;
	const solarS = stats(solarExposure);
	const skyS = stats(skyExposure);
	const mrtS = mrtArray && mrtArray.length > 0 ? stats(mrtArray) : null;
	const solarNonZero = countNonZero(solarExposure);
	const skyNonZero = countNonZero(skyExposure);
	const solarFirstNonZero = firstNonZeroIndices(solarExposure, 5);
	const skyFirstNonZero = firstNonZeroIndices(skyExposure, 5);

	console.log('\n--- WebGPU intermediates ---');
	console.log('numPoints:', numPoints, 'numHours:', numHours);

	// Probe: if solar shader has PROBE_FORCE_WRITE=true, index 0 should be 0.5. If we read 0, compute may not be writing to the buffer we read.
	const solarIndex0 = solarExposure[0] ?? 0;
	const probeOk = Math.abs(solarIndex0 - PROBE_EXPECTED_VALUE) < 0.001;
	console.log('\n[Probe] solarExposure[0] =', solarIndex0, probeOk ? '(expected 0.5: compute→readback path OK)' : '(if shader PROBE_FORCE_WRITE=true but we see 0, compute is not writing to the buffer we read)');

	// Sun vector samples (hours 0, 12, 23): expect non-zero and for daytime hours y > 0 (Y-up).
	if (rawDebug?.sunVectorSamples && rawDebug.sunVectorSamples.length >= 9) {
		const [x0, y0, z0, x12, y12, z12, x23, y23, z23] = rawDebug.sunVectorSamples;
		console.log('\n[Sun vectors] hour 0:  x=%s y=%s z=%s (y>0 = sun up)', x0.toFixed(4), y0.toFixed(4), z0.toFixed(4));
		console.log('[Sun vectors] hour 12: x=%s y=%s z=%s (y>0 = sun up)', x12.toFixed(4), y12.toFixed(4), z12.toFixed(4));
		console.log('[Sun vectors] hour 23: x=%s y=%s z=%s (y>0 = sun up)', x23.toFixed(4), y23.toFixed(4), z23.toFixed(4));
		if (y12 <= 0 && y0 <= 0 && y23 <= 0) {
			console.log('[Sun vectors] WARNING: all y <= 0 → shader treats all hours as night and writes 0.');
		}
	} else {
		console.log('\n[Sun vectors] no samples (__parityDebug__.sunVectorSamples missing)');
	}

	if (mrtS) {
		console.log('\nMRT (°C):');
		console.log('mrt:   n=%d mean=%s min=%s max=%s std=%s (plausible range e.g. 20–60 °C)', mrtS.n, mrtS.mean.toFixed(2), mrtS.min.toFixed(2), mrtS.max.toFixed(2), mrtS.std.toFixed(2));
	}
	if (rawDebug?.weatherSample && rawDebug.weatherSample.length > 0) {
		console.log('\nWeather sample (first', rawDebug.weatherSample.length, 'hours):');
		(rawDebug.weatherSample as Array<Record<string, number>>).forEach((row, i) => {
			console.log('  hour', i, row);
		});
	}

	console.log('\nStats:');
	console.log('solar: n=%d mean=%s min=%s max=%s std=%s nonZero=%d', solarS.n, solarS.mean.toFixed(4), solarS.min.toFixed(4), solarS.max.toFixed(4), solarS.std.toFixed(4), solarNonZero);
	console.log('sky:   n=%d mean=%s min=%s max=%s std=%s nonZero=%d', skyS.n, skyS.mean.toFixed(4), skyS.min.toFixed(4), skyS.max.toFixed(4), skyS.std.toFixed(4), skyNonZero);
	if (solarFirstNonZero.length > 0) console.log('solar first non-zero indices:', solarFirstNonZero);
	if (skyFirstNonZero.length > 0) console.log('sky first non-zero indices:', skyFirstNonZero);

	console.log('\nSamples across array (0%, 25%, 50%, 75%, 99% of length) — not just "first 20" so open vs shaded areas both appear:');
	console.log('solar:', sampledValues(solarExposure, 'solar').map(({ index, value }) => `[${index}]=${value}`).join(', '));
	console.log('sky:  ', sampledValues(skyExposure, 'sky').map(({ index, value }) => `[${index}]=${value}`).join(', '));

	// Optional: write to file for diff with reference
	const outPath = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday_webgpu_inspect.json');
	try {
		const fs = await import('node:fs');
		fs.writeFileSync(
			outPath,
			JSON.stringify(
				{
					numPoints,
					numHours,
					probeIndex0: solarIndex0,
					sunVectorSamples: rawDebug?.sunVectorSamples ?? null,
					weatherSample: rawDebug?.weatherSample ?? null,
					solarNonZero,
					skyNonZero,
					mrtStats: mrtS ?? null,
					solarExposure: solarExposure.slice(0, 1000),
					skyExposure: skyExposure.slice(0, 500),
					mrtSample: mrtArray?.slice(0, 100) ?? null,
					stats: { solar: solarS, sky: skyS },
					sampled: { solar: sampledValues(solarExposure, 'solar'), sky: sampledValues(skyExposure, 'sky') }
				},
				null,
				0
			),
			'utf8'
		);
		console.log('\nWrote sample to', outPath);
	} catch {
		// ignore
	}
});
