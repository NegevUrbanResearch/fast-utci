import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { loadReferenceIntermediatesFromFs } from '../../src/lib/parity/loadReferenceIntermediatesFromFs';
import { compareIntermediatesStats } from '../../src/lib/parity/compareIntermediates';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const BASE_PATH = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday');
/** Max wait for WebGPU to produce __parityIntermediates__ (loads in under 10s). */
const INTERMEDIATES_WAIT_MS = 15_000;
/** Poll often so we finish shortly after WebGPU is ready. */
const POLL_INTERVAL_MS = 300;
/** Distribution tolerance: mean and max of exposure (0–1) must match within this. */
const TOLERANCE_MEAN = 0.02;
const TOLERANCE_MAX = 0.05;
/** MRT in °C: mean within 1 °C, max within 2 °C. */
const MRT_TOLERANCE_MEAN = 1.0;
const MRT_TOLERANCE_MAX = 2.0;

/**
 * Intermediate-stage validation: compare WebGPU solar/sky exposure to reference by distribution only.
 * Single browser session: one page load, get both intermediates, then assert each stage if reference exists.
 */
test.describe('WebGPU intermediate validation (Ben-Gurion base, statistical)', () => {
	test('solar and sky exposure distributions match reference', async ({ page }) => {
		test.setTimeout(INTERMEDIATES_WAIT_MS + 10_000);

		const hasSolar = existsSync(BASE_PATH + '_solar.json');
		const hasSky = existsSync(BASE_PATH + '_sky.json');
		const hasMrt = existsSync(BASE_PATH + '_mrt.json');
		if (!hasSolar && !hasSky && !hasMrt) {
			test.skip();
			return;
		}

		const [refSolar, refSky, refMrt] = await Promise.all([
			hasSolar ? loadReferenceIntermediatesFromFs(BASE_PATH, 'solar') : null,
			hasSky ? loadReferenceIntermediatesFromFs(BASE_PATH, 'sky') : null,
			hasMrt ? loadReferenceIntermediatesFromFs(BASE_PATH, 'mrt') : null
		]);

		await page.goto(
			`/debug-webgpu-utci?analysis=${encodeURIComponent('Ben-Gurion/20250815_grid_2m_fullday')}`
		);

		await page.waitForFunction(
			() => {
				const w = window as unknown as { __parityIntermediates__?: unknown; __parityIntermediatesError__?: string };
				return w.__parityIntermediates__ != null || w.__parityIntermediatesError__ != null;
			},
			{ timeout: INTERMEDIATES_WAIT_MS, polling: POLL_INTERVAL_MS }
		);

		const { intermediates: rawIntermediates, error: readbackError, debug: rawDebug } = (await page.evaluate(() => {
			const w = window as unknown as {
				__parityIntermediates__?: {
					solarExposure: number[];
					skyExposure: number[];
					mrt?: number[];
					numPoints: number;
					numHours: number;
				};
				__parityIntermediatesError__?: string;
				__parityDebug__?: { mrt?: number[] };
			};
			return {
				intermediates: w.__parityIntermediates__ ?? null,
				error: w.__parityIntermediatesError__ ?? null,
				debug: (w as { __parityDebug__?: { mrt?: number[] } }).__parityDebug__ ?? null
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
			debug: { mrt?: number[] } | null;
		};
		if (readbackError) {
			throw new Error(`WebGPU readback failed: ${readbackError}`);
		}
		const intermediates = rawIntermediates;
		expect(intermediates).toBeDefined();

		// Run all stage checks and collect failures so solar doesn't block sky/MRT.
		const failures: string[] = [];

		if (hasSolar && refSolar && 'solarExposure' in refSolar) {
			const webgpuSolar = intermediates.solarExposure;
			if (!webgpuSolar.some((v) => v !== 0)) {
				failures.push(
					`WebGPU solar exposure is all zeros (numPoints=${intermediates.numPoints}, numHours=${intermediates.numHours}). Check exposure shader, BVH upload, and queue.onSubmittedWorkDone before readback.`
				);
			} else {
				const result = compareIntermediatesStats({
					ref: refSolar.solarExposure,
					webgpu: webgpuSolar,
					toleranceMean: TOLERANCE_MEAN,
					toleranceMax: TOLERANCE_MAX
				});
				if (!result.pass) {
					failures.push(
						`solar: meanDiff=${result.meanDiff.toFixed(4)} (ref mean=${result.refStats.mean.toFixed(4)}, webgpu mean=${result.webgpuStats.mean.toFixed(4)}), maxDiff=${result.maxDiff.toFixed(4)}`
					);
				}
			}
		}

		if (hasSky && refSky && 'skyExposure' in refSky) {
			const webgpuSky = intermediates.skyExposure;
			if (!webgpuSky.some((v) => v !== 0)) {
				failures.push(
					`WebGPU sky exposure is all zeros (numPoints=${intermediates.numPoints}). Check exposure shader, BVH upload, and queue.onSubmittedWorkDone before readback.`
				);
			} else {
				const result = compareIntermediatesStats({
					ref: refSky.skyExposure,
					webgpu: webgpuSky,
					toleranceMean: TOLERANCE_MEAN,
					toleranceMax: TOLERANCE_MAX
				});
				if (!result.pass) {
					failures.push(
						`sky: meanDiff=${result.meanDiff.toFixed(4)} (ref mean=${result.refStats.mean.toFixed(4)}, webgpu mean=${result.webgpuStats.mean.toFixed(4)}), maxDiff=${result.maxDiff.toFixed(4)}`
					);
				}
			}
		}

		const webgpuMrt = intermediates.mrt ?? rawDebug?.mrt;
		if (hasMrt && refMrt && 'mrt' in refMrt && webgpuMrt && webgpuMrt.length > 0) {
			const result = compareIntermediatesStats({
				ref: refMrt.mrt,
				webgpu: webgpuMrt,
				toleranceMean: MRT_TOLERANCE_MEAN,
				toleranceMax: MRT_TOLERANCE_MAX
			});
			if (!result.pass) {
				failures.push(
					`mrt: meanDiff=${result.meanDiff.toFixed(4)} °C (ref mean=${result.refStats.mean.toFixed(4)}, webgpu mean=${result.webgpuStats.mean.toFixed(4)}), maxDiff=${result.maxDiff.toFixed(4)} °C`
				);
			}
		}

		expect(
			failures,
			failures.length > 0 ? `Stage(s) failed:\n${failures.join('\n')}` : undefined
		).toEqual([]);
	});
});
