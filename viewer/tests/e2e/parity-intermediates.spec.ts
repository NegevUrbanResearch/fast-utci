import { test, expect, type Page } from '@playwright/test';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { loadReferenceIntermediatesFromFs } from '../../src/lib/parity/loadReferenceIntermediatesFromFs';
import { compareIntermediatesStats } from '../../src/lib/parity/compareIntermediates';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const BASE_PATH = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday');
const ANALYSIS_SLUG = 'Ben-Gurion/20250815_grid_2m_fullday';
/** Max wait for WebGPU parity collection/readback in CI/dev environments. */
const INTERMEDIATES_WAIT_MS = Number(process.env.PARITY_E2E_WAIT_MS ?? '90000');
/** Poll often so we finish shortly after WebGPU is ready. */
const POLL_INTERVAL_MS = 300;
const INCLUDE_MRT = process.env.PARITY_E2E_INCLUDE_MRT === '1';
/** Distribution tolerance: mean and max of exposure (0-1) must match within this. */
const TOLERANCE_MEAN = 0.02;
const TOLERANCE_MAX = 0.05;
/** MRT in C: mean within 1 C, max within 2 C. */
const MRT_TOLERANCE_MEAN = 1.0;
const MRT_TOLERANCE_MAX = 2.0;

type BrowserIntermediates = {
	solarExposure: number[];
	skyExposure: number[];
	mrt?: number[];
	numPoints: number;
	numHours: number;
};

async function collectIntermediates(page: Page, includeMrt: boolean): Promise<{
	intermediates: BrowserIntermediates;
	debugMrt: number[] | null;
}> {
	await page.goto(
		`/debug?parity=1&analysis=${encodeURIComponent(ANALYSIS_SLUG)}`
	);
	await page.waitForFunction(
		() => {
			const w = window as unknown as {
				__parityIntermediates__?: unknown;
				__parityIntermediatesError__?: string;
				__parityCollectionError__?: string;
				__parityCollectionStatus__?: { state: 'running' | 'success' | 'error' | 'timeout' };
			};
			if (w.__parityIntermediatesError__) return true;
			if (w.__parityCollectionError__) return true;
			const status = w.__parityCollectionStatus__?.state;
			if (status === 'error' || status === 'timeout') return true;
			if (status === 'success') return w.__parityIntermediates__ != null;
			return false;
		},
		{ timeout: INTERMEDIATES_WAIT_MS, polling: POLL_INTERVAL_MS }
	);

	const { intermediates: rawIntermediates, intermediatesError, collectionError, status, debugMrt } = await page.evaluate(
		(includeMrtFlag) => {
			const w = window as unknown as {
				__parityIntermediates__?: BrowserIntermediates;
				__parityIntermediatesError__?: string;
				__parityCollectionError__?: string;
				__parityCollectionStatus__?: { state: 'running' | 'success' | 'error' | 'timeout' };
				__parityDebug__?: { mrt?: number[] };
			};
			const raw = w.__parityIntermediates__ ?? null;
			const intermediates = raw
				? {
						solarExposure: raw.solarExposure,
						skyExposure: raw.skyExposure,
						mrt: includeMrtFlag ? raw.mrt : undefined,
						numPoints: raw.numPoints,
						numHours: raw.numHours
				  }
				: null;
			return {
				intermediates,
				intermediatesError: w.__parityIntermediatesError__ ?? null,
				collectionError: w.__parityCollectionError__ ?? null,
				status: w.__parityCollectionStatus__?.state ?? null,
				debugMrt: includeMrtFlag ? w.__parityDebug__?.mrt ?? null : null
			};
		},
		includeMrt
	);
	if (intermediatesError || collectionError || status !== 'success' || !rawIntermediates) {
		throw new Error(
			[
				'WebGPU intermediate readback failed.',
				`status=${String(status)}`,
				`intermediatesError=${intermediatesError ?? 'null'}`,
				`collectionError=${collectionError ?? 'null'}`
			].join(' ')
		);
	}
	expect(rawIntermediates.solarExposure.some((v) => v !== 0), 'solar exposure should not be all zeros').toBe(true);
	expect(rawIntermediates.skyExposure.some((v) => v !== 0), 'sky exposure should not be all zeros').toBe(true);
	return { intermediates: rawIntermediates, debugMrt };
}

/**
 * Intermediate-stage validation: compare WebGPU solar/sky exposure to reference by distribution only.
 * This default check avoids MRT payload transfer to keep local loops fast and memory-safe.
 */
test.describe('WebGPU intermediate validation (Ben-Gurion base, statistical)', () => {
	test('solar and sky exposure distributions match reference', async ({ page }) => {
		test.setTimeout(INTERMEDIATES_WAIT_MS + 10_000);

		const hasSolar = existsSync(BASE_PATH + '_solar.json');
		const hasSky = existsSync(BASE_PATH + '_sky.json');
		if (!hasSolar && !hasSky) {
			test.skip();
			return;
		}

		const [refSolar, refSky] = await Promise.all([
			hasSolar ? loadReferenceIntermediatesFromFs(BASE_PATH, 'solar') : null,
			hasSky ? loadReferenceIntermediatesFromFs(BASE_PATH, 'sky') : null,
		]);

		const { intermediates } = await collectIntermediates(page, false);

		// Run both checks and collect failures so one doesn't mask the other.
		const failures: string[] = [];

		if (hasSolar && refSolar && 'solarExposure' in refSolar) {
			const webgpuSolar = intermediates.solarExposure;
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

		if (hasSky && refSky && 'skyExposure' in refSky) {
			const webgpuSky = intermediates.skyExposure;
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
		expect(
			failures,
			failures.length > 0 ? `Stage(s) failed:\n${failures.join('\n')}` : undefined
		).toEqual([]);
	});

	test('mrt distribution matches reference (opt-in)', async ({ page }) => {
		if (!INCLUDE_MRT) {
			test.skip();
			return;
		}
		test.setTimeout(INTERMEDIATES_WAIT_MS + 10_000);
		const hasMrt = existsSync(BASE_PATH + '_mrt.json');
		if (!hasMrt) {
			test.skip();
			return;
		}

		const refMrt = await loadReferenceIntermediatesFromFs(BASE_PATH, 'mrt');
		if (!('mrt' in refMrt)) {
			throw new Error('Reference MRT artifact missing mrt vector');
		}
		const { intermediates, debugMrt } = await collectIntermediates(page, true);
		const webgpuMrt = intermediates.mrt ?? debugMrt;
		expect(webgpuMrt, 'MRT readback missing despite enabled MRT parity check').toBeTruthy();
		expect((webgpuMrt ?? []).length, 'MRT readback vector should be non-empty').toBeGreaterThan(0);

		const result = compareIntermediatesStats({
			ref: refMrt.mrt,
			webgpu: webgpuMrt as number[],
			toleranceMean: MRT_TOLERANCE_MEAN,
			toleranceMax: MRT_TOLERANCE_MAX
		});
		if (!result.pass) {
			throw new Error(
				`mrt: meanDiff=${result.meanDiff.toFixed(4)} C (ref mean=${result.refStats.mean.toFixed(4)}, webgpu mean=${result.webgpuStats.mean.toFixed(4)}), maxDiff=${result.maxDiff.toFixed(4)} C`
			);
		}
	});
});
