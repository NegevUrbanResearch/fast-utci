import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { writeFileSync } from 'node:fs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
const PARITY_BASE_PATH = process.env.PARITY_BASE_PATH || DEFAULT_BASE_PATH;
const basePath = resolve(REPO_ROOT, PARITY_BASE_PATH);

/** Total Tregenza weight; normalizing sky to 0–1 (idempotent if page already sends normalized). */
const TOTAL_TREGENZA_WEIGHT = 145.2488;

function normalizeSky(raw: number[]): number[] {
	return raw.map((v) => Math.max(0, Math.min(1, v / TOTAL_TREGENZA_WEIGHT)));
}

/**
 * WebGPU collect: load debug page, wait for parity data, write _webgpu_*.json files to disk.
 * Set PARITY_BASE_PATH (relative to repo root) to change output directory.
 */
test.describe('Collect WebGPU parity to files', () => {
	test('wait for parity data and write WebGPU JSON files', async ({ page }) => {
		test.setTimeout(120_000);

		// Analysis slug for URL: e.g. "Ben-Gurion/20250815_grid_2m_fullday"
		const analysisSlug = PARITY_BASE_PATH.replace(/^data[/\\]analyses[/\\]/, '').replace(/\\/g, '/');
		const url = `/debug-webgpu-utci?analysis=${encodeURIComponent(analysisSlug)}`;
		await page.goto(url);

		await page.waitForFunction(
			() => {
				const w = window as unknown as {
					__parityResults__?: unknown;
					__parityIntermediates__?: unknown;
					__parityIntermediatesError__?: string;
				};
				if (w.__parityIntermediatesError__) return true;
				return w.__parityResults__ != null && w.__parityIntermediates__ != null;
			},
			{ timeout: 100_000 }
		);

		const errorMsg = await page.evaluate(() => (window as unknown as { __parityIntermediatesError__?: string }).__parityIntermediatesError__);
		if (errorMsg) {
			throw new Error(`Parity intermediates error: ${errorMsg}`);
		}

		const { parityResults, parityIntermediates } = (await page.evaluate(() => {
			const w = window as unknown as {
				__parityResults__?: { utciByHour: number[][]; numPoints: number; numHours: number };
				__parityIntermediates__?: {
					numPoints: number;
					numHours: number;
					solarExposure: number[];
					skyExposure: number[];
					mrt?: number[];
				};
			};
			return {
				parityResults: w.__parityResults__,
				parityIntermediates: w.__parityIntermediates__,
			};
		})) as {
			parityResults: { utciByHour: number[][]; numPoints: number; numHours: number };
			parityIntermediates: {
				numPoints: number;
				numHours: number;
				solarExposure: number[];
				skyExposure: number[];
				mrt?: number[];
			};
		};

		const normalizedSky = normalizeSky(parityIntermediates.skyExposure);

		writeFileSync(
			`${basePath}_webgpu_solar.json`,
			JSON.stringify(
				{
					numPositions: parityIntermediates.numPoints,
					numHours: parityIntermediates.numHours,
					solarExposure: parityIntermediates.solarExposure,
				},
				null,
				0
			)
		);
		writeFileSync(
			`${basePath}_webgpu_sky.json`,
			JSON.stringify(
				{
					numPositions: parityIntermediates.numPoints,
					skyExposure: normalizedSky,
				},
				null,
				0
			)
		);
		if (parityIntermediates.mrt != null) {
			writeFileSync(
				`${basePath}_webgpu_mrt.json`,
				JSON.stringify(
					{
						numPositions: parityIntermediates.numPoints,
						numHours: parityIntermediates.numHours,
						mrt: parityIntermediates.mrt,
					},
					null,
				0
				)
			);
		}
		const utciByHour = parityResults.utciByHour;
		let min = Infinity;
		let max = -Infinity;
		let sum = 0;
		let count = 0;
		for (const hour of utciByHour) {
			for (const v of hour) {
				if (Number.isFinite(v)) {
					min = Math.min(min, v);
					max = Math.max(max, v);
					sum += v;
					count++;
				}
			}
		}
		const mean = count > 0 ? sum / count : 0;
		writeFileSync(
			`${basePath}_webgpu_utci.json`,
			JSON.stringify(
				{
					numPoints: parityResults.numPoints,
					numHours: parityResults.numHours,
					utciByHour: parityResults.utciByHour,
					utci_range: { min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max, mean },
				},
				null,
				0
			)
		);
	});
});
