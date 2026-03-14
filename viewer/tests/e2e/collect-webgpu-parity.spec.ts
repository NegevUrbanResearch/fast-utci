import { test } from '@playwright/test';
import { resolve } from 'node:path';
import { writeFileSync } from 'node:fs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
const PARITY_BASE_PATH = process.env.PARITY_BASE_PATH || DEFAULT_BASE_PATH;
const basePath = resolve(REPO_ROOT, PARITY_BASE_PATH);
const COLLECT_WAIT_MS = 180_000;

/**
 * WebGPU collect: load debug page, wait for parity data, write _webgpu_*.json files to disk.
 * Set PARITY_BASE_PATH (relative to repo root) to change output directory.
 */
test.describe('Collect WebGPU parity to files', () => {
	test('wait for parity data and write WebGPU JSON files', async ({ page }) => {
		test.setTimeout(240_000);
		const pageErrors: string[] = [];
		const failedRequests: string[] = [];
		const phaseLogs: string[] = [];
		page.on('console', (msg) => {
			const text = msg.text();
			if (text.includes('[parity:phase]')) phaseLogs.push(text);
		});
		page.on('pageerror', (err) => {
			pageErrors.push(err.message);
		});
		page.on('requestfailed', (request) => {
			const failure = request.failure();
			failedRequests.push(`${request.method()} ${request.url()} :: ${failure?.errorText ?? 'unknown request failure'}`);
		});

		// Analysis slug for URL: e.g. "Ben-Gurion/20250815_grid_2m_fullday"
		const analysisSlug = PARITY_BASE_PATH.replace(/^data[/\\]analyses[/\\]/, '').replace(/\\/g, '/');
		const url = `/debug-webgpu-utci?analysis=${encodeURIComponent(analysisSlug)}`;
		await page.goto(url);

		try {
			await page.waitForFunction(
				() => {
					const w = window as unknown as {
						__parityResults__?: unknown;
						__parityIntermediates__?: unknown;
						__parityIntermediatesError__?: string;
						__parityCollectionError__?: string;
						__parityCollectionStatus__?: {
							state: 'running' | 'success' | 'error' | 'timeout';
						};
					};
					if (w.__parityIntermediatesError__) return true;
					if (w.__parityCollectionError__) return true;
					if (w.__parityCollectionStatus__?.state === 'error') return true;
					if (w.__parityCollectionStatus__?.state === 'timeout') return true;
					if (w.__parityCollectionStatus__?.state === 'success') {
						return w.__parityResults__ != null && w.__parityIntermediates__ != null;
					}
					return false;
				},
				{ timeout: COLLECT_WAIT_MS, polling: 1000 }
			);
		} catch (waitErr) {
			const snapshot = page.isClosed()
				? { pageClosed: true as const }
				: await page.evaluate(() => {
						const w = window as unknown as {
							__parityCollectionStatus__?: unknown;
							__parityCollectionError__?: string;
							__parityIntermediatesError__?: string;
							__parityCollectionLog__?: unknown;
						};
						return {
							pageClosed: false as const,
							status: w.__parityCollectionStatus__ ?? null,
							collectionError: w.__parityCollectionError__ ?? null,
							intermediatesError: w.__parityIntermediatesError__ ?? null,
							log: w.__parityCollectionLog__ ?? null
						};
					});
			throw new Error(
				[
					`Timed out waiting for parity collection readiness: ${waitErr instanceof Error ? waitErr.message : String(waitErr)}`,
					`snapshot=${JSON.stringify(snapshot)}`,
					`phaseLogs=${JSON.stringify(phaseLogs.slice(-40))}`,
					`pageErrors=${JSON.stringify(pageErrors.slice(-10))}`,
					`failedRequests=${JSON.stringify(failedRequests.slice(-10))}`
				].join('\n')
			);
		}

		const readiness = await page.evaluate(() => {
			const w = window as unknown as {
				__parityResults__?: unknown;
				__parityIntermediates__?: unknown;
				__parityIntermediatesError__?: string;
				__parityCollectionError__?: string;
				__parityCollectionStatus__?: {
					runId: number;
					state: 'running' | 'success' | 'error' | 'timeout';
					phase: string;
					startedAt: number;
					updatedAt: number;
					message?: string;
				};
				__parityCollectionLog__?: unknown;
			};
			return {
				hasResults: w.__parityResults__ != null,
				hasIntermediates: w.__parityIntermediates__ != null,
				intermediatesError: w.__parityIntermediatesError__ ?? null,
				collectionError: w.__parityCollectionError__ ?? null,
				status: w.__parityCollectionStatus__ ?? null,
				log: w.__parityCollectionLog__ ?? null
			};
		});
		if (
			readiness.collectionError ||
			readiness.intermediatesError ||
			!readiness.hasResults ||
			!readiness.hasIntermediates ||
			readiness.status?.state !== 'success'
		) {
			throw new Error(
				[
					'Parity collection did not complete successfully.',
					`readiness=${JSON.stringify(readiness)}`,
					`phaseLogs=${JSON.stringify(phaseLogs.slice(-40))}`,
					`pageErrors=${JSON.stringify(pageErrors.slice(-10))}`,
					`failedRequests=${JSON.stringify(failedRequests.slice(-10))}`
				].join('\n')
			);
		}

		const t0 = Date.now();
		console.log(`[collect] parity status success; starting export at ${new Date(t0).toISOString()}`);

		const intermediatesJson = await page.evaluate(() => {
			const w = window as unknown as {
				__parityIntermediates__?: {
					numPoints: number;
					numHours: number;
					solarExposure: number[];
					skyExposure: number[];
					mrt?: number[];
					shortErf?: number[];
					longErf?: number[];
					shortDmrt?: number[];
					longDmrt?: number[];
				};
			};
			if (!w.__parityIntermediates__) {
				throw new Error('Missing __parityIntermediates__ at export time');
			}
			return JSON.stringify(w.__parityIntermediates__);
		});
		console.log(`[collect] intermediates JSON pulled in ${Date.now() - t0}ms`);
		const parityIntermediates = JSON.parse(intermediatesJson) as {
			numPoints: number;
			numHours: number;
			solarExposure: number[];
			skyExposure: number[];
			mrt?: number[];
			shortErf?: number[];
			longErf?: number[];
			shortDmrt?: number[];
			longDmrt?: number[];
		};

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
					// Contract: skyExposure is already normalized once in debug page to [0, 1].
					skyExposure: parityIntermediates.skyExposure,
				},
				null,
				0
			)
		);
		console.log(`[collect] wrote solar/sky in ${Date.now() - t0}ms`);
		if (parityIntermediates.mrt != null) {
			writeFileSync(
				`${basePath}_webgpu_mrt.json`,
				JSON.stringify(
					{
						numPositions: parityIntermediates.numPoints,
						numHours: parityIntermediates.numHours,
						mrt: parityIntermediates.mrt,
						...(parityIntermediates.shortErf
							? { short_erf: parityIntermediates.shortErf }
							: {}),
						...(parityIntermediates.longErf
							? { long_erf: parityIntermediates.longErf }
							: {}),
						...(parityIntermediates.shortDmrt
							? { short_dmrt: parityIntermediates.shortDmrt }
							: {}),
						...(parityIntermediates.longDmrt
							? { long_dmrt: parityIntermediates.longDmrt }
							: {}),
					},
					null,
				0
				)
			);
		}
		console.log(`[collect] wrote mrt (if present) in ${Date.now() - t0}ms`);
		const resultsJson = await page.evaluate(() => {
			const w = window as unknown as {
				__parityResults__?: {
					utciByHour: number[][];
					positions?: number[];
					numPoints: number;
					numHours: number;
				};
			};
			if (!w.__parityResults__) {
				throw new Error('Missing __parityResults__ at export time');
			}
			return JSON.stringify({
				utciByHour: w.__parityResults__.utciByHour,
				numPoints: w.__parityResults__.numPoints,
				numHours: w.__parityResults__.numHours
			});
		});
		const positionsJson = await page.evaluate(() => {
			const w = window as unknown as {
				__parityResults__?: {
					positions?: number[];
				};
			};
			return JSON.stringify(w.__parityResults__?.positions ?? null);
		});
		console.log(`[collect] results JSON pulled in ${Date.now() - t0}ms`);
		const parityResults = JSON.parse(resultsJson) as {
			utciByHour: number[][];
			numPoints: number;
			numHours: number;
		};
		const parityPositions = JSON.parse(positionsJson) as number[] | null;
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
					...(parityPositions ? { positions: parityPositions } : {}),
					utciByHour: parityResults.utciByHour,
					utci_range: { min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max, mean },
				},
				null,
				0
			)
		);
		console.log(`[collect] wrote utci in ${Date.now() - t0}ms`);
	});
});
