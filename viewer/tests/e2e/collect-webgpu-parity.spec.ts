import { expect, test } from '@playwright/test';
import { resolve } from 'node:path';
import { readFileSync, writeFileSync } from 'node:fs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
const PARITY_BASE_PATH = process.env.PARITY_BASE_PATH || DEFAULT_BASE_PATH;
const basePath = resolve(REPO_ROOT, PARITY_BASE_PATH);
const COLLECT_WAIT_MS = 180_000;
const COLLECT_MODE = process.env.PARITY_COLLECT_MODE === 'normal' ? 'normal' : 'parity';

function expectNumberArrayLength(arr: unknown, expected: number, label: string): asserts arr is number[] {
	expect(Array.isArray(arr), `${label} should be number[]`).toBe(true);
	expect((arr as unknown[]).length, `${label}.length mismatch`).toBe(expected);
}

/**
 * WebGPU collect: load debug page, wait for data, write _webgpu_*.json files to disk.
 * Set PARITY_BASE_PATH (relative to repo root) to change output directory.
 * Set PARITY_COLLECT_MODE=normal to collect the app-visible August slice from 12-month normal mode.
 */
test.describe('Collect WebGPU parity to files', () => {
	test('wait for WebGPU data and write JSON files', async ({ page }) => {
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
		const url =
			COLLECT_MODE === 'normal'
				? `/debug?collect=normal&analysis=${encodeURIComponent(analysisSlug)}`
				: `/debug?parity=1&analysis=${encodeURIComponent(analysisSlug)}`;
		await page.goto(url);

		try {
			await page.waitForFunction(
				(mode) => {
					const w = window as unknown as {
						__parityResults__?: unknown;
						__parityIntermediates__?: unknown;
						__normalUtciResults__?: unknown;
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
						if (mode === 'normal') {
							return w.__normalUtciResults__ != null;
						}
						return w.__parityResults__ != null && w.__parityIntermediates__ != null;
					}
					return false;
				},
				COLLECT_MODE,
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
				__normalUtciResults__?: unknown;
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
				hasNormalResults: w.__normalUtciResults__ != null,
				intermediatesError: w.__parityIntermediatesError__ ?? null,
				collectionError: w.__parityCollectionError__ ?? null,
				status: w.__parityCollectionStatus__ ?? null,
				log: w.__parityCollectionLog__ ?? null
			};
		});
		if (
			readiness.collectionError ||
			readiness.intermediatesError ||
			(COLLECT_MODE === 'normal'
				? !readiness.hasNormalResults
				: (!readiness.hasResults || !readiness.hasIntermediates)) ||
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
		console.log(`[collect] ${COLLECT_MODE} status success; starting export at ${new Date(t0).toISOString()}`);

		if (COLLECT_MODE === 'normal') {
			const normalResultsJson = await page.evaluate(() => {
				const w = window as unknown as {
					__normalUtciResults__?: {
						utciByHour: number[][];
						positions?: number[];
						numPoints: number;
						numHours: number;
						monthIndex: number;
					};
				};
				if (!w.__normalUtciResults__) {
					throw new Error('Missing __normalUtciResults__ at export time');
				}
				return JSON.stringify(w.__normalUtciResults__);
			});
			const normalResults = JSON.parse(normalResultsJson) as {
				utciByHour: number[][];
				positions?: number[];
				numPoints: number;
				numHours: number;
				monthIndex: number;
			};
			expect(normalResults.monthIndex).toBe(7);
			expect(normalResults.numHours).toBe(24);
			expect(normalResults.utciByHour.length).toBe(24);
			for (let hourIdx = 0; hourIdx < normalResults.utciByHour.length; hourIdx++) {
				expectNumberArrayLength(normalResults.utciByHour[hourIdx], normalResults.numPoints, `normal utciByHour[${hourIdx}]`);
			}
			if (normalResults.positions != null) {
				expectNumberArrayLength(normalResults.positions, normalResults.numPoints * 3, 'normal positions');
			}
			let min = Infinity;
			let max = -Infinity;
			let sum = 0;
			let count = 0;
			for (const hour of normalResults.utciByHour) {
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
				`${basePath}_webgpu_normal_utci.json`,
				JSON.stringify(
					{
						numPoints: normalResults.numPoints,
						numHours: normalResults.numHours,
						monthIndex: normalResults.monthIndex,
						...(normalResults.positions ? { positions: normalResults.positions } : {}),
						utciByHour: normalResults.utciByHour,
						utci_range: { min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max, mean },
					},
					null,
					0
				)
			);
			const writtenNormalUtci = JSON.parse(readFileSync(`${basePath}_webgpu_normal_utci.json`, 'utf8')) as Record<string, unknown>;
			expect(writtenNormalUtci.numPoints).toBe(normalResults.numPoints);
			expect(writtenNormalUtci.numHours).toBe(24);
			expect(writtenNormalUtci.monthIndex).toBe(7);
			expect(Array.isArray(writtenNormalUtci.utciByHour)).toBe(true);
			expect((writtenNormalUtci.utciByHour as unknown[]).length).toBe(24);
			console.log(`[collect] wrote normal UTCI in ${Date.now() - t0}ms`);
			return;
		}

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
		const expectedPointwiseLen = parityIntermediates.numPoints * parityIntermediates.numHours;
		expectNumberArrayLength(parityIntermediates.solarExposure, expectedPointwiseLen, 'solarExposure');
		expectNumberArrayLength(parityIntermediates.skyExposure, parityIntermediates.numPoints, 'skyExposure');
		if (parityIntermediates.mrt != null) {
			expectNumberArrayLength(parityIntermediates.mrt, expectedPointwiseLen, 'mrt');
		}
		if (parityIntermediates.shortErf != null) {
			expectNumberArrayLength(parityIntermediates.shortErf, expectedPointwiseLen, 'shortErf');
		}
		if (parityIntermediates.longErf != null) {
			expectNumberArrayLength(parityIntermediates.longErf, expectedPointwiseLen, 'longErf');
		}
		if (parityIntermediates.shortDmrt != null) {
			expectNumberArrayLength(parityIntermediates.shortDmrt, expectedPointwiseLen, 'shortDmrt');
		}
		if (parityIntermediates.longDmrt != null) {
			expectNumberArrayLength(parityIntermediates.longDmrt, expectedPointwiseLen, 'longDmrt');
		}

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
		expect(utciByHour.length).toBe(parityResults.numHours);
		for (let hourIdx = 0; hourIdx < utciByHour.length; hourIdx++) {
			expectNumberArrayLength(utciByHour[hourIdx], parityResults.numPoints, `utciByHour[${hourIdx}]`);
		}
		if (parityPositions != null) {
			expectNumberArrayLength(parityPositions, parityResults.numPoints * 3, 'positions');
		}
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
		const writtenMrtPath = `${basePath}_webgpu_mrt.json`;
		if (parityIntermediates.mrt != null) {
			const writtenMrt = JSON.parse(readFileSync(writtenMrtPath, 'utf8')) as Record<string, unknown>;
			expect(writtenMrt.numPositions).toBe(parityIntermediates.numPoints);
			expect(writtenMrt.numHours).toBe(parityIntermediates.numHours);
			expectNumberArrayLength(writtenMrt.mrt, expectedPointwiseLen, 'written mrt');
			if (parityIntermediates.shortErf != null) {
				expectNumberArrayLength(writtenMrt.short_erf, expectedPointwiseLen, 'written short_erf');
			}
			if (parityIntermediates.longErf != null) {
				expectNumberArrayLength(writtenMrt.long_erf, expectedPointwiseLen, 'written long_erf');
			}
			if (parityIntermediates.shortDmrt != null) {
				expectNumberArrayLength(writtenMrt.short_dmrt, expectedPointwiseLen, 'written short_dmrt');
			}
			if (parityIntermediates.longDmrt != null) {
				expectNumberArrayLength(writtenMrt.long_dmrt, expectedPointwiseLen, 'written long_dmrt');
			}
		}
		const writtenUtci = JSON.parse(readFileSync(`${basePath}_webgpu_utci.json`, 'utf8')) as Record<string, unknown>;
		expect(writtenUtci.numPoints).toBe(parityResults.numPoints);
		expect(writtenUtci.numHours).toBe(parityResults.numHours);
		expect(Array.isArray(writtenUtci.utciByHour)).toBe(true);
		expect((writtenUtci.utciByHour as unknown[]).length).toBe(parityResults.numHours);
		if (parityPositions != null) {
			expectNumberArrayLength(writtenUtci.positions, parityResults.numPoints * 3, 'written positions');
		}
		console.log(`[collect] wrote utci in ${Date.now() - t0}ms`);
	});
});
