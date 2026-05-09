import { test, expect, type Page } from '@playwright/test';
import { resolve, join } from 'node:path';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const manifestPath = resolve(REPO_ROOT, 'data/analyses/manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

const ON_DEMAND_TIME_INDEX = 12;
const REPRESENTATIVE_FULL_YEAR_HOURS = 12 * 24;
const GRASSHOPPER_ONE_HOUR_BASELINE_S = 15 * 60;
const PARITY_WAIT_MS = Number(process.env.BATCH_PARITY_WAIT_MS ?? '15000');
const FULL_YEAR_WAIT_MS = Number(process.env.BATCH_FULL_YEAR_WAIT_MS ?? '20000');
const ON_DEMAND_WAIT_MS = Number(process.env.BATCH_ON_DEMAND_WAIT_MS ?? '20000');
const PER_ANALYSIS_TIMEOUT_MS = PARITY_WAIT_MS + FULL_YEAR_WAIT_MS + ON_DEMAND_WAIT_MS + 15_000;

const batchAnalysisFilter = process.env.BATCH_ANALYSIS_FILTER?.trim().toLowerCase();
const batchAnalysisLimit = Number(process.env.BATCH_ANALYSIS_LIMIT ?? '0');
const batchAnalysisOffset = Number(process.env.BATCH_ANALYSIS_OFFSET ?? '0');

const analysesToRun = manifest.analyses
	.filter((analysis: { id: string }) =>
		!batchAnalysisFilter || analysis.id.toLowerCase().includes(batchAnalysisFilter)
	)
	.slice(
		Math.max(0, Number.isInteger(batchAnalysisOffset) ? batchAnalysisOffset : 0),
		batchAnalysisLimit > 0
			? Math.max(0, Number.isInteger(batchAnalysisOffset) ? batchAnalysisOffset : 0) +
					batchAnalysisLimit
			: undefined
	);

async function waitForCollectionSuccess(page: Page, label: string, timeoutMs: number) {
	await page.waitForFunction(
		() => {
			const w = window as unknown as {
				__parityCollectionError__?: string;
				__parityIntermediatesError__?: string;
				__parityCollectionStatus__?: { state: 'running' | 'success' | 'error' | 'timeout' };
			};
			if (typeof w.__parityCollectionError__ === 'string') return true;
			if (typeof w.__parityIntermediatesError__ === 'string') return true;
			return (
				w.__parityCollectionStatus__?.state === 'success' ||
				w.__parityCollectionStatus__?.state === 'error' ||
				w.__parityCollectionStatus__?.state === 'timeout'
			);
		},
		{ timeout: timeoutMs, polling: 1000 }
	);

	const snapshot = await page.evaluate(() => {
		const w = window as unknown as {
			__parityCollectionStatus__?: unknown;
			__parityCollectionError__?: string;
			__parityIntermediatesError__?: string;
			__parityCollectionLog__?: unknown;
		};
		return {
			status: w.__parityCollectionStatus__ ?? null,
			collectionError: w.__parityCollectionError__ ?? null,
			intermediatesError: w.__parityIntermediatesError__ ?? null,
			log: w.__parityCollectionLog__ ?? null
		};
	});

	if (
		snapshot.collectionError ||
		snapshot.intermediatesError ||
		(snapshot.status as { state?: string } | null)?.state !== 'success'
	) {
		throw new Error(
			`${label} collection did not complete successfully.\n` +
				`status=${JSON.stringify(snapshot.status)}\n` +
				`collectionError=${JSON.stringify(snapshot.collectionError)}\n` +
				`intermediatesError=${JSON.stringify(snapshot.intermediatesError)}\n` +
				`log=${JSON.stringify(snapshot.log)}`
		);
	}

	return snapshot;
}

	test(`Collect parity and timing for all analyses`, async ({ page }) => {
		page.on('console', (msg) => console.log(`  [browser] ${msg.text()}`));
		console.log(
			`[batch] Selected ${analysesToRun.length} analyses` +
				(batchAnalysisFilter ? ` (filter="${batchAnalysisFilter}")` : '') +
				(batchAnalysisOffset > 0 ? ` (offset=${batchAnalysisOffset})` : '') +
				(batchAnalysisLimit > 0 ? ` (limit=${batchAnalysisLimit})` : '') +
				` (timeouts: parity=${PARITY_WAIT_MS}ms, fullYear=${FULL_YEAR_WAIT_MS}ms, onDemand=${ON_DEMAND_WAIT_MS}ms)`
		);
		test.setTimeout(PER_ANALYSIS_TIMEOUT_MS * Math.max(1, analysesToRun.length) + 60_000);
		
		const resultsDir = resolve(REPO_ROOT, 'data/batch-parity-results');
		if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });

		for (const analysis of analysesToRun) {
			const analysisSlug = analysis.id;
			const basePath = resolve(REPO_ROOT, 'data/analyses', analysisSlug);
			const timingFile = join(resultsDir, `${analysisSlug.replace(/[/\\]/g, '_')}_timing.json`);
			
			console.log(`[batch] Starting ${analysisSlug}`);
			try {
				// --- Phase 1: 1-Month Mode (Parity & Timing) ---
				const url1m = `/debug-webgpu-utci?parity=1&analysis=${encodeURIComponent(analysisSlug)}`;
				console.log(`  [1m] Navigating...`);
				await page.goto(url1m);

				const readiness1m = await waitForCollectionSuccess(
					page,
					`[1m] ${analysisSlug}`,
					PARITY_WAIT_MS
				);
				const status1m = readiness1m.status as {
					startedAt: number;
					updatedAt: number;
				};
				const log1m = readiness1m.log;
				const tCompute1m = (status1m.updatedAt - status1m.startedAt) / 1000;
				
				console.log(`  [1m] Compute done: ${tCompute1m.toFixed(2)}s`);

				// Only collect heavy parity data if PARITY_COLLECT is set
				const shouldCollectParity = process.env.PARITY_COLLECT === '1' || process.env.PARITY_COLLECT === 'true';
				let collectTime1m = 0;

				if (shouldCollectParity) {
					const tStart = Date.now();
					console.log(`  [1m] Collecting heavy results...`);
					
					const utciJson = await page.evaluate(() => {
						const res = (window as any).__parityResults__;
						if (!res) return null;
						return JSON.stringify({
							numPoints: res.numPoints,
							numHours: res.numHours,
							positions: res.positions,
							utciByHour: res.utciByHour
						});
					});
					if (utciJson) writeFileSync(`${basePath}_webgpu_utci.json`, utciJson);

					// Intermediates
					const interMeta = await page.evaluate(() => {
						const inter = (window as any).__parityIntermediates__;
						if (!inter) return null;
						return {
							keys: Object.keys(inter).filter(k => (window as any).__parityIntermediates__[k] !== null && typeof (window as any).__parityIntermediates__[k] === 'object'),
							numPoints: inter.numPoints,
							numHours: inter.numHours
						};
					});

					if (interMeta) {
						for (const key of interMeta.keys) {
							const json = await page.evaluate((k) => {
								const val = (window as any).__parityIntermediates__[k];
								const data = (val instanceof Float32Array || val instanceof Uint32Array) ? Array.from(val) : val;
								return JSON.stringify(data);
							}, key);
							if (key === 'solarExposure') {
								writeFileSync(`${basePath}_webgpu_solar.json`, `{"numPositions":${interMeta.numPoints},"numHours":${interMeta.numHours},"solarExposure":${json}}`);
							} else if (key === 'skyExposure') {
								writeFileSync(`${basePath}_webgpu_sky.json`, `{"numPositions":${interMeta.numPoints},"skyExposure":${json}}`);
							}
						}
						const mrtJson = await page.evaluate(() => {
							const inter = (window as any).__parityIntermediates__;
							if (!inter.mrt) return null;
							return JSON.stringify({
								numPositions: inter.numPoints,
								numHours: inter.numHours,
								mrt: Array.from(inter.mrt),
								short_erf: inter.shortErf ? Array.from(inter.shortErf) : null,
								long_erf: inter.longErf ? Array.from(inter.longErf) : null,
								short_dmrt: inter.shortDmrt ? Array.from(inter.shortDmrt) : null,
								long_dmrt: inter.longDmrt ? Array.from(inter.longDmrt) : null
							});
						});
						if (mrtJson) writeFileSync(`${basePath}_webgpu_mrt.json`, mrtJson);
					}
					collectTime1m = (Date.now() - tStart) / 1000;
				}

				// --- Phase 2: 12-Month Mode (Timing Only) ---
				const url12m = `/debug-webgpu-utci?parity=0&analysis=${encodeURIComponent(analysisSlug)}`;
				console.log(`  [12m] Navigating...`);
				await page.goto(url12m);

				const readiness12m = await waitForCollectionSuccess(
					page,
					`[12m] ${analysisSlug}`,
					FULL_YEAR_WAIT_MS
				);
				const status12m = readiness12m.status as {
					startedAt: number;
					updatedAt: number;
				};
				const log12m = readiness12m.log;
				const tCompute12m = (status12m.updatedAt - status12m.startedAt) / 1000;
				console.log(`  [12m] Compute done: ${tCompute12m.toFixed(2)}s`);

				const preflight = await page.evaluate(() => (window as any).__computePreflight__);

				// --- Phase 3: Strict On-Demand One-Hour Mode (timing + allocation diagnostics) ---
				const urlOnDemand = `/debug-webgpu-utci?analysis=${encodeURIComponent(
					analysisSlug
				)}&onDemandPrototype=1&strictExposureOnly=1&timeIndex=${ON_DEMAND_TIME_INDEX}`;
				console.log(`  [on-demand] Navigating...`);
				let onDemandReport: Record<string, unknown> = {
					captureMode: 'strictExposureOnly',
					timeIndex: ON_DEMAND_TIME_INDEX,
					representativeFullYearHours: REPRESENTATIVE_FULL_YEAR_HOURS
				};

				try {
					const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
					if (!hasWebGpu) {
						onDemandReport = {
							...onDemandReport,
							state: 'unsupported',
							error: 'navigator.gpu is unavailable in this runtime.'
						};
					} else {
						const onDemandStart = Date.now();
						await page.goto(urlOnDemand);

						await page.waitForFunction(
							(expectedTimeIndex) => {
								const diagnostics = (window as any).__onDemandPrototypeDiagnostics__;
								return (
									diagnostics?.path === 'exposure-only-f32' &&
									diagnostics?.usedExposureOnlyPrecompute === true &&
									diagnostics?.usedRunAllForSelectedHour === false &&
									diagnostics?.liveAnalysisConstructedForSelectedHour === false &&
									Array.isArray(diagnostics?.timeIndices) &&
									diagnostics.timeIndices.includes(expectedTimeIndex) &&
									(diagnostics?.allHoursUtciBytesAllocated ?? -1) === 0 &&
									(diagnostics?.allHoursMrtBytesAllocated ?? -1) === 0 &&
									(diagnostics?.oneHourOutputBytes ?? 0) > 0 &&
									(diagnostics?.timings?.exposurePrecomputeMs ?? 0) > 0 &&
									(diagnostics?.timings?.oneHourDispatchMs ?? -1) >= 0 &&
									(diagnostics?.inFlightCount ?? 0) === 0
								);
							},
							ON_DEMAND_TIME_INDEX,
							{ timeout: ON_DEMAND_WAIT_MS }
						);

						const onDemandReadyS = (Date.now() - onDemandStart) / 1000;
						const onDemandDiagnostics = await page.evaluate(
							() => (window as any).__onDemandPrototypeDiagnostics__
						);
						const onDemandStatusText = await page
							.getByTestId('on-demand-prototype-status')
							.textContent()
							.catch(() => null);

						console.log(`  [on-demand] Ready: ${onDemandReadyS.toFixed(2)}s`);
						onDemandReport = {
							...onDemandReport,
							state: 'success',
							ready_s: onDemandReadyS,
							statusText: onDemandStatusText,
							diagnostics: onDemandDiagnostics
						};
					}
				} catch (error: any) {
					const diagnostics = await page
						.evaluate(() => (window as any).__onDemandPrototypeDiagnostics__)
						.catch(() => null);
					onDemandReport = {
						...onDemandReport,
						state: 'error',
						error: error?.message || String(error),
						diagnostics
					};
					console.warn(`  [on-demand] Capture failed for ${analysisSlug}: ${onDemandReport.error}`);
				}

				// Save combined timing report
				const timingReport = {
					analysisId: analysis.id,
					pythonRuntime: analysis.runtime_seconds,
					grasshopperOneHourBaseline_s: GRASSHOPPER_ONE_HOUR_BASELINE_S,
					representativeFullYearHours: REPRESENTATIVE_FULL_YEAR_HOURS,
					preflight,
					webgpu_1m: {
						compute_s: tCompute1m,
						collect_s: collectTime1m,
						status: status1m,
						log: log1m
					},
					webgpu_12m: {
						compute_s: tCompute12m,
						status: status12m,
						log: log12m
					},
					webgpu_on_demand: onDemandReport,
					timestamp: new Date().toISOString()
				};

				writeFileSync(timingFile, JSON.stringify(timingReport, null, 2));

				console.log(`[batch] Finished ${analysisSlug}. 1m: ${tCompute1m.toFixed(2)}s, 12m: ${tCompute12m.toFixed(2)}s.`);
			} finally {
				await page.goto('about:blank', { timeout: 5_000 }).catch(() => undefined);
			}
		}
	});
