import { test, expect } from '@playwright/test';
import { resolve, join } from 'node:path';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const manifestPath = resolve(REPO_ROOT, 'data/analyses/manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

const COLLECT_WAIT_MS = 300_000;

const analysesToRun = manifest.analyses;

	test(`Collect parity and timing for all analyses`, async ({ page }) => {
		// Massive timeout for the entire batch
		test.setTimeout(COLLECT_WAIT_MS * (analysesToRun.length + 1));
		
		const resultsDir = resolve(REPO_ROOT, 'data/batch-parity-results');
		if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });

		for (const analysis of analysesToRun) {
			const analysisSlug = analysis.id;
			const basePath = resolve(REPO_ROOT, 'data/analyses', analysisSlug);
			const timingFile = join(resultsDir, `${analysisSlug.replace(/[/\\]/g, '_')}_timing.json`);
			
			console.log(`[batch] Starting ${analysisSlug}`);

			// --- Phase 1: 1-Month Mode (Parity & Timing) ---
			const url1m = `/debug-webgpu-utci?parity=1&analysis=${encodeURIComponent(analysisSlug)}`;
			console.log(`  [1m] Navigating...`);
			await page.goto(url1m);

			await page.waitForFunction(
				() => (window as any).__parityCollectionStatus__?.state === 'success',
				{ timeout: COLLECT_WAIT_MS }
			);

			const status1m = await page.evaluate(() => (window as any).__parityCollectionStatus__);
			const log1m = await page.evaluate(() => (window as any).__parityCollectionLog__);
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

			await page.waitForFunction(
				() => (window as any).__parityCollectionStatus__?.state === 'success',
				{ timeout: COLLECT_WAIT_MS }
			);

			const status12m = await page.evaluate(() => (window as any).__parityCollectionStatus__);
			const tCompute12m = (status12m.updatedAt - status12m.startedAt) / 1000;
			console.log(`  [12m] Compute done: ${tCompute12m.toFixed(2)}s`);

			// Save combined timing report
			const timingReport = {
				analysisId: analysis.id,
				pythonRuntime: analysis.runtime_seconds,
				webgpu_1m: {
					compute_s: tCompute1m,
					collect_s: collectTime1m,
					status: status1m,
					log: log1m
				},
				webgpu_12m: {
					compute_s: tCompute12m,
					status: status12m
				},
				timestamp: new Date().toISOString()
			};
			writeFileSync(timingFile, JSON.stringify(timingReport, null, 2));

			console.log(`[batch] Finished ${analysisSlug}. 1m: ${tCompute1m.toFixed(2)}s, 12m: ${tCompute12m.toFixed(2)}s.`);
		}
	});
