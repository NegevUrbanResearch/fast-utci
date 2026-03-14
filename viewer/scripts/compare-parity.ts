/**
 * Offline parity compare: load Python ref + WebGPU collected files, assert per stage, exit 0/1.
 * Usage: npx tsx scripts/compare-parity.ts [--base-path <path>] [--report <file>]
 * Default base-path: data/analyses/Ben-Gurion/20250815_grid_2m_fullday (relative to repo root).
 * With --report <file>: write a detailed JSON report (diffs, percentiles, worst indices) for debugging.
 */
import { resolve } from 'node:path';
import { readFileSync, writeFileSync } from 'node:fs';
import { loadReferenceIntermediatesFromFs } from '../src/lib/parity/loadReferenceIntermediatesFromFs';
import { loadWebgpuCollectedFromFs } from '../src/lib/parity/loadWebgpuCollectedFromFs';
import { loadReferenceFromFs } from '../src/lib/parity/loadReferenceFromFs';
import { compareIntermediates, compareIntermediatesStats } from '../src/lib/parity/compareIntermediates';
import { compareUtciRange } from '../src/lib/parity/compareUtciRange';
import { buildParityReport } from '../src/lib/parity/buildParityReport';
import {
	computeSpatialComplexity,
	inferRectGridShapeFromPositions,
	type SpatialComplexityMetrics
} from '../src/lib/parity/spatialComplexity';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
type CompareMode = 'strict' | 'stats';

function parseArgs(): { basePath: string; reportPath: string | null; mode: CompareMode } {
	let basePath = process.env.PARITY_BASE_PATH ?? DEFAULT_BASE_PATH;
	let reportPath: string | null = null;
	let mode: CompareMode = 'strict';
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i++) {
		if (argv[i] === '--base-path' && argv[i + 1]) {
			basePath = argv[i + 1];
			i++;
		} else if (argv[i] === '--report' && argv[i + 1]) {
			reportPath = resolve(process.cwd(), argv[i + 1]);
			i++;
		} else if (argv[i] === '--mode' && argv[i + 1]) {
			const rawMode = argv[i + 1];
			if (rawMode === 'strict' || rawMode === 'stats') {
				mode = rawMode;
			} else {
				throw new Error(`Invalid --mode "${rawMode}". Expected strict|stats.`);
			}
			i++;
		}
	}
	return { basePath: resolve(REPO_ROOT, basePath), reportPath, mode };
}

function loadRefMetadataUtciRange(basePath: string): { min: number; max: number; mean: number } | null {
	try {
		const raw = readFileSync(`${basePath}.json`, 'utf8');
		const meta = JSON.parse(raw) as { utci_range?: { min: number; max: number; mean: number } };
		if (meta.utci_range && typeof meta.utci_range.min === 'number' && typeof meta.utci_range.max === 'number' && typeof meta.utci_range.mean === 'number') {
			return meta.utci_range;
		}
	} catch {
		// ignore
	}
	return null;
}

function meanMetrics(items: readonly SpatialComplexityMetrics[]): SpatialComplexityMetrics {
	if (items.length === 0) {
		return { gradientEnergy: 0, variance: 0, entropy: 0 };
	}
	const sum = items.reduce(
		(acc, curr) => {
			acc.gradientEnergy += curr.gradientEnergy;
			acc.variance += curr.variance;
			acc.entropy += curr.entropy;
			return acc;
		},
		{ gradientEnergy: 0, variance: 0, entropy: 0 }
	);
	return {
		gradientEnergy: sum.gradientEnergy / items.length,
		variance: sum.variance / items.length,
		entropy: sum.entropy / items.length
	};
}

function computeHourlyComplexity(
	utciByHour: readonly (readonly number[])[],
	width: number,
	height: number
): SpatialComplexityMetrics {
	const expectedLength = width * height;
	const malformed = utciByHour
		.map((hour, index) => ({ index, length: hour.length }))
		.filter((h) => h.length !== expectedLength);
	if (malformed.length > 0) {
		throw new Error(
			`UTCI hourly field shape mismatch: expected ${expectedLength}, got malformed hours: ${malformed
				.slice(0, 5)
				.map((h) => `${h.index}:${h.length}`)
				.join(', ')}${malformed.length > 5 ? ' ...' : ''}`
		);
	}
	if (utciByHour.length === 0) return { gradientEnergy: 0, variance: 0, entropy: 0 };
	return meanMetrics(utciByHour.map((hour) => computeSpatialComplexity(hour, width, height)));
}

async function main(): Promise<void> {
	const { basePath, reportPath, mode } = parseArgs();
	console.log('Compare parity, base path:', basePath);
	console.log('Compare mode:', mode);

	if (reportPath && mode === 'stats') {
		const report = await buildParityReport(basePath);
		writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
		// Print same-style stage lines as non-report run
		if (report.solar)
			console.log(
				`solar: ${report.solar.pass ? 'PASS' : 'FAIL'} (meanDiff=${report.solar.meanDiff?.toFixed(4) ?? '—'}, maxDiff=${report.solar.maxDiff?.toFixed(4) ?? '—'})${report.solar.detail?.diffStats ? ` [p99=${report.solar.detail.diffStats.p99.toFixed(4)}]` : ''}`
			);
		if (report.sky)
			console.log(
				`sky: ${report.sky.pass ? 'PASS' : 'FAIL'} (meanDiff=${report.sky.meanDiff?.toFixed(4) ?? '—'}, maxDiff=${report.sky.maxDiff?.toFixed(4) ?? '—'})${report.sky.detail?.diffStats ? ` [p99=${report.sky.detail.diffStats.p99.toFixed(4)}]` : ''}`
			);
		if (report.mrt)
			console.log(
				`mrt: ${report.mrt.pass ? 'PASS' : 'FAIL'} (meanDiff=${report.mrt.meanDiff?.toFixed(4) ?? '—'}, maxDiff=${report.mrt.maxDiff?.toFixed(4) ?? '—'})${report.mrt.detail?.diffStats ? ` [p99=${report.mrt.detail.diffStats.p99.toFixed(4)}]` : ''}`
			);
		if (report.utci)
			console.log(
				`utci: ${report.utci.pass ? 'PASS' : 'FAIL'} (minDiff=${report.utci.minDiff?.toFixed(3) ?? '—'}, maxDiff=${report.utci.maxDiff?.toFixed(3) ?? '—'}, meanDiff=${report.utci.meanDiff?.toFixed(3) ?? '—'})`
			);
		if (report.utciComplexity?.error) {
			console.log(`utci_complexity: skipped (${report.utciComplexity.error})`);
		} else if (report.utciComplexity) {
			console.log(
				`utci_complexity: gradientEnergy delta=${report.utciComplexity.delta.gradientEnergy.toFixed(4)}, variance delta=${report.utciComplexity.delta.variance.toFixed(4)}, entropy delta=${report.utciComplexity.delta.entropy.toFixed(4)}`
			);
		}
		console.log('Report written to:', reportPath);
		if (report.summary.failCount > 0)
			console.log('Inspect .detail.worstIndices and .detail.diffStats in the report to dig into locations.');
		process.exit(report.summary.failCount > 0 ? 1 : 0);
		return;
	}

	const webgpu = await loadWebgpuCollectedFromFs(basePath);
	let failCount = 0;
	const isStrictLengthMismatch = (message: string): boolean => /Length mismatch/.test(message);
	const strictReport: Record<string, unknown> = {
		basePath,
		mode,
		summary: { pass: true, failCount: 0 }
	};

	// Solar
	if (webgpu.solar) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'solar');
			if (mode === 'strict') {
				const result = compareIntermediates({
					ref: ref.solarExposure,
					webgpu: new Float32Array(webgpu.solar.solarExposure),
					tolerance: 0.05
				});
				const pass = result.pass;
				if (!pass) failCount++;
				strictReport.solar = { pass, rmse: result.rmse, maxError: result.maxError };
				console.log(
					`solar: ${pass ? 'PASS' : 'FAIL'} (rmse=${result.rmse.toFixed(4)}, maxError=${result.maxError.toFixed(4)})`
				);
			} else {
				const result = compareIntermediatesStats({
					ref: ref.solarExposure,
					webgpu: webgpu.solar.solarExposure,
					toleranceMean: 0.02,
					toleranceMax: 0.05
				});
				const pass = result.pass;
				if (!pass) failCount++;
				console.log(
					`solar: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`
				);
			}
		} catch (e) {
			failCount++;
			const msg = e instanceof Error ? e.message : String(e);
			if (mode === 'strict') strictReport.solar = { pass: false, error: msg };
			if (mode === 'strict' && isStrictLengthMismatch(msg)) {
				console.log('solar: FAIL (strict mode requires same length/order artifacts)');
			} else {
				console.log('solar: FAIL (no ref or error)', msg);
			}
		}
	} else {
		console.log('solar: skipped (no webgpu file)');
		if (mode === 'strict') strictReport.solar = { pass: true, skipped: true, reason: 'no webgpu file' };
	}

	// Sky
	if (webgpu.sky) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'sky');
			if (mode === 'strict') {
				const result = compareIntermediates({
					ref: ref.skyExposure,
					webgpu: new Float32Array(webgpu.sky.skyExposure),
					tolerance: 0.05
				});
				const pass = result.pass;
				if (!pass) failCount++;
				strictReport.sky = { pass, rmse: result.rmse, maxError: result.maxError };
				console.log(
					`sky: ${pass ? 'PASS' : 'FAIL'} (rmse=${result.rmse.toFixed(4)}, maxError=${result.maxError.toFixed(4)})`
				);
			} else {
				const result = compareIntermediatesStats({
					ref: ref.skyExposure,
					webgpu: webgpu.sky.skyExposure,
					toleranceMean: 0.02,
					toleranceMax: 0.05
				});
				const pass = result.pass;
				if (!pass) failCount++;
				console.log(
					`sky: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`
				);
			}
		} catch (e) {
			failCount++;
			const msg = e instanceof Error ? e.message : String(e);
			if (mode === 'strict') strictReport.sky = { pass: false, error: msg };
			if (mode === 'strict' && isStrictLengthMismatch(msg)) {
				console.log('sky: FAIL (strict mode requires same length/order artifacts)');
			} else {
				console.log('sky: FAIL (no ref or error)', msg);
			}
		}
	} else {
		console.log('sky: skipped (no webgpu file)');
		if (mode === 'strict') strictReport.sky = { pass: true, skipped: true, reason: 'no webgpu file' };
	}

	// MRT
	if (webgpu.mrt) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'mrt');
			const componentOut: Record<string, unknown> = {};
			const compareMrtComponent = (
				label: 'short_erf' | 'long_erf' | 'short_dmrt' | 'long_dmrt',
				toleranceStrict: number,
				toleranceMean: number,
				toleranceMax: number
			): boolean => {
				const refArr = (ref as Record<string, unknown>)[label] as Float32Array | undefined;
				const wgArrRaw = (webgpu.mrt as Record<string, unknown>)[label] as number[] | undefined;
				if (!refArr || !wgArrRaw) {
					console.log(`${label}: skipped (not available in both ref and webgpu files)`);
					if (mode === 'strict') {
						componentOut[label] = {
							pass: true,
							skipped: true,
							reason: 'not available in both ref and webgpu files'
						};
					}
					return true;
				}
				if (mode === 'strict') {
					const result = compareIntermediates({
						ref: refArr,
						webgpu: new Float32Array(wgArrRaw),
						tolerance: toleranceStrict
					});
					const pass = result.pass;
					componentOut[label] = { pass, rmse: result.rmse, maxError: result.maxError };
					console.log(
						`${label}: ${pass ? 'PASS' : 'FAIL'} (rmse=${result.rmse.toFixed(4)}, maxError=${result.maxError.toFixed(4)})`
					);
					return pass;
				}
				const result = compareIntermediatesStats({
					ref: refArr,
					webgpu: wgArrRaw,
					toleranceMean,
					toleranceMax
				});
				const pass = result.pass;
				componentOut[label] = { pass, meanDiff: result.meanDiff, maxDiff: result.maxDiff };
				console.log(
					`${label}: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`
				);
				return pass;
			};
			if (mode === 'strict') {
				const result = compareIntermediates({
					ref: ref.mrt,
					webgpu: new Float32Array(webgpu.mrt.mrt),
					tolerance: 2
				});
				const pass = result.pass;
				if (!pass) failCount++;
				strictReport.mrt = { pass, rmse: result.rmse, maxError: result.maxError };
				console.log(
					`mrt: ${pass ? 'PASS' : 'FAIL'} (rmse=${result.rmse.toFixed(4)}, maxError=${result.maxError.toFixed(4)})`
				);
			} else {
				const result = compareIntermediatesStats({
					ref: ref.mrt,
					webgpu: webgpu.mrt.mrt,
					toleranceMean: 1,
					toleranceMax: 2
				});
				const pass = result.pass;
				if (!pass) failCount++;
				console.log(
					`mrt: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`
				);
			}
			if (!compareMrtComponent('short_erf', 1.5, 0.5, 1.5)) failCount++;
			if (!compareMrtComponent('long_erf', 1.5, 0.5, 1.5)) failCount++;
			if (!compareMrtComponent('short_dmrt', 0.25, 0.1, 0.25)) failCount++;
			if (!compareMrtComponent('long_dmrt', 0.25, 0.1, 0.25)) failCount++;
			if (mode === 'strict') Object.assign(strictReport, componentOut);
		} catch (e) {
			failCount++;
			const msg = e instanceof Error ? e.message : String(e);
			if (mode === 'strict') strictReport.mrt = { pass: false, error: msg };
			if (mode === 'strict' && isStrictLengthMismatch(msg)) {
				console.log('mrt: FAIL (strict mode requires same length/order artifacts)');
			} else {
				console.log('mrt: FAIL (no ref or error)', msg);
			}
		}
	} else {
		console.log('mrt: skipped (no webgpu file)');
		if (mode === 'strict') strictReport.mrt = { pass: true, skipped: true, reason: 'no webgpu file' };
	}

	// UTCI range
	const refUtciRange = loadRefMetadataUtciRange(basePath);
	if (webgpu.utci && refUtciRange) {
		const result = compareUtciRange({
			ref: refUtciRange,
			webgpu: webgpu.utci.utci_range,
			toleranceMin: 2,
			toleranceMax: 2,
			toleranceMean: 1,
		});
		const pass = result.pass;
		if (!pass) failCount++;
		if (mode === 'strict') {
			strictReport.utci = {
				pass,
				minDiff: result.minDiff,
				maxDiff: result.maxDiff,
				meanDiff: result.meanDiff
			};
		}
		console.log(`utci: ${pass ? 'PASS' : 'FAIL'} (minDiff=${result.minDiff?.toFixed(3)}, maxDiff=${result.maxDiff?.toFixed(3)}, meanDiff=${result.meanDiff?.toFixed(3)})`);
	} else if (webgpu.utci) {
		console.log('utci: skipped (no ref metadata utci_range)');
		if (mode === 'strict') strictReport.utci = { pass: true, skipped: true, reason: 'no ref metadata utci_range' };
	} else {
		console.log('utci: skipped (no webgpu file)');
		if (mode === 'strict') strictReport.utci = { pass: true, skipped: true, reason: 'no webgpu file' };
	}

	// UTCI spatial complexity diagnostics
	if (webgpu.utci) {
		try {
			const ref = await loadReferenceFromFs(basePath);
			const refPositions = Array.from(ref.data.positions);
			const webgpuPositions =
				webgpu.utci.positions && webgpu.utci.positions.length === webgpu.utci.numPoints * 3
					? webgpu.utci.positions
					: null;
			const refShape = inferRectGridShapeFromPositions(refPositions);
			const webgpuShape = webgpuPositions ? inferRectGridShapeFromPositions(webgpuPositions) : null;
			if (mode === 'strict' && !webgpuPositions) {
				console.log(
					'utci_complexity: skipped (strict mode requires webgpu_utci.positions for topology-aware diagnostics)'
				);
				if (mode === 'strict') {
					strictReport.utci_complexity = {
						skipped: true,
						reason: 'strict mode requires webgpu_utci.positions for topology-aware diagnostics'
					};
				}
			} else if (!refShape || !webgpuShape) {
				console.log('utci_complexity: skipped (unable to infer rectangular grid shape)');
				if (mode === 'strict') {
					strictReport.utci_complexity = {
						skipped: true,
						reason: 'unable to infer rectangular grid shape'
					};
				}
			} else {
				const refComplexity = computeHourlyComplexity(
					ref.data.utciByHour.map((arr) => Array.from(arr)),
					refShape.width,
					refShape.height
				);
				const wgComplexity = computeHourlyComplexity(
					webgpu.utci.utciByHour,
					webgpuShape.width,
					webgpuShape.height
				);
				console.log(
					`utci_complexity: gradientEnergy ref=${refComplexity.gradientEnergy.toFixed(4)} webgpu=${wgComplexity.gradientEnergy.toFixed(4)} delta=${(wgComplexity.gradientEnergy - refComplexity.gradientEnergy).toFixed(4)}`
				);
				console.log(
					`utci_complexity: variance ref=${refComplexity.variance.toFixed(4)} webgpu=${wgComplexity.variance.toFixed(4)} delta=${(wgComplexity.variance - refComplexity.variance).toFixed(4)}`
				);
				console.log(
					`utci_complexity: entropy ref=${refComplexity.entropy.toFixed(4)} webgpu=${wgComplexity.entropy.toFixed(4)} delta=${(wgComplexity.entropy - refComplexity.entropy).toFixed(4)}`
				);
				if (mode === 'strict') {
					strictReport.utci_complexity = {
						ref: refComplexity,
						webgpu: wgComplexity,
						delta: {
							gradientEnergy: wgComplexity.gradientEnergy - refComplexity.gradientEnergy,
							variance: wgComplexity.variance - refComplexity.variance,
							entropy: wgComplexity.entropy - refComplexity.entropy
						}
					};
				}
			}
		} catch (e) {
			console.log(
				'utci_complexity: skipped (error)',
				e instanceof Error ? e.message : String(e)
			);
			if (mode === 'strict') {
				strictReport.utci_complexity = {
					skipped: true,
					reason: e instanceof Error ? e.message : String(e)
				};
			}
		}
	}
	if (reportPath && mode === 'strict') {
		(strictReport.summary as { pass: boolean; failCount: number }).pass = failCount === 0;
		(strictReport.summary as { pass: boolean; failCount: number }).failCount = failCount;
		writeFileSync(reportPath, JSON.stringify(strictReport, null, 2), 'utf8');
		console.log('Report written to:', reportPath);
	}

	process.exit(failCount > 0 ? 1 : 0);
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
