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
import { compareIntermediatesStats } from '../src/lib/parity/compareIntermediates';
import { compareUtciRange } from '../src/lib/parity/compareUtciRange';
import { buildParityReport } from '../src/lib/parity/buildParityReport';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';

function parseArgs(): { basePath: string; reportPath: string | null } {
	let basePath = process.env.PARITY_BASE_PATH ?? DEFAULT_BASE_PATH;
	let reportPath: string | null = null;
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i++) {
		if (argv[i] === '--base-path' && argv[i + 1]) {
			basePath = argv[i + 1];
			i++;
		} else if (argv[i] === '--report' && argv[i + 1]) {
			reportPath = resolve(process.cwd(), argv[i + 1]);
			i++;
		}
	}
	return { basePath: resolve(REPO_ROOT, basePath), reportPath };
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

async function main(): Promise<void> {
	const { basePath, reportPath } = parseArgs();
	console.log('Compare parity, base path:', basePath);

	if (reportPath) {
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
		console.log('Report written to:', reportPath);
		if (report.summary.failCount > 0)
			console.log('Inspect .detail.worstIndices and .detail.diffStats in the report to dig into locations.');
		process.exit(report.summary.failCount > 0 ? 1 : 0);
		return;
	}

	const webgpu = await loadWebgpuCollectedFromFs(basePath);
	let failCount = 0;

	// Solar
	if (webgpu.solar) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'solar');
			const result = compareIntermediatesStats({
				ref: ref.solarExposure,
				webgpu: webgpu.solar.solarExposure,
				toleranceMean: 0.02,
				toleranceMax: 0.05,
			});
			const pass = result.pass;
			if (!pass) failCount++;
			console.log(`solar: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`);
		} catch (e) {
			failCount++;
			console.log('solar: FAIL (no ref or error)', e instanceof Error ? e.message : String(e));
		}
	} else {
		console.log('solar: skipped (no webgpu file)');
	}

	// Sky
	if (webgpu.sky) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'sky');
			const result = compareIntermediatesStats({
				ref: ref.skyExposure,
				webgpu: webgpu.sky.skyExposure,
				toleranceMean: 0.02,
				toleranceMax: 0.05,
			});
			const pass = result.pass;
			if (!pass) failCount++;
			console.log(`sky: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`);
		} catch (e) {
			failCount++;
			console.log('sky: FAIL (no ref or error)', e instanceof Error ? e.message : String(e));
		}
	} else {
		console.log('sky: skipped (no webgpu file)');
	}

	// MRT
	if (webgpu.mrt) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'mrt');
			const result = compareIntermediatesStats({
				ref: ref.mrt,
				webgpu: webgpu.mrt.mrt,
				toleranceMean: 1,
				toleranceMax: 2,
			});
			const pass = result.pass;
			if (!pass) failCount++;
			console.log(`mrt: ${pass ? 'PASS' : 'FAIL'} (meanDiff=${result.meanDiff.toFixed(4)}, maxDiff=${result.maxDiff.toFixed(4)})`);
		} catch (e) {
			failCount++;
			console.log('mrt: FAIL (no ref or error)', e instanceof Error ? e.message : String(e));
		}
	} else {
		console.log('mrt: skipped (no webgpu file)');
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
		console.log(`utci: ${pass ? 'PASS' : 'FAIL'} (minDiff=${result.minDiff?.toFixed(3)}, maxDiff=${result.maxDiff?.toFixed(3)}, meanDiff=${result.meanDiff?.toFixed(3)})`);
	} else if (webgpu.utci) {
		console.log('utci: skipped (no ref metadata utci_range)');
	} else {
		console.log('utci: skipped (no webgpu file)');
	}

	process.exit(failCount > 0 ? 1 : 0);
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
