/**
 * Build a detailed parity report (pass/fail, stats, per-element diff analysis) for debugging.
 * Used by compare-parity.ts when --report is set.
 */

import { loadReferenceIntermediatesFromFs } from './loadReferenceIntermediatesFromFs';
import { loadWebgpuCollectedFromFs } from './loadWebgpuCollectedFromFs';
import { compareIntermediatesStats, analyzeDiffs } from './compareIntermediates';
import { compareUtciRange } from './compareUtciRange';
import { readFileSync } from 'node:fs';

export interface StageReport {
	pass: boolean;
	meanDiff?: number;
	maxDiff?: number;
	minDiff?: number;
	refStats?: { mean: number; max: number; min: number; std: number; n: number };
	webgpuStats?: { mean: number; max: number; min: number; std: number; n: number };
	/** When ref and webgpu same length: diff distribution and worst indices. */
	detail?: {
		sameLength: boolean;
		n: number;
		diffStats?: {
			mean: number;
			std: number;
			min: number;
			max: number;
			rmse: number;
			p50: number;
			p95: number;
			p99: number;
		};
		worstIndices?: Array<{ index: number; ref: number; webgpu: number; diff: number }>;
	};
	error?: string;
}

export interface ParityReport {
	basePath: string;
	summary: { pass: boolean; failCount: number };
	solar?: StageReport;
	sky?: StageReport;
	mrt?: StageReport;
	utci?: StageReport;
}

function loadRefMetadataUtciRange(basePath: string): { min: number; max: number; mean: number } | null {
	try {
		const raw = readFileSync(`${basePath}.json`, 'utf8');
		const meta = JSON.parse(raw) as { utci_range?: { min: number; max: number; mean: number } };
		if (
			meta.utci_range &&
			typeof meta.utci_range.min === 'number' &&
			typeof meta.utci_range.max === 'number' &&
			typeof meta.utci_range.mean === 'number'
		) {
			return meta.utci_range;
		}
	} catch {
		// ignore
	}
	return null;
}

export async function buildParityReport(basePath: string): Promise<ParityReport> {
	const webgpu = await loadWebgpuCollectedFromFs(basePath);
	let failCount = 0;
	const report: ParityReport = {
		basePath,
		summary: { pass: true, failCount: 0 }
	};

	const refToArr = (r: Float32Array | number[]) => (Array.isArray(r) ? r : Array.from(r));

	// Solar
	if (webgpu.solar) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'solar');
			const refArr = refToArr(ref.solarExposure);
			const wgArr = refToArr(webgpu.solar.solarExposure);
			const result = compareIntermediatesStats({
				ref: refArr,
				webgpu: wgArr,
				toleranceMean: 0.02,
				toleranceMax: 0.05
			});
			if (!result.pass) failCount++;
			const detail = analyzeDiffs({ ref: refArr, webgpu: wgArr, maxWorst: 20 });
			report.solar = {
				pass: result.pass,
				meanDiff: result.meanDiff,
				maxDiff: result.maxDiff,
				refStats: result.refStats,
				webgpuStats: result.webgpuStats,
				detail: {
					sameLength: detail.sameLength,
					n: detail.n,
					diffStats: detail.diffStats,
					worstIndices: detail.worstIndices
				}
			};
		} catch (e) {
			failCount++;
			report.solar = {
				pass: false,
				error: e instanceof Error ? e.message : String(e)
			};
		}
	}

	// Sky
	if (webgpu.sky) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'sky');
			const refArr = refToArr(ref.skyExposure);
			const wgArr = refToArr(webgpu.sky.skyExposure);
			const result = compareIntermediatesStats({
				ref: refArr,
				webgpu: wgArr,
				toleranceMean: 0.02,
				toleranceMax: 0.05
			});
			if (!result.pass) failCount++;
			const detail = analyzeDiffs({ ref: refArr, webgpu: wgArr, maxWorst: 20 });
			report.sky = {
				pass: result.pass,
				meanDiff: result.meanDiff,
				maxDiff: result.maxDiff,
				refStats: result.refStats,
				webgpuStats: result.webgpuStats,
				detail: {
					sameLength: detail.sameLength,
					n: detail.n,
					diffStats: detail.diffStats,
					worstIndices: detail.worstIndices
				}
			};
		} catch (e) {
			failCount++;
			report.sky = {
				pass: false,
				error: e instanceof Error ? e.message : String(e)
			};
		}
	}

	// MRT
	if (webgpu.mrt) {
		try {
			const ref = await loadReferenceIntermediatesFromFs(basePath, 'mrt');
			const refArr = refToArr(ref.mrt);
			const wgArr = refToArr(webgpu.mrt.mrt);
			const result = compareIntermediatesStats({
				ref: refArr,
				webgpu: wgArr,
				toleranceMean: 1,
				toleranceMax: 2
			});
			if (!result.pass) failCount++;
			const detail = analyzeDiffs({ ref: refArr, webgpu: wgArr, maxWorst: 20 });
			report.mrt = {
				pass: result.pass,
				meanDiff: result.meanDiff,
				maxDiff: result.maxDiff,
				refStats: result.refStats,
				webgpuStats: result.webgpuStats,
				detail: {
					sameLength: detail.sameLength,
					n: detail.n,
					diffStats: detail.diffStats,
					worstIndices: detail.worstIndices
				}
			};
		} catch (e) {
			failCount++;
			report.mrt = {
				pass: false,
				error: e instanceof Error ? e.message : String(e)
			};
		}
	}

	// UTCI range
	const refUtciRange = loadRefMetadataUtciRange(basePath);
	if (webgpu.utci && refUtciRange) {
		const result = compareUtciRange({
			ref: refUtciRange,
			webgpu: webgpu.utci.utci_range,
			toleranceMin: 2,
			toleranceMax: 2,
			toleranceMean: 1
		});
		if (!result.pass) failCount++;
		report.utci = {
			pass: result.pass,
			minDiff: result.minDiff,
			maxDiff: result.maxDiff,
			meanDiff: result.meanDiff
		};
	} else if (webgpu.utci) {
		report.utci = { pass: true, error: 'no ref metadata utci_range' };
	}

	report.summary = { pass: failCount === 0, failCount };
	return report;
}
