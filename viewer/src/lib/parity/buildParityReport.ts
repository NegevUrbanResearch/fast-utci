/**
 * Build a detailed parity report (pass/fail, stats, per-element diff analysis) for debugging.
 * Used by compare-parity.ts when --report is set.
 */

import { loadReferenceIntermediatesFromFs } from './loadReferenceIntermediatesFromFs';
import { loadWebgpuCollectedFromFs } from './loadWebgpuCollectedFromFs';
import { loadReferenceFromFs } from './loadReferenceFromFs';
import { compareIntermediatesStats, analyzeDiffs } from './compareIntermediates';
import { compareUtciRange } from './compareUtciRange';
import { compareUtciPointwise } from './compareUtciPointwise';
import {
	computeSpatialComplexity,
	inferRectGridShapeFromPositions,
	type SpatialComplexityMetrics
} from './spatialComplexity';
import { pointwiseIndexFromFlat } from './pointwiseIndex';
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
	utciPointwise?: {
		pass: boolean;
		rmse: number;
		maxError: number;
		meanDiff: number;
		worst: { hour: number; pointIndex: number; ref: number; webgpu: number; diff: number } | null;
		error?: string;
	};
	utciComplexity?: {
		ref: SpatialComplexityMetrics;
		webgpu: SpatialComplexityMetrics;
		delta: SpatialComplexityMetrics;
		error?: string;
	};
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

function enrichWorstIndices(
	worstIndices: Array<{ index: number; ref: number; webgpu: number; diff: number }> | undefined,
	numHours: number | undefined
): Array<{ index: number; ref: number; webgpu: number; diff: number; hourIndex: number; pointIndex: number }> | undefined {
	if (!worstIndices || numHours == null) return worstIndices as undefined;
	return worstIndices.map((row) => ({
		...row,
		...pointwiseIndexFromFlat(row.index, numHours)
	}));
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
					worstIndices: enrichWorstIndices(detail.worstIndices, ref.numHours)
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
					worstIndices: enrichWorstIndices(detail.worstIndices, ref.numHours)
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
	if (webgpu.utci) {
		try {
			const ref = await loadReferenceFromFs(basePath);
			const pointwise = compareUtciPointwise({
				ref: ref.data.utciByHour.map((arr) => Array.from(arr)),
				webgpu: webgpu.utci.utciByHour,
				tolerance: 2
			});
			report.utciPointwise = {
				pass: pointwise.pass,
				rmse: pointwise.rmse,
				maxError: pointwise.maxError,
				meanDiff: pointwise.meanDiff,
				worst: pointwise.worst
			};
		} catch (e) {
			failCount++;
			report.utciPointwise = {
				pass: false,
				rmse: 0,
				maxError: 0,
				meanDiff: 0,
				worst: null,
				error: e instanceof Error ? e.message : String(e)
			};
		}
	}

	// UTCI spatial complexity
	if (webgpu.utci) {
		try {
			const ref = await loadReferenceFromFs(basePath);
			const refPositions = Array.from(ref.data.positions);
			const webgpuPositions =
				webgpu.utci.positions && webgpu.utci.positions.length === webgpu.utci.numPoints * 3
					? webgpu.utci.positions
					: null;
			const refShape = inferRectGridShapeFromPositions(refPositions);
			const wgShape = webgpuPositions ? inferRectGridShapeFromPositions(webgpuPositions) : null;
			if (refShape && wgShape) {
				const meanMetrics = (items: readonly SpatialComplexityMetrics[]): SpatialComplexityMetrics => {
					if (items.length === 0) return { gradientEnergy: 0, variance: 0, entropy: 0 };
					const totals = items.reduce(
						(acc, curr) => {
							acc.gradientEnergy += curr.gradientEnergy;
							acc.variance += curr.variance;
							acc.entropy += curr.entropy;
							return acc;
						},
						{ gradientEnergy: 0, variance: 0, entropy: 0 }
					);
					return {
						gradientEnergy: totals.gradientEnergy / items.length,
						variance: totals.variance / items.length,
						entropy: totals.entropy / items.length
					};
				};
				const complexityByHour = (
					values: readonly (readonly number[])[],
					width: number,
					height: number
				): SpatialComplexityMetrics => {
					const expectedLength = width * height;
					const malformed = values
						.map((arr, index) => ({ index, length: arr.length }))
						.filter((h) => h.length !== expectedLength);
					if (malformed.length > 0) {
						throw new Error(
							`UTCI hourly field shape mismatch: expected ${expectedLength}, malformed hours: ${malformed
								.slice(0, 5)
								.map((h) => `${h.index}:${h.length}`)
								.join(', ')}${malformed.length > 5 ? ' ...' : ''}`
						);
					}
					return meanMetrics(values.map((arr) => computeSpatialComplexity(arr, width, height)));
				};
				const refMetrics = complexityByHour(
					ref.data.utciByHour.map((arr) => Array.from(arr)),
					refShape.width,
					refShape.height
				);
				const wgMetrics = complexityByHour(webgpu.utci.utciByHour, wgShape.width, wgShape.height);
				report.utciComplexity = {
					ref: refMetrics,
					webgpu: wgMetrics,
					delta: {
						gradientEnergy: wgMetrics.gradientEnergy - refMetrics.gradientEnergy,
						variance: wgMetrics.variance - refMetrics.variance,
						entropy: wgMetrics.entropy - refMetrics.entropy
					}
				};
			} else {
				report.utciComplexity = {
					ref: { gradientEnergy: 0, variance: 0, entropy: 0 },
					webgpu: { gradientEnergy: 0, variance: 0, entropy: 0 },
					delta: { gradientEnergy: 0, variance: 0, entropy: 0 },
					error: webgpuPositions
						? 'unable to infer rectangular grid shape from positions'
						: 'webgpu_utci.json is missing valid positions (numPoints*3), cannot compute topology-aware complexity'
				};
			}
		} catch (e) {
			report.utciComplexity = {
				ref: { gradientEnergy: 0, variance: 0, entropy: 0 },
				webgpu: { gradientEnergy: 0, variance: 0, entropy: 0 },
				delta: { gradientEnergy: 0, variance: 0, entropy: 0 },
				error: e instanceof Error ? e.message : String(e)
			};
		}
	}

	report.summary = { pass: failCount === 0, failCount };
	return report;
}
