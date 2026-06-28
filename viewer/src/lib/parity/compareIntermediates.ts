export interface CompareIntermediatesResult {
	pass: boolean;
	rmse: number;
	maxError: number;
	numPoints: number;
}

/** Stats for distribution comparison (grid sizes may differ). */
export interface IntermediatesStats {
	mean: number;
	max: number;
	min: number;
	std: number;
	n: number;
}

function computeStats(arr: Float32Array | number[]): IntermediatesStats {
	const n = arr.length;
	if (n === 0) return { mean: 0, max: 0, min: 0, std: 0, n: 0 };
	let sum = 0;
	let min = arr[0];
	let max = arr[0];
	for (let i = 0; i < n; i++) {
		const v = arr[i];
		sum += v;
		if (v < min) min = v;
		if (v > max) max = v;
	}
	const mean = sum / n;
	let sumSq = 0;
	for (let i = 0; i < n; i++) {
		const d = arr[i] - mean;
		sumSq += d * d;
	}
	const std = n > 1 ? Math.sqrt(sumSq / (n - 1)) : 0;
	return { mean, max, min, std, n };
}

export interface CompareIntermediatesStatsResult {
	pass: boolean;
	meanDiff: number;
	maxDiff: number;
	refStats: IntermediatesStats;
	webgpuStats: IntermediatesStats;
}

/**
 * Compare reference and WebGPU intermediate arrays by distribution (mean, max).
 * Does not require same grid size; use when pipelines use different point sets.
 */
export function compareIntermediatesStats(params: {
	ref: Float32Array | number[];
	webgpu: Float32Array | number[];
	/** Max allowed absolute difference in mean (e.g. 0.02 for 0–1 exposure). */
	toleranceMean?: number;
	/** Max allowed absolute difference in max (e.g. 0.05). */
	toleranceMax?: number;
}): CompareIntermediatesStatsResult {
	const { ref, webgpu, toleranceMean = 0.02, toleranceMax = 0.05 } = params;
	const refStats = computeStats(ref);
	const webgpuStats = computeStats(webgpu);
	const meanDiff = Math.abs(webgpuStats.mean - refStats.mean);
	const maxDiff = Math.abs(webgpuStats.max - refStats.max);
	const pass = meanDiff <= toleranceMean && maxDiff <= toleranceMax;
	return {
		pass,
		meanDiff,
		maxDiff,
		refStats,
		webgpuStats
	};
}

/**
 * Compare reference and WebGPU intermediate arrays (e.g. solar or sky exposure).
 * Pure function; safe to run in Node or browser.
 * Requires same length (same grid).
 */
export function compareIntermediates(params: {
	ref: Float32Array;
	webgpu: Float32Array;
	tolerance?: number;
	allowedOutliers?: number;
}): CompareIntermediatesResult & { outlierCount: number; allowedOutliers: number } {
	const { ref, webgpu, tolerance = 1e-5, allowedOutliers = 0 } = params;
	if (ref.length !== webgpu.length) {
		throw new Error(`Length mismatch: ref ${ref.length} vs webgpu ${webgpu.length}`);
	}
	const n = ref.length;
	if (n === 0) {
		return { pass: true, rmse: 0, maxError: 0, numPoints: 0, outlierCount: 0, allowedOutliers };
	}
	let sumSq = 0;
	let maxError = 0;
	let outlierCount = 0;
	for (let i = 0; i < n; i++) {
		const d = webgpu[i] - ref[i];
		sumSq += d * d;
		const absD = Math.abs(d);
		if (absD > tolerance) outlierCount++;
		if (absD > maxError) maxError = absD;
	}
	const rmse = Math.sqrt(sumSq / n);
	const pass = outlierCount <= allowedOutliers;
	return {
		pass,
		rmse,
		maxError,
		numPoints: n,
		outlierCount,
		allowedOutliers
	};
}

/** Percentile of a sorted array (0–1). */
function percentile(sorted: number[], p: number): number {
	if (sorted.length === 0) return 0;
	const i = (p / 100) * (sorted.length - 1);
	const lo = Math.floor(i);
	const hi = Math.ceil(i);
	if (lo === hi) return sorted[lo];
	return sorted[lo] + (i - lo) * (sorted[hi] - sorted[lo]);
}

export interface DiffDetail {
	index: number;
	ref: number;
	webgpu: number;
	diff: number;
}

export interface AnalyzeDiffsResult {
	/** Only set when ref.length === webgpu.length. */
	sameLength: boolean;
	n: number;
	/** Stats of (webgpu - ref) when same length. */
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
	/** Up to maxWorst indices with largest |diff| (point index, ref, webgpu, diff). */
	worstIndices?: DiffDetail[];
}

/**
 * Analyze per-element diffs for digging into parity failures.
 * When ref and webgpu have the same length, returns diff stats and worst indices.
 */
export function analyzeDiffs(params: {
	ref: Float32Array | number[];
	webgpu: Float32Array | number[];
	/** Max number of worst indices to return (default 20). */
	maxWorst?: number;
}): AnalyzeDiffsResult {
	const { ref, webgpu, maxWorst = 20 } = params;
	const nRef = ref.length;
	const nWeb = webgpu.length;
	if (nRef !== nWeb) {
		return { sameLength: false, n: nRef };
	}
	const n = nRef;
	if (n === 0) {
		return { sameLength: true, n: 0, diffStats: { mean: 0, std: 0, min: 0, max: 0, rmse: 0, p50: 0, p95: 0, p99: 0 }, worstIndices: [] };
	}
	const diffs: number[] = [];
	let sumSq = 0;
	let minDiff = Number.POSITIVE_INFINITY;
	let maxDiff = Number.NEGATIVE_INFINITY;
	for (let i = 0; i < n; i++) {
		const d = webgpu[i] - ref[i];
		diffs.push(d);
		sumSq += d * d;
		if (d < minDiff) minDiff = d;
		if (d > maxDiff) maxDiff = d;
	}
	const mean = diffs.reduce((a, b) => a + b, 0) / n;
	const std = n > 1 ? Math.sqrt(diffs.reduce((s, d) => s + (d - mean) ** 2, 0) / (n - 1)) : 0;
	const sorted = [...diffs].map((d) => Math.abs(d)).sort((a, b) => a - b);
	const diffStats = {
		mean,
		std,
		min: minDiff,
		max: maxDiff,
		rmse: Math.sqrt(sumSq / n),
		p50: percentile(sorted, 50),
		p95: percentile(sorted, 95),
		p99: percentile(sorted, 99)
	};
	const withIndex = diffs.map((d, i) => ({ index: i, ref: ref[i], webgpu: webgpu[i], diff: d }));
	withIndex.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
	const worstIndices = withIndex.slice(0, maxWorst);
	return {
		sameLength: true,
		n,
		diffStats,
		worstIndices
	};
}
