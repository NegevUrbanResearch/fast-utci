/**
 * Compare UTCI range (min, max, mean) between reference and WebGPU with tolerances.
 */

export interface CompareUtciRangeParams {
	ref: { min: number; max: number; mean: number };
	webgpu: { min: number; max: number; mean: number };
	toleranceMin?: number;
	toleranceMax?: number;
	toleranceMean?: number;
}

export interface CompareUtciRangeResult {
	pass: boolean;
	minDiff?: number;
	maxDiff?: number;
	meanDiff?: number;
}

export function compareUtciRange(params: CompareUtciRangeParams): CompareUtciRangeResult {
	const {
		ref,
		webgpu,
		toleranceMin = 2,
		toleranceMax = 2,
		toleranceMean = 1,
	} = params;
	const minDiff = Math.abs(webgpu.min - ref.min);
	const maxDiff = Math.abs(webgpu.max - ref.max);
	const meanDiff = Math.abs(webgpu.mean - ref.mean);
	const pass = minDiff <= toleranceMin && maxDiff <= toleranceMax && meanDiff <= toleranceMean;
	return {
		pass,
		minDiff,
		maxDiff,
		meanDiff,
	};
}
