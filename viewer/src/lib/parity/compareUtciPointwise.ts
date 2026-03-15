export interface UtciWorstCell {
	hour: number;
	pointIndex: number;
	ref: number;
	webgpu: number;
	diff: number;
}

export interface CompareUtciPointwiseResult {
	pass: boolean;
	rmse: number;
	maxError: number;
	meanDiff: number;
	numHours: number;
	numPoints: number;
	numValues: number;
	worst: UtciWorstCell | null;
}

export function compareUtciPointwise(params: {
	ref: readonly (readonly number[])[];
	webgpu: readonly (readonly number[])[];
	tolerance: number;
}): CompareUtciPointwiseResult {
	const { ref, webgpu, tolerance } = params;
	if (ref.length !== webgpu.length) {
		throw new Error(`UTCI hour count mismatch: ref ${ref.length} vs webgpu ${webgpu.length}`);
	}
	if (ref.length === 0) {
		return {
			pass: true,
			rmse: 0,
			maxError: 0,
			meanDiff: 0,
			numHours: 0,
			numPoints: 0,
			numValues: 0,
			worst: null
		};
	}

	const numHours = ref.length;
	const numPoints = ref[0].length;
	for (let hour = 0; hour < numHours; hour++) {
		if (ref[hour].length !== numPoints) {
			throw new Error(`UTCI ref shape mismatch at hour ${hour}: expected ${numPoints}, got ${ref[hour].length}`);
		}
		if (webgpu[hour].length !== numPoints) {
			throw new Error(
				`UTCI webgpu shape mismatch at hour ${hour}: expected ${numPoints}, got ${webgpu[hour].length}`
			);
		}
	}

	let sumSq = 0;
	let sumDiff = 0;
	let maxError = 0;
	let worst: UtciWorstCell | null = null;
	const numValues = numHours * numPoints;
	for (let hour = 0; hour < numHours; hour++) {
		for (let pointIndex = 0; pointIndex < numPoints; pointIndex++) {
			const refValue = ref[hour][pointIndex];
			const webgpuValue = webgpu[hour][pointIndex];
			const diff = webgpuValue - refValue;
			const absDiff = Math.abs(diff);
			sumSq += diff * diff;
			sumDiff += diff;
			if (absDiff > maxError || worst === null) {
				maxError = absDiff;
				worst = { hour, pointIndex, ref: refValue, webgpu: webgpuValue, diff };
			}
		}
	}

	return {
		pass: maxError <= tolerance,
		rmse: numValues > 0 ? Math.sqrt(sumSq / numValues) : 0,
		maxError,
		meanDiff: numValues > 0 ? sumDiff / numValues : 0,
		numHours,
		numPoints,
		numValues,
		worst
	};
}
