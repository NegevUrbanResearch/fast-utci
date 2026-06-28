export interface PointwiseIndex {
	pointIndex: number;
	hourIndex: number;
}

/**
 * Point-major flattening used by WebGPU and parity intermediates:
 * flatIndex = pointIndex * numHours + hourIndex
 */
export function pointwiseIndexFromFlat(flatIndex: number, numHours: number): PointwiseIndex {
	if (!Number.isInteger(flatIndex) || flatIndex < 0) {
		throw new Error(`flatIndex must be a non-negative integer, got ${flatIndex}`);
	}
	if (!Number.isInteger(numHours) || numHours <= 0) {
		throw new Error(`numHours must be a positive integer, got ${numHours}`);
	}
	return {
		pointIndex: Math.floor(flatIndex / numHours),
		hourIndex: flatIndex % numHours
	};
}

export function flatIndexFromPointwise(pointIndex: number, hourIndex: number, numHours: number): number {
	if (!Number.isInteger(pointIndex) || pointIndex < 0) {
		throw new Error(`pointIndex must be a non-negative integer, got ${pointIndex}`);
	}
	if (!Number.isInteger(hourIndex) || hourIndex < 0) {
		throw new Error(`hourIndex must be a non-negative integer, got ${hourIndex}`);
	}
	if (!Number.isInteger(numHours) || numHours <= 0) {
		throw new Error(`numHours must be a positive integer, got ${numHours}`);
	}
	if (hourIndex >= numHours) {
		throw new Error(`hourIndex must be < numHours (${numHours}), got ${hourIndex}`);
	}
	return pointIndex * numHours + hourIndex;
}
