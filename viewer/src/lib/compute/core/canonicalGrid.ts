import type { AnalysisRectangularBounds } from '$lib/types/analysis';
import { resolveCanonicalGridAxes } from '$lib/compute/core/canonicalGridAxes';
const FNV32_OFFSET = 0x811c9dc5;
const FNV32_PRIME = 0x01000193;

export interface CanonicalGridParams {
	bounds: AnalysisRectangularBounds;
	gridSize: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	zHeight?: number;
	originOffset?: { x: number; y: number; z: number };
}

export interface CanonicalGridResult {
	points: Float32Array;
	numPoints: number;
}

function roundToMicrounit(value: number): number {
	return Math.round(value * 1_000_000);
}

export function canonicalGridPoints(params: CanonicalGridParams): CanonicalGridResult {
	const { xValues, zValues, y, ox, oy, oz } = resolveCanonicalGridAxes(params);
	const numPoints = xValues.length * zValues.length;
	const points = new Float32Array(numPoints * 3);
	let offset = 0;
	for (const x of xValues) {
		for (const z of zValues) {
			points[offset++] = x + ox;
			points[offset++] = y + oy;
			points[offset++] = z + oz;
		}
	}

	return {
		points,
		numPoints
	};
}

export function canonicalGridPointsForActiveIndices(
	params: CanonicalGridParams & { activeCanonicalIndices: Uint32Array }
): CanonicalGridResult {
	const { activeCanonicalIndices } = params;
	const { xValues, zValues, y, ox, oy, oz } = resolveCanonicalGridAxes(params);
	const rowCount = zValues.length;
	const canonicalPointCount = xValues.length * rowCount;
	const values = new Float32Array(activeCanonicalIndices.length * 3);

	for (let i = 0; i < activeCanonicalIndices.length; i++) {
		const canonicalIndex = activeCanonicalIndices[i];
		if (canonicalIndex >= canonicalPointCount) {
			throw new Error(`active canonical index ${canonicalIndex} is out of range`);
		}

		const xIndex = Math.floor(canonicalIndex / rowCount);
		const zIndex = canonicalIndex % rowCount;
		const offset = i * 3;
		values[offset] = xValues[xIndex] + ox;
		values[offset + 1] = y + oy;
		values[offset + 2] = zValues[zIndex] + oz;
	}

	return {
		points: values,
		numPoints: activeCanonicalIndices.length
	};
}

export function canonicalGridChecksum(params: CanonicalGridParams): string {
	const { points } = canonicalGridPoints(params);
	let hash = FNV32_OFFSET;
	for (const point of points) {
		hash ^= roundToMicrounit(point) >>> 0;
		hash = Math.imul(hash, FNV32_PRIME) >>> 0;
	}
	return hash.toString(16).padStart(8, '0');
}
