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

function fnv1a32Hash(values: readonly number[]): string {
	let hash = FNV32_OFFSET;
	for (const value of values) {
		hash ^= value >>> 0;
		hash = Math.imul(hash, FNV32_PRIME) >>> 0;
	}
	return hash.toString(16).padStart(8, '0');
}

export function canonicalGridPoints(params: CanonicalGridParams): CanonicalGridResult {
	const { xValues, zValues, y, ox, oy, oz } = resolveCanonicalGridAxes(params);
	const values: number[] = [];
	for (const x of xValues) {
		for (const z of zValues) {
			values.push(x + ox, y + oy, z + oz);
		}
	}

	return {
		points: new Float32Array(values),
		numPoints: values.length / 3
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
	const quantized = new Array<number>(points.length);
	for (let i = 0; i < points.length; i++) {
		quantized[i] = roundToMicrounit(points[i]);
	}
	return fnv1a32Hash(quantized);
}
