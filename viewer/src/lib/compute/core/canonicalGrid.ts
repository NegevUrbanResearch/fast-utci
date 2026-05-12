import type { AnalysisBounds } from '$lib/compute/core/analysisGridFromBounds';
import { analysisBoundsToViewerRectangularBounds } from '$lib/compute/core/analysisGridFromBounds';

const EPSILON = 1e-9;
const FNV32_OFFSET = 0x811c9dc5;
const FNV32_PRIME = 0x01000193;

export interface CanonicalGridParams {
	bounds: AnalysisBounds;
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
	const { bounds, gridSize, coordinateSystem, zHeight, originOffset } = params;
	if (gridSize <= 0) {
		throw new Error('gridSize must be positive');
	}

	const { minX, maxX, minZ, maxZ } = analysisBoundsToViewerRectangularBounds({
		bounds,
		coordinateSystem
	});
	const y = zHeight ?? bounds.z ?? 0;
	const ox = originOffset?.x ?? 0;
	const oy = originOffset?.y ?? 0;
	const oz = originOffset?.z ?? 0;

	const values: number[] = [];
	for (let x = minX; x <= maxX + EPSILON; x += gridSize) {
		if (coordinateSystem === 'xy_ground') {
			for (let z = maxZ; z >= minZ - EPSILON; z -= gridSize) {
				values.push(x + ox, y + oy, z + oz);
			}
		} else {
			for (let z = minZ; z <= maxZ + EPSILON; z += gridSize) {
				values.push(x + ox, y + oy, z + oz);
			}
		}
	}

	return {
		points: new Float32Array(values),
		numPoints: values.length / 3
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
