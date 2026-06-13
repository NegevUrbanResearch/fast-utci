import type {
	AnalysisCoordinateSystem,
	AnalysisRectangularBounds,
	ProjectedTriangle2D,
	StudyAreaMask
} from '$lib/types/analysis';
import {
	canonicalAxisIndexRange,
	resolveCanonicalGridAxes
} from '$lib/compute/core/canonicalGridAxes';

const FNV32_OFFSET = 0x811c9dc5;
const FNV32_PRIME = 0x01000193;

export interface StudyAreaMaskParams {
	bounds: AnalysisRectangularBounds;
	gridSize: number;
	coordinateSystem: AnalysisCoordinateSystem;
	triangles: readonly ProjectedTriangle2D[];
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

function isPointInTriangle(
	px: number,
	py: number,
	triangle: ProjectedTriangle2D,
	epsilon = Number.EPSILON * 16
): boolean {
	const [ax, ay, bx, by, cx, cy] = triangle;
	const v0x = cx - ax;
	const v0y = cy - ay;
	const v1x = bx - ax;
	const v1y = by - ay;
	const v2x = px - ax;
	const v2y = py - ay;

	const dot00 = v0x * v0x + v0y * v0y;
	const dot01 = v0x * v1x + v0y * v1y;
	const dot02 = v0x * v2x + v0y * v2y;
	const dot11 = v1x * v1x + v1y * v1y;
	const dot12 = v1x * v2x + v1y * v2y;
	const denominator = dot00 * dot11 - dot01 * dot01;

	if (Math.abs(denominator) <= epsilon) {
		return false;
	}

	const invDenominator = 1 / denominator;
	const u = (dot11 * dot02 - dot01 * dot12) * invDenominator;
	const v = (dot00 * dot12 - dot01 * dot02) * invDenominator;

	return u >= -epsilon && v >= -epsilon && u + v <= 1 + epsilon;
}

function buildFootprintChecksum(params: StudyAreaMaskParams): string {
	const quantized: number[] = [
		roundToMicrounit(params.bounds.x_min),
		roundToMicrounit(params.bounds.x_max),
		roundToMicrounit(params.bounds.y_min),
		roundToMicrounit(params.bounds.y_max),
		roundToMicrounit(params.bounds.z ?? 0),
		roundToMicrounit(params.gridSize),
		params.coordinateSystem === 'xy_ground' ? 1 : 2
	];

	for (const triangle of params.triangles) {
		for (const value of triangle) {
			quantized.push(roundToMicrounit(value));
		}
	}

	return fnv1a32Hash(quantized);
}

export function buildStudyAreaMaskFromProjectedTriangles(
	params: StudyAreaMaskParams
): StudyAreaMask {
	const { bounds, gridSize, triangles, coordinateSystem } = params;
	if (gridSize <= 0) {
		throw new Error('gridSize must be positive');
	}

	const {
		xValues,
		zValues,
		minX: canonicalMinX,
		maxX: canonicalMaxX,
		minZ: canonicalMinZ,
		maxZ: canonicalMaxZ,
		width,
		height,
		descendingZ
	} =
		resolveCanonicalGridAxes({
			bounds,
			gridSize,
			coordinateSystem
		});
	const canonicalPointCount = width * height;
	const mask = new Uint8Array(canonicalPointCount);

	for (const triangle of triangles) {
		const xs = [triangle[0], triangle[2], triangle[4]];
		const ys = [triangle[1], triangle[3], triangle[5]];
		const minX = Math.min(...xs);
		const maxX = Math.max(...xs);
		const minY = Math.min(...ys);
		const maxY = Math.max(...ys);
		const { start: startX, end: endX } = canonicalAxisIndexRange({
			minAxisValue: canonicalMinX,
			maxAxisValue: canonicalMaxX,
			gridSize,
			minValue: minX,
			maxValue: maxX,
			axisLength: width
		});
		const { start: startZ, end: endZ } = canonicalAxisIndexRange({
			minAxisValue: canonicalMinZ,
			maxAxisValue: canonicalMaxZ,
			gridSize,
			minValue: minY,
			maxValue: maxY,
			axisLength: height,
			descending: descendingZ
		});

		for (let xIndex = startX; xIndex <= endX; xIndex++) {
			const x = xValues[xIndex];
			for (let zIndex = startZ; zIndex <= endZ; zIndex++) {
				const canonicalIndex = xIndex * height + zIndex;
				if (mask[canonicalIndex]) {
					continue;
				}

				if (isPointInTriangle(x, zValues[zIndex], triangle)) {
					mask[canonicalIndex] = 1;
				}
			}
		}
	}

	const activeCanonicalIndices = new Uint32Array(mask.reduce((count, active) => count + active, 0));
	let activeOffset = 0;
	for (let i = 0; i < mask.length; i++) {
		if (mask[i]) {
			activeCanonicalIndices[activeOffset++] = i;
		}
	}

	const footprintChecksum = buildFootprintChecksum(params);
	const maskChecksum = fnv1a32Hash(Array.from(mask));

	return {
		canonicalPointCount,
		activePointCount: activeCanonicalIndices.length,
		mask,
		activeCanonicalIndices,
		width,
		height,
		footprintChecksum,
		maskChecksum,
		signature: `${coordinateSystem}:${roundToMicrounit(gridSize)}:${width}x${height}:${footprintChecksum}:${maskChecksum}`
	};
}
