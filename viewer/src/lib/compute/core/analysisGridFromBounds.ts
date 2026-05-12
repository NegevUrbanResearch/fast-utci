import * as THREE from 'three';
import { createRectangularGridFromBounds } from '$lib/compute/core/grid-generator';

export interface AnalysisBounds {
	x_min: number;
	x_max: number;
	y_min: number;
	y_max: number;
	z?: number;
}

export interface ViewerRectangularBounds {
	minX: number;
	maxX: number;
	minZ: number;
	maxZ: number;
}

export function analysisBoundsToViewerRectangularBounds(params: {
	bounds: AnalysisBounds;
	coordinateSystem: 'xy_ground' | 'xz_ground';
}): ViewerRectangularBounds {
	const { bounds, coordinateSystem } = params;
	if (coordinateSystem === 'xy_ground') {
		return {
			minX: bounds.x_min,
			maxX: bounds.x_max,
			minZ: -bounds.y_max,
			maxZ: -bounds.y_min
		};
	}

	return {
		minX: bounds.x_min,
		maxX: bounds.x_max,
		minZ: bounds.y_min,
		maxZ: bounds.y_max
	};
}

/**
 * Map analysis metadata bounds to a rectangular grid in viewer Y-up world coordinates.
 * For xy_ground: analysis (x, y) with fixed z → viewer X = x, Z = -y, Y = bounds.z.
 * For xz_ground: analysis (x, z) with fixed y → viewer X = x, Z = z, Y = bounds.z (or y).
 */
export function analysisBoundsToRectangularGrid(params: {
	bounds: AnalysisBounds;
	gridSize: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	gridZHeight?: number;
}): { points: THREE.Vector3[]; normals: THREE.Vector3[] } {
	const { bounds, gridSize, coordinateSystem, gridZHeight } = params;
	const zHeight = gridZHeight ?? bounds.z ?? 0;
	const { minX, maxX, minZ, maxZ } = analysisBoundsToViewerRectangularBounds({
		bounds,
		coordinateSystem
	});
	return createRectangularGridFromBounds(
		{ min: [minX, minZ], max: [maxX, maxZ] },
		gridSize,
		zHeight
	);
}
