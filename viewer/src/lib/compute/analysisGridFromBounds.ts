import * as THREE from 'three';
import { createRectangularGridFromBounds } from '$lib/compute/grid-generator';

export interface AnalysisBounds {
	x_min: number;
	x_max: number;
	y_min: number;
	y_max: number;
	z?: number;
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
}): { points: THREE.Vector3[]; normals: THREE.Vector3[] } {
	const { bounds, gridSize, coordinateSystem } = params;
	const zHeight = bounds.z ?? 0;

	if (coordinateSystem === 'xy_ground') {
		// Analysis grid is (x, y) with z = bounds.z. Viewer: X_world = x, Z_world = -y, Y_world = bounds.z.
		const minX = bounds.x_min;
		const maxX = bounds.x_max;
		const minZ = -bounds.y_max;
		const maxZ = -bounds.y_min;
		return createRectangularGridFromBounds(
			{ min: [minX, minZ], max: [maxX, maxZ] },
			gridSize,
			zHeight
		);
	}

	// xz_ground: use bounds as (x_min, x_max), (y_min, y_max) as z range, Y = z height
	const minX = bounds.x_min;
	const maxX = bounds.x_max;
	const minZ = bounds.y_min;
	const maxZ = bounds.y_max;
	return createRectangularGridFromBounds(
		{ min: [minX, minZ], max: [maxX, maxZ] },
		gridSize,
		zHeight
	);
}
