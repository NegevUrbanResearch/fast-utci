import * as THREE from 'three';
import { createRectangularGridFromBounds } from '$lib/compute/core/grid-generator';
import {
	analysisBoundsToViewerRectangularBounds,
	type ViewerRectangularBounds
} from '$lib/compute/core/canonicalGridAxes';
import type { AnalysisCoordinateSystem, AnalysisRectangularBounds } from '$lib/types/analysis';

export type AnalysisBounds = AnalysisRectangularBounds;
export type { ViewerRectangularBounds };
export { analysisBoundsToViewerRectangularBounds };

/**
 * Map analysis metadata bounds to a rectangular grid in viewer Y-up world coordinates.
 * For xy_ground: analysis (x, y) with fixed z → viewer X = x, Z = -y, Y = bounds.z.
 * For xz_ground: analysis (x, z) with fixed y → viewer X = x, Z = z, Y = bounds.z (or y).
 */
export function analysisBoundsToRectangularGrid(params: {
	bounds: AnalysisBounds;
	gridSize: number;
	coordinateSystem: AnalysisCoordinateSystem;
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
