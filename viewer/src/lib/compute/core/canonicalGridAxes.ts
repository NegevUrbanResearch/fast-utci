import type { AnalysisCoordinateSystem, AnalysisRectangularBounds } from '$lib/types/analysis';

export const CANONICAL_GRID_EPSILON = 1e-9;

export interface CanonicalGridAxesParams {
	bounds: AnalysisRectangularBounds;
	gridSize: number;
	coordinateSystem: AnalysisCoordinateSystem;
	zHeight?: number;
	originOffset?: { x: number; y: number; z: number };
}

export interface CanonicalGridAxes {
	xValues: number[];
	zValues: number[];
	y: number;
	ox: number;
	oy: number;
	oz: number;
	minX: number;
	maxX: number;
	minZ: number;
	maxZ: number;
	width: number;
	height: number;
	descendingZ: boolean;
}

export interface ViewerRectangularBounds {
	minX: number;
	maxX: number;
	minZ: number;
	maxZ: number;
}

function buildAxisValues(params: {
	start: number;
	end: number;
	step: number;
	descending?: boolean;
}): number[] {
	const { start, end, step, descending = false } = params;
	const values: number[] = [];

	if (descending) {
		for (let value = end; value >= start - CANONICAL_GRID_EPSILON; value -= step) {
			values.push(value);
		}
		return values;
	}

	for (let value = start; value <= end + CANONICAL_GRID_EPSILON; value += step) {
		values.push(value);
	}
	return values;
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(Math.max(value, min), max);
}

export function resolveCanonicalViewerRectangularBounds(params: {
	bounds: AnalysisRectangularBounds;
	coordinateSystem: AnalysisCoordinateSystem;
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

export const analysisBoundsToViewerRectangularBounds = resolveCanonicalViewerRectangularBounds;

export function resolveCanonicalGridAxes(params: CanonicalGridAxesParams): CanonicalGridAxes {
	const { bounds, gridSize, coordinateSystem, zHeight, originOffset } = params;
	if (gridSize <= 0) {
		throw new Error('gridSize must be positive');
	}

	const { minX, maxX, minZ, maxZ } = resolveCanonicalViewerRectangularBounds({
		bounds,
		coordinateSystem
	});
	const descendingZ = coordinateSystem === 'xy_ground';
	const xValues = buildAxisValues({ start: minX, end: maxX, step: gridSize });
	const zValues = buildAxisValues({
		start: minZ,
		end: maxZ,
		step: gridSize,
		descending: descendingZ
	});

	return {
		xValues,
		zValues,
		y: zHeight ?? bounds.z ?? 0,
		ox: originOffset?.x ?? 0,
		oy: originOffset?.y ?? 0,
		oz: originOffset?.z ?? 0,
		minX,
		maxX,
		minZ,
		maxZ,
		width: xValues.length,
		height: zValues.length,
		descendingZ
	};
}

export function canonicalAxisIndexRange(params: {
	minAxisValue: number;
	maxAxisValue: number;
	gridSize: number;
	minValue: number;
	maxValue: number;
	axisLength: number;
	descending?: boolean;
}): { start: number; end: number } {
	const {
		minAxisValue,
		maxAxisValue,
		gridSize,
		minValue,
		maxValue,
		axisLength,
		descending = false
	} = params;

	if (descending) {
		return {
			start: clamp(
				Math.ceil((maxAxisValue - maxValue - CANONICAL_GRID_EPSILON) / gridSize),
				0,
				axisLength - 1
			),
			end: clamp(
				Math.floor((maxAxisValue - minValue + CANONICAL_GRID_EPSILON) / gridSize),
				0,
				axisLength - 1
			)
		};
	}

	return {
		start: clamp(
			Math.ceil((minValue - minAxisValue - CANONICAL_GRID_EPSILON) / gridSize),
			0,
			axisLength - 1
		),
		end: clamp(
			Math.floor((maxValue - minAxisValue + CANONICAL_GRID_EPSILON) / gridSize),
			0,
			axisLength - 1
		)
	};
}
