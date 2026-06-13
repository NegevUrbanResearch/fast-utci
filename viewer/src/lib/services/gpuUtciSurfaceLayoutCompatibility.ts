import type { F32MetricType } from '$lib/compute/on-demand/onDemandOutputFormat';
import type { UtciGridLayout } from './pointCloudService';

export type GpuNativeUtciSurfaceSource =
	| 'cpu-uploaded-selected-hour'
	| 'compute-buffer-selected-hour';

export type UtciGridLayoutPointCompatibilityEvaluation = {
	compatible: boolean | null;
	cellToPointMappingMatch: boolean | null;
	requiredExpensiveMappingComparison: boolean;
	performedExpensiveMappingComparison: boolean;
};

export type ComputeBufferUtciSurfaceLayoutCompatibilityStateSnapshot = {
	source: GpuNativeUtciSurfaceSource;
	metricType?: F32MetricType;
	width: number;
	height: number;
	gridSize: number;
	vertexCount: number;
	storageCount: number;
};

export type ComputeBufferUtciSurfaceLayoutCompatibilityEvaluation = {
	compatible: boolean | null;
	missingState: boolean;
	wrongSource: boolean;
	widthMatch: boolean | null;
	heightMatch: boolean | null;
	gridSizeMatch: boolean | null;
	metricTypeMatch: boolean | null;
	vertexCountMatch: boolean | null;
	storageCountMatch: boolean | null;
	pointCompatibility: UtciGridLayoutPointCompatibilityEvaluation | null;
};

export const GPU_NATIVE_SURFACE_STATE_KEY = 'gpuNativeUtciSurfaceState';
const SURFACE_VERTICES_PER_CELL = 6;

export function evaluateComputeBufferUtciSurfaceLayoutCompatibility(params: {
	state: ComputeBufferUtciSurfaceLayoutCompatibilityStateSnapshot | null | undefined;
	previousLayout: UtciGridLayout | null | undefined;
	nextLayout: UtciGridLayout;
	metricType?: F32MetricType;
	allowExpensiveMappingComparison?: boolean;
}): ComputeBufferUtciSurfaceLayoutCompatibilityEvaluation {
	const state = params.state ?? null;
	if (!state) {
		return {
			compatible: false,
			missingState: true,
			wrongSource: false,
			widthMatch: null,
			heightMatch: null,
			gridSizeMatch: null,
			metricTypeMatch: null,
			vertexCountMatch: null,
			storageCountMatch: null,
			pointCompatibility: null
		};
	}

	if (state.source !== 'compute-buffer-selected-hour') {
		return {
			compatible: false,
			missingState: false,
			wrongSource: true,
			widthMatch: null,
			heightMatch: null,
			gridSizeMatch: null,
			metricTypeMatch: null,
			vertexCountMatch: null,
			storageCountMatch: null,
			pointCompatibility: null
		};
	}

	const expectedVertexCount = getComputeBufferSurfaceVertexCount(params.nextLayout);
	const widthMatch = state.width === params.nextLayout.width;
	const metricTypeMatch =
		!params.metricType || !state.metricType || state.metricType === params.metricType;
	const heightMatch = state.height === params.nextLayout.height;
	const gridSizeMatch = state.gridSize === params.nextLayout.gridSize;
	const vertexCountMatch = state.vertexCount === expectedVertexCount;
	const storageCountMatch = state.storageCount === params.nextLayout.numPositions;

	if (!params.previousLayout) {
		return {
			compatible: false,
			missingState: false,
			wrongSource: false,
			widthMatch,
			heightMatch,
			gridSizeMatch,
			metricTypeMatch,
			vertexCountMatch,
			storageCountMatch,
			pointCompatibility: null
		};
	}

	const pointCompatibility = evaluateUtciGridLayoutsPointCompatibility(
		params.previousLayout,
		params.nextLayout,
		{
			allowExpensiveMappingComparison: params.allowExpensiveMappingComparison
		}
	);

	return {
		compatible:
			widthMatch &&
			metricTypeMatch &&
			heightMatch &&
			gridSizeMatch &&
			vertexCountMatch &&
			storageCountMatch
				? pointCompatibility.compatible
				: false,
		missingState: false,
		wrongSource: false,
		widthMatch,
		heightMatch,
		gridSizeMatch,
		metricTypeMatch,
		vertexCountMatch,
		storageCountMatch,
		pointCompatibility
	};
}

export function evaluateUtciGridLayoutsPointCompatibility(
	previousLayout: UtciGridLayout,
	nextLayout: UtciGridLayout,
	options?: {
		allowExpensiveMappingComparison?: boolean;
	}
): UtciGridLayoutPointCompatibilityEvaluation {
	const allowExpensiveMappingComparison =
		options?.allowExpensiveMappingComparison ?? true;

	if (
		previousLayout.numPositions !== nextLayout.numPositions ||
		previousLayout.coordinateSystem !== nextLayout.coordinateSystem ||
		previousLayout.minX !== nextLayout.minX ||
		previousLayout.minZ !== nextLayout.minZ ||
		previousLayout.baseY !== nextLayout.baseY
	) {
		return {
			compatible: false,
			cellToPointMappingMatch: null,
			requiredExpensiveMappingComparison: false,
			performedExpensiveMappingComparison: false
		};
	}

	if (
		previousLayout.cellToPointIndex &&
		nextLayout.cellToPointIndex &&
		previousLayout.cellToPointIndex.length === nextLayout.cellToPointIndex.length
	) {
		const cellCount = previousLayout.width * previousLayout.height;
		if (
			previousLayout.cellToPointIndex.length !== cellCount ||
			nextLayout.cellToPointIndex.length !== cellCount
		) {
			return {
				compatible: false,
				cellToPointMappingMatch: false,
				requiredExpensiveMappingComparison: false,
				performedExpensiveMappingComparison: false
			};
		}
		if (
			hasAmbiguousCellEntries(previousLayout.cellToPointIndex) ||
			hasAmbiguousCellEntries(nextLayout.cellToPointIndex)
		) {
			if (!allowExpensiveMappingComparison) {
				return {
					compatible: null,
					cellToPointMappingMatch: null,
					requiredExpensiveMappingComparison: true,
					performedExpensiveMappingComparison: false
				};
			}

			const mappingMatch = uint32ArraysEqual(
				createCellToPointIndexArray(previousLayout),
				createCellToPointIndexArray(nextLayout)
			);
			return {
				compatible: mappingMatch,
				cellToPointMappingMatch: mappingMatch,
				requiredExpensiveMappingComparison: true,
				performedExpensiveMappingComparison: true
			};
		}

		const mappingMatch = int32ArraysEqual(
			previousLayout.cellToPointIndex,
			nextLayout.cellToPointIndex
		);
		return {
			compatible: mappingMatch,
			cellToPointMappingMatch: mappingMatch,
			requiredExpensiveMappingComparison: false,
			performedExpensiveMappingComparison: false
		};
	}
	if (previousLayout.cellToPointIndex || nextLayout.cellToPointIndex) {
		return {
			compatible: false,
			cellToPointMappingMatch: false,
			requiredExpensiveMappingComparison: false,
			performedExpensiveMappingComparison: false
		};
	}

	const mappingMatch =
		uint32ArraysEqual(previousLayout.indexToRow, nextLayout.indexToRow) &&
		uint32ArraysEqual(previousLayout.indexToColumn, nextLayout.indexToColumn);
	return {
		compatible: mappingMatch,
		cellToPointMappingMatch: mappingMatch,
		requiredExpensiveMappingComparison: false,
		performedExpensiveMappingComparison: false
	};
}

export function areUtciGridLayoutsPointCompatible(
	previousLayout: UtciGridLayout,
	nextLayout: UtciGridLayout
): boolean {
	return (
		evaluateUtciGridLayoutsPointCompatibility(previousLayout, nextLayout, {
			allowExpensiveMappingComparison: true
		}).compatible ?? false
	);
}

export function createCellToPointIndexArray(layout: UtciGridLayout): Uint32Array {
	const cellCount = layout.width * layout.height;
	const inactivePointIndex = layout.numPositions;
	const cellToPoint = getCellToPointIndex(layout, cellCount);
	const indices = new Uint32Array(cellCount);
	for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
		const mappedPointIndex = cellToPoint[cellIndex] ?? -1;
		indices[cellIndex] = mappedPointIndex >= 0 ? mappedPointIndex : inactivePointIndex;
	}

	return indices;
}

export function createVertexToPointIndexArray(layout: UtciGridLayout): Uint32Array {
	const cellCount = layout.width * layout.height;
	const cellToPoint = createCellToPointIndexArray(layout);
	const indices = new Uint32Array(cellCount * SURFACE_VERTICES_PER_CELL);
	let offset = 0;
	for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
		const pointIndex = cellToPoint[cellIndex] ?? 0;
		for (let vertex = 0; vertex < SURFACE_VERTICES_PER_CELL; vertex += 1) {
			indices[offset++] = pointIndex;
		}
	}

	return indices;
}

export function getComputeBufferSurfaceVertexCount(layout: UtciGridLayout): number {
	return (layout.width + 1) * (layout.height + 1);
}

function getCellToPointIndex(layout: UtciGridLayout, cellCount: number): Int32Array {
	if (
		layout.cellToPointIndex?.length === cellCount &&
		!hasAmbiguousCellEntries(layout.cellToPointIndex)
	) {
		return layout.cellToPointIndex;
	}

	const cellToPoint = new Int32Array(cellCount);
	cellToPoint.fill(-1);

	for (let pointIndex = 0; pointIndex < layout.numPositions; pointIndex += 1) {
		const row = layout.indexToRow[pointIndex];
		const column = layout.indexToColumn[pointIndex];
		if (row >= layout.height || column >= layout.width) {
			continue;
		}

		cellToPoint[row * layout.width + column] = pointIndex;
	}

	return cellToPoint;
}

function hasAmbiguousCellEntries(cellToPointIndex: Int32Array): boolean {
	for (let index = 0; index < cellToPointIndex.length; index += 1) {
		if (cellToPointIndex[index] < -1) {
			return true;
		}
	}

	return false;
}

function int32ArraysEqual(left: Int32Array, right: Int32Array): boolean {
	if (left.length !== right.length) {
		return false;
	}
	for (let index = 0; index < left.length; index += 1) {
		if (left[index] !== right[index]) {
			return false;
		}
	}
	return true;
}

function uint32ArraysEqual(left: Uint32Array, right: Uint32Array): boolean {
	if (left.length !== right.length) {
		return false;
	}
	for (let index = 0; index < left.length; index += 1) {
		if (left[index] !== right[index]) {
			return false;
		}
	}
	return true;
}
