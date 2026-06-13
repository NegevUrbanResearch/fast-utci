import {
	createLargeBufferRequiredLimits,
	type WebgpuLargeBufferDeviceLimits,
	type WebgpuLargeBufferRequiredLimits
} from '$lib/compute/gpu/webgpuDeviceLimits';
import type { ActiveMaskUtciSurfaceShape } from '$lib/services/activeMaskUtciSurfaceGeometry';
const FLOAT32_BYTES = 4;
const UINT32_BYTES = 4;
const INT32_BYTES = 4;
const SURFACE_VERTICES_PER_CELL = 6;
const POSITION_COMPONENTS = 3;
const ACTIVE_INDEXED_VERTICES_PER_CELL = 4;
export const DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS = {
	jsLargestTypedArrayBytes: 256 * 1024 * 1024,
	jsTotalTypedArrayBytes: 1024 * 1024 * 1024,
	comfortableLimitRatio: 0.75
} as const;
export type ActiveMaskUtciSurfaceStrategy =
	| 'dense-indexed-rect'
	| 'active-non-indexed-quads'
	| 'active-indexed-quads'
	| 'active-instanced-quads'
	| 'active-tiled-indexed-quads'
	| 'active-tiled-instanced-quads';
export type ActiveMaskUtciSurfaceBudgetLimits = WebgpuLargeBufferRequiredLimits & {
	jsLargestTypedArrayBytes: number;
	jsTotalTypedArrayBytes: number;
	comfortableLimitRatio: number;
	source: {
		requested: WebgpuLargeBufferRequiredLimits;
		device?: WebgpuLargeBufferDeviceLimits;
	};
};
export type ActiveMaskUtciSurfaceStorageBufferEstimate = {
	name: string;
	bytes: number;
	fitsMaxBufferSize: boolean;
	fitsMaxStorageBufferBindingSize: boolean;
};
export type ActiveMaskUtciSurfaceBudgetEstimate = {
	strategy: ActiveMaskUtciSurfaceStrategy;
	totalJsTypedArrayBytes: number;
	largestSingleJsTypedArrayBytes: number;
	vertexBufferBytes: number;
	indexBufferBytes: number;
	storageBufferBytes: number;
	storageBuffers: ActiveMaskUtciSurfaceStorageBufferEstimate[];
	maxActiveCellsPerTile?: number;
	tileCount?: number;
	fits: {
		jsLargestTypedArray: boolean;
		jsTotalTypedArray: boolean;
		maxBufferSize: boolean;
		maxStorageBufferBindingSize: boolean;
		comfortableJsLargestTypedArray: boolean;
		comfortableJsTotalTypedArray: boolean;
	};
};
export type ActiveMaskUtciSurfaceBudgetLimitInputs = {
	requestedLimits?: WebgpuLargeBufferRequiredLimits;
	deviceLimits?: WebgpuLargeBufferDeviceLimits;
	jsLargestTypedArrayBytes?: number;
	jsTotalTypedArrayBytes?: number;
	comfortableLimitRatio?: number;
};
function resolveBudgetLimits(
	inputs: ActiveMaskUtciSurfaceBudgetLimitInputs = {}
): ActiveMaskUtciSurfaceBudgetLimits {
	const requested = inputs.requestedLimits ?? createLargeBufferRequiredLimits();
	return {
		maxStorageBufferBindingSize:
			inputs.deviceLimits?.maxStorageBufferBindingSize ??
			requested.maxStorageBufferBindingSize,
		maxBufferSize: inputs.deviceLimits?.maxBufferSize ?? requested.maxBufferSize,
		jsLargestTypedArrayBytes:
			inputs.jsLargestTypedArrayBytes ??
			DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS.jsLargestTypedArrayBytes,
		jsTotalTypedArrayBytes:
			inputs.jsTotalTypedArrayBytes ??
			DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS.jsTotalTypedArrayBytes,
		comfortableLimitRatio:
			inputs.comfortableLimitRatio ??
			DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS.comfortableLimitRatio,
		source: {
			requested: { ...requested },
			device: inputs.deviceLimits ? { ...inputs.deviceLimits } : undefined
		}
	};
}
function sum(values: number[]): number {
	return values.reduce((total, value) => total + value, 0);
}
function estimateStorageBuffers(
	buffers: { name: string; bytes: number }[],
	limits: ActiveMaskUtciSurfaceBudgetLimits
): ActiveMaskUtciSurfaceStorageBufferEstimate[] {
	return buffers.map((buffer) => ({
		...buffer,
		fitsMaxBufferSize: buffer.bytes <= limits.maxBufferSize,
		fitsMaxStorageBufferBindingSize: buffer.bytes <= limits.maxStorageBufferBindingSize
	}));
}
function buildEstimate(params: {
	strategy: ActiveMaskUtciSurfaceStrategy;
	jsArrays: number[];
	vertexBufferBytes: number;
	indexBufferBytes: number;
	storageBuffers: { name: string; bytes: number }[];
	limits: ActiveMaskUtciSurfaceBudgetLimits;
	maxActiveCellsPerTile?: number;
	tileCount?: number;
}): ActiveMaskUtciSurfaceBudgetEstimate {
	const storageBuffers = estimateStorageBuffers(params.storageBuffers, params.limits);
	const totalJsTypedArrayBytes = sum(params.jsArrays);
	const largestSingleJsTypedArrayBytes = Math.max(0, ...params.jsArrays);
	const bufferBytes = [
		params.vertexBufferBytes,
		params.indexBufferBytes,
		...storageBuffers.map((buffer) => buffer.bytes)
	];
	return {
		strategy: params.strategy,
		totalJsTypedArrayBytes,
		largestSingleJsTypedArrayBytes,
		vertexBufferBytes: params.vertexBufferBytes,
		indexBufferBytes: params.indexBufferBytes,
		storageBufferBytes: sum(storageBuffers.map((buffer) => buffer.bytes)),
		storageBuffers,
		maxActiveCellsPerTile: params.maxActiveCellsPerTile,
		tileCount: params.tileCount,
		fits: {
			jsLargestTypedArray:
				largestSingleJsTypedArrayBytes <= params.limits.jsLargestTypedArrayBytes,
			jsTotalTypedArray:
				totalJsTypedArrayBytes <= params.limits.jsTotalTypedArrayBytes,
			maxBufferSize: bufferBytes.every(
				(bufferSize) => bufferSize <= params.limits.maxBufferSize
			),
			maxStorageBufferBindingSize: storageBuffers.every(
				(buffer) => buffer.fitsMaxStorageBufferBindingSize
			),
			comfortableJsLargestTypedArray:
				largestSingleJsTypedArrayBytes <=
				params.limits.jsLargestTypedArrayBytes *
					params.limits.comfortableLimitRatio,
			comfortableJsTotalTypedArray:
				totalJsTypedArrayBytes <=
				params.limits.jsTotalTypedArrayBytes *
					params.limits.comfortableLimitRatio
		}
	};
}
function ceilDiv(numerator: number, denominator: number): number {
	return Math.ceil(numerator / denominator);
}
function maxCellsForTiledArrays(
	limits: ActiveMaskUtciSurfaceBudgetLimits,
	arrays: {
		bytesPerCell: number;
		binding: 'vertex-or-index' | 'storage';
	}[]
): number {
	const candidates = arrays.flatMap((array) => {
		const cappedByLargestArray = Math.floor(limits.jsLargestTypedArrayBytes / array.bytesPerCell);
		const cappedByBufferSize = Math.floor(limits.maxBufferSize / array.bytesPerCell);
		if (array.binding === 'storage') {
			return [
				cappedByLargestArray,
				cappedByBufferSize,
				Math.floor(limits.maxStorageBufferBindingSize / array.bytesPerCell)
			];
		}
		return [cappedByLargestArray, cappedByBufferSize];
	});
	return Math.max(1, Math.min(...candidates));
}
export function estimateActiveMaskUtciSurfaceStrategies(
	shape: ActiveMaskUtciSurfaceShape,
	limitInputs: ActiveMaskUtciSurfaceBudgetLimitInputs = {}
): { limits: ActiveMaskUtciSurfaceBudgetLimits; estimates: ActiveMaskUtciSurfaceBudgetEstimate[] } {
	const limits = resolveBudgetLimits(limitInputs);
	const activeCells = shape.activePointCount;
	const canonicalCells = shape.canonicalCellCount;
	const denseVertexCount = (shape.canonicalWidth + 1) * (shape.canonicalHeight + 1);
	const activeCellIndexBytes = activeCells * UINT32_BYTES;
	const activeUtciStorageBytes = activeCells * FLOAT32_BYTES;
	const densePositionBytes = denseVertexCount * POSITION_COMPONENTS * FLOAT32_BYTES;
	const denseIndexBytes = canonicalCells * SURFACE_VERTICES_PER_CELL * UINT32_BYTES;
	const denseCellMappingBytes = canonicalCells * UINT32_BYTES;
	const denseCellMappingLayoutBytes = canonicalCells * INT32_BYTES;
	const denseColorBufferBytes = canonicalCells * 4;
	const activeRowsBytes = activeCells * UINT32_BYTES;
	const activeColumnsBytes = activeCells * UINT32_BYTES;
	const activeTexelBytes = activeCells * UINT32_BYTES;
	const activeNonIndexedPositionBytes =
		activeCells * SURFACE_VERTICES_PER_CELL * POSITION_COMPONENTS * FLOAT32_BYTES;
	const activeNonIndexedVertexToPointBytes =
		activeCells * SURFACE_VERTICES_PER_CELL * UINT32_BYTES;
	const activeIndexedPositionBytes =
		activeCells * ACTIVE_INDEXED_VERTICES_PER_CELL * POSITION_COMPONENTS * FLOAT32_BYTES;
	const activeIndexedIndexBytes =
		activeCells * SURFACE_VERTICES_PER_CELL * UINT32_BYTES;
	const activeIndexedVertexToPointBytes =
		activeCells * ACTIVE_INDEXED_VERTICES_PER_CELL * UINT32_BYTES;
	const instancedVertexBufferBytes =
		ACTIVE_INDEXED_VERTICES_PER_CELL * POSITION_COMPONENTS * FLOAT32_BYTES;
	const instancedIndexBufferBytes = SURFACE_VERTICES_PER_CELL * UINT32_BYTES;
	const indexedCellsPerTile = maxCellsForTiledArrays(limits, [
		{
			bytesPerCell: ACTIVE_INDEXED_VERTICES_PER_CELL * POSITION_COMPONENTS * FLOAT32_BYTES,
			binding: 'vertex-or-index'
		},
		{ bytesPerCell: SURFACE_VERTICES_PER_CELL * UINT32_BYTES, binding: 'vertex-or-index' },
		{ bytesPerCell: ACTIVE_INDEXED_VERTICES_PER_CELL * UINT32_BYTES, binding: 'storage' },
		{ bytesPerCell: FLOAT32_BYTES, binding: 'storage' },
		{ bytesPerCell: UINT32_BYTES, binding: 'vertex-or-index' }
	]);
	const instancedCellsPerTile = maxCellsForTiledArrays(limits, [
		{ bytesPerCell: UINT32_BYTES, binding: 'storage' },
		{ bytesPerCell: FLOAT32_BYTES, binding: 'storage' }
	]);
	const indexedTileActiveCells = Math.min(activeCells, indexedCellsPerTile);
	const instancedTileActiveCells = Math.min(activeCells, instancedCellsPerTile);
	return {
		limits,
		estimates: [
			buildEstimate({
				strategy: 'dense-indexed-rect',
				jsArrays: [
					densePositionBytes,
					denseIndexBytes,
					activeRowsBytes,
					activeColumnsBytes,
					activeTexelBytes,
					denseCellMappingLayoutBytes,
					denseColorBufferBytes,
					activeUtciStorageBytes,
					denseCellMappingBytes
				],
				vertexBufferBytes: densePositionBytes,
				indexBufferBytes: denseIndexBytes,
				storageBuffers: [
					{ name: 'selected-hour-utci', bytes: activeUtciStorageBytes },
					{ name: 'cell-to-point-index', bytes: denseCellMappingBytes }
				],
				limits
			}),
			buildEstimate({
				strategy: 'active-non-indexed-quads',
				jsArrays: [
					activeNonIndexedPositionBytes,
					activeNonIndexedVertexToPointBytes,
					activeUtciStorageBytes,
					activeCellIndexBytes
				],
				vertexBufferBytes: activeNonIndexedPositionBytes,
				indexBufferBytes: 0,
				storageBuffers: [
					{ name: 'selected-hour-utci', bytes: activeUtciStorageBytes },
					{ name: 'vertex-to-point-index', bytes: activeNonIndexedVertexToPointBytes }
				],
				limits
			}),
			buildEstimate({
				strategy: 'active-indexed-quads',
				jsArrays: [
					activeIndexedPositionBytes,
					activeIndexedIndexBytes,
					activeIndexedVertexToPointBytes,
					activeUtciStorageBytes,
					activeCellIndexBytes
				],
				vertexBufferBytes: activeIndexedPositionBytes,
				indexBufferBytes: activeIndexedIndexBytes,
				storageBuffers: [
					{ name: 'selected-hour-utci', bytes: activeUtciStorageBytes },
					{ name: 'vertex-to-point-index', bytes: activeIndexedVertexToPointBytes }
				],
				limits
			}),
			buildEstimate({
				strategy: 'active-instanced-quads',
				jsArrays: [
					instancedVertexBufferBytes,
					instancedIndexBufferBytes,
					activeCellIndexBytes,
					activeUtciStorageBytes
				],
				vertexBufferBytes: instancedVertexBufferBytes,
				indexBufferBytes: instancedIndexBufferBytes,
				storageBuffers: [
					{ name: 'selected-hour-utci', bytes: activeUtciStorageBytes },
					{ name: 'active-canonical-indices', bytes: activeCellIndexBytes }
				],
				limits
			}),
			buildEstimate({
				strategy: 'active-tiled-indexed-quads',
				jsArrays: [
					indexedTileActiveCells *
						ACTIVE_INDEXED_VERTICES_PER_CELL *
						POSITION_COMPONENTS *
						FLOAT32_BYTES,
					indexedTileActiveCells * SURFACE_VERTICES_PER_CELL * UINT32_BYTES,
					indexedTileActiveCells * ACTIVE_INDEXED_VERTICES_PER_CELL * UINT32_BYTES,
					indexedTileActiveCells * FLOAT32_BYTES,
					indexedTileActiveCells * UINT32_BYTES
				],
				vertexBufferBytes:
					indexedTileActiveCells *
					ACTIVE_INDEXED_VERTICES_PER_CELL *
					POSITION_COMPONENTS *
					FLOAT32_BYTES,
				indexBufferBytes:
					indexedTileActiveCells * SURFACE_VERTICES_PER_CELL * UINT32_BYTES,
				storageBuffers: [
					{ name: 'selected-hour-utci-tile', bytes: indexedTileActiveCells * FLOAT32_BYTES },
					{
						name: 'vertex-to-point-index-tile',
						bytes:
							indexedTileActiveCells *
							ACTIVE_INDEXED_VERTICES_PER_CELL *
							UINT32_BYTES
					}
				],
				limits,
				maxActiveCellsPerTile: indexedCellsPerTile,
				tileCount: ceilDiv(activeCells, indexedCellsPerTile)
			}),
			buildEstimate({
				strategy: 'active-tiled-instanced-quads',
				jsArrays: [
					instancedVertexBufferBytes,
					instancedIndexBufferBytes,
					instancedTileActiveCells * UINT32_BYTES,
					instancedTileActiveCells * FLOAT32_BYTES
				],
				vertexBufferBytes: instancedVertexBufferBytes,
				indexBufferBytes: instancedIndexBufferBytes,
				storageBuffers: [
					{ name: 'selected-hour-utci-tile', bytes: instancedTileActiveCells * FLOAT32_BYTES },
					{ name: 'active-canonical-indices-tile', bytes: instancedTileActiveCells * UINT32_BYTES }
				],
				limits,
				maxActiveCellsPerTile: instancedCellsPerTile,
				tileCount: ceilDiv(activeCells, instancedCellsPerTile)
			})
		]
	};
}
