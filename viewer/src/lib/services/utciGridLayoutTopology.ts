import type * as THREE from 'three';
import { resolveCanonicalGridAxes } from '$lib/compute/core/canonicalGridAxes';
import type {
	AnalysisActiveMask,
	AnalysisCoordinateSystem,
	AnalysisMetadata,
	AnalysisRectangularBounds
} from '$lib/types/analysis';
import type {
	SelectedHourRenderLayoutConstructionMode,
	SelectedHourRenderLayoutNormalizationSignature
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

export interface UtciGridLayoutBase {
	width: number;
	height: number;
	gridSize: number;
	coordinateSystem: AnalysisCoordinateSystem;
	numPositions: number;
	minX: number;
	minZ: number;
	minY: number;
	maxY: number;
	centerX: number;
	centerZ: number;
	baseY: number;
	renderCellCount: number;
	canonicalCellCount: number;
	positionsIdentityId?: number;
	constructionMode?: SelectedHourRenderLayoutConstructionMode;
	normalizationSignature?: SelectedHourRenderLayoutNormalizationSignature;
	texture?: THREE.DataTexture;
}

export interface DenseUtciGridLayout extends UtciGridLayoutBase {
	renderTopology: 'dense-grid';
	indexToRow: Uint32Array;
	indexToColumn: Uint32Array;
	cellToPointIndex?: Int32Array;
	indexToTexel: Uint32Array;
	colorBuffer: Uint8Array;
}

export interface ActiveCellsUtciGridLayout extends UtciGridLayoutBase {
	renderTopology: 'active-cells';
	activeCanonicalIndices: Uint32Array;
	activeMaskSignature: string;
	indexToRow?: never;
	indexToColumn?: never;
	cellToPointIndex?: never;
	indexToTexel?: never;
	colorBuffer?: never;
}

export type UtciGridLayout = DenseUtciGridLayout | ActiveCellsUtciGridLayout;

export interface ActiveMaskViewerLayout {
	activeMask: AnalysisActiveMask;
	minX: number;
	maxX: number;
	minZ: number;
	maxZ: number;
	minY: number;
	maxY: number;
	width: number;
	height: number;
}

export function getRenderableActiveMask(
	metadata: AnalysisMetadata,
	numPositions: number
): AnalysisActiveMask | null {
	const activeMask = metadata.activeMask;
	if (
		(activeMask?.source !== 'base' && activeMask?.source !== 'base+road') ||
		activeMask.activeCanonicalIndices.length !== numPositions ||
		activeMask.activePointCount !== numPositions
	) {
		return null;
	}
	return activeMask;
}

export function resolveActiveMaskViewerLayout(params: {
	bounds: AnalysisRectangularBounds | undefined;
	activeMask: AnalysisActiveMask | null;
	gridSize: number;
	coordinateSystem: AnalysisCoordinateSystem;
	normalizationOffset: { x: number; y: number; z: number };
}): ActiveMaskViewerLayout | null {
	const { bounds, activeMask, gridSize, coordinateSystem, normalizationOffset } =
		params;
	if (!bounds || !activeMask) {
		return null;
	}

	const axes = resolveCanonicalGridAxes({
		bounds,
		gridSize,
		coordinateSystem
	});
	const minX = axes.minX + normalizationOffset.x;
	const maxX = axes.maxX + normalizationOffset.x;
	const minZ = axes.minZ + normalizationOffset.z;
	const maxZ = axes.maxZ + normalizationOffset.z;
	const minY = (bounds.z ?? 0) + normalizationOffset.y;
	const maxY = minY;

	return {
		activeMask,
		minX,
		maxX,
		minZ,
		maxZ,
		minY,
		maxY,
		width: axes.width,
		height: axes.height
	};
}

export function getActiveMaskGridCell(params: {
	canonicalIndex: number;
	width: number;
	height: number;
	coordinateSystem: AnalysisCoordinateSystem;
}): { row: number; col: number } {
	const canonicalRow = params.canonicalIndex % params.height;
	return {
		col: Math.min(params.width - 1, Math.floor(params.canonicalIndex / params.height)),
		row: Math.min(
			params.height - 1,
			params.coordinateSystem === 'xy_ground'
				? params.height - 1 - canonicalRow
				: canonicalRow
		)
	};
}

export function getActiveMaskPointGridCell(params: {
	layout: ActiveCellsUtciGridLayout;
	pointIndex: number;
}): { row: number; col: number } {
	const canonicalIndex = params.layout.activeCanonicalIndices[params.pointIndex];
	if (canonicalIndex === undefined) {
		throw new RangeError(`Active UTCI point index ${params.pointIndex} is out of range.`);
	}
	return getActiveMaskGridCell({
		canonicalIndex,
		width: params.layout.width,
		height: params.layout.height,
		coordinateSystem: params.layout.coordinateSystem
	});
}

export function buildActiveMaskUtciGridLayout(params: {
	activeMaskLayout: ActiveMaskViewerLayout;
	gridSize: number;
	coordinateSystem: AnalysisCoordinateSystem;
	numPositions: number;
	positionsIdentityId: number;
	normalizationSignature: SelectedHourRenderLayoutNormalizationSignature;
	visualLayerOffset: number;
}): ActiveCellsUtciGridLayout {
	const {
		activeMaskLayout,
		gridSize,
		coordinateSystem,
		numPositions,
		positionsIdentityId,
		normalizationSignature,
		visualLayerOffset
	} = params;

	return {
		renderTopology: 'active-cells',
		width: activeMaskLayout.width,
		height: activeMaskLayout.height,
		gridSize,
		coordinateSystem,
		numPositions,
		minX: activeMaskLayout.minX,
		minZ: activeMaskLayout.minZ,
		minY: activeMaskLayout.minY,
		maxY: activeMaskLayout.maxY,
		centerX: activeMaskLayout.minX + ((activeMaskLayout.width - 1) * gridSize) / 2,
		centerZ: activeMaskLayout.minZ + ((activeMaskLayout.height - 1) * gridSize) / 2,
		baseY: activeMaskLayout.minY + visualLayerOffset,
		renderCellCount: numPositions,
		canonicalCellCount: activeMaskLayout.width * activeMaskLayout.height,
		activeCanonicalIndices: activeMaskLayout.activeMask.activeCanonicalIndices,
		activeMaskSignature:
			activeMaskLayout.activeMask.signature ??
			activeMaskLayout.activeMask.activeMaskChecksum,
		positionsIdentityId,
		constructionMode: 'metadata-bounds-fallback',
		normalizationSignature
	};
}
