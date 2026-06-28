/**
 * Tooltip Service
 *
 * Handles raycasting and value lookup for metric tooltips on hover
 */

import * as THREE from 'three';
import type { Analysis } from '$lib/types/analysis';
import type { MetricType } from '$lib/types/viewer';
import { getUTCIForHour, getShadingIndex } from '$lib/services/dataLoader';
import { createRaycaster } from '$lib/utils/raycaster';
import { getNormalizedMousePosition } from '$lib/utils/mouse';
import type { UtciGridLayout } from './pointCloudService';
import {
	getCachedActiveMaskUtciSurfaceLookup,
	resolveActiveMaskUtciSurfaceCellLookup
} from './activeMaskUtciSurfaceLookup';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { sharedMetricPointReadbackCache } from '$lib/compute/gpu/metricPointReadback';

export interface TooltipData {
	value: number;
	position: { x: number; y: number; z: number };
	positionIndex: number;
}

export type TooltipProbeData = Omit<TooltipData, 'value'>;

export type TooltipMetricPointValueReader = {
	monthIndex: number;
	requestId?: number;
	ownerId?: string;
	readbackByteLength?: number;
	readMetricPointValue: (key: {
		metricType: MetricType;
		monthIndex: number;
		positionIndex: number;
		requestId?: number;
		ownerId?: string;
	}) => Promise<number>;
};

export type TooltipResolutionPath = 'none' | 'plane-cell' | 'mesh-raycast';

export interface TooltipMetricPointReadbackMeasurement {
	metricType: MetricType;
	monthIndex: number;
	positionIndex: number;
	requestId?: number;
	ownerId?: string;
	cacheHit: boolean;
	byteLength: number;
	latencyMs: number;
	success: boolean;
}

export interface TooltipInteractionMeasurement {
	hit: boolean;
	raycastMs: number;
	nearestPointMs: number;
	totalMs: number;
	overBudget: boolean;
	resolutionPath?: TooltipResolutionPath;
	directCellHit?: boolean;
	nearestScanUsed?: boolean;
	directCellMissCount?: number;
}

export interface TooltipInteractionDiagnostics {
	enabled: boolean;
	disabledByQuery: boolean;
	slowThresholdMs: number;
	hoverAttemptCount: number;
	suppressedHoverCount: number;
	throttledHoverCount: number;
	sampleCount: number;
	hitCount: number;
	missCount: number;
	overBudgetCount: number;
	lastOutcome: 'hit' | 'miss' | null;
	lastRaycastMs: number | null;
	maxRaycastMs: number;
	lastNearestPointMs: number | null;
	maxNearestPointMs: number;
	lastTotalMs: number | null;
	maxTotalMs: number;
	lastResolutionPath: TooltipResolutionPath | null;
	planeCellPathCount: number;
	meshRaycastPathCount: number;
	directCellHitCount: number;
	directCellMissCount: number;
	nearestScanFallbackCount: number;
	metricPointReadbackCount: number;
	metricPointReadbackBytes: number;
	metricPointReadbackLastBytes: number | null;
	metricPointReadbackCacheEntries: number;
	metricPointReadbackCacheHitCount: number;
	metricPointReadbackCacheMissCount: number;
	metricPointReadbackLastLatencyMs: number | null;
	metricPointReadbackMaxLatencyMs: number;
}

type TooltipWorldTransformContext = {
	coordinateSystem: 'xy_ground' | 'xz_ground';
	normalizationOffset: THREE.Vector3;
};

type TooltipSurfaceBounds = {
	planeWidth: number;
	planeHeight: number;
	halfWidth: number;
	halfHeight: number;
	epsilon: number;
};

type TooltipHitResolution = {
	position: { x: number; y: number; z: number };
	positionIndex: number;
	emitMeasurement: (hit: boolean) => void;
};

export const TOOLTIP_SLOW_BUDGET_MS = 8;
const rebuiltCellToPointIndexCache = new WeakMap<UtciGridLayout, Int32Array | null>();

export function createEmptyTooltipInteractionDiagnostics(
	disabledByQuery: boolean
): TooltipInteractionDiagnostics {
	return {
		enabled: !disabledByQuery,
		disabledByQuery,
		slowThresholdMs: TOOLTIP_SLOW_BUDGET_MS,
		hoverAttemptCount: 0,
		suppressedHoverCount: 0,
		throttledHoverCount: 0,
		sampleCount: 0,
		hitCount: 0,
		missCount: 0,
		overBudgetCount: 0,
		lastOutcome: null,
		lastRaycastMs: null,
		maxRaycastMs: 0,
		lastNearestPointMs: null,
		maxNearestPointMs: 0,
		lastTotalMs: null,
		maxTotalMs: 0,
		lastResolutionPath: null,
		planeCellPathCount: 0,
		meshRaycastPathCount: 0,
		directCellHitCount: 0,
		directCellMissCount: 0,
		nearestScanFallbackCount: 0,
		metricPointReadbackCount: 0,
		metricPointReadbackBytes: 0,
		metricPointReadbackLastBytes: null,
		metricPointReadbackCacheEntries: 0,
		metricPointReadbackCacheHitCount: 0,
		metricPointReadbackCacheMissCount: 0,
		metricPointReadbackLastLatencyMs: null,
		metricPointReadbackMaxLatencyMs: 0
	};
}

export function recordTooltipInteractionMeasurement(
	diagnostics: TooltipInteractionDiagnostics,
	measurement: TooltipInteractionMeasurement
): TooltipInteractionDiagnostics {
	const resolutionPath = measurement.resolutionPath ?? 'none';
	return {
		...diagnostics,
		enabled: !diagnostics.disabledByQuery,
		slowThresholdMs: TOOLTIP_SLOW_BUDGET_MS,
		sampleCount: diagnostics.sampleCount + 1,
		hitCount: diagnostics.hitCount + (measurement.hit ? 1 : 0),
		missCount: diagnostics.missCount + (measurement.hit ? 0 : 1),
		overBudgetCount: diagnostics.overBudgetCount + (measurement.overBudget ? 1 : 0),
		lastOutcome: measurement.hit ? 'hit' : 'miss',
		lastRaycastMs: measurement.raycastMs,
		maxRaycastMs: Math.max(diagnostics.maxRaycastMs, measurement.raycastMs),
		lastNearestPointMs: measurement.nearestPointMs,
		maxNearestPointMs: Math.max(diagnostics.maxNearestPointMs, measurement.nearestPointMs),
		lastTotalMs: measurement.totalMs,
		maxTotalMs: Math.max(diagnostics.maxTotalMs, measurement.totalMs),
		lastResolutionPath: resolutionPath,
		planeCellPathCount:
			diagnostics.planeCellPathCount + (resolutionPath === 'plane-cell' ? 1 : 0),
		meshRaycastPathCount:
			diagnostics.meshRaycastPathCount + (resolutionPath === 'mesh-raycast' ? 1 : 0),
		directCellHitCount:
			diagnostics.directCellHitCount + (measurement.directCellHit === true ? 1 : 0),
		directCellMissCount:
			diagnostics.directCellMissCount + (measurement.directCellMissCount ?? 0),
		nearestScanFallbackCount:
			diagnostics.nearestScanFallbackCount + (measurement.nearestScanUsed === true ? 1 : 0)
	};
}

function getTooltipSurfaceBounds(layout: UtciGridLayout): TooltipSurfaceBounds | null {
	if (
		!Number.isFinite(layout.width) ||
		!Number.isFinite(layout.height) ||
		!Number.isFinite(layout.gridSize) ||
		layout.width <= 0 ||
		layout.height <= 0 ||
		layout.gridSize <= 0
	) {
		return null;
	}

	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;
	if (!Number.isFinite(planeWidth) || !Number.isFinite(planeHeight) || planeWidth <= 0 || planeHeight <= 0) {
		return null;
	}

	const epsilon = Math.max(layout.gridSize * 1e-6, 1e-6);
	return {
		planeWidth,
		planeHeight,
		halfWidth: planeWidth / 2,
		halfHeight: planeHeight / 2,
		epsilon
	};
}

function getSurfaceLocalPoint(
	worldPoint: THREE.Vector3,
	object: THREE.Object3D
): THREE.Vector3 | null {
	const localPoint = object.worldToLocal(worldPoint.clone());
	if (
		!Number.isFinite(localPoint.x) ||
		!Number.isFinite(localPoint.y) ||
		!Number.isFinite(localPoint.z)
	) {
		return null;
	}

	return localPoint;
}

function getSurfaceCellCoordinates(
	localPoint: THREE.Vector3,
	layout: UtciGridLayout
): { column: number; row: number } | null {
	const bounds = getTooltipSurfaceBounds(layout);
	if (!bounds) {
		return null;
	}

	if (
		localPoint.x < -bounds.halfWidth - bounds.epsilon ||
		localPoint.x > bounds.halfWidth + bounds.epsilon ||
		localPoint.z < -bounds.halfHeight - bounds.epsilon ||
		localPoint.z > bounds.halfHeight + bounds.epsilon
	) {
		return null;
	}

	const normalizedX = THREE.MathUtils.clamp(
		localPoint.x + bounds.halfWidth,
		0,
		bounds.planeWidth - bounds.epsilon
	);
	const normalizedZ = THREE.MathUtils.clamp(
		localPoint.z + bounds.halfHeight,
		0,
		bounds.planeHeight - bounds.epsilon
	);
	const column = Math.floor(normalizedX / layout.gridSize);
	const row = Math.floor(normalizedZ / layout.gridSize);
	if (column < 0 || column >= layout.width || row < 0 || row >= layout.height) {
		return null;
	}

	return { column, row };
}

function getPositionIndexFromSurfaceCell(
	intersection: THREE.Intersection,
	layout: UtciGridLayout,
	analysis: Analysis,
	options?: { enforceDistanceThreshold?: boolean }
): { positionIndex: number | null; mappingMs: number; inactiveCell: boolean } {
	const mappingStartedAt = performance.now();
	const worldPoint = intersection.point;
	if (!worldPoint) return { positionIndex: null, mappingMs: 0, inactiveCell: false };

	const object = intersection.object;
	if (!object) return { positionIndex: null, mappingMs: 0, inactiveCell: false };

	const expectedCellCount = layout.width * layout.height;
	const localPoint = getSurfaceLocalPoint(worldPoint, object);
	if (!localPoint) {
		return { positionIndex: null, mappingMs: 0, inactiveCell: false };
	}

	const cellCoordinates = getSurfaceCellCoordinates(localPoint, layout);
	if (!cellCoordinates) {
		return { positionIndex: null, mappingMs: 0, inactiveCell: false };
	}

	const { column, row } = cellCoordinates;
	const cellLookup = getTooltipCellPointIndex(
		layout,
		analysis.data.numPositions,
		row,
		column,
		expectedCellCount
	);
	const pointIndex = cellLookup.positionIndex;
	if (pointIndex === -1) {
		return {
			positionIndex: null,
			mappingMs: performance.now() - mappingStartedAt,
			inactiveCell: cellLookup.inactiveCell
		};
	}
	if (pointIndex < 0 || pointIndex >= analysis.data.numPositions) {
		return {
			positionIndex: null,
			mappingMs: performance.now() - mappingStartedAt,
			inactiveCell: false
		};
	}

	const transformed = new THREE.Vector3();
	transformAnalysisPositionToWorld(
		analysis.data.positions,
		pointIndex,
		createTooltipWorldTransformContext(analysis),
		transformed
	);
	if (
		!Number.isFinite(transformed.x) ||
		!Number.isFinite(transformed.y) ||
		!Number.isFinite(transformed.z)
	) {
		return {
			positionIndex: null,
			mappingMs: performance.now() - mappingStartedAt,
			inactiveCell: false
		};
	}
	if (options?.enforceDistanceThreshold === false) {
		return {
			positionIndex: pointIndex,
			mappingMs: performance.now() - mappingStartedAt,
			inactiveCell: false
		};
	}

	const dx = worldPoint.x - transformed.x;
	const dz = worldPoint.z - transformed.z;
	const distance = Math.sqrt(dx * dx + dz * dz);
	if (!Number.isFinite(distance) || distance > getTooltipDistanceThreshold(analysis)) {
		return {
			positionIndex: null,
			mappingMs: performance.now() - mappingStartedAt,
			inactiveCell: false
		};
	}

	return {
		positionIndex: pointIndex,
		mappingMs: performance.now() - mappingStartedAt,
		inactiveCell: false
	};
}

function getTooltipCellPointIndex(
	layout: UtciGridLayout,
	numPositions: number,
	row: number,
	column: number,
	expectedCellCount: number
): { positionIndex: number; inactiveCell: boolean } {
	if (layout.renderTopology === 'active-cells') {
		const lookupResult = resolveActiveMaskUtciSurfaceCellLookup(
			getCachedActiveMaskUtciSurfaceLookup(layout),
			{ row, column }
		);
		return {
			positionIndex: lookupResult.positionIndex ?? -1,
			inactiveCell: lookupResult.inactiveCell
		};
	}

	const cellIndex = row * layout.width + column;
	const mappedPointIndex = layout.cellToPointIndex?.[cellIndex];
	if (
		layout.cellToPointIndex &&
		layout.cellToPointIndex.length === expectedCellCount &&
		mappedPointIndex !== undefined &&
		mappedPointIndex >= -1 &&
		mappedPointIndex < numPositions
	) {
		return {
			positionIndex: mappedPointIndex,
			inactiveCell: false
		};
	}

	return {
		positionIndex:
			getRebuiltCellToPointIndex(layout, numPositions, expectedCellCount)?.[cellIndex] ?? -1,
		inactiveCell: false
	};
}

function getRebuiltCellToPointIndex(
	layout: UtciGridLayout,
	numPositions: number,
	expectedCellCount: number
): Int32Array | null {
	if (layout.renderTopology !== 'dense-grid') {
		return null;
	}

	const cached = rebuiltCellToPointIndexCache.get(layout);
	if (cached !== undefined) {
		return cached;
	}

	if (
		expectedCellCount <= 0
	) {
		rebuiltCellToPointIndexCache.set(layout, null);
		return null;
	}
	if (
		layout.indexToRow.length < numPositions ||
		layout.indexToColumn.length < numPositions
	) {
		rebuiltCellToPointIndexCache.set(layout, null);
		return null;
	}

	const rebuilt = new Int32Array(expectedCellCount);
	rebuilt.fill(-1);

	for (let pointIndex = 0; pointIndex < numPositions; pointIndex += 1) {
		const row = layout.indexToRow[pointIndex];
		const column = layout.indexToColumn[pointIndex];
		if (row >= layout.height || column >= layout.width) {
			continue;
		}

		rebuilt[row * layout.width + column] = pointIndex;
	}

	rebuiltCellToPointIndexCache.set(layout, rebuilt);
	return rebuilt;
}

/**
 * Find the position index from a mesh intersection
 * Uses the intersection point's world coordinates to find the closest position
 * This accounts for coordinate transformations and mesh positioning
 */
function getNearestPositionIndexFromIntersection(
	intersection: THREE.Intersection,
	layout: UtciGridLayout,
	analysis: Analysis
): number | null {
	const worldPoint = intersection.point;
	if (!worldPoint) return null;

	const positions = analysis.data.positions;
	const numPositions = analysis.data.numPositions;
	const worldTransform = createTooltipWorldTransformContext(analysis);

	let closestIndex = -1;
	let minDistance = Infinity;
	const transformed = new THREE.Vector3();

	for (let i = 0; i < numPositions; i++) {
		transformAnalysisPositionToWorld(positions, i, worldTransform, transformed);

		const dx = worldPoint.x - transformed.x;
		const dz = worldPoint.z - transformed.z;
		const distance = Math.sqrt(dx * dx + dz * dz);

		if (distance < minDistance) {
			minDistance = distance;
			closestIndex = i;
		}
	}

	if (closestIndex === -1 || minDistance > getTooltipDistanceThreshold(analysis)) {
		return null;
	}

	return closestIndex;
}

function tryIntersectUtciSurfacePlane(
	raycaster: THREE.Raycaster,
	utciMesh: THREE.Mesh,
	layout: UtciGridLayout
): THREE.Intersection | null {
	if ((utciMesh as { isMesh?: boolean }).isMesh !== true || !getTooltipSurfaceBounds(layout)) {
		return null;
	}

	if (!utciMesh.matrixWorld.elements.every((value) => Number.isFinite(value))) {
		return null;
	}

	const planeOrigin = utciMesh.localToWorld(new THREE.Vector3(0, 0, 0));
	const planeNormal = new THREE.Vector3(0, 1, 0).transformDirection(utciMesh.matrixWorld);
	if (
		!Number.isFinite(planeOrigin.x) ||
		!Number.isFinite(planeOrigin.y) ||
		!Number.isFinite(planeOrigin.z) ||
		!Number.isFinite(planeNormal.x) ||
		!Number.isFinite(planeNormal.y) ||
		!Number.isFinite(planeNormal.z) ||
		planeNormal.lengthSq() <= 0
	) {
		return null;
	}

	const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(planeNormal, planeOrigin);
	const worldPoint = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
	if (!worldPoint) {
		return null;
	}

	const localPoint = getSurfaceLocalPoint(worldPoint, utciMesh);
	if (!localPoint || !getSurfaceCellCoordinates(localPoint, layout)) {
		return null;
	}

	return {
		distance: raycaster.ray.origin.distanceTo(worldPoint),
		point: worldPoint,
		object: utciMesh
	} as THREE.Intersection;
}

function resolveTooltipPositionIndex(
	intersection: THREE.Intersection,
	layout: UtciGridLayout,
	analysis: Analysis,
	diagnosticsEnabled: boolean,
	options?: { enforceDistanceThreshold?: boolean }
): {
	positionIndex: number | null;
	nearestPointMs: number;
	directCellHit: boolean;
	nearestScanUsed: boolean;
	inactiveCell: boolean;
} {
	const directResolution = getPositionIndexFromSurfaceCell(intersection, layout, analysis, {
		enforceDistanceThreshold: options?.enforceDistanceThreshold
	});
	if (directResolution.positionIndex !== null) {
		return {
			positionIndex: directResolution.positionIndex,
			nearestPointMs: diagnosticsEnabled ? directResolution.mappingMs : 0,
			directCellHit: true,
			nearestScanUsed: false,
			inactiveCell: false
		};
	}
	if (directResolution.inactiveCell) {
		return {
			positionIndex: null,
			nearestPointMs: diagnosticsEnabled ? directResolution.mappingMs : 0,
			directCellHit: false,
			nearestScanUsed: false,
			inactiveCell: true
		};
	}

	const nearestPointStart = diagnosticsEnabled ? performance.now() : 0;
	const positionIndex = getNearestPositionIndexFromIntersection(intersection, layout, analysis);
	return {
		positionIndex,
		nearestPointMs: diagnosticsEnabled ? performance.now() - nearestPointStart : 0,
		directCellHit: false,
		nearestScanUsed: true,
		inactiveCell: false
	};
}

export function resolvePositionIndexFromIntersection(
	intersection: THREE.Intersection,
	layout: UtciGridLayout,
	analysis: Analysis
): number | null {
	return resolveTooltipPositionIndex(intersection, layout, analysis, false, {
		enforceDistanceThreshold: true
	}).positionIndex;
}

/**
 * Get metric value for a position index
 */
function getMetricValue(
	analysis: Analysis,
	positionIndex: number,
	metricType: MetricType,
	hourIndex: number
): number | null {
	const { data } = analysis;

	if (metricType === 'shading_index') {
		const shadingIndexValues = getShadingIndex(data);
		if (!shadingIndexValues || positionIndex >= shadingIndexValues.length) {
			return null;
		}
		return shadingIndexValues[positionIndex];
	}

	const utciValues = getUTCIForHour(data, hourIndex);
	if (!utciValues || positionIndex >= utciValues.length) {
		return null;
	}
	return utciValues[positionIndex];
}

/**
 * Get position coordinates for a position index
 */
function getPositionCoordinates(
	analysis: Analysis,
	positionIndex: number
): { x: number; y: number; z: number } | null {
	const { data } = analysis;
	const positions = data.positions;

	if (positionIndex * 3 + 2 >= positions.length) {
		return null;
	}

	return {
		x: positions[positionIndex * 3],
		y: positions[positionIndex * 3 + 1],
		z: positions[positionIndex * 3 + 2]
	};
}

/**
 * Perform raycast to find hovered point and return tooltip data
 */
export function getTooltipData(
	event: MouseEvent,
	camera: THREE.Camera,
	utciMesh: THREE.Mesh | null,
	analysis: Analysis | null,
	metricType: MetricType,
	hourIndex: number,
	canvasRect: DOMRect,
	options?: {
		onDiagnosticsSample?: (measurement: TooltipInteractionMeasurement) => void;
	}
): TooltipData | null {
	const hit = resolveTooltipHit(event, camera, utciMesh, analysis, canvasRect, options);
	if (!hit) {
		return null;
	}

	const value = getMetricValue(analysis!, hit.positionIndex, metricType, hourIndex);
	if (value === null) {
		hit.emitMeasurement(false);
		return null;
	}

	hit.emitMeasurement(true);
	return {
		value,
		position: hit.position,
		positionIndex: hit.positionIndex
	};
}

function resolveTooltipHit(
	event: MouseEvent,
	camera: THREE.Camera,
	utciMesh: THREE.Mesh | null,
	analysis: Analysis | null,
	canvasRect: DOMRect,
	options?: {
		onDiagnosticsSample?: (measurement: TooltipInteractionMeasurement) => void;
	}
): TooltipHitResolution | null {
	const onDiagnosticsSample = options?.onDiagnosticsSample;
	const diagnosticsEnabled = typeof onDiagnosticsSample === 'function';
	const totalStart = diagnosticsEnabled ? performance.now() : 0;
	const emitMeasurement = (
		hit: boolean,
		raycastMs: number,
		nearestPointMs: number,
		resolutionPath: TooltipResolutionPath = 'none',
		directCellHit = false,
		nearestScanUsed = false,
		directCellMissCount = 0
	): void => {
		if (!diagnosticsEnabled) return;
		const totalMs = performance.now() - totalStart;
		onDiagnosticsSample({
			hit,
			raycastMs,
			nearestPointMs,
			totalMs,
			overBudget: totalMs > TOOLTIP_SLOW_BUDGET_MS,
			resolutionPath,
			directCellHit,
			nearestScanUsed,
			directCellMissCount
		});
	};

	if (!utciMesh || !analysis) {
		emitMeasurement(false, 0, 0);
		return null;
	}

	const layout: UtciGridLayout | undefined = utciMesh.userData.utciLayout;
	if (!layout) {
		emitMeasurement(false, 0, 0);
		return null;
	}

	const mouse = getNormalizedMousePosition(event, canvasRect);
	const gridSize = analysis.metadata.grid_size || 1.0;
	const raycaster = createRaycaster(camera, mouse, gridSize);
	let nearestPointMs = 0;
	let raycastMs = 0;
	let positionIndex: number | null = null;
	let resolutionPath: TooltipResolutionPath = 'none';
	let directCellHit = false;
	let nearestScanUsed = false;
	let directCellMissCount = 0;

	const planeIntersection = tryIntersectUtciSurfacePlane(raycaster, utciMesh, layout);
	if (planeIntersection) {
		resolutionPath = 'plane-cell';
		const planeResolution = resolveTooltipPositionIndex(
			planeIntersection,
			layout,
			analysis,
			diagnosticsEnabled,
			{ enforceDistanceThreshold: false }
		);
		positionIndex = planeResolution.positionIndex;
		nearestPointMs += planeResolution.nearestPointMs;
		directCellHit ||= planeResolution.directCellHit;
		nearestScanUsed ||= planeResolution.nearestScanUsed;
		directCellMissCount += planeResolution.directCellHit ? 0 : 1;
		if (planeResolution.inactiveCell) {
			emitMeasurement(
				false,
				raycastMs,
				nearestPointMs,
				resolutionPath,
				directCellHit,
				nearestScanUsed,
				directCellMissCount
			);
			return null;
		}
	}

	if (positionIndex === null) {
		resolutionPath = 'mesh-raycast';
		const raycastStart = diagnosticsEnabled ? performance.now() : 0;
		const intersections = raycaster.intersectObject(utciMesh, false);
		raycastMs = diagnosticsEnabled ? performance.now() - raycastStart : 0;
		if (intersections.length === 0) {
			emitMeasurement(
				false,
				raycastMs,
				nearestPointMs,
				resolutionPath,
				directCellHit,
				nearestScanUsed,
				directCellMissCount
			);
			return null;
		}

		const meshResolution = resolveTooltipPositionIndex(
			intersections[0],
			layout,
			analysis,
			diagnosticsEnabled,
			{ enforceDistanceThreshold: false }
		);
		positionIndex = meshResolution.positionIndex;
		nearestPointMs += meshResolution.nearestPointMs;
		directCellHit ||= meshResolution.directCellHit;
		nearestScanUsed ||= meshResolution.nearestScanUsed;
		directCellMissCount += meshResolution.directCellHit ? 0 : 1;
		if (meshResolution.inactiveCell) {
			emitMeasurement(
				false,
				raycastMs,
				nearestPointMs,
				resolutionPath,
				directCellHit,
				nearestScanUsed,
				directCellMissCount
			);
			return null;
		}
	}

	if (positionIndex === null || positionIndex < 0 || positionIndex >= analysis.data.numPositions) {
		emitMeasurement(
			false,
			raycastMs,
			nearestPointMs,
			resolutionPath,
			directCellHit,
			nearestScanUsed,
			directCellMissCount
		);
		return null;
	}

	const position = getPositionCoordinates(analysis, positionIndex);
	if (!position) {
		emitMeasurement(
			false,
			raycastMs,
			nearestPointMs,
			resolutionPath,
			directCellHit,
			nearestScanUsed,
			directCellMissCount
		);
		return null;
	}

	return {
		position,
		positionIndex,
		emitMeasurement: (hit: boolean) =>
			emitMeasurement(
				hit,
				raycastMs,
				nearestPointMs,
				resolutionPath,
				directCellHit,
				nearestScanUsed,
				directCellMissCount
			)
	};
}

export function getTooltipProbeData(
	event: MouseEvent,
	camera: THREE.Camera,
	utciMesh: THREE.Mesh | null,
	analysis: Analysis | null,
	canvasRect: DOMRect,
	options?: {
		onDiagnosticsSample?: (measurement: TooltipInteractionMeasurement) => void;
	}
): TooltipProbeData | null {
	const hit = resolveTooltipHit(event, camera, utciMesh, analysis, canvasRect, options);
	if (!hit) {
		return null;
	}

	return {
		position: hit.position,
		positionIndex: hit.positionIndex
	};
}

export async function getTooltipDataAsync(
	event: MouseEvent,
	camera: THREE.Camera,
	utciMesh: THREE.Mesh | null,
	analysis: Analysis | null,
	metricType: MetricType,
	hourIndex: number,
	canvasRect: DOMRect,
	options?: {
		onDiagnosticsSample?: (measurement: TooltipInteractionMeasurement) => void;
		metricPointValueReader?: TooltipMetricPointValueReader;
		onMetricPointReadbackSample?: (
			measurement: TooltipMetricPointReadbackMeasurement
		) => void;
	}
): Promise<TooltipData | null> {
	const hit = resolveTooltipHit(
		event,
		camera,
		utciMesh,
		analysis,
		canvasRect,
		options
	);
	if (!hit) {
		return null;
	}

	let value = analysis ? getMetricValue(analysis, hit.positionIndex, metricType, hourIndex) : null;
	if (
		value === null &&
		metricType === 'shading_index' &&
		options?.metricPointValueReader
	) {
		const reader = options.metricPointValueReader;
		const readbackStartedAt = performance.now();
		let cacheHit = false;
		let byteLength = 0;
		try {
			const key = {
				metricType,
				monthIndex: reader.monthIndex,
				positionIndex: hit.positionIndex,
				requestId: reader.requestId,
				ownerId: reader.ownerId
			};
			const result = await sharedMetricPointReadbackCache.getOrReadWithStats(
				key,
				() =>
					reader.readMetricPointValue({
						metricType,
						monthIndex: reader.monthIndex,
						positionIndex: hit.positionIndex,
						requestId: reader.requestId,
						ownerId: reader.ownerId
					})
			);
			value = result.value;
			cacheHit = result.cacheHit;
			byteLength = cacheHit ? 0 : (reader.readbackByteLength ?? 4);
			options.onMetricPointReadbackSample?.({
				...key,
				cacheHit,
				byteLength,
				latencyMs: performance.now() - readbackStartedAt,
				success: true
			});
		} catch {
			options.onMetricPointReadbackSample?.({
				metricType,
				monthIndex: reader.monthIndex,
				positionIndex: hit.positionIndex,
				requestId: reader.requestId,
				ownerId: reader.ownerId,
				cacheHit,
				byteLength,
				latencyMs: performance.now() - readbackStartedAt,
				success: false
			});
			value = null;
		}
	}
	if (value === null || !Number.isFinite(value)) {
		hit.emitMeasurement(false);
		return null;
	}

	hit.emitMeasurement(true);
	return {
		value,
		position: hit.position,
		positionIndex: hit.positionIndex
	};
}

function createTooltipWorldTransformContext(analysis: Analysis): TooltipWorldTransformContext {
	const coordinateSystem = analysis.metadata.coordinate_system || 'xy_ground';
	let normalizationOffset = new THREE.Vector3(0, 0, 0);

	if (isNormalizationEnabled()) {
		const scenarioOrigin = calculateScenarioOrigin(analysis.metadata as any);
		const anchorOffset = getAnchorOffset();
		let transformedOrigin: THREE.Vector3;
		if (coordinateSystem === 'xy_ground') {
			transformedOrigin = new THREE.Vector3(scenarioOrigin.x, scenarioOrigin.z, -scenarioOrigin.y);
		} else {
			transformedOrigin = scenarioOrigin.clone();
		}

		normalizationOffset = anchorOffset.clone().sub(transformedOrigin);
		if (normalizationOffset.lengthSq() <= 0.001) {
			normalizationOffset.set(0, 0, 0);
		}
	}

	return {
		coordinateSystem,
		normalizationOffset
	};
}

function transformAnalysisPositionToWorld(
	positions: Float32Array,
	positionIndex: number,
	context: TooltipWorldTransformContext,
	target: THREE.Vector3
): THREE.Vector3 {
	const x = positions[positionIndex * 3];
	const y = positions[positionIndex * 3 + 1];
	const z = positions[positionIndex * 3 + 2];

	if (context.coordinateSystem === 'xy_ground') {
		target.set(x, z, -y);
	} else {
		target.set(x, y, z);
	}

	target.add(context.normalizationOffset);
	return target;
}

function getTooltipDistanceThreshold(analysis: Analysis): number {
	return (analysis.metadata.grid_size || 1.0) * 0.7;
}
