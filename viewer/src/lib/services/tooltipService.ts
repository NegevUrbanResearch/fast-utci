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
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';

export interface TooltipData {
	value: number;
	position: { x: number; y: number; z: number };
	positionIndex: number;
}

export interface TooltipInteractionMeasurement {
	hit: boolean;
	raycastMs: number;
	nearestPointMs: number;
	totalMs: number;
	overBudget: boolean;
}

export interface TooltipInteractionDiagnostics {
	enabled: boolean;
	disabledByQuery: boolean;
	slowThresholdMs: number;
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

export const TOOLTIP_SLOW_BUDGET_MS = 8;

export function createEmptyTooltipInteractionDiagnostics(
	disabledByQuery: boolean
): TooltipInteractionDiagnostics {
	return {
		enabled: !disabledByQuery,
		disabledByQuery,
		slowThresholdMs: TOOLTIP_SLOW_BUDGET_MS,
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
		maxTotalMs: 0
	};
}

export function recordTooltipInteractionMeasurement(
	diagnostics: TooltipInteractionDiagnostics,
	measurement: TooltipInteractionMeasurement
): TooltipInteractionDiagnostics {
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
		maxTotalMs: Math.max(diagnostics.maxTotalMs, measurement.totalMs)
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
	analysis: Analysis
): number | null {
	const worldPoint = intersection.point;
	if (!worldPoint) return null;

	const object = intersection.object;
	if (!object) return null;

	const cellToPointIndex = layout.cellToPointIndex;
	const expectedCellCount = layout.width * layout.height;
	if (!cellToPointIndex || cellToPointIndex.length !== expectedCellCount) {
		return null;
	}

	const localPoint = getSurfaceLocalPoint(worldPoint, object);
	if (!localPoint) {
		return null;
	}

	const cellCoordinates = getSurfaceCellCoordinates(localPoint, layout);
	if (!cellCoordinates) {
		return null;
	}

	const { column, row } = cellCoordinates;
	const pointIndex = cellToPointIndex[row * layout.width + column];
	if (pointIndex < 0 || pointIndex >= analysis.data.numPositions) {
		return null;
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
		return null;
	}

	const dx = worldPoint.x - transformed.x;
	const dz = worldPoint.z - transformed.z;
	const distance = Math.sqrt(dx * dx + dz * dz);
	if (!Number.isFinite(distance) || distance > getTooltipDistanceThreshold(analysis)) {
		return null;
	}

	return pointIndex;
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
	diagnosticsEnabled: boolean
): { positionIndex: number | null; nearestPointMs: number } {
	const directPositionIndex = getPositionIndexFromSurfaceCell(intersection, layout, analysis);
	if (directPositionIndex !== null) {
		return {
			positionIndex: directPositionIndex,
			nearestPointMs: 0
		};
	}

	const nearestPointStart = diagnosticsEnabled ? performance.now() : 0;
	const positionIndex = getNearestPositionIndexFromIntersection(intersection, layout, analysis);
	return {
		positionIndex,
		nearestPointMs: diagnosticsEnabled ? performance.now() - nearestPointStart : 0
	};
}

export function resolvePositionIndexFromIntersection(
	intersection: THREE.Intersection,
	layout: UtciGridLayout,
	analysis: Analysis
): number | null {
	return resolveTooltipPositionIndex(intersection, layout, analysis, false).positionIndex;
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
	const onDiagnosticsSample = options?.onDiagnosticsSample;
	const diagnosticsEnabled = typeof onDiagnosticsSample === 'function';
	const totalStart = diagnosticsEnabled ? performance.now() : 0;
	const emitMeasurement = (
		hit: boolean,
		raycastMs: number,
		nearestPointMs: number
	): void => {
		if (!diagnosticsEnabled) return;
		const totalMs = performance.now() - totalStart;
		onDiagnosticsSample({
			hit,
			raycastMs,
			nearestPointMs,
			totalMs,
			overBudget: totalMs > TOOLTIP_SLOW_BUDGET_MS
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

	const planeIntersection = tryIntersectUtciSurfacePlane(raycaster, utciMesh, layout);
	if (planeIntersection) {
		const planeResolution = resolveTooltipPositionIndex(
			planeIntersection,
			layout,
			analysis,
			diagnosticsEnabled
		);
		positionIndex = planeResolution.positionIndex;
		nearestPointMs += planeResolution.nearestPointMs;
	}

	if (positionIndex === null) {
		const raycastStart = diagnosticsEnabled ? performance.now() : 0;
		const intersections = raycaster.intersectObject(utciMesh, false);
		raycastMs = diagnosticsEnabled ? performance.now() - raycastStart : 0;
		if (intersections.length === 0) {
			emitMeasurement(false, raycastMs, nearestPointMs);
			return null;
		}

		const meshResolution = resolveTooltipPositionIndex(
			intersections[0],
			layout,
			analysis,
			diagnosticsEnabled
		);
		positionIndex = meshResolution.positionIndex;
		nearestPointMs += meshResolution.nearestPointMs;
	}

	if (positionIndex === null || positionIndex < 0 || positionIndex >= analysis.data.numPositions) {
		emitMeasurement(false, raycastMs, nearestPointMs);
		return null;
	}

	const value = getMetricValue(analysis, positionIndex, metricType, hourIndex);
	if (value === null) {
		emitMeasurement(false, raycastMs, nearestPointMs);
		return null;
	}

	const position = getPositionCoordinates(analysis, positionIndex);
	if (!position) {
		emitMeasurement(false, raycastMs, nearestPointMs);
		return null;
	}

	emitMeasurement(true, raycastMs, nearestPointMs);
	return {
		value,
		position,
		positionIndex
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
