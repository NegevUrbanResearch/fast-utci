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

/**
 * Find the position index from a mesh intersection
 * Uses the intersection point's world coordinates to find the closest position
 * This accounts for coordinate transformations and mesh positioning
 */
function getPositionIndexFromIntersection(
	intersection: THREE.Intersection,
	layout: UtciGridLayout,
	analysis: Analysis
): number | null {
	// Get the 3D intersection point in world space
	const worldPoint = intersection.point;
	if (!worldPoint) return null;

	// Get positions from analysis data
	const positions = analysis.data.positions;
	const numPositions = analysis.data.numPositions;
	const coordinateSystem = analysis.metadata.coordinate_system || 'xy_ground';
	const gridSize = analysis.metadata.grid_size || 1.0;

	// Calculate normalization offset if enabled (matching pointCloudService.ts)
	let normalizationOffset = new THREE.Vector3(0, 0, 0);
	if (isNormalizationEnabled()) {
		const scenarioOrigin = calculateScenarioOrigin(analysis.metadata as any);
		const anchorOffset = getAnchorOffset();
		
		// Transform scenario origin to world space to match the coordinate system
		let transformedOrigin: THREE.Vector3;
		if (coordinateSystem === 'xy_ground') {
			// Transform origin to world space: (x, y, z) → (x, z, -y)
			transformedOrigin = new THREE.Vector3(scenarioOrigin.x, scenarioOrigin.z, -scenarioOrigin.y);
		} else {
			transformedOrigin = scenarioOrigin.clone();
		}
		
		// Calculate offset in world space (where anchorOffset already is)
		normalizationOffset = anchorOffset.clone().sub(transformedOrigin);
		
		if (normalizationOffset.lengthSq() <= 0.001) {
			normalizationOffset.set(0, 0, 0);
		}
	}

	// Transform function (matching pointCloudService.ts)
	function transformToWorld(x: number, y: number, z: number, target: THREE.Vector3): THREE.Vector3 {
		if (coordinateSystem === 'xy_ground') {
			target.set(x, z, -y);
		} else {
			target.set(x, y, z);
		}
		// Apply normalization offset
		target.add(normalizationOffset);
		return target;
	}

	// Find the closest position to the intersection point
	// Only compare X and Z coordinates (ground plane), ignore Y
	let closestIndex = -1;
	let minDistance = Infinity;
	const transformed = new THREE.Vector3();

	for (let i = 0; i < numPositions; i++) {
		const posX = positions[i * 3];
		const posY = positions[i * 3 + 1];
		const posZ = positions[i * 3 + 2];

		// Transform position to world space (matching the mesh transformation)
		transformToWorld(posX, posY, posZ, transformed);

		// Calculate distance in XZ plane only (ground plane)
		const dx = worldPoint.x - transformed.x;
		const dz = worldPoint.z - transformed.z;
		const distance = Math.sqrt(dx * dx + dz * dz);

		if (distance < minDistance) {
			minDistance = distance;
			closestIndex = i;
		}
	}

	// Only return if within reasonable distance (half grid size)
	if (closestIndex === -1 || minDistance > gridSize * 0.7) {
		return null;
	}

	return closestIndex;
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
	} else {
		// UTCI
		const utciValues = getUTCIForHour(data, hourIndex);
		if (!utciValues || positionIndex >= utciValues.length) {
			return null;
		}
		return utciValues[positionIndex];
	}
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
	canvasRect: DOMRect
): TooltipData | null {
	if (!utciMesh || !analysis) {
		return null;
	}

	// Get layout from mesh userData
	const layout: UtciGridLayout | undefined = utciMesh.userData.utciLayout;
	if (!layout) {
		return null;
	}

	// Get normalized mouse position
	const mouse = getNormalizedMousePosition(event, canvasRect);

	// Create raycaster
	const gridSize = analysis.metadata.grid_size || 1.0;
	const raycaster = createRaycaster(camera, mouse, gridSize);

	// Intersect with the UTCI mesh
	const intersections = raycaster.intersectObject(utciMesh, false);
	if (intersections.length === 0) {
		return null;
	}

	const intersection = intersections[0];

	// Get position index from intersection
	const positionIndex = getPositionIndexFromIntersection(intersection, layout, analysis);
	if (positionIndex === null || positionIndex < 0 || positionIndex >= analysis.data.numPositions) {
		return null;
	}

	// Get metric value
	const value = getMetricValue(analysis, positionIndex, metricType, hourIndex);
	if (value === null) {
		return null;
	}

	// Get position coordinates
	const position = getPositionCoordinates(analysis, positionIndex);
	if (!position) {
		return null;
	}

	return {
		value,
		position,
		positionIndex
	};
}

