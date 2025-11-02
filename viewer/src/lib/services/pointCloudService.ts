/**
 * Point Cloud Service
 * 
 * Service for creating and updating UTCI point cloud geometries
 */

import * as THREE from 'three';
import type { Analysis, UTCIData } from '$lib/types/analysis';
import { getUTCIForHour } from '$lib/services/dataLoader';
import { mapUTCIToColor } from '$lib/services/colorScale';
import { applyCoordinateTransform } from '$lib/utils/coordinates';

const VISUAL_OFFSET = 0.4; // Small vertical offset to prevent z-fighting

/**
 * Create UTCI point cloud geometry and material
 * @param analysis - Analysis data
 * @param hourIndex - Hour index (default: 0)
 * @param colorMode - Color mode ('normalized' or 'discrete')
 * @returns Object with geometry and material
 */
export function createPointCloudGeometry(
	analysis: Analysis,
	hourIndex: number = 0,
	colorMode: 'normalized' | 'discrete' = 'normalized'
): { geometry: THREE.BufferGeometry; material: THREE.PointsMaterial } {
	const { data, metadata } = analysis;
	const numPositions = data.numPositions;

	// Create position attribute with visual offset
	const positions = new Float32Array(data.positions.length);
	for (let i = 0; i < numPositions; i++) {
		positions[i * 3] = data.positions[i * 3];
		positions[i * 3 + 1] = data.positions[i * 3 + 1];
		positions[i * 3 + 2] = data.positions[i * 3 + 2] + VISUAL_OFFSET;
	}

	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	// Create color attribute
	const colors = createColors(analysis, hourIndex, colorMode);
	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

	// Create material
	const material = new THREE.PointsMaterial({
		size: 8,
		vertexColors: true,
		sizeAttenuation: false,
		transparent: true,
		opacity: 0.95,
		depthTest: true,
		depthWrite: false,
		alphaTest: 0.1
	});

	return { geometry, material };
}

/**
 * Create color array for point cloud
 * @param analysis - Analysis data
 * @param hourIndex - Hour index
 * @param colorMode - Color mode
 * @returns Color Float32Array
 */
export function createColors(
	analysis: Analysis,
	hourIndex: number,
	colorMode: 'normalized' | 'discrete'
): Float32Array {
	const { data, metadata } = analysis;
	const numPositions = data.numPositions;
	const utciValues = getUTCIForHour(data, hourIndex);

	// Determine range based on color mode
	let utciMin: number;
	let utciMax: number;

	if (colorMode === 'normalized') {
		utciMin = metadata.utci_range.min;
		utciMax = metadata.utci_range.max;
	} else {
		// Discrete mode - use per-hour range
		if (metadata.hour_statistics && metadata.hour_statistics[hourIndex]) {
			utciMin = metadata.hour_statistics[hourIndex].min;
			utciMax = metadata.hour_statistics[hourIndex].max;
		} else {
			utciMin = metadata.utci_range.min;
			utciMax = metadata.utci_range.max;
		}
	}

	// Create color attribute
	const colors = new Float32Array(numPositions * 3);

	for (let i = 0; i < numPositions; i++) {
		const utci = utciValues[i];
		const color = mapUTCIToColor(utci, utciMin, utciMax);
		colors[i * 3] = color.r;
		colors[i * 3 + 1] = color.g;
		colors[i * 3 + 2] = color.b;
	}

	return colors;
}

/**
 * Update point cloud colors
 * @param pointCloud - Three.js Points object
 * @param analysis - Analysis data
 * @param hourIndex - Hour index
 * @param colorMode - Color mode
 */
export function updatePointCloudColors(
	pointCloud: THREE.Points,
	analysis: Analysis,
	hourIndex: number,
	colorMode: 'normalized' | 'discrete'
): void {
	const { data } = analysis;

	if (data.numHours === 1) {
		console.warn('[WARN] Cannot update colors for single hour analysis');
		return;
	}

	const colors = createColors(analysis, hourIndex, colorMode);
	const colorAttribute = pointCloud.geometry.getAttribute('color') as THREE.BufferAttribute;

	for (let i = 0; i < colors.length / 3; i++) {
		colorAttribute.setXYZ(i, colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
	}

	colorAttribute.needsUpdate = true;
}


