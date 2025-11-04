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

const VISUAL_OFFSET = 0.0; // Object-level offset will be applied post-transform
const OVERLAP_FACTOR = 1.05; // Slight overlap to fill gaps between tiles

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

    // Create position attribute without baked vertical offset
	const positions = new Float32Array(data.positions.length);
	for (let i = 0; i < numPositions; i++) {
		positions[i * 3] = data.positions[i * 3];
		positions[i * 3 + 1] = data.positions[i * 3 + 1];
        positions[i * 3 + 2] = data.positions[i * 3 + 2];
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

/**
 * Create ground-aligned InstancedMesh geometry and material
 * @param analysis - Analysis data
 * @param hourIndex - Hour index (default: 0)
 * @param colorMode - Color mode ('normalized' or 'discrete')
 * @returns Object with geometry, material, instance count, and cell size
 */
export function createInstancedMeshGeometry(
	analysis: Analysis,
	hourIndex: number = 0,
	colorMode: 'normalized' | 'discrete' = 'normalized'
): { geometry: THREE.PlaneGeometry; material: THREE.MeshBasicMaterial; instanceCount: number; cellSize: number } {
	const { data, metadata } = analysis;
	const instanceCount = data.numPositions;
	// Make planes larger to ensure visibility - use 2x grid size for testing
	const cellSize = metadata.grid_size * 2.0; // Larger for visibility testing

	// Create plane geometry for each instance
	const geometry = new THREE.PlaneGeometry(cellSize, cellSize);

	// Create material with solid color for testing - no instance colors
	// Once we confirm rendering works, we'll add instance colors back
	const material = new THREE.MeshBasicMaterial({
		color: 0xff0000, // Bright red - solid color for testing
		transparent: false,
		side: THREE.DoubleSide,
		depthTest: true,
		depthWrite: false
	});

	return { geometry, material, instanceCount, cellSize };
}

/**
 * Update InstancedMesh instance matrices (positions) and colors
 * @param instancedMesh - Three.js InstancedMesh object
 * @param analysis - Analysis data
 * @param hourIndex - Hour index
 * @param colorMode - Color mode
 */
export function updateInstancedMesh(
	instancedMesh: THREE.InstancedMesh,
	analysis: Analysis,
	hourIndex: number,
	colorMode: 'normalized' | 'discrete'
): void {
	const { data, metadata } = analysis;
	const numPositions = data.numPositions;
	const instanceCount = Math.min(numPositions, instancedMesh.count);
	const coordinateSystem = metadata.coordinate_system || 'xy_ground';

	// Create colors
	const colors = createColors(analysis, hourIndex, colorMode);

	// Set instance matrices (positions) and colors
	const dummy = new THREE.Object3D();
	const color = new THREE.Color();

	// TEST: Render first instance at origin to verify rendering works
	const testCount = Math.min(instanceCount, 1);
	console.log('[UTCI] Rendering', testCount, 'instances for testing');
	
	for (let i = 0; i < testCount; i++) {
		// Transform positions based on coordinate system BEFORE setting instance matrix
		// This way we don't need to rotate the mesh object
		let x = data.positions[i * 3];
		let y = data.positions[i * 3 + 1];
		let z = data.positions[i * 3 + 2];
		
		// Transform coordinates: xy_ground (Z-up) -> Three.js (Y-up)
		if (coordinateSystem === 'xy_ground') {
			// Swap Y and Z: (X, Y, Z) -> (X, Z, -Y)
			const temp = y;
			y = z;
			z = -temp;
		}
		
		// Log first position for debugging
		if (i === 0) {
			console.log('[UTCI] First instance position:', x, y, z, 'original:', data.positions[0], data.positions[1], data.positions[2]);
		}
		
		// TEST: Position at origin for first instance
		if (i === 0) {
			dummy.position.set(0, 0, 0);
			console.log('[UTCI] TEST: First instance at origin (0,0,0)');
		} else {
			dummy.position.set(x, y, z);
		}
		
		// Planes should be horizontal (flat on ground)
		// PlaneGeometry is in XY plane facing +Z, rotate to face +Y (up) so visible from above
		// Try +90° around X to face +Y
		dummy.rotation.set(Math.PI / 2, 0, 0);
		dummy.scale.set(1, 1, 1);
		dummy.updateMatrix();
		instancedMesh.setMatrixAt(i, dummy.matrix);

		// Skip instance colors for now - testing with solid material color
		// Once we confirm planes render, we'll enable instance colors
		// color.setRGB(1, 0, 0);
		// instancedMesh.setColorAt(i, color);
	}
	
	// Hide remaining instances by setting their scale to 0
	for (let i = testCount; i < instanceCount; i++) {
		dummy.position.set(0, 0, 0);
		dummy.rotation.set(0, 0, 0);
		dummy.scale.set(0, 0, 0);
		dummy.updateMatrix();
		instancedMesh.setMatrixAt(i, dummy.matrix);
	}

	instancedMesh.instanceMatrix.needsUpdate = true;
	if (instancedMesh.instanceColor) {
		instancedMesh.instanceColor.needsUpdate = true;
	}
	
	// CRITICAL: Compute bounding sphere for frustum culling
	// Without this, Three.js may cull the entire InstancedMesh
	instancedMesh.computeBoundingSphere();
}


