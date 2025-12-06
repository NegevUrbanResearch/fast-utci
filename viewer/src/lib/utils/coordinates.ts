/**
 * Shared Coordinate Transformation Utilities
 * 
 * Generalizes coordinate system transformations used in ModelLoader and UTCIRenderer
 */

import * as THREE from 'three';

/**
 * Apply coordinate transformation based on coordinate system
 * @param object - Three.js object to transform
 * @param coordinateSystem - Coordinate system ('xy_ground' or 'xz_ground')
 */
export function applyCoordinateTransform(
	object: THREE.Object3D,
	coordinateSystem: 'xy_ground' | 'xz_ground'
): void {
	if (coordinateSystem === 'xy_ground') {
		// Model uses Z-up (XY is ground plane)
		// Rotate -90 degrees around X axis to convert Z-up to Y-up
		object.rotation.x = -Math.PI / 2;
	}
	// xz_ground uses Y-up by default (Three.js standard), no transform needed
}

/**
 * Calculate scenario origin from metadata bounds
 * Returns the center point of the bounds in the original coordinate system
 * @param metadata - Analysis metadata with bounds
 * @returns Center point as Vector3
 */
export function calculateScenarioOrigin(metadata: { bounds?: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number } }): THREE.Vector3 {
	const bounds = metadata.bounds;
	if (!bounds) {
		// Fallback: return zero if no bounds available
		return new THREE.Vector3(0, 0, 0);
	}

	const centerX = (bounds.x_min + bounds.x_max) / 2;
	const centerY = (bounds.y_min + bounds.y_max) / 2;
	const centerZ = bounds.z ?? 0;

	return new THREE.Vector3(centerX, centerY, centerZ);
}

/**
 * Apply offset translation to align model to anchor point
 * This should be called AFTER coordinate system rotation
 * @param object - Three.js object to translate
 * @param offset - Offset vector to apply
 */
export function applyModelOffset(object: THREE.Object3D, offset: THREE.Vector3): void {
	object.position.add(offset);
}
