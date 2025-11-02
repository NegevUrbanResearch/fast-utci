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


