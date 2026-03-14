/**
 * Shared Bounds Calculation Utilities
 * 
 * Generalizes model bounds calculations used across components
 */

import * as THREE from 'three';

/**
 * Calculate model bounds
 * @param model - Three.js model (Group or Object3D)
 * @returns Bounding box
 */
export function calculateModelBounds(model: THREE.Object3D): THREE.Box3 {
	return new THREE.Box3().setFromObject(model);
}

/**
 * Calculate model center
 * @param model - Three.js model (Group or Object3D)
 * @returns Center point
 */
export function calculateModelCenter(model: THREE.Object3D): THREE.Vector3 {
	const box = calculateModelBounds(model);
	const center = new THREE.Vector3();
	box.getCenter(center);
	return center;
}

/**
 * Calculate model size
 * @param model - Three.js model (Group or Object3D)
 * @returns Size vector
 */
export function calculateModelSize(model: THREE.Object3D): THREE.Vector3 {
	const box = calculateModelBounds(model);
	const size = new THREE.Vector3();
	box.getSize(size);
	return size;
}

/**
 * Get bounds, center, and size from a single setFromObject pass (avoids triple traversal).
 */
export function getBoundsCenterAndSize(model: THREE.Object3D): {
	bounds: THREE.Box3;
	center: THREE.Vector3;
	size: THREE.Vector3;
} {
	const bounds = calculateModelBounds(model);
	const center = new THREE.Vector3();
	const size = new THREE.Vector3();
	bounds.getCenter(center);
	bounds.getSize(size);
	return { bounds, center, size };
}

/**
 * Get ground level (minimum Y coordinate)
 * @param model - Three.js model (Group or Object3D)
 * @returns Ground level Y coordinate
 */
export function getGroundLevel(model: THREE.Object3D): number {
	const box = calculateModelBounds(model);
	return box.min.y;
}


