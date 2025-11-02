/**
 * Shared Raycaster Utilities
 * 
 * Generalizes raycaster creation for both UTCI point picking and sun marker picking
 */

import * as THREE from 'three';

/**
 * Create a raycaster for point picking
 * @param camera - Three.js camera
 * @param mouse - Normalized mouse coordinates (-1 to 1)
 * @param gridSize - Grid spacing in meters (for dynamic threshold). Default: 10.0
 * @returns Raycaster configured for point picking
 */
export function createRaycaster(
	camera: THREE.Camera,
	mouse: THREE.Vector2,
	gridSize: number = 10.0
): THREE.Raycaster {
	const raycaster = new THREE.Raycaster();
	// Set threshold to half the grid size for accurate point picking
	// This allows hovering within half a grid spacing to select a point
	raycaster.params.Points.threshold = gridSize * 0.5;
	raycaster.setFromCamera(mouse, camera);
	return raycaster;
}

/**
 * Create a raycaster for sun marker picking (fixed threshold)
 * @param camera - Three.js camera
 * @param mouse - Normalized mouse coordinates (-1 to 1)
 * @returns Raycaster configured for marker picking
 */
export function createSunMarkerRaycaster(
	camera: THREE.Camera,
	mouse: THREE.Vector2
): THREE.Raycaster {
	const raycaster = new THREE.Raycaster();
	raycaster.params.Points.threshold = 10;
	raycaster.setFromCamera(mouse, camera);
	return raycaster;
}


