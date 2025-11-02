/**
 * Camera Store
 * 
 * Svelte store for managing camera state (position, target, zoom limits)
 */

import { writable, type Writable } from 'svelte/store';
import * as THREE from 'three';

/**
 * Camera state
 */
export interface CameraState {
	position: THREE.Vector3;
	target: THREE.Vector3;
	minDistance: number;
	maxDistance: number;
	zoomSpeed: number;
}

/**
 * Default camera state
 */
const defaultCameraState: CameraState = {
	position: new THREE.Vector3(-2000, 300, -400),
	target: new THREE.Vector3(-2000, 0, -400),
	minDistance: 200,
	maxDistance: 2000,
	zoomSpeed: 1.0  // Use default Three.js zoom speed
};

/**
 * Camera store
 */
export const cameraStore: Writable<CameraState> = writable<CameraState>(defaultCameraState);

/**
 * Set camera position
 * @param position - Camera position
 */
export function setCameraPosition(position: THREE.Vector3): void {
	cameraStore.update((state) => ({
		...state,
		position: position.clone()
	}));
}

/**
 * Set camera target
 * @param target - Camera target
 */
export function setCameraTarget(target: THREE.Vector3): void {
	cameraStore.update((state) => ({
		...state,
		target: target.clone()
	}));
}

/**
 * Focus camera on model bounds
 * @param center - Model center
 * @param size - Model size
 */
export function focusCameraOnModel(center: THREE.Vector3, size: THREE.Vector3): void {
	const maxDim = Math.max(size.x, size.y, size.z);
	const fov = 60 * (Math.PI / 180);
	let cameraDistance = Math.abs(maxDim / 2 / Math.tan(fov / 2));
	cameraDistance *= 1.5; // Add margin

	// Position camera at an angle for better perspective
	const position = new THREE.Vector3(
		center.x + cameraDistance * 0.7,
		center.y + cameraDistance * 0.5,
		center.z + cameraDistance * 0.7
	);

	cameraStore.update((state) => ({
		...state,
		position,
		target: center.clone()
	}));
}


