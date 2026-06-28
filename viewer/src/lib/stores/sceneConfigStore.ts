import { writable, type Writable } from 'svelte/store';
import * as THREE from 'three';

export interface SceneConfig {
	sceneRadius: number;
	cameraNear: number;
	cameraFar: number;
}

const defaultSceneConfig: SceneConfig = {
	sceneRadius: 1000,
	cameraNear: 1,
	cameraFar: 1000
};

export const sceneConfigStore: Writable<SceneConfig> = writable<SceneConfig>(
	defaultSceneConfig
);

export function updateSceneConfigFromBounds(bounds: THREE.Box3): void {
	const size = new THREE.Vector3();
	bounds.getSize(size);
	const center = new THREE.Vector3();
	bounds.getCenter(center);

	// Approximate scene radius as half the diagonal of the bounds.
	const radius = size.length() * 0.5;

	// Derive near/far planes from radius. Keep a reasonable ratio to preserve
	// depth precision while allowing zoom-out over larger projects.
	const near = Math.max(radius / 500, 0.5);
	const far = radius * 6;

	sceneConfigStore.set({
		sceneRadius: radius,
		cameraNear: near,
		cameraFar: far
	});
}

