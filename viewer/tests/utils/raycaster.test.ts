import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import { createRaycaster } from '$lib/utils/raycaster';

describe('raycaster utilities', () => {
	let camera: THREE.PerspectiveCamera;
	let mouse: THREE.Vector2;

	beforeEach(() => {
		camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
		camera.position.set(0, 0, 10);
		camera.lookAt(0, 0, 0);
		mouse = new THREE.Vector2(0, 0);
	});

	it('should create raycaster with default threshold', () => {
		const raycaster = createRaycaster(camera, mouse);
		expect(raycaster).toBeInstanceOf(THREE.Raycaster);
		expect(raycaster.params.Points.threshold).toBe(5);
	});

	it('should create raycaster with custom threshold', () => {
		const raycaster = createRaycaster(camera, mouse, 10.0);
		expect(raycaster.params.Points.threshold).toBe(5); // Should be half of gridSize
	});

	it('should set raycaster from camera and mouse position', () => {
		const raycaster = createRaycaster(camera, mouse);
		expect(raycaster.ray.origin).toBeDefined();
		expect(raycaster.ray.direction).toBeDefined();
	});
});


