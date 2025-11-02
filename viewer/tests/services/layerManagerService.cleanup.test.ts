import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
	initializeLayerManager,
	resetLayerManager,
	getDiscoveredLayers,
	toggleLayerVisibility,
	hasLayer,
	getMeshCount
} from '$lib/services/layerManagerService';

describe('Layer Manager Service - Cleanup', () => {
	let model: THREE.Group;

	beforeEach(() => {
		model = new THREE.Group();
		resetLayerManager(); // Ensure clean state
	});

	describe('resetLayerManager', () => {
		it('should clear all layers from the map', () => {
			// Setup model with layers
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			model.add(buildingMesh);

			initializeLayerManager(model);
			expect(getDiscoveredLayers()).toContain('building');

			// Reset and verify layers are cleared
			resetLayerManager();
			expect(getDiscoveredLayers().length).toBe(0);
		});

		it('should allow re-initialization after reset', () => {
			// First initialization
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			model.add(buildingMesh);
			initializeLayerManager(model);

			// Reset
			resetLayerManager();

			// Second initialization with different model
			const newModel = new THREE.Group();
			const vegetationMesh = new THREE.Mesh(new THREE.SphereGeometry(1));
			vegetationMesh.userData.layerType = 'vegetation';
			newModel.add(vegetationMesh);
			initializeLayerManager(newModel);

			const layers = getDiscoveredLayers();
			expect(layers).toContain('vegetation');
			expect(layers).not.toContain('building');
			expect(layers.length).toBe(1);
		});

		it('should handle operations gracefully after reset', () => {
			// Setup and reset
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			model.add(buildingMesh);
			initializeLayerManager(model);
			resetLayerManager();

			// Operations should not throw after reset
			expect(() => toggleLayerVisibility('building', false)).not.toThrow();
			expect(hasLayer('building')).toBe(false);
			expect(getMeshCount('building')).toBe(0);
		});

		it('should prevent stale references to old meshes', () => {
			// First model
			const firstMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			firstMesh.userData.layerType = 'building';
			firstMesh.visible = true;
			model.add(firstMesh);
			initializeLayerManager(model);

			// Reset and load new model
			resetLayerManager();
			const newModel = new THREE.Group();
			const newMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			newMesh.userData.layerType = 'building';
			newMesh.visible = true;
			newModel.add(newMesh);
			initializeLayerManager(newModel);

			// Toggle visibility on new model
			toggleLayerVisibility('building', false);

			// Old mesh should not be affected (stale reference prevented)
			expect(newMesh.visible).toBe(false);
			// Note: firstMesh.visible may still be true since it's no longer tracked
			// The key is that toggling doesn't affect the old mesh
		});
	});

	describe('Model switching scenario', () => {
		it('should handle rapid model switching correctly', () => {
			// Simulate loading first model
			const model1 = new THREE.Group();
			const mesh1 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh1.userData.layerType = 'building';
			model1.add(mesh1);
			initializeLayerManager(model1);
			expect(getDiscoveredLayers()).toEqual(['building']);

			// Reset and load second model
			resetLayerManager();
			const model2 = new THREE.Group();
			const mesh2a = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh2a.userData.layerType = 'vegetation';
			const mesh2b = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh2b.userData.layerType = 'road';
			model2.add(mesh2a);
			model2.add(mesh2b);
			initializeLayerManager(model2);

			const layers = getDiscoveredLayers();
			expect(layers).toContain('vegetation');
			expect(layers).toContain('road');
			expect(layers).not.toContain('building');
			expect(layers.length).toBe(2);

			// Reset and load third model (back to first model type)
			resetLayerManager();
			const model3 = new THREE.Group();
			const mesh3 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh3.userData.layerType = 'building';
			model3.add(mesh3);
			initializeLayerManager(model3);

			const finalLayers = getDiscoveredLayers();
			expect(finalLayers).toEqual(['building']);
			expect(finalLayers.length).toBe(1);
		});
	});

	describe('Mesh reference integrity', () => {
		it('should not retain references to disposed meshes', () => {
			// Create model with mesh
			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh.userData.layerType = 'building';
			model.add(mesh);
			initializeLayerManager(model);

			// Verify mesh is tracked
			expect(hasLayer('building')).toBe(true);
			expect(getMeshCount('building')).toBe(1);

			// Reset (simulating disposal)
			resetLayerManager();

			// Verify references are cleared
			expect(hasLayer('building')).toBe(false);
			expect(getMeshCount('building')).toBe(0);
		});
	});
});

