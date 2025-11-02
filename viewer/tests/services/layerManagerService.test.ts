import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
	discoverLayers,
	toggleLayerVisibility,
	getDiscoveredLayers,
	initializeLayerManager
} from '$lib/services/layerManagerService';

describe('Layer Manager Service', () => {
	let model: THREE.Group;

	beforeEach(() => {
		model = new THREE.Group();
		// Reset service state
	});

	describe('discoverLayers', () => {
		it('should discover layers from model scene graph', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			const layers = discoverLayers(model);

			expect(layers.has('building')).toBe(true);
			expect(layers.get('building')?.length).toBe(1);
			expect(layers.get('building')?.[0]).toBe(buildingMesh);
		});

		it('should discover multiple layer types', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			buildingParent.add(buildingMesh);

			const vegetationParent = new THREE.Group();
			vegetationParent.name = 'Trees';
			const vegetationMesh = new THREE.Mesh(new THREE.SphereGeometry(1));
			vegetationMesh.userData.layerType = 'vegetation';
			vegetationParent.add(vegetationMesh);

			model.add(buildingParent);
			model.add(vegetationParent);

			const layers = discoverLayers(model);

			expect(layers.has('building')).toBe(true);
			expect(layers.has('vegetation')).toBe(true);
			expect(layers.get('building')?.length).toBe(1);
			expect(layers.get('vegetation')?.length).toBe(1);
		});

		it('should group multiple meshes of the same layer type', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			
			const mesh1 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh1.userData.layerType = 'building';
			const mesh2 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh2.userData.layerType = 'building';
			
			buildingParent.add(mesh1);
			buildingParent.add(mesh2);
			model.add(buildingParent);

			const layers = discoverLayers(model);

			expect(layers.has('building')).toBe(true);
			expect(layers.get('building')?.length).toBe(2);
		});

		it('should ignore meshes without layerType in userData', () => {
			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			// No userData.layerType
			model.add(mesh);

			const layers = discoverLayers(model);

			expect(layers.size).toBe(0);
		});

		it('should handle empty model', () => {
			const layers = discoverLayers(model);
			expect(layers.size).toBe(0);
		});
	});

	describe('toggleLayerVisibility', () => {
		it('should set visibility of all meshes in a layer', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			
			const mesh1 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh1.userData.layerType = 'building';
			mesh1.visible = true;
			
			const mesh2 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh2.userData.layerType = 'building';
			mesh2.visible = true;
			
			buildingParent.add(mesh1);
			buildingParent.add(mesh2);
			model.add(buildingParent);

			// Discover layers first
			initializeLayerManager(model);

			// Toggle to hidden
			toggleLayerVisibility('building', false);

			expect(mesh1.visible).toBe(false);
			expect(mesh2.visible).toBe(false);

			// Toggle to visible
			toggleLayerVisibility('building', true);

			expect(mesh1.visible).toBe(true);
			expect(mesh2.visible).toBe(true);
		});

		it('should only affect meshes of the specified layer type', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			buildingMesh.visible = true;

			const vegetationParent = new THREE.Group();
			vegetationParent.name = 'Trees';
			const vegetationMesh = new THREE.Mesh(new THREE.SphereGeometry(1));
			vegetationMesh.userData.layerType = 'vegetation';
			vegetationMesh.visible = true;

			buildingParent.add(buildingMesh);
			vegetationParent.add(vegetationMesh);
			model.add(buildingParent);
			model.add(vegetationParent);

			initializeLayerManager(model);

			// Hide buildings only
			toggleLayerVisibility('building', false);

			expect(buildingMesh.visible).toBe(false);
			expect(vegetationMesh.visible).toBe(true);
		});

		it('should handle layer type that does not exist', () => {
			initializeLayerManager(model);

			// Should not throw error
			expect(() => toggleLayerVisibility('nonexistent', false)).not.toThrow();
		});
	});

	describe('getDiscoveredLayers', () => {
		it('should return array of discovered layer types', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			buildingParent.add(buildingMesh);

			const vegetationParent = new THREE.Group();
			vegetationParent.name = 'Trees';
			const vegetationMesh = new THREE.Mesh(new THREE.SphereGeometry(1));
			vegetationMesh.userData.layerType = 'vegetation';
			vegetationParent.add(vegetationMesh);

			model.add(buildingParent);
			model.add(vegetationParent);

			initializeLayerManager(model);

			const layers = getDiscoveredLayers();
			expect(layers).toContain('building');
			expect(layers).toContain('vegetation');
			expect(layers.length).toBe(2);
		});

		it('should return empty array if no layers discovered', () => {
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			expect(layers).toEqual([]);
		});
	});

	describe('initializeLayerManager', () => {
		it('should initialize layer manager with discovered layers', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			initializeLayerManager(model);

			const layers = getDiscoveredLayers();
			expect(layers).toContain('building');
		});

		it('should clear previous layers when reinitializing', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingMesh.userData.layerType = 'building';
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			initializeLayerManager(model);
			expect(getDiscoveredLayers()).toContain('building');

			// Reinitialize with empty model
			const emptyModel = new THREE.Group();
			initializeLayerManager(emptyModel);

			const layers = getDiscoveredLayers();
			expect(layers).not.toContain('building');
			expect(layers.length).toBe(0);
		});
	});
});

