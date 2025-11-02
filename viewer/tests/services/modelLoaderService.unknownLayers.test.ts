import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import { applyLayerMaterials } from '$lib/services/modelLoaderService';
import { initializeLayerManager, getDiscoveredLayers } from '$lib/services/layerManagerService';

describe('Model Loader Service - Unknown Layer Handling', () => {
	let model: THREE.Group;

	beforeEach(() => {
		model = new THREE.Group();
	});

	describe('Unknown layer remapping', () => {
		it('should remap unknown layer to base when no base layer exists', () => {
			// Create model with only unknown layer
			const unknownParent = new THREE.Group();
			unknownParent.name = 'SomeUnknownLayer';
			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			unknownParent.add(mesh);
			model.add(unknownParent);

			// Process the model
			applyLayerMaterials(model);

			// Verify the mesh was remapped to 'base'
			mesh.traverse((child) => {
				if (child.isMesh) {
					const meshChild = child as THREE.Mesh;
					expect(meshChild.userData.layerType).toBe('base');
				}
			});

			// Verify layer manager sees it as 'base'
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			expect(layers).toContain('base');
			expect(layers).not.toContain('unknown');
		});

		it('should keep unknown layer when base layer already exists', () => {
			// Create model with both unknown and base layers
			const baseParent = new THREE.Group();
			baseParent.name = 'ground';
			const baseMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			baseParent.add(baseMesh);

			const unknownParent = new THREE.Group();
			unknownParent.name = 'SomeUnknownLayer';
			const unknownMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			unknownParent.add(unknownMesh);

			model.add(baseParent);
			model.add(unknownParent);

			// Process the model
			applyLayerMaterials(model);

			// Verify base mesh stays as base
			baseMesh.traverse((child) => {
				if (child.isMesh) {
					const meshChild = child as THREE.Mesh;
					expect(meshChild.userData.layerType).toBe('base');
				}
			});

			// Verify unknown mesh stays as unknown (not remapped)
			unknownMesh.traverse((child) => {
				if (child.isMesh) {
					const meshChild = child as THREE.Mesh;
					expect(meshChild.userData.layerType).toBe('unknown');
				}
			});

			// Verify layer manager sees both layers
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			expect(layers).toContain('base');
			expect(layers).toContain('unknown');
		});

		it('should remap multiple unknown meshes to base', () => {
			// Create model with multiple meshes in unknown layer
			const unknownParent = new THREE.Group();
			unknownParent.name = 'UnknownLayerGroup';
			
			const mesh1 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			const mesh2 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			const mesh3 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			
			unknownParent.add(mesh1);
			unknownParent.add(mesh2);
			unknownParent.add(mesh3);
			model.add(unknownParent);

			// Process the model
			applyLayerMaterials(model);

			// Verify all meshes were remapped to 'base'
			let baseMeshCount = 0;
			model.traverse((child) => {
				if (child.isMesh) {
					const mesh = child as THREE.Mesh;
					expect(mesh.userData.layerType).toBe('base');
					baseMeshCount++;
				}
			});
			
			// Should have merged into 1 mesh
			expect(baseMeshCount).toBe(1);

			// Verify layer manager sees it as 'base'
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			expect(layers).toContain('base');
			expect(layers).not.toContain('unknown');
		});

		it('should handle model with no unknown layers', () => {
			// Create model with only standard layers
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingParent.add(buildingMesh);

			const vegetationParent = new THREE.Group();
			vegetationParent.name = 'Trees';
			const vegetationMesh = new THREE.Mesh(new THREE.SphereGeometry(1));
			vegetationParent.add(vegetationMesh);

			model.add(buildingParent);
			model.add(vegetationParent);

			// Process the model
			applyLayerMaterials(model);

			// Verify layers are correctly identified
			buildingMesh.traverse((child) => {
				if (child.isMesh) {
					expect((child as THREE.Mesh).userData.layerType).toBe('building');
				}
			});

			vegetationMesh.traverse((child) => {
				if (child.isMesh) {
					expect((child as THREE.Mesh).userData.layerType).toBe('vegetation');
				}
			});

			// Verify layer manager sees both layers (no 'unknown')
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			expect(layers).toContain('building');
			expect(layers).toContain('vegetation');
			expect(layers).not.toContain('unknown');
		});

		it('should apply correct material when remapping unknown to base', () => {
			// Create model with unknown layer
			const unknownParent = new THREE.Group();
			unknownParent.name = 'mystery_layer';
			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			unknownParent.add(mesh);
			model.add(unknownParent);

			// Process the model
			applyLayerMaterials(model);

			// Verify the mesh has base material properties
			mesh.traverse((child) => {
				if (child.isMesh) {
					const meshChild = child as THREE.Mesh;
					expect(meshChild.material).toBeDefined();
					// Material should be the base layer material (not unknown material)
					expect(meshChild.userData.layerType).toBe('base');
				}
			});
		});
	});

	describe('Complex scenarios', () => {
		it('should handle model with unknown, base, and other standard layers', () => {
			// Create a complex model
			const baseParent = new THREE.Group();
			baseParent.name = 'ground';
			const baseMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			baseParent.add(baseMesh);

			const unknownParent = new THREE.Group();
			unknownParent.name = 'mystery_object';
			const unknownMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			unknownParent.add(unknownMesh);

			const buildingParent = new THREE.Group();
			buildingParent.name = 'building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingParent.add(buildingMesh);

			model.add(baseParent);
			model.add(unknownParent);
			model.add(buildingParent);

			// Process the model
			applyLayerMaterials(model);

			// Verify all layers are correctly identified
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			
			expect(layers).toContain('base');
			expect(layers).toContain('unknown');
			expect(layers).toContain('building');
			expect(layers.length).toBe(3);
		});

		it('should handle empty model gracefully', () => {
			// Process empty model
			applyLayerMaterials(model);

			// Verify no layers are discovered
			initializeLayerManager(model);
			const layers = getDiscoveredLayers();
			expect(layers.length).toBe(0);
		});
	});
});

