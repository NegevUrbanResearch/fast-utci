import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
	getLayerName,
	mapLayerNameToType,
	applyLayerMaterials,
	mergeLayerMeshes
} from '$lib/services/modelLoaderService';
import { LAYER_NAME_MAPPING } from '$lib/types/layerMaterials';

describe('Model Loader Service', () => {
	describe('getLayerName', () => {
		it('should return layer name from parent chain', () => {
			const parent = new THREE.Group();
			parent.name = 'Building';
			const mesh = new THREE.Mesh();
			parent.add(mesh);

			const layerName = getLayerName(mesh);
			expect(layerName).toBe('Building');
		});

		it('should traverse multiple parent levels', () => {
			const grandParent = new THREE.Group();
			grandParent.name = 'Scene';
			const parent = new THREE.Group();
			parent.name = 'Vegetation';
			const mesh = new THREE.Mesh();
			grandParent.add(parent);
			parent.add(mesh);

			const layerName = getLayerName(mesh);
			expect(layerName).toBe('Vegetation');
		});

		it('should skip generic GLTF names', () => {
			const parent = new THREE.Group();
			parent.name = 'GLTF_123';
			const mesh = new THREE.Mesh();
			parent.add(mesh);

			const layerName = getLayerName(mesh);
			expect(layerName).toBe('unknown');
		});

		it('should skip numeric-only names', () => {
			const parent = new THREE.Group();
			parent.name = '123';
			const mesh = new THREE.Mesh();
			parent.add(mesh);

			const layerName = getLayerName(mesh);
			expect(layerName).toBe('unknown');
		});

		it('should skip "Scene" root name', () => {
			const parent = new THREE.Group();
			parent.name = 'Scene';
			const mesh = new THREE.Mesh();
			parent.add(mesh);

			const layerName = getLayerName(mesh);
			expect(layerName).toBe('unknown');
		});

		it('should skip generic "Layer_XX" names', () => {
			const parent = new THREE.Group();
			parent.name = 'Layer_01';
			const mesh = new THREE.Mesh();
			parent.add(mesh);

			const layerName = getLayerName(mesh);
			expect(layerName).toBe('unknown');
		});

		it('should return "unknown" if no valid layer name found', () => {
			const mesh = new THREE.Mesh();
			// No parent
			const layerName = getLayerName(mesh);
			expect(layerName).toBe('unknown');
		});

		it('should stop at max 20 levels to prevent infinite loops', () => {
			// Create a deep chain of 25 parents, all with generic names except first valid
			let current: THREE.Object3D = new THREE.Mesh();
			const mesh = current as THREE.Mesh;
			
			// Create chain where first valid name is at level 5
			for (let i = 0; i < 25; i++) {
				const parent = new THREE.Group();
				parent.name = i === 5 ? 'ValidLayer' : i < 5 ? 'GLTF_123' : 'Intermediate';
				current.parent = parent;
				parent.add(current);
				current = parent;
			}

			const layerName = getLayerName(mesh);
			// Should find "ValidLayer" (the first valid name encountered)
			expect(layerName).toBe('ValidLayer');
			
			// Test with all invalid names - should stop at 20 levels and return 'unknown'
			let current2: THREE.Object3D = new THREE.Mesh();
			const mesh2 = current2 as THREE.Mesh;
			for (let i = 0; i < 25; i++) {
				const parent = new THREE.Group();
				parent.name = 'GLTF_123'; // All invalid
				current2.parent = parent;
				parent.add(current2);
				current2 = parent;
			}
			
			const layerName2 = getLayerName(mesh2);
			expect(layerName2).toBe('unknown');
		});
	});

	describe('mapLayerNameToType', () => {
		it('should map known layer names using LAYER_NAME_MAPPING', () => {
			expect(mapLayerNameToType('building')).toBe('building');
			expect(mapLayerNameToType('buildings')).toBe('building');
			expect(mapLayerNameToType('vegetation')).toBe('vegetation');
			expect(mapLayerNameToType('trees')).toBe('vegetation');
		});

		it('should handle case-insensitive matching', () => {
			expect(mapLayerNameToType('Building')).toBe('building');
			expect(mapLayerNameToType('VEGETATION')).toBe('vegetation');
			expect(mapLayerNameToType('New Building')).toBe('new_building');
		});

		it('should handle substring matching for variants', () => {
			expect(mapLayerNameToType('new building')).toBe('new_building');
			expect(mapLayerNameToType('proposed trees')).toBe('new_vegetation');
		});

		it('should return "unknown" for unmapped layer names', () => {
			expect(mapLayerNameToType('unknown_layer')).toBe('unknown');
			expect(mapLayerNameToType('custom_type')).toBe('unknown');
		});

		it('should return "unknown" for empty string', () => {
			expect(mapLayerNameToType('')).toBe('unknown');
		});
	});

	describe('applyLayerMaterials', () => {
		let model: THREE.Group;

		beforeEach(() => {
			model = new THREE.Group();
		});

		it('should apply materials to meshes based on layer type', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			applyLayerMaterials(model);

			expect(buildingMesh.material).toBeInstanceOf(THREE.Material);
			expect(buildingMesh.userData.layerType).toBeDefined();
		});

	it('should mark meshes with unknown layer names as "unknown" (remapped to base)', () => {
		const unknownParent = new THREE.Group();
		unknownParent.name = 'UnknownLayer';
		const unknownMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
		unknownParent.add(unknownMesh);
		model.add(unknownParent);

		applyLayerMaterials(model);

		// Unknown layers are now remapped to 'base' when no base layer exists (Issue #5)
		expect(unknownMesh.userData.layerType).toBe('base');
	});

		it('should store layer name in userData', () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			applyLayerMaterials(model);

			expect(buildingMesh.userData.layerName).toBe('Building');
			expect(buildingMesh.userData.layerType).toBe('building');
		});

		it('should remove line objects that are not building edges', () => {
			const lineGeometry = new THREE.BufferGeometry();
			const line = new THREE.Line(lineGeometry);
			line.name = 'SomeLine';
			model.add(line);

			applyLayerMaterials(model);

			expect(model.children).not.toContain(line);
		});

		it('should keep building edge lines', () => {
			const lineGeometry = new THREE.BufferGeometry();
			const line = new THREE.Line(lineGeometry);
			line.name = 'building_edges';
			model.add(line);

			applyLayerMaterials(model);

			expect(model.children).toContain(line);
		});

		it('should handle models with multiple layer types', () => {
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

			applyLayerMaterials(model);

			expect(buildingMesh.userData.layerType).toBe('building');
			expect(vegetationMesh.userData.layerType).toBe('vegetation');
		});
	});

	describe('mergeLayerMeshes', () => {
		it('should merge multiple meshes of the same layer type', () => {
			const model = new THREE.Group();
			const meshes: THREE.Mesh[] = [];

			// Create 3 building meshes
			for (let i = 0; i < 3; i++) {
				const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
				mesh.userData.layerType = 'building';
				meshes.push(mesh);
				model.add(mesh);
			}

			mergeLayerMeshes(model, 'building', meshes);

			// Should have one merged mesh instead of 3
			const buildingMeshes = model.children.filter(
				(child) => child.userData.layerType === 'building'
			);
			expect(buildingMeshes.length).toBe(1);
		});

		it('should preserve layer type in userData after merging', () => {
			const model = new THREE.Group();
			const meshes: THREE.Mesh[] = [];

			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh.userData.layerType = 'building';
			meshes.push(mesh);
			model.add(mesh);

			mergeLayerMeshes(model, 'building', meshes);

			const mergedMesh = model.children.find(
				(child) => child.userData.layerType === 'building'
			);
			expect(mergedMesh?.userData.layerType).toBe('building');
		});

		it('should add building edges for building layer type', () => {
			const model = new THREE.Group();
			const meshes: THREE.Mesh[] = [];

			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh.userData.layerType = 'building';
			meshes.push(mesh);
			model.add(mesh);

			mergeLayerMeshes(model, 'building', meshes);

			const mergedMesh = model.children.find(
				(child) => child.userData.layerType === 'building'
			) as THREE.Mesh;

			// Should have edges as a child
			const edges = mergedMesh.children.find((child) =>
				child.name.includes('_edges')
			);
			expect(edges).toBeDefined();
			expect(edges).toBeInstanceOf(THREE.LineSegments);
		});

		it('should set vegetation to not cast shadows', () => {
			const model = new THREE.Group();
			const meshes: THREE.Mesh[] = [];

			const mesh = new THREE.Mesh(new THREE.SphereGeometry(1));
			mesh.userData.layerType = 'vegetation';
			meshes.push(mesh);
			model.add(mesh);

			mergeLayerMeshes(model, 'vegetation', meshes);

			const mergedMesh = model.children.find(
				(child) => child.userData.layerType === 'vegetation'
			) as THREE.Mesh;

			expect(mergedMesh.castShadow).toBe(false);
			expect(mergedMesh.receiveShadow).toBe(true);
		});

		it('should set other layers to cast shadows', () => {
			const model = new THREE.Group();
			const meshes: THREE.Mesh[] = [];

			const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			mesh.userData.layerType = 'building';
			meshes.push(mesh);
			model.add(mesh);

			mergeLayerMeshes(model, 'building', meshes);

			const mergedMesh = model.children.find(
				(child) => child.userData.layerType === 'building'
			) as THREE.Mesh;

			expect(mergedMesh.castShadow).toBe(true);
			expect(mergedMesh.receiveShadow).toBe(true);
		});
	});
});

