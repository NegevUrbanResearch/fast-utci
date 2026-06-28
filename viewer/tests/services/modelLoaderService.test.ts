import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
	getLayerName,
	mapLayerNameToType,
	applyLayerMaterials,
	mergeLayerMeshes,
	resolveComputeBvhEligibility
} from '$lib/services/modelLoaderService';
import { discoverLayers } from '$lib/services/layerManagerService';
import { prepareMeshPayloadForWorker } from '$lib/compute/gpu/mergeAndBvhWorkerClient';
import { LAYER_NAME_MAPPING } from '$lib/types/layerMaterials';

function addLayerMesh(
	model: THREE.Group,
	layerName: string,
	geometry: THREE.BufferGeometry = new THREE.BoxGeometry(1, 1, 1)
): THREE.Mesh {
	const parent = new THREE.Group();
	parent.name = layerName;
	const mesh = new THREE.Mesh(geometry);
	parent.add(mesh);
	model.add(parent);
	return mesh;
}

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
			expect(mapLayerNameToType('existing_buildings')).toBe('building');
			expect(mapLayerNameToType('vegetation')).toBe('vegetation');
			expect(mapLayerNameToType('trees')).toBe('vegetation');
			expect(mapLayerNameToType('trees_canopy')).toBe('vegetation');
			expect(mapLayerNameToType('trees_camopy')).toBe('vegetation');
			expect(mapLayerNameToType('tree_canopy')).toBe('vegetation');
			expect(mapLayerNameToType('street')).toBe('road');
			expect(mapLayerNameToType('train track')).toBe('train_track');
			expect(mapLayerNameToType('train tracks')).toBe('train_track');
			expect(mapLayerNameToType('train_tracks')).toBe('train_track');
			expect(mapLayerNameToType('district_outline')).toBe('ignored');
			expect(mapLayerNameToType('trees_point')).toBe('ignored');
			expect(mapLayerNameToType('tree_point')).toBe('ignored');
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

	describe('resolveComputeBvhEligibility', () => {
		it('separates Innovation District visual ground-family layers from compute occluders', () => {
			expect(resolveComputeBvhEligibility({ rawLayerName: 'existing_buildings', layerType: 'building' })).toBe(true);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'trees_canopy', layerType: 'vegetation' })).toBe(true);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'street', layerType: 'road' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'road', layerType: 'road' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'roads', layerType: 'road' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'highway', layerType: 'road' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'train track', layerType: 'train_track' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'train tracks', layerType: 'train_track' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'train_tracks', layerType: 'train_track' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'ground', layerType: 'base' })).toBe(false);
			expect(resolveComputeBvhEligibility({ rawLayerName: 'legacy_unknown', layerType: 'base' })).toBeUndefined();
		});
	});

	describe('applyLayerMaterials', () => {
		let model: THREE.Group;

		beforeEach(() => {
			model = new THREE.Group();
		});

		it('should apply materials to meshes based on layer type', async () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			await applyLayerMaterials(model);

			expect(buildingMesh.material).toBeInstanceOf(THREE.Material);
			expect(buildingMesh.userData.layerType).toBeDefined();
		});

	it('should mark meshes with unknown layer names as "unknown" (remapped to base)', async () => {
		const unknownParent = new THREE.Group();
		unknownParent.name = 'UnknownLayer';
		const unknownMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
		unknownParent.add(unknownMesh);
		model.add(unknownParent);

		await applyLayerMaterials(model);

		// Unknown layers are now remapped to 'base' when no base layer exists (Issue #5)
		expect(unknownMesh.userData.layerType).toBe('base');
	});

		it('should store layer name in userData', async () => {
			const buildingParent = new THREE.Group();
			buildingParent.name = 'Building';
			const buildingMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
			buildingParent.add(buildingMesh);
			model.add(buildingParent);

			await applyLayerMaterials(model);

			expect(buildingMesh.userData.layerName).toBe('Building');
			expect(buildingMesh.userData.layerType).toBe('building');
		});

		it('should remove line objects that are not building edges', async () => {
			const lineGeometry = new THREE.BufferGeometry();
			const line = new THREE.Line(lineGeometry);
			line.name = 'SomeLine';
			model.add(line);

			await applyLayerMaterials(model);

			expect(model.children).not.toContain(line);
		});

		it('should keep building edge lines', async () => {
			const lineGeometry = new THREE.BufferGeometry();
			const line = new THREE.Line(lineGeometry);
			line.name = 'building_edges';
			model.add(line);

			await applyLayerMaterials(model);

			expect(model.children).toContain(line);
		});

		it('should handle models with multiple layer types', async () => {
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

			await applyLayerMaterials(model);

			expect(buildingMesh.userData.layerType).toBe('building');
			expect(vegetationMesh.userData.layerType).toBe('vegetation');
		});

		it('stores compute BVH eligibility only for explicit layer metadata decisions', async () => {
			const buildingMesh = addLayerMesh(model, 'existing_buildings');
			const treeCanopyMesh = addLayerMesh(model, 'trees_canopy');
			const streetMesh = addLayerMesh(model, 'street');
			const trainTrackMesh = addLayerMesh(model, 'train_tracks');
			const groundMesh = addLayerMesh(model, 'ground');
			const legacyUnknownRemappedToBase = addLayerMesh(model, 'legacy_unknown');

			await applyLayerMaterials(model);

			expect(buildingMesh.userData.includeInComputeBvh).toBe(true);
			expect(treeCanopyMesh.userData.includeInComputeBvh).toBe(true);
			expect(streetMesh.userData.includeInComputeBvh).toBe(false);
			expect(trainTrackMesh.userData.includeInComputeBvh).toBe(false);
			expect(groundMesh.userData.includeInComputeBvh).toBe(false);
			expect(legacyUnknownRemappedToBase.userData.includeInComputeBvh).toBeUndefined();
		});

		it('renders merged roads as an outline-only context layer above UTCI', async () => {
			addLayerMesh(model, 'street', new THREE.PlaneGeometry(10, 4));
			addLayerMesh(model, 'street', new THREE.PlaneGeometry(6, 3));

			await applyLayerMaterials(model);

			const mergedRoad = model.children.find((child) => child.userData.layerType === 'road') as THREE.Mesh;
			const roadMaterial = mergedRoad.material as THREE.MeshStandardMaterial;
			const roadOutline = mergedRoad.children.find((child) => child.name === 'road_outline') as THREE.LineSegments;
			const outlineMaterial = roadOutline.material as THREE.LineBasicMaterial;

			expect(mergedRoad.name).toBe('road_merged');
			expect(mergedRoad.userData.includeInComputeBvh).toBe(false);
			expect(mergedRoad.renderOrder).toBeGreaterThan(2);
			expect(roadMaterial.opacity).toBe(0);
			expect(roadMaterial.transparent).toBe(true);
			expect(roadMaterial.depthWrite).toBe(false);
			expect(roadOutline).toBeInstanceOf(THREE.LineSegments);
			expect(roadOutline.renderOrder).toBe(mergedRoad.renderOrder);
			expect(outlineMaterial.color.getHexString()).toBe('f5f7fa');
			expect(outlineMaterial.toneMapped).toBe(false);
			expect(outlineMaterial.depthWrite).toBe(false);
		});

		it('keeps train tracks as geometry rendered above UTCI instead of converting them to outlines', async () => {
			addLayerMesh(model, 'train_tracks', new THREE.PlaneGeometry(10, 1));
			addLayerMesh(model, 'train_tracks', new THREE.PlaneGeometry(8, 1));

			await applyLayerMaterials(model);

			const mergedTracks = model.children.find((child) => child.userData.layerType === 'train_track') as THREE.Mesh;
			const trackMaterial = mergedTracks.material as THREE.MeshStandardMaterial;

			expect(mergedTracks.name).toBe('train_track_merged');
			expect(mergedTracks.userData.includeInComputeBvh).toBe(false);
			expect(mergedTracks.renderOrder).toBeGreaterThan(2);
			expect(trackMaterial.opacity).toBeGreaterThan(0);
			expect(trackMaterial.depthWrite).toBe(true);
			expect(
				mergedTracks.children.some((child) => child.name === 'road_outline')
			).toBe(false);
		});

		it('preserves compute BVH eligibility through processed scene clones', async () => {
			addLayerMesh(model, 'existing_buildings');
			addLayerMesh(model, 'trees_canopy');
			addLayerMesh(model, 'street');
			addLayerMesh(model, 'train_tracks');
			addLayerMesh(model, 'ground');

			await applyLayerMaterials(model);
			const cloned = model.clone(true);
			const eligibilityByLayer = new Map<string, unknown>();
			cloned.traverse((child) => {
				if (child instanceof THREE.Mesh && child.userData.layerType) {
					eligibilityByLayer.set(child.userData.layerType, child.userData.includeInComputeBvh);
				}
			});

			expect(eligibilityByLayer.get('building')).toBe(true);
			expect(eligibilityByLayer.get('vegetation')).toBe(true);
			expect(eligibilityByLayer.get('road')).toBe(false);
			expect(eligibilityByLayer.get('train_track')).toBe(false);
			expect(eligibilityByLayer.get('base')).toBe(false);
		});

		it('removes ignored district outline primitives before layer discovery and compute payloads', async () => {
			const outlineParent = new THREE.Group();
			outlineParent.name = 'district_outline';
			const lineGeometry = new THREE.BufferGeometry().setFromPoints([
				new THREE.Vector3(0, 0, 0),
				new THREE.Vector3(10, 0, 0)
			]);
			const outlineLine = new THREE.Line(lineGeometry);
			outlineParent.add(outlineLine);
			model.add(outlineParent);
			addLayerMesh(model, 'existing_buildings');

			await applyLayerMaterials(model);

			expect(outlineParent.children).not.toContain(outlineLine);
			const discoveredLayers = discoverLayers(model);
			expect(discoveredLayers.has('ignored')).toBe(false);
			const payload = prepareMeshPayloadForWorker(model);
			expect(payload.meshes).toHaveLength(1);
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

		it('preserves explicit compute BVH eligibility when merging meshes', () => {
			const model = new THREE.Group();
			const meshes = [new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1))];
			meshes[0].userData.includeInComputeBvh = false;
			meshes[1].userData.includeInComputeBvh = false;
			meshes.forEach((mesh) => model.add(mesh));

			mergeLayerMeshes(model, 'road', meshes);

			const mergedMesh = model.children.find((child) => child.userData.layerType === 'road') as THREE.Mesh;
			expect(mergedMesh.userData.includeInComputeBvh).toBe(false);
		});

		it('leaves merged compute BVH eligibility undefined when every source mesh is missing metadata', () => {
			const model = new THREE.Group();
			const meshes = [new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1))];
			meshes.forEach((mesh) => model.add(mesh));

			mergeLayerMeshes(model, 'building', meshes);

			const mergedMesh = model.children.find((child) => child.userData.layerType === 'building') as THREE.Mesh;
			expect(mergedMesh.userData.includeInComputeBvh).toBeUndefined();
		});

		it('keeps merged compute BVH eligibility true when any source mesh is eligible', () => {
			const model = new THREE.Group();
			const meshes = [new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1))];
			meshes[0].userData.includeInComputeBvh = false;
			meshes[1].userData.includeInComputeBvh = true;
			meshes.forEach((mesh) => model.add(mesh));

			mergeLayerMeshes(model, 'building', meshes);

			const mergedMesh = model.children.find((child) => child.userData.layerType === 'building') as THREE.Mesh;
			expect(mergedMesh.userData.includeInComputeBvh).toBe(true);
		});

		it('preserves compute BVH eligibility through batched layer merges in applyLayerMaterials', async () => {
			const model = new THREE.Group();
			const makeGeometry = () => new THREE.PlaneGeometry(1, 1, 100, 100);

			addLayerMesh(model, 'existing_buildings', makeGeometry());
			addLayerMesh(model, 'existing_buildings', makeGeometry());
			addLayerMesh(model, 'existing_buildings', makeGeometry());
			addLayerMesh(model, 'street', makeGeometry());
			addLayerMesh(model, 'street', makeGeometry());
			addLayerMesh(model, 'street', makeGeometry());
			addLayerMesh(model, 'legacy_unknown', makeGeometry());
			addLayerMesh(model, 'legacy_unknown', makeGeometry());
			addLayerMesh(model, 'legacy_unknown', makeGeometry());

			await applyLayerMaterials(model);

			const mergedBuilding = model.children.find((child) => child.userData.layerType === 'building') as THREE.Mesh;
			const mergedRoad = model.children.find((child) => child.userData.layerType === 'road') as THREE.Mesh;
			const mergedBase = model.children.find((child) => child.userData.layerType === 'base') as THREE.Mesh;

			expect(mergedBuilding.name).toBe('building_merged');
			expect(mergedBuilding.userData.includeInComputeBvh).toBe(true);
			expect(mergedRoad.name).toBe('road_merged');
			expect(mergedRoad.userData.includeInComputeBvh).toBe(false);
			expect(mergedBase.name).toBe('base_merged');
			expect(mergedBase.userData.includeInComputeBvh).toBeUndefined();
		});
	});
});
