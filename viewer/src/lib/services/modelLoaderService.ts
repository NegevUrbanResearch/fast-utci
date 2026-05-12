/**
 * Model Loader Service
 * 
 * Loads GLTF models and applies materials based on layer names from the scene graph.
 * Uses Three.js utilities for geometry merging and edge generation.
 * No geometric detection fallback - marks as 'unknown' if no layer name found.
 */

import * as THREE from 'three';

function yieldToMain(): Promise<void> {
	return new Promise((resolve) => {
		if (typeof requestAnimationFrame !== 'undefined') {
			requestAnimationFrame(() => resolve());
		} else {
			setTimeout(() => resolve(), 0);
		}
	});
}
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { getMaterial } from './materialPool';
import { LAYER_NAME_MAPPING } from '$lib/types/layerMaterials';

/** Above this triangle count per layer we skip merge on the main thread to avoid freeze/crash. */
const MAX_TRIANGLES_PER_LAYER_DISPLAY = 500_000;
/** When layer is over this count we merge in batches with yields between batches. */
const BATCH_MERGE_THRESHOLD_TRIANGLES = 50_000;
const BATCH_TRIANGLE_TARGET = 50_000;
const BATCH_MESH_MAX = 20;

function countTrianglesForMeshes(meshes: THREE.Mesh[]): number {
	let total = 0;
	for (const mesh of meshes) {
		const geom = mesh.geometry;
		if (!geom) continue;
		const idx = geom.getIndex();
		const pos = geom.getAttribute('position');
		if (idx) total += idx.count / 3;
		else if (pos) total += pos.count / 3;
	}
	return total;
}

/**
 * Get layer name from scene graph by traversing parent chain
 * 
 * @param mesh - Mesh object to find layer for
 * @returns Layer name from GLB scene graph, or 'unknown' if not found
 */
export function getLayerName(mesh: THREE.Mesh): string {
	let current: THREE.Object3D = mesh;
	
	// Traverse up parent chain (max 20 levels to prevent infinite loops)
	for (let i = 0; i < 20; i++) {
		if (!current.parent) break;
		
		current = current.parent;
		const name = current.name;
		
		// Check if this is a layer node (meaningful name)
		if (name && 
			!name.match(/^\d+$/) &&              // Not just digits
			!name.startsWith('GLTF') &&          // Not auto-generated GLTF name
			!name.startsWith('Layer_') &&        // Ignore generic Layer_XX names
			name !== 'Scene' &&                  // Not root scene
			name !== '') {
			return name;  // Found the layer name!
		}
	}
	
	return 'unknown';
}

/**
 * Map actual GLB layer name to standard material type
 * 
 * @param layerName - Layer name from scene graph
 * @returns Standard layer type (building, road, vegetation, etc.) or 'unknown'
 */
export function mapLayerNameToType(layerName: string): string {
	if (!layerName || layerName === '') {
		return 'unknown';
	}

	const nameLower = layerName.toLowerCase();
	
	// Try exact match first
	if (LAYER_NAME_MAPPING[nameLower]) {
		return LAYER_NAME_MAPPING[nameLower];
	}
	
	// Fallback to substring matching
	for (const [key, type] of Object.entries(LAYER_NAME_MAPPING)) {
		if (nameLower.includes(key)) {
			return type;
		}
	}
	
	return 'unknown';
}

/**
 * Apply layer materials to loaded model
 * 
 * Traverses scene graph to find layer names and applies appropriate materials.
 * Also groups meshes by layer type for performance optimization.
 * 
 * @param model - Loaded GLTF model
 * @returns Model with applied materials (Promise so we can yield between layer merges and avoid main-thread freeze).
 */
export async function applyLayerMaterials(model: THREE.Group): Promise<THREE.Group> {
	await yieldToMain();

	const meshesByLayer = new Map<string, THREE.Mesh[]>();  // Group meshes by layer type for merging
	const layerStats: Record<string, { count: number; layerName: string }> = {};
	const itemsToRemove: THREE.Object3D[] = [];  // Track non-mesh items to remove
	const finalMeshesByLayer = new Map<string, THREE.Mesh[]>();  // Track final meshes (merged or single)

	model.traverse((child) => {
		// Remove lines/curves that aren't needed (not building edges we add)
		if (child instanceof THREE.Line || child instanceof THREE.LineSegments) {
			if (!child.name.includes('_edges')) {
				itemsToRemove.push(child);
			}
			return;
		}
		
		if (child instanceof THREE.Mesh) {
			// Extract layer name from scene graph
			const layerName = getLayerName(child);
			const layerType = mapLayerNameToType(layerName);
			
			// Apply material based on layer type (from pool)
			child.material = getMaterial(layerType);
			
			// Store layer info in userData for later use (visibility toggles, etc.)
			child.userData.layerType = layerType;
			child.userData.layerName = layerName;
			
			// Track layer statistics
			if (!layerStats[layerType]) {
				layerStats[layerType] = { count: 0, layerName: layerName };
			}
			layerStats[layerType].count++;
			
			// Group meshes by layer for merging
			if (!meshesByLayer.has(layerType)) {
				meshesByLayer.set(layerType, []);
			}
			meshesByLayer.get(layerType)!.push(child);
			
			// Shadow settings (all layers cast/receive shadows initially)
			child.castShadow = true;
			child.receiveShadow = true;
		}
	});

	await yieldToMain();

	// Remove unwanted lines/curves
	itemsToRemove.forEach(item => {
		if (item.parent) {
			item.parent.remove(item);
		}
	});
	if (itemsToRemove.length > 0) {
		console.log(`[FILTER] Removed ${itemsToRemove.length} line/curve objects`);
	}
	
	// Print layer summary
	console.log('[LAYERS] Discovered layers:');
	for (const [layerType, stats] of Object.entries(layerStats)) {
		console.log(`  ${layerType}: ${stats.count} meshes (from '${stats.layerName}')`);
	}
	
	// Handle 'unknown' layers - remap to 'base' if appropriate
	if (layerStats['unknown'] && !layerStats['base']) {
		console.log('[LAYERS] Remapping unknown layer to base (ground layer)');
		
		// Update all 'unknown' meshes to 'base' type
		const unknownMeshes = meshesByLayer.get('unknown');
		if (unknownMeshes) {
			unknownMeshes.forEach(mesh => {
				mesh.userData.layerType = 'base';
				mesh.material = getMaterial('base');
			});
			
			// Move meshes from 'unknown' to 'base' in the map
			meshesByLayer.set('base', unknownMeshes);
			meshesByLayer.delete('unknown');
			
			// Update stats
			layerStats['base'] = layerStats['unknown'];
			delete layerStats['unknown'];
		}
	}

	const onlyBaseLayer = Object.keys(layerStats).length === 1 && layerStats['base'];
	
	// Merge geometries by layer type for massive performance improvement.
	// Yield between layers so the main thread stays responsive (avoids freeze/crash on large models).
	// Skip merge for layers over the cap to prevent one huge sync merge on the main thread.
	console.log('[PERF] Merging geometries by layer type...');
	for (const [layerType, meshes] of meshesByLayer.entries()) {
		if (meshes.length > 1) {
			const layerTriangles = countTrianglesForMeshes(meshes);
			if (layerTriangles > MAX_TRIANGLES_PER_LAYER_DISPLAY) {
				console.warn(
					`[PERF] Skipping merge for ${layerType} (${(layerTriangles / 1e6).toFixed(2)}M triangles > ${MAX_TRIANGLES_PER_LAYER_DISPLAY / 1e6}M cap)`
				);
				finalMeshesByLayer.set(layerType, meshes);
			} else {
				const mergedMesh =
					layerTriangles > BATCH_MERGE_THRESHOLD_TRIANGLES
						? await mergeLayerMeshesBatched(model, layerType, meshes)
						: mergeLayerMeshes(model, layerType, meshes);
				if (mergedMesh) {
					finalMeshesByLayer.set(layerType, [mergedMesh]);
				}
			}
			await yieldToMain();
		} else {
			// Single mesh - keep it
			finalMeshesByLayer.set(layerType, meshes);
		}
	}

	// If the model only has a base layer, make it fully visible
	if (onlyBaseLayer) {
		const baseMaterial = getMaterial('base').clone();
		(baseMaterial as THREE.Material & { opacity?: number; transparent?: boolean; depthWrite?: boolean; polygonOffset?: boolean }).opacity = 1;
		(baseMaterial as THREE.Material & { opacity?: number; transparent?: boolean; depthWrite?: boolean; polygonOffset?: boolean }).transparent = false;
		(baseMaterial as THREE.Material & { opacity?: number; transparent?: boolean; depthWrite?: boolean; polygonOffset?: boolean }).depthWrite = true;
		(baseMaterial as THREE.Material & { opacity?: number; transparent?: boolean; depthWrite?: boolean; polygonOffset?: boolean }).polygonOffset = false;

		model.traverse((child) => {
			if (child instanceof THREE.Mesh && child.userData.layerType === 'base') {
				child.material = baseMaterial;
			}
		});
	}
	
	return model;
}

/**
 * Merge layer meshes into a single geometry for massive performance improvement
 * 
 * Uses Three.js BufferGeometryUtils.mergeGeometries() to merge geometries.
 * For buildings, also adds edges using Three.js EdgesGeometry.
 * 
 * @param model - Model to add merged mesh to
 * @param layerType - Type of layer being merged
 * @param meshes - Array of meshes to merge
 * @returns The merged mesh (for layer manager tracking)
 */
export function mergeLayerMeshes(
	model: THREE.Group,
	layerType: string,
	meshes: THREE.Mesh[]
): THREE.Mesh | null {
	const geometries: THREE.BufferGeometry[] = [];
	
	console.log(`[PERF] Merging ${meshes.length} ${layerType} meshes...`);
	
	// Collect all geometries with world transforms
	meshes.forEach(mesh => {
		const geometry = mesh.geometry.clone();
		mesh.updateWorldMatrix(true, false);
		geometry.applyMatrix4(mesh.matrixWorld);

		// Ensure normals exist on each source geometry so that the merged
		// result has a stable "normal" attribute for lit materials / TSL.
		if (!geometry.getAttribute('normal')) {
			geometry.computeVertexNormals();
		}

		geometries.push(geometry);
		
		// Remove original mesh from scene
		if (mesh.parent) {
			mesh.parent.remove(mesh);
		}
	});
	
	// Merge all geometries into one using Three.js BufferGeometryUtils
	const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
	
	if (merged) {
		// Some GLBs may not provide normals on every mesh; after merging we
		// guarantee a "normal" attribute for all lighting-aware materials.
		if (!merged.getAttribute('normal')) {
			merged.computeVertexNormals();
		}

		// Create single mesh for this layer (using material from pool)
		const material = getMaterial(layerType);
		const mergedMesh = new THREE.Mesh(merged, material);
		mergedMesh.name = `${layerType}_merged`;
		mergedMesh.userData.layerType = layerType;
		mergedMesh.userData.layerName = layerType;
		
		// Shadow settings based on layer type
		if (layerType === 'vegetation') {
			mergedMesh.castShadow = false;   // Vegetation doesn't cast shadows (performance)
			mergedMesh.receiveShadow = true;
		} else {
			mergedMesh.castShadow = true;
			mergedMesh.receiveShadow = true;
		}
		
		model.add(mergedMesh);
		
		const vertexCount = (merged.attributes.position.count / 1000).toFixed(1);
		console.log(`[PERF] ${layerType}: ${meshes.length} meshes -> 1 mesh (${vertexCount}k vertices)`);
		
		// Add building edges after merging for better performance
		// Uses Three.js EdgesGeometry utility
		// Applied to both 'building' and 'new_building' layers
		if (layerType === 'building' || layerType === 'new_building') {
			const edges = new THREE.EdgesGeometry(merged, 15);  // 15 degree threshold
			const lineMaterial = new THREE.LineBasicMaterial({ 
				color: 0x888888,  // Medium gray
				linewidth: 1
			});
			const lineSegments = new THREE.LineSegments(edges, lineMaterial);
			lineSegments.name = `${layerType}_edges`;
			mergedMesh.add(lineSegments);
			console.log(`[PERF] Added ${layerType} edges to merged geometry`);
		}
		
		return mergedMesh;
	} else {
		console.warn(`[PERF] Failed to merge ${layerType} geometry`);
		return null;
	}
}

/**
 * Merge a large layer in batches with yields between batches to keep the main thread responsive.
 * Used when layer triangle count is over BATCH_MERGE_THRESHOLD_TRIANGLES but under the display cap.
 */
async function mergeLayerMeshesBatched(
	model: THREE.Group,
	layerType: string,
	meshes: THREE.Mesh[]
): Promise<THREE.Mesh | null> {
	function meshTriCount(m: THREE.Mesh): number {
		const g = m.geometry;
		if (!g) return 0;
		const idx = g.getIndex();
		const pos = g.getAttribute('position');
		return idx ? idx.count / 3 : pos ? pos.count / 3 : 0;
	}

	const batches: THREE.Mesh[][] = [];
	let current: THREE.Mesh[] = [];
	let currentTri = 0;

	for (const mesh of meshes) {
		const tri = meshTriCount(mesh);
		if (current.length >= BATCH_MESH_MAX || (currentTri + tri > BATCH_TRIANGLE_TARGET && current.length > 0)) {
			batches.push(current);
			current = [];
			currentTri = 0;
		}
		current.push(mesh);
		currentTri += tri;
	}
	if (current.length > 0) batches.push(current);

	const batchGeometries: THREE.BufferGeometry[] = [];

	for (const batch of batches) {
		const geometries: THREE.BufferGeometry[] = [];
		for (const mesh of batch) {
			const geometry = mesh.geometry.clone();
			mesh.updateWorldMatrix(true, false);
			geometry.applyMatrix4(mesh.matrixWorld);
			if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
			geometries.push(geometry);
			if (mesh.parent) mesh.parent.remove(mesh);
		}
		const mergedBatch = BufferGeometryUtils.mergeGeometries(geometries, false);
		if (mergedBatch) batchGeometries.push(mergedBatch);
		await yieldToMain();
	}

	if (batchGeometries.length === 0) return null;
	const merged = batchGeometries.length === 1 ? batchGeometries[0] : BufferGeometryUtils.mergeGeometries(batchGeometries, false);
	if (!merged) return null;

	if (!merged.getAttribute('normal')) merged.computeVertexNormals();

	const material = getMaterial(layerType);
	const mergedMesh = new THREE.Mesh(merged, material);
	mergedMesh.name = `${layerType}_merged`;
	mergedMesh.userData.layerType = layerType;
	mergedMesh.userData.layerName = layerType;

	if (layerType === 'vegetation') {
		mergedMesh.castShadow = false;
		mergedMesh.receiveShadow = true;
	} else {
		mergedMesh.castShadow = true;
		mergedMesh.receiveShadow = true;
	}

	model.add(mergedMesh);

	const vertexCount = (merged.attributes.position.count / 1000).toFixed(1);
	console.log(`[PERF] ${layerType}: ${meshes.length} meshes -> 1 mesh (${vertexCount}k vertices, batched)`);

	if (layerType === 'building' || layerType === 'new_building') {
		const edges = new THREE.EdgesGeometry(merged, 15);
		const lineMaterial = new THREE.LineBasicMaterial({ color: 0x888888, linewidth: 1 });
		const lineSegments = new THREE.LineSegments(edges, lineMaterial);
		lineSegments.name = `${layerType}_edges`;
		mergedMesh.add(lineSegments);
	}

	return mergedMesh;
}
