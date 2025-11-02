/**
 * Model Loader Service
 * 
 * Loads GLTF models and applies materials based on layer names from the scene graph.
 * Uses Three.js utilities for geometry merging and edge generation.
 * No geometric detection fallback - marks as 'unknown' if no layer name found.
 */

import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { getMaterial } from './materialPool';
import { LAYER_NAME_MAPPING } from '$lib/types/layerMaterials';

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
 * @returns Model with applied materials and array of final meshes (for layer manager)
 */
export function applyLayerMaterials(model: THREE.Group): THREE.Group {
	const meshesByLayer = new Map<string, THREE.Mesh[]>();  // Group meshes by layer type for merging
	const layerStats: Record<string, { count: number; layerName: string }> = {};
	const itemsToRemove: THREE.Object3D[] = [];  // Track non-mesh items to remove
	const finalMeshesByLayer = new Map<string, THREE.Mesh[]>();  // Track final meshes (merged or single)
	
	model.traverse((child) => {
		// Remove lines/curves that aren't needed (not building edges we add)
		if (child.isLine || child.isLineSegments) {
			if (!child.name.includes('_edges')) {
				itemsToRemove.push(child);
			}
			return;
		}
		
		if (child.isMesh) {
			const mesh = child as THREE.Mesh;
			
			// Extract layer name from scene graph
			const layerName = getLayerName(mesh);
			const layerType = mapLayerNameToType(layerName);
			
			// Apply material based on layer type (from pool)
			mesh.material = getMaterial(layerType);
			
			// Store layer info in userData for later use (visibility toggles, etc.)
			mesh.userData.layerType = layerType;
			mesh.userData.layerName = layerName;
			
			// Track layer statistics
			if (!layerStats[layerType]) {
				layerStats[layerType] = { count: 0, layerName: layerName };
			}
			layerStats[layerType].count++;
			
			// Group meshes by layer for merging
			if (!meshesByLayer.has(layerType)) {
				meshesByLayer.set(layerType, []);
			}
			meshesByLayer.get(layerType)!.push(mesh);
			
			// Shadow settings (all layers cast/receive shadows initially)
			mesh.castShadow = true;
			mesh.receiveShadow = true;
		}
	});
	
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
	
	// Merge geometries by layer type for massive performance improvement
	console.log('[PERF] Merging geometries by layer type...');
	for (const [layerType, meshes] of meshesByLayer.entries()) {
		if (meshes.length > 1) {
			const mergedMesh = mergeLayerMeshes(model, layerType, meshes);
			if (mergedMesh) {
				finalMeshesByLayer.set(layerType, [mergedMesh]);
			}
		} else {
			// Single mesh - keep it
			finalMeshesByLayer.set(layerType, meshes);
		}
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
		geometries.push(geometry);
		
		// Remove original mesh from scene
		if (mesh.parent) {
			mesh.parent.remove(mesh);
		}
	});
	
	// Merge all geometries into one using Three.js BufferGeometryUtils
	const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
	
	if (merged) {
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

