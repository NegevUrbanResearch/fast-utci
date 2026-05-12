/**
 * Layer Manager Service
 * 
 * Service to track meshes by layer type and toggle visibility efficiently.
 * Discovers layers from model scene graph and manages visibility state.
 */

import * as THREE from 'three';

// Singleton layer map: layerType -> Mesh[]
let layerMap = new Map<string, THREE.Mesh[]>();

// Callback to trigger scene re-render (set by Threlte component)
let invalidateCallback: (() => void) | null = null;

/**
 * Set the invalidate callback from Threlte
 * This is called from the Scene component to enable forced re-renders
 * @param callback - Function to call when scene needs re-rendering
 */
export function setInvalidateCallback(callback: (() => void) | null): void {
	invalidateCallback = callback;
}

/**
 * Discover which layer types exist in the loaded model
 * 
 * Scans the model for meshes with userData.layerType and groups them by type.
 * This should be called AFTER geometry merging to track the merged meshes.
 * 
 * @param model - Loaded model to scan
 * @returns Map of layer type to array of meshes
 */
export function discoverLayers(model: THREE.Group): Map<string, THREE.Mesh[]> {
	const layers = new Map<string, THREE.Mesh[]>();
	
	model.traverse((child) => {
		// Only track meshes that have layerType in userData
		// Skip line segments (edges) - they're children of merged meshes
		if (child instanceof THREE.Mesh && child.userData.layerType) {
			const layerType = child.userData.layerType as string;
			
			// Skip edge lines (they're not the actual layer meshes)
			if (child.name.includes('_edges')) {
				return;
			}
			
			if (!layers.has(layerType)) {
				layers.set(layerType, []);
			}
			layers.get(layerType)!.push(child);
		}
	});
	
	console.log(`[LAYER MANAGER] Discovered ${layers.size} layer types with ${Array.from(layers.values()).reduce((sum, arr) => sum + arr.length, 0)} total meshes`);
	
	return layers;
}

/**
 * Initialize layer manager with discovered layers from model
 * 
 * @param model - Loaded model to discover layers from
 */
export function initializeLayerManager(model: THREE.Group): void {
	layerMap = discoverLayers(model);
	
	const layerTypes = Array.from(layerMap.keys());
	console.log(`[LAYER MANAGER] Initialized with ${layerTypes.length} layer types: ${layerTypes.join(', ')}`);
}

/**
 * Toggle visibility of all meshes in a layer type
 * 
 * This also handles edge lines that are children of merged meshes.
 * 
 * @param layerType - Layer type to toggle
 * @param visible - New visibility state
 */
export function toggleLayerVisibility(layerType: string, visible: boolean): void {
	const meshes = layerMap.get(layerType);
	if (!meshes || meshes.length === 0) {
		console.warn(`[LAYER MANAGER] Layer type '${layerType}' not found in layerMap. Available layers: ${Array.from(layerMap.keys()).join(', ')}`);
		return;
	}
	
	let actualCount = 0;
	meshes.forEach((mesh) => {
		// Set visibility on the mesh itself
		mesh.visible = visible;
		actualCount++;
		
		// Also handle edge lines if they exist (children of merged building meshes)
		if (mesh.children.length > 0) {
			mesh.children.forEach((child) => {
				if (child instanceof THREE.LineSegments && child.name.includes('_edges')) {
					child.visible = visible;
				}
			});
		}
	});
	
	console.log(`[LAYER MANAGER] ${visible ? 'Show' : 'Hide'} ${layerType}: ${actualCount} meshes`);
	
	// Trigger Threlte to re-render the scene
	if (invalidateCallback) {
		invalidateCallback();
	}
}

/**
 * Get all discovered layer types
 * 
 * @returns Array of layer type IDs
 */
export function getDiscoveredLayers(): string[] {
	return Array.from(layerMap.keys());
}

/**
 * Get meshes for a specific layer type
 * 
 * @param layerType - Layer type to get meshes for
 * @returns Array of meshes for this layer type, or empty array if not found
 */
export function getLayerMeshes(layerType: string): THREE.Mesh[] {
	return layerMap.get(layerType) || [];
}

/**
 * Check if a layer type exists in the model
 * 
 * @param layerType - Layer type to check
 * @returns True if layer type exists
 */
export function hasLayer(layerType: string): boolean {
	return layerMap.has(layerType);
}

/**
 * Get mesh count for a layer type
 * 
 * @param layerType - Layer type
 * @returns Number of meshes in this layer type
 */
export function getMeshCount(layerType: string): number {
	const meshes = layerMap.get(layerType);
	return meshes ? meshes.length : 0;
}

/**
 * Apply visibility state to all layers based on provided state map
 * This should be called after initialization to sync meshes with store state
 * 
 * @param visibilityState - Map of layer type to visibility boolean
 */
export function applyLayerVisibilityState(visibilityState: Record<string, boolean>): void {
	console.log('[LAYER MANAGER] Applying initial visibility state');
	
	// Apply visibility for each layer type in the map
	layerMap.forEach((meshes, layerType) => {
        const visible = visibilityState[layerType] ?? false; // Default to hidden if not specified
		toggleLayerVisibility(layerType, visible);
	});
}

/**
 * Reset layer manager (clear all tracked meshes)
 * This should be called when switching models to prevent stale references
 */
export function resetLayerManager(): void {
	layerMap.clear();
	console.log('[LAYER MANAGER] Reset - all layers cleared');
}

