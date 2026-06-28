/**
 * Layer Store
 * 
 * Svelte store for managing 3D model layer visibility state
 */

import { writable, derived, type Writable } from 'svelte/store';
import type { StandardLayerType } from '$lib/types/layers';
import { STANDARD_LAYER_TYPES } from '$lib/types/layerMaterials';
import { toggleLayerVisibility } from '$lib/services/layerManagerService';

/**
 * Layer visibility state - maps layer ID to visibility boolean
 */
export interface LayerVisibility {
	[layerId: string]: boolean;
}

/**
 * Initialize layer visibility from standard layer types
 */
function initializeLayerVisibility(): LayerVisibility {
	const visibility: LayerVisibility = {};
	STANDARD_LAYER_TYPES.forEach((layer) => {
		visibility[layer.id] = layer.defaultVisible;
	});
	return visibility;
}

/**
 * Layer store - holds layer visibility state
 */
export const layerStore: Writable<LayerVisibility> = writable<LayerVisibility>(
	initializeLayerVisibility()
);

/**
 * Discovered layers store - holds array of layer types that exist in the model
 */
export const discoveredLayersStore: Writable<string[]> = writable<string[]>([]);

/**
 * Set discovered layers from model
 * @param layerTypes - Array of layer type IDs discovered in the model
 */
export function setDiscoveredLayers(layerTypes: string[]): void {
	discoveredLayersStore.set(layerTypes);
	const onlyBaseLayer =
		layerTypes.length === 1 &&
		(layerTypes[0] === 'base' || layerTypes[0] === 'unknown');
	
	// Initialize visibility for discovered layers (use defaults from STANDARD_LAYER_TYPES)
	let nextState: LayerVisibility | null = null;
	layerStore.update((state) => {
		const newState = { ...state };
		layerTypes.forEach((layerId) => {
			const standardLayer = STANDARD_LAYER_TYPES.find((l) => l.id === layerId);
			if (standardLayer && !(layerId in newState)) {
				newState[layerId] = standardLayer.defaultVisible;
			}
		});
		if (onlyBaseLayer) {
			newState[layerTypes[0]] = true;
		}
		nextState = newState;
		return newState;
	});

	// Apply visibility to meshes now that layers are known
	if (nextState) {
		layerTypes.forEach((layerId) => {
			toggleLayerVisibility(layerId, nextState?.[layerId] ?? false);
		});
	}
}

/**
 * Toggle layer visibility
 * This now connects to layerManagerService to actually toggle mesh visibility
 * @param layerId - Layer identifier
 */
export function toggleLayer(layerId: string): void {
	let newVisible = false;
	layerStore.update((state) => {
		newVisible = !state[layerId];
		return {
			...state,
			[layerId]: newVisible
		};
	});
	
	// Toggle actual mesh visibility using layerManagerService AFTER store update
	// This ensures the store state is updated first, then the meshes
	// Three.js automatically handles rendering updates when mesh.visible changes
	toggleLayerVisibility(layerId, newVisible);
}

/**
 * Set layer visibility
 * @param layerId - Layer identifier
 * @param visible - Visibility state
 */
export function setLayerVisible(layerId: string, visible: boolean): void {
	layerStore.update((state) => ({
		...state,
		[layerId]: visible
	}));
	
	// Also set actual mesh visibility using layerManagerService
	// Three.js automatically handles rendering updates when mesh.visible changes
	toggleLayerVisibility(layerId, visible);
}

/**
 * Get layer visibility
 * @param layerId - Layer identifier
 * @returns Visibility state (defaults to false if layer not found)
 */
export function getLayerVisible(layerId: string): boolean {
	let visible = false;
	layerStore.subscribe((state) => {
		visible = state[layerId] ?? false;
	})();
	return visible;
}

/**
 * Reset layer visibility to defaults
 */
export function resetLayerVisibility(): void {
	layerStore.set(initializeLayerVisibility());
}
