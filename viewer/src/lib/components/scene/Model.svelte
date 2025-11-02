<script lang="ts" context="module">
	// CRITICAL: Global cache-busting counter that persists across component instances
	// This ensures each model load gets a truly unique URL, preventing Threlte from
	// serving a cached (and mutated) version when we return to a previously loaded model
	let globalModelVersion = 0;
</script>

<script lang="ts">
	import { GLTF } from '@threlte/extras';
	import { applyCoordinateTransform } from '$lib/utils/coordinates';
	import { createEventDispatcher, onDestroy } from 'svelte';
	import type { Group } from 'three';
	import type { AnalysisMetadata } from '$lib/types/analysis';
	import type { ThrelteGltf } from '@threlte/extras';
	import { applyLayerMaterials } from '$lib/services/modelLoaderService';
	import {
		initializeLayerManager,
		getDiscoveredLayers,
		resetLayerManager,
		applyLayerVisibilityState
	} from '$lib/services/layerManagerService';
	import { layerStore } from '$lib/stores/layerStore';
	import { get } from 'svelte/store';

	export let modelPath: string;
	export let coordinateSystem: 'xy_ground' | 'xz_ground' = 'xy_ground';
	// Metadata is available but not directly used in rendering - kept for future use
	export const metadata: AnalysisMetadata | undefined = undefined;

	const dispatch = createEventDispatcher<{
		modelLoaded: Group;
		layersDiscovered: string[];
	}>();

	let gltfGroup: Group | undefined;
	let lastModelPath: string | undefined;
	
	// Loading sequence ID to prevent race conditions
	// When modelPath changes, we increment this to invalidate any in-flight loads
	let loadSequenceId = 0;
	
	// Local cache-busting version for this component's current model
	// Gets assigned from global counter when modelPath changes
	let currentModelVersion = 0;

    // Handle model loading
    function handleLoad(gltf: ThrelteGltf, sequenceId: number) {
        // Check if this load is still valid (not superseded by a newer load)
        if (sequenceId !== loadSequenceId) {
            console.log(`[MODEL] Ignoring stale load (sequence ${sequenceId} vs current ${loadSequenceId})`);
            return;
        }
        
        // Always reset layer manager when loading a new model (even if path seems same)
        // This ensures clean state when switching scenarios
        resetLayerManager();
        
        gltfGroup = gltf.scene;
        lastModelPath = modelPath;
		
		// Apply layer materials FIRST (before coordinate transform)
		// This discovers layers, applies materials, and merges geometries
		applyLayerMaterials(gltfGroup);
		
		// IMPORTANT: Initialize layer manager AFTER merging is complete
		// This ensures we track the merged meshes, not the original ones
		initializeLayerManager(gltfGroup);
		
		// CRITICAL: Apply current layer visibility state from store to meshes
		// This ensures default-hidden layers (roads, sidewalks, etc.) are actually hidden
		const currentLayerState = get(layerStore);
		applyLayerVisibilityState(currentLayerState);
		
		// Get discovered layer types and dispatch event
		const discoveredLayers = getDiscoveredLayers();
		console.log(`[MODEL] Discovered ${discoveredLayers.length} layers: ${discoveredLayers.join(', ')}`);
		dispatch('layersDiscovered', discoveredLayers);
		
		// Apply coordinate transform AFTER materials (to preserve layer info)
		if (gltfGroup && coordinateSystem) {
			applyCoordinateTransform(gltfGroup, coordinateSystem);
		}
		
		dispatch('modelLoaded', gltfGroup);
	}
	
	// React to modelPath changes - increment sequence to invalidate old loads
	// eslint-disable-next-line @typescript-eslint/no-unused-expressions
	$: modelPath, (() => {
		// When modelPath changes, increment sequence ID and get next version from global counter
		// - loadSequenceId: invalidates any in-flight loads from previous model
		// - currentModelVersion: gets unique version number from global counter (persists across instances)
		// Note: globalModelVersion mutation intentional - we want a persistent counter across component instances
		loadSequenceId++;
		// @ts-expect-error - Svelte warns about module var mutation but it's intentional here
		globalModelVersion++;
		currentModelVersion = globalModelVersion;
		console.log(`[MODEL] Model path changed to: ${modelPath} (sequence: ${loadSequenceId}, version: ${currentModelVersion})`);
	})()
	
	// Create wrapper function that captures current sequence ID
	// This ensures we can detect stale loads
	function createLoadHandler() {
		const currentSequenceId = loadSequenceId;
		return (gltf: ThrelteGltf) => handleLoad(gltf, currentSequenceId);
	}
	
	// Reactive load handler that updates when sequence changes
	$: onloadHandler = createLoadHandler();

	// Apply coordinate transformation when coordinate system changes
	$: if (gltfGroup && coordinateSystem) {
		applyCoordinateTransform(gltfGroup, coordinateSystem);
	}

	// Cleanup on destroy
	onDestroy(() => {
		// Threlte's GLTF component handles disposal automatically
		// But we should reset the layer manager to clear any stale references
		resetLayerManager();
	});
</script>

<!-- Add cache-busting version parameter to force reload on model change -->
<GLTF url={`${modelPath}?v=${currentModelVersion}`} onload={onloadHandler}>
	<slot />
</GLTF>

