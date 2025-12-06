<script lang="ts" context="module">
	// CRITICAL: Global cache-busting counter that persists across component instances
	// This ensures each model load gets a truly unique URL, preventing Threlte from
	// serving a cached (and mutated) version when we return to a previously loaded model
	let globalModelVersion = 0;
</script>

<script lang="ts">
	import { GLTF } from '@threlte/extras';
	import { applyCoordinateTransform, calculateScenarioOrigin, applyModelOffset } from '$lib/utils/coordinates';
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
	import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
	import { getCachedModel, cacheModel, hasModelInCache } from '$lib/services/modelCacheService';
	import * as THREE from 'three';

	export let modelPath: string;
	export let coordinateSystem: 'xy_ground' | 'xz_ground' = 'xy_ground';
	export let metadata: AnalysisMetadata | undefined = undefined;

	const dispatch = createEventDispatcher<{
		modelLoaded: Group;
		layersDiscovered: string[];
	}>();

	let gltfGroup: Group | undefined;
	let lastModelPath: string | undefined;
	let useCache = false;
	let cachedScene: Group | undefined;
	
	// Loading sequence ID to prevent race conditions
	// When modelPath changes, we increment this to invalidate any in-flight loads
	let loadSequenceId = 0;
	
	// Local cache-busting version for this component's current model
	// Gets assigned from global counter when modelPath changes
	// Only used when not using cache
	let currentModelVersion = 0;

	// Process a loaded scene (from cache or GLTF loader)
	function processScene(scene: Group, fromCache: boolean = false) {
		// Always reset layer manager when loading a new model (even if path seems same)
		// This ensures clean state when switching scenarios
		resetLayerManager();
		
		// If from cache, the scene is already processed (materials, coordinate transform applied)
		// Just clone it and apply normalization offset
		if (fromCache) {
			gltfGroup = scene.clone(true);
			// Clone materials to avoid sharing references
			gltfGroup.traverse((child) => {
				if (child instanceof THREE.Mesh && child.material) {
					if (Array.isArray(child.material)) {
						child.material = child.material.map(mat => mat.clone());
					} else {
						child.material = child.material.clone();
					}
				}
			});
			
			// Re-initialize layer manager for the cloned scene
			initializeLayerManager(gltfGroup);
			
			// Apply current layer visibility state
			const currentLayerState = get(layerStore);
			applyLayerVisibilityState(currentLayerState);
			
			// Get discovered layer types
			const discoveredLayers = getDiscoveredLayers();
			console.log(`[MODEL] Discovered ${discoveredLayers.length} layers: ${discoveredLayers.join(', ')}`);
			dispatch('layersDiscovered', discoveredLayers);
		} else {
			// Not from cache - process the raw scene
			gltfGroup = scene;
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
			
			// Cache the processed scene (before normalization)
			// This allows us to apply scenario-specific normalization offsets later
			if (!hasModelInCache(modelPath)) {
				// Clone the processed scene for caching
				const sceneToCache = gltfGroup.clone(true);
				// Clone materials for cached version
				sceneToCache.traverse((child) => {
					if (child instanceof THREE.Mesh && child.material) {
						if (Array.isArray(child.material)) {
							child.material = child.material.map(mat => mat.clone());
						} else {
							child.material = child.material.clone();
						}
					}
				});
				cacheModel(modelPath, sceneToCache);
			}
		}
		
		// Apply normalization offset if enabled
		// This must be applied each time since different scenarios may have different offsets
		if (gltfGroup && isNormalizationEnabled() && metadata) {
			const scenarioOrigin = calculateScenarioOrigin(metadata);
			const anchorOffset = getAnchorOffset();
			
			// Transform scenario origin to world space to match the coordinate system
			// For xy_ground: rotation around X by -90° transforms (x, y, z) → (x, z, -y)
			let transformedOrigin: THREE.Vector3;
			if (coordinateSystem === 'xy_ground') {
				// Transform origin to world space: (x, y, z) → (x, z, -y)
				transformedOrigin = new THREE.Vector3(scenarioOrigin.x, scenarioOrigin.z, -scenarioOrigin.y);
			} else {
				transformedOrigin = scenarioOrigin.clone();
			}
			
			// Calculate offset in world space (where anchorOffset already is)
			const offset = anchorOffset.clone().sub(transformedOrigin);
			
			if (offset.lengthSq() > 0.001) { // Only apply if offset is significant
				console.log(`[MODEL] Applying normalization offset:`, offset);
				applyModelOffset(gltfGroup, offset);
			}
		}
		
		dispatch('modelLoaded', gltfGroup);
	}

    // Handle model loading from GLTF loader
    function handleLoad(gltf: ThrelteGltf, sequenceId: number) {
        // Check if this load is still valid (not superseded by a newer load)
        if (sequenceId !== loadSequenceId) {
            console.log(`[MODEL] Ignoring stale load (sequence ${sequenceId} vs current ${loadSequenceId})`);
            return;
        }
        
        processScene(gltf.scene, false);
	}
	
	// React to modelPath changes - check cache first, then load if needed
	// eslint-disable-next-line @typescript-eslint/no-unused-expressions
	$: modelPath, (() => {
		loadSequenceId++;
		
		// Check cache first
		const cached = getCachedModel(modelPath);
		if (cached) {
			console.log(`[MODEL] Using cached model: ${modelPath}`);
			useCache = true;
			cachedScene = cached.scene;
			// Process cached scene
			processScene(cached.scene, true);
		} else {
			// Not in cache, load via GLTF
			console.log(`[MODEL] Loading model from file: ${modelPath}`);
			useCache = false;
			cachedScene = undefined;
			// @ts-expect-error - Svelte warns about module var mutation but it's intentional here
			globalModelVersion++;
			currentModelVersion = globalModelVersion;
		}
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

<!-- Only render GLTF component if not using cache -->
{#if !useCache}
	<!-- Add cache-busting version parameter to force reload on model change -->
	<GLTF url={`${modelPath}?v=${currentModelVersion}`} onload={onloadHandler}>
		<slot />
	</GLTF>
{/if}
