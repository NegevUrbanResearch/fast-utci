<script lang="ts" context="module">
	// Keep loader URLs unique across component instances.
	let globalModelVersion = 0;

	function nextGlobalModelVersion(): number {
		globalModelVersion++;
		return globalModelVersion;
	}
</script>

<script lang="ts">
	import { processModelScene } from '$lib/services/modelSceneProcessor';
	import { GLTF } from '@threlte/extras';
	import { applyModelCoordinateTransform } from '$lib/utils/coordinates';
	import { createEventDispatcher, onDestroy } from 'svelte';
	import type { Group } from 'three';
	import type { AnalysisMetadata } from '$lib/types/analysis';
	import type { ThrelteGltf } from '@threlte/extras';
	import { resetLayerManager } from '$lib/services/layerManagerService';
	import { getCachedModel } from '$lib/services/modelCacheService';
	import { setError, setLoading } from '$lib/stores/viewerStore';
	import { T } from '@threlte/core';
	import * as THREE from 'three';

	export let modelPath: string;
	export let coordinateSystem: 'xy_ground' | 'xz_ground' = 'xy_ground';
	export let metadata: AnalysisMetadata | undefined = undefined;
	export let isRequestCurrent: () => boolean = () => true;

	const dispatch = createEventDispatcher<{
		modelLoaded: Group;
		layersDiscovered: string[];
	}>();

	let gltfGroup: Group | undefined;
	let useCache = false;
	
	// Loading sequence ID to prevent race conditions
	// When modelPath changes, we increment this to invalidate any in-flight loads
	let loadSequenceId = 0;
	
	// Local cache-busting version for this component's current model
	// Gets assigned from global counter when modelPath changes
	// Only used when not using cache
	let currentModelVersion = 0;

	async function processScene(scene: Group, fromCache = false, sequenceId = loadSequenceId) {
		const snapshot = { modelPath, coordinateSystem, metadata };
		const ownsSelection = isRequestCurrent;
		const isCurrent = () => sequenceId === loadSequenceId && ownsSelection();
		const processed = await processModelScene(scene, fromCache, snapshot, {
			modelLoaded: (scene) => dispatch('modelLoaded', scene),
			layersDiscovered: (layers) => dispatch('layersDiscovered', layers)
		}, isCurrent, (error) => {
			console.error('[MODEL] Failed to process model:', error);
			setError(error instanceof Error ? error.message : 'Failed to process model');
			setLoading(false);
		});
		if (isCurrent() && processed) gltfGroup = processed;
	}

	function handleLoad(gltf: ThrelteGltf, sequenceId: number) {
		if (sequenceId !== loadSequenceId || !isRequestCurrent()) return;
		void processScene(gltf.scene, false, sequenceId);
	}
	
	// React to modelPath changes - check cache first, then load if needed
	$: modelPath, (() => {
		loadSequenceId++;
		
		// Check cache first
		const cached = getCachedModel(modelPath);
		if (cached) {
			console.log(`[MODEL] Using cached model: ${modelPath}`);
			useCache = true;
			// Process cached scene (async for API consistency; no heavy merge when from cache)
			void processScene(cached.scene, true);
		} else {
			// Not in cache, load via GLTF
			console.log(`[MODEL] Loading model from file: ${modelPath}`);
			useCache = false;
			currentModelVersion = nextGlobalModelVersion();
		}
	})()
	
	// Create wrapper function that captures current sequence ID
	// This ensures we can detect stale loads
	function createLoadHandler(currentSequenceId: number) {
		return (gltf: ThrelteGltf) => handleLoad(gltf, currentSequenceId);
	}
	
	// Reactive load handler that updates when sequence changes
	$: onloadHandler = createLoadHandler(loadSequenceId);

	// Apply coordinate transformation only when coordinateSystem *changes* (not when gltfGroup is first set).
	// This prevents double-apply: processScene applies once after merge; we re-apply here only if user changes coord system.
	let previousCoordinateSystem: typeof coordinateSystem | undefined = undefined;
	let prevGltfGroup: Group | undefined = undefined;
	$: if (gltfGroup) {
		if (prevGltfGroup !== gltfGroup) {
			prevGltfGroup = gltfGroup;
			previousCoordinateSystem = coordinateSystem;
		} else if (coordinateSystem !== previousCoordinateSystem) {
			applyModelCoordinateTransform(gltfGroup, coordinateSystem);
			previousCoordinateSystem = coordinateSystem;
		}
	}

	// Cleanup on destroy
	onDestroy(() => {
		loadSequenceId++;
		// Threlte's GLTF component handles disposal automatically
		// But we should reset the layer manager to clear any stale references
		resetLayerManager();
	});
</script>

<!-- When cached, render the cloned scene directly; otherwise load via GLTF -->
{#if useCache && gltfGroup}
	<T is={THREE.Group} oncreate={(ref) => {
		ref.add(gltfGroup!);
	}}>
		<slot />
	</T>
{:else}
	<!-- Add cache-busting version parameter to force reload on model change -->
	<GLTF url={`${modelPath}?v=${currentModelVersion}`} onload={onloadHandler}>
		<slot />
	</GLTF>
{/if}
