<script lang="ts">
	/**
	 * ComparisonRenderer Component
	 *
	 * ABOUTME: Implements scissor-test dual rendering for comparing base and comparison scenes.
	 * The base scene renders on the left side (0 to curtain position) and the comparison
	 * scene renders on the right side (curtain position to end). Camera and layers are synced.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { useThrelte } from '@threlte/core';
	import { T } from '@threlte/core';
	import { comparisonStore, curtainPosition, comparisonAnalysis, unifiedUtciRange } from '$lib/stores/comparisonStore';
	import { cameraStore } from '$lib/stores/cameraStore';
	import { layerStore, discoveredLayersStore, setDiscoveredLayers } from '$lib/stores/layerStore';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { get } from 'svelte/store';
	import { base } from '$app/paths';
	import {
		applyLayerMaterials,
		mapLayerNameToType
	} from '$lib/services/modelLoaderService';
	import {
		applyLayerVisibilityState,
		discoverLayers
	} from '$lib/services/layerManagerService';
	import {
		getCachedModel,
		cacheModel,
		hasModelInCache
	} from '$lib/services/modelCacheService';
	import {
		createUtciSurfaceMesh,
		updateUtciSurfaceTexture
	} from '$lib/services/pointCloudService';
	import { applyCoordinateTransform, calculateScenarioOrigin, applyModelOffset } from '$lib/utils/coordinates';
	import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
	import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
	import * as THREE from 'three';
	import type { Group, PerspectiveCamera, Mesh, MeshBasicMaterial } from 'three';

	// Props
	export let baseCamera: PerspectiveCamera | undefined = undefined;

	const { renderer, scene, invalidate } = useThrelte();

	// Comparison scene (separate from base scene)
	let comparisonScene: THREE.Scene | null = null;
	let comparisonModel: Group | null = null;
	let comparisonCamera: PerspectiveCamera | null = null;

	// Layer map for comparison model
	let comparisonLayerMap = new Map<string, THREE.Mesh[]>();

	// Track layers from comparison model for merging with base layers
	let comparisonLayerTypes: string[] = [];

	// UTCI surface mesh for comparison scene
	let comparisonUtciMesh: Mesh | null = null;

	/**
	 * Get the comparison UTCI mesh for external use (e.g., tooltip raycasting)
	 */
	export function getComparisonUtciMesh(): Mesh | null {
		return comparisonUtciMesh;
	}

	// Loading state
	let isLoading = false;
	let loadError: string | null = null;

	// Cached model version for cache busting
	let modelVersion = 0;

	// Track which analysis we've loaded to avoid reloading
	let loadedAnalysisId: string | null = null;

	// Track base model layers before comparison started (for restoration)
	let baseLayerTypesSnapshot: string[] = [];

	// GLTF loader
	const gltfLoader = new GLTFLoader();

	/**
	 * Add lights to comparison scene matching the base scene's Lights component
	 */
	function addLightsToScene(targetScene: THREE.Scene): void {
		// Match the default values from Lights.svelte
		const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
		const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
		directionalLight.position.set(100, 200, 100);
		
		targetScene.add(ambientLight);
		targetScene.add(directionalLight);
	}

	/**
	 * Merge comparison layers with base layers in the store
	 */
	function mergeComparisonLayers(comparisonLayers: string[]): void {
		// Snapshot current base layers before merging (only on first comparison load)
		if (baseLayerTypesSnapshot.length === 0) {
			baseLayerTypesSnapshot = [...get(discoveredLayersStore)];
		}
		
		// Merge: union of base layers and comparison layers
		const merged = [...new Set([...baseLayerTypesSnapshot, ...comparisonLayers])];
		setDiscoveredLayers(merged);
		
		console.log(`[COMPARISON RENDERER] Merged layers - base: ${baseLayerTypesSnapshot.length}, comparison: ${comparisonLayers.length}, merged: ${merged.length}`);
	}

	/**
	 * Restore base-only layers when comparison ends
	 */
	function restoreBaseOnlyLayers(): void {
		if (baseLayerTypesSnapshot.length > 0) {
			setDiscoveredLayers(baseLayerTypesSnapshot);
			console.log(`[COMPARISON RENDERER] Restored ${baseLayerTypesSnapshot.length} base layers`);
			baseLayerTypesSnapshot = [];
		}
	}

	/**
	 * Create UTCI surface mesh for comparison analysis
	 */
	function createComparisonUtciMesh(analysis: typeof $comparisonAnalysis): void {
		if (!analysis || !comparisonScene) return;

		// Dispose existing mesh first
		disposeComparisonUtciMesh();

		// Get current viewer state for UTCI visualization
		const currentViewerState = get(viewerStore);
		const hourIndex = currentViewerState.currentHour ?? 0;
		const colorMode = currentViewerState.colorMode ?? 'normalized';
		const metricType = currentViewerState.metricType ?? 'utci';
		
		// Use unified range for consistent color mapping between base and comparison
		const rangeOverride = get(unifiedUtciRange) ?? undefined;

		try {
			comparisonUtciMesh = createUtciSurfaceMesh(analysis, hourIndex, colorMode, metricType, rangeOverride);
			comparisonScene.add(comparisonUtciMesh);
			
			// Apply visibility based on viewer state
			comparisonUtciMesh.visible = currentViewerState.utciVisible ?? true;
			
			console.log(`[COMPARISON RENDERER] Created UTCI mesh for comparison analysis`);
		} catch (error) {
			console.error('[COMPARISON RENDERER] Failed to create UTCI mesh:', error);
		}
	}

	/**
	 * Update UTCI surface mesh with new viewer state
	 */
	function updateComparisonUtciMesh(): void {
		if (!comparisonUtciMesh || !$comparisonAnalysis) return;

		const currentViewerState = get(viewerStore);
		const hourIndex = currentViewerState.currentHour ?? 0;
		const colorMode = currentViewerState.colorMode ?? 'normalized';
		const metricType = currentViewerState.metricType ?? 'utci';
		
		// Use unified range for consistent color mapping between base and comparison
		const rangeOverride = get(unifiedUtciRange) ?? undefined;

		try {
			updateUtciSurfaceTexture(
				comparisonUtciMesh,
				$comparisonAnalysis,
				hourIndex,
				colorMode,
				metricType,
				rangeOverride
			);
			
			// Update visibility
			comparisonUtciMesh.visible = currentViewerState.utciVisible ?? true;
			
			invalidate();
		} catch (error) {
			console.error('[COMPARISON RENDERER] Failed to update UTCI mesh:', error);
		}
	}

	/**
	 * Dispose comparison UTCI mesh
	 */
	function disposeComparisonUtciMesh(): void {
		if (!comparisonUtciMesh) return;

		if (comparisonScene) {
			comparisonScene.remove(comparisonUtciMesh);
		}

		// Dispose materials and textures
		const materials = Array.isArray(comparisonUtciMesh.material)
			? comparisonUtciMesh.material
			: [comparisonUtciMesh.material];

		materials.forEach((mat) => {
			const material = mat as MeshBasicMaterial;
			material.map?.dispose();
			material.dispose();
		});

		comparisonUtciMesh.geometry.dispose();
		comparisonUtciMesh = null;
		
		console.log(`[COMPARISON RENDERER] Disposed UTCI mesh`);
	}

	// Derive data directory from base path
	const getDataBasePath = () => {
		const basePath = base || '';
		return basePath.replace(/\/viewer\/build$/, '');
	};

	/**
	 * Load comparison model when analysis changes
	 */
	async function loadComparisonModel(analysis: typeof $comparisonAnalysis) {
		if (!analysis) {
			cleanupComparison();
			return;
		}

		const modelPath = analysis.metadata.model_file.replace('data/', `${getDataBasePath()}/data/`);
		const analysisId = $comparisonStore.comparisonAnalysisId;
		
		// Verify the analysis matches the current store request to prevent stale loads
		// This handles race conditions when switching scenarios rapidly
		if (!analysisId) {
			console.log(`[COMPARISON RENDERER] No active comparison, skipping`);
			return;
		}

		// Skip if this exact model is already loaded
		if (loadedAnalysisId === modelPath && comparisonModel) {
			console.log(`[COMPARISON RENDERER] Model already loaded for ${modelPath}`);
			return;
		}

		console.log(`[COMPARISON RENDERER] Loading model: ${modelPath}`);
		isLoading = true;
		loadError = null;

		try {
			// Initialize comparison scene if needed
			if (!comparisonScene) {
				comparisonScene = new THREE.Scene();
				// Copy background from base scene for visual consistency
				if (scene.background) {
					comparisonScene.background = scene.background;
				}
				// Add lights to match the base scene
				addLightsToScene(comparisonScene);
			}

			// Initialize comparison camera (clone of base camera)
			if (!comparisonCamera && baseCamera) {
				comparisonCamera = baseCamera.clone() as PerspectiveCamera;
			}

			// Check cache first
			const cached = getCachedModel(modelPath);
			let modelGroup: Group;

			if (cached) {
				console.log(`[COMPARISON RENDERER] Using cached model: ${modelPath}`);
				modelGroup = cloneAndProcessCachedModel(cached.scene, analysis);
			} else {
				console.log(`[COMPARISON RENDERER] Loading model from file: ${modelPath}`);
				modelVersion++;
				const gltf = await new Promise<any>((resolve, reject) => {
					gltfLoader.load(
						`${modelPath}?v=${modelVersion}`,
						resolve,
						undefined,
						reject
					);
				});

				modelGroup = processLoadedModel(gltf.scene, analysis, modelPath);
			}

			// Clear old model from comparison scene
			if (comparisonModel && comparisonScene) {
				comparisonScene.remove(comparisonModel);
				disposeGroup(comparisonModel);
			}

			// Add new model to comparison scene
			comparisonModel = modelGroup;
			comparisonScene.add(comparisonModel);

			// Discover layers for visibility sync
			comparisonLayerMap = discoverLayersForComparison(comparisonModel);
			
			// Store layer types from comparison model
			comparisonLayerTypes = Array.from(comparisonLayerMap.keys());
			
			// Merge comparison layers with base layers in the store
			// This makes new layers (like "new_buildings", "new_trees") appear in LayerControls
			mergeComparisonLayers(comparisonLayerTypes);

			// Apply current layer visibility
			applyLayerVisibilityToComparison($layerStore);

			// Create UTCI surface mesh for comparison analysis data
			createComparisonUtciMesh(analysis);

			loadedAnalysisId = modelPath;
			isLoading = false;

			console.log(`[COMPARISON RENDERER] Model loaded successfully`);
			invalidate();
		} catch (error) {
			console.error('[COMPARISON RENDERER] Failed to load model:', error);
			loadError = error instanceof Error ? error.message : 'Failed to load comparison model';
			isLoading = false;
		}
	}

	/**
	 * Clone and process a cached model
	 */
	function cloneAndProcessCachedModel(cachedScene: Group, analysis: typeof $comparisonAnalysis): Group {
		const modelGroup = cachedScene.clone(true);

		// Clone materials to avoid sharing
		modelGroup.traverse((child) => {
			if (child instanceof THREE.Mesh && child.material) {
				if (Array.isArray(child.material)) {
					child.material = child.material.map((mat) => mat.clone());
				} else {
					child.material = child.material.clone();
				}
			}
		});

		// Apply normalization offset if needed
		applyNormalizationOffset(modelGroup, analysis);

		return modelGroup;
	}

	/**
	 * Process a freshly loaded model
	 */
	function processLoadedModel(
		loadedScene: Group,
		analysis: typeof $comparisonAnalysis,
		modelPath: string
	): Group {
		const modelGroup = loadedScene;

		// Apply layer materials
		applyLayerMaterials(modelGroup);

		// Apply coordinate transform
		const coordinateSystem = analysis?.metadata.coordinate_system || 'xy_ground';
		applyCoordinateTransform(modelGroup, coordinateSystem);

		// Cache the processed scene (before normalization)
		if (!hasModelInCache(modelPath)) {
			const sceneToCache = modelGroup.clone(true);
			sceneToCache.traverse((child) => {
				if (child instanceof THREE.Mesh && child.material) {
					if (Array.isArray(child.material)) {
						child.material = child.material.map((mat) => mat.clone());
					} else {
						child.material = child.material.clone();
					}
				}
			});
			cacheModel(modelPath, sceneToCache);
		}

		// Apply normalization offset
		applyNormalizationOffset(modelGroup, analysis);

		return modelGroup;
	}

	/**
	 * Apply normalization offset to model
	 */
	function applyNormalizationOffset(modelGroup: Group, analysis: typeof $comparisonAnalysis): void {
		if (!analysis || !isNormalizationEnabled()) return;

		const metadata = analysis.metadata;
		const coordinateSystem = metadata.coordinate_system || 'xy_ground';
		// Cast to any since bounds may exist dynamically but not in TypeScript type
		const scenarioOrigin = calculateScenarioOrigin(metadata as any);
		const anchorOffset = getAnchorOffset();

		let transformedOrigin: THREE.Vector3;
		if (coordinateSystem === 'xy_ground') {
			transformedOrigin = new THREE.Vector3(scenarioOrigin.x, scenarioOrigin.z, -scenarioOrigin.y);
		} else {
			transformedOrigin = scenarioOrigin.clone();
		}

		const offset = anchorOffset.clone().sub(transformedOrigin);

		if (offset.lengthSq() > 0.001) {
			console.log(`[COMPARISON RENDERER] Applying normalization offset:`, offset);
			applyModelOffset(modelGroup, offset);
		}
	}

	/**
	 * Discover layers in comparison model
	 */
	function discoverLayersForComparison(model: Group): Map<string, THREE.Mesh[]> {
		const layers = new Map<string, THREE.Mesh[]>();

		model.traverse((child) => {
			if (child instanceof THREE.Mesh && child.userData.layerType) {
				const layerType = child.userData.layerType as string;
				if (child.name.includes('_edges')) return;

				if (!layers.has(layerType)) {
					layers.set(layerType, []);
				}
				layers.get(layerType)!.push(child);
			}
		});

		return layers;
	}

	/**
	 * Apply layer visibility to comparison model
	 */
	function applyLayerVisibilityToComparison(visibilityState: Record<string, boolean>): void {
		comparisonLayerMap.forEach((meshes, layerType) => {
			const visible = visibilityState[layerType] ?? false;
			meshes.forEach((mesh) => {
				mesh.visible = visible;
				// Handle edge lines
				mesh.children.forEach((child) => {
					if (child instanceof THREE.LineSegments && child.name.includes('_edges')) {
						child.visible = visible;
					}
				});
			});
		});
	}

	/**
	 * Dispose of a group and its resources
	 */
	function disposeGroup(group: Group): void {
		group.traverse((child) => {
			if (child instanceof THREE.Mesh) {
				child.geometry?.dispose();
				if (Array.isArray(child.material)) {
					child.material.forEach((mat) => mat.dispose());
				} else if (child.material) {
					child.material.dispose();
				}
			}
		});
	}

	/**
	 * Cleanup comparison resources
	 */
	function cleanupComparison(): void {
		if (comparisonModel && comparisonScene) {
			comparisonScene.remove(comparisonModel);
			disposeGroup(comparisonModel);
			comparisonModel = null;
		}

		// Dispose UTCI mesh
		disposeComparisonUtciMesh();

		comparisonLayerMap.clear();
		comparisonLayerTypes = [];
		loadedAnalysisId = null;
		
		// Reset UTCI update state tracking
		lastUtciUpdateState = null;
		
		// Restore base-only layers in the store
		restoreBaseOnlyLayers();
	}

	// Store original render function
	let originalAutoRender: boolean | undefined;
	let renderLoopActive = false;

	/**
	 * Custom render loop with scissor-test rendering
	 */
	function customRenderLoop(): void {
		if (!renderer || !$comparisonStore.isComparing || !comparisonScene || !comparisonCamera) {
			return;
		}

		// Use baseCamera prop which is passed from the parent component
		const actualCamera = baseCamera;
		if (!actualCamera) return;

		// Sync comparison camera with base camera
		comparisonCamera.position.copy(actualCamera.position);
		comparisonCamera.quaternion.copy(actualCamera.quaternion);
		if ('zoom' in actualCamera && 'zoom' in comparisonCamera) {
			comparisonCamera.zoom = actualCamera.zoom;
		}
		comparisonCamera.updateProjectionMatrix();

		// Get canvas dimensions
		const canvas = renderer.domElement;
		const width = canvas.clientWidth;
		const height = canvas.clientHeight;

		// Calculate scissor positions based on curtain position
		const curtain = $curtainPosition;
		const curtainX = Math.floor(width * curtain);

		// Store original state
		const originalScissorTest = renderer.getScissorTest();

		// Enable scissor test
		renderer.setScissorTest(true);

		// Clear entire canvas first
		renderer.setViewport(0, 0, width, height);
		renderer.setScissor(0, 0, width, height);
		renderer.clear();

		// Render base scene on LEFT side (0 to curtainX)
		if (curtainX > 0) {
			renderer.setViewport(0, 0, width, height);
			renderer.setScissor(0, 0, curtainX, height);
			renderer.render(scene, actualCamera);
		}

		// Render comparison scene on RIGHT side (curtainX to end)
		if (curtainX < width && comparisonScene) {
			renderer.setViewport(0, 0, width, height);
			renderer.setScissor(curtainX, 0, width - curtainX, height);
			renderer.render(comparisonScene, comparisonCamera);
		}

		// Restore scissor test state
		renderer.setScissorTest(originalScissorTest);
	}

	// Animation frame ID for cleanup
	let animationFrameId: number | null = null;

	/**
	 * Start custom render loop
	 */
	function startRenderLoop(): void {
		if (renderLoopActive) return;

		renderLoopActive = true;
		const loop = () => {
			if (!renderLoopActive) return;
			customRenderLoop();
			animationFrameId = requestAnimationFrame(loop);
		};
		animationFrameId = requestAnimationFrame(loop);
	}

	/**
	 * Stop custom render loop and restore normal rendering
	 */
	function stopRenderLoop(): void {
		renderLoopActive = false;
		if (animationFrameId !== null) {
			cancelAnimationFrame(animationFrameId);
			animationFrameId = null;
		}

		// Reset renderer scissor state to allow normal rendering
		if (renderer) {
			renderer.setScissorTest(false);
			
			// Reset viewport to full canvas size
			const canvas = renderer.domElement;
			const width = canvas.clientWidth;
			const height = canvas.clientHeight;
			renderer.setViewport(0, 0, width, height);
			renderer.setScissor(0, 0, width, height);
			
			console.log(`[COMPARISON RENDERER] Renderer state reset, triggering re-render`);
		}

		// Trigger a re-render of the base scene to update the view immediately
		invalidate();
	}

	// React to comparison analysis changes
	$: if ($comparisonStore.isComparing && $comparisonAnalysis) {
		loadComparisonModel($comparisonAnalysis);
	}

	// React to layer visibility changes
	$: if ($comparisonStore.isComparing && comparisonModel) {
		applyLayerVisibilityToComparison($layerStore);
		invalidate();
	}

	// Track previous viewer state for comparison UTCI updates
	// Only trigger expensive texture updates when relevant viewer properties change
	let lastUtciUpdateState: { hour: number; colorMode: string; metricType: string; visible: boolean } | null = null;

	// React to viewer state changes (hour, metric type, color mode, visibility)
	// Note: We deliberately do NOT reference $comparisonStore here to avoid
	// triggering expensive UTCI updates on curtain position changes
	$: if (comparisonUtciMesh && $viewerStore) {
		const currentState = {
			hour: $viewerStore.currentHour,
			colorMode: $viewerStore.colorMode,
			metricType: $viewerStore.metricType ?? 'utci',
			visible: $viewerStore.utciVisible ?? true
		};
		
		// Only update if comparison is active and state actually changed
		if (get(comparisonStore).isComparing && (
			!lastUtciUpdateState ||
			lastUtciUpdateState.hour !== currentState.hour ||
			lastUtciUpdateState.colorMode !== currentState.colorMode ||
			lastUtciUpdateState.metricType !== currentState.metricType ||
			lastUtciUpdateState.visible !== currentState.visible
		)) {
			updateComparisonUtciMesh();
			lastUtciUpdateState = currentState;
		}
	}

	// Note: No need to invalidate() on curtain position changes - the custom render loop
	// already runs continuously via requestAnimationFrame when comparison is active

	// Start/stop render loop based on comparison state
	$: if ($comparisonStore.isComparing && comparisonScene && comparisonCamera) {
		startRenderLoop();
	} else {
		stopRenderLoop();
	}

	onMount(() => {
		console.log('[COMPARISON RENDERER] Mounted');
	});

	onDestroy(() => {
		console.log('[COMPARISON RENDERER] Destroying');
		stopRenderLoop();
		cleanupComparison();

		if (comparisonScene) {
			comparisonScene = null;
		}
		if (comparisonCamera) {
			comparisonCamera = null;
		}
	});
</script>

<!-- 
	This component doesn't render any visual elements directly.
	It hooks into Threlte's render loop to implement scissor-based dual rendering.
-->
