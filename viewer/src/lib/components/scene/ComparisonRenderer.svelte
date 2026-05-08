<script lang="ts">
	/**
	 * ComparisonRenderer Component
	 *
	 * ABOUTME: Implements scissor-test dual rendering for comparing base and comparison scenes.
	 * The base scene renders on the left side (0 to curtain position) and the comparison
	 * scene renders on the right side (curtain position to end). Camera and layers are synced.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { useThrelte, useTask } from '@threlte/core';
	import { T } from '@threlte/core';
	import { comparisonStore, curtainPosition, comparisonAnalysis, unifiedUtciRange, setComparisonModelLoading } from '$lib/stores/comparisonStore';
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
		disposeUtciSurfaceMesh,
		type UtciSurfaceBackendType,
		updateUtciSurfaceMesh
	} from '$lib/services/pointCloudService';
	import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';
	import { applyModelCoordinateTransform, calculateScenarioOrigin, applyModelOffset } from '$lib/utils/coordinates';
	import { resolveAnalysisModelPath } from '$lib/utils/analysisPaths';
	import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
	import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
	import * as THREE from 'three';
	import type { Group, PerspectiveCamera, Mesh } from 'three';

	// Props
	export let baseCamera: PerspectiveCamera | undefined = undefined;
	export let utciSurfaceBackend: UtciSurfaceBackendType = 'dataTexture';

	const { renderer, scene, invalidate, autoRender, renderStage } = useThrelte();

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
	let lastComparisonAnalysis: typeof $comparisonAnalysis = null;
	let lastBackend: UtciSurfaceBackendType | null = null;

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
	function createComparisonUtciMesh(analysis: typeof $comparisonAnalysis): boolean {
		if (!analysis || !comparisonScene) return false;

		// Dispose existing mesh first
		disposeComparisonUtciMesh();

		// Get current viewer state for UTCI visualization
		const currentViewerState = get(viewerStore);
		const hourIndex = currentViewerState.currentHour ?? 0;
		const monthIndex = currentViewerState.currentMonth ?? 7;
		const effectiveHourIndex = getEffectiveHourIndex(analysis, hourIndex, monthIndex);
		const colorMode = currentViewerState.colorMode ?? 'normalized';
		const metricType = currentViewerState.metricType ?? 'utci';

		// Use unified range for consistent color mapping between base and comparison
		const rangeOverride = get(unifiedUtciRange) ?? undefined;

		try {
			comparisonUtciMesh = createUtciSurfaceMesh({
				analysis,
				hourIndex: effectiveHourIndex,
				colorMode,
				metricType,
				rangeOverride,
				monthIndex,
				backend: utciSurfaceBackend
			});
			comparisonScene.add(comparisonUtciMesh);
			
			// Apply visibility based on viewer state
			comparisonUtciMesh.visible = currentViewerState.utciVisible ?? true;
			lastComparisonAnalysis = analysis;
			lastBackend = utciSurfaceBackend;
			
			console.log(`[COMPARISON RENDERER] Created UTCI mesh for comparison analysis`);
			return true;
		} catch (error) {
			console.error('[COMPARISON RENDERER] Failed to create UTCI mesh:', error);
			return false;
		}
	}

	/**
	 * Update UTCI surface mesh with new viewer state
	 */
	function updateComparisonUtciMesh(): boolean {
		if (!comparisonUtciMesh || !$comparisonAnalysis) return false;

		const currentViewerState = get(viewerStore);
		const hourIndex = currentViewerState.currentHour ?? 0;
		const monthIndex = currentViewerState.currentMonth ?? 7;
		const effectiveHourIndex = getEffectiveHourIndex(
			$comparisonAnalysis,
			hourIndex,
			monthIndex
		);
		const colorMode = currentViewerState.colorMode ?? 'normalized';
		const metricType = currentViewerState.metricType ?? 'utci';

		// Use unified range for consistent color mapping between base and comparison
		const rangeOverride = get(unifiedUtciRange) ?? undefined;

		try {
			const updated = updateUtciSurfaceMesh(
				comparisonUtciMesh,
				{
					analysis: $comparisonAnalysis,
					hourIndex: effectiveHourIndex,
					colorMode,
					metricType,
					rangeOverride,
					monthIndex,
					backend: utciSurfaceBackend
				}
			);
			if (!updated) {
				return createComparisonUtciMesh($comparisonAnalysis);
			}
			
			// Update visibility
			comparisonUtciMesh.visible = currentViewerState.utciVisible ?? true;
			
			invalidate();
			return true;
		} catch (error) {
			console.error('[COMPARISON RENDERER] Failed to update UTCI mesh:', error);
			return false;
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

		disposeUtciSurfaceMesh(comparisonUtciMesh);
		comparisonUtciMesh = null;
		lastComparisonAnalysis = null;
		lastBackend = null;
		
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

		const analysisId = $comparisonStore.comparisonAnalysisId;
		const modelPath = resolveAnalysisModelPath(
			analysis.metadata,
			analysisId
		).replace('data/', `${getDataBasePath()}/data/`);
		
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
		setComparisonModelLoading(true);
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

				modelGroup = await processLoadedModel(gltf.scene, analysis, modelPath);
			}

			// Add new model to comparison scene FIRST (before removing old one)
			// This prevents the scene from being empty during the transition
			comparisonScene.add(modelGroup);

			// Clear old model from comparison scene AFTER new one is added
			if (comparisonModel && comparisonScene) {
				comparisonScene.remove(comparisonModel);
				disposeGroup(comparisonModel);
			}

			// Update reference to new model
			comparisonModel = modelGroup;

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
			setComparisonModelLoading(false);

			console.log(`[COMPARISON RENDERER] Model loaded successfully`);
			invalidate();
		} catch (error) {
			console.error('[COMPARISON RENDERER] Failed to load model:', error);
			loadError = error instanceof Error ? error.message : 'Failed to load comparison model';
			isLoading = false;
			setComparisonModelLoading(false);
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
	 * Process a freshly loaded model (async so layer merge can yield and avoid main-thread freeze).
	 */
	async function processLoadedModel(
		loadedScene: Group,
		analysis: typeof $comparisonAnalysis,
		modelPath: string
	): Promise<Group> {
		const modelGroup = loadedScene;

		// Apply layer materials (yields between layers for large models)
		await applyLayerMaterials(modelGroup);

		// Apply coordinate transform
		const coordinateSystem = analysis?.metadata.coordinate_system || 'xy_ground';
		applyModelCoordinateTransform(modelGroup, coordinateSystem);

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
		
		// Reset loading state
		isLoading = false;
		setComparisonModelLoading(false);
		
		// Restore base-only layers in the store
		restoreBaseOnlyLayers();
	}

	// Comparison render task: replaces Threlte's default auto-render while this
	// component is mounted (i.e. while comparison mode is active) so we can
	// fully control scissor-based passes.
	function renderComparison() {
		if (!renderer) return;
		if (!$comparisonStore.isComparing) return;
		if (!comparisonScene || !comparisonCamera) return;

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

		// Use renderer size rather than DOM element size for WebGPU safety
		const size = new THREE.Vector2();
		renderer.getSize(size);
		const width = size.x;
		const height = size.y;
		if (width === 0 || height === 0) return;

		// Calculate scissor positions based on curtain position
		const curtain = $curtainPosition;
		const curtainX = Math.floor(width * curtain);

		// Store original state (WebGLRenderer) if the API exists. WebGPURenderer
		// does not currently expose setScissorTest / getScissorTest, but still
		// supports setScissor + setViewport.
		const canToggleScissorTest =
			typeof (renderer as any).getScissorTest === 'function' &&
			typeof (renderer as any).setScissorTest === 'function';
		const originalScissorTest = canToggleScissorTest
			? (renderer as any).getScissorTest()
			: undefined;

		if (canToggleScissorTest) {
			(renderer as any).setScissorTest(true);
		}

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
		// Only render if scene exists AND model is loaded (prevents empty scene flash)
		if (curtainX < width && comparisonScene && comparisonModel) {
			renderer.setViewport(0, 0, width, height);
			renderer.setScissor(curtainX, 0, width - curtainX, height);
			renderer.render(comparisonScene, comparisonCamera);
		}

		// Restore scissor test state if supported
		if (canToggleScissorTest && originalScissorTest !== undefined) {
			(renderer as any).setScissorTest(originalScissorTest);
		}
	}

	// Drive the comparison renderer as a dedicated render task. While this
	// component is mounted we disable Threlte's autoRender so this task becomes
	// the only render path, then restore the previous behavior on destroy.
	const { start: startComparisonRender, stop: stopComparisonRender } = useTask(renderComparison, {
		autoStart: false,
		autoInvalidate: false,
		stage: renderStage
	});

	let comparisonRenderActive = false;
	let previousAutoRender: boolean | null = null;

	// React to comparison analysis changes
	$: if ($comparisonStore.isComparing && $comparisonAnalysis) {
		loadComparisonModel($comparisonAnalysis);
	}

	// React to layer visibility changes
	$: if ($comparisonStore.isComparing && comparisonModel) {
		applyLayerVisibilityToComparison($layerStore);
		invalidate();
	}

	// Track previous viewer state for comparison UTCI updates.
	let lastUtciUpdateState: {
		hour: number;
		month: number;
		colorMode: string;
		metricType: string;
		visible: boolean;
		unifiedRangeMin: number | null;
		unifiedRangeMax: number | null;
	} | null = null;

	// React to viewer state changes without depending on comparison store fields
	// that change every frame, like curtain position.
	$: {
		const viewerState = $viewerStore;
		const currentUnifiedRange = $unifiedUtciRange;
		const currentComparisonAnalysis = $comparisonAnalysis;
		const currentState = viewerState ? {
			hour: viewerState.currentHour ?? 0,
			month: viewerState.currentMonth ?? 7,
			colorMode: viewerState.colorMode ?? 'normalized',
			metricType: viewerState.metricType ?? 'utci',
			visible: viewerState.utciVisible ?? true,
			unifiedRangeMin: currentUnifiedRange?.utciMin ?? null,
			unifiedRangeMax: currentUnifiedRange?.utciMax ?? null
		} : null;

		if (!viewerState || !get(comparisonStore).isComparing || !currentComparisonAnalysis) {
			if (!currentComparisonAnalysis) {
				lastUtciUpdateState = null;
			}
		} else if (!currentState) {
			// Unreachable with the guard above, but keeps TypeScript happy in the
			// subsequent state comparisons.
		} else if (!comparisonUtciMesh) {
			if (comparisonScene && currentState) {
				const recreated = createComparisonUtciMesh(currentComparisonAnalysis);
				if (recreated) {
					lastUtciUpdateState = currentState;
					invalidate();
				}
			}
		} else {
			const needsRecreate =
				currentComparisonAnalysis !== lastComparisonAnalysis ||
				utciSurfaceBackend !== lastBackend;
			const stateChanged =
				!lastUtciUpdateState ||
				lastUtciUpdateState.hour !== currentState.hour ||
				lastUtciUpdateState.month !== currentState.month ||
				lastUtciUpdateState.colorMode !== currentState.colorMode ||
				lastUtciUpdateState.metricType !== currentState.metricType ||
				lastUtciUpdateState.visible !== currentState.visible ||
				lastUtciUpdateState.unifiedRangeMin !== currentState.unifiedRangeMin ||
				lastUtciUpdateState.unifiedRangeMax !== currentState.unifiedRangeMax;

			if (needsRecreate) {
				const recreated = createComparisonUtciMesh(currentComparisonAnalysis);
				if (recreated) {
					lastUtciUpdateState = currentState;
					invalidate();
				}
			} else if (stateChanged) {
				const updated = updateComparisonUtciMesh();
				if (updated) {
					lastUtciUpdateState = currentState;
				}
			}
		}
	}

	// Note: No need to invalidate() on curtain position changes - the custom render loop
	// already runs continuously via requestAnimationFrame when comparison is active

	onMount(() => {
		console.log('[COMPARISON RENDERER] Mounted');

		// Take over rendering while comparison mode is active (this component
		// is only mounted when comparing). We disable Threlte's autoRender
		// and run our custom scissor-based task instead.
		previousAutoRender = autoRender.current;
		autoRender.set(false);
		startComparisonRender();
		comparisonRenderActive = true;
	});

	onDestroy(() => {
		console.log('[COMPARISON RENDERER] Destroying');

		// Stop comparison render task and restore original autoRender state so
		// the base scene resumes normal rendering.
		if (comparisonRenderActive) {
			stopComparisonRender();
			comparisonRenderActive = false;
		}
		if (previousAutoRender !== null) {
			autoRender.set(previousAutoRender);
			previousAutoRender = null;
		}

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
