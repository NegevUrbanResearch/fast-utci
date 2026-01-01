<script lang="ts">
	import { useThrelte } from '@threlte/core';
	import { onDestroy } from 'svelte';
	import type { Analysis } from '$lib/types/analysis';
	import {
		createUtciSurfaceMesh,
		updateUtciSurfaceTexture
	} from '$lib/services/pointCloudService';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { unifiedUtciRange } from '$lib/stores/comparisonStore';
	import type { Group, Mesh, MeshBasicMaterial } from 'three';

	export let analysis: Analysis | null = null;
	export let model: Group | null = null;

	// Expose mesh for tooltip raycasting
	export let utciSurface: Mesh | null = null;
	let lastAnalysis: Analysis | null = null;
	const { scene, invalidate } = useThrelte();

	function disposeUtciSurface() {
		if (!utciSurface) return;

		scene.remove(utciSurface);

		const materials = Array.isArray(utciSurface.material)
			? utciSurface.material
			: [utciSurface.material];

		materials.forEach((mat) => {
			const material = mat as MeshBasicMaterial;
			material.map?.dispose();
			material.dispose();
		});

		utciSurface.geometry.dispose();
		utciSurface = null;
		lastAnalysis = null;
	}

	// Track last update state to avoid redundant texture updates
	let lastUpdateState: {
		hour: number;
		colorMode: string;
		metricType: string;
		unifiedRangeMin: number | null;
		unifiedRangeMax: number | null;
	} | null = null;

	/**
	 * Check if the update state has changed and we need to refresh the texture.
	 * This consolidates all texture update triggers into a single reactive block.
	 */
	function hasStateChanged(
		viewerState: typeof $viewerStore,
		unifiedRange: typeof $unifiedUtciRange
	): boolean {
		const currentState = {
			hour: viewerState.currentHour,
			colorMode: viewerState.colorMode,
			metricType: viewerState.metricType ?? 'utci',
			unifiedRangeMin: unifiedRange?.utciMin ?? null,
			unifiedRangeMax: unifiedRange?.utciMax ?? null
		};

		if (!lastUpdateState) {
			lastUpdateState = currentState;
			return true;
		}

		const changed =
			lastUpdateState.hour !== currentState.hour ||
			lastUpdateState.colorMode !== currentState.colorMode ||
			lastUpdateState.metricType !== currentState.metricType ||
			lastUpdateState.unifiedRangeMin !== currentState.unifiedRangeMin ||
			lastUpdateState.unifiedRangeMax !== currentState.unifiedRangeMax;

		if (changed) {
			lastUpdateState = currentState;
		}

		return changed;
	}

	// Unified reactive block that handles all texture updates
	// By directly referencing $unifiedUtciRange here, Svelte properly tracks it as a dependency
	$: {
		const viewerState = $viewerStore;
		const currentUnifiedRange = $unifiedUtciRange;
		const rangeOverride = currentUnifiedRange ?? undefined;

		if (!analysis) {
			disposeUtciSurface();
			lastUpdateState = null;
		} else if (analysis !== lastAnalysis) {
			// Analysis changed - recreate the mesh
			disposeUtciSurface();
			utciSurface = createUtciSurfaceMesh(
				analysis,
				viewerState?.currentHour ?? 0,
				viewerState?.colorMode ?? 'normalized',
				viewerState?.metricType ?? 'utci',
				rangeOverride
			);
			scene.add(utciSurface);
			lastAnalysis = analysis;
			// Update state tracking
			lastUpdateState = {
				hour: viewerState?.currentHour ?? 0,
				colorMode: viewerState?.colorMode ?? 'normalized',
				metricType: viewerState?.metricType ?? 'utci',
				unifiedRangeMin: currentUnifiedRange?.utciMin ?? null,
				unifiedRangeMax: currentUnifiedRange?.utciMax ?? null
			};
			invalidate();
		} else if (utciSurface && viewerState && hasStateChanged(viewerState, currentUnifiedRange)) {
			// Viewer state or unified range changed - update texture
			updateUtciSurfaceTexture(
				utciSurface,
				analysis,
				viewerState.currentHour,
				viewerState.colorMode,
				viewerState.metricType ?? 'utci',
				rangeOverride
			);
			invalidate();
		}
	}

	$: {
		if (utciSurface) {
			utciSurface.visible = Boolean(analysis && $viewerStore?.utciVisible);
			if (utciSurface.visible) {
				invalidate();
			}
		}
	}

	onDestroy(() => {
		disposeUtciSurface();
	});
</script>

