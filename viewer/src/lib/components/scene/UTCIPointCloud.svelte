<script lang="ts">
	import { useThrelte } from '@threlte/core';
	import { onDestroy } from 'svelte';
	import type { Analysis } from '$lib/types/analysis';
	import {
		createUtciSurfaceMesh,
		disposeUtciSurfaceMesh,
		type UtciSurfaceBackendType,
		updateUtciSurfaceMesh
	} from '$lib/services/pointCloudService';
	import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { unifiedUtciRange } from '$lib/stores/comparisonStore';
	import type { Group, Mesh } from 'three';

	type UtciSurfaceDiagnostics = {
		utciSurfaceSource?: string;
		selectedHourTransferCount?: number;
		dataTextureBuildCount?: number;
	};

	export let analysis: Analysis | null = null;
	export let model: Group | null = null;
	export let utciSurfaceBackend: UtciSurfaceBackendType = 'dataTexture';
	export let onUtciSurfaceDiagnostics:
		| ((diagnostics: UtciSurfaceDiagnostics) => void)
		| undefined = undefined;

	// Expose mesh for tooltip raycasting
	export let utciSurface: Mesh | null = null;
	let lastAnalysis: Analysis | null = null;
	let lastBackend: UtciSurfaceBackendType | null = null;
	const { scene, invalidate } = useThrelte();

	function disposeUtciSurface() {
		if (!utciSurface) return;

		disposeUtciSurfaceMesh(utciSurface);
		utciSurface = null;
		lastAnalysis = null;
		lastBackend = null;
		onUtciSurfaceDiagnostics?.({});
	}

	function publishUtciSurfaceDiagnostics(): void {
		onUtciSurfaceDiagnostics?.({
			utciSurfaceSource: utciSurface?.userData.utciSurfaceSource as string | undefined,
			selectedHourTransferCount: utciSurface?.userData.selectedHourTransferCount as
				| number
				| undefined,
			dataTextureBuildCount: utciSurface?.userData.dataTextureBuildCount as
				| number
				| undefined
		});
	}

	function buildSurfaceOptions(
		activeAnalysis: Analysis,
		viewerState: typeof $viewerStore,
		rangeOverride: typeof $unifiedUtciRange | undefined
	) {
		return {
			analysis: activeAnalysis,
			hourIndex: getEffectiveHourIndex(
				activeAnalysis,
				viewerState?.currentHour ?? 0,
				viewerState?.currentMonth ?? 7
			),
			colorMode: viewerState?.colorMode ?? 'normalized',
			metricType: viewerState?.metricType ?? 'utci',
			rangeOverride: rangeOverride ?? undefined,
			monthIndex: viewerState?.currentMonth ?? 7,
			backend: utciSurfaceBackend
		} as const;
	}

	function recreateUtciSurface(
		activeAnalysis: Analysis,
		viewerState: typeof $viewerStore,
		rangeOverride: typeof $unifiedUtciRange | undefined
	): void {
		disposeUtciSurface();
		utciSurface = createUtciSurfaceMesh(buildSurfaceOptions(activeAnalysis, viewerState, rangeOverride));
		scene.add(utciSurface);
		lastAnalysis = activeAnalysis;
		lastBackend = utciSurfaceBackend;
		publishUtciSurfaceDiagnostics();
	}

	// Track last update state to avoid redundant texture updates
	let lastUpdateState: {
		hour: number;
		month: number;
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
			month: viewerState.currentMonth ?? 7,
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
			lastUpdateState.month !== currentState.month ||
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
		} else if (!utciSurface || analysis !== lastAnalysis || utciSurfaceBackend !== lastBackend) {
			// Analysis or backend changed - recreate the mesh
			recreateUtciSurface(analysis, viewerState, currentUnifiedRange);
			// Update state tracking
			lastUpdateState = {
				hour: viewerState?.currentHour ?? 0,
				month: viewerState?.currentMonth ?? 7,
				colorMode: viewerState?.colorMode ?? 'normalized',
				metricType: viewerState?.metricType ?? 'utci',
				unifiedRangeMin: currentUnifiedRange?.utciMin ?? null,
				unifiedRangeMax: currentUnifiedRange?.utciMax ?? null
			};
			invalidate();
		} else if (utciSurface && viewerState && hasStateChanged(viewerState, currentUnifiedRange)) {
			// Viewer state or unified range changed - update existing surface or recreate if required
			const updated = updateUtciSurfaceMesh(
				utciSurface,
				buildSurfaceOptions(analysis, viewerState, currentUnifiedRange)
			);
			if (!updated) {
				recreateUtciSurface(analysis, viewerState, currentUnifiedRange);
			} else {
				publishUtciSurfaceDiagnostics();
			}
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

