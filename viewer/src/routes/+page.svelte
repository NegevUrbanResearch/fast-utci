<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { base } from '$app/paths';
	import { analysisStore, loadAnalysisData } from '$lib/stores/analysisStore';
	import { viewerStore, setAnalysisId, setLoading, setError } from '$lib/stores/viewerStore';
	import { cameraStore, focusCameraOnModel } from '$lib/stores/cameraStore';
	import { setDiscoveredLayers } from '$lib/stores/layerStore';
	import { calculateModelBounds, calculateModelCenter, calculateModelSize } from '$lib/utils/bounds';
	import Scene from '$lib/components/scene/Scene.svelte';
	import Camera from '$lib/components/scene/Camera.svelte';
	import Lights from '$lib/components/scene/Lights.svelte';
	import GridHelper from '$lib/components/scene/GridHelper.svelte';
	import Model from '$lib/components/scene/Model.svelte';
	import UTCIPointCloud from '$lib/components/scene/UTCIPointCloud.svelte';
	import TimeSlider from '$lib/components/ui/TimeSlider.svelte';
	import LayerControls from '$lib/components/ui/LayerControls.svelte';
	import ColorLegend from '$lib/components/ui/ColorLegend.svelte';
	import ScenarioSelector from '$lib/components/ui/ScenarioSelector.svelte';
	import AnalyticsPanel from '$lib/components/ui/AnalyticsPanel.svelte';
	import '$lib/styles/variables.css';
	import type { Group } from 'three';

	// Data base path: strip /viewer/build from base path to get project root
	const getDataBasePath = () => {
		const basePath = base || '';
		return basePath.replace(/\/viewer\/build$/, '');
	};

	let model: Group | null = null;
	let gridVisible = false;

	// Get analysis ID from URL parameters (client-side only)
	let analysisId: string = '20250815_grid_2m_fullday';
	let mounted = false;

	async function loadAnalysis(id: string) {
		try {
			setLoading(true);
			setError(null);
			setAnalysisId(id);
			await loadAnalysisData(id);
			
			// Focus camera on model once loaded
			if (model && $analysisStore) {
				const bounds = calculateModelBounds(model);
				const center = calculateModelCenter(model);
				const size = calculateModelSize(model);
				focusCameraOnModel(center, size);
			}
		} catch (error) {
			console.error('[ERROR] Failed to load analysis:', error);
			setError(error instanceof Error ? error.message : 'Failed to load analysis');
		} finally {
			setLoading(false);
		}
	}

	onMount(() => {
		// Client-side only: get analysis ID from URL parameters
		if (typeof window !== 'undefined') {
			const params = new URLSearchParams(window.location.search);
			analysisId = params.get('analysis') || '20250815_grid_2m_fullday';
			
			console.log('[OK] Viewer initialized');
			mounted = true;
			loadAnalysis(analysisId);
		}
	});

	// React to URL changes (client-side only)
	$: if (typeof window !== 'undefined' && $page.url.searchParams && mounted) {
		const newAnalysisId = $page.url.searchParams.get('analysis') || '20250815_grid_2m_fullday';
		if (newAnalysisId !== analysisId) {
			analysisId = newAnalysisId;
			loadAnalysis(analysisId);
		}
	}
</script>

<div class="viewer-container">
	{#if $viewerStore.loading}
		<div class="loading">
			<div class="loading-message">Loading analysis data...</div>
		</div>
	{/if}

	{#if $viewerStore.error}
		<div class="error">
			<div class="error-message">Error: {$viewerStore.error}</div>
		</div>
	{/if}

	<Scene>
		<Camera />
		<Lights />
		
		{#if $analysisStore}
			{#key $analysisStore.metadata.model_file}
				<Model
					modelPath={$analysisStore.metadata.model_file.replace('data/', `${getDataBasePath()}/data/`)}
					coordinateSystem={$analysisStore.metadata.coordinate_system || 'xy_ground'}
					on:modelLoaded={(e) => {
						model = e.detail;
						if (model) {
							const bounds = calculateModelBounds(model);
							const center = calculateModelCenter(model);
							const size = calculateModelSize(model);
							focusCameraOnModel(center, size);
						}
					}}
					on:layersDiscovered={(e) => {
						setDiscoveredLayers(e.detail);
					}}
				/>
			{/key}
			
			{#if model}
				<GridHelper {model} visible={gridVisible} />
				<UTCIPointCloud analysis={$analysisStore} model={model} />
			{/if}
		{/if}
	</Scene>

	<ScenarioSelector />
	<LayerControls />
	<AnalyticsPanel />
	{#if $analysisStore && $analysisStore.metadata.analysis_type === 'full_day'}
		<TimeSlider />
	{/if}
	<ColorLegend />
</div>

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		overflow: hidden;
		width: 100%;
		height: 100%;
	}

	.viewer-container {
		width: 100vw;
		height: 100vh;
		position: relative;
		overflow: hidden;
	}

	.loading,
	.error {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		background: var(--color-bg-panel);
		padding: var(--spacing-xl);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		z-index: calc(var(--z-tooltip) + 1);
	}

	.loading-message,
	.error-message {
		font-size: 16px;
		color: var(--color-text-primary);
		text-align: center;
	}

	.error-message {
		color: #e74c3c;
	}
</style>
