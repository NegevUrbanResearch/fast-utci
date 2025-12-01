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
	import RadialTimePicker from '$lib/components/ui/RadialTimePicker.svelte';
	import LayerControls from '$lib/components/ui/LayerControls.svelte';
	import ColorLegend from '$lib/components/ui/ColorLegend.svelte';
	import ScenarioSelector from '$lib/components/ui/ScenarioSelector.svelte';
	import AnalyticsPanel from '$lib/components/ui/AnalyticsPanel.svelte';
	import ThemeToggle from '$lib/components/ui/ThemeToggle.svelte';
	import '$lib/styles/variables.css';
	import nurLogo from '$lib/assets/Nur Logo white.svg';
	import mitLogo from '$lib/assets/MIT.svg';
	import bguLogo from '$lib/assets/bgu-logo.svg';
	import type { Group } from 'three';

	const getDataBasePath = () => {
		const basePath = base || '';
		return basePath.replace(/\/viewer\/build$/, '');
	};

	let model: Group | null = null;
	let gridVisible = false;

	let analyticsOpen = false;

	let analysisId: string = '20250815_grid_2m_fullday';
	let mounted = false;

	async function loadAnalysis(id: string) {
		try {
			setLoading(true);
			setError(null);
			setAnalysisId(id);
			await loadAnalysisData(id);

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
		if (typeof window !== 'undefined') {
			const params = new URLSearchParams(window.location.search);
			analysisId = params.get('analysis') || '20250815_grid_2m_fullday';

			console.log('[OK] Viewer initialized');
			mounted = true;
			loadAnalysis(analysisId);
		}
	});

	$: if (typeof window !== 'undefined' && $page.url.searchParams && mounted) {
		const newAnalysisId = $page.url.searchParams.get('analysis') || '20250815_grid_2m_fullday';
		if (newAnalysisId !== analysisId) {
			analysisId = newAnalysisId;
			loadAnalysis(analysisId);
		}
	}
</script>

<div class="viewer-shell">
	<header class="app-header">
		<div class="header-left">
			<div class="partner-logos">
				<img src={nurLogo} alt="NUR Negev Urban Research" class="logo logo-nur" />
				<img src={bguLogo} alt="BGU" class="logo logo-bgu" />
				<img src={mitLogo} alt="MIT" class="logo logo-mit" />
			</div>
			<div class="header-title">
				<div class="title-kicker">CityScope</div>
				<div class="title-main">Urban Comfort Lab</div>
			</div>
		</div>
		<div class="header-right">
			<ThemeToggle />
		</div>
	</header>

	<div class="app-body">
		<aside class="app-sidebar">
			<div class="sidebar-section">
				<div class="section-header">Scenario</div>
				<ScenarioSelector />
			</div>

			<div class="sidebar-section analytics-section">
				<button
					type="button"
					class="section-header section-header-toggle"
					on:click={() => (analyticsOpen = !analyticsOpen)}
				>
					<span>Analytics</span>
					<span class:open={analyticsOpen} class="chevron">▾</span>
				</button>
				{#if analyticsOpen}
					<AnalyticsPanel />
				{/if}
			</div>

			<div class="sidebar-section">
				<div class="section-header">Model Layers</div>
				<LayerControls />
			</div>

			<div class="sidebar-section">
				<div class="section-header">UTCI Legend &amp; Scale</div>
				<ColorLegend />
			</div>

			{#if $analysisStore && $analysisStore.metadata.analysis_type === 'full_day'}
				<div class="sidebar-section">
					<div class="section-header">Time of Day</div>
					<div class="section-subtitle">Select analysis hour for UTCI</div>
					<RadialTimePicker />
				</div>
			{/if}
		</aside>

		<main class="app-main">
			{#if $viewerStore.loading}
				<div class="overlay-message">Loading analysis data...</div>
			{/if}

			{#if $viewerStore.error}
				<div class="overlay-message error">Error: {$viewerStore.error}</div>
			{/if}

			<Scene backgroundColor={$viewerStore.theme === 'light' ? 0x4b5563 : 0x111827}>
				<Camera />
				<Lights />

				{#if $analysisStore}
					{#key $analysisStore.metadata.model_file}
						<Model
							modelPath={$analysisStore.metadata.model_file.replace(
								'data/',
								`${getDataBasePath()}/data/`
							)}
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
		</main>
	</div>
</div>

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		overflow: hidden;
		width: 100%;
		height: 100%;
		font-family: var(--font-family);
		background: var(--color-bg-page);
		color: var(--color-text-primary);
	}

	.viewer-shell {
		width: 100vw;
		height: 100vh;
		display: flex;
		flex-direction: column;
		background:
			radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 55%),
			var(--color-bg-page);
	}

	.app-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 8px 18px;
		border-bottom: 1px solid var(--color-border-subtle);
		background: var(--color-bg-header);
		backdrop-filter: blur(16px);
		z-index: 10;
	}

	.header-left {
		display: flex;
		align-items: center;
		gap: 14px;
	}

	.header-title {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.header-title .title-kicker {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		color: var(--color-text-secondary);
	}

	.header-title .title-main {
		font-size: 16px;
		font-weight: 600;
		letter-spacing: 0.02em;
	}

	.header-right {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.partner-logos {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.logo {
		height: 39px;
		object-fit: contain;
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.4));
		display: block;
	}

	.logo-nur {
		height: 39px;
	}

	.logo-bgu {
		height: 39px;
	}

	:global(html[data-theme='dark'] .logo-nur) {
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.6));
	}

	:global(html[data-theme='light'] .logo-nur) {
		filter: brightness(0) drop-shadow(0 0 4px rgba(0, 0, 0, 0.3));
	}

	:global(html[data-theme='dark'] .logo-bgu) {
		filter: drop-shadow(0 0 3px rgba(0, 0, 0, 0.55));
	}

	:global(html[data-theme='light'] .logo-bgu) {
		filter: drop-shadow(0 0 3px rgba(15, 23, 42, 0.45));
	}

	:global(html[data-theme='dark'] .logo-mit) {
		filter: invert(1) drop-shadow(0 0 4px rgba(0, 0, 0, 0.6));
	}

	:global(html[data-theme='light'] .logo-mit) {
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.4));
	}

	.app-body {
		flex: 1;
		display: grid;
		grid-template-columns: 320px minmax(0, 1fr);
		height: 100%;
	}

	.app-sidebar {
		border-right: 1px solid var(--color-border-subtle);
		background: var(--color-bg-sidebar);
		padding: 12px 10px;
		display: flex;
		flex-direction: column;
		gap: 10px;
		overflow-y: auto;
	}

	.sidebar-section {
		background: var(--color-bg-panel);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		padding: 10px 12px;
		border: 1px solid var(--color-border-subtle);
	}

	:global(html[data-theme='dark'] .app-sidebar .sidebar-section) {
		box-shadow:
			0 14px 30px rgba(15, 23, 42, 0.7),
			0 0 0 1px rgba(248, 250, 252, 0.03);
	}

	.section-header {
		font-size: 12px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		margin-bottom: 8px;
		color: var(--color-text-secondary);
	}

	.section-subtitle {
		font-size: 13px;
		color: var(--color-text-muted);
		margin-bottom: 8px;
	}

	.section-header-toggle {
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 6px;
		background: transparent;
		border: none;
		padding: 0;
		cursor: pointer;
	}

	.analytics-section {
		padding-top: 8px;
	}

	.analytics-section .section-header {
		margin-bottom: 4px;
	}

	.chevron {
		transition: transform 0.15s ease;
	}

	.chevron.open {
		transform: rotate(180deg);
	}

	.app-main {
		position: relative;
		background: var(--color-bg-page);
	}

	.overlay-message {
		position: absolute;
		top: 16px;
		left: 50%;
		transform: translateX(-50%);
		z-index: var(--z-tooltip);
		padding: 10px 16px;
		border-radius: 999px;
		background: var(--color-bg-panel);
		color: var(--color-text-primary);
		box-shadow: var(--shadow-panel);
		font-size: 13px;
	}

	.overlay-message.error {
		border: 1px solid var(--color-danger);
	}
</style>
