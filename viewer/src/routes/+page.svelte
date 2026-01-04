<script lang="ts">
	/**
	 * Main Page Component
	 *
	 * ABOUTME: The main viewer page that integrates the 3D scene, UI panels, and comparison features.
	 * When comparison mode is active, renders both base and comparison scenes with a curtain slider.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { page } from '$app/stores';
	import { base } from '$app/paths';
	import { analysisStore, loadAnalysisData } from '$lib/stores/analysisStore';
	import { viewerStore, setAnalysisId, setLoading, setError } from '$lib/stores/viewerStore';
	import { cameraStore, focusCameraOnModel } from '$lib/stores/cameraStore';
	import { setDiscoveredLayers } from '$lib/stores/layerStore';
	import { comparisonStore, comparisonAnalysis } from '$lib/stores/comparisonStore';
	import { calculateModelBounds, calculateModelCenter, calculateModelSize } from '$lib/utils/bounds';
	import Scene from '$lib/components/scene/Scene.svelte';
	import Camera from '$lib/components/scene/Camera.svelte';
	import Lights from '$lib/components/scene/Lights.svelte';
	import GridHelper from '$lib/components/scene/GridHelper.svelte';
	import Model from '$lib/components/scene/Model.svelte';
	import UTCIPointCloud from '$lib/components/scene/UTCIPointCloud.svelte';
	import ComparisonRenderer from '$lib/components/scene/ComparisonRenderer.svelte';
	import ComparisonCurtain from '$lib/components/ui/ComparisonCurtain.svelte';
	import RadialTimePicker from '$lib/components/ui/RadialTimePicker.svelte';
	import LayerControls from '$lib/components/ui/LayerControls.svelte';
	import ColorLegend from '$lib/components/ui/ColorLegend.svelte';
	import ScenarioSelector from '$lib/components/ui/ScenarioSelector.svelte';
	import AnalyticsPanel from '$lib/components/ui/AnalyticsPanel.svelte';
	import MetricTooltip from '$lib/components/ui/MetricTooltip.svelte';
	import '$lib/styles/variables.css';
	import nurLogo from '$lib/assets/Nur Logo white.svg';
	import mitLogo from '$lib/assets/MIT.svg';
	import bguLogo from '$lib/assets/bgu-logo.svg';
	import sceLogo from '$lib/assets/sce-logo.svg';
	import * as THREE from 'three';
	import type { Group, Mesh, PerspectiveCamera } from 'three';
	import { getTooltipData } from '$lib/services/tooltipService';

	const getDataBasePath = () => {
		const basePath = base || '';
		return basePath.replace(/\/viewer\/build$/, '');
	};

	let model: Group | null = null;
	let gridVisible = false;

	let analyticsOpen = false;
	let modelLoading = true;
	let hasFitOnce = false;
	let lastModelFile: string | null = null;

	let analysisId: string = '20250815_grid_2m_fullday';
	let mounted = false;

	// Tooltip state
	let tooltipVisible = false;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipValue: number | null = null;
	let utciMesh: Mesh | null = null;
	let canvasElement: HTMLCanvasElement | null = null;

	// Camera reference - will be set from Camera component
	let cameraRef: PerspectiveCamera | undefined = undefined;

	// Main viewport element reference for comparison curtain positioning
	let mainViewportElement: HTMLElement | null = null;

	// Scenario selector component reference
	let scenarioSelector: ScenarioSelector;

	// ComparisonRenderer component reference for accessing comparison UTCI mesh
	let comparisonRenderer: ComparisonRenderer;

	// Track comparison mode
	$: isComparing = $comparisonStore.isComparing;

	// Reactive scenario name for comparison curtain label
	// Watch comparisonAnalysisId to trigger updates when scenarios change
	$: comparisonScenarioName = (isComparing && scenarioSelector && $comparisonStore.comparisonAnalysisId)
		? scenarioSelector.getScenarioName()
		: 'Comparison';

	async function loadAnalysis(id: string) {
		try {
			modelLoading = true;
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

	// Trigger model loading overlay when the model file changes
	$: if ($analysisStore && $analysisStore.metadata?.model_file) {
		const currentModelFile = $analysisStore.metadata.model_file;
		if (currentModelFile !== lastModelFile) {
			modelLoading = true;
			lastModelFile = currentModelFile;
		}
	}

	// Throttle tooltip updates for performance
	let lastTooltipUpdate = 0;
	const TOOLTIP_THROTTLE_MS = 16; // ~60fps

	// Handle mouse move for tooltip
	function handleMouseMove(event: MouseEvent) {
		const now = performance.now();
		if (now - lastTooltipUpdate < TOOLTIP_THROTTLE_MS) {
			return; // Throttle updates
		}
		lastTooltipUpdate = now;

		if (!utciMesh || !$analysisStore || !$viewerStore.utciVisible || !canvasElement || !cameraRef) {
			tooltipVisible = false;
			return;
		}

		const canvasRect = canvasElement.getBoundingClientRect();

		// Determine which side of the comparison curtain the mouse is on
		// If in comparison mode and mouse is on the right side, use comparison data
		let meshToRaycast = utciMesh;
		let analysisToUse = $analysisStore;

		if (isComparing && mainViewportElement) {
			const viewportRect = mainViewportElement.getBoundingClientRect();
			const mouseXRelative = (event.clientX - viewportRect.left) / viewportRect.width;
			const curtainPos = $comparisonStore.curtainPosition;

			// If mouse is on the right side of the curtain, use comparison data
			if (mouseXRelative > curtainPos) {
				const comparisonMesh = comparisonRenderer?.getComparisonUtciMesh();
				if (comparisonMesh && $comparisonAnalysis) {
					meshToRaycast = comparisonMesh;
					analysisToUse = $comparisonAnalysis;
				}
			}
		}

		const tooltipData = getTooltipData(
			event,
			cameraRef,
			meshToRaycast,
			analysisToUse,
			$viewerStore.metricType,
			$viewerStore.currentHour,
			canvasRect
		);

		if (tooltipData) {
			tooltipVisible = true;
			tooltipX = event.clientX;
			tooltipY = event.clientY;
			tooltipValue = tooltipData.value;
		} else {
			tooltipVisible = false;
		}
	}

	function handleMouseLeave() {
		tooltipVisible = false;
	}

	// Attach event listeners to canvas element when available
	let eventListenersAttached = false;

	$: if (canvasElement && mounted && !eventListenersAttached) {
		const canvas = canvasElement;
		canvas.addEventListener('mousemove', handleMouseMove, { passive: true });
		canvas.addEventListener('mouseleave', handleMouseLeave, { passive: true });
		eventListenersAttached = true;
	}

	onDestroy(() => {
		if (canvasElement && eventListenersAttached) {
			canvasElement.removeEventListener('mousemove', handleMouseMove);
			canvasElement.removeEventListener('mouseleave', handleMouseLeave);
			eventListenersAttached = false;
		}
	});

</script>

<div class="viewer-shell">
	<header class="app-header">
		<div class="header-left">
			<div class="header-title">
				<div class="title-kicker">CityScope</div>
				<div class="title-main">Urban Comfort Lab</div>
			</div>
		</div>
		<div class="header-center">
			<div class="layers-section">
				<div class="layers-title">Layers</div>
				<LayerControls />
			</div>
		</div>
		<div class="header-right">
			<div class="partner-logos">
				<img src={nurLogo} alt="NUR Negev Urban Research" class="logo logo-nur" />
				<img src={bguLogo} alt="BGU" class="logo logo-bgu" />
				<img src={mitLogo} alt="MIT" class="logo logo-mit" />
				<img src={sceLogo} alt="SCE" class="logo logo-sce" />
			</div>
		</div>
	</header>

	<div class="app-body">
		<aside class="app-sidebar">
			<div class="sidebar-section">
				<div class="section-header">Scenario</div>
				<ScenarioSelector bind:this={scenarioSelector} />
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

			{#if $analysisStore && $analysisStore.metadata.analysis_type === 'full_day' && $viewerStore.metricType === 'utci'}
				<div class="sidebar-section">
					<div class="section-header">Time of Day</div>
					<div class="section-subtitle">Select analysis hour for UTCI</div>
					<RadialTimePicker />
				</div>
			{/if}
		</aside>

		<main class="app-main" bind:this={mainViewportElement}>
			<!-- Color Legend positioned at bottom right of screen -->
			<div class="legend-container">
				<ColorLegend />
			</div>

			<!-- Metric Tooltip -->
			<MetricTooltip
				visible={tooltipVisible}
				x={tooltipX}
				y={tooltipY}
				value={tooltipValue}
				metricType={$viewerStore.metricType}
			/>
			{#if $viewerStore.loading}
				<div class="overlay-message">Loading analysis data...</div>
			{/if}

			{#if $viewerStore.error}
				<div class="overlay-message error">Error: {$viewerStore.error}</div>
			{/if}

			{#if modelLoading || ($comparisonStore.isComparing && $comparisonStore.modelLoading)}
				<div 
					class="model-loading-backdrop" 
					class:comparison-mode={$comparisonStore.isComparing && $comparisonStore.modelLoading && !modelLoading}
					style={$comparisonStore.isComparing && $comparisonStore.modelLoading && !modelLoading 
						? `--curtain-position: ${$comparisonStore.curtainPosition}` 
						: ''}
					aria-hidden="true"
				></div>
				<div 
					class="model-loading-overlay" 
					class:comparison-mode={$comparisonStore.isComparing && $comparisonStore.modelLoading && !modelLoading}
					style={$comparisonStore.isComparing && $comparisonStore.modelLoading && !modelLoading 
						? `--curtain-position: ${$comparisonStore.curtainPosition}` 
						: ''}
					aria-live="polite"
				>
					<div class="spinner"></div>
					<div class="loading-text">Preparing model…</div>
				</div>
			{/if}

			<Scene
				backgroundColor={$viewerStore.theme === 'light' ? 0x4b5563 : 0x111827}
				bind:canvasElement={canvasElement}
			>
				<Camera bind:cameraRef={cameraRef} />
				<Lights />

				{#if $analysisStore}
					{#key $analysisStore.metadata.model_file}
						<Model
							modelPath={$analysisStore.metadata.model_file.replace(
								'data/',
								`${getDataBasePath()}/data/`
							)}
							coordinateSystem={$analysisStore.metadata.coordinate_system || 'xy_ground'}
							metadata={$analysisStore.metadata}
							on:modelLoaded={(e) => {
								model = e.detail;
								modelLoading = false;
								if (!hasFitOnce && model) {
									const bounds = calculateModelBounds(model);
									const center = calculateModelCenter(model);
									const size = calculateModelSize(model);
									// Bird's-eye, closer top-down fit
									const maxDim = Math.max(size.x, size.y, size.z);
									const distance = maxDim * 1.05;
									const position = center.clone().add(new THREE.Vector3(0, distance, 0.01));
									cameraStore.update((state) => ({
										...state,
										position,
										target: center.clone()
									}));
									hasFitOnce = true;
								}
							}}
							on:layersDiscovered={(e) => {
								setDiscoveredLayers(e.detail);
							}}
						/>
					{/key}

					{#if model}
						<GridHelper {model} visible={gridVisible} />
						<UTCIPointCloud
							analysis={$analysisStore}
							model={model}
							bind:utciSurface={utciMesh}
						/>
					{/if}

					<!-- Comparison renderer (only active when comparing) -->
					{#if isComparing}
						<ComparisonRenderer bind:this={comparisonRenderer} baseCamera={cameraRef} />
					{/if}
				{/if}
			</Scene>

			<!-- Comparison curtain overlay (only visible when comparing) -->
			{#if isComparing}
				<ComparisonCurtain
					containerElement={mainViewportElement}
					comparisonScenarioName={comparisonScenarioName}
				/>
			{/if}
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
		background: var(--color-bg-header);
		backdrop-filter: blur(16px);
		z-index: 10;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
		gap: 20px;
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

	.header-center {
		display: flex;
		align-items: center;
		flex: 1;
		justify-content: center;
		min-width: 0;
	}

	.layers-section {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		justify-content: center;
		gap: 4px;
	}

	.layers-title {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--color-text-secondary);
		line-height: 1.2;
		margin: 0;
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

	.logo-sce {
		height: 39px;
		filter: invert(1) drop-shadow(0 0 4px rgba(0, 0, 0, 0.6));
	}

	.app-body {
		flex: 1;
		display: grid;
		grid-template-columns: minmax(320px, 320px) 1fr;
		grid-template-areas: 'sidebar main';
		height: 100%;
		overflow: hidden;
		position: relative;
	}

	.app-sidebar {
		grid-area: sidebar;
		background: var(--color-bg-sidebar);
		padding: 12px 10px;
		display: flex;
		flex-direction: column;
		gap: 10px;
		overflow-y: auto;
		overflow-x: hidden;
		scrollbar-gutter: stable;
		width: 320px;
		min-width: 320px;
		max-width: 320px;
		box-sizing: border-box;
		flex-shrink: 0;
		contain: layout size;
		position: relative;
		box-shadow: 2px 0 12px rgba(0, 0, 0, 0.12);
	}

	.app-main {
		grid-area: main;
	}

	.sidebar-section {
		background: var(--color-bg-panel);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		padding: 10px 12px;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
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
		min-width: 0;
		overflow: hidden;
	}

	.legend-container {
		position: absolute;
		bottom: 20px;
		right: 20px;
		z-index: var(--z-tooltip);
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

	.model-loading-backdrop {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		z-index: calc(var(--z-tooltip) - 1);
		background: rgba(17, 24, 39, 0.4);
		backdrop-filter: blur(12px);
		-webkit-backdrop-filter: blur(12px);
		pointer-events: none;
		transition: opacity 0.25s ease-out;
	}

	.model-loading-backdrop.comparison-mode {
		left: calc(var(--curtain-position) * 100%);
		right: 0;
	}

	@media (prefers-reduced-motion: reduce) {
		.model-loading-backdrop {
			transition: none;
		}
	}

	.model-loading-overlay {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		z-index: var(--z-tooltip);
		min-width: 180px;
		padding: 14px 18px;
		border-radius: 14px;
		background: rgba(17, 24, 39, 0.82);
		backdrop-filter: blur(10px);
		color: white;
		box-shadow:
			0 14px 30px rgba(0, 0, 0, 0.35),
			0 0 0 1px rgba(255, 255, 255, 0.05);
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 10px;
		text-align: center;
	}

	.model-loading-overlay.comparison-mode {
		left: calc(50% + var(--curtain-position) * 50%);
		transform: translate(-50%, -50%);
	}

	.model-loading-overlay .loading-text {
		font-size: 13px;
		letter-spacing: 0.04em;
	}

	.spinner {
		width: 36px;
		height: 36px;
		border-radius: 50%;
		border: 3px solid rgba(255, 255, 255, 0.18);
		border-top-color: var(--color-accent);
		animation: spin 0.9s linear infinite;
	}

	@keyframes spin {
		from {
			transform: rotate(0deg);
		}
		to {
			transform: rotate(360deg);
		}
	}

</style>
