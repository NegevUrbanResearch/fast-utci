<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import { page } from "$app/stores";
	import { goto } from "$app/navigation";
	import { base } from "$app/paths";
	import {
		analysisStore,
		loadAnalysisData,
	} from "$lib/stores/analysisStore";
	import {
		viewerStore,
		setAnalysisId,
		setLoading,
		setError,
	} from "$lib/stores/viewerStore";
	import { cameraStore, focusCameraOnModel } from "$lib/stores/cameraStore";
	import { setDiscoveredLayers } from "$lib/stores/layerStore";
	import {
		calculateModelBounds,
		calculateModelCenter,
		calculateModelSize,
	} from "$lib/utils/bounds";
	import Scene from "$lib/components/scene/Scene.svelte";
	import Camera from "$lib/components/scene/Camera.svelte";
	import Lights from "$lib/components/scene/Lights.svelte";
	import GridHelper from "$lib/components/scene/GridHelper.svelte";
	import Model from "$lib/components/scene/Model.svelte";
	import UTCIPointCloud from "$lib/components/scene/UTCIPointCloud.svelte";
	import DebugUtciScissor from "$lib/components/scene/DebugUtciScissor.svelte";
	import ComparisonCurtain from "$lib/components/ui/ComparisonCurtain.svelte";
	import RadialTimePicker from "$lib/components/ui/RadialTimePicker.svelte";
	import LayerControls from "$lib/components/ui/LayerControls.svelte";
	import ColorLegend from "$lib/components/ui/ColorLegend.svelte";
	import ProjectSelector from "$lib/components/ui/ProjectSelector.svelte";
	import MetricTooltip from "$lib/components/ui/MetricTooltip.svelte";
	import "$lib/styles/variables.css";
	import { getDefaultAnalysisId } from "$lib/config/projects";
	import {
		resolveModelPath,
		resolveProjectId,
	} from "$lib/utils/analysisPaths";
	import { getInitialAnalysisId } from "$lib/utils/analysisQuery";
	import nurLogo from "$lib/assets/Nur Logo white.svg";
	import mitLogo from "$lib/assets/MIT.svg";
	import bguLogo from "$lib/assets/bgu-logo.svg";
	import sceLogo from "$lib/assets/sce-logo.svg";
	import * as THREE from "three";
	import type { Group, Mesh, PerspectiveCamera } from "three";
	import { getTooltipData } from "$lib/services/tooltipService";
	import {
		sceneConfigStore,
		updateSceneConfigFromBounds,
	} from "$lib/stores/sceneConfigStore";
	import type { Analysis } from "$lib/types/analysis";
import { createLiveUtciAnalysisFromCompute } from "$lib/compute/liveUtciAnalysis";
import { createWebgpuUtciPipeline } from "$lib/compute/webgpuUtciPipeline";
import { comparisonStore, curtainPosition } from "$lib/stores/comparisonStore";
import { get } from "svelte/store";

	const getDataBasePath = () => {
		const basePath = base || "";
		return basePath.replace(/\/viewer\/build$/, "");
	};

	// Hard-coded EPW mapping for current projects.
	const getEpwUrlForProject = (projectId: string): string => {
		const basePath = getDataBasePath();
		if (projectId === "Ben-Gurion") {
			return `${basePath}/data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw`;
		}
		// Default to Tel Aviv / Bet Dagan for Ness-Tziona and others.
		return `${basePath}/data/weather/ISR_TA_Tel.Aviv-Bet.Dagan.401790_TMYx/ISR_TA_Tel.Aviv-Bet.Dagan.401790_TMYx.epw`;
	};

	let model: Group | null = null;
	let gridVisible = false;

	let modelLoading = true;
	let hasFitOnce = false;
	let lastModelFile: string | null = null;

	const DEFAULT_ANALYSIS_ID = getDefaultAnalysisId();
	let analysisId: string = DEFAULT_ANALYSIS_ID;
	let mounted = false;

	// Tooltip state
	let tooltipVisible = false;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipValue: number | null = null;
	let utciMesh: Mesh | null = null;
	let liveUtciMesh: Mesh | null = null;
	let canvasElement: HTMLCanvasElement | null = null;

	// Camera reference - will be set from Camera component
	let cameraRef: PerspectiveCamera | undefined = undefined;

	// Main viewport element reference for curtain positioning
	let mainViewportElement: HTMLElement | null = null;

	// Live analysis state
	let liveAnalysis: Analysis | null = null;
	let liveLoading = false;
	let liveError: string | null = null;
	let lastLiveKey: string | null = null;

	function selectSampleMesh(group: Group): Mesh | null {
		let firstMesh: Mesh | null = null;
		group.traverse((child) => {
			if (child instanceof THREE.Mesh && !firstMesh) {
				firstMesh = child;
			}
		});
		return firstMesh;
	}

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
			console.error("[ERROR] Failed to load analysis:", error);
			setError(
				error instanceof Error
					? error.message
					: "Failed to load analysis",
			);
		} finally {
			setLoading(false);
		}
	}

	async function handleProjectSelection(newAnalysisId: string) {
		if (!newAnalysisId || newAnalysisId === analysisId) return;
		analysisId = newAnalysisId;
		if (typeof window !== "undefined") {
			const url = new URL(window.location.href);
			url.searchParams.set("analysis", newAnalysisId);
			goto(`${url.pathname}?${url.searchParams.toString()}`, {
				replaceState: true,
				noScroll: true,
			});
		}
		await loadAnalysis(newAnalysisId);
	}

	onMount(() => {
		if (typeof window !== "undefined") {
			analysisId = getInitialAnalysisId(
				window.location.search,
				DEFAULT_ANALYSIS_ID,
			);

			console.log("[OK] Debug WebGPU UTCI viewer initialized");
			mounted = true;
			loadAnalysis(analysisId);
		}
	});

	$: if (typeof window !== "undefined" && $page.url.searchParams && mounted) {
		const newAnalysisId = getInitialAnalysisId(
			`?${$page.url.searchParams.toString()}`,
			DEFAULT_ANALYSIS_ID,
		);
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

	// Live analysis trigger: when base analysis or model changes, recompute.
	async function computeLiveAnalysis() {
		const base = $analysisStore;
		if (!base || !model) {
			liveAnalysis = null;
			liveError = null;
			return;
		}

		const projectId = resolveProjectId(analysisId) ?? "Ben-Gurion";
		const epwUrl = getEpwUrlForProject(projectId);
		const sampleMesh = selectSampleMesh(model);
		if (!sampleMesh) {
			console.warn("[DEBUG UTCI] Could not find sample mesh for live grid.");
			liveAnalysis = null;
			return;
		}

		const liveKey = `${base.metadata.model_file}|${base.metadata.grid_size}|${analysisId}`;
		if (liveKey === lastLiveKey && liveAnalysis) {
			return;
		}

		lastLiveKey = liveKey;
		liveLoading = true;
		liveError = null;

		try {
			const response = await fetch(epwUrl);
			if (!response.ok) {
				throw new Error(
					`Failed to load EPW file for project ${projectId}: ${response.status}`,
				);
			}
			const epwContent = await response.text();

			const pipeline = await createWebgpuUtciPipeline();

			const result = await createLiveUtciAnalysisFromCompute(
				{
					analysisId,
					baseMetadata: base.metadata,
					sampleMesh,
					epwContent,
					gridResolution: base.metadata.grid_size || 2,
					zHeight: 1.5,
					numHours: base.data.numHours ?? base.metadata.hours.length ?? 24,
					startMonth: 8,
				},
				{ pipeline },
			);

			liveAnalysis = result;

			// Treat the live WebGPU analysis as the comparison analysis so that
			// unifiedUtciRange can provide a shared color scale across .bin and
			// live UTCI surfaces in this debug view.
			comparisonStore.update((state) => ({
				...state,
				isComparing: true,
				comparisonAnalysis: result,
			}));
		} catch (error) {
			console.error("[DEBUG UTCI] Failed to compute live UTCI:", error);
			liveError =
				error instanceof Error
					? error.message
					: "Failed to compute live UTCI";
			liveAnalysis = null;
		} finally {
			liveLoading = false;
		}
	}

	$: if ($analysisStore && model && mounted) {
		// Fire and forget; internal guards prevent excessive recomputation.
		void computeLiveAnalysis();
	}

	// Tooltip: support hover on both .bin (left) and live WebGPU (right) UTCI.
	let lastTooltipUpdate = 0;
	const TOOLTIP_THROTTLE_MS = 16;

	function handleMouseMove(event: MouseEvent) {
		const now = performance.now();
		if (now - lastTooltipUpdate < TOOLTIP_THROTTLE_MS) {
			return;
		}
		lastTooltipUpdate = now;

		if (
			!$viewerStore.utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			tooltipVisible = false;
			return;
		}

		const canvasRect = canvasElement.getBoundingClientRect();

		// Decide which UTCI mesh and analysis to sample based on curtain
		// position, so hover works naturally on both sides.
		const relativeX = (event.clientX - canvasRect.left) / canvasRect.width;
		const curtain = get(curtainPosition);

		let targetMesh: Mesh | null = null;
		let targetAnalysis: Analysis | null = null;

		if (relativeX <= curtain) {
			targetMesh = utciMesh;
			targetAnalysis = $analysisStore;
		} else {
			targetMesh = liveUtciMesh;
			targetAnalysis = liveAnalysis;
		}

		const tooltipData = getTooltipData(
			event,
			cameraRef,
			targetMesh,
			targetAnalysis,
			$viewerStore.metricType,
			$viewerStore.currentHour,
			canvasRect,
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

	let eventListenersAttached = false;

	$: if (canvasElement && mounted && !eventListenersAttached) {
		const canvas = canvasElement;
		canvas.addEventListener("mousemove", handleMouseMove, {
			passive: true,
		});
		canvas.addEventListener("mouseleave", handleMouseLeave, {
			passive: true,
		});
		eventListenersAttached = true;
	}

	onDestroy(() => {
		if (canvasElement && eventListenersAttached) {
			canvasElement.removeEventListener("mousemove", handleMouseMove);
			canvasElement.removeEventListener("mouseleave", handleMouseLeave);
			eventListenersAttached = false;
		}
	});

	$: currentProjectId = resolveProjectId(analysisId) ?? "Ben-Gurion";
</script>

<svelte:head></svelte:head>

<div class="viewer-shell">
	<header class="app-header">
		<div class="header-left">
			<div class="partner-logos">
				<img
					src={nurLogo}
					alt="NUR Negev Urban Research"
					class="logo logo-nur"
				/>
				<img src={bguLogo} alt="BGU" class="logo logo-bgu" />
				<img src={mitLogo} alt="MIT" class="logo logo-mit" />
				<img src={sceLogo} alt="SCE" class="logo logo-sce" />
			</div>
		</div>
		<div class="header-center">
			<div class="header-title">
				<div class="logo-final">
					<div class="text">Score.CH</div>
					<div class="underline-grad"></div>
				</div>
				<div class="debug-label">
					WebGPU UTCI Debug Viewer · .bin vs live compute (no parity
					guaranteed)
				</div>
			</div>
		</div>
		<div class="header-right">
			{#key analysisId}
				<ProjectSelector
					analysisId={analysisId}
					onSelect={handleProjectSelection}
				/>
			{/key}
		</div>
	</header>

	<div class="app-body">
		<aside class="app-sidebar">
			<div class="sidebar-section">
				<div class="section-header">Project & Time</div>
				<div class="section-subtitle">
					Select project and UTCI hour; left side uses .bin, right side
					uses live compute.
				</div>
				{#if $analysisStore && $analysisStore.metadata.analysis_type === "full_day" && $viewerStore.metricType === "utci"}
					<RadialTimePicker />
				{/if}
			</div>

			<div class="sidebar-section layers-sidebar-section">
				<div class="section-header">Layers</div>
				<LayerControls placement="sidebar" />
			</div>

			{#if liveLoading}
				<div class="sidebar-section">
					<div class="section-header">Live UTCI</div>
					<div class="section-subtitle">
						Computing live UTCI via debug pipeline…
					</div>
				</div>
			{:else if liveError}
				<div class="sidebar-section">
					<div class="section-header">Live UTCI</div>
					<div class="section-subtitle error">
						Failed to compute live UTCI: {liveError}
					</div>
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
				<div class="overlay-message error">
					Error: {$viewerStore.error}
				</div>
			{/if}

			{#if modelLoading}
				<div
					class="model-loading-backdrop"
					aria-hidden="true"
				></div>
				<div
					class="model-loading-overlay"
					aria-live="polite"
				>
					<div class="spinner"></div>
					<div class="loading-text">Preparing model…</div>
				</div>
			{/if}

			<Scene
				backgroundColor={$viewerStore.theme === "light"
					? 0x4b5563
					: 0x111827}
				bind:canvasElement
			>
				<Camera
					bind:cameraRef
					near={$sceneConfigStore.cameraNear}
					far={$sceneConfigStore.cameraFar}
				/>
				<Lights />

				{#if $analysisStore}
					{#key $analysisStore.metadata.model_file}
						<Model
							modelPath={resolveModelPath(
								$analysisStore.metadata.model_file,
								analysisId,
							).replace("data/", `${getDataBasePath()}/data/`)}
							coordinateSystem={$analysisStore.metadata
								.coordinate_system || "xy_ground"}
							metadata={$analysisStore.metadata}
							on:modelLoaded={(e) => {
								model = e.detail;
								modelLoading = false;
								if (model) {
									const bounds = calculateModelBounds(model);
									const center = calculateModelCenter(model);
									const size = calculateModelSize(model);
									updateSceneConfigFromBounds(bounds);
									if (!hasFitOnce) {
										const maxDim = Math.max(
											size.x,
											size.y,
											size.z,
										);
										const distance = maxDim * 1.05;
										const position = center
											.clone()
											.add(
												new THREE.Vector3(
													0,
													distance,
													0.01,
												),
											);
										cameraStore.update((state) => ({
											...state,
											position,
											target: center.clone(),
										}));
										hasFitOnce = true;
									}
								}
							}}
							on:layersDiscovered={(e) => {
								setDiscoveredLayers(e.detail);
							}}
						/>
					{/key}

					{#if model}
						<GridHelper {model} visible={gridVisible} />
						<!-- Left: .bin-backed UTCI -->
						<UTCIPointCloud
							analysis={$analysisStore}
							{model}
							bind:utciSurface={utciMesh}
						/>

						<!-- Right: live-computed UTCI using adapter -->
						{#if liveAnalysis}
							<UTCIPointCloud
								analysis={liveAnalysis}
								{model}
								bind:utciSurface={liveUtciMesh}
							/>
						{/if}

						{#if utciMesh && liveUtciMesh && cameraRef}
							<DebugUtciScissor
								baseCamera={cameraRef}
								binUtciMesh={utciMesh}
								liveUtciMesh={liveUtciMesh}
							/>
						{/if}
					{/if}
				{/if}
			</Scene>

			<!-- Comparison curtain overlay reused for visual UX -->
			<ComparisonCurtain
				containerElement={mainViewportElement}
				comparisonScenarioName="Live WebGPU UTCI"
			/>
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
		background: radial-gradient(
				circle at top left,
				rgba(56, 189, 248, 0.18),
				transparent 55%
			),
			var(--color-bg-page);
	}

	.app-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
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
		flex: 1 1 0;
		min-width: 0;
		justify-content: flex-start;
		overflow: hidden;
	}

	.header-title {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
	}

	.logo-final {
		position: relative;
		font-family: "Space Grotesk", sans-serif;
	}

	.logo-final .text {
		font-size: 34px;
		font-weight: 700;
		color: var(--color-text-primary);
		letter-spacing: -0.03em;
		padding-bottom: 2px;
	}

	.logo-final .underline-grad {
		position: absolute;
		bottom: -2px;
		left: 0;
		right: 0;
		height: 4px;
		background: linear-gradient(
			90deg,
			#313695,
			#4575b4,
			#74add1,
			#abd9e9,
			#e0f3f8,
			#ffffbf,
			#fee090,
			#fdae61,
			#f46d43,
			#d73027,
			#a50026
		);
		border-radius: 2px;
		opacity: 0.9;
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
	}

	.debug-label {
		font-size: 11px;
		color: var(--color-text-secondary);
		margin-top: 6px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.header-center {
		display: flex;
		align-items: center;
		flex: 0 0 auto;
		justify-content: center;
		min-width: 0;
	}

	.header-right {
		display: flex;
		align-items: center;
		flex: 1 1 0;
		min-width: 0;
		justify-content: flex-end;
	}

	.partner-logos {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: nowrap;
		max-width: 100%;
	}

	.logo {
		height: 30px;
		object-fit: contain;
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.4));
		display: block;
	}

	.logo-nur {
		height: 50px;
	}

	.logo-bgu {
		height: 35px;
	}

	.app-body {
		flex: 1;
		display: grid;
		grid-template-columns: minmax(320px, 320px) 1fr;
		grid-template-areas: "sidebar main";
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
		position: relative;
		box-shadow: 2px 0 12px rgba(0, 0, 0, 0.12);
	}

	.app-main {
		grid-area: main;
		position: relative;
		background: var(--color-bg-page);
		min-width: 0;
		overflow: hidden;
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

	.section-header {
		font-size: var(--font-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		margin-bottom: 8px;
		color: var(--color-text-secondary);
	}

	.section-subtitle {
		font-size: var(--font-sm);
		color: var(--color-text-muted);
		margin-bottom: 8px;
	}

	.section-subtitle.error {
		color: var(--color-danger);
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
		pointer-events: none;
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

