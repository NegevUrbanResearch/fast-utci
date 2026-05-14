<script lang="ts">
	/**
	 * Main Page Component
	 *
	 * ABOUTME: The main viewer page that integrates the 3D scene, UI panels, and comparison features.
	 * When comparison mode is active, renders both base and comparison scenes with a curtain slider.
	 */
	import { onMount, onDestroy } from "svelte";
	import { page } from "$app/stores";
	import { goto } from "$app/navigation";
	import { base } from "$app/paths";
	import { analysisStore, loadAnalysisData } from "$lib/stores/analysisStore";
	import {
		viewerStore,
		setAnalysisId,
		setLoading,
		setError,
	} from "$lib/stores/viewerStore";
	import { cameraStore, focusCameraOnModel } from "$lib/stores/cameraStore";
	import { setDiscoveredLayers } from "$lib/stores/layerStore";
	import {
		comparisonStore,
		comparisonAnalysis,
	} from "$lib/stores/comparisonStore";
	import {
		EMPTY_PERFORMANCE_SNAPSHOT,
		performanceStore,
	} from "$lib/stores/performanceStore";
	import {
		calculateModelBounds,
		calculateModelCenter,
		calculateModelSize,
	} from "$lib/utils/bounds";
	import type {
		WebgpuLargeBufferDeviceLimits,
		WebgpuLargeBufferRequiredLimits,
	} from "$lib/compute/gpu/webgpuDeviceLimits";
	import type ComparisonRenderer from "$lib/components/scene/ComparisonRenderer.svelte";
	import RadialTimePicker from "$lib/components/ui/RadialTimePicker.svelte";
	import LayerControls from "$lib/components/ui/LayerControls.svelte";
	import ColorLegend from "$lib/components/ui/ColorLegend.svelte";
	import ScenarioSelector from "$lib/components/ui/ScenarioSelector.svelte";
	import ProjectSelector from "$lib/components/ui/ProjectSelector.svelte";
	import PerformancePanel from "$lib/components/ui/PerformancePanel.svelte";
	import ViewerShell from "$lib/components/viewer/ViewerShell.svelte";
	import "$lib/styles/variables.css";
	import { getDefaultAnalysisId } from "$lib/config/projects";
	import { resolveProjectId } from "$lib/utils/analysisPaths";
	import type { Analysis } from "$lib/types/analysis";
	import type { LiveSelectedHourControllerSurfaceDiagnostics } from "$lib/compute/selected-hour/liveSelectedHourController";
	import type { LiveSelectedHourPublishedRenderContext } from "$lib/compute/selected-hour/liveSelectedHourRenderContext";
	import { projectMainRouteLiveSceneState } from "$lib/compute/selected-hour/liveSelectedHourRouteProjection";
	import type { LiveSelectedHourSurfaceIdentity } from "$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity";
	import { createLiveSelectedHourRouteHost } from "$lib/compute/selected-hour/liveSelectedHourRouteHost";
	import { resolveLiveSelectedHourTimeIndex } from "$lib/compute/selected-hour/liveUtciSelectedHour";
	import type { SelectedHourGpuResidentOutput } from "$lib/compute/selected-hour/liveUtciSelectedHourSession";
	import type { Group, Mesh, PerspectiveCamera } from "three";
	import {
		sceneConfigStore,
		updateSceneConfigFromBounds,
	} from "$lib/stores/sceneConfigStore";
	import {
		DEFAULT_MAIN_UTCI_RENDER_MODE,
		parseUtciRenderMode,
		resolveMainRouteUtciSurfaceBackend,
		type UtciRendererBackend,
	} from "$lib/utciRenderMode";
	import { buildMainRoutePerformanceSnapshot } from "$lib/performance/mainRoutePerformanceTelemetry";
	import { getMainRouteOverlayGating } from "./mainRouteOverlayGating";
	import {
		publishMainRouteUtciDiagnostics,
		releaseBaseAcceptedGpuResidentOutput,
		releaseComparisonAcceptedGpuResidentOutput,
		type MainRouteAcceptedGpuResidentOutputReleaseParams,
		type MainRouteLiveSelectedHourDiagnosticsParams,
		type MainRouteWindow,
	} from "./main/liveSelectedHour";
	import {
		buildProjectSelectionHref,
		getAnalysisSyncAfterMount,
		getModelReloadState,
		getMountedAnalysisId,
	} from "./main/modelSelection";
	import { getMainRouteModelLoadedEffects } from "./main/modelLifecycle";
	import MainRouteOverlays from "./main/MainRouteOverlays.svelte";
	import MainRouteTooltipLayer from "./main/MainRouteTooltipLayer.svelte";
	import MainRouteViewport from "./main/MainRouteViewport.svelte";

	const getDataBasePath = () => {
		const basePath = base || "";
		return basePath.replace(/\/viewer\/build$/, "");
	};

	let model: Group | null = null;
	let gridVisible = false;

	let performanceOpen = false;
	let modelLoading = true;
	let hasFitOnce = false;
	let lastModelFile: string | null = null;

	const DEFAULT_ANALYSIS_ID = getDefaultAnalysisId();
	let analysisId: string = DEFAULT_ANALYSIS_ID;
	let mounted = false;
	// Main route UTCI rendering follows the selected/default render mode.
	$: utciRenderMode = parseUtciRenderMode(
		$page.url.searchParams,
		DEFAULT_MAIN_UTCI_RENDER_MODE,
	);
	type UtciOnDemandMode = "f32";
	const utciOnDemandMode: UtciOnDemandMode = "f32";
	let rendererBackend: UtciRendererBackend = "unknown";
	let rendererDeviceForMain: GPUDevice | undefined = undefined;
	$: resolvedUtciSurfaceBackend = resolveMainRouteUtciSurfaceBackend({
		mode: utciRenderMode,
		rendererBackend,
		isComparing: $comparisonStore.isComparing,
	});
	$: utciRenderDiagnosticsEnabled =
		$page.url.searchParams.get("utciRenderDiagnostics") === "1";

	type MainRouteUtciSurfaceDiagnostics = LiveSelectedHourControllerSurfaceDiagnostics;

	let rendererRequiredLimits: WebgpuLargeBufferRequiredLimits | undefined = undefined;
	let rendererDeviceLimits: WebgpuLargeBufferDeviceLimits | undefined = undefined;
	let lastBaseGpuResidentCopyFailure:
		| { error?: string; requestId?: number }
		| undefined = undefined;
	$: requestLargeWebgpuLimits = utciRenderMode !== "data";
	const liveRouteHost = createLiveSelectedHourRouteHost({
		dataBasePath: getDataBasePath(),
	});
	let liveRouteState = liveRouteHost.getState();
	const unsubscribeLiveRouteHost = liveRouteHost.subscribe((state) => {
		liveRouteState = state;
	});
	let comparisonModelForLiveCompute: Group | null = null;
	let baseDisplayedAnalysis: Analysis | null = null;
	let comparisonRendererDisplayAnalysis: Analysis | null | undefined = undefined;
	let useLiveUtciOnMainRoute = false;
	let baseLiveReady = false;
	let comparisonLiveReady = true;
	let baseHasVisibleLiveSurface = false;
	let comparisonHasVisibleLiveSurface = false;
	let baseSceneAnalysis: Analysis | null = null;
	let comparisonSceneAnalysis: Analysis | null | undefined = undefined;
	let baseSceneRenderContext: LiveSelectedHourPublishedRenderContext | null = null;
	let comparisonSceneRenderContext:
		| LiveSelectedHourPublishedRenderContext
		| null
		| undefined = undefined;
	let baseSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null = null;
	let comparisonSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null | undefined = undefined;
	let basePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	let comparisonPendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	let basePendingRenderUpdateStartedAt: number | undefined = undefined;
	let comparisonPendingRenderUpdateStartedAt: number | undefined = undefined;
	let showMainRouteOverlay = false;
	let showMainRouteComparisonOverlay = false;
	let tooltipHoverSampleCount = 0;
	let cameraWheelEventCount = 0;

	function updateUtciRenderDiagnostics(
		params: MainRouteLiveSelectedHourDiagnosticsParams,
	): void {
		if (typeof window === "undefined") return;

		const win = window as MainRouteWindow;
		publishMainRouteUtciDiagnostics(win, params);
	}

	function handleRendererDiagnostics(diagnostics: {
		rendererBackend: UtciRendererBackend;
		rendererDevice?: GPUDevice;
		rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
		rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
		error?: string;
	}): void {
		rendererBackend = diagnostics.rendererBackend;
		rendererDeviceForMain = diagnostics.rendererDevice;
		rendererRequiredLimits = diagnostics.rendererRequiredLimits;
		rendererDeviceLimits = diagnostics.rendererDeviceLimits;
	}

	function handleUtciSurfaceDiagnostics(
		diagnostics: MainRouteUtciSurfaceDiagnostics,
	): void {
		if (diagnostics.gpuResidentCopyStatus === "failed") {
			lastBaseGpuResidentCopyFailure = {
				error: diagnostics.gpuResidentCopyError,
				requestId: diagnostics.gpuResidentCopyRequestId,
			};
		}
		liveRouteHost.handleBaseSurfaceDiagnostics(diagnostics);
	}

	function handleComparisonUtciSurfaceDiagnostics(
		diagnostics: MainRouteUtciSurfaceDiagnostics,
	): void {
		liveRouteHost.handleComparisonSurfaceDiagnostics(diagnostics);
	}

	function handleBaseAcceptedGpuResidentOutputRelease(
		params: MainRouteAcceptedGpuResidentOutputReleaseParams,
	): void {
		releaseBaseAcceptedGpuResidentOutput(liveRouteHost, params);
	}

	function handleComparisonAcceptedGpuResidentOutputRelease(
		params: MainRouteAcceptedGpuResidentOutputReleaseParams,
	): void {
		releaseComparisonAcceptedGpuResidentOutput(liveRouteHost, params);
	}

	function handleMainRouteModelLoaded(event: CustomEvent<Group>): void {
		model = event.detail;
		modelLoading = false;
		if (!model) return;

		const bounds = calculateModelBounds(model);
		const center = calculateModelCenter(model);
		const size = calculateModelSize(model);
		const effects = getMainRouteModelLoadedEffects({
			bounds,
			center,
			size,
			hasFitOnce,
		});

		updateSceneConfigFromBounds(effects.sceneBounds);
		if (effects.cameraFit) {
			cameraStore.update((state) => ({
				...state,
				position: effects.cameraFit!.position,
				target: effects.cameraFit!.target,
			}));
		}
		hasFitOnce = effects.nextHasFitOnce;
	}

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
	$: currentProjectId = resolveProjectId(analysisId) ?? "Ben-Gurion";
	$: comparisonModelForLiveCompute =
		isComparing && comparisonRenderer && !$comparisonStore.modelLoading
			? comparisonRenderer.getComparisonModel()
			: null;
	$: useLiveUtciOnMainRoute =
		$viewerStore.metricType === "utci" &&
		$analysisStore?.metadata.analysis_type === "full_day";
	$: selectedMonthIndex = $viewerStore.currentMonth ?? 7;
	$: selectedHourIndex = $viewerStore.currentHour;
	$: selectedTimeIndex = resolveLiveSelectedHourTimeIndex({
		monthIndex: selectedMonthIndex,
		hourIndex: selectedHourIndex,
	});
	$: liveRouteHost.setRouteInputs({
		enabled: useLiveUtciOnMainRoute,
		analysisId,
		baseAnalysis: $analysisStore,
		baseModel: modelLoading ? null : model,
		selection: {
			monthIndex: selectedMonthIndex,
			hourIndex: selectedHourIndex,
			timeIndex: selectedTimeIndex,
			selectionKey: [analysisId, selectedMonthIndex, selectedHourIndex].join("|"),
		},
		colorMode: $viewerStore.colorMode,
		utciRenderMode,
		rendererBackend,
		rendererDevice: rendererDeviceForMain,
		utciSurfaceBackend: resolvedUtciSurfaceBackend,
		comparison: {
			active: isComparing,
			analysisId: $comparisonStore.comparisonAnalysisId,
			sourceAnalysis: $comparisonStore.isLoading ? null : $comparisonAnalysis,
			model: comparisonModelForLiveCompute,
			rendererDevice: rendererDeviceForMain,
		},
	});
	$: ({
		baseDisplayedAnalysis,
		comparisonRendererDisplayAnalysis,
		baseLiveReady,
		comparisonLiveReady,
		baseHasVisibleLiveSurface,
		comparisonHasVisibleLiveSurface,
		baseSceneAnalysis,
		comparisonSceneAnalysis,
		baseSceneRenderContext,
		comparisonSceneRenderContext,
		baseSceneSurfaceIdentity,
		comparisonSceneSurfaceIdentity,
		basePendingGpuResidentOutput,
		comparisonPendingGpuResidentOutput,
		basePendingRenderUpdateStartedAt,
		comparisonPendingRenderUpdateStartedAt,
	} = projectMainRouteLiveSceneState({
		useLiveUtciOnMainRoute,
		isComparing,
		baseAnalysis: $analysisStore,
		comparisonAnalysis: $comparisonAnalysis,
		liveRouteState,
	}));
	$: ({
		showOverlay: showMainRouteOverlay,
		showComparisonModeOverlay: showMainRouteComparisonOverlay,
	} = getMainRouteOverlayGating({
		modelLoading,
		useLiveUtciOnMainRoute,
		baseLiveLoading: liveRouteState.base.loading,
		baseHasVisibleLiveSurface,
		isComparing: $comparisonStore.isComparing,
		comparisonModelLoading: $comparisonStore.modelLoading,
		comparisonLiveLoading: liveRouteState.comparison.loading,
		comparisonHasVisibleLiveSurface,
	}));

	// Reactive scenario name for comparison curtain label
	// Watch comparisonAnalysisId to trigger updates when scenarios change
	$: comparisonScenarioName =
		isComparing && scenarioSelector && $comparisonStore.comparisonAnalysisId
			? scenarioSelector.getScenarioName()
			: "Comparison";

	async function loadAnalysis(id: string) {
		try {
			modelLoading = true;
			setLoading(true);
			setError(null);
			setAnalysisId(id);
			await loadAnalysisData(id, undefined, { metadataOnly: true });

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
			goto(buildProjectSelectionHref(window.location.href, newAnalysisId), {
				replaceState: true,
				noScroll: true,
			});
		}
		await loadAnalysis(newAnalysisId);
	}

	onMount(() => {
		if (typeof window !== "undefined") {
			analysisId = getMountedAnalysisId(
				window.location.search,
				DEFAULT_ANALYSIS_ID,
			);

			console.log("[OK] Viewer initialized");
			mounted = true;
			loadAnalysis(analysisId);
		}
	});

	$: if (typeof window !== "undefined" && $page.url.searchParams && mounted) {
		const syncResult = getAnalysisSyncAfterMount({
			mounted,
			currentAnalysisId: analysisId,
			pageSearchParams: $page.url.searchParams,
			defaultAnalysisId: DEFAULT_ANALYSIS_ID,
		});
		if (syncResult.shouldLoad) {
			analysisId = syncResult.analysisId;
			loadAnalysis(analysisId);
		}
	}

	// Trigger model loading overlay when the model file changes
	$: if ($analysisStore && $analysisStore.metadata?.model_file) {
		const modelReloadState = getModelReloadState({
			currentModelFile: $analysisStore.metadata.model_file,
			lastModelFile,
		});
		if (modelReloadState.shouldResetModel) {
			modelLoading = true;
			model = null;
		}
		lastModelFile = modelReloadState.nextLastModelFile;
	}

	$: if (typeof window !== "undefined") {
		updateUtciRenderDiagnostics({
			enabled: utciRenderDiagnosticsEnabled,
			utciOnDemand: utciOnDemandMode,
			utciRenderRequested: utciRenderMode,
			utciRenderResolved: resolvedUtciSurfaceBackend,
			rendererBackend,
			rendererRequiredLimits,
			rendererDeviceLimits,
			liveRouteState,
			lastBaseGpuResidentCopyFailure,
			baseLiveReady,
			comparisonLiveReady,
			selectedMonthIndex,
			selectedHourIndex,
			selectedTimeIndex,
			baseSceneRenderContextTimeIndex: baseSceneRenderContext?.timeIndex,
			baseAcceptedUtciRange: basePendingGpuResidentOutput?.utciRange ?? undefined,
			tooltipHoverSampleCount,
			cameraWheelEventCount,
		});
	}

	$: if (typeof window !== "undefined") {
		performanceStore.set(
			buildMainRoutePerformanceSnapshot({
				analysisId,
				projectLabel: currentProjectId,
				pointCount: $analysisStore?.metadata?.num_positions ?? null,
				gridSizeMeters: $analysisStore?.metadata?.grid_size ?? null,
				selectedMonthIndex,
				selectedHourIndex,
				diagnostics: {
					baseLiveReady,
					timings: liveRouteState.base.runtimeDiagnostics?.timings,
					trackedGpuAllocationBytes:
						liveRouteState.base.runtimeDiagnostics?.trackedGpuAllocationBytes,
					error: liveRouteState.base.error ?? undefined,
				},
				now: performance.now(),
			}),
		);
	}

	onDestroy(() => {
		if (typeof window !== "undefined") {
			(window as MainRouteWindow).__utciRenderDiagnostics__ = undefined;
		}
		performanceStore.set(EMPTY_PERFORMANCE_SNAPSHOT);

		unsubscribeLiveRouteHost();
		liveRouteHost.dispose();
	});
</script>

<svelte:head></svelte:head>

<ViewerShell
	bind:mainViewportElement
	showTimeSection={$analysisStore != null &&
		$analysisStore.metadata.analysis_type === "full_day" &&
		$viewerStore.metricType === "utci"}
>
	<svelte:fragment slot="headerRight">
		{#key analysisId}
			<ProjectSelector
				analysisId={analysisId}
				onSelect={handleProjectSelection}
			/>
		{/key}
	</svelte:fragment>

	<svelte:fragment slot="scenario">
		<div class="section-header">Scenario</div>
		<ScenarioSelector
			bind:this={scenarioSelector}
			projectId={currentProjectId}
			mode="compare"
		/>
	</svelte:fragment>

	<svelte:fragment slot="analytics">
		<button
			type="button"
			class="section-header section-header-toggle"
			on:click={() => (performanceOpen = !performanceOpen)}
		>
			<span>Performance</span>
			<span class:open={performanceOpen} class="chevron">v</span>
		</button>
		{#if performanceOpen}
			<PerformancePanel />
		{/if}
	</svelte:fragment>

	<svelte:fragment slot="layers">
		<div class="section-header">Layers</div>
		<LayerControls placement="sidebar" />
	</svelte:fragment>

	<svelte:fragment slot="time">
		{#if $analysisStore && $analysisStore.metadata.analysis_type === "full_day" && $viewerStore.metricType === "utci"}
			<div class="section-header">Time of Day</div>
			<div class="section-subtitle">
				Select analysis hour for UTCI
			</div>
			<RadialTimePicker />
		{/if}
	</svelte:fragment>

	<svelte:fragment slot="legend">
		<ColorLegend
			displayAnalysis={useLiveUtciOnMainRoute ? baseDisplayedAnalysis : null}
			utciRangeOverride={useLiveUtciOnMainRoute
				? (basePendingGpuResidentOutput?.utciRange ?? null)
				: undefined}
		/>
	</svelte:fragment>

	<svelte:fragment slot="tooltip">
		<MainRouteTooltipLayer
			bind:tooltipHoverSampleCount
			bind:cameraWheelEventCount
			{canvasElement}
			{cameraRef}
			baseMesh={utciMesh}
			{baseDisplayedAnalysis}
			baseSceneTimeIndex={baseSceneRenderContext?.timeIndex}
			comparisonDisplayedAnalysis={useLiveUtciOnMainRoute
				? comparisonRendererDisplayAnalysis
				: $comparisonAnalysis}
			comparisonSceneTimeIndex={comparisonSceneRenderContext?.timeIndex}
			getComparisonUtciMesh={() => comparisonRenderer?.getComparisonUtciMesh() ?? null}
			{useLiveUtciOnMainRoute}
			{isComparing}
			{mainViewportElement}
			curtainPosition={$comparisonStore.curtainPosition}
			viewerCurrentHour={$viewerStore.currentHour}
			metricType={$viewerStore.metricType}
			utciVisible={$viewerStore.utciVisible}
		/>
	</svelte:fragment>

	<svelte:fragment slot="overlays">
		<MainRouteOverlays
			loading={$viewerStore.loading}
			error={$viewerStore.error}
			baseLiveError={liveRouteState.base.error}
			comparisonLiveError={liveRouteState.comparison.error}
			{showMainRouteOverlay}
			{showMainRouteComparisonOverlay}
			curtainPosition={$comparisonStore.curtainPosition}
			{modelLoading}
			comparisonModelLoading={$comparisonStore.modelLoading}
			{useLiveUtciOnMainRoute}
			{isComparing}
			{mainViewportElement}
			{comparisonScenarioName}
		/>
	</svelte:fragment>

	<svelte:fragment slot="viewport">
		<MainRouteViewport
			bind:canvasElement
			bind:cameraRef
			bind:utciMesh
			bind:comparisonRenderer
			analysis={$analysisStore}
			{analysisId}
			dataBasePath={getDataBasePath()}
			theme={$viewerStore.theme}
			{requestLargeWebgpuLimits}
			cameraNear={$sceneConfigStore.cameraNear}
			cameraFar={$sceneConfigStore.cameraFar}
			{gridVisible}
			{model}
			{isComparing}
			{baseSceneAnalysis}
			{comparisonSceneAnalysis}
			{basePendingGpuResidentOutput}
			{comparisonPendingGpuResidentOutput}
			{baseSceneRenderContext}
			{comparisonSceneRenderContext}
			{baseSceneSurfaceIdentity}
			{comparisonSceneSurfaceIdentity}
			{basePendingRenderUpdateStartedAt}
			{comparisonPendingRenderUpdateStartedAt}
			{resolvedUtciSurfaceBackend}
			onRendererDiagnostics={handleRendererDiagnostics}
			onBaseUtciSurfaceDiagnostics={handleUtciSurfaceDiagnostics}
			onComparisonUtciSurfaceDiagnostics={handleComparisonUtciSurfaceDiagnostics}
			onBaseAcceptedGpuResidentOutputRelease={handleBaseAcceptedGpuResidentOutputRelease}
			onComparisonAcceptedGpuResidentOutputRelease={handleComparisonAcceptedGpuResidentOutputRelease}
			on:modelLoaded={handleMainRouteModelLoaded}
			on:layersDiscovered={(event) => setDiscoveredLayers(event.detail)}
		/>
	</svelte:fragment>
</ViewerShell>
