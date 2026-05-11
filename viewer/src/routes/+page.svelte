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
		calculateModelBounds,
		calculateModelCenter,
		calculateModelSize,
	} from "$lib/utils/bounds";
	import Scene from "$lib/components/scene/Scene.svelte";
	import type {
		WebgpuLargeBufferDeviceLimits,
		WebgpuLargeBufferRequiredLimits,
	} from "$lib/compute/webgpuDeviceLimits";
	import Camera from "$lib/components/scene/Camera.svelte";
	import Lights from "$lib/components/scene/Lights.svelte";
	import GridHelper from "$lib/components/scene/GridHelper.svelte";
	import Model from "$lib/components/scene/Model.svelte";
	import UTCIPointCloud from "$lib/components/scene/UTCIPointCloud.svelte";
	import ComparisonRenderer from "$lib/components/scene/ComparisonRenderer.svelte";
	import ComparisonCurtain from "$lib/components/ui/ComparisonCurtain.svelte";
	import RadialTimePicker from "$lib/components/ui/RadialTimePicker.svelte";
	import LayerControls from "$lib/components/ui/LayerControls.svelte";
	import ColorLegend from "$lib/components/ui/ColorLegend.svelte";
	import ScenarioSelector from "$lib/components/ui/ScenarioSelector.svelte";
	import ProjectSelector from "$lib/components/ui/ProjectSelector.svelte";
	import AnalyticsPanel from "$lib/components/ui/AnalyticsPanel.svelte";
	import MetricTooltip from "$lib/components/ui/MetricTooltip.svelte";
	import ViewerShell from "$lib/components/viewer/ViewerShell.svelte";
	import {
		createCanvasInteractionController,
		type CanvasInteractionController,
	} from "$lib/components/viewer/canvasInteractionController";
	import "$lib/styles/variables.css";
	import { getDefaultAnalysisId } from "$lib/config/projects";
	import { resolveAnalysisModelPath, resolveProjectId } from "$lib/utils/analysisPaths";
	import type { Analysis } from "$lib/types/analysis";
	import type { LiveSelectedHourControllerSurfaceDiagnostics } from "$lib/compute/liveSelectedHourController";
	import type { LiveSelectedHourPublishedRenderContext } from "$lib/compute/liveSelectedHourRenderContext";
	import { projectMainRouteLiveSceneState } from "$lib/compute/liveSelectedHourRouteProjection";
	import type { LiveSelectedHourSurfaceIdentity } from "$lib/compute/liveSelectedHourSurfaceIdentity";
	import { createLiveSelectedHourRouteHost } from "$lib/compute/liveSelectedHourRouteHost";
	import { resolveLiveSelectedHourTimeIndex } from "$lib/compute/liveUtciSelectedHour";
	import type { SelectedHourGpuResidentOutput } from "$lib/compute/liveUtciSelectedHourSession";
	import * as THREE from "three";
	import type { Group, Mesh, PerspectiveCamera } from "three";
	import { getTooltipData } from "$lib/services/tooltipService";
	import {
		armTooltipMotionSuppression,
		createTooltipMotionSuppressionState,
		releaseTooltipMotionPointer,
		setTooltipMotionPointerDown,
	} from "$lib/services/tooltipMotionSuppression";
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
	import {
		getMainRouteTooltipHoverPolicy,
		resolveMainRouteTooltipTarget,
	} from "./main/tooltip";

	const getDataBasePath = () => {
		const basePath = base || "";
		return basePath.replace(/\/viewer\/build$/, "");
	};

	let model: Group | null = null;
	let gridVisible = false;

	let analyticsOpen = false;
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

	// Tooltip state
	let tooltipVisible = false;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipValue: number | null = null;
	let tooltipPosition: { x: number; y: number; z: number } | null = null;
	let utciMesh: Mesh | null = null;
	let canvasElement: HTMLCanvasElement | null = null;

	// Camera reference - will be set from Camera component
	let cameraRef: PerspectiveCamera | undefined = undefined;
	let tooltipMotionSuppression = createTooltipMotionSuppressionState();

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

	// Throttle tooltip updates for performance
	let lastTooltipUpdate = 0;
	const TOOLTIP_THROTTLE_MS = 16; // ~60fps

	function hideTooltip() {
		tooltipVisible = false;
		tooltipPosition = null;
	}

	function handleTooltipMotionPointerDown() {
		tooltipMotionSuppression = setTooltipMotionPointerDown(
			tooltipMotionSuppression,
			true,
			performance.now(),
		);
		hideTooltip();
	}

	function handleTooltipMotionPointerRelease() {
		const hadCanvasPointerInteraction = tooltipMotionSuppression.pointerDown;
		tooltipMotionSuppression = releaseTooltipMotionPointer(
			tooltipMotionSuppression,
			performance.now(),
		);
		if (hadCanvasPointerInteraction) {
			hideTooltip();
		}
	}

	function handleTooltipMotionWheel() {
		cameraWheelEventCount += 1;
		tooltipMotionSuppression = armTooltipMotionSuppression(
			tooltipMotionSuppression,
			performance.now(),
		);
		hideTooltip();
	}

	// Handle mouse move for tooltip
	function handleMouseMove(event: MouseEvent) {
		tooltipHoverSampleCount += 1;
		const now = performance.now();
		const hoverPolicy = getMainRouteTooltipHoverPolicy({
			tooltipMotionSuppression,
			now,
			lastTooltipUpdate,
			throttleMs: TOOLTIP_THROTTLE_MS,
		});
		if (hoverPolicy.shouldSuppress) {
			hideTooltip();
			return;
		}
		if (hoverPolicy.shouldThrottle) {
			return; // Throttle updates
		}
		lastTooltipUpdate = hoverPolicy.nextTooltipUpdate;

		if (
			!utciMesh ||
			!baseDisplayedAnalysis ||
			!$viewerStore.utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			hideTooltip();
			return;
		}

		const canvasRect = canvasElement.getBoundingClientRect();

		// Determine which side of the comparison curtain the mouse is on
		// If in comparison mode and mouse is on the right side, use comparison data
		const tooltipTarget = resolveMainRouteTooltipTarget({
			baseMesh: utciMesh,
			baseAnalysis: baseDisplayedAnalysis,
			baseSceneTimeIndex: baseSceneRenderContext?.timeIndex,
			comparisonMesh: comparisonRenderer?.getComparisonUtciMesh() ?? null,
			comparisonAnalysis: useLiveUtciOnMainRoute
				? comparisonRendererDisplayAnalysis
				: $comparisonAnalysis,
			comparisonSceneTimeIndex: comparisonSceneRenderContext?.timeIndex,
			useLiveUtciOnMainRoute,
			isComparing,
			mouseClientX: event.clientX,
			mainViewportRect: mainViewportElement
				? mainViewportElement.getBoundingClientRect()
				: null,
			curtainPosition: $comparisonStore.curtainPosition,
			viewerCurrentHour: $viewerStore.currentHour,
		});

		const tooltipData = getTooltipData(
			event,
			cameraRef,
			tooltipTarget.meshToRaycast,
			tooltipTarget.analysisToUse,
			$viewerStore.metricType,
			tooltipTarget.tooltipHourIndex,
			canvasRect,
		);

		if (tooltipData) {
			tooltipVisible = true;
			tooltipX = event.clientX;
			tooltipY = event.clientY;
			tooltipValue = tooltipData.value;
			tooltipPosition = tooltipData.position;
		} else {
			hideTooltip();
		}
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

	function handleMouseLeave() {
		hideTooltip();
	}

	let hoverInteractionController: CanvasInteractionController | null = null;
	let hoverInteractionCanvas: HTMLCanvasElement | null = null;
	let tooltipMotionInteractionController: CanvasInteractionController | null = null;
	let tooltipMotionInteractionCanvas: HTMLCanvasElement | null = null;

	function detachHoverListeners() {
		hoverInteractionController?.dispose();
		hoverInteractionController = null;
		hoverInteractionCanvas = null;
	}

	function detachTooltipMotionListeners() {
		tooltipMotionInteractionController?.dispose();
		tooltipMotionInteractionController = null;
		tooltipMotionInteractionCanvas = null;
	}

	$: if (mounted) {
		if (hoverInteractionCanvas && hoverInteractionCanvas !== canvasElement) {
			detachHoverListeners();
		}
		if (canvasElement && hoverInteractionCanvas !== canvasElement) {
			hoverInteractionController = createCanvasInteractionController({
				canvas: canvasElement,
				onPointerMove: handleMouseMove,
				onPointerLeave: handleMouseLeave,
			});
			hoverInteractionCanvas = canvasElement;
		}

		if (
			tooltipMotionInteractionCanvas &&
			tooltipMotionInteractionCanvas !== canvasElement
		) {
			detachTooltipMotionListeners();
		}
		if (canvasElement && tooltipMotionInteractionCanvas !== canvasElement) {
			tooltipMotionInteractionController = createCanvasInteractionController({
				canvas: canvasElement,
				windowTarget: window,
				onPointerDown: handleTooltipMotionPointerDown,
				onWheel: handleTooltipMotionWheel,
				onWindowPointerUp: handleTooltipMotionPointerRelease,
				onWindowPointerCancel: handleTooltipMotionPointerRelease,
			});
			tooltipMotionInteractionCanvas = canvasElement;
		}
	}

	onDestroy(() => {
		if (typeof window !== "undefined") {
			(window as MainRouteWindow).__utciRenderDiagnostics__ = undefined;
		}

		unsubscribeLiveRouteHost();
		liveRouteHost.dispose();

		detachHoverListeners();
		detachTooltipMotionListeners();
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
			on:click={() => (analyticsOpen = !analyticsOpen)}
		>
			<span>Analytics</span>
			<span class:open={analyticsOpen} class="chevron">v</span>
		</button>
		{#if analyticsOpen}
			<AnalyticsPanel />
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
		<MetricTooltip
			visible={tooltipVisible}
			x={tooltipX}
			y={tooltipY}
			value={tooltipValue}
			position={tooltipPosition}
			metricType={$viewerStore.metricType}
		/>
	</svelte:fragment>

	<svelte:fragment slot="overlays">
		{#if $viewerStore.loading}
			<div class="overlay-message">Loading analysis data...</div>
		{/if}

		{#if $viewerStore.error}
			<div class="overlay-message error">
				Error: {$viewerStore.error}
			</div>
		{/if}

		{#if liveRouteState.base.error}
			<div class="overlay-message error">
				Live UTCI error: {liveRouteState.base.error}
			</div>
		{/if}

		{#if liveRouteState.comparison.error}
			<div class="overlay-message error comparison-note">
				Scenario live UTCI error: {liveRouteState.comparison.error}
			</div>
		{/if}

		{#if showMainRouteOverlay}
			<div
				class="model-loading-backdrop"
				class:comparison-mode={showMainRouteComparisonOverlay}
				style={showMainRouteComparisonOverlay
					? `--curtain-position: ${$comparisonStore.curtainPosition}`
					: ""}
				aria-hidden="true"
			></div>
			<div
				class="model-loading-overlay"
				class:comparison-mode={showMainRouteComparisonOverlay}
				style={showMainRouteComparisonOverlay
					? `--curtain-position: ${$comparisonStore.curtainPosition}`
					: ""}
				aria-live="polite"
			>
				<div class="spinner"></div>
				<div class="loading-text">
					{#if modelLoading || $comparisonStore.modelLoading}
						Preparing model...
					{:else if useLiveUtciOnMainRoute}
						Computing live UTCI...
					{:else}
						Loading analysis...
					{/if}
				</div>
			</div>
		{/if}

		{#if isComparing}
			<ComparisonCurtain
				containerElement={mainViewportElement}
				{comparisonScenarioName}
			/>
		{/if}
	</svelte:fragment>

	<svelte:fragment slot="viewport">
		{#key requestLargeWebgpuLimits}
			<Scene
				backgroundColor={$viewerStore.theme === "light"
					? 0x4b5563
					: 0x111827}
				bind:canvasElement
				onRendererDiagnostics={handleRendererDiagnostics}
				{requestLargeWebgpuLimits}
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
							modelPath={resolveAnalysisModelPath(
								$analysisStore.metadata,
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
						<UTCIPointCloud
							analysis={baseSceneAnalysis}
							{model}
							bind:utciSurface={utciMesh}
							acceptedGpuResidentOutput={basePendingGpuResidentOutput}
							selectedHourRenderContext={baseSceneRenderContext}
							liveSelectedHourSurfaceIdentity={baseSceneSurfaceIdentity}
							onUtciSurfaceDiagnostics={handleUtciSurfaceDiagnostics}
							onAcceptedGpuResidentOutputRelease={handleBaseAcceptedGpuResidentOutputRelease}
							pendingRenderUpdateStartedAt={basePendingRenderUpdateStartedAt}
							utciSurfaceBackend={resolvedUtciSurfaceBackend}
						/>
					{/if}

					{#if isComparing}
						<ComparisonRenderer
							bind:this={comparisonRenderer}
							acceptedGpuResidentOutput={comparisonPendingGpuResidentOutput}
							baseCamera={cameraRef}
							displayAnalysis={comparisonSceneAnalysis}
							selectedHourRenderContext={comparisonSceneRenderContext}
							liveSelectedHourSurfaceIdentity={comparisonSceneSurfaceIdentity}
							onUtciSurfaceDiagnostics={handleComparisonUtciSurfaceDiagnostics}
							onAcceptedGpuResidentOutputRelease={handleComparisonAcceptedGpuResidentOutputRelease}
							pendingRenderUpdateStartedAt={comparisonPendingRenderUpdateStartedAt}
							utciSurfaceBackend={resolvedUtciSurfaceBackend}
						/>
					{/if}
				{/if}
			</Scene>
		{/key}
	</svelte:fragment>
</ViewerShell>


<style>
	.model-loading-backdrop.comparison-mode {
		left: calc(var(--curtain-position) * 100%);
		right: 0;
	}

	.model-loading-overlay.comparison-mode {
		left: calc(50% + var(--curtain-position) * 50%);
		transform: translate(-50%, -50%);
	}
</style>
