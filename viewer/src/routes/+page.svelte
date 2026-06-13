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
	import {
		parseMainRouteGridResolution,
		type MainRouteGridResolution,
	} from "$lib/utils/analysisQuery";
	import { resolveProjectId } from "$lib/utils/analysisPaths";
	import type { Analysis } from "$lib/types/analysis";
	import type { LiveSelectedHourControllerSurfaceDiagnostics } from "$lib/compute/selected-hour/liveSelectedHourController";
	import type { LiveSelectedHourPublishedRenderContext } from "$lib/compute/selected-hour/liveSelectedHourRenderContext";
	import { projectMainRouteLiveSceneState } from "$lib/compute/selected-hour/liveSelectedHourRouteProjection";
	import type { LiveSelectedHourSurfaceIdentity } from "$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity";
	import { createLiveSelectedHourRouteHost } from "$lib/compute/selected-hour/liveSelectedHourRouteHost";
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
	import { parseExposureSchedulingFromSearchParams } from "$lib/compute/gpu/exposureScheduling";
	import { buildMainRoutePerformanceSnapshot } from "$lib/performance/mainRoutePerformanceTelemetry";
	import { getMainRouteOverlayGating } from "./mainRouteOverlayGating";
	import {
		createMainRouteRenderPublicationProjectionTracker,
		publishMainRouteUtciDiagnostics,
		resolveMainRouteLiveMetricSelection,
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
	import {
		createEmptyTooltipInteractionDiagnostics,
		type TooltipInteractionDiagnostics,
	} from "$lib/services/tooltipService";
	import {
		createEmptyCameraInteractionTelemetry,
		type CameraInteractionDiagnostics,
	} from "$lib/services/cameraInteractionTelemetry";

	const MAIN_ROUTE_DIAGNOSTICS_GRID_RESOLUTIONS = new Set<number>([
		10,
		8,
		6,
		4,
		2,
		1,
		0.5,
	]);

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
	let selectedGridResolutionMeters = parseMainRouteGridResolution(
		typeof window !== "undefined" ? window.location.search : "",
	);
	$: resolvedUtciSurfaceBackend = resolveMainRouteUtciSurfaceBackend({
		mode: utciRenderMode,
		rendererBackend,
		isComparing: $comparisonStore.isComparing,
	});
	$: utciRenderDiagnosticsEnabled =
		$page.url.searchParams.get("utciRenderDiagnostics") === "1";
	$: exposureScheduling = parseExposureSchedulingFromSearchParams(
		$page.url.searchParams,
	);

	type MainRouteUtciSurfaceDiagnostics = LiveSelectedHourControllerSurfaceDiagnostics;
	type MainRouteDiagnosticsPreservedBaseSceneSurface = {
		analysis: Analysis;
		renderContext: LiveSelectedHourPublishedRenderContext;
		surfaceIdentity: LiveSelectedHourSurfaceIdentity;
		gpuResidentOutput: SelectedHourGpuResidentOutput;
		pendingRenderUpdateStartedAt: number | undefined;
		requestId: number;
	};
	type MainRouteDiagnosticsWindow = MainRouteWindow & {
		__mainRouteDiagnosticsSetGridResolution?:
			| ((resolutionMeters: number) => boolean)
			| undefined;
	};

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
	let useLiveMetricOnMainRoute = false;
	let liveRouteEnabled = false;
	let liveShadingMetricAvailable = false;
	let liveMetricSelectionKey = "";
	let liveMetricUnavailableError: string | null = null;
	let showTimeSection = false;
	let fixedTimePickerMode: "month" | null = null;
	let selectedMonthIndex = 7;
	let selectedHourIndex = 0;
	let selectedTimeIndex = 0;
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
	let diagnosticsPreservedBaseSceneSurface:
		| MainRouteDiagnosticsPreservedBaseSceneSurface
		| null = null;
	let viewportBaseSceneAnalysis: Analysis | null = null;
	let viewportBaseSceneRenderContext:
		| LiveSelectedHourPublishedRenderContext
		| null = null;
	let viewportBaseSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null = null;
	let viewportBasePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	let viewportBasePendingRenderUpdateStartedAt: number | undefined = undefined;
	let showMainRouteOverlay = false;
	let showMainRouteComparisonOverlay = false;
	let tooltipHoverSampleCount = 0;
	let cameraWheelEventCount = 0;
	let tooltipInteractionDiagnostics: TooltipInteractionDiagnostics & {
		hoverSampleCount: number;
	} = {
		...createEmptyTooltipInteractionDiagnostics(false),
		hoverSampleCount: 0,
	};
	let cameraInteractionDiagnostics: CameraInteractionDiagnostics =
		createEmptyCameraInteractionTelemetry().diagnostics;
	const mainRouteRenderPublicationProjectionTracker =
		createMainRouteRenderPublicationProjectionTracker();

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

	function handleGridResolutionChange(value: MainRouteGridResolution): void {
		selectedGridResolutionMeters = value;
	}

	function captureDiagnosticsBaseSceneSurface():
		| MainRouteDiagnosticsPreservedBaseSceneSurface
		| null {
		if (
			!baseHasVisibleLiveSurface ||
			baseSceneAnalysis == null ||
			baseSceneRenderContext == null ||
			baseSceneSurfaceIdentity == null ||
			basePendingGpuResidentOutput == null
		) {
			return null;
		}

		return {
			analysis: baseSceneAnalysis,
			renderContext: baseSceneRenderContext,
			surfaceIdentity: baseSceneSurfaceIdentity,
			gpuResidentOutput: basePendingGpuResidentOutput,
			pendingRenderUpdateStartedAt: basePendingRenderUpdateStartedAt,
			requestId: basePendingGpuResidentOutput.requestId,
		};
	}

	function handleDiagnosticsGridResolutionChange(
		resolutionMeters: number,
	): boolean {
		if (
			!utciRenderDiagnosticsEnabled ||
			!MAIN_ROUTE_DIAGNOSTICS_GRID_RESOLUTIONS.has(resolutionMeters) ||
			resolutionMeters === selectedGridResolutionMeters
		) {
			return false;
		}

		const preservedSurface = captureDiagnosticsBaseSceneSurface();
		if (!preservedSurface) {
			return false;
		}

		diagnosticsPreservedBaseSceneSurface = preservedSurface;
		selectedGridResolutionMeters = resolutionMeters as MainRouteGridResolution;
		return true;
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
	$: ({
		useLiveMetricOnMainRoute,
		liveRouteEnabled,
		liveShadingMetricAvailable,
		selectedMonthIndex,
		selectedHourIndex,
		selectedTimeIndex,
		selectionKey: liveMetricSelectionKey,
		showTimeSection,
		fixedTimePickerMode,
		liveMetricUnavailableError,
	} = resolveMainRouteLiveMetricSelection({
		analysis: $analysisStore,
		analysisId,
		metricType: $viewerStore.metricType,
		currentMonth: $viewerStore.currentMonth,
		currentHour: $viewerStore.currentHour,
		rendererBackend,
		rendererDevice: rendererDeviceForMain,
		utciSurfaceBackend: resolvedUtciSurfaceBackend,
	}));
	$: liveRouteHost.setRouteInputs({
		enabled: liveRouteEnabled,
		analysisId,
		baseAnalysis: $analysisStore,
		baseModel: modelLoading ? null : model,
		metricType: $viewerStore.metricType,
		selection: {
			monthIndex: selectedMonthIndex,
			hourIndex: selectedHourIndex,
			timeIndex: selectedTimeIndex,
			selectionKey: liveMetricSelectionKey,
		},
		gridResolutionMeters: selectedGridResolutionMeters,
		exposureScheduling,
		diagnosticsEnabled: utciRenderDiagnosticsEnabled,
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
		useLiveUtciOnMainRoute: useLiveMetricOnMainRoute,
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
		useLiveUtciOnMainRoute: useLiveMetricOnMainRoute,
		baseLiveLoading: liveRouteState.base.loading,
		baseHasVisibleLiveSurface,
		isComparing: $comparisonStore.isComparing,
		comparisonModelLoading: $comparisonStore.modelLoading,
		comparisonLiveLoading: liveRouteState.comparison.loading,
		comparisonHasVisibleLiveSurface,
	}));

	$: if (!utciRenderDiagnosticsEnabled) {
		diagnosticsPreservedBaseSceneSurface = null;
	}

	$: if (
		diagnosticsPreservedBaseSceneSurface != null &&
		basePendingGpuResidentOutput != null &&
		basePendingGpuResidentOutput.requestId !==
			diagnosticsPreservedBaseSceneSurface.requestId
	) {
		diagnosticsPreservedBaseSceneSurface = null;
	}

	$: {
		const preservedSurface =
			utciRenderDiagnosticsEnabled &&
			diagnosticsPreservedBaseSceneSurface != null &&
			basePendingGpuResidentOutput == null
				? diagnosticsPreservedBaseSceneSurface
				: null;
		viewportBaseSceneAnalysis =
			preservedSurface?.analysis ?? baseSceneAnalysis;
		viewportBaseSceneRenderContext =
			preservedSurface?.renderContext ?? baseSceneRenderContext;
		viewportBaseSceneSurfaceIdentity =
			preservedSurface?.surfaceIdentity ?? baseSceneSurfaceIdentity;
		viewportBasePendingGpuResidentOutput =
			preservedSurface?.gpuResidentOutput ?? basePendingGpuResidentOutput;
		viewportBasePendingRenderUpdateStartedAt =
			preservedSurface?.pendingRenderUpdateStartedAt ??
			basePendingRenderUpdateStartedAt;
	}

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
		(window as MainRouteDiagnosticsWindow).__mainRouteDiagnosticsSetGridResolution =
			utciRenderDiagnosticsEnabled
				? handleDiagnosticsGridResolutionChange
				: undefined;
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
			baseColorMode: $viewerStore.colorMode,
			basePointCount: liveRouteState.base.analysis?.metadata.num_positions ?? null,
			baseMetadataGridSize: liveRouteState.base.analysis?.metadata.grid_size ?? null,
			exposureScheduling,
			baseSceneRenderContextTimeIndex: baseSceneRenderContext?.timeIndex,
			baseAcceptedUtciRange:
				$viewerStore.metricType === "utci"
					? (basePendingGpuResidentOutput?.utciRange ?? undefined)
					: undefined,
			tooltipInteraction: tooltipInteractionDiagnostics,
			cameraInteraction: cameraInteractionDiagnostics,
			timingsOverride: mainRouteRenderPublicationProjectionTracker.apply({
				enabled: useLiveMetricOnMainRoute,
				timings: liveRouteState.base.runtimeDiagnostics?.timings,
				projectedSceneSurfaceIdentity: baseSceneSurfaceIdentity,
				publishedSurfaceIdentity: liveRouteState.baseSurfaceIdentity,
				sceneRenderContextTimeIndex: baseSceneRenderContext?.timeIndex,
				selectedTimeIndex,
			}),
		});
	}

	$: if (typeof window !== "undefined") {
		performanceStore.set(
			buildMainRoutePerformanceSnapshot({
				analysisId,
				projectLabel: currentProjectId,
				pointCount: useLiveMetricOnMainRoute
					? (liveRouteState.base.analysis?.metadata.num_positions ?? null)
					: ($analysisStore?.metadata?.num_positions ?? null),
				gridSizeMeters: useLiveMetricOnMainRoute
					? (liveRouteState.base.analysis?.metadata.grid_size ??
							selectedGridResolutionMeters)
					: ($analysisStore?.metadata?.grid_size ?? null),
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
			const win = window as MainRouteDiagnosticsWindow;
			win.__utciRenderDiagnostics__ = undefined;
			win.__mainRouteDiagnosticsSetGridResolution = undefined;
		}
		performanceStore.set(EMPTY_PERFORMANCE_SNAPSHOT);

		unsubscribeLiveRouteHost();
		liveRouteHost.dispose();
	});
</script>

<svelte:head></svelte:head>

<ViewerShell
	bind:mainViewportElement
	{showTimeSection}
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
			<PerformancePanel
				selectedGridResolutionMeters={selectedGridResolutionMeters}
				onGridResolutionChange={handleGridResolutionChange}
			/>
		{/if}
	</svelte:fragment>

	<svelte:fragment slot="layers">
		<div class="section-header">Layers</div>
		<LayerControls placement="sidebar" />
	</svelte:fragment>

	<svelte:fragment slot="time">
		{#if showTimeSection}
			<div class="section-header">
				{$viewerStore.metricType === "shading_index" ? "Month" : "Time of Day"}
			</div>
			<div class="section-subtitle">
				{$viewerStore.metricType === "shading_index"
					? "Select representative month"
					: "Select analysis hour for UTCI"}
			</div>
			<RadialTimePicker fixedMode={fixedTimePickerMode} />
		{/if}
	</svelte:fragment>

	<svelte:fragment slot="legend">
		<ColorLegend
			displayAnalysis={useLiveMetricOnMainRoute ? baseDisplayedAnalysis : null}
			liveShadingMetricAvailable={liveShadingMetricAvailable}
			utciRangeOverride={$viewerStore.metricType === "utci" && useLiveMetricOnMainRoute
				? (basePendingGpuResidentOutput?.utciRange ?? null)
				: undefined}
		/>
	</svelte:fragment>

	<svelte:fragment slot="tooltip">
		<MainRouteTooltipLayer
			bind:tooltipHoverSampleCount
			bind:cameraWheelEventCount
			bind:tooltipInteractionDiagnostics
			bind:cameraInteractionDiagnostics
			{canvasElement}
			{cameraRef}
			baseMesh={utciMesh}
			{baseDisplayedAnalysis}
			baseSceneTimeIndex={baseSceneRenderContext?.timeIndex}
			{basePendingGpuResidentOutput}
			comparisonDisplayedAnalysis={useLiveMetricOnMainRoute
				? comparisonRendererDisplayAnalysis
				: $comparisonAnalysis}
			comparisonSceneTimeIndex={comparisonSceneRenderContext?.timeIndex}
			{comparisonPendingGpuResidentOutput}
			rendererDevice={rendererDeviceForMain}
			getComparisonUtciMesh={() => comparisonRenderer?.getComparisonUtciMesh() ?? null}
			useLiveUtciOnMainRoute={useLiveMetricOnMainRoute}
			{isComparing}
			{mainViewportElement}
			curtainPosition={$comparisonStore.curtainPosition}
			viewerCurrentHour={$viewerStore.currentHour}
			metricType={$viewerStore.metricType}
			utciVisible={$viewerStore.utciVisible}
			diagnosticsEnabled={utciRenderDiagnosticsEnabled}
		/>
	</svelte:fragment>

	<svelte:fragment slot="overlays">
		<MainRouteOverlays
			loading={$viewerStore.loading}
			error={$viewerStore.error}
			baseLiveError={liveMetricUnavailableError ?? liveRouteState.base.error}
			comparisonLiveError={liveRouteState.comparison.error}
			{showMainRouteOverlay}
			{showMainRouteComparisonOverlay}
			curtainPosition={$comparisonStore.curtainPosition}
			{modelLoading}
			comparisonModelLoading={$comparisonStore.modelLoading}
			useLiveUtciOnMainRoute={useLiveMetricOnMainRoute}
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
			baseSceneAnalysis={viewportBaseSceneAnalysis}
			{comparisonSceneAnalysis}
			basePendingGpuResidentOutput={viewportBasePendingGpuResidentOutput}
			{comparisonPendingGpuResidentOutput}
			baseSceneRenderContext={viewportBaseSceneRenderContext}
			{comparisonSceneRenderContext}
			baseSceneSurfaceIdentity={viewportBaseSceneSurfaceIdentity}
			{comparisonSceneSurfaceIdentity}
			basePendingRenderUpdateStartedAt={viewportBasePendingRenderUpdateStartedAt}
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
