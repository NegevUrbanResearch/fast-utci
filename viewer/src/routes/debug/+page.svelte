<script lang="ts">
	import { browser } from "$app/environment";
	import { T } from "@threlte/core";
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
		setCurrentHour,
		setCurrentMonth,
		setAnalysisId,
		setLoading,
		setError,
	} from "$lib/stores/viewerStore";
	import { cameraStore, focusCameraOnModel } from "$lib/stores/cameraStore";
	import { setDiscoveredLayers } from "$lib/stores/layerStore";
	import {
		getBoundsCenterAndSize,
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
	import ScenarioSelector from "$lib/components/ui/ScenarioSelector.svelte";
	import ProjectSelector from "$lib/components/ui/ProjectSelector.svelte";
	import MetricTooltip from "$lib/components/ui/MetricTooltip.svelte";
	import ViewerShell from "$lib/components/viewer/ViewerShell.svelte";
	import "$lib/styles/variables.css";
	import { getDefaultAnalysisId } from "$lib/config/projects";
	import {
		resolveAnalysisModelPath,
		resolveProjectId,
	} from "$lib/utils/analysisPaths";
	import { getInitialAnalysisId } from "$lib/utils/analysisQuery";
	import * as THREE from "three";
	import type { Group, Mesh, PerspectiveCamera } from "three";
	import {
		prepareMeshPayloadForWorkerAsync,
		runMergeAndBvhInWorker,
		MAX_GRID_POINTS_GUARD,
	} from "$lib/compute/mergeAndBvhWorkerClient";
	import {
		createEmptyTooltipInteractionDiagnostics,
		getTooltipData,
		recordTooltipInteractionMeasurement,
		type TooltipInteractionDiagnostics,
	} from "$lib/services/tooltipService";
	import {
		TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS,
		armTooltipMotionSuppression,
		createTooltipMotionSuppressionState,
		releaseTooltipMotionPointer,
		setTooltipMotionPointerDown,
		shouldSuppressTooltipMotion,
	} from "$lib/services/tooltipMotionSuppression";
	import {
		createEmptyCameraInteractionTelemetry,
		recordCameraInteractionFrame,
		type CameraInteractionDiagnostics,
		type CameraInteractionTelemetry,
	} from "$lib/services/cameraInteractionTelemetry";
	import { getUTCIForHour, getUtciByHourForExport } from "$lib/services/dataLoader";
	import { getEffectiveHourIndex } from "$lib/utils/effectiveHourIndex";
	import {
		sceneConfigStore,
		updateSceneConfigFromBounds,
	} from "$lib/stores/sceneConfigStore";
	import type { Analysis, AnalysisMetadata } from "$lib/types/analysis";
	import { createLiveUtciAnalysisFromCompute } from "$lib/compute/liveUtciAnalysis";
	import {
		buildSelectedHourLiveAnalysis,
		resolveAcceptedGpuResidentUtciRange,
	} from "$lib/compute/liveUtciSelectedHour";
	import { ComputeManager } from "$lib/compute/compute-manager";
	import {
		buildGpuResidentSurfaceResetPatch,
		recordColdStartLifecycleTiming,
		resetColdStartLifecycleTimings,
		createEmptyOnDemandDiagnostics,
		mergeSelectedHourRenderTimings,
		prepareSelectedHourCycleTimings,
		type SelectedHourRenderTimingSubsteps,
		type OnDemandRuntimeDiagnostics,
	} from "$lib/compute/onDemandDiagnostics";
	import {
		deriveOnDemandPrototypeStatus,
		type OnDemandPrototypeStatus,
	} from "$lib/compute/onDemandPrototypeStatus";
	import type {
		WebgpuLargeBufferDeviceLimits,
		WebgpuLargeBufferRequiredLimits,
	} from "$lib/compute/webgpuDeviceLimits";
	import type { LiveSelectedHourControllerSurfaceDiagnostics } from "$lib/compute/liveSelectedHourController";
	import type { LiveSelectedHourPublishedRenderContext } from "$lib/compute/liveSelectedHourRenderContext";
	import { projectMainRouteLiveSceneState } from "$lib/compute/liveSelectedHourRouteProjection";
	import { createLiveSelectedHourRouteHost } from "$lib/compute/liveSelectedHourRouteHost";
	import type { LiveSelectedHourSurfaceIdentity } from "$lib/compute/liveSelectedHourSurfaceIdentity";
	import type { SelectedHourGpuResidentOutput } from "$lib/compute/liveUtciSelectedHourSession";
	import {
		createOnDemandScrubState,
		markOnDemandRequestCompleted,
		startOnDemandRequest,
	} from "$lib/compute/onDemandScrubState";
	import { createWebgpuUtciPipeline } from "$lib/compute/webgpuUtciPipeline";
	import { normalizeSkyExposureToViewFactor } from "$lib/parity/skyScale";
	import type { SerializedBvhForGpu, UTCIComputePipeline } from "$lib/compute/gpu-pipeline";
	import { comparisonStore, curtainPosition } from "$lib/stores/comparisonStore";
	import { get } from "svelte/store";
	import { emitComputeTelemetry } from "$lib/compute/telemetry";
	import {
		createSyntheticGpuUtciBridge,
		type SyntheticGpuUtciBridge,
	} from "$lib/services/gpuUtciRenderBridge";
	import {
		DEFAULT_DEBUG_UTCI_RENDER_MODE,
		parseUtciRenderMode,
		resolveUtciSurfaceBackend,
		type UtciRenderMode,
		type UtciRendererBackend,
	} from "$lib/utciRenderMode";
	import {
		deriveDebugWebgpuUtciDiagnosticsState,
		shouldExposeDebugWindowDiagnostics,
		type DebugSelectedHourEngine,
	} from "$lib/debug/debugWebgpuUtciDiagnostics";
	import {
		buildDebugSelectedHourDispatchCounters,
		shouldUseDebugSharedSelectedHourHost,
	} from "$lib/debug/debugSelectedHourMode";
	import { parseDebugWebgpuUtciQuery } from "$lib/debug/debugWebgpuUtciQuery";
	import { calculateScenarioOrigin } from "$lib/utils/coordinates";
	import { getAnchorOffset, isNormalizationEnabled } from "$lib/config/viewerConfig";

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
	/** Full-screen overlay until model is loaded and live UTCI has finished or failed. */
	$: gpuResidentSelectedHourVisible =
		liveUtciSurfaceDiagnostics.gpuResidentCopyStatus === "complete" &&
		liveUtciSurfaceDiagnostics.utciSurfaceSource === "compute-buffer-selected-hour";
	$: debugSharedSelectedHourVisible =
		useDebugSharedSelectedHourHost &&
		(debugSharedBaseHasVisibleSurface ||
			debugSharedBasePendingGpuResidentOutput !== null ||
			(debugSharedRouteState.base.renderSurfaceDiagnostics.gpuResidentCopyStatus ===
				"complete" &&
				debugSharedRouteState.base.renderSurfaceDiagnostics.utciSurfaceSource ===
					"compute-buffer-selected-hour"));
	$: suppressFullLoadOverlayForOnDemandScrub =
		onDemandPrototypeEnabled &&
		debugOnDemandMode === "f32" &&
		!strictExposureOnlyEnabled &&
		!useDebugSharedSelectedHourHost &&
		(acceptedGpuResidentUtciOutput !== null ||
			liveUtciSurfaceDiagnostics.gpuResidentCopyStatus === "pending");
	$: showFullLoadOverlay =
		modelLoading ||
		(model != null &&
			liveAnalysis === null &&
			!gpuResidentSelectedHourVisible &&
			!debugSharedSelectedHourVisible &&
			!suppressFullLoadOverlayForOnDemandScrub &&
			liveError === null &&
			!strictExposureOnlyEnabled);
	let hasFitOnce = false;
	let lastModelFile: string | null = null;
	/** Model file path that the current `model` was loaded for. Used to avoid running compute with a stale model after project switch. */
	let modelFileForLoadedModel: string | null = null;
	/** Last WebGPU pipeline instance; disposed before creating a new one and on page destroy to avoid leaks/crashes. */
	let lastPipeline: UTCIComputePipeline | null = null;
	/** AbortController for the current live run; aborted when project/model changes so only one run is active. */
	let liveAbortController: AbortController | null = null;
	/** Progress during 12-month compute: { current, total } or null. */
	let liveComputeProgress: { current: number; total: number } | null = null;
	let rerunLiveAnalysisAfterCurrentCompute = false;

	const DEFAULT_ANALYSIS_ID = getDefaultAnalysisId();
	let analysisId: string = DEFAULT_ANALYSIS_ID;
	let mounted = false;
	/** When true (?parity=1), keep Python comparison hour-local while f32 on-demand still computes month/hour selections. */
	$: debugQueryState = parseDebugWebgpuUtciQuery($page.url.searchParams);
	$: parityMode = debugQueryState.parityMode;
	/** E2E-only normal-mode export; keeps parityMode false while exposing one app-visible month slice. */
	$: normalCollectMode = debugQueryState.collectMode === "normal";
	$: debugOnDemandMode = debugQueryState.debugOnDemandMode;
	// Default the debug route to auto so prototype coverage exercises the live
	// renderer-backend resolution path.
	$: utciRenderMode = parseUtciRenderMode(
		$page.url.searchParams,
		DEFAULT_DEBUG_UTCI_RENDER_MODE,
	);
	let rendererBackend: UtciRendererBackend = "unknown";
	let rendererDeviceForDebug: GPUDevice | undefined = undefined;
	let rendererRequiredLimits: WebgpuLargeBufferRequiredLimits | undefined = undefined;
	let rendererDeviceLimits: WebgpuLargeBufferDeviceLimits | undefined = undefined;
	$: requestLargeWebgpuLimits = debugOnDemandMode === "f32" && utciRenderMode === "gpu";
	$: resolvedUtciSurfaceBackend = resolveUtciSurfaceBackend(
		utciRenderMode,
		rendererBackend,
	);
	const debugSharedRouteHost = createLiveSelectedHourRouteHost({
		dataBasePath: getDataBasePath(),
	});
	let debugSharedRouteState = debugSharedRouteHost.getState();
	const unsubscribeDebugSharedRouteHost = debugSharedRouteHost.subscribe((state) => {
		debugSharedRouteState = state;
	});

	let debugSharedBaseLiveReady = false;
	let debugSharedBaseHasVisibleSurface = false;
	let debugSharedBaseSceneAnalysis: Analysis | null = null;
	let debugSharedBaseRenderContext: LiveSelectedHourPublishedRenderContext | null = null;
	let debugSharedBaseSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null = null;
	let debugSharedBasePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	let debugSharedBasePendingRenderUpdateStartedAt: number | undefined = undefined;

	function handleDebugSharedBaseSurfaceDiagnostics(
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics,
	): void {
		debugSharedRouteHost.handleBaseSurfaceDiagnostics(diagnostics);
	}

	// Tooltip state
	let tooltipVisible = false;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipValue: number | null = null;
	let tooltipPosition: { x: number; y: number; z: number } | null = null;
	let copiedPointStatus: string | null = null;
	let utciMesh: Mesh | null = null;
	let liveUtciMesh: Mesh | null = null;
	let liveUtciSurfaceDiagnostics: {
		utciSurfaceSource?: string;
		selectedHourTransferCount?: number;
		dataTextureBuildCount?: number;
		gpuResidentCopyStatus?: "idle" | "pending" | "complete" | "failed";
		gpuResidentCopyError?: string;
		gpuResidentCopyRequestId?: number;
	} & SelectedHourRenderTimingSubsteps = {};
	let canvasElement: HTMLCanvasElement | null = null;

	// Camera reference - will be set from Camera component
	let cameraRef: PerspectiveCamera | undefined = undefined;
	const CAMERA_INTERACTION_ARM_WINDOW_MS = TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS;
	const CAMERA_INTERACTION_POSITION_EPSILON_SQ = 1e-6;
	const CAMERA_INTERACTION_QUATERNION_EPSILON = 1e-6;
	let tooltipMotionSuppression = createTooltipMotionSuppressionState();
	let cameraInteractionTelemetry: CameraInteractionTelemetry =
		createEmptyCameraInteractionTelemetry();
	let cameraInteractionFrameId: number | null = null;
	let cameraInteractionLastFrameTimeMs: number | null = null;
	let cameraInteractionPointerDown = false;
	let cameraInteractionSequenceActive = false;
	let cameraInteractionArmedUntilMs = 0;
	let hasCameraInteractionSnapshot = false;
	const lastCameraInteractionPosition = new THREE.Vector3();
	const lastCameraInteractionQuaternion = new THREE.Quaternion();

	// Main viewport element reference for curtain positioning
	let mainViewportElement: HTMLElement | null = null;

	// Live analysis state
	let liveAnalysis: Analysis | null = null;
	let liveLoading = false;
	let liveError: string | null = null;
	let lastLiveKey: string | null = null;
	let lastLiveComputeModeKey: string | null = null;
	// Large models (e.g. Ness Tziona) with 12 months need several minutes for 288-slice readback
	const LIVE_COMPUTE_WATCHDOG_MS = 300_000;
	let liveComputeWatchdog: ReturnType<typeof setTimeout> | null = null;
	let liveRunCounter = 0;
	type ParityCollectionPhase =
		| "preflight"
		| "epw"
		| "worker"
		| "pipelineInit"
		| "runAll"
		| "readback"
		| "done";

	type ParityCollectionStatus = {
		runId: number;
		state: "running" | "success" | "error" | "timeout";
		phase: ParityCollectionPhase;
		startedAt: number;
		updatedAt: number;
		message?: string;
	};

	type ParityCollectionLogEntry = {
		runId: number;
		state: "running" | "success" | "error" | "timeout";
		phase: ParityCollectionPhase;
		timestamp: number;
		message?: string;
	};

	type OnDemandPrototypeDiagnostics = Partial<OnDemandRuntimeDiagnostics> & {
		navigatorGpu: boolean;
		rendererBackend: "webgpu" | "unknown";
		utciRenderRequested?: UtciRenderMode;
		utciRenderResolved?: "dataTexture" | "gpuNative";
		utciSurfaceSource?: string;
		bridgeAttached?: boolean;
		visibleColorVariance?: number;
		debugComparisonReference?: "python-bin";
		pythonBinComparisonActive?: boolean;
		binComparisonEnabled?: boolean;
		binComparisonValid?: boolean;
		pythonBaselineStatus?: "available-august" | "unavailable-non-august";
		debugComparisonMonthIndex?: number;
		pythonComparisonHourIndex?: number;
		webgpuComparisonHourIndex?: number;
		pythonBinSampleComparison?: OnDemandPythonSampleComparison;
		appVisibleSelectedHour?: boolean;
		selectedHourReadbackCount?: number;
		liveAnalysisConstructedForSelectedHour?: boolean;
		pendingReadbackRequestId?: number;
		pendingReadbackTimeIndex?: number;
		acceptedGpuResidentUtciRange?: { min: number; max: number };
		surfaceRequestId?: number;
		selectionKey?: string;
		sceneSurfaceRequestId?: number;
		sceneSelectionKey?: string;
		selectedHourIndex?: number;
		renderContextTimeIndex?: number;
		acceptedUtciRange?: { min: number; max: number };
		tooltipInteraction?: TooltipInteractionDiagnostics;
		cameraInteraction?: CameraInteractionDiagnostics;
		tooltipProbeClientPoint?: { clientX: number; clientY: number } | null;
		selectedHourEngine?: DebugSelectedHourEngine;
		legacySelectedHourDispatchCount?: number;
		legacyScrubScheduleCount?: number;
	};

	type OnDemandPythonSampleRecord = {
		pointIndex: number;
		debugValue: number;
		referenceValue: number;
		absDiff: number;
	};

	type OnDemandPythonSampleComparison = {
		numCompared: number;
		maxAbsDiff: number;
		samples: OnDemandPythonSampleRecord[];
	};

	type AcceptedGpuResidentUtciOutput = {
		requestId: number;
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		output: Awaited<ReturnType<ComputeManager["runUtciForTimeIndex"]>>;
		utciRange: { min: number; max: number };
		tooltipUtciValues?: Float32Array;
	};

	type DeferredCpuFallbackSelectedHour = {
		requestId: number;
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		base: Analysis;
		utciValues: Float32Array;
	};

	type OnDemandPrototypeComparison = {
		timeIndex: number;
		numCompared: number;
		maxAbsDiff: number;
		rmse: number;
		debugReadbackCount: number;
	};

	type OnDemandMultiHourComparisonResult = {
		hour: number;
		numCompared: number;
		maxAbsDiff: number;
		rmse: number;
		onDemandAt31079?: number;
		baselineAt31079?: number;
		diffAt31079?: number;
	};

	type OnDemandPrototypeMultiHourComparison = {
		baselineSource: "separateRunAll";
		baselineMonthContext: {
			monthIndex: 0;
			sliceKind: "representative-day-full-year";
			note: string;
		};
		strictPath: OnDemandRuntimeDiagnostics;
		hours: number[];
		hourResults: OnDemandMultiHourComparisonResult[];
		knownPoint31079?: {
			pointIndex: number;
			hours: Array<{
				hour: number;
				onDemand: number;
				baseline: number;
				diff: number;
			}>;
		};
	};

	type OnDemandMonthHourComparisonPair = {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
	};

	type OnDemandMonthHourComparisonPairResult = {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		numCompared: number;
		maxAbsDiff: number;
		rmse: number;
		onDemandAt31079?: number;
		baselineAt31079?: number;
		diffAt31079?: number;
	};

	type OnDemandMonthHourComparisonResult = {
		status: "idle" | "running" | "complete" | "error";
		baselineSource: "separateRunAll";
		pairs: OnDemandMonthHourComparisonPairResult[];
		error?: string;
	};

	type ParityWindow = Window & {
		__parityIntermediatesError__?: string;
		__parityCollectionError__?: string;
		__parityCollectionStatus__?: ParityCollectionStatus;
		__parityCollectionLog__?: ParityCollectionLogEntry[];
		__parityResults__?: unknown;
		__parityIntermediates__?: unknown;
		__normalUtciResults__?: unknown;
		__parityMetadata__?: AnalysisMetadata;
		__parityModel__?: Group | null;
		__parityThree__?: typeof THREE;
		__onDemandPrototypeDiagnostics__?: OnDemandPrototypeDiagnostics;
		__debugTooltipProbe__?: (() => { clientX: number; clientY: number } | null) | undefined;
		__onDemandPrototypeComparison__?: OnDemandPrototypeComparison;
		__onDemandMultiHourComparison__?: OnDemandPrototypeMultiHourComparison;
		__onDemandMonthHourComparison__?: OnDemandMonthHourComparisonResult;
	};

	const getParityWindow = (): ParityWindow =>
		window as unknown as ParityWindow;

	const PARITY_SAMPLE_HEIGHT_OFFSET_M = 0.9;

	function buildSunVectorsFixtureFromMetadata(params: {
		baseMetadata: AnalysisMetadata;
		numHours: number;
		numMonths: number;
	}): { sunVectors: Float32Array; sunAltitudes: Float32Array } | undefined {
		const { baseMetadata, numHours, numMonths } = params;
		const raw = (baseMetadata as unknown as { sun_positions?: unknown }).sun_positions;
		if (!Array.isArray(raw) || raw.length < numHours) return undefined;

		type SunPositionLike = {
			vector?: [number, number, number];
			altitude?: number;
			is_up?: boolean;
		};

		const firstDay = raw as SunPositionLike[];
		const sunVectors = new Float32Array(numMonths * numHours * 3);
		const sunAltitudes = new Float32Array(numMonths * numHours);
		for (let monthOffset = 0; monthOffset < numMonths; monthOffset += 1) {
			for (let hour = 0; hour < numHours; hour += 1) {
				const source = firstDay[hour];
				const vector = source?.vector;
				const baseIndex = (monthOffset * numHours + hour) * 3;
				if (vector && vector.length === 3) {
					sunVectors[baseIndex] = vector[0];
					sunVectors[baseIndex + 1] = vector[2];
					sunVectors[baseIndex + 2] = -vector[1];
				}
				const altitudeDegrees = Number.isFinite(source?.altitude)
					? (source?.altitude as number)
					: 0;
				const isUp = source?.is_up === true || altitudeDegrees > 0;
				sunAltitudes[monthOffset * numHours + hour] = isUp
					? (altitudeDegrees * Math.PI) / 180
					: 0;
			}
		}

		return { sunVectors, sunAltitudes };
	}

	function getGridOriginOffset(
		baseMetadata: AnalysisMetadata,
	): { x: number; y: number; z: number } | undefined {
		if (!isNormalizationEnabled()) return undefined;

		const coordinateSystem =
			(baseMetadata.coordinate_system as "xy_ground" | "xz_ground") ?? "xy_ground";
		const scenarioOrigin = calculateScenarioOrigin(baseMetadata as any);
		const anchorOffset = getAnchorOffset();
		const transformedOrigin =
			coordinateSystem === "xy_ground"
				? new THREE.Vector3(
					scenarioOrigin.x,
					scenarioOrigin.z,
					-scenarioOrigin.y,
				)
				: scenarioOrigin.clone();
		const normalizationOffset = anchorOffset.clone().sub(transformedOrigin);
		return normalizationOffset.lengthSq() > 0.001
			? {
				x: normalizationOffset.x,
				y: normalizationOffset.y,
				z: normalizationOffset.z,
			}
			: undefined;
	}

	function getStrictExposureOnlyTimeIndex(): number {
		const raw = Number($page.url.searchParams.get("timeIndex") ?? "12");
		return Number.isInteger(raw) && raw >= 0 ? raw : 12;
	}

	function getDebugQueryMonthIndex(defaultMonthIndex: number): number {
		const raw = Number($page.url.searchParams.get("monthIndex") ?? String(defaultMonthIndex));
		if (!Number.isInteger(raw)) return defaultMonthIndex;
		return Math.min(Math.max(raw, 0), 11);
	}

	function getDebugOnDemandSelection(params: {
		monthIndex: number;
		hourIndex: number;
		parityMode: boolean;
	}): { monthIndex: number; hourIndex: number; timeIndex: number } {
		const { monthIndex, hourIndex, parityMode } = params;
		return {
			monthIndex,
			hourIndex,
			timeIndex: monthIndex * 24 + hourIndex,
		};
	}

	function getDebugComparisonSelectionView(): { monthIndex: number; hourIndex: number } {
		return {
			monthIndex: $viewerStore.currentMonth ?? 7,
			hourIndex: $viewerStore.currentHour,
		};
	}

	function getGpuResidentTooltipAnalysis(): Analysis | null {
		if (!acceptedGpuResidentUtciOutput || !$analysisStore) return null;
		const values = acceptedGpuResidentUtciOutput.tooltipUtciValues;
		if (!values) return $analysisStore;

		return {
			...$analysisStore,
			data: {
				numPositions: $analysisStore.data.numPositions,
				numHours: 1,
				positions: $analysisStore.data.positions,
				utciValues: values,
				shadingIndex: $analysisStore.data.shadingIndex,
			},
		};
	}

	function getCompareHoursFromQuery(): number[] {
		const raw = $page.url.searchParams.get("compareHours");
		if (!raw) return [];
		return raw
			.split(",")
			.map((value) => Number(value.trim()))
			.filter((value) => Number.isInteger(value) && value >= 0);
	}

	function getCompareMonthHourPairsFromQuery(): OnDemandMonthHourComparisonPair[] {
		const raw = $page.url.searchParams.get("compareMonthHours");
		if (!raw) return [];

		return raw
			.split(",")
			.map((entry) => entry.trim())
			.filter(Boolean)
			.map((entry) => {
				const [monthRaw, hourRaw, ...rest] = entry.split(":");
				const monthIndex = Number(monthRaw);
				const hourIndex = Number(hourRaw);
				if (
					rest.length > 0 ||
					!Number.isInteger(monthIndex) ||
					!Number.isInteger(hourIndex) ||
					monthIndex < 0 ||
					monthIndex > 11 ||
					hourIndex < 0 ||
					hourIndex > 23
				) {
					throw new Error(
						`Invalid compareMonthHours entry "${entry}". Expected month:hour with month 0-11 and hour 0-23.`,
					);
				}
				return {
					monthIndex,
					hourIndex,
					timeIndex: monthIndex * 24 + hourIndex,
				};
			});
	}

	function clearOnDemandPrototypeComparison(): void {
		if (!browser) return;
		getParityWindow().__onDemandPrototypeComparison__ = undefined;
	}

	function clearOnDemandMultiHourComparison(): void {
		if (!browser) return;
		getParityWindow().__onDemandMultiHourComparison__ = undefined;
	}

	function clearOnDemandMonthHourComparison(): void {
		if (!browser) return;
		getParityWindow().__onDemandMonthHourComparison__ = {
			status: "idle",
			baselineSource: "separateRunAll",
			pairs: [],
		};
	}

	let onDemandPrototypeComparisonRunToken = 0;
	let lastOnDemandPrototypeComparisonAttemptKey: string | null = null;
	let onDemandMonthHourComparisonRunToken = 0;
	let lastOnDemandMonthHourComparisonAttemptKey: string | null = null;

	function invalidateOnDemandPrototypeComparison(options?: { resetAttemptKey?: boolean }): void {
		onDemandPrototypeComparisonRunToken += 1;
		onDemandMonthHourComparisonRunToken += 1;
		if (options?.resetAttemptKey !== false) {
			lastOnDemandPrototypeComparisonAttemptKey = null;
			lastOnDemandMonthHourComparisonAttemptKey = null;
		}
		clearOnDemandPrototypeComparison();
		clearOnDemandMultiHourComparison();
		clearOnDemandMonthHourComparison();
	}

	function hasOnDemandPrototypeComparison(): boolean {
		return browser && Boolean(getParityWindow().__onDemandPrototypeComparison__);
	}

	function hasOnDemandMultiHourComparison(): boolean {
		return browser && Boolean(getParityWindow().__onDemandMultiHourComparison__);
	}

	function hasCompletedOnDemandMonthHourComparison(): boolean {
		if (!browser) return false;
		const status = getParityWindow().__onDemandMonthHourComparison__?.status;
		return status === "complete" || status === "error";
	}

	function getOnDemandPrototypeDebugReadbackCount(): number {
		return getParityWindow().__onDemandPrototypeDiagnostics__?.debugReadbackCount ?? 0;
	}

	function getOnDemandPrototypeComparisonKey(): string | null {
		if (
			!browser ||
			!compareOneHourEnabled ||
			!liveAnalysis ||
			!lastPipeline ||
			!lastLiveKey ||
			liveLoading ||
			liveError
		) {
			return null;
		}
		return `${lastLiveKey}|compareOneHour`;
	}

	let onDemandPrototypeStatus: OnDemandPrototypeStatus = "idle";
	let onDemandPrototypeError: string | null = null;
	let syntheticBridge: SyntheticGpuUtciBridge | null = null;
	let syntheticBridgeKey: string | null = null;
	let syntheticBridgeMountedKey: string | null = null;
	let syntheticBridgeValidationRunId = 0;
	let syntheticBridgeValidationStartedForKey: string | null = null;
	let syntheticBridgeValidationTimer: ReturnType<typeof setTimeout> | null = null;
	let lastDebugOnDemandScrubTriggerKey: string | null = null;
	let debugOnDemandScrubScheduleRunId = 0;
	let legacySelectedHourDispatchCount = 0;
	let legacyScrubScheduleCount = 0;
	let acceptedGpuResidentUtciOutput: AcceptedGpuResidentUtciOutput | null = null;
	let deferredCpuFallbackSelectedHour: DeferredCpuFallbackSelectedHour | null = null;
	let onDemandDebugPrepared:
		| {
				computeManager: ComputeManager;
				pipeline: UTCIComputePipeline;
				deviceSource?: "standalone" | "renderer";
				numPoints: number;
				numHours: number;
				numMonths: number;
				base: Analysis;
				signal: AbortSignal;
				runId: number;
				zHeight: number;
				exposureReady: boolean;
				exposurePrecomputePromise: Promise<void> | null;
				coldStartStartedAt: number;
				pendingRenderUpdate:
					| {
							requestId: number;
							monthIndex: number;
							timeIndex: number;
							startedAt: number;
						}
					| null;
			}
		| undefined;
	let onDemandScrubState = createOnDemandScrubState();

	function destroyOnDemandGpuBuffer(
		output: Awaited<ReturnType<ComputeManager["runUtciForTimeIndex"]>> | undefined,
	): void {
		const buffer = output?.gpuBuffer as GPUBuffer | undefined;
		buffer?.destroy?.();
		if (output && "gpuBuffer" in output) {
			output.gpuBuffer = undefined;
		}
	}

	function setAcceptedGpuResidentUtciOutput(next: AcceptedGpuResidentUtciOutput | null): void {
		const previous = acceptedGpuResidentUtciOutput;
		if (previous && previous.output !== next?.output) {
			destroyOnDemandGpuBuffer(previous.output);
		}
		acceptedGpuResidentUtciOutput = next;
	}
	let wasOnDemandPrototypeEnabled = false;
	let wasCompareOneHourEnabled = false;
	$: debugTooltipHoverDisabled = $page.url.searchParams.get("disableTooltip") === "1";
	$: compareOneHourEnabled =
		onDemandPrototypeEnabled &&
		browser &&
		$page.url.searchParams.get("compareOneHour") === "1";
	$: strictExposureOnlyEnabled =
		onDemandPrototypeEnabled &&
		browser &&
		$page.url.searchParams.get("strictExposureOnly") === "1";
	$: compareHours = strictExposureOnlyEnabled ? getCompareHoursFromQuery() : [];
	$: compareHoursEnabled =
		strictExposureOnlyEnabled &&
		compareHours.length > 0 &&
		$page.url.searchParams.get("baseline") === "separateRunAll";
	$: compareMonthHoursEnabled =
		strictExposureOnlyEnabled &&
		Boolean($page.url.searchParams.get("compareMonthHours")) &&
		$page.url.searchParams.get("baseline") === "separateRunAll";
	$: debugOnDemandMonthIndex = getDebugQueryMonthIndex($viewerStore.currentMonth ?? 7);
	$: debugOnDemandSelection = getDebugOnDemandSelection({
		monthIndex: debugOnDemandMonthIndex,
		hourIndex: $page.url.searchParams.has("timeIndex")
			? getStrictExposureOnlyTimeIndex()
			: $viewerStore.currentHour,
		parityMode,
	});
	$: legacySelectedHourDispatchCounters = buildDebugSelectedHourDispatchCounters({
		legacySelectedHourDispatchCount,
		legacyScrubScheduleCount,
	});
	$: debugDiagnosticsState = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode,
		collectMode: normalCollectMode ? "normal" : "off",
		debugOnDemandMode,
		utciRenderMode,
		selectedMonthIndex: debugOnDemandSelection.monthIndex,
		selectedHourIndex: debugOnDemandSelection.hourIndex,
		selectedTimeIndex: debugOnDemandSelection.timeIndex,
		selectedHourEngine: "legacy-debug",
		binComparisonValid: parityMode && debugOnDemandSelection.monthIndex === 7,
		...legacySelectedHourDispatchCounters,
	});
	$: onDemandPrototypeEnabled = browser && debugDiagnosticsState.onDemandEnabled;
	$: useDebugSharedSelectedHourHost = shouldUseDebugSharedSelectedHourHost({
		onDemandPrototypeEnabled,
		debugOnDemandMode,
		parityMode,
		normalCollectMode,
		strictExposureOnlyEnabled,
		compareOneHourEnabled,
	});
	$: debugSharedRouteHost.setRouteInputs({
		enabled: useDebugSharedSelectedHourHost,
		analysisId,
		baseAnalysis: $analysisStore,
		baseModel:
			modelFileForLoadedModel === $analysisStore?.metadata.model_file ? model : null,
		selection: {
			monthIndex: debugOnDemandSelection.monthIndex,
			hourIndex: debugOnDemandSelection.hourIndex,
			timeIndex: debugOnDemandSelection.timeIndex,
			selectionKey: [
				analysisId,
				debugOnDemandSelection.monthIndex,
				debugOnDemandSelection.hourIndex,
			].join("|"),
		},
		colorMode: $viewerStore.colorMode,
		utciRenderMode,
		rendererBackend,
		rendererDevice: rendererDeviceForDebug,
		utciSurfaceBackend: resolvedUtciSurfaceBackend,
		comparison: {
			active: false,
			analysisId: null,
			sourceAnalysis: null,
			model: null,
			rendererDevice: rendererDeviceForDebug,
		},
	});
	$: ({
		baseLiveReady: debugSharedBaseLiveReady,
		baseHasVisibleLiveSurface: debugSharedBaseHasVisibleSurface,
		baseSceneAnalysis: debugSharedBaseSceneAnalysis,
		baseSceneRenderContext: debugSharedBaseRenderContext,
		baseSceneSurfaceIdentity: debugSharedBaseSurfaceIdentity,
		basePendingGpuResidentOutput: debugSharedBasePendingGpuResidentOutput,
		basePendingRenderUpdateStartedAt: debugSharedBasePendingRenderUpdateStartedAt,
	} = projectMainRouteLiveSceneState({
		useLiveUtciOnMainRoute: useDebugSharedSelectedHourHost,
		isComparing: false,
		baseAnalysis: $analysisStore,
		comparisonAnalysis: null,
		liveRouteState: debugSharedRouteState,
	}));
	$: debugOnDemandSelectionKey =
		debugDiagnosticsState.onDemandEnabled
			? `${debugDiagnosticsState.selection.monthIndex}:${debugDiagnosticsState.selection.timeIndex}`
			: "off";
	$: if (
		browser &&
		debugDiagnosticsState.onDemandEnabled &&
		($page.url.searchParams.has("timeIndex") || $page.url.searchParams.has("monthIndex"))
	) {
		if (($viewerStore.currentMonth ?? 7) !== debugOnDemandSelection.monthIndex) {
			setCurrentMonth(debugOnDemandSelection.monthIndex);
		}
		if ($viewerStore.currentHour !== debugOnDemandSelection.hourIndex) {
			setCurrentHour(debugOnDemandSelection.hourIndex);
		}
	}
	$: if (
		browser &&
		mounted &&
		onDemandPrototypeEnabled &&
		debugOnDemandMode === "f32" &&
		!useDebugSharedSelectedHourHost &&
		!strictExposureOnlyEnabled &&
		$analysisStore &&
		model &&
		modelFileForLoadedModel === $analysisStore.metadata.model_file &&
		onDemandDebugPrepared
	) {
		const nextScrubTriggerKey = `${analysisId}|${liveComputeModeKey}|${debugOnDemandSelectionKey}`;
		if (nextScrubTriggerKey !== lastDebugOnDemandScrubTriggerKey) {
			lastDebugOnDemandScrubTriggerKey = nextScrubTriggerKey;
			scheduleDebugOnDemandScrubRecompute(nextScrubTriggerKey);
		}
	} else if (
		useDebugSharedSelectedHourHost ||
		!browser ||
		!onDemandPrototypeEnabled ||
		debugOnDemandMode !== "f32" ||
		strictExposureOnlyEnabled
	) {
		lastDebugOnDemandScrubTriggerKey = null;
		debugOnDemandScrubScheduleRunId += 1;
	}
	$: syntheticBridgeEnabled =
		onDemandPrototypeEnabled &&
		browser &&
		$page.url.searchParams.get("syntheticBridge") === "1";
	$: liveComputeModeKey = strictExposureOnlyEnabled
		? parityMode
			? `strict-parity-${getStrictExposureOnlyTimeIndex()}${compareHoursEnabled ? `|compareHours=${compareHours.join(",")}|baseline=separateRunAll` : ""}${compareMonthHoursEnabled ? `|compareMonthHours=${$page.url.searchParams.get("compareMonthHours") ?? ""}|baseline=separateRunAll` : ""}`
			: `strict-full-year-${getStrictExposureOnlyTimeIndex()}${compareHoursEnabled ? `|compareHours=${compareHours.join(",")}|baseline=separateRunAll` : ""}${compareMonthHoursEnabled ? `|compareMonthHours=${$page.url.searchParams.get("compareMonthHours") ?? ""}|baseline=separateRunAll` : ""}`
		: debugDiagnosticsState.onDemandEnabled
			? debugDiagnosticsState.binComparisonEnabled
				? "debug-on-demand-parity-f32"
				: "debug-on-demand-full-year-f32"
			: debugDiagnosticsState.binComparisonEnabled
				? "parity"
				: debugDiagnosticsState.collectNormalMode
					? "collect-normal"
					: "full-year";

	$: if (mounted && lastLiveComputeModeKey !== null && lastLiveComputeModeKey !== liveComputeModeKey) {
		liveAbortController?.abort();
		liveAnalysis = null;
		liveError = null;
		lastLiveKey = null;
		onDemandDebugPrepared = undefined;
		lastDebugOnDemandScrubTriggerKey = null;
		debugOnDemandScrubScheduleRunId += 1;
		onDemandScrubState = createOnDemandScrubState();
		invalidateOnDemandPrototypeComparison();
	}

	$: lastLiveComputeModeKey = liveComputeModeKey;

	const SYNTHETIC_BRIDGE_VALIDATION_ATTEMPTS = 10;
	const SYNTHETIC_BRIDGE_VALIDATION_DELAY_MS = 48;
	const SYNTHETIC_BRIDGE_SAMPLE_SIZE = 48;

	function getSyntheticBridgeIdentity(targetModel: Group): string {
		return `${analysisId}:${targetModel.uuid}`;
	}

	function resetSyntheticBridgeDiagnostics(error?: string): void {
		updateOnDemandPrototypeDiagnostics({
			bridgeAttached: false,
			debugReadbackCount: 0,
			dataTextureBuildCount: 0,
			visibleColorVariance: 0,
			error,
		});
	}

	function cancelSyntheticBridgeValidation(): void {
		syntheticBridgeValidationRunId += 1;
		syntheticBridgeValidationStartedForKey = null;
		if (syntheticBridgeValidationTimer) {
			clearTimeout(syntheticBridgeValidationTimer);
			syntheticBridgeValidationTimer = null;
		}
	}

	function computeCanvasVisibleColorVariance(canvas: HTMLCanvasElement): number {
		const sampleWidth = Math.min(SYNTHETIC_BRIDGE_SAMPLE_SIZE, Math.max(1, canvas.width));
		const sampleHeight = Math.min(SYNTHETIC_BRIDGE_SAMPLE_SIZE, Math.max(1, canvas.height));
		const sourceWidth = Math.max(1, Math.floor(canvas.width * 0.5));
		const sourceHeight = Math.max(1, Math.floor(canvas.height * 0.5));
		const sourceX = Math.max(0, Math.floor((canvas.width - sourceWidth) / 2));
		const sourceY = Math.max(0, Math.floor((canvas.height - sourceHeight) / 2));

		const sampleCanvas = document.createElement("canvas");
		sampleCanvas.width = sampleWidth;
		sampleCanvas.height = sampleHeight;
		const context = sampleCanvas.getContext("2d", { willReadFrequently: true });
		if (!context) {
			throw new Error("Synthetic bridge validation could not acquire a 2D canvas context.");
		}

		context.drawImage(
			canvas,
			sourceX,
			sourceY,
			sourceWidth,
			sourceHeight,
			0,
			0,
			sampleWidth,
			sampleHeight,
		);

		const { data } = context.getImageData(0, 0, sampleWidth, sampleHeight);
		if (data.length <= 4) return 0;

		let luminanceSum = 0;
		const sampleCount = data.length / 4;
		for (let index = 0; index < data.length; index += 4) {
			luminanceSum +=
				(data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722) /
				255;
		}

		const mean = luminanceSum / sampleCount;
		let squaredDeviationSum = 0;
		for (let index = 0; index < data.length; index += 4) {
			const luminance =
				(data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722) /
				255;
			const delta = luminance - mean;
			squaredDeviationSum += delta * delta;
		}

		return squaredDeviationSum / sampleCount;
	}

	function maybeStartSyntheticBridgeRenderValidation(): void {
		if (
			!browser ||
			!syntheticBridgeEnabled ||
			!syntheticBridge ||
			!syntheticBridgeKey ||
			!canvasElement ||
			syntheticBridgeMountedKey !== syntheticBridgeKey
		) {
			return;
		}

		const diagnostics = getParityWindow().__onDemandPrototypeDiagnostics__;
		if (
			diagnostics?.rendererBackend !== "webgpu" ||
			diagnostics.bridgeAttached !== true ||
			(diagnostics.visibleColorVariance ?? 0) > 0 ||
			diagnostics.error ||
			syntheticBridgeValidationStartedForKey === syntheticBridgeKey
		) {
			return;
		}

		cancelSyntheticBridgeValidation();
		const runId = syntheticBridgeValidationRunId;
		const validationKey = syntheticBridgeKey;
		syntheticBridgeValidationStartedForKey = validationKey;
		let attempts = 0;

		const validate = () => {
			if (
				runId !== syntheticBridgeValidationRunId ||
				validationKey !== syntheticBridgeKey ||
				validationKey !== syntheticBridgeMountedKey
			) {
				return;
			}

			requestAnimationFrame(() => {
				if (
					runId !== syntheticBridgeValidationRunId ||
					validationKey !== syntheticBridgeKey ||
					validationKey !== syntheticBridgeMountedKey ||
					!canvasElement
				) {
					return;
				}

				try {
					const visibleColorVariance = computeCanvasVisibleColorVariance(canvasElement);
					if (visibleColorVariance > 0) {
						syntheticBridgeValidationTimer = null;
						updateOnDemandPrototypeDiagnostics({
							visibleColorVariance,
							error: undefined,
						});
						return;
					}
				} catch (error) {
					syntheticBridgeValidationTimer = null;
					updateOnDemandPrototypeDiagnostics({
						bridgeAttached: true,
						debugReadbackCount: 0,
						dataTextureBuildCount: 0,
						visibleColorVariance: 0,
						error: error instanceof Error ? error.message : String(error),
					});
					return;
				}

				attempts += 1;
				if (attempts >= SYNTHETIC_BRIDGE_VALIDATION_ATTEMPTS) {
					syntheticBridgeValidationTimer = null;
					updateOnDemandPrototypeDiagnostics({
						bridgeAttached: true,
						debugReadbackCount: 0,
						dataTextureBuildCount: 0,
						visibleColorVariance: 0,
						error: "Synthetic bridge mounted but render validation observed no visible color variance.",
					});
					return;
				}

				syntheticBridgeValidationTimer = setTimeout(
					validate,
					SYNTHETIC_BRIDGE_VALIDATION_DELAY_MS,
				);
			});
		};

		validate();
	}

	function initializeOnDemandPrototypeDiagnostics(): void {
		if (!browser || !onDemandPrototypeEnabled) return;

		cameraInteractionTelemetry = createEmptyCameraInteractionTelemetry();
		cameraInteractionPointerDown = false;
		cameraInteractionSequenceActive = false;
		cameraInteractionArmedUntilMs = 0;
		cameraInteractionLastFrameTimeMs = null;
		hasCameraInteractionSnapshot = false;
		tooltipMotionSuppression = createTooltipMotionSuppressionState();
		const navigatorGpu = Boolean(navigator.gpu);
		const error =
			syntheticBridgeEnabled && !navigatorGpu
				? "Synthetic bridge requires a WebGPU-capable browser runtime."
				: undefined;
		invalidateOnDemandPrototypeComparison();
		updateOnDemandPrototypeDiagnostics(
			{
				...createEmptyOnDemandDiagnostics(),
				navigatorGpu,
				rendererBackend,
				utciRenderRequested: utciRenderMode,
				utciRenderResolved: resolvedUtciSurfaceBackend,
				debugComparisonReference:
					parityMode && debugDiagnosticsState.binComparisonValid ? "python-bin" : undefined,
				pythonBinComparisonActive: parityMode && debugDiagnosticsState.binComparisonValid,
				binComparisonEnabled: debugDiagnosticsState.binComparisonEnabled,
				binComparisonValid: debugDiagnosticsState.binComparisonValid,
				pythonBaselineStatus: parityMode
					? debugDiagnosticsState.binComparisonValid
						? "available-august"
						: "unavailable-non-august"
					: undefined,
				...(syntheticBridgeEnabled
					? {
						bridgeAttached: false,
						visibleColorVariance: 0,
					}
					: {}),
				...(error ? { error } : {}),
				tooltipInteraction: createEmptyTooltipInteractionDiagnostics(
					debugTooltipHoverDisabled,
				),
				cameraInteraction: cameraInteractionTelemetry.diagnostics,
			},
			{ replace: true },
		);
	}

	function canUseGpuResidentRender(
		computeManager?: ComputeManager,
	): { available: boolean; sameDevice: boolean | null; error?: string } {
		const computeDevice = computeManager?.getDeviceForDebug();
		const rendererDevice = rendererDeviceForDebug;

		if (!computeDevice && !rendererDevice) {
			return {
				available: false,
				sameDevice: null,
				error:
					"GPU-resident render feasibility gate failed: compute GPUDevice is unavailable and renderer GPUDevice is unavailable.",
			};
		}
		if (!computeDevice) {
			return {
				available: false,
				sameDevice: null,
				error:
					"GPU-resident render feasibility gate failed: compute GPUDevice is unavailable.",
			};
		}
		if (!rendererDevice) {
			return {
				available: false,
				sameDevice: null,
				error:
					"GPU-resident render feasibility gate failed: renderer GPUDevice is unavailable.",
			};
		}
		if (computeDevice !== rendererDevice) {
			return {
				available: false,
				sameDevice: false,
				error:
					"GPU-resident render feasibility gate failed: compute and render are using different GPUDevice instances.",
			};
		}
		return {
			available: true,
			sameDevice: true,
		};
	}

	function buildGpuResidentRenderDiagnosticsPatch(
		computeManager?: ComputeManager,
	): Pick<
		OnDemandRuntimeDiagnostics,
		| "gpuResidentRenderAvailable"
		| "sameDeviceForComputeAndRender"
		| "gpuResidentCopyStatus"
		| "gpuResidentCopyError"
	> {
		const feasibility = canUseGpuResidentRender(computeManager);
		if (feasibility.available) {
			return {
				gpuResidentRenderAvailable: true,
				sameDeviceForComputeAndRender: true,
				gpuResidentCopyStatus: "idle",
				gpuResidentCopyError: undefined,
			};
		}

		return {
			gpuResidentRenderAvailable: false,
			sameDeviceForComputeAndRender: feasibility.sameDevice,
			gpuResidentCopyStatus:
				feasibility.sameDevice === false ? "failed" : "idle",
			gpuResidentCopyError: feasibility.error,
		};
	}

	function resetOnDemandPrototypeColdStartTimings(): void {
		if (!browser || !onDemandPrototypeEnabled) return;

		const existingDiagnostics = getParityWindow().__onDemandPrototypeDiagnostics__;
		if (!existingDiagnostics) return;
		const nextDiagnostics = resetColdStartLifecycleTimings({
			...createEmptyOnDemandDiagnostics(),
			...existingDiagnostics,
		});
		updateOnDemandPrototypeDiagnostics({
			timings: nextDiagnostics.timings,
		});
	}

	function updateOnDemandPrototypeColdStartTiming(
		key: "payloadPrepareMs" | "workerBvhMs" | "pipelineUploadMs" | "firstSelectedHourReadyMs" | "firstSelectedHourVisibleMs",
		value: number | undefined,
	): void {
		if (!browser || !onDemandPrototypeEnabled || value === undefined) return;

		const existingDiagnostics = getParityWindow().__onDemandPrototypeDiagnostics__;
		if (!existingDiagnostics) return;
		const nextDiagnostics = recordColdStartLifecycleTiming(
			{
				...createEmptyOnDemandDiagnostics(),
				...existingDiagnostics,
			},
			key,
			value,
		);
		if (nextDiagnostics === existingDiagnostics) return;
		updateOnDemandPrototypeDiagnostics({
			timings: nextDiagnostics.timings,
		});
	}

	function updateOnDemandPrototypeDiagnostics(
		diagnostics: Partial<OnDemandPrototypeDiagnostics>,
		options?: { replace?: boolean; computeManager?: ComputeManager },
	): void {
		if (
			!browser ||
			!onDemandPrototypeEnabled ||
			!shouldExposeDebugWindowDiagnostics(debugDiagnosticsState)
		) return;

		const win = getParityWindow();
		const existing = options?.replace ? undefined : win.__onDemandPrototypeDiagnostics__;
		const feasibilityDiagnostics = buildGpuResidentRenderDiagnosticsPatch(
			options?.computeManager ?? onDemandDebugPrepared?.computeManager,
		);
		const nextDiagnostics: OnDemandPrototypeDiagnostics = {
			...createEmptyOnDemandDiagnostics(),
			...existing,
			...feasibilityDiagnostics,
			...diagnostics,
			navigatorGpu: diagnostics.navigatorGpu ?? existing?.navigatorGpu ?? Boolean(navigator.gpu),
			rendererBackend: diagnostics.rendererBackend ?? existing?.rendererBackend ?? "unknown",
			utciRenderRequested:
				diagnostics.utciRenderRequested ?? existing?.utciRenderRequested ?? utciRenderMode,
			utciRenderResolved:
				diagnostics.utciRenderResolved ??
				existing?.utciRenderResolved ??
				resolvedUtciSurfaceBackend,
			utciSurfaceSource:
				"utciSurfaceSource" in diagnostics
					? diagnostics.utciSurfaceSource
					: existing?.utciSurfaceSource,
			selectedHourEngine:
				diagnostics.selectedHourEngine ??
				(useDebugSharedSelectedHourHost ? existing?.selectedHourEngine : undefined) ??
				debugDiagnosticsState.selectedHourEngine,
			binComparisonEnabled:
				diagnostics.binComparisonEnabled ??
				existing?.binComparisonEnabled ??
				debugDiagnosticsState.binComparisonEnabled,
			binComparisonValid:
				diagnostics.binComparisonValid ??
				existing?.binComparisonValid ??
				debugDiagnosticsState.binComparisonValid,
			...buildDebugSelectedHourDispatchCounters({
				legacySelectedHourDispatchCount:
					diagnostics.legacySelectedHourDispatchCount ??
					legacySelectedHourDispatchCount,
				legacyScrubScheduleCount:
					diagnostics.legacyScrubScheduleCount ?? legacyScrubScheduleCount,
			}),
			tooltipInteraction:
				diagnostics.tooltipInteraction ??
				existing?.tooltipInteraction ??
				createEmptyTooltipInteractionDiagnostics(debugTooltipHoverDisabled),
			cameraInteraction:
				diagnostics.cameraInteraction ??
				existing?.cameraInteraction ??
				cameraInteractionTelemetry.diagnostics,
		};
		win.__onDemandPrototypeDiagnostics__ = nextDiagnostics;
		onDemandPrototypeError = nextDiagnostics.error ?? null;
		onDemandPrototypeStatus = deriveOnDemandPrototypeStatus({
			diagnostics: nextDiagnostics,
			syntheticBridgeEnabled,
			strictExposureOnlyEnabled,
			compareOneHourEnabled,
			hasOnDemandPrototypeComparison: hasOnDemandPrototypeComparison(),
			compareHoursEnabled,
			hasOnDemandMultiHourComparison: hasOnDemandMultiHourComparison(),
			compareMonthHoursEnabled,
			hasCompletedOnDemandMonthHourComparison:
				hasCompletedOnDemandMonthHourComparison(),
		});
		if (syntheticBridgeEnabled) {
			maybeStartSyntheticBridgeRenderValidation();
		}
	}

	$: if (
		useDebugSharedSelectedHourHost &&
		debugSharedRouteState.base.renderTransport === "compute-buffer-selected-hour" &&
		debugSharedRouteState.base.sameDeviceForComputeAndRender === true &&
		debugSharedRouteState.base.renderSurfaceDiagnostics.utciSurfaceSource ===
			"compute-buffer-selected-hour"
	) {
		const renderSurfaceDiagnostics =
			debugSharedRouteState.base.renderSurfaceDiagnostics as typeof debugSharedRouteState.base.renderSurfaceDiagnostics & {
				selectedHourReadbackCount?: number;
			};
		updateOnDemandPrototypeDiagnostics({
			selectedHourEngine: "shared-host",
			legacySelectedHourDispatchCount,
			legacyScrubScheduleCount,
			surfaceRequestId: debugSharedRouteState.baseSurfaceIdentity?.requestId,
			selectionKey: debugSharedRouteState.baseSurfaceIdentity?.selectionKey,
			sceneSurfaceRequestId: debugSharedRouteState.baseSceneSurfaceIdentity?.requestId,
			sceneSelectionKey: debugSharedRouteState.baseSceneSurfaceIdentity?.selectionKey,
			selectedMonthIndex: debugOnDemandSelection.monthIndex,
			selectedHourIndex: debugOnDemandSelection.hourIndex,
			selectedTimeIndex: debugOnDemandSelection.timeIndex,
			renderContextTimeIndex: debugSharedBaseRenderContext?.timeIndex,
			acceptedUtciRange: debugSharedBasePendingGpuResidentOutput?.utciRange,
			renderTransport: debugSharedRouteState.base.renderTransport,
			sameDeviceForComputeAndRender:
				debugSharedRouteState.base.sameDeviceForComputeAndRender,
			utciSurfaceSource: renderSurfaceDiagnostics.utciSurfaceSource,
			selectedHourReadbackCount:
				renderSurfaceDiagnostics.selectedHourReadbackCount ?? 0,
			selectedHourTransferCount:
				renderSurfaceDiagnostics.selectedHourTransferCount,
			dataTextureBuildCount: renderSurfaceDiagnostics.dataTextureBuildCount,
			gpuResidentCopyStatus: renderSurfaceDiagnostics.gpuResidentCopyStatus,
			gpuResidentCopyError: renderSurfaceDiagnostics.gpuResidentCopyError,
			gpuResidentCopyRequestId: renderSurfaceDiagnostics.gpuResidentCopyRequestId,
		});
	}

	function armCameraInteractionTelemetry(windowMs = CAMERA_INTERACTION_ARM_WINDOW_MS): void {
		cameraInteractionArmedUntilMs = Math.max(
			cameraInteractionArmedUntilMs,
			performance.now() + windowMs,
		);
	}

	function snapshotCameraInteractionTransform(camera: PerspectiveCamera): void {
		lastCameraInteractionPosition.copy(camera.position);
		lastCameraInteractionQuaternion.copy(camera.quaternion);
		hasCameraInteractionSnapshot = true;
	}

	function hasCameraInteractionTransformChanged(camera: PerspectiveCamera): boolean {
		return (
			camera.position.distanceToSquared(lastCameraInteractionPosition) >
				CAMERA_INTERACTION_POSITION_EPSILON_SQ ||
			1 - Math.abs(camera.quaternion.dot(lastCameraInteractionQuaternion)) >
				CAMERA_INTERACTION_QUATERNION_EPSILON
		);
	}

	function stopCameraInteractionTelemetryLoop(): void {
		if (cameraInteractionFrameId !== null) {
			cancelAnimationFrame(cameraInteractionFrameId);
			cameraInteractionFrameId = null;
		}
		cameraInteractionLastFrameTimeMs = null;
		hasCameraInteractionSnapshot = false;
	}

	function startCameraInteractionTelemetryLoop(): void {
		if (!browser || cameraInteractionFrameId !== null) return;

		const tick = (frameTimeMs: number) => {
			cameraInteractionFrameId = requestAnimationFrame(tick);

			const activeCamera = cameraRef;
			if (!activeCamera) {
				cameraInteractionLastFrameTimeMs = frameTimeMs;
				hasCameraInteractionSnapshot = false;
				return;
			}

			if (!hasCameraInteractionSnapshot) {
				snapshotCameraInteractionTransform(activeCamera);
				cameraInteractionLastFrameTimeMs = frameTimeMs;
				return;
			}

			const frameMs =
				cameraInteractionLastFrameTimeMs == null
					? 0
					: frameTimeMs - cameraInteractionLastFrameTimeMs;
			const cameraChanged = hasCameraInteractionTransformChanged(activeCamera);
			const interactionArmed =
				cameraInteractionPointerDown ||
				frameTimeMs <= cameraInteractionArmedUntilMs ||
				cameraInteractionSequenceActive;

			if (cameraChanged && interactionArmed) {
				cameraInteractionTelemetry = recordCameraInteractionFrame(
					cameraInteractionTelemetry,
					frameMs,
				);
				cameraInteractionSequenceActive = true;
				armCameraInteractionTelemetry();
				tooltipMotionSuppression = armTooltipMotionSuppression(
					tooltipMotionSuppression,
					frameTimeMs,
				);
				hideTooltip();
				updateOnDemandPrototypeDiagnostics({
					cameraInteraction: cameraInteractionTelemetry.diagnostics,
				});
			} else if (!cameraChanged) {
				cameraInteractionSequenceActive = false;
			}

			snapshotCameraInteractionTransform(activeCamera);
			cameraInteractionLastFrameTimeMs = frameTimeMs;
		};

		cameraInteractionFrameId = requestAnimationFrame(tick);
	}

	function disposeSyntheticBridge(): void {
		cancelSyntheticBridgeValidation();
		syntheticBridge?.dispose();
		syntheticBridge = null;
		syntheticBridgeKey = null;
		syntheticBridgeMountedKey = null;
	}

	async function runOnDemandPrototypeComparison(params: {
		pipeline: UTCIComputePipeline;
		numPoints: number;
		numHours: number;
		numMonths: number;
		comparisonKey: string;
	}): Promise<void> {
		if (!browser || !onDemandPrototypeEnabled || !compareOneHourEnabled) {
			clearOnDemandPrototypeComparison();
			return;
		}

		const { pipeline, numPoints, numHours, numMonths, comparisonKey } = params;
		if (!pipeline.runUtciForTimeIndex || !pipeline.readOnDemandUtciForDebug) {
			clearOnDemandPrototypeComparison();
			updateOnDemandPrototypeDiagnostics({
				error:
					"Current WebGPU UTCI pipeline does not expose the on-demand comparison APIs."
			});
			return;
		}

		const timeIndex = 12;
		const requestToken = ++onDemandPrototypeComparisonRunToken;
		const expectedLiveKey = lastLiveKey;
		lastOnDemandPrototypeComparisonAttemptKey = comparisonKey;
		clearOnDemandPrototypeComparison();

		const isStaleRequest = () =>
			!browser ||
			!onDemandPrototypeEnabled ||
			!compareOneHourEnabled ||
			onDemandPrototypeComparisonRunToken !== requestToken ||
			lastPipeline !== pipeline ||
			lastLiveKey !== expectedLiveKey ||
			lastOnDemandPrototypeComparisonAttemptKey !== comparisonKey;

		try {
			await pipeline.runUtciForTimeIndex({
				timeIndex,
				numPoints,
				numHours,
				numMonths,
				format: "f32-utci",
			});
			const onDemandUtci = await pipeline.readOnDemandUtciForDebug({ numPoints });
			const baselineUtci = await pipeline.readUtcisSlice({
				monthIndex: 0,
				hourIndex: timeIndex,
				numPoints,
				numHours,
				numMonths,
			});

			const numCompared = Math.min(onDemandUtci.length, baselineUtci.length, numPoints);
			if (numCompared <= 0) {
				throw new Error("On-demand one-hour comparison produced no comparable values.");
			}

			let maxAbsDiff = 0;
			let sumSquaredDiff = 0;
			for (let index = 0; index < numCompared; index += 1) {
				const delta = onDemandUtci[index] - baselineUtci[index];
				if (!Number.isFinite(delta)) {
					throw new Error(`On-demand comparison produced a non-finite delta at index ${index}.`);
				}
				const absDelta = Math.abs(delta);
				if (absDelta > maxAbsDiff) {
					maxAbsDiff = absDelta;
				}
				sumSquaredDiff += delta * delta;
			}

			if (isStaleRequest()) {
				return;
			}

			const debugReadbackCount = getOnDemandPrototypeDebugReadbackCount() + 1;
			getParityWindow().__onDemandPrototypeComparison__ = {
				timeIndex,
				numCompared,
				maxAbsDiff,
				rmse: Math.sqrt(sumSquaredDiff / numCompared),
				debugReadbackCount,
			};
			updateOnDemandPrototypeDiagnostics({
				debugReadbackCount,
				error: undefined,
			});
		} catch (error) {
			if (isStaleRequest()) {
				return;
			}
			clearOnDemandPrototypeComparison();
			updateOnDemandPrototypeDiagnostics({
				error: error instanceof Error ? error.message : String(error),
			});
		}
	}

	function buildStrictOnDemandPrototypeDiagnostics(params: {
		strictDiagnostics?: OnDemandRuntimeDiagnostics;
		pointCount: number;
		modelId: string;
		fallbackTimeIndices: number[];
	}): OnDemandPrototypeDiagnostics {
		const { strictDiagnostics, pointCount, modelId, fallbackTimeIndices } = params;
		return {
			...createEmptyOnDemandDiagnostics(),
			...strictDiagnostics,
			bridgeAttached: false,
			visibleColorVariance: 0,
			debugReadbackCount: 0,
			dataTextureBuildCount: 0,
			navigatorGpu: Boolean(navigator.gpu),
			rendererBackend,
			modelId,
			pointCount,
			timeIndices:
				strictDiagnostics?.timeIndices?.length
					? [...strictDiagnostics.timeIndices]
					: [...fallbackTimeIndices],
			timings: strictDiagnostics?.timings ? { ...strictDiagnostics.timings } : {},
			utciRenderRequested: utciRenderMode,
			utciRenderResolved: resolvedUtciSurfaceBackend,
			debugComparisonReference: undefined,
			pythonBinComparisonActive: false,
			binComparisonEnabled: debugDiagnosticsState.binComparisonEnabled,
			binComparisonValid: debugDiagnosticsState.binComparisonValid,
			pythonBaselineStatus: undefined,
			debugComparisonMonthIndex: undefined,
			pythonComparisonHourIndex: undefined,
			webgpuComparisonHourIndex: undefined,
			pythonBinSampleComparison: undefined,
			selectedHourReadbackCount: 0,
			liveAnalysisConstructedForSelectedHour: false,
			error: undefined,
		};
	}

	$: if (acceptedGpuResidentUtciOutput && $analysisStore) {
		const colorMode = $viewerStore.colorMode;
		const nextRange = resolveAcceptedGpuResidentUtciRange({
			base: $analysisStore,
			monthIndex: acceptedGpuResidentUtciOutput.monthIndex,
			hourIndex: acceptedGpuResidentUtciOutput.hourIndex,
			colorMode,
			selectedHourUtci: acceptedGpuResidentUtciOutput.tooltipUtciValues,
		});
		const currentRange = acceptedGpuResidentUtciOutput.utciRange;
		if (nextRange.min !== currentRange.min || nextRange.max !== currentRange.max) {
			setAcceptedGpuResidentUtciOutput({
				...acceptedGpuResidentUtciOutput,
				utciRange: nextRange,
			});
			updateOnDemandPrototypeDiagnostics({
				acceptedGpuResidentUtciRange: nextRange,
			});
		}
	}

	function shouldAttemptGpuResidentSelectedHourRender(
		computeManager: ComputeManager,
	): boolean {
		return (
			resolvedUtciSurfaceBackend === "gpuNative" &&
			canUseGpuResidentRender(computeManager).available
		);
	}

	async function activateDeferredCpuFallbackIfAvailable(params: {
		requestId: number;
		monthIndex: number;
		timeIndex: number;
	}): Promise<boolean> {
		const fallback = deferredCpuFallbackSelectedHour;
		if (
			fallback &&
			fallback.requestId === params.requestId &&
			fallback.monthIndex === params.monthIndex &&
			fallback.timeIndex === params.timeIndex
		) {
			const fallbackAnalysis = buildSelectedHourLiveAnalysis({
				base: fallback.base,
				utciValues: fallback.utciValues,
				monthIndex: fallback.monthIndex,
				timeIndex: fallback.timeIndex,
			});
			setAcceptedGpuResidentUtciOutput(null);
			deferredCpuFallbackSelectedHour = null;
			liveAnalysis = fallbackAnalysis;
			comparisonStore.update((state) => ({
				...state,
				isComparing: true,
				comparisonAnalysis: fallbackAnalysis,
			}));
			return true;
		}

		const prepared = onDemandDebugPrepared;
		if (
			!prepared ||
			!doesDebugOnDemandRequestStillOwnSelection({
				prepared,
				requestId: params.requestId,
				monthIndex: params.monthIndex,
				timeIndex: params.timeIndex,
			})
		) {
			return false;
		}

		const utciValues = await prepared.pipeline.readOnDemandUtciForDebug?.({
			numPoints: prepared.numPoints,
		});
		if (!utciValues) return false;

		const fallbackAnalysis = buildSelectedHourLiveAnalysis({
			base: prepared.base,
			utciValues,
			monthIndex: params.monthIndex,
			timeIndex: params.timeIndex,
		});
		setAcceptedGpuResidentUtciOutput(null);
		deferredCpuFallbackSelectedHour = null;
		liveAnalysis = fallbackAnalysis;
		comparisonStore.update((state) => ({
			...state,
			isComparing: true,
			comparisonAnalysis: fallbackAnalysis,
		}));
		return true;
	}

	function buildOnDemandScrubStateDiagnosticsPatch(
		extra?: Partial<OnDemandPrototypeDiagnostics>,
	): Partial<OnDemandPrototypeDiagnostics> {
		return {
			selectedMonthIndex: onDemandScrubState.selectedMonthIndex,
			selectedTimeIndex: onDemandScrubState.selectedTimeIndex,
			completedMonthIndex: onDemandScrubState.completedMonthIndex,
			completedTimeIndex: onDemandScrubState.completedTimeIndex,
			activeRequestId: onDemandScrubState.activeRequestId,
			completedRequestId: onDemandScrubState.completedRequestId,
			staleResultDiscardCount: onDemandScrubState.staleResultDiscardCount,
			inFlightCount: onDemandScrubState.inFlightCount,
			scrubSampleCount: onDemandScrubState.scrubSampleCount,
			...extra,
		};
	}

	async function runDebugOnDemandSelectedHour(params: {
		base: Analysis;
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		readbackForComparison: boolean;
	}): Promise<Awaited<ReturnType<ComputeManager["runUtciForTimeIndex"]>> | undefined> {
		let pendingReadbackPublished = false;
		if (!onDemandDebugPrepared) {
			const zHeight = params.base.metadata.bounds?.z ?? 0.9;
			const signal = liveAbortController?.signal ?? new AbortController().signal;
			const prepared = await prepareWebgpuDebugInputsForCurrentSelection({
				base: params.base,
				signal,
				runId: liveRunCounter,
				zHeight,
			});
			onDemandDebugPrepared = {
				...prepared,
				base: params.base,
				signal,
				runId: liveRunCounter,
				zHeight,
				exposureReady: false,
				exposurePrecomputePromise: null,
				pendingRenderUpdate: null,
			};
		}
		const prepared = onDemandDebugPrepared;
		if (!prepared.exposureReady) {
			if (!prepared.exposurePrecomputePromise) {
				prepared.exposurePrecomputePromise = prepared.computeManager
					.runExposurePrecompute({
						numPoints: prepared.numPoints,
						numHours: prepared.numHours,
						numMonths: prepared.numMonths,
					})
					.then(() => {
						prepared.exposureReady = true;
					})
					.finally(() => {
						if (prepared.exposurePrecomputePromise) {
							prepared.exposurePrecomputePromise = null;
						}
					});
			}
			await prepared.exposurePrecomputePromise;
			if (onDemandDebugPrepared !== prepared || prepared.signal.aborted) {
				return undefined;
			}
		}
		if (prepared.signal.aborted) {
			return undefined;
		}
		legacySelectedHourDispatchCount += 1;

		const started = startOnDemandRequest(onDemandScrubState, {
			monthIndex: params.monthIndex,
			timeIndex: params.timeIndex,
		});
		onDemandScrubState = started.state;

		const output = await prepared.computeManager.runUtciForTimeIndex({
			timeIndex: params.timeIndex,
			numPoints: prepared.numPoints,
			numHours: prepared.numHours,
			numMonths: prepared.numMonths,
			format: "f32-utci",
		});

		const forcedOverlapMs = Number($page.url.searchParams.get("forceOnDemandOverlapMs") ?? "0");
		if (forcedOverlapMs > 0) {
			await new Promise((resolve) => setTimeout(resolve, forcedOverlapMs));
		}

		const completed = markOnDemandRequestCompleted(onDemandScrubState, started.request);
		onDemandScrubState = completed.state;
		if (!completed.accepted) {
			destroyOnDemandGpuBuffer(output);
			return undefined;
		}

		updateOnDemandPrototypeDiagnostics(
			buildOnDemandScrubStateDiagnosticsPatch({
				pendingReadbackRequestId: started.request.requestId,
				pendingReadbackTimeIndex: params.timeIndex,
			}),
		);
		pendingReadbackPublished = true;

		try {
			const postAcceptDelayMs = Number($page.url.searchParams.get("forceOnDemandPostAcceptDelayMs") ?? "0");
			if (postAcceptDelayMs > 0) {
				await new Promise((resolve) => setTimeout(resolve, postAcceptDelayMs));
			}

			const useGpuResidentSelectedHourRender =
				shouldAttemptGpuResidentSelectedHourRender(prepared.computeManager);
			const needsSelectedHourAnalysisForImmediateRender = !useGpuResidentSelectedHourRender;
			const shouldReadbackForComparison =
				params.readbackForComparison && parityMode && debugDiagnosticsState.binComparisonValid;
			const shouldReadbackSelectedHour =
				shouldReadbackForComparison ||
				needsSelectedHourAnalysisForImmediateRender;
			const selectedHourReadbackStart = shouldReadbackSelectedHour
				? performance.now()
				: undefined;
			const selectedHourUtci = shouldReadbackSelectedHour
				? await prepared.pipeline.readOnDemandUtciForDebug?.({
						numPoints: prepared.numPoints,
					})
				: undefined;
			const selectedHourReadbackMs =
				selectedHourReadbackStart === undefined
					? undefined
					: performance.now() - selectedHourReadbackStart;

			if (
				!doesDebugOnDemandRequestStillOwnSelection({
					prepared,
					requestId: started.request.requestId,
					monthIndex: params.monthIndex,
					timeIndex: params.timeIndex,
				})
			) {
				onDemandScrubState = {
					...onDemandScrubState,
					staleResultDiscardCount: onDemandScrubState.staleResultDiscardCount + 1,
				};
				updateOnDemandPrototypeDiagnostics(
					buildOnDemandScrubStateDiagnosticsPatch({
						pendingReadbackRequestId: undefined,
						pendingReadbackTimeIndex: undefined,
					}),
				);
				pendingReadbackPublished = false;
				destroyOnDemandGpuBuffer(output);
				return output;
			}

			let selectedHourAnalysisBuildMs: number | undefined;
			let selectedHourAnalysis: Analysis | undefined;
			if (selectedHourUtci && needsSelectedHourAnalysisForImmediateRender) {
				const selectedHourAnalysisBuildStart = performance.now();
				selectedHourAnalysis = buildSelectedHourLiveAnalysis({
					base: prepared.base,
					utciValues: selectedHourUtci,
					monthIndex: params.monthIndex,
					timeIndex: params.timeIndex,
				});
				selectedHourAnalysisBuildMs = performance.now() - selectedHourAnalysisBuildStart;
			}

			prepared.pendingRenderUpdate = {
				requestId: started.request.requestId,
				monthIndex: params.monthIndex,
				timeIndex: params.timeIndex,
				startedAt: performance.now(),
			};

			if (useGpuResidentSelectedHourRender) {
				const acceptedUtciRange = resolveAcceptedGpuResidentUtciRange({
					base: prepared.base,
					monthIndex: params.monthIndex,
					hourIndex: params.hourIndex,
					colorMode: $viewerStore.colorMode,
					selectedHourUtci,
				});
				liveAnalysis = null;
				setAcceptedGpuResidentUtciOutput({
					requestId: started.request.requestId,
					monthIndex: params.monthIndex,
					hourIndex: params.hourIndex,
					timeIndex: params.timeIndex,
					output,
					utciRange: acceptedUtciRange,
					tooltipUtciValues: selectedHourUtci,
				});
				updateOnDemandPrototypeDiagnostics({
					acceptedGpuResidentUtciRange: acceptedUtciRange,
				});
				deferredCpuFallbackSelectedHour = selectedHourUtci
					? {
						requestId: started.request.requestId,
						monthIndex: params.monthIndex,
						hourIndex: params.hourIndex,
						timeIndex: params.timeIndex,
						base: prepared.base,
						utciValues: selectedHourUtci,
					}
					: null;
				comparisonStore.update((state) => ({
					...state,
					isComparing: true,
					comparisonAnalysis: null,
				}));
			} else {
				setAcceptedGpuResidentUtciOutput(null);
				deferredCpuFallbackSelectedHour = null;
				if (selectedHourAnalysis) {
					liveAnalysis = selectedHourAnalysis;
					comparisonStore.update((state) => ({
						...state,
						isComparing: true,
						comparisonAnalysis: selectedHourAnalysis,
					}));
				} else {
					liveAnalysis = null;
					comparisonStore.update((state) => ({
						...state,
						isComparing: false,
						comparisonAnalysis: null,
					}));
				}
			}
			updateOnDemandPrototypeColdStartTiming(
				"firstSelectedHourReadyMs",
				performance.now() - prepared.coldStartStartedAt,
			);
			const firstSelectedHourReadyMs =
				getParityWindow().__onDemandPrototypeDiagnostics__?.timings?.firstSelectedHourReadyMs;

			const pipelineDiagnostics = prepared.computeManager.getOnDemandDiagnostics();
			const pythonBinSampleComparison =
				shouldReadbackForComparison && selectedHourUtci
				? compareSampledPointsAgainstPythonBin({
					referenceAnalysis: params.base,
					debugValues: selectedHourUtci,
					monthIndex: params.monthIndex,
					hourIndex: params.hourIndex,
					timeIndex: params.timeIndex,
				})
				: undefined;
			updateOnDemandPrototypeDiagnostics({
				...pipelineDiagnostics,
				...buildOnDemandScrubStateDiagnosticsPatch(
					useGpuResidentSelectedHourRender
						? {
							pendingReadbackRequestId: started.request.requestId,
							pendingReadbackTimeIndex: params.timeIndex,
						}
						: {
							pendingReadbackRequestId: undefined,
							pendingReadbackTimeIndex: undefined,
						},
				),
				debugComparisonReference: shouldReadbackForComparison ? "python-bin" : undefined,
				pythonBinComparisonActive: shouldReadbackForComparison,
				binComparisonEnabled: debugDiagnosticsState.binComparisonEnabled,
				binComparisonValid: debugDiagnosticsState.binComparisonValid,
				pythonBaselineStatus: parityMode
					? debugDiagnosticsState.binComparisonValid
						? "available-august"
						: "unavailable-non-august"
					: undefined,
				debugComparisonMonthIndex: shouldReadbackForComparison ? params.monthIndex : undefined,
				pythonComparisonHourIndex: shouldReadbackForComparison ? params.hourIndex : undefined,
				webgpuComparisonHourIndex: shouldReadbackForComparison ? params.hourIndex : undefined,
				pythonBinSampleComparison,
				appVisibleSelectedHour: useGpuResidentSelectedHourRender
					? false
					: Boolean(selectedHourAnalysis),
				selectedHourReadbackCount:
					useGpuResidentSelectedHourRender || !selectedHourAnalysis ? 0 : 1,
				liveAnalysisConstructedForSelectedHour:
					useGpuResidentSelectedHourRender ? false : Boolean(selectedHourAnalysis),
				renderTransport:
					useGpuResidentSelectedHourRender
						? "none"
						: selectedHourAnalysis
							? "cpu-uploaded-selected-hour"
							: "none",
				gpuResidentCopyStatus: useGpuResidentSelectedHourRender ? "pending" : undefined,
				timings: prepareSelectedHourCycleTimings({
					existingTimings: getParityWindow().__onDemandPrototypeDiagnostics__?.timings,
					pipelineTimings: pipelineDiagnostics?.timings,
					firstSelectedHourReadyMs,
					selectedHourReadbackMs: needsSelectedHourAnalysisForImmediateRender
						? selectedHourReadbackMs
						: undefined,
					selectedHourAnalysisBuildMs: useGpuResidentSelectedHourRender
						? undefined
						: selectedHourAnalysisBuildMs,
				}),
			});
			pendingReadbackPublished = false;
		} catch (error) {
			if (pendingReadbackPublished) {
				updateOnDemandPrototypeDiagnostics(
					buildOnDemandScrubStateDiagnosticsPatch({
						pendingReadbackRequestId: undefined,
						pendingReadbackTimeIndex: undefined,
					}),
				);
			}
			throw error;
		}

		return output;
	}

	function compareFloatArrays(
		hour: number,
		onDemand: Float32Array,
		baseline: Float32Array,
	): OnDemandMultiHourComparisonResult {
		const numCompared = Math.min(onDemand.length, baseline.length);
		if (numCompared <= 0) {
			throw new Error(`On-demand multi-hour comparison produced no comparable values for hour ${hour}.`);
		}

		let maxAbsDiff = 0;
		let sumSquaredDiff = 0;
		for (let index = 0; index < numCompared; index += 1) {
			const delta = onDemand[index] - baseline[index];
			if (!Number.isFinite(delta)) {
				throw new Error(
					`On-demand multi-hour comparison produced a non-finite delta at hour ${hour}, index ${index}.`,
				);
			}
			const absDelta = Math.abs(delta);
			if (absDelta > maxAbsDiff) {
				maxAbsDiff = absDelta;
			}
			sumSquaredDiff += delta * delta;
		}

		const result: OnDemandMultiHourComparisonResult = {
			hour,
			numCompared,
			maxAbsDiff,
			rmse: Math.sqrt(sumSquaredDiff / numCompared),
		};

		if (numCompared > 31079) {
			result.onDemandAt31079 = onDemand[31079];
			result.baselineAt31079 = baseline[31079];
			result.diffAt31079 = onDemand[31079] - baseline[31079];
		}

		return result;
	}

	function compareMonthHourFloatArrays(
		pair: OnDemandMonthHourComparisonPair,
		onDemand: Float32Array,
		baseline: Float32Array,
	): OnDemandMonthHourComparisonPairResult {
		const baseResult = compareFloatArrays(pair.hourIndex, onDemand, baseline);
		return {
			monthIndex: pair.monthIndex,
			hourIndex: pair.hourIndex,
			timeIndex: pair.timeIndex,
			numCompared: baseResult.numCompared,
			maxAbsDiff: baseResult.maxAbsDiff,
			rmse: baseResult.rmse,
			onDemandAt31079: baseResult.onDemandAt31079,
			baselineAt31079: baseResult.baselineAt31079,
			diffAt31079: baseResult.diffAt31079,
		};
	}

	function compareSampledPointsAgainstPythonBin(params: {
		referenceAnalysis: Analysis | null;
		debugValues: Float32Array;
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
	}): OnDemandPythonSampleComparison {
		const uniquePointIndices = [...new Set([0, 31079, params.debugValues.length - 1])].filter(
			(pointIndex) => pointIndex >= 0 && pointIndex < params.debugValues.length,
		);
		const samples: OnDemandPythonSampleRecord[] = [];
		let maxAbsDiff = 0;

		for (const pointIndex of uniquePointIndices) {
			const referenceHourIndex = getEffectiveHourIndex(
				params.referenceAnalysis,
				params.hourIndex,
				params.monthIndex,
			);
			const referenceValue = getUtciAtPoint(params.referenceAnalysis, pointIndex, referenceHourIndex);
			if (referenceValue === null) {
				continue;
			}

			const debugValue = params.debugValues[pointIndex];
			if (!Number.isFinite(debugValue) || !Number.isFinite(referenceValue)) {
				throw new Error(
					`Python sampled comparison produced a non-finite value at monthIndex ${params.monthIndex}, timeIndex ${params.timeIndex}, point ${pointIndex}.`,
				);
			}

			const absDiff = Math.abs(debugValue - referenceValue);
			if (absDiff > maxAbsDiff) {
				maxAbsDiff = absDiff;
			}

			samples.push({
				pointIndex,
				debugValue,
				referenceValue,
				absDiff,
			});
		}

		return {
			numCompared: samples.length,
			maxAbsDiff,
			samples,
		};
	}

	async function createSeparateRunAllBaselineManager(params: {
		prepared: Awaited<ReturnType<typeof prepareWebgpuDebugInputsForCurrentSelection>>;
		signal: AbortSignal;
	}): Promise<{
		baselineManager: ComputeManager;
		baselinePipeline: UTCIComputePipeline;
	}> {
		const { prepared, signal } = params;
		const { analysisParams, numHours, numMonths, gridResolution } = prepared;
		const baseMetadata = analysisParams.baseMetadata;
		const bounds = baseMetadata.bounds as
			| { x_min: number; x_max: number; y_min: number; y_max: number; z?: number }
			| undefined;
		if (!bounds) {
			throw new Error(
				"Analysis metadata is missing bounds; separate runAll baseline cannot build canonical grid.",
			);
		}

		const baselinePipeline = await createWebgpuUtciPipeline({ enableDiagnostics: parityMode });
		const baselineManager = new ComputeManager(baselinePipeline, {
			numMonths,
			numHoursPerDay: numHours,
			startMonth: analysisParams.startMonth,
		});
		const coordinateSystem =
			(baseMetadata.coordinate_system as "xy_ground" | "xz_ground") ?? "xy_ground";
		const gridOriginOffset = getGridOriginOffset(baseMetadata);
		const computeGridHeight =
			(bounds.z ?? analysisParams.zHeight) + PARITY_SAMPLE_HEIGHT_OFFSET_M;
		const sunVectorsFixture =
			numMonths === 1
				? buildSunVectorsFixtureFromMetadata({
					baseMetadata,
					numHours,
					numMonths,
				})
				: undefined;

		await baselineManager.initFromModelAndWeather({
			serializedBvh: analysisParams.workerResult.serializedBvh,
			sunVectorsFixture,
			useRectangularGridFromBounds: true,
			analysisBounds: bounds,
			coordinateSystem,
			gridOriginOffset,
			epwContent: analysisParams.epwContent,
			gridResolution,
			zHeight: computeGridHeight,
			signal,
		});

		return {
			baselineManager,
			baselinePipeline,
		};
	}

	async function runOnDemandMultiHourComparison(params: {
		prepared: Awaited<ReturnType<typeof prepareWebgpuDebugInputsForCurrentSelection>>;
		signal: AbortSignal;
	}): Promise<void> {
		if (!browser || !onDemandPrototypeEnabled || !strictExposureOnlyEnabled || !compareHoursEnabled) {
			clearOnDemandMultiHourComparison();
			return;
		}

		const { prepared, signal } = params;
		const strictPipeline = prepared.pipeline;
		if (!strictPipeline.readOnDemandUtciForDebug) {
			clearOnDemandMultiHourComparison();
			updateOnDemandPrototypeDiagnostics(
				{
					error:
						"Current WebGPU UTCI pipeline does not expose the strict multi-hour debug readback API.",
				},
				{ computeManager: prepared.computeManager },
			);
			return;
		}

		clearOnDemandMultiHourComparison();
		let baselinePipeline: UTCIComputePipeline | null = null;

		try {
			const { baselineManager, baselinePipeline: createdBaselinePipeline } =
				await createSeparateRunAllBaselineManager({
					prepared,
					signal,
				});
			baselinePipeline = createdBaselinePipeline;
			const strictPath =
				prepared.computeManager.getOnDemandDiagnostics() ?? createEmptyOnDemandDiagnostics();

			const hourResults: OnDemandMultiHourComparisonResult[] = [];
			for (const hour of compareHours) {
				if (signal.aborted) {
					return;
				}

				await prepared.computeManager.runUtciForTimeIndex({
					timeIndex: hour,
					numPoints: prepared.numPoints,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
					format: "f32-utci",
				});
				const onDemandUtci = await strictPipeline.readOnDemandUtciForDebug({
					numPoints: prepared.numPoints,
				});
				const baselineUtci = await baselineManager.getUtcisForMonthHour({
					monthIndex: 0,
					hourIndex: hour,
					numPoints: prepared.numPoints,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
				});
				hourResults.push(compareFloatArrays(hour, onDemandUtci, baselineUtci));
			}

			if (signal.aborted) {
				return;
			}
			const knownPointResults =
				prepared.numPoints > 31079
					? hourResults.filter(
						(result) =>
							(result.hour === 16 || result.hour === 17) &&
							result.onDemandAt31079 !== undefined,
					)
					: [];

			getParityWindow().__onDemandMultiHourComparison__ = {
				baselineSource: "separateRunAll",
				baselineMonthContext: {
					monthIndex: 0,
					sliceKind: "representative-day-full-year",
					note: "compareHours uses the separate runAll baseline monthIndex 0 representative-day slice.",
				},
				strictPath,
				hours: [...compareHours],
				hourResults,
				...(knownPointResults.length > 0
					? {
						knownPoint31079: {
							pointIndex: 31079,
							hours: knownPointResults.map((result) => ({
								hour: result.hour,
								onDemand: result.onDemandAt31079 ?? Number.NaN,
								baseline: result.baselineAt31079 ?? Number.NaN,
								diff: result.diffAt31079 ?? Number.NaN,
							})),
						},
					}
					: {}),
			};

			updateOnDemandPrototypeDiagnostics(
				buildStrictOnDemandPrototypeDiagnostics({
					strictDiagnostics: strictPath,
					pointCount: prepared.numPoints,
					modelId: prepared.analysisParams.baseMetadata.model_file ?? analysisId,
					fallbackTimeIndices: compareHours,
				}),
				{ replace: true, computeManager: prepared.computeManager },
			);
		} catch (error) {
			if (signal.aborted) {
				return;
			}
			clearOnDemandMultiHourComparison();
			updateOnDemandPrototypeDiagnostics(
				{ error: error instanceof Error ? error.message : String(error) },
				{ computeManager: prepared.computeManager },
			);
		} finally {
			baselinePipeline?.dispose?.();
		}
	}

	async function runOnDemandMonthHourComparison(params: {
		prepared: Awaited<ReturnType<typeof prepareWebgpuDebugInputsForCurrentSelection>>;
		signal: AbortSignal;
		comparisonKey: string;
	}): Promise<void> {
		if (
			!browser ||
			!onDemandPrototypeEnabled ||
			!strictExposureOnlyEnabled ||
			!compareMonthHoursEnabled
		) {
			clearOnDemandMonthHourComparison();
			return;
		}

		const { prepared, signal, comparisonKey } = params;
		const requestToken = ++onDemandMonthHourComparisonRunToken;
		lastOnDemandMonthHourComparisonAttemptKey = comparisonKey;
		const strictPipeline = prepared.pipeline;
		const publishResult = (result: OnDemandMonthHourComparisonResult): void => {
			if (
				!browser ||
				signal.aborted ||
				onDemandMonthHourComparisonRunToken !== requestToken ||
				lastOnDemandMonthHourComparisonAttemptKey !== comparisonKey
			) {
				return;
			}
			getParityWindow().__onDemandMonthHourComparison__ = result;
		};

		publishResult({
			status: "running",
			baselineSource: "separateRunAll",
			pairs: [],
		});

		if (!strictPipeline.readOnDemandUtciForDebug) {
			const errorMessage =
				"Current WebGPU UTCI pipeline does not expose the strict month/hour debug readback API.";
			publishResult({
				status: "error",
				baselineSource: "separateRunAll",
				pairs: [],
				error: errorMessage,
			});
			updateOnDemandPrototypeDiagnostics(
				{ error: errorMessage },
				{ computeManager: prepared.computeManager },
			);
			return;
		}

		let baselinePipeline: UTCIComputePipeline | null = null;
		try {
			const pairs = getCompareMonthHourPairsFromQuery();
			const { baselineManager, baselinePipeline: createdBaselinePipeline } =
				await createSeparateRunAllBaselineManager({
					prepared,
					signal,
				});
			baselinePipeline = createdBaselinePipeline;

			const results: OnDemandMonthHourComparisonPairResult[] = [];
			for (const pair of pairs) {
				if (signal.aborted) return;

				await prepared.computeManager.runUtciForTimeIndex({
					timeIndex: pair.timeIndex,
					numPoints: prepared.numPoints,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
					format: "f32-utci",
				});
				const onDemandUtci = await strictPipeline.readOnDemandUtciForDebug({
					numPoints: prepared.numPoints,
				});
				const baselineUtci = await baselineManager.getUtcisForMonthHour({
					monthIndex: pair.monthIndex,
					hourIndex: pair.hourIndex,
					numPoints: prepared.numPoints,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
				});
				results.push(compareMonthHourFloatArrays(pair, onDemandUtci, baselineUtci));
			}

			publishResult({
				status: "complete",
				baselineSource: "separateRunAll",
				pairs: results,
			});
			updateOnDemandPrototypeDiagnostics({}, { computeManager: prepared.computeManager });
		} catch (error) {
			if (signal.aborted) return;
			const errorMessage = error instanceof Error ? error.message : String(error);
			publishResult({
				status: "error",
				baselineSource: "separateRunAll",
				pairs: [],
				error: errorMessage,
			});
			updateOnDemandPrototypeDiagnostics(
				{ error: errorMessage },
				{ computeManager: prepared.computeManager },
			);
		} finally {
			baselinePipeline?.dispose?.();
		}
	}

	function attachSyntheticBridge(targetModel: Group): void {
		const nextKey = getSyntheticBridgeIdentity(targetModel);
		if (syntheticBridgeKey === nextKey && syntheticBridge) return;

		disposeSyntheticBridge();
		resetSyntheticBridgeDiagnostics();

		try {
			const { center, size } = getBoundsCenterAndSize(targetModel);
			syntheticBridge = createSyntheticGpuUtciBridge({ center, size });
			syntheticBridgeKey = nextKey;
		} catch (error) {
			resetSyntheticBridgeDiagnostics(
				error instanceof Error ? error.message : String(error),
			);
		}
	}

	function resetAnalysisSceneState(): void {
		liveAbortController?.abort();
		model = null;
		utciMesh = null;
		liveUtciMesh = null;
		modelFileForLoadedModel = null;
		liveAnalysis = null;
		liveError = null;
		liveComputeProgress = null;
		liveUtciSurfaceDiagnostics = {};
		setAcceptedGpuResidentUtciOutput(null);
		deferredCpuFallbackSelectedHour = null;
		lastLiveKey = null;
		onDemandDebugPrepared = undefined;
		lastDebugOnDemandScrubTriggerKey = null;
		debugOnDemandScrubScheduleRunId += 1;
		onDemandScrubState = createOnDemandScrubState();
		invalidateOnDemandPrototypeComparison();
		disposeSyntheticBridge();
		if (browser) {
			const parityWindow = getParityWindow();
			parityWindow.__parityModel__ = null;
			parityWindow.__parityThree__ = undefined;
		}
		if (onDemandPrototypeEnabled && compareOneHourEnabled) {
			updateOnDemandPrototypeDiagnostics({
				debugReadbackCount: 0,
				dataTextureBuildCount: 0,
				error: undefined,
			});
		}
		if (onDemandPrototypeEnabled) {
			resetSyntheticBridgeDiagnostics();
		}
	}

	function handleSyntheticBridgeMount(ref: Group): void {
		if (!syntheticBridge || !syntheticBridgeKey) return;

		ref.add(syntheticBridge.group);
		syntheticBridgeMountedKey = syntheticBridgeKey;
		updateOnDemandPrototypeDiagnostics({
			bridgeAttached: true,
			debugReadbackCount: 0,
			dataTextureBuildCount: 0,
			visibleColorVariance: 0,
			error: undefined,
		});
	}

	function handleRendererDiagnostics(diagnostics: {
		rendererBackend: UtciRendererBackend;
		rendererDevice?: GPUDevice;
		rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
		rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
		error?: string;
	}): Promise<void> {
		const previousRendererDevice = rendererDeviceForDebug;
		const previousOnDemandDiagnostics = onDemandPrototypeEnabled
			? getParityWindow().__onDemandPrototypeDiagnostics__
			: undefined;
		rendererBackend = diagnostics.rendererBackend;
		rendererDeviceForDebug = diagnostics.rendererDevice;
		rendererRequiredLimits = diagnostics.rendererRequiredLimits;
		rendererDeviceLimits = diagnostics.rendererDeviceLimits;
		if (!onDemandPrototypeEnabled) return;

		updateOnDemandPrototypeDiagnostics({
			rendererBackend: diagnostics.rendererBackend,
			rendererRequestedMaxStorageBufferBindingSize:
				diagnostics.rendererRequiredLimits?.maxStorageBufferBindingSize,
			rendererRequestedMaxBufferSize: diagnostics.rendererRequiredLimits?.maxBufferSize,
			rendererDeviceMaxStorageBufferBindingSize:
				diagnostics.rendererDeviceLimits?.maxStorageBufferBindingSize,
			rendererDeviceMaxBufferSize: diagnostics.rendererDeviceLimits?.maxBufferSize,
			utciRenderRequested: utciRenderMode,
			utciRenderResolved: resolveUtciSurfaceBackend(
				utciRenderMode,
				diagnostics.rendererBackend,
			),
			error: diagnostics.error,
		});

		const rendererDeviceJustBecameAvailable =
			previousRendererDevice === undefined && diagnostics.rendererDevice !== undefined;
		if (
			rendererDeviceJustBecameAvailable &&
			onDemandDebugPrepared?.deviceSource === "standalone" &&
			debugOnDemandMode === "f32" &&
			resolvedUtciSurfaceBackend === "gpuNative"
		) {
			onDemandDebugPrepared.pipeline.dispose?.();
			if (lastPipeline === onDemandDebugPrepared.pipeline) {
				lastPipeline = null;
			}
			onDemandDebugPrepared = undefined;
			setAcceptedGpuResidentUtciOutput(null);
			deferredCpuFallbackSelectedHour = null;
			lastDebugOnDemandScrubTriggerKey = null;
			debugOnDemandScrubScheduleRunId += 1;
			updateOnDemandPrototypeDiagnostics({
				renderTransport: "none",
				appVisibleSelectedHour: false,
				utciSurfaceSource: undefined,
				selectedHourReadbackCount: 0,
				selectedHourTransferCount: 0,
				gpuResidentCopyStatus: "idle",
				gpuResidentCopyError: undefined,
			});
			if (!useDebugSharedSelectedHourHost) {
				scheduleDebugOnDemandScrubRecompute(debugOnDemandSelectionKey);
			}
		}
		const retryGpuResidentSelectedHour =
			rendererDeviceJustBecameAvailable &&
			previousOnDemandDiagnostics?.renderTransport === "cpu-uploaded-selected-hour" &&
			previousOnDemandDiagnostics?.gpuResidentRenderAvailable === false &&
			previousOnDemandDiagnostics?.sameDeviceForComputeAndRender === null &&
			debugOnDemandMode === "f32" &&
			!strictExposureOnlyEnabled &&
			mounted &&
			$analysisStore != null &&
			model != null &&
			onDemandDebugPrepared != null &&
			modelFileForLoadedModel === $analysisStore.metadata.model_file &&
			acceptedGpuResidentUtciOutput == null &&
			canUseGpuResidentRender(onDemandDebugPrepared.computeManager).available;

		if (retryGpuResidentSelectedHour) {
			const retryBase = $analysisStore;
			const retryMonthIndex = debugOnDemandSelection.monthIndex;
			const retryHourIndex = debugOnDemandSelection.hourIndex;
			const retryTimeIndex = debugOnDemandSelection.timeIndex;
			requestAnimationFrame(() => {
				if (
					!browser ||
					!mounted ||
					!onDemandPrototypeEnabled ||
					debugOnDemandMode !== "f32" ||
					useDebugSharedSelectedHourHost ||
					strictExposureOnlyEnabled ||
					!$analysisStore ||
					!model ||
					!onDemandDebugPrepared ||
					$analysisStore !== retryBase ||
					modelFileForLoadedModel !== $analysisStore.metadata.model_file
				) {
					return;
				}

				void runDebugOnDemandSelectedHour({
					base: retryBase,
					monthIndex: retryMonthIndex,
					hourIndex: retryHourIndex,
					timeIndex: retryTimeIndex,
					readbackForComparison: parityMode,
				});
			});
		}
	}

	async function handleLiveUtciSurfaceDiagnostics(diagnostics: {
		utciSurfaceSource?: string;
		selectedHourTransferCount?: number;
		dataTextureBuildCount?: number;
		gpuResidentCopyStatus?: "idle" | "pending" | "complete" | "failed";
		gpuResidentCopyError?: string;
		gpuResidentCopyRequestId?: number;
	} & SelectedHourRenderTimingSubsteps): Promise<void> {
		liveUtciSurfaceDiagnostics = diagnostics;
		if (!onDemandPrototypeEnabled) return;

		const nextDiagnostics: Partial<OnDemandPrototypeDiagnostics> = {};
		if (!Object.keys(diagnostics).length) {
			Object.assign(
				nextDiagnostics,
				buildGpuResidentSurfaceResetPatch({
					existingTimings: getParityWindow().__onDemandPrototypeDiagnostics__?.timings,
				}),
			);
		} else {
			if (diagnostics.utciSurfaceSource !== undefined) {
				nextDiagnostics.utciSurfaceSource = diagnostics.utciSurfaceSource;
			}
			if (diagnostics.selectedHourTransferCount !== undefined) {
				nextDiagnostics.selectedHourTransferCount = diagnostics.selectedHourTransferCount;
			}
			if (diagnostics.dataTextureBuildCount !== undefined) {
				nextDiagnostics.dataTextureBuildCount = diagnostics.dataTextureBuildCount;
			}
			if (diagnostics.gpuResidentCopyStatus !== undefined) {
				nextDiagnostics.gpuResidentCopyStatus = diagnostics.gpuResidentCopyStatus;
			}
			if ("gpuResidentCopyError" in diagnostics) {
				nextDiagnostics.gpuResidentCopyError = diagnostics.gpuResidentCopyError;
			}
			if (
				diagnostics.gpuResidentCopyStatus === "complete" &&
				getParityWindow().__onDemandPrototypeDiagnostics__?.sameDeviceForComputeAndRender === true &&
				diagnostics.utciSurfaceSource === "compute-buffer-selected-hour"
			) {
				if (
					diagnostics.gpuResidentCopyRequestId !== undefined &&
					deferredCpuFallbackSelectedHour?.requestId === diagnostics.gpuResidentCopyRequestId
				) {
					deferredCpuFallbackSelectedHour = null;
				}
				nextDiagnostics.renderTransport = "compute-buffer-selected-hour";
				nextDiagnostics.selectedHourReadbackCount = 0;
				nextDiagnostics.selectedHourTransferCount = 0;
				nextDiagnostics.dataTextureBuildCount = 0;
				nextDiagnostics.appVisibleSelectedHour = true;
				nextDiagnostics.liveAnalysisConstructedForSelectedHour = false;
				nextDiagnostics.pendingReadbackRequestId = undefined;
				nextDiagnostics.pendingReadbackTimeIndex = undefined;
			} else if (
				diagnostics.gpuResidentCopyStatus === "failed" &&
				diagnostics.gpuResidentCopyRequestId !== undefined &&
				onDemandDebugPrepared &&
				await activateDeferredCpuFallbackIfAvailable({
					requestId: diagnostics.gpuResidentCopyRequestId,
					monthIndex:
						deferredCpuFallbackSelectedHour?.monthIndex ??
						debugOnDemandSelection.monthIndex,
					timeIndex:
						deferredCpuFallbackSelectedHour?.timeIndex ??
						debugOnDemandSelection.timeIndex,
				})
			) {
				nextDiagnostics.renderTransport = "cpu-uploaded-selected-hour";
				nextDiagnostics.selectedHourReadbackCount = 1;
				nextDiagnostics.appVisibleSelectedHour = true;
				nextDiagnostics.liveAnalysisConstructedForSelectedHour = true;
				nextDiagnostics.pendingReadbackRequestId = undefined;
				nextDiagnostics.pendingReadbackTimeIndex = undefined;
			}
		}
		const prepared = onDemandDebugPrepared;
		const pendingRenderUpdate = prepared?.pendingRenderUpdate;
		if (
			prepared &&
			pendingRenderUpdate &&
			(
				(diagnostics.selectedHourTransferCount ?? 0) > 0 ||
				(diagnostics.dataTextureBuildCount ?? 0) > 0 ||
				diagnostics.gpuResidentCopyStatus === "complete"
			)
		) {
			const stillOwned = doesDebugOnDemandRequestStillOwnSelection({
				prepared,
				requestId: pendingRenderUpdate.requestId,
				monthIndex: pendingRenderUpdate.monthIndex,
				timeIndex: pendingRenderUpdate.timeIndex,
			});
			if (
				stillOwned
			) {
				const surfaceUpdateMs = performance.now() - pendingRenderUpdate.startedAt;
				updateOnDemandPrototypeColdStartTiming(
					"firstSelectedHourVisibleMs",
					performance.now() - prepared.coldStartStartedAt,
				);
				const firstSelectedHourVisibleMs =
					getParityWindow().__onDemandPrototypeDiagnostics__?.timings?.firstSelectedHourVisibleMs;
				nextDiagnostics.timings = mergeSelectedHourRenderTimings({
					existingTimings: getParityWindow().__onDemandPrototypeDiagnostics__?.timings,
					renderUpdateMs: surfaceUpdateMs,
					gpuSurfaceUpdateMs: surfaceUpdateMs,
					firstSelectedHourVisibleMs,
					renderSubsteps: {
						renderSceneSyncStartDelayMs: diagnostics.renderSceneSyncStartDelayMs,
						renderSceneSyncTotalMs: diagnostics.renderSceneSyncTotalMs,
						renderLayoutBuildMs: diagnostics.renderLayoutBuildMs,
						renderSurfaceMeshMs: diagnostics.renderSurfaceMeshMs,
						renderStorageInitWaitMs: diagnostics.renderStorageInitWaitMs,
						renderBufferCopyMs: diagnostics.renderBufferCopyMs,
						renderQueueDrainMs: diagnostics.renderQueueDrainMs,
					},
				});
				prepared.pendingRenderUpdate = null;
			} else if (!stillOwned) {
				prepared.pendingRenderUpdate = null;
			}
		}

		updateOnDemandPrototypeDiagnostics(nextDiagnostics);
	}

	$: if (browser) {
		if (onDemandPrototypeEnabled) {
			if (!wasOnDemandPrototypeEnabled) {
				initializeOnDemandPrototypeDiagnostics();
			} else if (!getParityWindow().__onDemandPrototypeDiagnostics__) {
				initializeOnDemandPrototypeDiagnostics();
			}
		} else if (wasOnDemandPrototypeEnabled) {
			onDemandPrototypeStatus = "idle";
			onDemandPrototypeError = null;
			onDemandDebugPrepared = undefined;
			debugOnDemandScrubScheduleRunId += 1;
			onDemandScrubState = createOnDemandScrubState();
			liveUtciSurfaceDiagnostics = {};
			setAcceptedGpuResidentUtciOutput(null);
			deferredCpuFallbackSelectedHour = null;
			getParityWindow().__onDemandPrototypeDiagnostics__ = undefined;
			invalidateOnDemandPrototypeComparison();
		}
		wasOnDemandPrototypeEnabled = onDemandPrototypeEnabled;
	}

	$: if (browser) {
		if (compareOneHourEnabled && !wasCompareOneHourEnabled) {
			invalidateOnDemandPrototypeComparison();
			if (onDemandPrototypeEnabled) {
				updateOnDemandPrototypeDiagnostics({
					debugReadbackCount: 0,
					dataTextureBuildCount: 0,
					error: undefined,
				});
			}
		} else if (!compareOneHourEnabled && wasCompareOneHourEnabled) {
			invalidateOnDemandPrototypeComparison();
			if (onDemandPrototypeEnabled) {
				updateOnDemandPrototypeDiagnostics({
					debugReadbackCount: 0,
					dataTextureBuildCount: 0,
					error: undefined,
				});
			}
		}
		wasCompareOneHourEnabled = compareOneHourEnabled;
	}

	$: if (browser && onDemandPrototypeEnabled) {
		updateOnDemandPrototypeDiagnostics({
			rendererBackend,
			rendererRequestedMaxStorageBufferBindingSize:
				rendererRequiredLimits?.maxStorageBufferBindingSize,
			rendererRequestedMaxBufferSize: rendererRequiredLimits?.maxBufferSize,
			rendererDeviceMaxStorageBufferBindingSize:
				rendererDeviceLimits?.maxStorageBufferBindingSize,
			rendererDeviceMaxBufferSize: rendererDeviceLimits?.maxBufferSize,
			utciRenderRequested: utciRenderMode,
			utciRenderResolved: resolvedUtciSurfaceBackend,
		});
	}

	$: if (browser) {
		const comparisonKey = getOnDemandPrototypeComparisonKey();
		if (
			comparisonKey &&
			lastOnDemandPrototypeComparisonAttemptKey !== comparisonKey &&
			lastPipeline &&
			liveAnalysis
		) {
			void runOnDemandPrototypeComparison({
				pipeline: lastPipeline,
				numPoints: liveAnalysis.data.numPositions,
				numHours: liveAnalysis.metadata.hours.length || 24,
				numMonths: liveAnalysis.metadata.num_months ?? (parityMode ? 1 : 12),
				comparisonKey,
			});
		}
	}

	$: if (syntheticBridgeEnabled && model) {
		attachSyntheticBridge(model);
	}

	$: if (
		syntheticBridgeEnabled &&
		syntheticBridge &&
		syntheticBridgeKey &&
		syntheticBridgeMountedKey === syntheticBridgeKey &&
		canvasElement
	) {
		maybeStartSyntheticBridgeRenderValidation();
	}

	$: if ((!syntheticBridgeEnabled || !model) && syntheticBridge) {
		disposeSyntheticBridge();
		if (onDemandPrototypeEnabled) {
			resetSyntheticBridgeDiagnostics();
		}
	}

	const setParityStatus = (
		runId: number,
		state: "running" | "success" | "error" | "timeout",
		phase: ParityCollectionPhase,
		message?: string,
	) => {
		const now = Date.now();
		const win = getParityWindow();
		const startedAt =
			win.__parityCollectionStatus__?.runId === runId
				? win.__parityCollectionStatus__.startedAt
				: now;
		win.__parityCollectionStatus__ = {
			runId,
			state,
			phase,
			startedAt,
			updatedAt: now,
			...(message ? { message } : {}),
		};
		const log = win.__parityCollectionLog__ ?? [];
		log.push({ runId, state, phase, timestamp: now, ...(message ? { message } : {}) });
		win.__parityCollectionLog__ = log;
		console.info(
			`[parity:phase] run=${runId} state=${state} phase=${phase}${message ? ` msg=${message}` : ""}`,
		);
	};

	const setParityError = (
		runId: number,
		phase: ParityCollectionPhase,
		message: string,
		state: "error" | "timeout" = "error",
	) => {
		const win = getParityWindow();
		win.__parityCollectionError__ = message;
		win.__parityIntermediatesError__ = message;
		setParityStatus(runId, state, phase, message);
	};

	async function loadAnalysis(id: string) {
		try {
			modelLoading = true;
			resetAnalysisSceneState();
			setLoading(true);
			setError(null);
			setAnalysisId(id);
			await loadAnalysisData(id);

			if (model && $analysisStore) {
				const { center, size } = getBoundsCenterAndSize(model);
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

	// Trigger model loading overlay when the model file changes; clear which model we have so we don't run compute with a stale model.
	$: if ($analysisStore && $analysisStore.metadata?.model_file) {
		const currentModelFile = $analysisStore.metadata.model_file;
		if (currentModelFile !== lastModelFile) {
			modelLoading = true;
			lastModelFile = currentModelFile;
			resetAnalysisSceneState();
		}
	}

	// Validation: expose __parityIntermediates__ after every successful compute for statistical e2e checks (grid sizes not required to match).

	// Live analysis trigger: when base analysis or model changes, recompute.
	// Only run when the loaded model is for the current analysis's model_file (avoids Ben Gurion grid on Nes Ziona after project switch).
	function scheduleLiveAnalysisRecompute(): void {
		requestAnimationFrame(() => {
			if (
				$analysisStore &&
				model &&
				mounted &&
				modelFileForLoadedModel === $analysisStore?.metadata?.model_file
			) {
				void computeLiveAnalysis();
			}
		});
	}

	function scheduleDebugOnDemandScrubRecompute(expectedScrubTriggerKey: string): void {
		if (useDebugSharedSelectedHourHost) {
			return;
		}
		legacyScrubScheduleCount += 1;
		const scheduledRunId = ++debugOnDemandScrubScheduleRunId;
		const scheduledAnalysisId = analysisId;
		const scheduledLiveComputeModeKey = liveComputeModeKey;
		const scheduledDebugOnDemandSelectionKey = debugOnDemandSelectionKey;
		const scheduledBase = $analysisStore;
		const scheduledPrepared = onDemandDebugPrepared;
		const scheduledMonthIndex = debugOnDemandSelection.monthIndex;
		const scheduledHourIndex = debugOnDemandSelection.hourIndex;
		const scheduledTimeIndex = debugOnDemandSelection.timeIndex;

		requestAnimationFrame(() => {
			if (
				scheduledRunId !== debugOnDemandScrubScheduleRunId ||
				lastDebugOnDemandScrubTriggerKey !== expectedScrubTriggerKey ||
				!browser ||
				!mounted ||
				!onDemandPrototypeEnabled ||
				debugOnDemandMode !== "f32" ||
				useDebugSharedSelectedHourHost ||
				strictExposureOnlyEnabled ||
				!$analysisStore ||
				!model ||
				!onDemandDebugPrepared ||
				!scheduledBase ||
				modelFileForLoadedModel !== $analysisStore.metadata.model_file ||
				analysisId !== scheduledAnalysisId ||
				liveComputeModeKey !== scheduledLiveComputeModeKey ||
				debugOnDemandSelectionKey !== scheduledDebugOnDemandSelectionKey ||
				$analysisStore !== scheduledBase ||
				onDemandDebugPrepared !== scheduledPrepared
			) {
				return;
			}

			void runDebugOnDemandSelectedHour({
				base: scheduledBase,
				monthIndex: scheduledMonthIndex,
				hourIndex: scheduledHourIndex,
				timeIndex: scheduledTimeIndex,
				readbackForComparison: parityMode,
			});
		});
	}

	function doesDebugOnDemandRequestStillOwnSelection(params: {
		prepared: NonNullable<typeof onDemandDebugPrepared>;
		requestId: number;
		monthIndex: number;
		timeIndex: number;
	}): boolean {
		return (
			onDemandDebugPrepared === params.prepared &&
			!params.prepared.signal.aborted &&
			onDemandScrubState.activeRequestId === null &&
			onDemandScrubState.completedRequestId === params.requestId &&
			onDemandScrubState.completedMonthIndex === params.monthIndex &&
			onDemandScrubState.completedTimeIndex === params.timeIndex &&
			onDemandScrubState.selectedMonthIndex === params.monthIndex &&
			onDemandScrubState.selectedTimeIndex === params.timeIndex
		);
	}

	type PreparedWebgpuDebugInputs = {
		analysisParams: {
			analysisId: string;
			baseMetadata: AnalysisMetadata;
			workerResult: { serializedBvh: SerializedBvhForGpu; gridPoints?: Float32Array };
			epwContent: string;
			gridResolution: number;
			zHeight: number;
			numHours: number;
			startMonth: number;
			numMonths: number;
		};
		pipeline: UTCIComputePipeline;
		computeManager: ComputeManager;
		numPoints: number;
		numHours: number;
		numMonths: number;
		gridResolution: number;
		deviceSource: "renderer" | "standalone";
		coldStartStartedAt: number;
	};

	function getPreferredDebugComputeDevice(): GPUDevice | undefined {
		if (
			debugOnDemandMode === "f32" &&
			resolvedUtciSurfaceBackend === "gpuNative" &&
			requestLargeWebgpuLimits
		) {
			return rendererDeviceForDebug;
		}
		return undefined;
	}

	async function prepareWebgpuDebugInputsForCurrentSelection(params: {
		base: Analysis;
		signal: AbortSignal;
		runId: number;
		zHeight: number;
	}): Promise<PreparedWebgpuDebugInputs> {
		if (!model) {
			throw new Error("Model is not loaded for the current selection.");
		}

		const { base, signal, runId, zHeight } = params;
		const coldStartStartedAt = performance.now();
		resetOnDemandPrototypeColdStartTimings();
		const projectId = resolveProjectId(analysisId) ?? "Ben-Gurion";
		const epwUrl = getEpwUrlForProject(projectId);
		const baseGrid = base.metadata.grid_size || 2;
		const numHours = base.data.numHours ?? base.metadata.hours.length ?? 24;
		const numMonths = debugOnDemandMode === "f32" ? 12 : parityMode ? 1 : 12;
		const startMonth = numMonths > 1 ? 1 : 8;

		setParityStatus(runId, "running", "preflight");
		const GRID_FALLBACKS = [2, 4, 6, 8];
		const startIdx = GRID_FALLBACKS.findIndex((resolution) => resolution >= baseGrid);
		const resolutionsToTry =
			startIdx >= 0 ? GRID_FALLBACKS.slice(startIdx) : [Math.max(baseGrid, 8)];

		let meshes: Awaited<ReturnType<typeof prepareMeshPayloadForWorkerAsync>>["meshes"] | null =
			null;
		let totalTriangles = 0;
		let preflight: Awaited<
			ReturnType<typeof prepareMeshPayloadForWorkerAsync>
		>["preflight"] | null = null;
		let effectiveGridResolution = baseGrid;
		let lastErr: unknown = null;
		const payloadPrepareStartedAt = performance.now();

		for (const tryRes of resolutionsToTry) {
			try {
				const result = await prepareMeshPayloadForWorkerAsync(model, {
					signal,
					gridResolution: tryRes,
					numHours,
					numMonths,
					hasWorkerSupport: typeof Worker !== "undefined",
				});
				meshes = result.meshes;
				totalTriangles = result.totalTriangles;
				preflight = result.preflight;
				effectiveGridResolution = tryRes;
				updateOnDemandPrototypeColdStartTiming(
					"payloadPrepareMs",
					performance.now() - payloadPrepareStartedAt,
				);
				break;
			} catch (error) {
				lastErr = error;
				const message = error instanceof Error ? error.message : String(error);
				if (
					message.includes("exceeds budget") &&
					tryRes < resolutionsToTry[resolutionsToTry.length - 1]
				) {
					continue;
				}
				throw error;
			}
		}

		if (!meshes || !preflight) {
			throw lastErr ?? new Error("Preflight failed");
		}

		emitComputeTelemetry("live.preflight.done", {
			data: {
				totalTriangles,
				estimatedGridPoints: preflight.estimatedGridPoints,
				estimatedBytes: preflight.estimatedBytes,
				effectiveGridResolution,
			},
		});
		if (effectiveGridResolution !== baseGrid) {
			console.warn(
				`[DEBUG UTCI] Memory budget: using ${effectiveGridResolution}m grid (requested ${baseGrid}m) for ~${(preflight.estimatedBytes / (1024 * 1024)).toFixed(0)} MB estimate`,
			);
		}
		(window as Window & {
			__computePreflight__?: {
				numPoints: number;
				estimatedBytes: number;
				effectiveGridResolution: number;
			};
		}).__computePreflight__ = {
			numPoints: preflight.estimatedGridPoints,
			estimatedBytes: preflight.estimatedBytes,
			effectiveGridResolution,
		};

		setParityStatus(runId, "running", "epw");
		const response = await fetch(epwUrl);
		if (!response.ok) {
			throw new Error(
				`Failed to load EPW file for project ${projectId}: ${response.status}`,
			);
		}
		const epwContent = await response.text();

		setParityStatus(runId, "running", "pipelineInit");
		lastPipeline?.dispose?.();
		lastPipeline = null;
		const preferredDevice = getPreferredDebugComputeDevice();
		const pipeline = await createWebgpuUtciPipeline({
			enableDiagnostics: parityMode,
			device: preferredDevice,
		});
		lastPipeline = pipeline;

		let workerResult:
			| { gridPoints: Float32Array; serializedBvh: SerializedBvhForGpu }
			| null = null;

		if (typeof Worker !== "undefined") {
			setParityStatus(runId, "running", "worker");
			try {
				const workerBvhStartedAt = performance.now();
				workerResult = await runMergeAndBvhInWorker({
					meshes,
					gridResolution: effectiveGridResolution,
					zHeight,
					signal,
					maxGridPoints: MAX_GRID_POINTS_GUARD,
					bvhOnly: true,
				});
				updateOnDemandPrototypeColdStartTiming(
					"workerBvhMs",
					performance.now() - workerBvhStartedAt,
				);
			} catch (workerError) {
				if (workerError instanceof DOMException && workerError.name === "AbortError") {
					throw workerError;
				}
				throw new Error(
					`Worker BVH generation failed; rectangular parity path requires workerResult.serializedBvh (triangles ${(totalTriangles / 1e6).toFixed(1)}M): ${workerError instanceof Error ? workerError.message : String(workerError)}`,
				);
			}
		}

		if (!workerResult) {
			throw new Error(
				"Worker did not produce BVH output; rectangular parity path requires workerResult.serializedBvh.",
			);
		}

		const bounds = base.metadata.bounds as
			| { x_min: number; x_max: number; y_min: number; y_max: number; z?: number }
			| undefined;
		if (!bounds) {
			throw new Error(
				"Analysis metadata is missing bounds; rectangular parity path cannot build canonical grid.",
			);
		}

		const coordinateSystem =
			(base.metadata.coordinate_system as "xy_ground" | "xz_ground") ?? "xy_ground";
		const gridOriginOffset = getGridOriginOffset(base.metadata);
		const computeGridHeight = (bounds.z ?? zHeight) + PARITY_SAMPLE_HEIGHT_OFFSET_M;
		const sunVectorsFixture =
			numMonths === 1
				? buildSunVectorsFixtureFromMetadata({
					baseMetadata: base.metadata,
					numHours,
					numMonths,
				})
				: undefined;

		const uploadOnlyPipeline: UTCIComputePipeline = {
			uploadStaticData: (uploadParams) => pipeline.uploadStaticData(uploadParams),
			runAll: async () => {},
			readUtcisSlice: (sliceParams) => pipeline.readUtcisSlice(sliceParams),
		};
		const uploadManager = new ComputeManager(uploadOnlyPipeline, {
			numMonths,
			numHoursPerDay: numHours,
			startMonth,
		});
		const computeManager = new ComputeManager(pipeline, {
			numMonths,
			numHoursPerDay: numHours,
			startMonth,
		});

		const uploadStartedAt = performance.now();
		const initResult = await uploadManager.initFromModelAndWeather({
			serializedBvh: workerResult.serializedBvh,
			sunVectorsFixture,
			useRectangularGridFromBounds: true,
			analysisBounds: bounds,
			coordinateSystem,
			gridOriginOffset,
			epwContent,
			gridResolution: effectiveGridResolution,
			zHeight: computeGridHeight,
			signal,
		});
		updateOnDemandPrototypeColdStartTiming(
			"pipelineUploadMs",
			performance.now() - uploadStartedAt,
		);

		emitComputeTelemetry("pipeline.upload.done", {
			ms: performance.now() - uploadStartedAt,
			data: { numPoints: initResult.numPoints, numHours, numMonths },
		});

		return {
			analysisParams: {
				analysisId,
				baseMetadata: base.metadata,
				workerResult,
				epwContent,
				gridResolution: effectiveGridResolution,
				zHeight,
				numHours,
				startMonth,
				numMonths,
			},
			pipeline,
			computeManager,
			numPoints: initResult.numPoints,
			numHours,
			numMonths,
			gridResolution: effectiveGridResolution,
			deviceSource: preferredDevice ? "renderer" : "standalone",
			coldStartStartedAt,
		};
	}

	function computeTooltipProbeClientPoint(params: {
		canvas: HTMLCanvasElement | null;
		camera: PerspectiveCamera | undefined;
		mesh: Mesh | null;
	}): { clientX: number; clientY: number } | null {
		const { canvas, camera, mesh } = params;
		if (!canvas || !camera || !mesh) return null;
		const positionAttribute = mesh.geometry?.getAttribute?.("position");
		if (!positionAttribute || positionAttribute.count <= 0) return null;

		const rect = canvas.getBoundingClientRect();
		const sampleStep = Math.max(1, Math.floor(positionAttribute.count / 256));
		const localPoint = new THREE.Vector3();
		const worldPoint = new THREE.Vector3();
		const projectedPoint = new THREE.Vector3();
		let bestPoint: { clientX: number; clientY: number } | null = null;
		let bestDistanceToCenter = Number.POSITIVE_INFINITY;

		for (let index = 0; index < positionAttribute.count; index += sampleStep) {
			localPoint.fromBufferAttribute(positionAttribute, index);
			worldPoint.copy(localPoint);
			mesh.localToWorld(worldPoint);
			projectedPoint.copy(worldPoint).project(camera);
			if (
				!Number.isFinite(projectedPoint.x) ||
				!Number.isFinite(projectedPoint.y) ||
				!Number.isFinite(projectedPoint.z) ||
				projectedPoint.z < -1 ||
				projectedPoint.z > 1 ||
				projectedPoint.x < -1 ||
				projectedPoint.x > 1 ||
				projectedPoint.y < -1 ||
				projectedPoint.y > 1
			) {
				continue;
			}

			const clientX = rect.left + ((projectedPoint.x + 1) * 0.5) * rect.width;
			const clientY = rect.top + ((1 - projectedPoint.y) * 0.5) * rect.height;
			const hitElement: Element | null = canvas.ownerDocument.elementFromPoint(clientX, clientY);
			if (hitElement !== canvas) {
				continue;
			}
			const distanceToCenter = Math.abs(projectedPoint.x) + Math.abs(projectedPoint.y);
			if (distanceToCenter < bestDistanceToCenter) {
				bestDistanceToCenter = distanceToCenter;
				bestPoint = { clientX, clientY };
			}
		}

		return bestPoint;
	}

	let lastTooltipProbeClientPointKey: string | null = null;

	function publishTooltipProbePoint(params: {
		enabled: boolean;
		canvas: HTMLCanvasElement | null;
		camera: PerspectiveCamera | undefined;
		mesh: Mesh | null;
	}): void {
		if (!browser) return;

		const win = getParityWindow();
		win.__debugTooltipProbe__ = params.enabled
			? () =>
					computeTooltipProbeClientPoint({
						canvas: params.canvas,
						camera: params.camera,
						mesh: params.mesh,
					})
			: undefined;

		if (!params.enabled) {
			lastTooltipProbeClientPointKey = null;
			return;
		}

		const nextProbePoint = computeTooltipProbeClientPoint(params);
		const nextProbeKey = nextProbePoint
			? `${nextProbePoint.clientX.toFixed(2)}:${nextProbePoint.clientY.toFixed(2)}`
			: "null";
		if (nextProbeKey !== lastTooltipProbeClientPointKey) {
			updateOnDemandPrototypeDiagnostics({
				tooltipProbeClientPoint: nextProbePoint,
			});
			lastTooltipProbeClientPointKey = nextProbeKey;
		}
	}

	$: if (browser) {
		const probeEnabled = onDemandPrototypeEnabled;
		const probeCanvas = canvasElement;
		const probeCamera = cameraRef;
		const probeMesh = liveUtciMesh;
		publishTooltipProbePoint({
			enabled: probeEnabled,
			canvas: probeCanvas,
			camera: probeCamera,
			mesh: probeMesh,
		});
	}

	let lastTooltipDisableState: boolean | null = null;

	$: if (browser) {
		if (onDemandPrototypeEnabled) {
			if (lastTooltipDisableState !== debugTooltipHoverDisabled) {
				updateOnDemandPrototypeDiagnostics({
					tooltipInteraction: createEmptyTooltipInteractionDiagnostics(
						debugTooltipHoverDisabled,
					),
				});
				lastTooltipDisableState = debugTooltipHoverDisabled;
				if (debugTooltipHoverDisabled) {
					tooltipVisible = false;
					tooltipPosition = null;
				}
			}
		} else {
			lastTooltipDisableState = null;
		}
	}

	async function computeLiveAnalysis() {
		if (liveLoading) {
			rerunLiveAnalysisAfterCurrentCompute = true;
			return;
		}
		rerunLiveAnalysisAfterCurrentCompute = false;
		const base = $analysisStore;
		if (!base || !model) {
			liveAnalysis = null;
			liveError = null;
			return;
		}
		if (modelFileForLoadedModel !== base.metadata.model_file) {
			return;
		}

		const gridResolution = base.metadata.grid_size || 2;
		const zHeight = base.metadata.bounds?.z ?? 0.9;
		const strictTimeIndex = strictExposureOnlyEnabled
			? getStrictExposureOnlyTimeIndex()
			: "full";
		const liveKey = `${base.metadata.model_file}|${base.metadata.grid_size}|${analysisId}|${liveComputeModeKey}|${strictTimeIndex}`;
		if (
			liveKey === lastLiveKey &&
			(liveAnalysis ||
				(strictExposureOnlyEnabled &&
					(onDemandPrototypeStatus === "ready" || onDemandPrototypeStatus === "error")))
		) {
			return;
		}

		lastLiveKey = liveKey;
		liveLoading = true;
		liveError = null;
		const runId = ++liveRunCounter;
		const parityWin = getParityWindow();
		parityWin.__parityIntermediatesError__ = undefined;
		parityWin.__parityCollectionError__ = undefined;
		parityWin.__parityResults__ = undefined;
		parityWin.__parityIntermediates__ = undefined;
		parityWin.__normalUtciResults__ = undefined;
		parityWin.__parityMetadata__ = base.metadata;
		parityWin.__parityCollectionLog__ = [];
		setParityStatus(runId, "running", "preflight");
		emitComputeTelemetry("live.compute.start", {
			data: { gridResolution, zHeight }
		});

		liveAbortController?.abort();
		liveAbortController = new AbortController();
		const signal = liveAbortController.signal;
		if (liveComputeWatchdog) clearTimeout(liveComputeWatchdog);
		liveComputeWatchdog = setTimeout(() => {
			if (runId !== liveRunCounter) return;
			const timeoutMessage = `Live compute exceeded ${LIVE_COMPUTE_WATCHDOG_MS}ms watchdog.`;
			setParityError(runId, "done", timeoutMessage, "timeout");
			liveAbortController?.abort();
		}, LIVE_COMPUTE_WATCHDOG_MS);

		try {
			if (strictExposureOnlyEnabled) {
				const prepared = await prepareWebgpuDebugInputsForCurrentSelection({
					base,
					signal,
					runId,
					zHeight,
				});

				liveAnalysis = null;
				liveComputeProgress = null;
				comparisonStore.update((state) => ({
					...state,
					isComparing: false,
					comparisonAnalysis: null,
				}));
				await prepared.computeManager.runExposurePrecompute({
					numPoints: prepared.numPoints,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
				});
				await prepared.computeManager.runUtciForTimeIndex({
					timeIndex: getStrictExposureOnlyTimeIndex(),
					numPoints: prepared.numPoints,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
					format: "f32-utci",
				});
				const strictDiagnostics = prepared.computeManager.getOnDemandDiagnostics();
				if (compareHoursEnabled) {
					clearOnDemandMultiHourComparison();
				}
				clearOnDemandMonthHourComparison();
				const mergedDiagnostics = buildStrictOnDemandPrototypeDiagnostics({
					strictDiagnostics,
					pointCount: prepared.numPoints,
					modelId: base.metadata.model_file ?? analysisId,
					fallbackTimeIndices: [getStrictExposureOnlyTimeIndex()],
				});
				updateOnDemandPrototypeDiagnostics(mergedDiagnostics, {
					replace: true,
					computeManager: prepared.computeManager,
				});
				if (compareHoursEnabled) {
					await runOnDemandMultiHourComparison({
						prepared,
						signal,
					});
				}
				if (compareMonthHoursEnabled) {
					await runOnDemandMonthHourComparison({
						prepared,
						signal,
						comparisonKey: `${liveComputeModeKey}|compareMonthHours`,
					});
				}
				setParityStatus(runId, "success", "done");
				return;
			}

			if (debugOnDemandMode === "f32" && !compareOneHourEnabled) {
				if (useDebugSharedSelectedHourHost) {
					liveAnalysis = null;
					liveComputeProgress = null;
					setParityStatus(runId, "success", "done");
					return;
				}
				if (!onDemandDebugPrepared) {
					const prepared = await prepareWebgpuDebugInputsForCurrentSelection({
						base,
						signal,
						runId,
						zHeight,
					});
					onDemandDebugPrepared = {
						...prepared,
						base,
						signal,
						runId,
						zHeight,
						exposureReady: false,
						exposurePrecomputePromise: null,
						pendingRenderUpdate: null,
					};
				} else {
					onDemandDebugPrepared = {
						...onDemandDebugPrepared,
						base,
						signal,
						runId,
						zHeight,
					};
				}

				liveAnalysis = null;
				liveComputeProgress = null;
				lastDebugOnDemandScrubTriggerKey =
					`${analysisId}|${liveComputeModeKey}|${debugOnDemandSelectionKey}`;
				const output = await runDebugOnDemandSelectedHour({
					base,
					monthIndex: debugOnDemandSelection.monthIndex,
					hourIndex: debugOnDemandSelection.hourIndex,
					timeIndex: debugOnDemandSelection.timeIndex,
					readbackForComparison: parityMode,
				});
				if (!output) return;
				const comparisonKey =
					compareOneHourEnabled && lastLiveKey ? `${lastLiveKey}|compareOneHour` : null;
				if (comparisonKey && onDemandDebugPrepared) {
					await runOnDemandPrototypeComparison({
						pipeline: onDemandDebugPrepared.pipeline,
						numPoints: onDemandDebugPrepared.numPoints,
						numHours: onDemandDebugPrepared.numHours,
						numMonths: onDemandDebugPrepared.numMonths,
						comparisonKey,
					});
				}
				setParityStatus(runId, "success", "done");
				return;
			}

			const prepared = await prepareWebgpuDebugInputsForCurrentSelection({
				base,
				signal,
				runId,
				zHeight,
			});
			const pipeline = prepared.pipeline;

			setParityStatus(runId, "running", "runAll");
			liveComputeProgress = null;
			const result = await createLiveUtciAnalysisFromCompute(
				prepared.analysisParams,
				{
					pipeline: prepared.pipeline,
					signal,
					onProgress: (completed, total) => {
						liveComputeProgress = { current: completed, total };
					},
					onPhase: (phase) => {
						if (phase === "readback") {
							setParityStatus(runId, "running", "readback");
						}
					},
				},
			);

			liveAnalysis = result;
			liveComputeProgress = null;

			const fullDayData = result.data && "numHours" in result.data ? (result.data as import("$lib/types/analysis").FullDayData) : null;
			const comparisonKey =
				compareOneHourEnabled && lastLiveKey ? `${lastLiveKey}|compareOneHour` : null;

			if (comparisonKey) {
				await runOnDemandPrototypeComparison({
					pipeline: prepared.pipeline,
					numPoints: result.data.numPositions,
					numHours: prepared.numHours,
					numMonths: prepared.numMonths,
					comparisonKey,
				});
			}


			// Treat the live WebGPU analysis as the comparison analysis so that
			// unifiedUtciRange can provide a shared color scale across .bin and
			// live UTCI surfaces in this debug view.
			comparisonStore.update((state) => ({
				...state,
				isComparing: true,
				comparisonAnalysis: result,
			}));


			if (normalCollectMode && fullDayData && (fullDayData.utciByHour || fullDayData.utciStorage)) {
				const monthIndex = 7;
				const utciByHour: number[][] = [];
				for (let hour = 0; hour < 24; hour++) {
					const effectiveHour = getEffectiveHourIndex(result, hour, monthIndex);
					utciByHour.push(Array.from(getUTCIForHour(fullDayData, effectiveHour)));
				}
				const win = getParityWindow();
				win.__normalUtciResults__ = {
					utciByHour,
					positions: Array.from(fullDayData.positions),
					numPoints: fullDayData.numPositions,
					numHours: 24,
					monthIndex,
				};
			}

			// Expose results and intermediates for e2e validation when ?parity=1 or ?collect=normal.
			if (parityMode || normalCollectMode) {
			if (fullDayData && (fullDayData.utciByHour || fullDayData.utciStorage)) {
				const win = window as unknown as {
					__parityResults__?: unknown;
					__parityIntermediates__?: {
						solarExposure: number[];
						skyExposure: number[];
						mrt?: number[];
						shortErf?: number[];
						longErf?: number[];
						shortDmrt?: number[];
						longDmrt?: number[];
						numPoints: number;
						numHours: number;
						numMonths?: number;
					};
				};
				win.__parityResults__ = {
					utciByHour: getUtciByHourForExport(fullDayData),
					positions: Array.from(fullDayData.positions),
					computeGridPointsWorld:
						(result as unknown as { __computeGridPointsWorld?: number[] })
							.__computeGridPointsWorld ?? null,
					numPoints: fullDayData.numPositions,
					numHours: fullDayData.numHours,
				};
				if (
					pipeline.readSolarExposureFull &&
					pipeline.readSkyExposure &&
					lastPipeline === pipeline
				) {
					const numPoints = result.data.numPositions;
					const numMonths = result.metadata.num_months ?? 1;
					const numHours = 24;
					try {
						const readPromises: [
							Promise<Float32Array>,
							Promise<Float32Array>,
							Promise<Float32Array>?
						] = [
							pipeline.readSolarExposureFull({ numPoints, numHours, numMonths }),
							pipeline.readSkyExposure({ numPoints }),
						];
						if (pipeline.readMrtFull) {
							readPromises.push(pipeline.readMrtFull({ numPoints, numHours, numMonths }));
						}
						const results = await Promise.all(readPromises);
						const solarExposure = Array.from(results[0]);
						const skyExposure = normalizeSkyExposureToViewFactor(results[1]);
						const mrtArray = results[2];
						const mrtComponents =
							pipeline.readMrtComponentsFull &&
							(pipeline.supportsMrtComponentDiagnostics?.() ?? false) &&
							mrtArray !== undefined
								? await pipeline.readMrtComponentsFull({ numPoints, numHours, numMonths })
								: undefined;
						const augustStart = 7 * 24 * numPoints;
						const augustEnd = 8 * 24 * numPoints;
						const TOME_WEIGHT = 145.24881; // Matches WGSL total_tregenza_weight

						win.__parityIntermediates__ = {
							solarExposure: results[0] ? results[0].slice(parityMode ? 0 : augustStart, parityMode ? results[0].length : augustEnd) : null,
							skyExposure: results[1] ? results[1].map(v => v / TOME_WEIGHT) : null,
							mrt: results[2] ? results[2].slice(parityMode ? 0 : augustStart, parityMode ? results[2].length : augustEnd) : null,
							shortErf: mrtComponents?.shortErf ? mrtComponents.shortErf.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.shortErf.length : augustEnd) : null,
							longErf: mrtComponents?.longErf ? mrtComponents.longErf.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.longErf.length : augustEnd) : null,
							shortDmrt: mrtComponents?.shortDmrt ? mrtComponents.shortDmrt.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.shortDmrt.length : augustEnd) : null,
							longDmrt: mrtComponents?.longDmrt ? mrtComponents.longDmrt.slice(parityMode ? 0 : augustStart, parityMode ? mrtComponents.longDmrt.length : augustEnd) : null,
							numPoints,
							numHours: 24,
							numMonths: 1,
						};
						const debugWin = win as unknown as {
							__parityDebug__?: {
								sunVectorSamples: number[] | null;
								mrt?: number[];
								shortErf?: number[];
								longErf?: number[];
								shortDmrt?: number[];
								longDmrt?: number[];
								weatherSample?: Array<{
									air_temp: number;
									direct_normal: number;
									diffuse_horizontal: number;
									horiz_infrared: number;
									wind_speed: number;
									rel_humidity: number;
								}>;
							};
						};
						debugWin.__parityDebug__ = {
							sunVectorSamples: pipeline.getSunVectorSamples?.() ?? null,
							...(mrtArray !== undefined ? { mrt: Array.from(mrtArray) } : {}),
							...(mrtComponents
								? {
										shortErf: Array.from(mrtComponents.shortErf),
										longErf: Array.from(mrtComponents.longErf),
										shortDmrt: Array.from(mrtComponents.shortDmrt),
										longDmrt: Array.from(mrtComponents.longDmrt)
									}
								: {}),
							weatherSample: pipeline.getWeatherSample?.(3) ?? [],
						};
						// Validation: fail fast when readback is all zeros (exposure not computed or not visible to CPU).
						const solarAllZero = solarExposure.length > 0 && solarExposure.every((v) => v === 0);
						const skyAllZero = skyExposure.length > 0 && skyExposure.every((v) => v === 0);
						if (solarAllZero && skyAllZero) {
							setParityError(
								runId,
								"readback",
								"Exposure readback returned all zeros. Check: exposure passes ran (BVH present), buffers have COPY_SRC, queue.onSubmittedWorkDone before readback, and shader logic (sun vectors Y-up/daytime, BVH raycast). Run tests/e2e/inspect-intermediates.spec.ts for diagnostics.",
							);
						}
					} catch (intermediateErr) {
						console.warn("[validation] Failed to read back intermediates:", intermediateErr);
						// Expose error for e2e so test fails with a clear message instead of timing out on zeros
						setParityError(
							runId,
							"readback",
							intermediateErr instanceof Error
								? intermediateErr.message
								: String(intermediateErr),
						);
					}
				}
			}
			}
			setParityStatus(runId, "success", "done");
		} catch (error) {
			liveComputeProgress = null;
			if (error instanceof DOMException && error.name === "AbortError") {
				return;
			}
			console.error("[DEBUG UTCI] Failed to compute live UTCI:", error);
			liveError =
				error instanceof Error
					? error.message
					: "Failed to compute live UTCI";
			if (onDemandPrototypeEnabled) {
				updateOnDemandPrototypeDiagnostics({
					error: liveError,
					liveAnalysisConstructedForSelectedHour: strictExposureOnlyEnabled
						? false
						: undefined,
				});
			}
			(
				window as unknown as {
					__parityIntermediatesError__?: string;
					__parityCollectionError__?: string;
				}
			).__parityCollectionError__ = liveError;
			setParityError(runId, "done", liveError, "error");
			emitComputeTelemetry("live.compute.error", {
				data: {
					message: error instanceof Error ? error.message : String(error),
				},
			});
			liveAnalysis = null;
		} finally {
			if (liveComputeWatchdog) {
				clearTimeout(liveComputeWatchdog);
				liveComputeWatchdog = null;
			}
			liveLoading = false;
			emitComputeTelemetry("live.compute.finish");
			if (rerunLiveAnalysisAfterCurrentCompute) {
				rerunLiveAnalysisAfterCurrentCompute = false;
				scheduleLiveAnalysisRecompute();
			}
		}
	}

	$: if (
		$analysisStore &&
		model &&
		mounted &&
		modelFileForLoadedModel === $analysisStore?.metadata?.model_file &&
		!useDebugSharedSelectedHourHost &&
		!(
			onDemandPrototypeEnabled &&
			debugOnDemandMode === "f32" &&
			!strictExposureOnlyEnabled &&
			onDemandDebugPrepared
		)
	) {
		const scheduledLiveComputeModeKey = liveComputeModeKey;
		const scheduledDebugOnDemandSelectionKey = debugOnDemandSelectionKey;
		// Defer by one frame so the first paint after model load can run before sync triangle count and payload prep.
		requestAnimationFrame(() => {
			if (
				$analysisStore &&
				model &&
				mounted &&
				modelFileForLoadedModel === $analysisStore?.metadata?.model_file &&
				scheduledLiveComputeModeKey === liveComputeModeKey &&
				scheduledDebugOnDemandSelectionKey === debugOnDemandSelectionKey
			) {
				void computeLiveAnalysis();
			}
		});
	}

	// Tooltip: support hover on both .bin (left) and live WebGPU (right) UTCI.
	let lastTooltipUpdate = 0;
	const TOOLTIP_THROTTLE_MS = 16;

	function getDebugTooltipTarget(event: MouseEvent): {
		mesh: Mesh | null;
		analysis: Analysis | null;
		hourIndex: number;
		side: "python" | "webgpu";
	} {
		const comparing = get(comparisonStore).isComparing;
		const comparisonSelection = getDebugComparisonSelectionView();

		if (comparing && canvasElement) {
			const canvasRect = canvasElement.getBoundingClientRect();
			const relativeX = (event.clientX - canvasRect.left) / canvasRect.width;
			const curtain = get(curtainPosition);
			if (relativeX <= curtain) {
				return {
					mesh: utciMesh,
					analysis: $analysisStore,
					hourIndex: comparisonSelection.hourIndex,
					side: "python"
				};
			}
		}

		return {
			mesh: liveUtciMesh,
			analysis: liveAnalysis ?? getGpuResidentTooltipAnalysis(),
			hourIndex: liveAnalysis
				? getEffectiveHourIndex(
						liveAnalysis,
						comparisonSelection.hourIndex,
						comparisonSelection.monthIndex
					)
				: acceptedGpuResidentUtciOutput?.tooltipUtciValues
					? 0
					: comparisonSelection.hourIndex,
			side: "webgpu"
		};
	}

	function getUtciAtPoint(analysis: Analysis | null, pointIndex: number, hourIndex: number): number | null {
		if (!analysis || pointIndex < 0 || pointIndex >= analysis.data.numPositions) return null;
		const values = getUTCIForHour(analysis.data, hourIndex);
		if (!values || pointIndex >= values.length) return null;
		return values[pointIndex];
	}

	async function copyTextToClipboard(text: string): Promise<void> {
		if (navigator.clipboard?.writeText) {
			try {
				await navigator.clipboard.writeText(text);
				return;
			} catch {
				// Fall through to textarea-based copy below.
			}
		}

		const textarea = document.createElement("textarea");
		textarea.value = text;
		textarea.setAttribute("readonly", "true");
		textarea.style.position = "fixed";
		textarea.style.left = "-9999px";
		textarea.style.top = "0";
		document.body.appendChild(textarea);
		textarea.focus();
		textarea.select();
		const copied = document.execCommand("copy");
		document.body.removeChild(textarea);
		if (!copied) {
			throw new Error("Clipboard copy was rejected by the browser.");
		}
	}

	async function copyClickedPointData(event: MouseEvent | PointerEvent) {
		if (!$viewerStore.utciVisible || !canvasElement || !cameraRef) return;
		const canvasRect = canvasElement.getBoundingClientRect();
		const target = getDebugTooltipTarget(event);
		const tooltipData = getTooltipData(
			event,
			cameraRef,
			target.mesh,
			target.analysis,
			$viewerStore.metricType,
			target.hourIndex,
			canvasRect,
		);
		if (!tooltipData) return;

		const comparisonSelection = getDebugComparisonSelectionView();
		const pythonHourIndex = comparisonSelection.hourIndex;
		const webgpuHourIndex = liveAnalysis
			? getEffectiveHourIndex(
					liveAnalysis,
					comparisonSelection.hourIndex,
					comparisonSelection.monthIndex
				)
			: acceptedGpuResidentUtciOutput?.tooltipUtciValues
				? 0
				: comparisonSelection.hourIndex;
		const pythonUtci = getUtciAtPoint($analysisStore, tooltipData.positionIndex, pythonHourIndex);
		const webgpuUtci = getUtciAtPoint(
			liveAnalysis ?? getGpuResidentTooltipAnalysis(),
			tooltipData.positionIndex,
			webgpuHourIndex
		);
		const payload = {
			source: "debug",
			analysisId,
			parityMode,
			clickedSide: target.side,
			pointIndex: tooltipData.positionIndex,
			coords: tooltipData.position,
			hour: comparisonSelection.hourIndex,
			monthIndex: comparisonSelection.monthIndex,
			pythonHourIndex,
			webgpuHourIndex,
			pythonUtci,
			webgpuUtci,
			diff: pythonUtci == null || webgpuUtci == null ? null : webgpuUtci - pythonUtci
		};
		const text = JSON.stringify(payload, null, 2);
		try {
			await copyTextToClipboard(text);
			copiedPointStatus = `Copied point ${tooltipData.positionIndex}`;
			console.info("[debug] Copied point comparison:", payload);
		} catch (error) {
			copiedPointStatus = "Copy failed - see console";
			console.warn("[debug] Failed to copy point comparison; payload:", payload, error);
		}
		window.setTimeout(() => {
			copiedPointStatus = null;
		}, 2000);
	}

	function hideTooltip(): void {
		tooltipVisible = false;
		tooltipPosition = null;
	}

	function handleMouseMove(event: MouseEvent) {
		if (debugTooltipHoverDisabled) {
			hideTooltip();
			return;
		}

		const now = performance.now();
		if (shouldSuppressTooltipMotion(tooltipMotionSuppression, now)) {
			hideTooltip();
			return;
		}
		if (now - lastTooltipUpdate < TOOLTIP_THROTTLE_MS) {
			return;
		}
		lastTooltipUpdate = now;

		if (
			!$viewerStore.utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			hideTooltip();
			return;
		}

		const canvasRect = canvasElement.getBoundingClientRect();
		const target = getDebugTooltipTarget(event);

		const tooltipData = getTooltipData(
			event,
			cameraRef,
			target.mesh,
			target.analysis,
			$viewerStore.metricType,
			target.hourIndex,
			canvasRect,
			{
				onDiagnosticsSample: (measurement) => {
					if (!browser || !onDemandPrototypeEnabled) return;
					const currentDiagnostics =
						getParityWindow().__onDemandPrototypeDiagnostics__?.tooltipInteraction ??
						createEmptyTooltipInteractionDiagnostics(debugTooltipHoverDisabled);
					updateOnDemandPrototypeDiagnostics({
						tooltipInteraction: recordTooltipInteractionMeasurement(
							currentDiagnostics,
							measurement,
						),
					});
				},
			},
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

	function handleMouseLeave() {
		hideTooltip();
	}

	let hoverListenersCanvas: HTMLCanvasElement | null = null;
	let pointerListenerCanvas: HTMLCanvasElement | null = null;
	let cameraInteractionListenerCanvas: HTMLCanvasElement | null = null;

	function detachHoverListeners(): void {
		if (!hoverListenersCanvas) return;
		hoverListenersCanvas.removeEventListener("mousemove", handleMouseMove);
		hoverListenersCanvas.removeEventListener("mouseleave", handleMouseLeave);
		hoverListenersCanvas = null;
	}

	function detachPointerListener(): void {
		if (!pointerListenerCanvas) return;
		pointerListenerCanvas.removeEventListener("pointerdown", copyClickedPointData);
		pointerListenerCanvas = null;
	}

	function handleCameraInteractionPointerDown(): void {
		tooltipMotionSuppression = setTooltipMotionPointerDown(
			tooltipMotionSuppression,
			true,
			performance.now(),
		);
		cameraInteractionPointerDown = true;
		armCameraInteractionTelemetry();
		hideTooltip();
	}

	function handleCameraInteractionPointerRelease(): void {
		const hadCanvasPointerInteraction = tooltipMotionSuppression.pointerDown;
		tooltipMotionSuppression = releaseTooltipMotionPointer(
			tooltipMotionSuppression,
			performance.now(),
		);
		cameraInteractionPointerDown = false;
		if (hadCanvasPointerInteraction) {
			armCameraInteractionTelemetry();
			hideTooltip();
		}
	}

	function handleCameraInteractionWheel(): void {
		tooltipMotionSuppression = armTooltipMotionSuppression(
			tooltipMotionSuppression,
			performance.now(),
		);
		armCameraInteractionTelemetry();
		hideTooltip();
	}

	function detachCameraInteractionListeners(): void {
		if (cameraInteractionListenerCanvas) {
			cameraInteractionListenerCanvas.removeEventListener(
				"pointerdown",
				handleCameraInteractionPointerDown,
			);
			cameraInteractionListenerCanvas.removeEventListener(
				"wheel",
				handleCameraInteractionWheel,
			);
			cameraInteractionListenerCanvas = null;
		}
		if (browser) {
			window.removeEventListener(
				"pointerup",
				handleCameraInteractionPointerRelease,
			);
			window.removeEventListener(
				"pointercancel",
				handleCameraInteractionPointerRelease,
			);
		}
	}

	$: if (mounted) {
		if (hoverListenersCanvas && hoverListenersCanvas !== canvasElement) {
			detachHoverListeners();
		}
		if (
			canvasElement &&
			!debugTooltipHoverDisabled &&
			hoverListenersCanvas !== canvasElement
		) {
			canvasElement.addEventListener("mousemove", handleMouseMove, {
				passive: true,
			});
			canvasElement.addEventListener("mouseleave", handleMouseLeave, {
				passive: true,
			});
			hoverListenersCanvas = canvasElement;
		} else if (debugTooltipHoverDisabled) {
			detachHoverListeners();
		}

		if (pointerListenerCanvas && pointerListenerCanvas !== canvasElement) {
			detachPointerListener();
		}
		if (canvasElement && pointerListenerCanvas !== canvasElement) {
			canvasElement.addEventListener("pointerdown", copyClickedPointData);
			pointerListenerCanvas = canvasElement;
		}

		if (
			cameraInteractionListenerCanvas &&
			cameraInteractionListenerCanvas !== canvasElement
		) {
			detachCameraInteractionListeners();
		}
		if (canvasElement && cameraInteractionListenerCanvas !== canvasElement) {
			canvasElement.addEventListener(
				"pointerdown",
				handleCameraInteractionPointerDown,
				{ passive: true },
			);
			canvasElement.addEventListener("wheel", handleCameraInteractionWheel, {
				passive: true,
			});
			window.addEventListener("pointerup", handleCameraInteractionPointerRelease, {
				passive: true,
			});
			window.addEventListener(
				"pointercancel",
				handleCameraInteractionPointerRelease,
				{ passive: true },
			);
			cameraInteractionListenerCanvas = canvasElement;
		}
	}

	$: if (browser) {
		if (onDemandPrototypeEnabled) {
			startCameraInteractionTelemetryLoop();
		} else {
			stopCameraInteractionTelemetryLoop();
		}
	}

	onDestroy(() => {
		liveAbortController?.abort();
		liveAbortController = null;
		unsubscribeDebugSharedRouteHost();
		debugSharedRouteHost.dispose();
		disposeSyntheticBridge();
		detachHoverListeners();
		detachPointerListener();
		detachCameraInteractionListeners();
		stopCameraInteractionTelemetryLoop();
		lastPipeline?.dispose?.();
		lastPipeline = null;
	});

	$: currentProjectId = resolveProjectId(analysisId) ?? "Ben-Gurion";

	$: if (!$comparisonStore.isComparing && utciMesh) {
		utciMesh = null;
	}
</script>

<svelte:head></svelte:head>

<ViewerShell
	bind:mainViewportElement
	debugLabel="WebGPU UTCI Debug Viewer - .bin vs live compute"
	showTimeSection={$analysisStore != null &&
		$analysisStore.metadata.analysis_type === "full_day" &&
		$viewerStore.metricType === "utci"}
	showSidebarExtraSection={Boolean(liveError)}
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
			projectId={currentProjectId}
			mode="replace-analysis"
			activeAnalysisId={analysisId}
			onSelectScenarioAnalysisId={handleProjectSelection}
		/>
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

	<svelte:fragment slot="layers">
		<div class="section-header">Layers</div>
		<LayerControls placement="sidebar" />
	</svelte:fragment>

	<svelte:fragment slot="sidebarExtra">
		{#if liveError}
			<div class="section-header">Live UTCI</div>
			<div class="section-subtitle error">
				Failed to compute live UTCI: {liveError}
			</div>
		{/if}
	</svelte:fragment>

	<svelte:fragment slot="legend">
		<ColorLegend
			displayAnalysis={
				$comparisonStore.isComparing ? null : liveAnalysis
			}
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
		{#if copiedPointStatus}
			<div class="copy-status">{copiedPointStatus}</div>
		{/if}
		{#if $viewerStore.loading}
			<div class="overlay-message">Loading analysis data...</div>
		{/if}

		{#if $viewerStore.error}
			<div class="overlay-message error">
				Error: {$viewerStore.error}
			</div>
		{/if}

		{#if showFullLoadOverlay}
			<div class="model-loading-backdrop" aria-hidden="true"></div>
			<div class="model-loading-overlay" aria-live="polite">
				<div class="spinner"></div>
				<div class="loading-text">
					{modelLoading
						? "Preparing model..."
						: liveComputeProgress
							? `Computing month ${liveComputeProgress.current}/${liveComputeProgress.total}...`
							: "Computing UTCI..."}
				</div>
			</div>
		{/if}

		{#if $comparisonStore.isComparing}
			<ComparisonCurtain
				containerElement={mainViewportElement}
				comparisonScenarioName="Live WebGPU UTCI"
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
								const parityWindow = getParityWindow();
								parityWindow.__parityModel__ = model;
								parityWindow.__parityThree__ = THREE;
								modelFileForLoadedModel = $analysisStore?.metadata?.model_file ?? null;
								modelLoading = false;
								if (model) {
									requestAnimationFrame(() => {
										if (!model) return;
										const { bounds, center, size } = getBoundsCenterAndSize(model);
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
									});
								}
							}}
							on:layersDiscovered={(e) => {
								setDiscoveredLayers(e.detail);
							}}
						/>
					{/key}

					{#if model}
						<GridHelper {model} visible={gridVisible} />
						{#if syntheticBridge}
							{#key syntheticBridge.group.uuid}
								<T is={THREE.Group} oncreate={handleSyntheticBridgeMount} />
							{/key}
						{/if}
						{#if $comparisonStore.isComparing}
							<UTCIPointCloud
								analysis={$analysisStore}
								{model}
								bind:utciSurface={utciMesh}
								utciSurfaceBackend={resolvedUtciSurfaceBackend}
							/>
						{/if}

						{#if liveAnalysis || acceptedGpuResidentUtciOutput || debugSharedBaseSceneAnalysis || debugSharedBasePendingGpuResidentOutput}
							<UTCIPointCloud
								analysis={useDebugSharedSelectedHourHost
									? debugSharedBaseSceneAnalysis
									: liveAnalysis ?? $analysisStore}
								{model}
								bind:utciSurface={liveUtciMesh}
								utciSurfaceBackend={resolvedUtciSurfaceBackend}
								acceptedGpuResidentOutput={useDebugSharedSelectedHourHost
									? debugSharedBasePendingGpuResidentOutput
									: acceptedGpuResidentUtciOutput}
								selectedHourRenderContext={useDebugSharedSelectedHourHost
									? debugSharedBaseRenderContext
									: null}
								liveSelectedHourSurfaceIdentity={useDebugSharedSelectedHourHost
									? debugSharedBaseSurfaceIdentity
									: null}
								pendingRenderUpdateStartedAt={useDebugSharedSelectedHourHost
									? debugSharedBasePendingRenderUpdateStartedAt
									: onDemandDebugPrepared?.pendingRenderUpdate?.startedAt}
								onUtciSurfaceDiagnostics={useDebugSharedSelectedHourHost
									? handleDebugSharedBaseSurfaceDiagnostics
									: handleLiveUtciSurfaceDiagnostics}
							/>
						{/if}

						{#if $comparisonStore.isComparing && utciMesh && liveUtciMesh && cameraRef}
							<DebugUtciScissor
								baseCamera={cameraRef}
								binUtciMesh={utciMesh}
								liveUtciMesh={liveUtciMesh}
							/>
						{/if}
					{/if}
				{/if}
			</Scene>
		{/key}
	</svelte:fragment>
</ViewerShell>


<style>
	.copy-status {
		position: absolute;
		left: 50%;
		bottom: 20px;
		transform: translateX(-50%);
		z-index: var(--z-tooltip);
		padding: 6px 10px;
		border: 1px solid var(--color-border-subtle);
		border-radius: var(--radius-panel);
		background: var(--color-bg-panel);
		color: var(--color-text-secondary);
		box-shadow: var(--shadow-tooltip);
		font-size: 12px;
		font-weight: 600;
		pointer-events: none;
	}

</style>
