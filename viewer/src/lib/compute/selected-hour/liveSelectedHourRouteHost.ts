import type { Group } from 'three';
import type { Analysis } from '$lib/types/analysis';
import {
	createLiveSelectedHourController,
	type LiveSelectedHourAcceptedVisibleSurface,
	type LiveSelectedHourController,
	type LiveSelectedHourGpuResidentRelease,
	type LiveSelectedHourControllerRequest,
	type LiveSelectedHourControllerState,
	type LiveSelectedHourControllerSurfaceDiagnostics,
	type LiveSelectedHourSessionConfig
} from '$lib/compute/selected-hour/liveSelectedHourController';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import { getEpwUrlForAnalysis } from '$lib/compute/weather/projectWeather';
import {
	createLiveSelectedHourPublishedRenderContext,
	type LiveSelectedHourPublishedRenderContext,
	type LiveSelectedHourRangeOverride
} from '$lib/compute/selected-hour/liveSelectedHourRenderContext';
import type { UtciRenderMode } from '$lib/utciRenderMode';
import { getUtciRangeForDisplay } from '$lib/utils/effectiveHourIndex';
import { resolveProjectId } from '$lib/utils/analysisPaths';

export type LiveSelectedHourRouteInputs = {
	enabled: boolean;
	analysisId: string | null;
	baseAnalysis: Analysis | null;
	baseModel: Group | null;
	selection: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		selectionKey: string;
	};
	gridResolutionMeters?: number;
	colorMode: 'normalized' | 'discrete';
	utciRenderMode: UtciRenderMode;
	rendererBackend: 'unknown' | 'webgpu';
	rendererDevice?: GPUDevice;
	utciSurfaceBackend: 'dataTexture' | 'gpuNative';
	comparison: {
		active: boolean;
		analysisId: string | null;
		sourceAnalysis: Analysis | null;
		model: Group | null;
		rendererDevice?: GPUDevice;
	};
};

export type LiveSelectedHourRouteState = {
	base: LiveSelectedHourControllerState;
	comparison: LiveSelectedHourControllerState;
	baseDisplayAnalysis: Analysis | null;
	comparisonDisplayAnalysis: Analysis | null | undefined;
	// Primary is base-first when both slots have visible selected-hour surfaces.
	primaryAcceptedVisibleSurface: LiveSelectedHourAcceptedVisibleSurface | null;
	baseAcceptedVisibleSurface: LiveSelectedHourAcceptedVisibleSurface | null;
	comparisonAcceptedVisibleSurface: LiveSelectedHourAcceptedVisibleSurface | null;
	acceptedRequestId: number | undefined;
	acceptedSelectionKey: string | undefined;
	acceptedVisibleAtMs: number | undefined;
	baseHasVisibleLiveSurface: boolean;
	comparisonHasVisibleLiveSurface: boolean;
	baseSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	comparisonSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null | undefined;
	baseSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	comparisonSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	baseRenderContext: LiveSelectedHourPublishedRenderContext | null;
	comparisonRenderContext: LiveSelectedHourPublishedRenderContext | null | undefined;
	baseReady: boolean;
	comparisonReady: boolean;
	comparisonSourceAnalysisId: string | null;
	liveUnifiedRange: LiveSelectedHourRangeOverride | null;
};

export type LiveSelectedHourRouteHost = {
	setRouteInputs(inputs: LiveSelectedHourRouteInputs): void;
	releaseBaseAcceptedGpuResidentOutput(release: LiveSelectedHourGpuResidentRelease): void;
	releaseComparisonAcceptedGpuResidentOutput(
		release: LiveSelectedHourGpuResidentRelease
	): void;
	handleBaseSurfaceDiagnostics(diagnostics: LiveSelectedHourControllerSurfaceDiagnostics): void;
	handleComparisonSurfaceDiagnostics(
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	): void;
	getState(): LiveSelectedHourRouteState;
	subscribe(listener: (state: LiveSelectedHourRouteState) => void): () => void;
	flush(): Promise<void>;
	dispose(): void;
};

export type LiveSelectedHourRouteHostDeps = {
	createController?: () => LiveSelectedHourController;
	prepareSession?: NonNullable<
		Parameters<typeof createLiveSelectedHourController>[0]
	>['prepareSession'];
	resolveEpwUrl?: (params: {
		analysisId?: string | null;
		fallbackProjectId?: string | null;
	}) => string;
	dataBasePath?: string;
};

type ControllerSlot = 'base' | 'comparison';

type ComparisonSourceContext = {
	analysisId: string;
	analysis: Analysis;
	model: Group;
	rendererDevice?: GPUDevice;
};

type SelectionPlan = {
	preferGpuResident: boolean;
	preferredDevice?: GPUDevice;
	controllerIdentity: string;
	selectionTriggerKey: string;
};

type PublishedSurfaceSnapshot = {
	controllerIdentity: string;
	selectionTriggerKey: string;
	acceptedVisibleSurface: LiveSelectedHourAcceptedVisibleSurface | null;
	analysis: Analysis;
	surfaceIdentity: LiveSelectedHourSurfaceIdentity;
	renderContext: LiveSelectedHourPublishedRenderContext;
};

const objectIdentityIds = new WeakMap<object, number>();
let nextObjectIdentityId = 1;

const IDLE_CONTROLLER_STATE: LiveSelectedHourControllerState = {
	analysis: null,
	acceptedGpuResidentOutput: null,
	surfaceIdentity: null,
	acceptedVisibleSurface: null,
	acceptedRequestId: undefined,
	acceptedSelectionKey: undefined,
	acceptedVisibleAtMs: undefined,
	visibleSelectedHourReadbackCount: undefined,
	readbackInstrumentation: 'not-instrumented',
	selectedHourReadbackReasons: [],
	selectedHourReadbackReasonCounts: {},
	loading: false,
	error: null,
	renderTransport: 'idle',
	sameDeviceForComputeAndRender: null,
	pendingRenderUpdateStartedAt: undefined,
	renderSurfaceDiagnostics: {},
	ready: false,
	renderReady: false,
	awaitingGpuSurface: false
};

function cloneState(state: LiveSelectedHourRouteState): LiveSelectedHourRouteState {
	return {
		...state,
		base: {
			...state.base,
			acceptedVisibleSurface: state.base.acceptedVisibleSurface
				? { ...state.base.acceptedVisibleSurface }
				: null,
			selectedHourReadbackReasons: [...state.base.selectedHourReadbackReasons],
			selectedHourReadbackReasonCounts: { ...state.base.selectedHourReadbackReasonCounts },
			surfaceIdentity: state.base.surfaceIdentity ? { ...state.base.surfaceIdentity } : null,
			runtimeDiagnostics: state.base.runtimeDiagnostics
				? {
						timings: { ...state.base.runtimeDiagnostics.timings },
						trackedGpuAllocationBytes: {
							...state.base.runtimeDiagnostics.trackedGpuAllocationBytes
						}
					}
				: state.base.runtimeDiagnostics,
			renderSurfaceDiagnostics: { ...state.base.renderSurfaceDiagnostics }
		},
		comparison: {
			...state.comparison,
			acceptedVisibleSurface: state.comparison.acceptedVisibleSurface
				? { ...state.comparison.acceptedVisibleSurface }
				: null,
			selectedHourReadbackReasons: [...state.comparison.selectedHourReadbackReasons],
			selectedHourReadbackReasonCounts: {
				...state.comparison.selectedHourReadbackReasonCounts
			},
			surfaceIdentity: state.comparison.surfaceIdentity
				? { ...state.comparison.surfaceIdentity }
				: null,
			runtimeDiagnostics: state.comparison.runtimeDiagnostics
				? {
						timings: { ...state.comparison.runtimeDiagnostics.timings },
						trackedGpuAllocationBytes: {
							...state.comparison.runtimeDiagnostics.trackedGpuAllocationBytes
						}
					}
				: state.comparison.runtimeDiagnostics,
			renderSurfaceDiagnostics: { ...state.comparison.renderSurfaceDiagnostics }
		},
		primaryAcceptedVisibleSurface: state.primaryAcceptedVisibleSurface
			? { ...state.primaryAcceptedVisibleSurface }
			: null,
		baseAcceptedVisibleSurface: state.baseAcceptedVisibleSurface
			? { ...state.baseAcceptedVisibleSurface }
			: null,
		comparisonAcceptedVisibleSurface: state.comparisonAcceptedVisibleSurface
			? { ...state.comparisonAcceptedVisibleSurface }
			: null,
		baseSurfaceIdentity: state.baseSurfaceIdentity ? { ...state.baseSurfaceIdentity } : null,
		baseSceneSurfaceIdentity: state.baseSceneSurfaceIdentity
			? { ...state.baseSceneSurfaceIdentity }
			: null,
		comparisonSurfaceIdentity: state.comparisonSurfaceIdentity
			? { ...state.comparisonSurfaceIdentity }
			: null,
		comparisonSceneSurfaceIdentity: state.comparisonSceneSurfaceIdentity
			? { ...state.comparisonSceneSurfaceIdentity }
			: state.comparisonSceneSurfaceIdentity,
		baseRenderContext: state.baseRenderContext ? { ...state.baseRenderContext } : null,
		comparisonRenderContext: state.comparisonRenderContext
			? { ...state.comparisonRenderContext }
			: state.comparisonRenderContext,
		liveUnifiedRange: state.liveUnifiedRange ? { ...state.liveUnifiedRange } : null
	};
}

function buildControllerIdentity(params: {
	analysisId: string;
	analysis: Analysis;
	model: Group;
	preferredDevice?: GPUDevice;
	gridResolutionMeters?: number;
}): string {
	let analysisIdentity = objectIdentityIds.get(params.analysis);
	if (analysisIdentity === undefined) {
		analysisIdentity = nextObjectIdentityId++;
		objectIdentityIds.set(params.analysis, analysisIdentity);
	}

	let modelIdentity = objectIdentityIds.get(params.model);
	if (modelIdentity === undefined) {
		modelIdentity = nextObjectIdentityId++;
		objectIdentityIds.set(params.model, modelIdentity);
	}

	let deviceIdentity: number | null = null;
	if (params.preferredDevice) {
		deviceIdentity = objectIdentityIds.get(params.preferredDevice) ?? null;
		if (deviceIdentity == null) {
			deviceIdentity = nextObjectIdentityId++;
			objectIdentityIds.set(params.preferredDevice, deviceIdentity);
		}
	}

	return [
		params.analysisId,
		params.analysis.metadata.model_file,
		`analysis:${analysisIdentity}`,
		`model:${modelIdentity}`,
		`device:${deviceIdentity ?? 'none'}`,
		`grid:${params.gridResolutionMeters ?? 'base'}`,
		params.preferredDevice ? 'renderer' : 'standalone'
	].join('|');
}

function buildSelectionTriggerKey(params: {
	controllerIdentity: string;
	selectionKey: string;
	colorMode: 'normalized' | 'discrete';
	preferGpuResident: boolean;
	gridResolutionMeters?: number;
}): string {
	return [
		params.controllerIdentity,
		params.selectionKey,
		params.colorMode,
		`grid:${params.gridResolutionMeters ?? 'base'}`,
		params.preferGpuResident ? 'gpu' : 'cpu'
	].join('|');
}

function isFullDayAnalysis(analysis: Analysis | null): analysis is Analysis {
	return analysis?.metadata.analysis_type === 'full_day';
}

function resolveComparisonSourceContext(
	inputs: LiveSelectedHourRouteInputs | null
): ComparisonSourceContext | null {
	if (
		!inputs?.comparison.active ||
		!inputs.comparison.analysisId ||
		inputs.comparison.model == null ||
		!isFullDayAnalysis(inputs.comparison.sourceAnalysis)
	) {
		return null;
	}

	const sourceAnalysisId =
		inputs.comparison.sourceAnalysis.metadata.source_analysis_id ??
		inputs.comparison.analysisId;
	if (sourceAnalysisId !== inputs.comparison.analysisId) {
		return null;
	}

	return {
		analysisId: inputs.comparison.analysisId,
		analysis: inputs.comparison.sourceAnalysis,
		model: inputs.comparison.model,
		rendererDevice: inputs.comparison.rendererDevice
	};
}

function resolvePreferGpuResident(inputs: LiveSelectedHourRouteInputs | null): boolean {
	return (
		inputs?.rendererBackend === 'webgpu' && inputs.utciSurfaceBackend === 'gpuNative'
	);
}

function shouldDeferLiveStartup(inputs: LiveSelectedHourRouteInputs | null): boolean {
	if (!inputs?.enabled) return false;
	if (inputs.utciRenderMode === 'data') {
		return false;
	}
	return (
		inputs.rendererBackend !== 'webgpu' ||
		inputs.rendererDevice == null ||
		inputs.utciSurfaceBackend !== 'gpuNative'
	);
}

function buildSelectionPlan(params: {
	analysisId: string;
	analysis: Analysis;
	model: Group;
	rendererBackend: LiveSelectedHourRouteInputs['rendererBackend'];
	rendererDevice?: GPUDevice;
	utciSurfaceBackend: LiveSelectedHourRouteInputs['utciSurfaceBackend'];
	selectionKey: string;
	colorMode: LiveSelectedHourRouteInputs['colorMode'];
	gridResolutionMeters?: number;
}): SelectionPlan {
	const preferGpuResident =
		params.rendererBackend === 'webgpu' && params.utciSurfaceBackend === 'gpuNative';
	const preferredDevice = preferGpuResident ? params.rendererDevice : undefined;
	const controllerIdentity = buildControllerIdentity({
		analysisId: params.analysisId,
		analysis: params.analysis,
		model: params.model,
		preferredDevice,
		gridResolutionMeters: params.gridResolutionMeters
	});
	return {
		preferGpuResident,
		preferredDevice,
		controllerIdentity,
		selectionTriggerKey: buildSelectionTriggerKey({
			controllerIdentity,
			selectionKey: params.selectionKey,
			colorMode: params.colorMode,
			gridResolutionMeters: params.gridResolutionMeters,
			preferGpuResident
		})
	};
}

function resolveDisplayRange(params: {
	analysis: Analysis;
	colorMode: 'normalized' | 'discrete';
	hourIndex: number;
	monthIndex: number;
}): { utciMin: number; utciMax: number } {
	return getUtciRangeForDisplay(
		params.analysis.metadata,
		params.colorMode,
		params.hourIndex,
		params.monthIndex
	);
}

function syncRequestedRenderContextAnalysis(params: {
	renderContext: LiveSelectedHourPublishedRenderContext;
	analysis: Analysis;
}): LiveSelectedHourPublishedRenderContext {
	return createLiveSelectedHourPublishedRenderContext({
		...params.renderContext,
		analysis: params.analysis
	});
}

function attachControllerIdentityToSurfaceIdentity(params: {
	surfaceIdentity: LiveSelectedHourSurfaceIdentity;
	controllerIdentity: string;
	controllerInstanceId: number;
}): LiveSelectedHourSurfaceIdentity {
	return {
		...params.surfaceIdentity,
		controllerIdentity: params.controllerIdentity,
		controllerInstanceId: params.controllerInstanceId
	};
}

export function createLiveSelectedHourRouteHost(
	deps: LiveSelectedHourRouteHostDeps = {}
): LiveSelectedHourRouteHost {
	const createController =
		deps.createController ??
		(() =>
			createLiveSelectedHourController({
				prepareSession: deps.prepareSession
			}));
	const resolveEpwUrl =
		deps.resolveEpwUrl ??
		((params: { analysisId?: string | null; fallbackProjectId?: string | null }) =>
			getEpwUrlForAnalysis({
				analysisId: params.analysisId,
				fallbackProjectId: params.fallbackProjectId,
				dataBasePath: deps.dataBasePath ?? ''
			}));

	let disposed = false;
	let currentInputs: LiveSelectedHourRouteInputs | null = null;
	let pendingWork = Promise.resolve();
	let reconcileQueued = false;
	let baseController = createController();
	let comparisonController = createController();
	let baseControllerState = baseController.getState();
	let comparisonControllerState = comparisonController.getState();
	let baseControllerIdentity: string | null = null;
	let comparisonControllerIdentity: string | null = null;
	let baseSelectionTriggerKey: string | null = null;
	let comparisonSelectionTriggerKey: string | null = null;
	let baseRequestedRenderContext: LiveSelectedHourPublishedRenderContext | null = null;
	let comparisonRequestedRenderContext: LiveSelectedHourPublishedRenderContext | null = null;
	let baseControllerGeneration = 0;
	let comparisonControllerGeneration = 0;
	let nextControllerInstanceId = 1;
	let baseControllerInstanceId = nextControllerInstanceId++;
	let comparisonControllerInstanceId = nextControllerInstanceId++;
	let basePublishedSurface: PublishedSurfaceSnapshot | null = null;
	let comparisonPublishedSurface: PublishedSurfaceSnapshot | null = null;
	let unsubscribeBaseController: () => void = () => undefined;
	let unsubscribeComparisonController: () => void = () => undefined;
	const listeners = new Set<(state: LiveSelectedHourRouteState) => void>();
	let state: LiveSelectedHourRouteState;

	function emit(): void {
		const snapshot = cloneState(state);
		for (const listener of listeners) {
			listener(snapshot);
		}
	}

	function createPublishedSurfaceSnapshot(params: {
		controllerIdentity: string;
		controllerInstanceId: number;
		selectionTriggerKey: string;
		acceptedVisibleSurface: LiveSelectedHourAcceptedVisibleSurface | null;
		analysis: Analysis;
		surfaceIdentity: LiveSelectedHourSurfaceIdentity;
		renderContext: LiveSelectedHourPublishedRenderContext;
	}): PublishedSurfaceSnapshot {
		return {
			controllerIdentity: params.controllerIdentity,
			selectionTriggerKey: params.selectionTriggerKey,
			acceptedVisibleSurface: params.acceptedVisibleSurface
				? { ...params.acceptedVisibleSurface }
				: null,
			analysis: params.analysis,
			surfaceIdentity: attachControllerIdentityToSurfaceIdentity({
				surfaceIdentity: params.surfaceIdentity,
				controllerIdentity: params.controllerIdentity,
				controllerInstanceId: params.controllerInstanceId
			}),
			renderContext: params.renderContext
		};
	}

	function publishState(): void {
		if (
			baseControllerIdentity &&
			baseSelectionTriggerKey &&
			baseRequestedRenderContext != null &&
			baseControllerState.renderReady &&
			!baseControllerState.loading &&
			baseControllerState.error == null &&
			baseControllerState.surfaceIdentity != null
		) {
			basePublishedSurface = createPublishedSurfaceSnapshot({
				controllerIdentity: baseControllerIdentity,
				controllerInstanceId: baseControllerInstanceId,
				selectionTriggerKey: baseSelectionTriggerKey,
				acceptedVisibleSurface: baseControllerState.acceptedVisibleSurface,
				analysis: baseControllerState.analysis ?? baseRequestedRenderContext.analysis,
				surfaceIdentity: baseControllerState.surfaceIdentity,
				renderContext: baseRequestedRenderContext
			});
		}

		if (
			comparisonControllerIdentity &&
			comparisonSelectionTriggerKey &&
			comparisonRequestedRenderContext != null &&
			comparisonControllerState.renderReady &&
			!comparisonControllerState.loading &&
			comparisonControllerState.error == null &&
			comparisonControllerState.surfaceIdentity != null
		) {
			comparisonPublishedSurface = createPublishedSurfaceSnapshot({
				controllerIdentity: comparisonControllerIdentity,
				controllerInstanceId: comparisonControllerInstanceId,
				selectionTriggerKey: comparisonSelectionTriggerKey,
				acceptedVisibleSurface: comparisonControllerState.acceptedVisibleSurface,
				analysis:
					comparisonControllerState.analysis ?? comparisonRequestedRenderContext.analysis,
				surfaceIdentity: comparisonControllerState.surfaceIdentity,
				renderContext: comparisonRequestedRenderContext
			});
		}

		const startupDeferred = shouldDeferLiveStartup(currentInputs);
		const comparisonSourceContext = resolveComparisonSourceContext(currentInputs);
		const comparisonActive = currentInputs?.comparison.active ?? false;
		const comparisonEligible = comparisonSourceContext != null;
		const liveEnabled = (currentInputs?.enabled ?? false) && !startupDeferred;
		let baseControllerIsCurrent = false;
		let basePublishedSurfaceIsCurrent = false;
		let baseHasVisiblePublishedSurface = false;
		let baseVisibleSurface: PublishedSurfaceSnapshot | null = null;
		if (
			liveEnabled &&
			currentInputs?.analysisId &&
			isFullDayAnalysis(currentInputs.baseAnalysis) &&
			currentInputs.baseModel
		) {
			const selectionPlan = buildSelectionPlan({
				analysisId: currentInputs.analysisId,
				analysis: currentInputs.baseAnalysis,
				model: currentInputs.baseModel,
				rendererBackend: currentInputs.rendererBackend,
				rendererDevice: currentInputs.rendererDevice,
				utciSurfaceBackend: currentInputs.utciSurfaceBackend,
				selectionKey: currentInputs.selection.selectionKey,
				colorMode: currentInputs.colorMode,
				gridResolutionMeters: currentInputs.gridResolutionMeters
			});
			baseControllerIsCurrent = baseControllerIdentity === selectionPlan.controllerIdentity;
			basePublishedSurfaceIsCurrent =
				baseControllerIsCurrent &&
				!baseControllerState.loading &&
				baseControllerState.error == null &&
				basePublishedSurface?.controllerIdentity === selectionPlan.controllerIdentity &&
				basePublishedSurface.selectionTriggerKey === selectionPlan.selectionTriggerKey;
			baseHasVisiblePublishedSurface =
				basePublishedSurface?.controllerIdentity === selectionPlan.controllerIdentity &&
				baseControllerIsCurrent &&
				baseControllerState.error == null;
			baseVisibleSurface = baseHasVisiblePublishedSurface ? basePublishedSurface : null;
		}

		let comparisonControllerIsCurrent = false;
		let comparisonPublishedSurfaceIsCurrent = false;
		let comparisonHasVisiblePublishedSurface = false;
		let comparisonVisibleSurface: PublishedSurfaceSnapshot | null = null;
		if (liveEnabled && comparisonEligible && currentInputs) {
			const selectionPlan = buildSelectionPlan({
				analysisId: comparisonSourceContext.analysisId,
				analysis: comparisonSourceContext.analysis,
				model: comparisonSourceContext.model,
				rendererBackend: currentInputs.rendererBackend,
				rendererDevice:
					comparisonSourceContext.rendererDevice ?? currentInputs.rendererDevice,
				utciSurfaceBackend: currentInputs.utciSurfaceBackend,
				selectionKey: currentInputs.selection.selectionKey,
				colorMode: currentInputs.colorMode,
				gridResolutionMeters: currentInputs.gridResolutionMeters
			});
			comparisonControllerIsCurrent =
				comparisonControllerIdentity === selectionPlan.controllerIdentity;
			comparisonPublishedSurfaceIsCurrent =
				comparisonControllerIsCurrent &&
				!comparisonControllerState.loading &&
				comparisonControllerState.error == null &&
				comparisonPublishedSurface?.controllerIdentity === selectionPlan.controllerIdentity &&
				comparisonPublishedSurface.selectionTriggerKey === selectionPlan.selectionTriggerKey;
			comparisonHasVisiblePublishedSurface =
				comparisonPublishedSurface?.controllerIdentity === selectionPlan.controllerIdentity &&
				comparisonControllerIsCurrent &&
				comparisonControllerState.error == null;
			comparisonVisibleSurface = comparisonHasVisiblePublishedSurface
				? comparisonPublishedSurface
				: null;
		}

		const exposedBaseState =
			liveEnabled && baseControllerIsCurrent ? baseControllerState : IDLE_CONTROLLER_STATE;
		const exposedComparisonState =
			liveEnabled && comparisonEligible && comparisonControllerIsCurrent
				? comparisonControllerState
				: IDLE_CONTROLLER_STATE;
		const baseBootstrapAnalysis =
			liveEnabled &&
			!baseVisibleSurface &&
			baseControllerIsCurrent &&
			baseControllerState.error == null &&
			baseRequestedRenderContext != null
				? baseControllerState.analysis ?? baseRequestedRenderContext.analysis
				: null;
		const comparisonBootstrapAnalysis =
			!comparisonActive
				? undefined
				: !comparisonEligible
					? null
					: liveEnabled &&
						  !comparisonVisibleSurface &&
						  comparisonControllerIsCurrent &&
						  comparisonControllerState.error == null &&
						  comparisonRequestedRenderContext != null
						? comparisonControllerState.analysis ??
							comparisonRequestedRenderContext.analysis
						: null;
		const baseDisplayAnalysis =
			!(currentInputs?.enabled ?? false)
				? currentInputs?.baseAnalysis ?? null
				: startupDeferred
					? null
				: baseVisibleSurface
					? baseVisibleSurface.analysis
					: baseBootstrapAnalysis;
		const comparisonDisplayAnalysis = !comparisonActive
			? undefined
			: !comparisonEligible
				? null
				: !(currentInputs?.enabled ?? false)
					? comparisonSourceContext.analysis
					: startupDeferred
						? null
					: comparisonVisibleSurface
						? comparisonVisibleSurface.analysis
						: comparisonBootstrapAnalysis;
		const baseSceneSurfaceIdentity =
			!liveEnabled
				? null
				: baseControllerIsCurrent &&
					  baseControllerState.error == null &&
					  baseControllerIdentity &&
					  baseControllerState.surfaceIdentity != null
					? attachControllerIdentityToSurfaceIdentity({
							surfaceIdentity: baseControllerState.surfaceIdentity,
							controllerIdentity: baseControllerIdentity,
							controllerInstanceId: baseControllerInstanceId
					  })
					: baseVisibleSurface
						? baseVisibleSurface.surfaceIdentity
						: null;
		const comparisonSceneSurfaceIdentity = !comparisonActive
			? undefined
			: !comparisonEligible
				? null
				: !liveEnabled
					? null
					: comparisonControllerIsCurrent &&
						  comparisonControllerState.error == null &&
						  comparisonControllerIdentity &&
						  comparisonControllerState.surfaceIdentity != null
						? attachControllerIdentityToSurfaceIdentity({
								surfaceIdentity: comparisonControllerState.surfaceIdentity,
								controllerIdentity: comparisonControllerIdentity,
								controllerInstanceId: comparisonControllerInstanceId
						  })
						: comparisonVisibleSurface
							? comparisonVisibleSurface.surfaceIdentity
							: null;

		let liveUnifiedRange: LiveSelectedHourRangeOverride | null = null;
		if (
			liveEnabled &&
			comparisonEligible &&
			baseVisibleSurface &&
			comparisonVisibleSurface
		) {
			const baseRange = resolveDisplayRange({
				analysis: baseVisibleSurface.analysis,
				colorMode: baseVisibleSurface.renderContext.colorMode,
				hourIndex: baseVisibleSurface.renderContext.hourIndex,
				monthIndex: baseVisibleSurface.renderContext.monthIndex
			});
			const comparisonRange = resolveDisplayRange({
				analysis: comparisonVisibleSurface.analysis,
				colorMode: comparisonVisibleSurface.renderContext.colorMode,
				hourIndex: comparisonVisibleSurface.renderContext.hourIndex,
				monthIndex: comparisonVisibleSurface.renderContext.monthIndex
			});
			liveUnifiedRange = {
				utciMin: Math.min(baseRange.utciMin, comparisonRange.utciMin),
				utciMax: Math.max(baseRange.utciMax, comparisonRange.utciMax)
			};
		}
		const currentBaseRequestedRenderContext =
			liveEnabled &&
			baseControllerIsCurrent &&
			baseControllerState.error == null &&
			baseRequestedRenderContext != null
				? baseRequestedRenderContext
				: null;
		const currentComparisonRequestedRenderContext =
			liveEnabled &&
			comparisonEligible &&
			comparisonControllerIsCurrent &&
			comparisonControllerState.error == null &&
			comparisonRequestedRenderContext != null
				? comparisonRequestedRenderContext
				: null;
		const baseRenderContext = currentBaseRequestedRenderContext
				? createLiveSelectedHourPublishedRenderContext({
						...currentBaseRequestedRenderContext,
						analysis:
							baseControllerState.analysis ?? currentBaseRequestedRenderContext.analysis,
						rangeOverride: liveUnifiedRange
				  })
				: liveEnabled && baseVisibleSurface
					? createLiveSelectedHourPublishedRenderContext({
							...baseVisibleSurface.renderContext,
							rangeOverride: liveUnifiedRange
					  })
					: null;
		const comparisonRenderContext = !comparisonActive
			? undefined
			: currentComparisonRequestedRenderContext
				? createLiveSelectedHourPublishedRenderContext({
						...currentComparisonRequestedRenderContext,
						analysis:
							comparisonControllerState.analysis ??
							currentComparisonRequestedRenderContext.analysis,
						rangeOverride: liveUnifiedRange
				  })
				: liveEnabled && comparisonEligible && comparisonVisibleSurface
					? createLiveSelectedHourPublishedRenderContext({
							...comparisonVisibleSurface.renderContext,
							rangeOverride: liveUnifiedRange
					  })
					: null;
		const baseAcceptedVisibleSurface = baseVisibleSurface?.acceptedVisibleSurface ?? null;
		const comparisonAcceptedVisibleSurface =
			comparisonVisibleSurface?.acceptedVisibleSurface ?? null;
		const primaryAcceptedVisibleSurface =
			baseAcceptedVisibleSurface ?? comparisonAcceptedVisibleSurface;

		state = {
			base: exposedBaseState,
			comparison: exposedComparisonState,
			baseDisplayAnalysis,
			comparisonDisplayAnalysis,
			primaryAcceptedVisibleSurface: liveEnabled ? primaryAcceptedVisibleSurface : null,
			baseAcceptedVisibleSurface: liveEnabled ? baseAcceptedVisibleSurface : null,
			comparisonAcceptedVisibleSurface: liveEnabled
				? comparisonAcceptedVisibleSurface
				: null,
			acceptedRequestId: liveEnabled ? primaryAcceptedVisibleSurface?.requestId : undefined,
			acceptedSelectionKey: liveEnabled
				? primaryAcceptedVisibleSurface?.selectionKey
				: undefined,
			acceptedVisibleAtMs: liveEnabled
				? primaryAcceptedVisibleSurface?.visibleAtMs
				: undefined,
			baseHasVisibleLiveSurface: liveEnabled && baseVisibleSurface != null,
			comparisonHasVisibleLiveSurface:
				liveEnabled && comparisonEligible && comparisonVisibleSurface != null,
			baseSceneSurfaceIdentity,
			comparisonSceneSurfaceIdentity,
			baseSurfaceIdentity: liveEnabled && baseVisibleSurface
				? baseVisibleSurface.surfaceIdentity
				: null,
			comparisonSurfaceIdentity:
				liveEnabled && comparisonEligible && comparisonVisibleSurface
					? comparisonVisibleSurface.surfaceIdentity
					: null,
			baseRenderContext,
			comparisonRenderContext,
			baseReady:
				liveEnabled && basePublishedSurfaceIsCurrent
					? exposedBaseState.renderReady
					: startupDeferred
						? false
						: !liveEnabled && currentInputs?.baseAnalysis != null,
			comparisonReady: !comparisonActive
				? true
				: !comparisonEligible
					? false
					: liveEnabled
						? comparisonPublishedSurfaceIsCurrent
							? exposedComparisonState.renderReady
							: false
						: startupDeferred
							? false
							: true,
			comparisonSourceAnalysisId: comparisonSourceContext?.analysisId ?? null,
			liveUnifiedRange
		};
		emit();
	}

	function bindController(slot: ControllerSlot, controller: LiveSelectedHourController): () => void {
		return controller.subscribe((nextState) => {
			if (slot === 'base') {
				baseControllerState = nextState;
			} else {
				comparisonControllerState = nextState;
			}
			publishState();
		});
	}

	unsubscribeBaseController = bindController('base', baseController);
	unsubscribeComparisonController = bindController('comparison', comparisonController);

	function replaceController(slot: ControllerSlot): void {
		if (slot === 'base') {
			unsubscribeBaseController();
			baseController.dispose();
			baseController = createController();
			baseControllerState = baseController.getState();
			baseControllerIdentity = null;
			baseSelectionTriggerKey = null;
			baseRequestedRenderContext = null;
			basePublishedSurface = null;
			baseControllerGeneration += 1;
			baseControllerInstanceId = nextControllerInstanceId++;
			unsubscribeBaseController = bindController('base', baseController);
			return;
		}

		unsubscribeComparisonController();
		comparisonController.dispose();
		comparisonController = createController();
		comparisonControllerState = comparisonController.getState();
		comparisonControllerIdentity = null;
		comparisonSelectionTriggerKey = null;
		comparisonRequestedRenderContext = null;
		comparisonPublishedSurface = null;
		comparisonControllerGeneration += 1;
		comparisonControllerInstanceId = nextControllerInstanceId++;
		unsubscribeComparisonController = bindController('comparison', comparisonController);
	}

	function queueReconcile(): void {
		if (disposed) return;
		reconcileQueued = true;
		pendingWork = pendingWork
			.catch(() => undefined)
			.then(async () => {
				while (reconcileQueued && !disposed) {
					reconcileQueued = false;
					await reconcileLatestInputs();
				}
			});
	}

	function queueTask(task: () => Promise<void>): void {
		if (disposed) return;
		pendingWork = pendingWork
			.catch(() => undefined)
			.then(async () => {
				if (disposed) return;
				await task();
			});
	}

	async function requestControllerSelection(
		controller: LiveSelectedHourController,
		request: LiveSelectedHourControllerRequest
	) {
		return controller.requestSelection(request);
	}

	function createSessionConfig(params: {
		analysisId: string;
		analysis: Analysis;
		model: Group;
		preferredDevice?: GPUDevice;
		fallbackProjectId?: string | null;
		gridResolutionMeters?: number;
	}): LiveSelectedHourSessionConfig {
		return {
			analysisId: params.analysisId,
			base: params.analysis,
			model: params.model,
			epwUrl: resolveEpwUrl({
				analysisId: params.analysisId,
				fallbackProjectId: params.fallbackProjectId
			}),
			preferredDevice: params.preferredDevice,
			gridResolution: params.gridResolutionMeters
		};
	}

	async function reconcileBase(inputs: LiveSelectedHourRouteInputs): Promise<void> {
		if (
			!inputs.enabled ||
			shouldDeferLiveStartup(inputs) ||
			!inputs.analysisId ||
			!isFullDayAnalysis(inputs.baseAnalysis) ||
			!inputs.baseModel
		) {
			if (baseControllerIdentity !== null || baseControllerState.ready || baseControllerState.loading) {
				replaceController('base');
				publishState();
			}
			return;
		}

		const selectionPlan = buildSelectionPlan({
			analysisId: inputs.analysisId,
			analysis: inputs.baseAnalysis,
			model: inputs.baseModel,
			rendererBackend: inputs.rendererBackend,
			rendererDevice: inputs.rendererDevice,
			utciSurfaceBackend: inputs.utciSurfaceBackend,
			selectionKey: inputs.selection.selectionKey,
			colorMode: inputs.colorMode,
			gridResolutionMeters: inputs.gridResolutionMeters
		});
		const controllerIdentity = selectionPlan.controllerIdentity;
		if (baseControllerIdentity !== controllerIdentity) {
			if (baseControllerIdentity !== null) {
				replaceController('base');
				publishState();
			}
			baseControllerIdentity = controllerIdentity;
			baseSelectionTriggerKey = null;
			baseRequestedRenderContext = null;
		}

		const selectionTriggerKey = selectionPlan.selectionTriggerKey;
		if (baseSelectionTriggerKey === selectionTriggerKey && baseControllerState.error == null) {
			return;
		}

		const sessionConfig = createSessionConfig({
			analysisId: inputs.analysisId,
			analysis: inputs.baseAnalysis,
			model: inputs.baseModel,
			preferredDevice: selectionPlan.preferredDevice,
			fallbackProjectId: resolveProjectId(inputs.analysisId),
			gridResolutionMeters: inputs.gridResolutionMeters
		});
		baseSelectionTriggerKey = selectionTriggerKey;
		baseRequestedRenderContext = createLiveSelectedHourPublishedRenderContext({
			analysis: inputs.baseAnalysis,
			monthIndex: inputs.selection.monthIndex,
			hourIndex: inputs.selection.hourIndex,
			timeIndex: inputs.selection.timeIndex,
			selectionKey: inputs.selection.selectionKey,
			colorMode: inputs.colorMode
		});
		const requestResult = await requestControllerSelection(baseController, {
			sessionKey: controllerIdentity,
			sessionConfig,
			monthIndex: inputs.selection.monthIndex,
			hourIndex: inputs.selection.hourIndex,
			timeIndex: inputs.selection.timeIndex,
			selectionKey: inputs.selection.selectionKey,
			colorMode: inputs.colorMode,
			preferGpuResident: selectionPlan.preferGpuResident,
			rendererDevice: selectionPlan.preferredDevice
		});
		if (
			requestResult.accepted &&
			baseControllerIdentity === controllerIdentity &&
			baseSelectionTriggerKey === selectionTriggerKey &&
			baseRequestedRenderContext != null &&
			requestResult.state.analysis != null
		) {
			baseRequestedRenderContext = syncRequestedRenderContextAnalysis({
				renderContext: baseRequestedRenderContext,
				analysis: requestResult.state.analysis
			});
			publishState();
		}
		if (
			!requestResult.accepted &&
			baseControllerIdentity === controllerIdentity &&
			baseSelectionTriggerKey === selectionTriggerKey
		) {
			baseSelectionTriggerKey = null;
			baseRequestedRenderContext = null;
			publishState();
		}
	}

	async function reconcileComparison(inputs: LiveSelectedHourRouteInputs): Promise<void> {
		const comparisonSourceContext = resolveComparisonSourceContext(inputs);
		if (
			!inputs.enabled ||
			shouldDeferLiveStartup(inputs) ||
			comparisonSourceContext == null ||
			comparisonSourceContext.model == null
		) {
			if (
				comparisonControllerIdentity !== null ||
				comparisonControllerState.ready ||
				comparisonControllerState.loading
			) {
				replaceController('comparison');
				publishState();
			}
			return;
		}

		const selectionPlan = buildSelectionPlan({
			analysisId: comparisonSourceContext.analysisId,
			analysis: comparisonSourceContext.analysis,
			model: comparisonSourceContext.model,
			rendererBackend: inputs.rendererBackend,
			rendererDevice: comparisonSourceContext.rendererDevice ?? inputs.rendererDevice,
			utciSurfaceBackend: inputs.utciSurfaceBackend,
			selectionKey: inputs.selection.selectionKey,
			colorMode: inputs.colorMode,
			gridResolutionMeters: inputs.gridResolutionMeters
		});
		const controllerIdentity = selectionPlan.controllerIdentity;
		if (comparisonControllerIdentity !== controllerIdentity) {
			if (comparisonControllerIdentity !== null) {
				replaceController('comparison');
				publishState();
			}
			comparisonControllerIdentity = controllerIdentity;
			comparisonSelectionTriggerKey = null;
			comparisonRequestedRenderContext = null;
		}

		const selectionTriggerKey = selectionPlan.selectionTriggerKey;
		if (
			comparisonSelectionTriggerKey === selectionTriggerKey &&
			comparisonControllerState.error == null
		) {
			return;
		}

		const sessionConfig = createSessionConfig({
			analysisId: comparisonSourceContext.analysisId,
			analysis: comparisonSourceContext.analysis,
			model: comparisonSourceContext.model,
			preferredDevice: selectionPlan.preferredDevice,
			fallbackProjectId: resolveProjectId(comparisonSourceContext.analysisId),
			gridResolutionMeters: inputs.gridResolutionMeters
		});
		comparisonSelectionTriggerKey = selectionTriggerKey;
		comparisonRequestedRenderContext = createLiveSelectedHourPublishedRenderContext({
			analysis: comparisonSourceContext.analysis,
			monthIndex: inputs.selection.monthIndex,
			hourIndex: inputs.selection.hourIndex,
			timeIndex: inputs.selection.timeIndex,
			selectionKey: inputs.selection.selectionKey,
			colorMode: inputs.colorMode
		});
		const requestResult = await requestControllerSelection(comparisonController, {
			sessionKey: controllerIdentity,
			sessionConfig,
			monthIndex: inputs.selection.monthIndex,
			hourIndex: inputs.selection.hourIndex,
			timeIndex: inputs.selection.timeIndex,
			selectionKey: inputs.selection.selectionKey,
			colorMode: inputs.colorMode,
			preferGpuResident: selectionPlan.preferGpuResident,
			rendererDevice: selectionPlan.preferredDevice,
			selectedHourReadbackReason: 'comparison'
		});
		if (
			requestResult.accepted &&
			comparisonControllerIdentity === controllerIdentity &&
			comparisonSelectionTriggerKey === selectionTriggerKey &&
			comparisonRequestedRenderContext != null &&
			requestResult.state.analysis != null
		) {
			comparisonRequestedRenderContext = syncRequestedRenderContextAnalysis({
				renderContext: comparisonRequestedRenderContext,
				analysis: requestResult.state.analysis
			});
			publishState();
		}
		if (
			!requestResult.accepted &&
			comparisonControllerIdentity === controllerIdentity &&
			comparisonSelectionTriggerKey === selectionTriggerKey
		) {
			comparisonSelectionTriggerKey = null;
			comparisonRequestedRenderContext = null;
			publishState();
		}
	}

	async function reconcileLatestInputs(): Promise<void> {
		const inputs = currentInputs;
		if (disposed || inputs == null) {
			return;
		}

		await reconcileBase(inputs);
		await reconcileComparison(inputs);
		publishState();
	}

	state = {
		base: baseControllerState,
		comparison: comparisonControllerState,
		baseDisplayAnalysis: null,
		comparisonDisplayAnalysis: undefined,
		primaryAcceptedVisibleSurface: null,
		baseAcceptedVisibleSurface: null,
		comparisonAcceptedVisibleSurface: null,
		acceptedRequestId: undefined,
		acceptedSelectionKey: undefined,
		acceptedVisibleAtMs: undefined,
		baseHasVisibleLiveSurface: false,
		comparisonHasVisibleLiveSurface: false,
		baseSceneSurfaceIdentity: null,
		comparisonSceneSurfaceIdentity: undefined,
		baseSurfaceIdentity: null,
		comparisonSurfaceIdentity: null,
		baseRenderContext: null,
		comparisonRenderContext: undefined,
		baseReady: false,
		comparisonReady: true,
		comparisonSourceAnalysisId: null,
		liveUnifiedRange: null
	};

	return {
		setRouteInputs(inputs) {
			if (disposed) {
				return;
			}

			currentInputs = inputs;
			publishState();
			queueReconcile();
		},

		releaseBaseAcceptedGpuResidentOutput(release) {
			if (disposed) {
				return;
			}
			if (
				release.controllerIdentity !== baseControllerIdentity ||
				release.controllerInstanceId !== baseControllerInstanceId
			) {
				return;
			}
			baseController.releaseAcceptedGpuResidentOutput(release);
		},

		releaseComparisonAcceptedGpuResidentOutput(release) {
			if (disposed) {
				return;
			}
			if (
				release.controllerIdentity !== comparisonControllerIdentity ||
				release.controllerInstanceId !== comparisonControllerInstanceId
			) {
				return;
			}
			comparisonController.releaseAcceptedGpuResidentOutput(release);
		},

		handleBaseSurfaceDiagnostics(diagnostics) {
			const controllerAtEvent = baseController;
			const generationAtEvent = baseControllerGeneration;
			queueTask(async () => {
				if (
					controllerAtEvent !== baseController ||
					generationAtEvent !== baseControllerGeneration
				) {
					return;
				}
				await controllerAtEvent.handleRenderSurfaceDiagnostics(diagnostics);
			});
		},

		handleComparisonSurfaceDiagnostics(diagnostics) {
			const controllerAtEvent = comparisonController;
			const generationAtEvent = comparisonControllerGeneration;
			queueTask(async () => {
				if (
					controllerAtEvent !== comparisonController ||
					generationAtEvent !== comparisonControllerGeneration
				) {
					return;
				}
				await controllerAtEvent.handleRenderSurfaceDiagnostics(diagnostics);
			});
		},

		getState() {
			return cloneState(state);
		},

		subscribe(listener) {
			listeners.add(listener);
			return () => {
				listeners.delete(listener);
			};
		},

		async flush() {
			await pendingWork;
		},

		dispose() {
			if (disposed) {
				return;
			}

			disposed = true;
			unsubscribeBaseController();
			unsubscribeComparisonController();
			baseController.dispose();
			comparisonController.dispose();
			listeners.clear();
		}
	};
}
