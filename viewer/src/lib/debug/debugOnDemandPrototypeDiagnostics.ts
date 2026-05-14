import {
	createEmptyOnDemandDiagnostics,
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/on-demand/onDemandDiagnostics';
import {
	buildSelectedHourRuntimeContract,
	type SelectedHourReadbackInstrumentation,
	type SelectedHourRuntimeContract
} from '$lib/diagnostics/selectedHourRuntimeContract';
import type { TooltipInteractionDiagnostics } from '$lib/services/tooltipService';
import type { CameraInteractionDiagnostics } from '$lib/services/cameraInteractionTelemetry';
import type { DebugSelectedHourEngine } from '$lib/debug/debugWebgpuUtciDiagnostics';
import type { UtciRenderMode } from '$lib/utciRenderMode';

export interface DebugOnDemandPythonSampleRecord {
	pointIndex: number;
	debugValue: number;
	referenceValue: number;
	absDiff: number;
}

export interface DebugOnDemandPythonSampleComparison {
	numCompared: number;
	maxAbsDiff: number;
	samples: DebugOnDemandPythonSampleRecord[];
}

export type DebugOnDemandPrototypeDiagnostics = Partial<OnDemandRuntimeDiagnostics> & {
	navigatorGpu: boolean;
	rendererBackend: 'webgpu' | 'unknown';
	utciRenderRequested?: UtciRenderMode;
	utciRenderResolved?: 'dataTexture' | 'gpuNative';
	utciSurfaceSource?: string;
	bridgeAttached?: boolean;
	visibleColorVariance?: number;
	debugComparisonReference?: 'python-bin';
	pythonBinComparisonActive?: boolean;
	binComparisonEnabled?: boolean;
	binComparisonValid?: boolean;
	pythonBaselineStatus?: 'available-august' | 'unavailable-non-august';
	debugComparisonMonthIndex?: number;
	pythonComparisonHourIndex?: number;
	webgpuComparisonHourIndex?: number;
	pythonSelectedHourMeanUtci?: number;
	webgpuSelectedHourMeanUtci?: number;
	pythonDerivedOneHourMs?: number;
	pythonBinSampleComparison?: DebugOnDemandPythonSampleComparison;
	appVisibleSelectedHour?: boolean;
	selectedHourReadbackCount?: number;
	visibleSelectedHourReadbackCount?: number;
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
	selectedHourRuntimeContract?: SelectedHourRuntimeContract;
};

export interface DebugOnDemandPrototypeDiagnosticsDefaults {
	navigatorGpu: boolean;
	rendererBackend: DebugOnDemandPrototypeDiagnostics['rendererBackend'];
	utciRenderRequested: DebugOnDemandPrototypeDiagnostics['utciRenderRequested'];
	utciRenderResolved: DebugOnDemandPrototypeDiagnostics['utciRenderResolved'];
	selectedHourEngine: NonNullable<DebugOnDemandPrototypeDiagnostics['selectedHourEngine']>;
	binComparisonEnabled: boolean;
	binComparisonValid: boolean;
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	tooltipInteraction: NonNullable<DebugOnDemandPrototypeDiagnostics['tooltipInteraction']>;
	cameraInteraction: NonNullable<DebugOnDemandPrototypeDiagnostics['cameraInteraction']>;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
}

export interface BuildDebugOnDemandPrototypeDiagnosticsParams {
	existing?: DebugOnDemandPrototypeDiagnostics;
	patch: Partial<DebugOnDemandPrototypeDiagnostics>;
	defaults: DebugOnDemandPrototypeDiagnosticsDefaults;
	replace?: boolean;
}

function normalizeTransport(
	value:
		| DebugOnDemandPrototypeDiagnostics['renderTransport']
		| DebugOnDemandPrototypeDiagnostics['utciSurfaceSource']
) {
	return value === 'compute-buffer-selected-hour' || value === 'cpu-uploaded-selected-hour'
		? value
		: 'none';
}

export function buildDebugOnDemandPrototypeDiagnostics(
	params: BuildDebugOnDemandPrototypeDiagnosticsParams
): DebugOnDemandPrototypeDiagnostics {
	const existing = params.replace ? undefined : params.existing;
	const patch = params.patch;
	const defaults = params.defaults;

	const nextDiagnostics: DebugOnDemandPrototypeDiagnostics = {
		...createEmptyOnDemandDiagnostics(),
		...existing,
		...patch,
		navigatorGpu: patch.navigatorGpu ?? existing?.navigatorGpu ?? defaults.navigatorGpu,
		rendererBackend: patch.rendererBackend ?? existing?.rendererBackend ?? defaults.rendererBackend,
		utciRenderRequested:
			patch.utciRenderRequested ?? existing?.utciRenderRequested ?? defaults.utciRenderRequested,
		utciRenderResolved:
			patch.utciRenderResolved ?? existing?.utciRenderResolved ?? defaults.utciRenderResolved,
		utciSurfaceSource:
			'utciSurfaceSource' in patch ? patch.utciSurfaceSource : existing?.utciSurfaceSource,
		selectedHourEngine:
			patch.selectedHourEngine ?? existing?.selectedHourEngine ?? defaults.selectedHourEngine,
		binComparisonEnabled:
			patch.binComparisonEnabled ??
			existing?.binComparisonEnabled ??
			defaults.binComparisonEnabled,
		binComparisonValid:
			patch.binComparisonValid ?? existing?.binComparisonValid ?? defaults.binComparisonValid,
		legacySelectedHourDispatchCount:
			patch.legacySelectedHourDispatchCount ?? defaults.legacySelectedHourDispatchCount,
		legacyScrubScheduleCount:
			patch.legacyScrubScheduleCount ?? defaults.legacyScrubScheduleCount,
		tooltipInteraction:
			patch.tooltipInteraction ?? existing?.tooltipInteraction ?? defaults.tooltipInteraction,
		cameraInteraction:
			patch.cameraInteraction ?? existing?.cameraInteraction ?? defaults.cameraInteraction
	};

	nextDiagnostics.selectedHourRuntimeContract =
		patch.selectedHourRuntimeContract ??
		buildSelectedHourRuntimeContract({
			route: 'debug',
			selectedHourEngine: nextDiagnostics.selectedHourEngine ?? defaults.selectedHourEngine,
			renderTransport: normalizeTransport(nextDiagnostics.renderTransport),
			utciSurfaceSource: normalizeTransport(nextDiagnostics.utciSurfaceSource),
			sameDeviceForComputeAndRender: nextDiagnostics.sameDeviceForComputeAndRender === true,
			dataTextureBuildCount: nextDiagnostics.dataTextureBuildCount,
			visibleSelectedHourReadbackCount: nextDiagnostics.visibleSelectedHourReadbackCount,
			readbackInstrumentation: defaults.readbackInstrumentation,
			legacySelectedHourDispatchCount: nextDiagnostics.legacySelectedHourDispatchCount,
			legacyScrubScheduleCount: nextDiagnostics.legacyScrubScheduleCount,
			requestId: nextDiagnostics.surfaceRequestId,
			sceneRequestId: nextDiagnostics.sceneSurfaceRequestId,
			selectionKey: nextDiagnostics.selectionKey,
			sceneSelectionKey: nextDiagnostics.sceneSelectionKey,
			readbackReasons: nextDiagnostics.selectedHourReadbackReasons,
			readbackReasonCounts: nextDiagnostics.selectedHourReadbackReasonCounts
		});

	return nextDiagnostics;
}
