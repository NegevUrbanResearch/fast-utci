import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';
import type { ExposureSchedulingMode } from '$lib/compute/gpu/exposureScheduling';
import {
	copyRenderPublicationDiagnostics,
	type SelectedHourRenderPublicationDiagnostics
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

export type OnDemandRendererBackend = 'webgpu' | 'unknown';

export type OnDemandPath = 'idle' | 'run-all-baseline' | 'exposure-only-f32' | 'error';

export type ExposureSchedulerSliceTraceSample = {
	sliceIndex: number;
	pointStart: number;
	pointCount: number;
	workgroupCount: number;
	encodeMs: number;
	submitAtMs: number;
	queueWaitMs: number;
	yieldStartedAtMs?: number;
	yieldRafCallbackAtMs?: number;
	yieldCompletedAtMs?: number;
	yieldWaitMs?: number;
	yieldPostRafTimeoutMs?: number;
};

export type ExposureSchedulerSliceWindow = {
	sliceIndex: number;
	startMs: number;
	endMs: number;
	queueWaitMs: number;
	yieldWaitMs?: number;
};

export type ExposureSchedulerBreathingTrace = {
	version: 1;
	mode: ExposureSchedulingMode;
	maxWorkgroupsPerSlice: number;
	sliceCount: number;
	submitCount: number;
	yieldCount: number;
	queueWaitTotalMs: number;
	queueWaitMaxMs: number;
	queueWaitMinMs: number;
	queueWaitAverageMs: number;
	encodeTotalMs: number;
	yieldWaitTotalMs: number;
	yieldWaitMaxMs: number;
	yieldWaitAverageMs: number;
	yieldPostRafTimeoutMaxMs: number;
	yieldPostRafTimeoutAverageMs: number;
	allSliceWindows: ExposureSchedulerSliceWindow[];
	firstSamples: ExposureSchedulerSliceTraceSample[];
	worstQueueWaitSamples: ExposureSchedulerSliceTraceSample[];
	worstYieldSamples: ExposureSchedulerSliceTraceSample[];
	lastSamples: ExposureSchedulerSliceTraceSample[];
};

export interface OnDemandTimings {
	payloadPrepareMs?: number;
	workerBvhMs?: number;
	pipelineUploadMs?: number;
	staticUploadTrace?: StaticUploadTrace;
	firstSelectedHourReadyMs?: number;
	firstSelectedHourVisibleMs?: number;
	exposurePrecomputeMs?: number;
	exposureWeatherBufferEnsureMs?: number;
	exposureCommandEncodeTotalMs?: number;
	exposureSolarEncodeMs?: number;
	exposureSkyEncodeMs?: number;
	exposureQueueWaitMs?: number;
	exposurePointCount?: number;
	exposureTotalTimeSteps?: number;
	exposureDaylightTimeSteps?: number;
	exposurePointChunks?: number;
	exposureSolarDispatchCount?: number;
	exposureSkyDispatchCount?: number;
	exposureSolarRayBudget?: number;
	exposureSkyRayBudget?: number;
	exposureSchedulerMode?: ExposureSchedulingMode;
	exposureSchedulerSliceCount?: number;
	exposurePointDispatchChunkCount?: number;
	exposureSchedulerMaxWorkgroupsPerSlice?: number;
	exposureSchedulerQueueWaitTotalMs?: number;
	exposureSchedulerQueueWaitMaxMs?: number;
	exposureSchedulerQueueWaitMinMs?: number;
	exposureSchedulerYieldCount?: number;
	exposureSchedulerSubmitCount?: number;
	exposureSchedulerBreathingTrace?: ExposureSchedulerBreathingTrace;
	oneHourDispatchMs?: number;
	shadingIndexDispatchMs?: number;
	shadingIndexQueueWaitMs?: number;
	shadingIndexOutputBytes?: number;
	shadingIndexSnapshotBytes?: number;
	renderUpdateMs?: number;
	renderSceneSyncStartDelayMs?: number;
	renderSceneSyncTotalMs?: number;
	renderLayoutBuildMs?: number;
	renderSurfaceMeshMs?: number;
	renderStorageInitWaitMs?: number;
	renderBufferCopyMs?: number;
	renderQueueDrainMs?: number;
	renderPublication?: SelectedHourRenderPublicationDiagnostics;
	debugReadbackMs?: number;
	selectedHourReadbackMs?: number;
	selectedDayRangeSummaryMs?: number;
	selectedDayRangeSummaryDispatchMs?: number;
	selectedDayRangeSummaryReadbackMs?: number;
	selectedDayRangeSummaryReadbackBytes?: number;
	selectedDayRangeSummaryReadbackCount?: number;
	selectedDayRangeSummaryComputedHourCount?: number;
	selectedDayRangeSummaryReductionPassCount?: number;
	selectedDayRangeFullReadbackAvoidedCount?: number;
	selectedHourRangeSummaryMs?: number;
	selectedHourRangeSummaryDispatchMs?: number;
	selectedHourRangeSummaryReadbackMs?: number;
	selectedHourRangeSummaryReadbackBytes?: number;
	selectedHourRangeSummaryReadbackCount?: number;
	selectedHourRangeSummaryReductionPassCount?: number;
	selectedHourRangeFullReadbackAvoidedCount?: number;
	selectedHourAnalysisBuildMs?: number;
	cpuColorBuildMs?: number;
	gpuSurfaceUpdateMs?: number;
}

export interface StaticUploadTrace {
	totalMs: number;
	weatherSnapshotMs?: number;
	weatherTimeStepCount?: number;
	solarExposureBufferCreateMs?: number;
	solarZeroFillMs?: number;
	solarZeroFillWriteCount?: number;
	solarZeroFillBytes?: number;
	skyExposureBufferCreateMs?: number;
	skyZeroFillMs?: number;
	skyZeroFillWriteCount?: number;
	skyZeroFillBytes?: number;
	gridBufferCreateMs?: number;
	gridWriteMs?: number;
	gridWriteBytes?: number;
	sunBufferCreateMs?: number;
	sunWriteMs?: number;
	sunWriteBytes?: number;
	sunAltitudeBufferCreateMs?: number;
	sunAltitudeWriteMs?: number;
	sunAltitudeWriteBytes?: number;
	domeVectorBufferCreateMs?: number;
	domeVectorWriteMs?: number;
	domeVectorWriteBytes?: number;
	domeWeightBufferCreateMs?: number;
	domeWeightWriteMs?: number;
	domeWeightWriteBytes?: number;
	bvhSerializeMs?: number;
	bvhNodeBufferCreateMs?: number;
	bvhNodeWriteMs?: number;
	bvhNodeWriteBytes?: number;
	bvhIndexBufferCreateMs?: number;
	bvhIndexWriteMs?: number;
	bvhIndexWriteBytes?: number;
	bvhVertexBufferCreateMs?: number;
	bvhVertexWriteMs?: number;
	bvhVertexWriteBytes?: number;
	bvhParamBufferCreateMs?: number;
	bvhParamWriteMs?: number;
	bvhParamWriteBytes?: number;
}

export function invokeDiagnosticsCallbackSafely<T>(
	callback: ((payload: T) => void | Promise<void>) | undefined,
	payload: T,
	contextLabel: string
): void {
	try {
		const result = callback?.(payload);
		if (result && typeof (result as Promise<void>).catch === 'function') {
			void result.catch((error) => {
				console.error(`[${contextLabel}] diagnostics callback failed.`, error);
			});
		}
	} catch (error) {
		console.error(`[${contextLabel}] diagnostics callback failed.`, error);
	}
}

export type SelectedHourRenderTimingSubsteps = Pick<
	OnDemandTimings,
	| 'renderSceneSyncStartDelayMs'
	| 'renderSceneSyncTotalMs'
	| 'renderLayoutBuildMs'
	| 'renderSurfaceMeshMs'
	| 'renderStorageInitWaitMs'
	| 'renderBufferCopyMs'
	| 'renderQueueDrainMs'
> & {
	renderPublication?: SelectedHourRenderPublicationDiagnostics;
};

const SELECTED_HOUR_RENDER_SUBSTEP_KEYS = [
	'renderSceneSyncStartDelayMs',
	'renderSceneSyncTotalMs',
	'renderLayoutBuildMs',
	'renderSurfaceMeshMs',
	'renderStorageInitWaitMs',
	'renderBufferCopyMs',
	'renderQueueDrainMs'
] as const satisfies ReadonlyArray<keyof SelectedHourRenderTimingSubsteps>;

const SELECTED_HOUR_RENDER_TIMING_KEYS = [
	'renderUpdateMs',
	'gpuSurfaceUpdateMs',
	...SELECTED_HOUR_RENDER_SUBSTEP_KEYS
] as const satisfies ReadonlyArray<keyof OnDemandTimings>;

export const COLD_START_TIMING_KEYS = [
	'payloadPrepareMs',
	'workerBvhMs',
	'pipelineUploadMs',
	'firstSelectedHourReadyMs',
	'firstSelectedHourVisibleMs'
] as const;

export type ColdStartTimingKey = (typeof COLD_START_TIMING_KEYS)[number];

export interface TrackedGpuAllocationBytes {
	persistentExposureBytes: number;
	allHoursOutputBytes: number;
	selectedHourOutputBytes: number;
	selectedHourOutputBytesHighWatermark: number;
	renderOwnedSelectedHourBytes?: number;
	renderOwnedSelectedHourBytesHighWatermark?: number;
	trackingScope: 'utci-owned-webgpu-buffers';
}

export type TrackedGpuAllocationBytesPatch = Partial<
	Omit<TrackedGpuAllocationBytes, 'trackingScope' | 'selectedHourOutputBytesHighWatermark'>
>;

export interface OnDemandRuntimeDiagnostics {
	navigatorGpu: boolean;
	rendererBackend: OnDemandRendererBackend;
	path: OnDemandPath;
	gpuResidentRenderAvailable: boolean;
	sameDeviceForComputeAndRender: boolean | null;
	gpuResidentCopyStatus: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
	adapterInfo?: string | null;
	maxStorageBufferBindingSize?: number | null;
	maxBufferSize?: number | null;
	maxStorageBuffersPerShaderStage?: number | null;
	rendererRequestedMaxStorageBufferBindingSize?: number | null;
	rendererRequestedMaxBufferSize?: number | null;
	rendererDeviceMaxStorageBufferBindingSize?: number | null;
	rendererDeviceMaxBufferSize?: number | null;
	modelId?: string | null;
	scenarioId?: string | null;
	gridResolution?: number | null;
	pointCount?: number | null;
	selectedMonthIndex: number | null;
	selectedTimeIndex: number | null;
	completedMonthIndex: number | null;
	completedTimeIndex: number | null;
	activeRequestId: number | null;
	completedRequestId: number | null;
	staleResultDiscardCount: number;
	inFlightCount: number;
	scrubSampleCount: number;
	timeIndices: number[];
	usedRunAllForSelectedHour: boolean;
	usedExposureOnlyPrecompute: boolean;
	allHoursUtciBytesAllocated: number;
	allHoursMrtBytesAllocated: number;
	oneHourOutputBytes: number;
	selectedHourTransferCount: number;
	trackedGpuAllocationBytes: TrackedGpuAllocationBytes;
	renderTransport:
		| 'none'
		| 'compute-buffer-selected-hour'
		| 'cpu-uploaded-selected-hour';
	visibleSelectedHourReadbackCount?: number;
	selectedHourReadbackReasons?: SelectedHourReadbackReason[];
	selectedHourReadbackReasonCounts?: Partial<Record<SelectedHourReadbackReason, number>>;
	debugReadbackCount: number;
	dataTextureBuildCount: number;
	timings: OnDemandTimings;
	error?: string;
}

export function createEmptyOnDemandDiagnostics(): OnDemandRuntimeDiagnostics {
	return {
		navigatorGpu: false,
		rendererBackend: 'unknown',
		path: 'idle',
		gpuResidentRenderAvailable: false,
		sameDeviceForComputeAndRender: null,
		gpuResidentCopyStatus: 'idle',
		selectedMonthIndex: null,
		selectedTimeIndex: null,
		completedMonthIndex: null,
		completedTimeIndex: null,
		activeRequestId: null,
		completedRequestId: null,
		staleResultDiscardCount: 0,
		inFlightCount: 0,
		scrubSampleCount: 0,
		timeIndices: [],
		usedRunAllForSelectedHour: false,
		usedExposureOnlyPrecompute: false,
		allHoursUtciBytesAllocated: 0,
		allHoursMrtBytesAllocated: 0,
		oneHourOutputBytes: 0,
		selectedHourTransferCount: 0,
		trackedGpuAllocationBytes: {
			persistentExposureBytes: 0,
			allHoursOutputBytes: 0,
			selectedHourOutputBytes: 0,
			selectedHourOutputBytesHighWatermark: 0,
			renderOwnedSelectedHourBytes: 0,
			renderOwnedSelectedHourBytesHighWatermark: 0,
			trackingScope: 'utci-owned-webgpu-buffers'
		},
		renderTransport: 'none',
		debugReadbackCount: 0,
		dataTextureBuildCount: 0,
		timings: {}
	};
}

export function recordOnDemandTiming<K extends keyof OnDemandTimings>(
	diagnostics: OnDemandRuntimeDiagnostics,
	key: K,
	value: OnDemandTimings[K]
): OnDemandRuntimeDiagnostics {
	return {
		...diagnostics,
		timings: {
			...diagnostics.timings,
			[key]: value
		}
	};
}

export function recordSelectedHourReadbackReason(
	diagnostics: OnDemandRuntimeDiagnostics,
	reason: SelectedHourReadbackReason
): OnDemandRuntimeDiagnostics {
	const existingReasons = diagnostics.selectedHourReadbackReasons ?? [];
	const existingCounts = diagnostics.selectedHourReadbackReasonCounts ?? {};
	return {
		...diagnostics,
		selectedHourReadbackReasons: [...existingReasons, reason],
		selectedHourReadbackReasonCounts: {
			...existingCounts,
			[reason]: (existingCounts[reason] ?? 0) + 1
		}
	};
}

export function mergeSelectedHourRenderTimings(params: {
	existingTimings?: OnDemandTimings;
	renderUpdateMs: number;
	gpuSurfaceUpdateMs: number;
	firstSelectedHourVisibleMs?: number;
	renderSubsteps?: SelectedHourRenderTimingSubsteps;
}): OnDemandTimings {
	const {
		existingTimings,
		renderUpdateMs,
		gpuSurfaceUpdateMs,
		firstSelectedHourVisibleMs,
		renderSubsteps
	} = params;

	const nextTimings: OnDemandTimings = {
		...existingTimings,
		renderUpdateMs,
		gpuSurfaceUpdateMs
	};
	for (const key of SELECTED_HOUR_RENDER_SUBSTEP_KEYS) delete nextTimings[key];

	if (firstSelectedHourVisibleMs !== undefined) {
		nextTimings.firstSelectedHourVisibleMs = firstSelectedHourVisibleMs;
	}

	delete nextTimings.renderPublication;

	if (!renderSubsteps) {
		return nextTimings;
	}

	nextTimings.renderPublication = copyRenderPublicationDiagnostics(
		renderSubsteps.renderPublication
	);

	for (const key of SELECTED_HOUR_RENDER_SUBSTEP_KEYS) {
		const value = renderSubsteps[key];
		if (value !== undefined) {
			nextTimings[key] = value;
		}
	}

	return nextTimings;
}

export function clearSelectedHourRenderTimings(existingTimings?: OnDemandTimings): OnDemandTimings {
	const nextTimings: OnDemandTimings = { ...existingTimings };
	for (const key of SELECTED_HOUR_RENDER_TIMING_KEYS) {
		delete nextTimings[key];
	}
	delete nextTimings.renderPublication;
	return nextTimings;
}

export function prepareSelectedHourCycleTimings(params: {
	existingTimings?: OnDemandTimings;
	pipelineTimings?: OnDemandTimings;
	firstSelectedHourReadyMs?: number;
	selectedHourReadbackMs?: number;
	selectedHourAnalysisBuildMs?: number;
}): OnDemandTimings {
	const {
		existingTimings,
		pipelineTimings,
		firstSelectedHourReadyMs,
		selectedHourReadbackMs,
		selectedHourAnalysisBuildMs
	} = params;

	return {
		...clearSelectedHourRenderTimings(existingTimings),
		...pipelineTimings,
		firstSelectedHourReadyMs,
		selectedHourReadbackMs,
		selectedHourAnalysisBuildMs
	};
}

export function buildGpuResidentSurfaceResetPatch(params: {
	existingTimings?: OnDemandTimings;
}): {
	utciSurfaceSource: undefined;
	selectedHourTransferCount: 0;
	dataTextureBuildCount: 0;
	gpuResidentCopyStatus: 'idle';
	gpuResidentCopyError: undefined;
	gpuResidentCopyRequestId: undefined;
	timings: OnDemandTimings;
} {
	return {
		utciSurfaceSource: undefined,
		selectedHourTransferCount: 0,
		dataTextureBuildCount: 0,
		gpuResidentCopyStatus: 'idle',
		gpuResidentCopyError: undefined,
		gpuResidentCopyRequestId: undefined,
		timings: clearSelectedHourRenderTimings(params.existingTimings)
	};
}

export function resetColdStartLifecycleTimings(
	diagnostics: OnDemandRuntimeDiagnostics
): OnDemandRuntimeDiagnostics {
	const nextTimings: OnDemandTimings = { ...diagnostics.timings };
	for (const key of COLD_START_TIMING_KEYS) {
		delete nextTimings[key];
	}
	return {
		...diagnostics,
		timings: nextTimings
	};
}

export function recordColdStartLifecycleTiming<K extends ColdStartTimingKey>(
	diagnostics: OnDemandRuntimeDiagnostics,
	key: K,
	value: OnDemandTimings[K]
): OnDemandRuntimeDiagnostics {
	if (value === undefined || diagnostics.timings[key] !== undefined) {
		return diagnostics;
	}
	return recordOnDemandTiming(diagnostics, key, value);
}

export function mergeTrackedGpuAllocationBytes(
	diagnostics: OnDemandRuntimeDiagnostics,
	patch: TrackedGpuAllocationBytesPatch
): OnDemandRuntimeDiagnostics {
	const selectedHourOutputBytes =
		patch.selectedHourOutputBytes ?? diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes;
	const renderOwnedSelectedHourBytes =
		patch.renderOwnedSelectedHourBytes ??
		diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes ??
		0;
	return {
		...diagnostics,
		trackedGpuAllocationBytes: {
			...diagnostics.trackedGpuAllocationBytes,
			...patch,
			selectedHourOutputBytes,
			selectedHourOutputBytesHighWatermark: Math.max(
				diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark,
				selectedHourOutputBytes
			),
			renderOwnedSelectedHourBytes,
			renderOwnedSelectedHourBytesHighWatermark: Math.max(
				diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytesHighWatermark ??
					0,
				renderOwnedSelectedHourBytes
			),
			trackingScope: 'utci-owned-webgpu-buffers'
		}
	};
}
