export type OnDemandRendererBackend = 'webgpu' | 'unknown';

export type OnDemandPath = 'idle' | 'run-all-baseline' | 'exposure-only-f32' | 'error';

export interface OnDemandTimings {
	exposurePrecomputeMs?: number;
	oneHourDispatchMs?: number;
	renderUpdateMs?: number;
	debugReadbackMs?: number;
}

export interface TrackedGpuAllocationBytes {
	persistentExposureBytes: number;
	allHoursOutputBytes: number;
	selectedHourOutputBytes: number;
	selectedHourOutputBytesHighWatermark: number;
	trackingScope: 'utci-owned-webgpu-buffers';
}

export type TrackedGpuAllocationBytesPatch = Partial<
	Omit<TrackedGpuAllocationBytes, 'trackingScope' | 'selectedHourOutputBytesHighWatermark'>
>;

export interface OnDemandRuntimeDiagnostics {
	navigatorGpu: boolean;
	rendererBackend: OnDemandRendererBackend;
	path: OnDemandPath;
	adapterInfo?: string | null;
	maxStorageBufferBindingSize?: number | null;
	maxBufferSize?: number | null;
	maxStorageBuffersPerShaderStage?: number | null;
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

export function mergeTrackedGpuAllocationBytes(
	diagnostics: OnDemandRuntimeDiagnostics,
	patch: TrackedGpuAllocationBytesPatch
): OnDemandRuntimeDiagnostics {
	const selectedHourOutputBytes =
		patch.selectedHourOutputBytes ?? diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes;
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
			trackingScope: 'utci-owned-webgpu-buffers'
		}
	};
}
