import * as THREE from 'three';
import type { Group } from 'three';
import type {
	Analysis,
	AnalysisActiveMask,
	ClassifiedAnalysisActiveMask,
	AnalysisMetadata
} from '$lib/types/analysis';
import {
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker
} from '$lib/compute/gpu/mergeAndBvhWorkerClient';
import { ComputeManager } from '$lib/compute/compute-manager';
import type {
	F32MetricOutput,
	OnDemandUtciOutput,
	SerializedBvhForGpu,
	UTCIComputePipeline
} from '$lib/compute/gpu/gpu-pipeline';
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';
import { createWebgpuUtciPipeline } from '$lib/compute/gpu/webgpuUtciPipeline';
import { emitComputeTelemetry } from '$lib/compute/telemetry';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import {
	buildSelectedHourLiveShadingAnalysis,
	buildSelectedHourLiveAnalysis,
	resolveLiveGpuResidentUtciRange
} from '$lib/compute/selected-hour/liveUtciSelectedHour';
import {
	createSelectedHourOutputHandle,
	disposeSelectedHourOutputHandle,
	type SelectedHourOutputHandle
} from '$lib/compute/gpu/selectedHourOutputHandle';
import {
	createEmptyOnDemandDiagnostics,
	recordSelectedHourReadbackReason,
	type OnDemandTimings,
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/on-demand/onDemandDiagnostics';
import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';
import {
	copyRenderPublicationDiagnostics,
	stampRenderPublicationTimeline,
	type SelectedHourRenderPublicationPath
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import { buildClassifiedAnalysisActiveMask } from '$lib/compute/selected-hour/activeMaskSurfaceClassification';

const GRID_RESOLUTION_FALLBACKS = [0.5, 1, 2, 4, 6, 8, 10];
const PARITY_SAMPLE_HEIGHT_OFFSET_M = 0.9;
const LIVE_SELECTED_HOUR_MAX_GRID_POINTS = Number.MAX_SAFE_INTEGER;
const LIVE_SELECTED_HOUR_MAX_ESTIMATED_BYTES = Number.POSITIVE_INFINITY;
export type SelectedHourLiveMetricType = 'utci' | 'shading_index';

function stampSessionRenderTimeline(
	diagnostics: OnDemandRuntimeDiagnostics,
	timeline: Parameters<typeof stampRenderPublicationTimeline>[0]['timeline'],
	renderTransport?: SelectedHourLiveResult['renderTransport']
): void {
	const renderPublicationPath: SelectedHourRenderPublicationPath =
		renderTransport === 'compute-buffer-selected-hour' ||
		renderTransport === 'cpu-uploaded-selected-hour'
			? renderTransport
			: renderTransport === 'live-render-pending'
				? 'none'
				: (diagnostics.timings.renderPublication?.renderPublicationPath ??
					'cpu-uploaded-selected-hour');
	diagnostics.timings.renderPublication = stampRenderPublicationTimeline({
		current: diagnostics.timings.renderPublication,
		timeline,
		fallback: {
			renderPublicationPath,
			renderPublicationPhase: 'unknown',
			renderPublicationMeshAction: 'skipped'
		}
	});
}

export type SelectedHourGpuResidentOutput = {
	requestId: number;
	metricType?: SelectedHourLiveMetricType;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	output: OnDemandUtciOutput | F32MetricOutput;
	gpuOutputHandle?: SelectedHourOutputHandle;
	utciRange: { min: number; max: number };
	shadingIndexRange?: { min: number; max: number };
	tooltipUtciValues?: Float32Array;
};

export type SelectedHourCpuFallbackOutput = {
	analysis: Analysis;
	cpuFallbackValues: Float32Array;
};

export type SelectedHourLiveResult = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	analysis: Analysis | null;
	gpuResidentOutput: SelectedHourGpuResidentOutput | null;
	cpuFallbackValues?: Float32Array;
	loadCpuFallback?: () => Promise<SelectedHourCpuFallbackOutput>;
	selectedHourVisibleStartedAt?: number;
	pendingRenderUpdateStartedAt: number;
	renderTransport:
		| 'cpu-uploaded-selected-hour'
		| 'compute-buffer-selected-hour'
		| 'live-render-pending';
	sameDeviceForComputeAndRender: boolean | null;
	diagnostics: OnDemandRuntimeDiagnostics;
};

export type SelectedHourLiveSession = {
	base: Analysis;
	numPoints: number;
	numHours: number;
	numMonths: number;
	deviceSource: 'renderer' | 'standalone';
	runSelectedHour(params: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		metricType?: SelectedHourLiveMetricType;
		colorMode: 'normalized' | 'discrete';
		preferGpuResident: boolean;
		rendererDevice?: GPUDevice;
		selectedHourReadbackReason?: SelectedHourReadbackReason;
	}): Promise<SelectedHourLiveResult>;
	dispose(): void;
};

type PreparedSessionState = {
	base: Analysis;
	pipeline: UTCIComputePipeline;
	computeManager: ComputeManager;
	numPoints: number;
	numHours: number;
	numMonths: number;
	deviceSource: 'renderer' | 'standalone';
	signal: AbortSignal;
	exposureScheduling?: ExposureSchedulingOptions;
	diagnosticsEnabled?: boolean;
	exposureReady: boolean;
	exposurePrecomputePromise: Promise<void> | null;
	requestSequence: number;
	selectedDayRangeCache: Map<string, { min: number; max: number }>;
	activeMask?: ClassifiedAnalysisActiveMask;
	lifecycleTimings: Pick<
		OnDemandTimings,
		'payloadPrepareMs' | 'workerBvhMs' | 'pipelineUploadMs'
	>;
	coldStartStartedAt: number;
};

function copyRuntimeDiagnosticsSnapshot(
	diagnostics: OnDemandRuntimeDiagnostics | undefined
): Pick<OnDemandRuntimeDiagnostics, 'timings' | 'trackedGpuAllocationBytes'> | undefined {
	if (!diagnostics) return undefined;
	return {
		timings: {
			...diagnostics.timings,
			renderPublication: copyRenderPublicationDiagnostics(
				diagnostics.timings.renderPublication
			)
		},
		trackedGpuAllocationBytes: { ...diagnostics.trackedGpuAllocationBytes }
	};
}

function applyRuntimeDiagnosticsSnapshot(
	target: OnDemandRuntimeDiagnostics,
	snapshot:
		| Pick<OnDemandRuntimeDiagnostics, 'timings' | 'trackedGpuAllocationBytes'>
		| undefined,
	lifecycleTimings: PreparedSessionState['lifecycleTimings']
): void {
	target.timings = {
		...lifecycleTimings,
		...(snapshot?.timings ?? {})
	};
	if (snapshot) {
		target.trackedGpuAllocationBytes = snapshot.trackedGpuAllocationBytes;
	}
}

function recordSelectedHourReadyTiming(
	diagnostics: OnDemandRuntimeDiagnostics,
	startedAt: number
): void {
	diagnostics.timings = {
		...diagnostics.timings,
		firstSelectedHourReadyMs: performance.now() - startedAt
	};
}

function getGridOriginOffset(
	baseMetadata: AnalysisMetadata
): { x: number; y: number; z: number } | undefined {
	if (!isNormalizationEnabled()) return undefined;

	const coordinateSystem =
		(baseMetadata.coordinate_system as 'xy_ground' | 'xz_ground') ?? 'xy_ground';
	const scenarioOrigin = calculateScenarioOrigin(baseMetadata as never);
	const anchorOffset = getAnchorOffset();
	const transformedOrigin =
		coordinateSystem === 'xy_ground'
			? new THREE.Vector3(scenarioOrigin.x, scenarioOrigin.z, -scenarioOrigin.y)
			: scenarioOrigin.clone();
	const normalizationOffset = anchorOffset.clone().sub(transformedOrigin);

	return normalizationOffset.lengthSq() > 0.001
		? {
				x: normalizationOffset.x,
				y: normalizationOffset.y,
				z: normalizationOffset.z
			}
		: undefined;
}

function worldToAnalysisCoords(
	x: number,
	y: number,
	z: number,
	coordinateSystem: 'xy_ground' | 'xz_ground'
): [number, number, number] {
	if (coordinateSystem === 'xy_ground') {
		return [x, -z, y];
	}
	return [x, y, z];
}

function applyActiveMaskDiagnostics(
	diagnostics: OnDemandRuntimeDiagnostics,
	activeMask: AnalysisActiveMask | undefined
): void {
	if (!activeMask) return;
	diagnostics.activeMaskSource = activeMask.source;
	diagnostics.canonicalPointCount = activeMask.canonicalPointCount;
	diagnostics.activePointCount = activeMask.activePointCount;
	diagnostics.inactivePointCount = activeMask.inactivePointCount;
	diagnostics.activePointRatio = activeMask.activePointRatio;
	diagnostics.activeMaskChecksum = activeMask.activeMaskChecksum;
}

function requiresClassifiedActiveMask(analysisId: string): boolean {
	return analysisId === 'Innovation-District' || analysisId.startsWith('Innovation-District/');
}

function buildResolutionsToTry(requestedGridResolution: number): number[] {
	const requested = Number.isFinite(requestedGridResolution) && requestedGridResolution > 0
		? requestedGridResolution
		: 2;
	return [
		requested,
		...GRID_RESOLUTION_FALLBACKS.filter(
			(resolution) => resolution > requested && resolution <= 10
		)
	];
}

function buildGeneratedGridBaseAnalysis(params: {
	base: Analysis;
	effectiveGridResolution: number;
	numPoints: number;
	gridPoints: Float32Array | undefined;
	gridOriginOffset: { x: number; y: number; z: number } | undefined;
	activeMask?: ClassifiedAnalysisActiveMask;
}): Analysis {
	const expectedLength = params.numPoints * 3;
	const coordinateSystem =
		(params.base.metadata.coordinate_system as 'xy_ground' | 'xz_ground') ?? 'xy_ground';
	if (!params.gridPoints || params.gridPoints.length !== expectedLength) {
		throw new Error(
			`Live selected-hour generated grid point length mismatch: expected ${expectedLength}, got ${
				params.gridPoints?.length ?? 'missing'
			}.`
		);
	}

	const positions = new Float32Array(expectedLength);

	const offset = params.gridOriginOffset ?? { x: 0, y: 0, z: 0 };
	for (let pointIndex = 0; pointIndex < params.numPoints; pointIndex += 1) {
		const baseIndex = pointIndex * 3;
		const worldX = params.gridPoints[baseIndex] - offset.x;
		const worldY =
			params.gridPoints[baseIndex + 1] - offset.y - PARITY_SAMPLE_HEIGHT_OFFSET_M;
		const worldZ = params.gridPoints[baseIndex + 2] - offset.z;
		const [x, y, z] = worldToAnalysisCoords(
			worldX,
			worldY,
			worldZ,
			coordinateSystem
		);
		positions[baseIndex] = x;
		positions[baseIndex + 1] = y;
		positions[baseIndex + 2] = z;
	}

	return {
		metadata: {
			...params.base.metadata,
			grid_size: params.effectiveGridResolution,
			num_positions: params.numPoints,
			activeMask: params.activeMask
		},
		data: {
			...params.base.data,
			numPositions: params.numPoints,
			positions,
			utciValues:
				'utciValues' in params.base.data &&
				params.base.data.utciValues?.length === params.numPoints
					? params.base.data.utciValues
					: undefined,
			utciByHour:
				'utciByHour' in params.base.data &&
				params.base.data.utciByHour?.every((slice) => slice.length === params.numPoints)
					? params.base.data.utciByHour
					: undefined,
			utciStorage:
				'utciStorage' in params.base.data &&
				params.base.data.utciStorage?.numPoints === params.numPoints
					? params.base.data.utciStorage
					: undefined,
			shadingIndex:
				'shadingIndex' in params.base.data &&
				params.base.data.shadingIndex?.length === params.numPoints
					? params.base.data.shadingIndex
					: undefined
		} as Analysis['data']
	};
}

function ensureNotAborted(signal: AbortSignal): void {
	if (signal.aborted) {
		throw new DOMException('Aborted', 'AbortError');
	}
}

function disposeGpuBuffer(
	output:
		| Pick<OnDemandUtciOutput | F32MetricOutput, 'gpuOutputHandle' | 'gpuBuffer'>
		| undefined
): void {
	disposeSelectedHourOutputHandle(output?.gpuOutputHandle);
	if (!output?.gpuOutputHandle && output?.gpuBuffer) {
		const buffer = output.gpuBuffer as GPUBuffer;
		buffer.destroy?.();
	}
	if (output && 'gpuBuffer' in output) {
		output.gpuBuffer = undefined;
	}
}

export function disposeSelectedHourGpuResidentOutput(
	value: SelectedHourGpuResidentOutput | null
): void {
	disposeGpuBuffer(value?.output);
}

async function ensureExposurePrecompute(state: PreparedSessionState): Promise<void> {
	if (state.exposureReady) return;
	if (!state.exposurePrecomputePromise) {
		state.exposurePrecomputePromise = state.computeManager
			.runExposurePrecompute({
				numPoints: state.numPoints,
				numHours: state.numHours,
				numMonths: state.numMonths,
				exposureScheduling: state.exposureScheduling,
				diagnosticsEnabled: state.diagnosticsEnabled,
				signal: state.signal
			})
			.then(() => {
				state.exposureReady = true;
			})
			.finally(() => {
				state.exposurePrecomputePromise = null;
			});
	}
	await state.exposurePrecomputePromise;
	ensureNotAborted(state.signal);
}

function resolveSameDevice(params: {
	computeManager: ComputeManager;
	rendererDevice?: GPUDevice;
}): boolean | null {
	const computeDevice = params.computeManager.getDeviceForDebug();
	const { rendererDevice } = params;
	if (!computeDevice || !rendererDevice) return null;
	return computeDevice === rendererDevice;
}

async function readSelectedHourCpuFallback(params: {
	pipeline: UTCIComputePipeline;
	base: Analysis;
	numPoints: number;
	monthIndex: number;
	timeIndex: number;
	readbackReason: SelectedHourReadbackReason;
	recordReadback: (reason: SelectedHourReadbackReason) => void;
}): Promise<SelectedHourCpuFallbackOutput> {
	const selectedHourUtci = await params.pipeline.readOnDemandUtciForDebug?.({
		numPoints: params.numPoints
	});
	if (!selectedHourUtci) {
		throw new Error(
			'Selected-hour live session requires pipeline.readOnDemandUtciForDebug() for CPU fallback output.'
		);
	}
	params.recordReadback(params.readbackReason);

	return {
		analysis: buildSelectedHourLiveAnalysis({
			base: params.base,
			utciValues: selectedHourUtci,
			monthIndex: params.monthIndex,
			timeIndex: params.timeIndex
		}),
		cpuFallbackValues: selectedHourUtci
	};
}

function getUtciValuesRange(values: Float32Array): { min: number; max: number } | null {
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (const value of values) {
		if (!Number.isFinite(value)) continue;
		if (value < min) min = value;
		if (value > max) max = value;
	}
	return Number.isFinite(min) && Number.isFinite(max) && max > min ? { min, max } : null;
}

function accumulateUtciRange(
	current: { min: number; max: number } | null,
	values: Float32Array
): { min: number; max: number } | null {
	const range = getUtciValuesRange(values);
	if (!range) return current;
	if (!current) return range;
	return {
		min: Math.min(current.min, range.min),
		max: Math.max(current.max, range.max)
	};
}

function accumulateUtciRangeSummary(
	current: { min: number; max: number } | null,
	range: { min: number; max: number } | null
): { min: number; max: number } | null {
	if (!range) return current;
	if (!current) return range;
	return {
		min: Math.min(current.min, range.min),
		max: Math.max(current.max, range.max)
	};
}

const SELECTED_HOUR_RANGE_SUMMARY_TIMING_KEYS = [
	'selectedHourRangeSummaryMs',
	'selectedHourRangeSummaryDispatchMs',
	'selectedHourRangeSummaryReadbackMs',
	'selectedHourRangeSummaryReadbackBytes',
	'selectedHourRangeSummaryReadbackCount',
	'selectedHourRangeSummaryReductionPassCount',
	'selectedHourRangeFullReadbackAvoidedCount'
] as const satisfies ReadonlyArray<keyof OnDemandTimings>;

function pickSelectedHourRangeSummaryTimings(
	timings: OnDemandTimings | undefined
): OnDemandTimings {
	const picked: OnDemandTimings = {};
	if (!timings) return picked;
	for (const key of SELECTED_HOUR_RANGE_SUMMARY_TIMING_KEYS) {
		const value = timings[key];
		if (value !== undefined) {
			picked[key] = value as never;
		}
	}
	return picked;
}

async function resolveSelectedHourUtciRange(params: {
	state: PreparedSessionState;
	output: OnDemandUtciOutput;
	timeIndex: number;
	colorMode: 'normalized' | 'discrete';
	selectedHourUtci?: Float32Array;
}): Promise<{
	range: { min: number; max: number } | null;
	resolutionPath:
		| 'compact-gpu-summary'
		| 'cpu-scan-existing-values'
		| 'unavailable'
		| 'not-needed';
	readbackCount: number;
	cpuScanCount: number;
	summaryReadbackCount: number;
	summaryReadbackBytes: number;
	fullReadbackAvoidedCount: number;
	reductionPassCount: number;
	timings: OnDemandTimings;
}> {
	if (params.colorMode !== 'discrete') {
		return {
			range: null,
			resolutionPath: 'not-needed',
			readbackCount: 0,
			cpuScanCount: 0,
			summaryReadbackCount: 0,
			summaryReadbackBytes: 0,
			fullReadbackAvoidedCount: 0,
			reductionPassCount: 0,
			timings: {}
		};
	}

	if (params.state.pipeline.runUtciRangeSummaryForOutput) {
		const summaryStartedAt = performance.now();
		const summary = await params.state.computeManager.runUtciRangeSummaryForOutput({
			timeIndex: params.timeIndex,
			numPoints: params.state.numPoints,
			format: 'f32-utci',
			output: params.output,
			signal: params.state.signal
		});
		const pipelineTimings = pickSelectedHourRangeSummaryTimings(
			params.state.computeManager.getOnDemandDiagnostics?.()?.timings
		);
		return {
			range: summary.range,
			resolutionPath: 'compact-gpu-summary',
			readbackCount: 0,
			cpuScanCount: 0,
			summaryReadbackCount: 1,
			summaryReadbackBytes: summary.readbackBytes,
			fullReadbackAvoidedCount: 1,
			reductionPassCount: summary.reductionPassCount,
			timings: {
				selectedHourRangeSummaryMs: performance.now() - summaryStartedAt,
				selectedHourRangeSummaryReadbackBytes: summary.readbackBytes,
				selectedHourRangeSummaryReadbackCount: 1,
				selectedHourRangeSummaryReductionPassCount: summary.reductionPassCount,
				selectedHourRangeFullReadbackAvoidedCount: 1,
				...pipelineTimings
			}
		};
	}

	if (params.selectedHourUtci) {
		return {
			range: getUtciValuesRange(params.selectedHourUtci),
			resolutionPath: 'cpu-scan-existing-values',
			readbackCount: 0,
			cpuScanCount: 1,
			summaryReadbackCount: 0,
			summaryReadbackBytes: 0,
			fullReadbackAvoidedCount: 0,
			reductionPassCount: 0,
			timings: {}
		};
	}

	return {
		range: null,
		resolutionPath: 'unavailable',
		readbackCount: 0,
		cpuScanCount: 0,
		summaryReadbackCount: 0,
		summaryReadbackBytes: 0,
		fullReadbackAvoidedCount: 0,
		reductionPassCount: 0,
		timings: {}
	};
}

function ensureSelectedHourOutputHandle(params: {
	output: OnDemandUtciOutput;
	requestId: number;
	timeIndex: number;
	byteLength: number;
}): SelectedHourOutputHandle | null {
	const existingHandle = params.output.gpuOutputHandle;
	if (existingHandle) {
		existingHandle.requestId = params.requestId;
		existingHandle.timeIndex = params.timeIndex;
		const legacyBuffer = params.output.gpuBuffer as GPUBuffer | undefined;
		if (legacyBuffer && legacyBuffer !== existingHandle.buffer) {
			legacyBuffer.destroy?.();
		}
		params.output.gpuBuffer = existingHandle.buffer;
		return existingHandle;
	}

	const buffer = params.output.gpuBuffer as GPUBuffer | undefined;
	if (!buffer) return null;

	const handle = createSelectedHourOutputHandle({
		buffer,
		byteLength: params.output.outputBytes ?? params.byteLength,
		source: 'webgpu-on-demand-snapshot',
		requestId: params.requestId,
		timeIndex: params.timeIndex
	});
	params.output.gpuOutputHandle = handle;
	return handle;
}

async function resolveSelectedDayUtciRange(params: {
	state: PreparedSessionState;
	monthIndex: number;
	selectedTimeIndex: number;
	selectedHourUtci?: Float32Array;
	recordReadback: (reason: SelectedHourReadbackReason) => void;
}): Promise<{
	range: { min: number; max: number } | null;
	cacheKey: string;
	cacheHit: boolean;
	cacheSizeBefore: number;
	cacheSizeAfter: number;
	readbackCount: number;
	computedHourCount: number;
	resolutionPath: 'full-readback' | 'compact-gpu-summary' | 'cache-hit' | 'unavailable';
	summaryReadbackCount: number;
	summaryReadbackBytes: number;
	fullReadbackAvoidedCount: number;
}> {
	const cacheKey = `${params.monthIndex}:${params.state.numHours}`;
	const cacheSizeBefore = params.state.selectedDayRangeCache.size;
	const cached = params.state.selectedDayRangeCache.get(cacheKey);
	if (cached) {
		return {
			range: cached,
			cacheKey,
			cacheHit: true,
			cacheSizeBefore,
			cacheSizeAfter: params.state.selectedDayRangeCache.size,
			readbackCount: 0,
			computedHourCount: 0,
			resolutionPath: 'cache-hit',
			summaryReadbackCount: 0,
			summaryReadbackBytes: 0,
			fullReadbackAvoidedCount: 0
		};
	}
	const supportsCompactSummary = Boolean(params.state.pipeline.runUtciRangeSummaryForTimeIndex);
	const readOnDemandUtciForDebug = params.state.pipeline.readOnDemandUtciForDebug;
	if (!supportsCompactSummary && !readOnDemandUtciForDebug) {
		return {
			range: null,
			cacheKey,
			cacheHit: false,
			cacheSizeBefore,
			cacheSizeAfter: params.state.selectedDayRangeCache.size,
			readbackCount: 0,
			computedHourCount: 0,
			resolutionPath: 'unavailable',
			summaryReadbackCount: 0,
			summaryReadbackBytes: 0,
			fullReadbackAvoidedCount: 0
		};
	}

	let dayRange: { min: number; max: number } | null = null;
	let readbackCount = 0;
	let computedHourCount = 0;
	let summaryReadbackCount = 0;
	let summaryReadbackBytes = 0;
	let fullReadbackAvoidedCount = 0;
	const monthStart = params.monthIndex * params.state.numHours;
	const monthEnd = Math.min(
		monthStart + params.state.numHours,
		params.state.numMonths * params.state.numHours
	);
	for (let timeIndex = monthStart; timeIndex < monthEnd; timeIndex += 1) {
		ensureNotAborted(params.state.signal);
		if (timeIndex === params.selectedTimeIndex && params.selectedHourUtci) {
			dayRange = accumulateUtciRange(dayRange, params.selectedHourUtci);
			continue;
		}

		if (supportsCompactSummary) {
			const summary = await params.state.computeManager.runUtciRangeSummaryForTimeIndex({
				timeIndex,
				numPoints: params.state.numPoints,
				numHours: params.state.numHours,
				numMonths: params.state.numMonths,
				format: 'f32-utci',
				signal: params.state.signal
			});
			computedHourCount += 1;
			summaryReadbackCount += 1;
			summaryReadbackBytes += summary.readbackBytes;
			fullReadbackAvoidedCount += 1;
			dayRange = accumulateUtciRangeSummary(dayRange, summary.range);
			continue;
		}

		if (!readOnDemandUtciForDebug) {
			throw new Error('Selected-day range fallback readback is unavailable.');
		}
		await params.state.computeManager.runUtciForTimeIndex({
			timeIndex,
			numPoints: params.state.numPoints,
			numHours: params.state.numHours,
			numMonths: params.state.numMonths,
			format: 'f32-utci'
		});
		computedHourCount += 1;
		const values = await readOnDemandUtciForDebug({
			numPoints: params.state.numPoints
		});
		params.recordReadback('range');
		readbackCount += 1;
		dayRange = accumulateUtciRange(dayRange, values);
	}

	if (dayRange) params.state.selectedDayRangeCache.set(cacheKey, dayRange);
	return {
		range: dayRange,
		cacheKey,
		cacheHit: false,
		cacheSizeBefore,
		cacheSizeAfter: params.state.selectedDayRangeCache.size,
		readbackCount,
		computedHourCount,
		resolutionPath: supportsCompactSummary ? 'compact-gpu-summary' : 'full-readback',
		summaryReadbackCount,
		summaryReadbackBytes,
		fullReadbackAvoidedCount
	};
}

function createSelectedHourLiveSession(state: PreparedSessionState): SelectedHourLiveSession {
	return {
		base: state.base,
		numPoints: state.numPoints,
		numHours: state.numHours,
		numMonths: state.numMonths,
		deviceSource: state.deviceSource,
		async runSelectedHour(params) {
			const selectedHourVisibleStartedAt = performance.now();
			const metricType = params.metricType ?? 'utci';
			const requestReadyStartedAt =
				state.requestSequence === 0 ? state.coldStartStartedAt : selectedHourVisibleStartedAt;
			const diagnostics = createEmptyOnDemandDiagnostics();
			applyActiveMaskDiagnostics(diagnostics, state.activeMask);
			const recordReadback = (reason: SelectedHourReadbackReason) => {
				Object.assign(diagnostics, recordSelectedHourReadbackReason(diagnostics, reason));
			};
			ensureNotAborted(state.signal);
			await ensureExposurePrecompute(state);
			ensureNotAborted(state.signal);
			const afterExposureDiagnostics = copyRuntimeDiagnosticsSnapshot(
				state.computeManager.getOnDemandDiagnostics?.()
			);
			applyRuntimeDiagnosticsSnapshot(
				diagnostics,
				afterExposureDiagnostics,
				state.lifecycleTimings
			);

			const requestId = ++state.requestSequence;
			stampSessionRenderTimeline(diagnostics, {
				sessionMetricType: metricType
			});

			if (metricType === 'shading_index') {
				let shadingOutput: F32MetricOutput | null = null;
				try {
					shadingOutput = await state.computeManager.runShadingIndex({
						numPoints: state.numPoints,
						numHours: state.numHours,
						numMonths: state.numMonths,
						monthIndex: params.monthIndex,
						startTimeIndex: params.monthIndex * state.numHours,
						timeCount: state.numHours,
						signal: state.signal
					});
					const sessionComputeOutputReturnedAtMs = performance.now();
					const afterDispatchDiagnostics = copyRuntimeDiagnosticsSnapshot(
						state.computeManager.getOnDemandDiagnostics?.()
					);
					if (shadingOutput.period.kind !== 'month-index') {
						throw new Error(
							'Selected-hour Shading Index publication requires a month-index output period.'
						);
					}
					applyRuntimeDiagnosticsSnapshot(
						diagnostics,
						afterDispatchDiagnostics,
						state.lifecycleTimings
					);
					const sessionDiagnosticsAppliedAtMs = performance.now();
					const shadingIndexOutputBytes =
						shadingOutput.outputBytes ?? state.numPoints * 4;
					const shadingIndexSnapshotBytes =
						diagnostics.timings.shadingIndexSnapshotBytes ??
						diagnostics.timings.shadingIndexOutputBytes ??
						shadingIndexOutputBytes;
					const gpuOutputHandle = shadingOutput.gpuOutputHandle;
					const sessionGpuOutputHandleReadyAtMs = performance.now();
					ensureNotAborted(state.signal);

					const sameDeviceForComputeAndRender = resolveSameDevice({
						computeManager: state.computeManager,
						rendererDevice: params.rendererDevice
					});
					const preferGpuResident =
						params.preferGpuResident &&
						sameDeviceForComputeAndRender === true &&
						Boolean(gpuOutputHandle);
					const pendingRenderUpdateStartedAt = performance.now();
					if (!preferGpuResident) {
						disposeGpuBuffer(shadingOutput);
						shadingOutput = null;
						throw new Error(
							'Selected-hour Shading Index publication requires same-device GPU-resident output; CPU fallback is not available.'
						);
					}

					stampSessionRenderTimeline(
						diagnostics,
						{
							sessionMetricType: 'shading_index',
							sessionMetricPeriodKind: shadingOutput.period.kind,
							sessionMetricPeriodIndex: shadingOutput.period.index,
							sessionMetricPeriodStartTimeIndex: shadingOutput.period.startTimeIndex,
							sessionMetricPeriodTimeCount: shadingOutput.period.timeCount,
							sessionComputeOutputReturnedAtMs,
							sessionDiagnosticsAppliedAtMs,
							sessionGpuOutputHandleReadyAtMs,
							sessionPreferGpuResidentResolvedAtMs: pendingRenderUpdateStartedAt,
							sessionOutputBytes: shadingIndexOutputBytes,
							sessionCompactSummaryBytes: 0,
							sessionShadingIndexDispatchMs:
								diagnostics.timings.shadingIndexDispatchMs,
							sessionShadingIndexQueueWaitMs:
								diagnostics.timings.shadingIndexQueueWaitMs,
							sessionShadingIndexOutputBytes: shadingIndexOutputBytes,
							sessionShadingIndexSnapshotBytes: shadingIndexSnapshotBytes,
							sessionShadingIndexSource: 'fresh-dispatch',
							sessionShadingIndexMonthCacheHit: false,
							sessionFullSolarReadbackCount: 0,
							sessionTooltipPointReadbackCount: 0,
							sessionTooltipPointReadbackBytes: 0
						},
						'compute-buffer-selected-hour'
					);
					const sessionSelectedHourAnalysisBuildStartedAtMs = performance.now();
					const selectedHourAnalysis = buildSelectedHourLiveShadingAnalysis({
						base: state.base,
						shadingOutput,
						monthIndex: params.monthIndex,
						timeIndex: params.timeIndex
					});
					stampSessionRenderTimeline(diagnostics, {
						sessionSelectedHourAnalysisBuildStartedAtMs,
						sessionSelectedHourAnalysisBuildCompletedAtMs: performance.now()
					});

					recordSelectedHourReadyTiming(diagnostics, requestReadyStartedAt);
					stampSessionRenderTimeline(diagnostics, {
						sessionResultReadyAtMs: performance.now()
					});
					const shadingIndexRange = { min: 0, max: 1 };
					const gpuResidentOutput = {
						requestId,
						metricType: 'shading_index' as const,
						monthIndex: params.monthIndex,
						hourIndex: params.hourIndex,
						timeIndex: params.timeIndex,
						output: shadingOutput,
						gpuOutputHandle: gpuOutputHandle ?? undefined,
						utciRange: shadingIndexRange,
						shadingIndexRange
					};
					const result: SelectedHourLiveResult = {
						requestId,
						monthIndex: params.monthIndex,
						hourIndex: params.hourIndex,
						timeIndex: params.timeIndex,
						analysis: selectedHourAnalysis,
						gpuResidentOutput,
						selectedHourVisibleStartedAt,
						pendingRenderUpdateStartedAt,
						renderTransport: 'compute-buffer-selected-hour',
						sameDeviceForComputeAndRender,
						diagnostics
					};
					stampSessionRenderTimeline(diagnostics, {
						sessionResultReadyAtMs: performance.now(),
						sessionResultReturnedAtMs: performance.now()
					});
					shadingOutput = null;
					return result;
				} catch (error) {
					disposeGpuBuffer(shadingOutput ?? undefined);
					throw error;
				}
			}

			const output = await state.computeManager.runUtciForTimeIndex({
				timeIndex: params.timeIndex,
				numPoints: state.numPoints,
				numHours: state.numHours,
				numMonths: state.numMonths,
				format: 'f32-utci'
			});
			const sessionComputeOutputReturnedAtMs = performance.now();
			const afterDispatchDiagnostics = copyRuntimeDiagnosticsSnapshot(
				state.computeManager.getOnDemandDiagnostics?.()
			);
			applyRuntimeDiagnosticsSnapshot(
				diagnostics,
				afterDispatchDiagnostics,
				state.lifecycleTimings
			);
			const sessionDiagnosticsAppliedAtMs = performance.now();
			const gpuOutputHandle = ensureSelectedHourOutputHandle({
				output,
				requestId,
				timeIndex: params.timeIndex,
				byteLength: state.numPoints * 4
			});
			const sessionGpuOutputHandleReadyAtMs = performance.now();
			ensureNotAborted(state.signal);

			const sameDeviceForComputeAndRender = resolveSameDevice({
				computeManager: state.computeManager,
				rendererDevice: params.rendererDevice
			});
			const preferGpuResident =
				params.preferGpuResident &&
				sameDeviceForComputeAndRender === true &&
				Boolean(gpuOutputHandle);
			const pendingRenderUpdateStartedAt = performance.now();
			const renderTransport = preferGpuResident
				? 'compute-buffer-selected-hour'
				: 'cpu-uploaded-selected-hour';
			stampSessionRenderTimeline(
				diagnostics,
				{
					sessionMetricType: 'utci',
					sessionComputeOutputReturnedAtMs,
					sessionDiagnosticsAppliedAtMs,
					sessionGpuOutputHandleReadyAtMs,
					sessionPreferGpuResidentResolvedAtMs: pendingRenderUpdateStartedAt,
					sessionOutputBytes: output.outputBytes
				},
				renderTransport
			);

			if (!preferGpuResident) {
				const fallback = await readSelectedHourCpuFallback({
					pipeline: state.pipeline,
					base: state.base,
					numPoints: state.numPoints,
					monthIndex: params.monthIndex,
					timeIndex: params.timeIndex,
					readbackReason: 'visible-fallback',
					recordReadback
				}).catch((error) => {
					disposeGpuBuffer(output);
					throw error;
				});
				disposeGpuBuffer(output);
				recordSelectedHourReadyTiming(diagnostics, requestReadyStartedAt);
				return {
					requestId,
					monthIndex: params.monthIndex,
					hourIndex: params.hourIndex,
					timeIndex: params.timeIndex,
					analysis: fallback.analysis,
					gpuResidentOutput: null,
					cpuFallbackValues: fallback.cpuFallbackValues,
					selectedHourVisibleStartedAt,
					pendingRenderUpdateStartedAt,
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender,
					diagnostics
				};
			}

			const sessionDebugReadbackStartedAtMs = performance.now();
			const selectedHourUtci = await state.pipeline.readOnDemandUtciForDebug?.({
				numPoints: state.numPoints
			});
			stampSessionRenderTimeline(diagnostics, {
				sessionDebugReadbackStartedAtMs,
				sessionDebugReadbackCompletedAtMs: performance.now()
			});
			if (selectedHourUtci) {
				recordReadback(params.selectedHourReadbackReason ?? 'tooltip');
			}
			const sessionSelectedHourRangeScanStartedAtMs = performance.now();
			const selectedHourUtciRangeResult = await resolveSelectedHourUtciRange({
				state,
				output,
				timeIndex: params.timeIndex,
				colorMode: params.colorMode,
				selectedHourUtci
			});
			const selectedHourUtciRange = selectedHourUtciRangeResult.range;
			diagnostics.timings = {
				...diagnostics.timings,
				...selectedHourUtciRangeResult.timings
			};
			stampSessionRenderTimeline(diagnostics, {
				sessionSelectedHourRangeScanStartedAtMs,
				sessionSelectedHourRangeScanCompletedAtMs: performance.now(),
				sessionSelectedHourRangeResolutionPath:
					selectedHourUtciRangeResult.resolutionPath,
				sessionSelectedHourRangeReadbackCount:
					selectedHourUtciRangeResult.readbackCount,
				sessionSelectedHourRangeCpuScanCount:
					selectedHourUtciRangeResult.cpuScanCount,
				sessionSelectedHourRangeSummaryReadbackCount:
					selectedHourUtciRangeResult.summaryReadbackCount,
				sessionSelectedHourRangeSummaryReadbackBytes:
					selectedHourUtciRangeResult.summaryReadbackBytes,
				sessionSelectedHourRangeFullReadbackAvoidedCount:
					selectedHourUtciRangeResult.fullReadbackAvoidedCount,
				sessionSelectedHourRangeSummaryReductionPassCount:
					selectedHourUtciRangeResult.reductionPassCount
			});
			const sessionRangeResolveStartedAtMs = performance.now();
			const selectedDayUtciRangeResult =
				params.colorMode === 'normalized'
					? await resolveSelectedDayUtciRange({
							state,
							monthIndex: params.monthIndex,
							selectedTimeIndex: params.timeIndex,
							selectedHourUtci,
							recordReadback
					  })
					: null;
			const selectedDayUtciRange = selectedDayUtciRangeResult?.range ?? null;
			stampSessionRenderTimeline(diagnostics, {
				sessionRangeResolveStartedAtMs,
				sessionRangeResolveCompletedAtMs: performance.now(),
				sessionSelectedDayRangeCacheKey: selectedDayUtciRangeResult?.cacheKey,
				sessionSelectedDayRangeCacheHit: selectedDayUtciRangeResult?.cacheHit,
				sessionSelectedDayRangeCacheSizeBefore:
					selectedDayUtciRangeResult?.cacheSizeBefore,
				sessionSelectedDayRangeCacheSizeAfter:
					selectedDayUtciRangeResult?.cacheSizeAfter,
				sessionSelectedDayRangeReadbackCount:
					selectedDayUtciRangeResult?.readbackCount,
				sessionSelectedDayRangeComputedHourCount:
					selectedDayUtciRangeResult?.computedHourCount,
				sessionSelectedDayRangeResolutionPath:
					selectedDayUtciRangeResult?.resolutionPath,
				sessionSelectedDayRangeSummaryReadbackCount:
					selectedDayUtciRangeResult?.summaryReadbackCount,
				sessionSelectedDayRangeSummaryReadbackBytes:
					selectedDayUtciRangeResult?.summaryReadbackBytes,
				sessionSelectedDayRangeFullReadbackAvoidedCount:
					selectedDayUtciRangeResult?.fullReadbackAvoidedCount
			});
			const selectedHourDisplayRange =
				params.colorMode === 'normalized'
					? selectedDayUtciRange
					: selectedHourUtciRange;
			const sessionSelectedHourAnalysisBuildStartedAtMs = performance.now();
			const selectedHourAnalysis = selectedHourUtci
				? buildSelectedHourLiveAnalysis({
						base: state.base,
						utciValues: selectedHourUtci,
						utciRange: selectedHourDisplayRange,
						monthIndex: params.monthIndex,
						timeIndex: params.timeIndex
				  })
				: null;
			stampSessionRenderTimeline(diagnostics, {
				sessionSelectedHourAnalysisBuildStartedAtMs,
				sessionSelectedHourAnalysisBuildCompletedAtMs: performance.now()
			});
			const sessionCpuFallbackSetupStartedAtMs = performance.now();
			const loadCpuFallback = selectedHourUtci
				? async () => ({
						analysis: selectedHourAnalysis as Analysis,
						cpuFallbackValues: selectedHourUtci
				  })
				: () =>
						readSelectedHourCpuFallback({
							pipeline: state.pipeline,
							base: state.base,
							numPoints: state.numPoints,
							monthIndex: params.monthIndex,
							timeIndex: params.timeIndex,
							readbackReason: 'visible-fallback',
							recordReadback
						});
			stampSessionRenderTimeline(diagnostics, {
				sessionCpuFallbackSetupStartedAtMs,
				sessionCpuFallbackSetupCompletedAtMs: performance.now()
			});

			recordSelectedHourReadyTiming(diagnostics, requestReadyStartedAt);
			const sessionGpuResidentResultAssemblyStartedAtMs = performance.now();
			const sessionGpuResidentRangeResolveStartedAtMs = performance.now();
			const utciRange = resolveLiveGpuResidentUtciRange({
				colorMode: params.colorMode,
				selectedHourUtci,
				selectedHourUtciRange,
				selectedDayUtciRange
			});
			const sessionGpuResidentRangeResolveCompletedAtMs = performance.now();
			const sessionTooltipValuesHandoffStartedAtMs = performance.now();
			const tooltipUtciValues = selectedHourUtci;
			const sessionTooltipValuesHandoffCompletedAtMs = performance.now();
			const gpuResidentOutput = {
				requestId,
				metricType: 'utci' as const,
				monthIndex: params.monthIndex,
				hourIndex: params.hourIndex,
				timeIndex: params.timeIndex,
				output,
				gpuOutputHandle: gpuOutputHandle ?? undefined,
				utciRange,
				tooltipUtciValues
			};
			stampSessionRenderTimeline(diagnostics, {
				sessionGpuResidentRangeResolveStartedAtMs,
				sessionGpuResidentRangeResolveCompletedAtMs,
				sessionTooltipValuesHandoffStartedAtMs,
				sessionTooltipValuesHandoffCompletedAtMs,
				sessionGpuResidentResultAssemblyStartedAtMs,
				sessionGpuResidentResultAssemblyCompletedAtMs: performance.now()
			});
			const result: SelectedHourLiveResult = {
				requestId,
				monthIndex: params.monthIndex,
				hourIndex: params.hourIndex,
				timeIndex: params.timeIndex,
				analysis: selectedHourAnalysis,
				gpuResidentOutput,
				loadCpuFallback,
				selectedHourVisibleStartedAt,
				pendingRenderUpdateStartedAt,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender,
				diagnostics
			};
			const sessionResultReadyAtMs = performance.now();
			stampSessionRenderTimeline(diagnostics, {
				sessionResultReadyAtMs,
				sessionResultReturnedAtMs: performance.now()
			});
			return result;
		},
		dispose() {
			state.pipeline.dispose?.();
		}
	};
}

export async function prepareSelectedHourLiveSession(params: {
	analysisId: string;
	base: Analysis;
	model: Group;
	epwUrl: string;
	signal: AbortSignal;
	preferredDevice?: GPUDevice;
	gridResolution?: number;
	exposureScheduling?: ExposureSchedulingOptions;
	diagnosticsEnabled?: boolean;
	numMonths?: number;
	startMonth?: number;
	zHeight?: number;
}): Promise<SelectedHourLiveSession> {
	const {
		analysisId,
		base,
		model,
		epwUrl,
		signal,
		preferredDevice,
		gridResolution,
		exposureScheduling,
		diagnosticsEnabled = false,
		numMonths = 12,
		startMonth = 1,
		zHeight = base.metadata.bounds?.z ?? 0.9
	} = params;
	ensureNotAborted(signal);

	const requestedGridResolution = gridResolution ?? base.metadata.grid_size ?? 2;
	const numHours = base.data.numHours ?? base.metadata.hours.length ?? 24;
	const resolutionsToTry = buildResolutionsToTry(requestedGridResolution);
	const coldStartStartedAt = performance.now();
	const lifecycleTimings: PreparedSessionState['lifecycleTimings'] = {};

	let meshes: Awaited<ReturnType<typeof prepareMeshPayloadForWorkerAsync>>['meshes'] | null = null;
	let totalTriangles = 0;
	let preflight: Awaited<
		ReturnType<typeof prepareMeshPayloadForWorkerAsync>
	>['preflight'] | null = null;
	let effectiveGridResolution = requestedGridResolution;
	let lastError: unknown = null;

	for (const tryResolution of resolutionsToTry) {
		const payloadPrepareStartedAt = performance.now();
		try {
			const result = await prepareMeshPayloadForWorkerAsync(model, {
				signal,
				gridResolution: tryResolution,
				numHours,
				numMonths,
				maxGridPoints: LIVE_SELECTED_HOUR_MAX_GRID_POINTS,
				maxEstimatedBytes: LIVE_SELECTED_HOUR_MAX_ESTIMATED_BYTES,
				hasWorkerSupport: typeof Worker !== 'undefined',
				analysisBounds: base.metadata.bounds,
				coordinateSystem: base.metadata.coordinate_system ?? 'xy_ground'
			});
			meshes = result.meshes;
			totalTriangles = result.totalTriangles;
			preflight = result.preflight;
			effectiveGridResolution = tryResolution;
			break;
		} catch (error) {
			lastError = error;
			const message = error instanceof Error ? error.message : String(error);
			if (
				message.includes('exceeds budget') &&
				tryResolution < resolutionsToTry[resolutionsToTry.length - 1]
			) {
				continue;
			}
			throw error;
		} finally {
			lifecycleTimings.payloadPrepareMs =
				(lifecycleTimings.payloadPrepareMs ?? 0) +
				(performance.now() - payloadPrepareStartedAt);
		}
	}

	if (!meshes || !preflight) {
		throw lastError ?? new Error('Live selected-hour preflight failed.');
	}

	emitComputeTelemetry('live.preflight.done', {
		data: {
			totalTriangles,
			estimatedGridPoints: preflight.estimatedGridPoints,
			estimatedBytes: preflight.estimatedBytes,
			effectiveGridResolution
		}
	});

	const response = await fetch(epwUrl, { signal });
	if (!response.ok) {
		throw new Error(`Failed to load EPW file for ${analysisId}: ${response.status}`);
	}
	const epwContent = await response.text();
	ensureNotAborted(signal);

	let pipeline: UTCIComputePipeline | null = null;
	try {
		pipeline = await createWebgpuUtciPipeline({
			enableDiagnostics: false,
			device: preferredDevice
		});
		const activePipeline = pipeline;

		let workerResult: { gridPoints: Float32Array; serializedBvh: SerializedBvhForGpu } | null =
			null;
		if (typeof Worker !== 'undefined') {
			try {
				const workerBvhStartedAt = performance.now();
				workerResult = await runMergeAndBvhInWorker({
					meshes,
					gridResolution: effectiveGridResolution,
					zHeight,
					signal,
					maxGridPoints: LIVE_SELECTED_HOUR_MAX_GRID_POINTS,
					bvhOnly: true
				});
				lifecycleTimings.workerBvhMs = performance.now() - workerBvhStartedAt;
			} catch (error) {
				if (error instanceof DOMException && error.name === 'AbortError') {
					throw error;
				}
				throw new Error(
					`Worker BVH generation failed for ${analysisId}: ${
						error instanceof Error ? error.message : String(error)
					}`
				);
			}
		}

		if (!workerResult) {
			throw new Error(
				'Worker did not produce BVH output; selected-hour live session requires serializedBvh.'
			);
		}

		const bounds = base.metadata.bounds as
			| { x_min: number; x_max: number; y_min: number; y_max: number; z?: number }
			| undefined;
		if (!bounds) {
			throw new Error('Analysis metadata is missing bounds for selected-hour live session.');
		}

		const coordinateSystem =
			(base.metadata.coordinate_system as 'xy_ground' | 'xz_ground') ?? 'xy_ground';
		const gridOriginOffset = getGridOriginOffset(base.metadata);
		const computeGridHeight = (bounds.z ?? zHeight) + PARITY_SAMPLE_HEIGHT_OFFSET_M;
		const activeMask = buildClassifiedAnalysisActiveMask({
			model,
			bounds,
			gridResolution: effectiveGridResolution,
			coordinateSystem,
			gridOriginOffset,
			requireClassifiedSurface: requiresClassifiedActiveMask(analysisId)
		});

		const uploadOnlyPipeline: UTCIComputePipeline = {
			uploadStaticData: (uploadParams) => activePipeline.uploadStaticData(uploadParams),
			runAll: async () => {},
			readUtcisSlice: (sliceParams) => activePipeline.readUtcisSlice(sliceParams)
		};
		const uploadManager = new ComputeManager(uploadOnlyPipeline, {
			numMonths,
			numHoursPerDay: numHours,
			startMonth
		});
		const computeManager = new ComputeManager(activePipeline, {
			numMonths,
			numHoursPerDay: numHours,
			startMonth
		});

		const pipelineUploadStartedAt = performance.now();
		const initResult = await uploadManager.initFromModelAndWeather({
			serializedBvh: workerResult.serializedBvh,
			useRectangularGridFromBounds: true,
			analysisBounds: bounds,
			coordinateSystem,
			gridOriginOffset,
			activeCanonicalIndices: activeMask?.activeCanonicalIndices,
			epwContent,
			gridResolution: effectiveGridResolution,
			zHeight: computeGridHeight,
			signal
		});
		ensureNotAborted(signal);
		lifecycleTimings.pipelineUploadMs = performance.now() - pipelineUploadStartedAt;

		emitComputeTelemetry('pipeline.upload.done', {
			data: { numPoints: initResult.numPoints, numHours, numMonths }
		});
		const sessionBase = buildGeneratedGridBaseAnalysis({
			base,
			effectiveGridResolution,
			numPoints: initResult.numPoints,
			gridPoints: initResult.gridPoints,
			gridOriginOffset,
			activeMask
		});

		return createSelectedHourLiveSession({
			base: sessionBase,
			pipeline: activePipeline,
			computeManager,
			numPoints: initResult.numPoints,
			numHours,
			numMonths,
			deviceSource: preferredDevice ? 'renderer' : 'standalone',
			signal,
			exposureScheduling,
			diagnosticsEnabled,
			exposureReady: false,
			exposurePrecomputePromise: null,
			requestSequence: 0,
			selectedDayRangeCache: new Map(),
			activeMask,
			lifecycleTimings,
			coldStartStartedAt
		});
	} catch (error) {
		pipeline?.dispose?.();
		throw error;
	}
}
