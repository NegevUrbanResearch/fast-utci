import * as THREE from 'three';
import type { Group } from 'three';
import type { Analysis, AnalysisMetadata } from '$lib/types/analysis';
import {
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker,
	MAX_GRID_POINTS_GUARD
} from '$lib/compute/mergeAndBvhWorkerClient';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { OnDemandUtciOutput, SerializedBvhForGpu, UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { createWebgpuUtciPipeline } from '$lib/compute/webgpuUtciPipeline';
import { emitComputeTelemetry } from '$lib/compute/telemetry';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import {
	buildSelectedHourLiveAnalysis,
	resolveLiveGpuResidentUtciRange
} from '$lib/compute/liveUtciSelectedHour';

const GRID_FALLBACKS = [2, 4, 6, 8];
const PARITY_SAMPLE_HEIGHT_OFFSET_M = 0.9;

export type SelectedHourGpuResidentOutput = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	output: OnDemandUtciOutput;
	utciRange: { min: number; max: number };
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
	pendingRenderUpdateStartedAt: number;
	renderTransport: 'cpu-uploaded-selected-hour' | 'compute-buffer-selected-hour';
	sameDeviceForComputeAndRender: boolean | null;
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
		colorMode: 'normalized' | 'discrete';
		preferGpuResident: boolean;
		rendererDevice?: GPUDevice;
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
	exposureReady: boolean;
	exposurePrecomputePromise: Promise<void> | null;
	requestSequence: number;
	selectedDayRangeCache: Map<string, { min: number; max: number }>;
};

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

function ensureNotAborted(signal: AbortSignal): void {
	if (signal.aborted) {
		throw new DOMException('Aborted', 'AbortError');
	}
}

function disposeGpuBuffer(output: OnDemandUtciOutput | undefined): void {
	const buffer = output?.gpuBuffer as GPUBuffer | undefined;
	buffer?.destroy?.();
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
				numMonths: state.numMonths
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
}): Promise<SelectedHourCpuFallbackOutput> {
	const selectedHourUtci = await params.pipeline.readOnDemandUtciForDebug?.({
		numPoints: params.numPoints
	});
	if (!selectedHourUtci) {
		throw new Error(
			'Selected-hour live session requires pipeline.readOnDemandUtciForDebug() for CPU fallback output.'
		);
	}

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

async function resolveSelectedDayUtciRange(params: {
	state: PreparedSessionState;
	monthIndex: number;
	selectedTimeIndex: number;
	selectedHourUtci?: Float32Array;
}): Promise<{ min: number; max: number } | null> {
	const cacheKey = `${params.monthIndex}:${params.state.numHours}`;
	const cached = params.state.selectedDayRangeCache.get(cacheKey);
	if (cached) return cached;
	if (!params.state.pipeline.readOnDemandUtciForDebug) return null;

	let dayRange: { min: number; max: number } | null = null;
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

		await params.state.computeManager.runUtciForTimeIndex({
			timeIndex,
			numPoints: params.state.numPoints,
			numHours: params.state.numHours,
			numMonths: params.state.numMonths,
			format: 'f32-utci'
		});
		const values = await params.state.pipeline.readOnDemandUtciForDebug({
			numPoints: params.state.numPoints
		});
		dayRange = accumulateUtciRange(dayRange, values);
	}

	if (dayRange) params.state.selectedDayRangeCache.set(cacheKey, dayRange);
	return dayRange;
}

function createSelectedHourLiveSession(state: PreparedSessionState): SelectedHourLiveSession {
	return {
		base: state.base,
		numPoints: state.numPoints,
		numHours: state.numHours,
		numMonths: state.numMonths,
		deviceSource: state.deviceSource,
		async runSelectedHour(params) {
			ensureNotAborted(state.signal);
			await ensureExposurePrecompute(state);
			ensureNotAborted(state.signal);

			const requestId = ++state.requestSequence;
			const output = await state.computeManager.runUtciForTimeIndex({
				timeIndex: params.timeIndex,
				numPoints: state.numPoints,
				numHours: state.numHours,
				numMonths: state.numMonths,
				format: 'f32-utci'
			});
			ensureNotAborted(state.signal);

			const sameDeviceForComputeAndRender = resolveSameDevice({
				computeManager: state.computeManager,
				rendererDevice: params.rendererDevice
			});
			const preferGpuResident =
				params.preferGpuResident &&
				sameDeviceForComputeAndRender === true &&
				Boolean(output.gpuBuffer);
			const pendingRenderUpdateStartedAt = performance.now();

			if (!preferGpuResident) {
				const fallback = await readSelectedHourCpuFallback({
					pipeline: state.pipeline,
					base: state.base,
					numPoints: state.numPoints,
					monthIndex: params.monthIndex,
					timeIndex: params.timeIndex
				}).catch((error) => {
					disposeGpuBuffer(output);
					throw error;
				});
				disposeGpuBuffer(output);
				return {
					requestId,
					monthIndex: params.monthIndex,
					hourIndex: params.hourIndex,
					timeIndex: params.timeIndex,
					analysis: fallback.analysis,
					gpuResidentOutput: null,
					cpuFallbackValues: fallback.cpuFallbackValues,
					pendingRenderUpdateStartedAt,
					renderTransport: 'cpu-uploaded-selected-hour',
					sameDeviceForComputeAndRender
				};
			}

			const selectedHourUtci = await state.pipeline.readOnDemandUtciForDebug?.({
				numPoints: state.numPoints
			});
			const selectedHourAnalysis = selectedHourUtci
				? buildSelectedHourLiveAnalysis({
						base: state.base,
						utciValues: selectedHourUtci,
						monthIndex: params.monthIndex,
						timeIndex: params.timeIndex
				  })
				: null;
			const selectedDayUtciRange =
				params.colorMode === 'normalized'
					? await resolveSelectedDayUtciRange({
							state,
							monthIndex: params.monthIndex,
							selectedTimeIndex: params.timeIndex,
							selectedHourUtci
					  })
					: null;
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
							timeIndex: params.timeIndex
						});

			return {
				requestId,
				monthIndex: params.monthIndex,
				hourIndex: params.hourIndex,
				timeIndex: params.timeIndex,
				analysis: selectedHourAnalysis,
				gpuResidentOutput: {
					requestId,
					monthIndex: params.monthIndex,
					hourIndex: params.hourIndex,
					timeIndex: params.timeIndex,
					output,
					utciRange: resolveLiveGpuResidentUtciRange({
						colorMode: params.colorMode,
						selectedHourUtci,
						selectedDayUtciRange
					}),
					tooltipUtciValues: selectedHourUtci
				},
				loadCpuFallback,
				pendingRenderUpdateStartedAt,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender
			};
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
		numMonths = 12,
		startMonth = 1,
		zHeight = base.metadata.bounds?.z ?? 0.9
	} = params;
	ensureNotAborted(signal);

	const baseGrid = base.metadata.grid_size || 2;
	const numHours = base.data.numHours ?? base.metadata.hours.length ?? 24;
	const startIdx = GRID_FALLBACKS.findIndex((resolution) => resolution >= baseGrid);
	const resolutionsToTry =
		startIdx >= 0 ? GRID_FALLBACKS.slice(startIdx) : [Math.max(baseGrid, 8)];

	let meshes: Awaited<ReturnType<typeof prepareMeshPayloadForWorkerAsync>>['meshes'] | null = null;
	let totalTriangles = 0;
	let preflight: Awaited<
		ReturnType<typeof prepareMeshPayloadForWorkerAsync>
	>['preflight'] | null = null;
	let effectiveGridResolution = baseGrid;
	let lastError: unknown = null;

	for (const tryResolution of resolutionsToTry) {
		try {
			const result = await prepareMeshPayloadForWorkerAsync(model, {
				signal,
				gridResolution: tryResolution,
				numHours,
				numMonths,
				hasWorkerSupport: typeof Worker !== 'undefined'
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
				workerResult = await runMergeAndBvhInWorker({
					meshes,
					gridResolution: effectiveGridResolution,
					zHeight,
					signal,
					maxGridPoints: MAX_GRID_POINTS_GUARD,
					bvhOnly: true
				});
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

		const initResult = await uploadManager.initFromModelAndWeather({
			serializedBvh: workerResult.serializedBvh,
			useRectangularGridFromBounds: true,
			analysisBounds: bounds,
			coordinateSystem,
			gridOriginOffset,
			epwContent,
			gridResolution: effectiveGridResolution,
			zHeight: computeGridHeight,
			signal
		});
		ensureNotAborted(signal);

		emitComputeTelemetry('pipeline.upload.done', {
			data: { numPoints: initResult.numPoints, numHours, numMonths }
		});

		return createSelectedHourLiveSession({
			base,
			pipeline: activePipeline,
			computeManager,
			numPoints: initResult.numPoints,
			numHours,
			numMonths,
			deviceSource: preferredDevice ? 'renderer' : 'standalone',
			signal,
			exposureReady: false,
			exposurePrecomputePromise: null,
			requestSequence: 0,
			selectedDayRangeCache: new Map()
		});
	} catch (error) {
		pipeline?.dispose?.();
		throw error;
	}
}
