import * as THREE from 'three';
import type { Group } from 'three';
import type { Analysis, AnalysisMetadata } from '$lib/types/analysis';
import {
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker
} from '$lib/compute/gpu/mergeAndBvhWorkerClient';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { OnDemandUtciOutput, SerializedBvhForGpu, UTCIComputePipeline } from '$lib/compute/gpu/gpu-pipeline';
import { createWebgpuUtciPipeline } from '$lib/compute/gpu/webgpuUtciPipeline';
import { emitComputeTelemetry } from '$lib/compute/telemetry';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import {
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
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/on-demand/onDemandDiagnostics';
import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';

const GRID_RESOLUTION_FALLBACKS = [0.5, 1, 2, 4, 6, 8, 10];
const PARITY_SAMPLE_HEIGHT_OFFSET_M = 0.9;
const LIVE_SELECTED_HOUR_MAX_GRID_POINTS = Number.MAX_SAFE_INTEGER;
const LIVE_SELECTED_HOUR_MAX_ESTIMATED_BYTES = Number.POSITIVE_INFINITY;

export type SelectedHourGpuResidentOutput = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	output: OnDemandUtciOutput;
	gpuOutputHandle?: SelectedHourOutputHandle;
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
	selectedHourVisibleStartedAt?: number;
	pendingRenderUpdateStartedAt: number;
	renderTransport: 'cpu-uploaded-selected-hour' | 'compute-buffer-selected-hour';
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
	exposureReady: boolean;
	exposurePrecomputePromise: Promise<void> | null;
	requestSequence: number;
	selectedDayRangeCache: Map<string, { min: number; max: number }>;
};

function copyRuntimeDiagnosticsSnapshot(
	diagnostics: OnDemandRuntimeDiagnostics | undefined
): Pick<OnDemandRuntimeDiagnostics, 'timings' | 'trackedGpuAllocationBytes'> | undefined {
	if (!diagnostics) return undefined;
	return {
		timings: { ...diagnostics.timings },
		trackedGpuAllocationBytes: { ...diagnostics.trackedGpuAllocationBytes }
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
			num_positions: params.numPoints
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

function disposeGpuBuffer(output: OnDemandUtciOutput | undefined): void {
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
		params.recordReadback('range');
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
			const selectedHourVisibleStartedAt = performance.now();
			const diagnostics = createEmptyOnDemandDiagnostics();
			const recordReadback = (reason: SelectedHourReadbackReason) => {
				Object.assign(diagnostics, recordSelectedHourReadbackReason(diagnostics, reason));
			};
			ensureNotAborted(state.signal);
			await ensureExposurePrecompute(state);
			ensureNotAborted(state.signal);
			const afterExposureDiagnostics = copyRuntimeDiagnosticsSnapshot(
				state.computeManager.getOnDemandDiagnostics?.()
			);
			if (afterExposureDiagnostics) {
				diagnostics.timings = afterExposureDiagnostics.timings;
				diagnostics.trackedGpuAllocationBytes =
					afterExposureDiagnostics.trackedGpuAllocationBytes;
			}

			const requestId = ++state.requestSequence;
			const output = await state.computeManager.runUtciForTimeIndex({
				timeIndex: params.timeIndex,
				numPoints: state.numPoints,
				numHours: state.numHours,
				numMonths: state.numMonths,
				format: 'f32-utci'
			});
			const afterDispatchDiagnostics = copyRuntimeDiagnosticsSnapshot(
				state.computeManager.getOnDemandDiagnostics?.()
			);
			if (afterDispatchDiagnostics) {
				diagnostics.timings = afterDispatchDiagnostics.timings;
				diagnostics.trackedGpuAllocationBytes =
					afterDispatchDiagnostics.trackedGpuAllocationBytes;
			}
			const gpuOutputHandle = ensureSelectedHourOutputHandle({
				output,
				requestId,
				timeIndex: params.timeIndex,
				byteLength: state.numPoints * 4
			});
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

			const selectedHourUtci = await state.pipeline.readOnDemandUtciForDebug?.({
				numPoints: state.numPoints
			});
			if (selectedHourUtci) {
				recordReadback(params.selectedHourReadbackReason ?? 'tooltip');
			}
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
							selectedHourUtci,
							recordReadback
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
							timeIndex: params.timeIndex,
							readbackReason: 'visible-fallback',
							recordReadback
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
					gpuOutputHandle: gpuOutputHandle ?? undefined,
					utciRange: resolveLiveGpuResidentUtciRange({
						colorMode: params.colorMode,
						selectedHourUtci,
						selectedDayUtciRange
					}),
					tooltipUtciValues: selectedHourUtci
				},
				loadCpuFallback,
				selectedHourVisibleStartedAt,
				pendingRenderUpdateStartedAt,
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender,
				diagnostics
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
	gridResolution?: number;
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
		numMonths = 12,
		startMonth = 1,
		zHeight = base.metadata.bounds?.z ?? 0.9
	} = params;
	ensureNotAborted(signal);

	const requestedGridResolution = gridResolution ?? base.metadata.grid_size ?? 2;
	const numHours = base.data.numHours ?? base.metadata.hours.length ?? 24;
	const resolutionsToTry = buildResolutionsToTry(requestedGridResolution);

	let meshes: Awaited<ReturnType<typeof prepareMeshPayloadForWorkerAsync>>['meshes'] | null = null;
	let totalTriangles = 0;
	let preflight: Awaited<
		ReturnType<typeof prepareMeshPayloadForWorkerAsync>
	>['preflight'] | null = null;
	let effectiveGridResolution = requestedGridResolution;
	let lastError: unknown = null;

	for (const tryResolution of resolutionsToTry) {
		try {
			const result = await prepareMeshPayloadForWorkerAsync(model, {
				signal,
				gridResolution: tryResolution,
				numHours,
				numMonths,
				maxGridPoints: LIVE_SELECTED_HOUR_MAX_GRID_POINTS,
				maxEstimatedBytes: LIVE_SELECTED_HOUR_MAX_ESTIMATED_BYTES,
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
					maxGridPoints: LIVE_SELECTED_HOUR_MAX_GRID_POINTS,
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
		const sessionBase = buildGeneratedGridBaseAnalysis({
			base,
			effectiveGridResolution,
			numPoints: initResult.numPoints,
			gridPoints: initResult.gridPoints,
			gridOriginOffset
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
