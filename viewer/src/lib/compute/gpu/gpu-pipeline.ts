import type { OnDemandOutputFormat } from '$lib/compute/on-demand/onDemandOutputFormat';
import type { OnDemandRuntimeDiagnostics } from '$lib/compute/on-demand/onDemandDiagnostics';
import type { SelectedHourOutputHandle } from '$lib/compute/gpu/selectedHourOutputHandle';
import type { ExposureSchedulingOptions } from '$lib/compute/gpu/exposureScheduling';

export interface PipelineConfig {
	numPoints: number;
	numHours: number;
	numMonths: number;
	solarExposureBufferSize: number;
	skyExposureBufferSize: number;
	utciResultBufferSize: number;
	mrtResultBufferSize: number;
}

/**
 * Create a static configuration description for the UTCI WebGPU pipeline.
 * This is pure arithmetic and safe to use in vitest (no WebGPU required).
 */
export function createPipelineConfig(params: {
	numPoints: number;
	numHours: number;
	numMonths: number;
}): PipelineConfig {
	const { numPoints, numHours, numMonths } = params;

	if (numPoints <= 0 || numHours <= 0 || numMonths <= 0) {
		throw new Error('numPoints, numHours and numMonths must all be positive');
	}

	const totalTimeSteps = numHours * numMonths;

	return {
		numPoints,
		numHours,
		numMonths,
		// Bit-packed: 1 bit per point × time step, stored as u32 words
		solarExposureBufferSize: Math.ceil((numPoints * totalTimeSteps) / 32) * 4,
		// Single sky view factor per point
		skyExposureBufferSize: numPoints * 4,
		// f32 per point × hour × month
		utciResultBufferSize: numPoints * totalTimeSteps * 4,
		// f32 per point × hour × month (same layout as UTCI for MRT readback)
		mrtResultBufferSize: numPoints * totalTimeSteps * 4
	};
}

export interface DispatchDimensions {
	x: number;
	y: number;
}

export const MAX_WEBGPU_WORKGROUPS_PER_DIMENSION = 65_535;

export interface PointDispatchChunk {
	pointOffset: number;
	pointCount: number;
	workgroupsX: number;
}

/**
 * Compute flat UTCI index for point-major layout:
 * flatIndex = pointIndex * totalTimeSteps + timeIndex
 */
export function getUtciFlatIndex(
	pointIndex: number,
	timeIndex: number,
	totalTimeSteps: number
): number {
	if (pointIndex < 0 || timeIndex < 0 || totalTimeSteps <= 0) {
		throw new Error('pointIndex and timeIndex must be >= 0, totalTimeSteps must be > 0');
	}
	return pointIndex * totalTimeSteps + timeIndex;
}

/**
 * Calculate 2D dispatch dimensions for a (points, hours×months) layout.
 * Workgroups are sized along X; Y is the time dimension.
 */
export function calculateDispatch(
	numPoints: number,
	numHours: number,
	numMonths: number,
	workgroupSize: number
): DispatchDimensions {
	if (numPoints <= 0 || numHours <= 0 || numMonths <= 0 || workgroupSize <= 0) {
		throw new Error('numPoints, numHours, numMonths and workgroupSize must all be positive');
	}

	const totalTimeSteps = numHours * numMonths;

	return {
		x: Math.ceil(numPoints / workgroupSize),
		y: totalTimeSteps
	};
}

export function createPointDispatchChunks(
	numPoints: number,
	workgroupSize: number,
	maxWorkgroupsX: number = MAX_WEBGPU_WORKGROUPS_PER_DIMENSION
): PointDispatchChunk[] {
	if (numPoints <= 0 || workgroupSize <= 0 || maxWorkgroupsX <= 0) {
		throw new Error('numPoints, workgroupSize and maxWorkgroupsX must all be positive');
	}

	const maxPointsPerChunk = workgroupSize * maxWorkgroupsX;
	const chunks: PointDispatchChunk[] = [];
	for (let pointOffset = 0; pointOffset < numPoints; pointOffset += maxPointsPerChunk) {
		const pointCount = Math.min(maxPointsPerChunk, numPoints - pointOffset);
		chunks.push({
			pointOffset,
			pointCount,
			workgroupsX: Math.ceil(pointCount / workgroupSize)
		});
	}
	return chunks;
}

/**
 * Minimal interface for the UTCI GPU compute pipeline orchestrator.
 * 
 * The concrete WebGPU implementation lives in browser-only code; tests can
 * provide a fake implementation that conforms to this interface.
 */
/** Pre-serialized BVH from a worker (merge + BVH + grid off main thread). */
export interface SerializedBvhForGpu {
	bvhNodeBuffer: ArrayBuffer;
	bvhIndexBuffer: ArrayBuffer;
	vertexBuffer: Float32Array;
	indexBuffer: Uint32Array;
}

export interface ExposurePrecomputeParams {
	numPoints: number;
	numHours: number;
	numMonths: number;
	exposureScheduling?: ExposureSchedulingOptions;
	signal?: AbortSignal;
}

export interface RunUtciForTimeIndexParams {
	timeIndex: number;
	numPoints: number;
	numHours: number;
	numMonths: number;
	format: OnDemandOutputFormat;
}

export interface RunUtciRangeSummaryForTimeIndexParams extends RunUtciForTimeIndexParams {
	format: 'f32-utci';
	signal?: AbortSignal;
}

export interface OnDemandUtciOutput {
	format: OnDemandOutputFormat;
	numPoints: number;
	timeIndex: number;
	gpuBuffer?: unknown;
	gpuOutputHandle?: SelectedHourOutputHandle;
	outputBytes?: number;
	debugLabel?: string;
}

export interface UtciRangeSummary {
	timeIndex: number;
	range: { min: number; max: number } | null;
	validCount: number;
	readbackBytes: number;
	reductionPassCount: number;
	debugLabel: 'webgpu-on-demand-f32-utci-range-summary';
}

export interface UTCIComputePipeline {
	/**
	 * Upload static data required for all dispatches.
	 * These arrays must match the sizes described by PipelineConfig.
	 */
	uploadStaticData(params: {
		gridPoints: Float32Array; // vec3<f32>[numPoints]
		sunVectors: Float32Array; // vec3<f32>[numMonths * numHours]
		sunAltitudes?: Float32Array; // f32[numMonths * numHours] in radians, for MRT shader
		weather: Float32Array; // packed per-hour weather structs
		domeVectors?: Float32Array; // Tregenza 145 patches, Y-up, for sky exposure
		domeWeights?: Float32Array; // 145 weights
		mesh?: { geometry: unknown }; // optional; when set, BVH is serialized from mesh for GPU exposure
		serializedBvh?: SerializedBvhForGpu; // when set (e.g. from worker), BVH is used directly; mesh not needed for BVH
	}): Promise<void>;

	/**
	 * Run the three compute passes (solar exposure, sky exposure, MRT+UTCI)
	 * for the configured number of points / hours / months.
	 */
	runAll(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
		workgroupSize?: number;
	}): Promise<void>;

	/**
	 * Read back a UTCI slice for the given (month, hour) as a Float32Array
	 * of length numPoints. In the real implementation this will map a GPU
	 * buffer and copy the required range into CPU memory.
	 */
	readUtcisSlice(params: {
		monthIndex: number; // 0-11
		hourIndex: number; // 0-23
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array>;

	runExposurePrecompute?(params: ExposurePrecomputeParams): Promise<void>;

	runUtciForTimeIndex?(params: RunUtciForTimeIndexParams): Promise<OnDemandUtciOutput>;

	runUtciRangeSummaryForTimeIndex?(
		params: RunUtciRangeSummaryForTimeIndexParams
	): Promise<UtciRangeSummary>;

	readOnDemandUtciForDebug?(params: { numPoints: number }): Promise<Float32Array>;

	getOnDemandDiagnostics?(): OnDemandRuntimeDiagnostics;

	getDeviceForDebug?(): GPUDevice;

	/**
	 * Read the entire UTCI results buffer in a single GPU→CPU transfer.
	 * Returns a flat Float32Array of length numPoints × numHours × numMonths,
	 * in point-major layout: [p0_t0, p0_t1, ..., p0_tN, p1_t0, ...].
	 * 
	 * This eliminates 288 serial mapAsync round-trips by doing one bulk copy.
	 * Optional; falls back to per-slice reading if not implemented.
	 */
	readUtciBulk?(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array>;

	/**
	 * Read full solar exposure buffer (point-major: [p0_h0..p0_h23, p1_h0..], one month only).
	 * Optional; only WebGPU implementation provides this (for intermediate parity).
	 */
	readSolarExposureFull?(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array>;

	/**
	 * Read full sky exposure buffer (one value per point).
	 * Optional; only WebGPU implementation provides this (for intermediate parity).
	 */
	readSkyExposure?(params: { numPoints: number }): Promise<Float32Array>;

	/**
	 * Read full MRT buffer (point-major: [p0_h0..p0_h23, p1_h0..], one month).
	 * Optional; only WebGPU implementation provides this (for intermediate parity).
	 */
	readMrtFull?(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array>;

	/**
	 * Read SolarCal MRT component buffers (point-major, one month).
	 * Optional; provided by the WebGPU implementation for parity diagnostics.
	 */
	readMrtComponentsFull?(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<{
		shortErf: Float32Array;
		longErf: Float32Array;
		shortDmrt: Float32Array;
		longDmrt: Float32Array;
	}>;

	/**
	 * Whether this pipeline instance can expose MRT component diagnostics.
	 * Adapters with low storage-buffer limits may disable this path.
	 */
	supportsMrtComponentDiagnostics?(): boolean;

	/**
	 * Return last-uploaded sun vector samples (hours 0, 12, 23) for debugging exposure zeros.
	 * Optional; only WebGPU implementation provides this.
	 */
	getSunVectorSamples?(): number[] | null;

	/**
	 * Return first N hours of uploaded weather as objects (air_temp, direct_normal, etc.) for parity comparison.
	 * Optional; only WebGPU implementation provides this.
	 */
	getWeatherSample?(numHours?: number): Array<{
		air_temp: number;
		direct_normal: number;
		diffuse_horizontal: number;
		horiz_infrared: number;
		wind_speed: number;
		rel_humidity: number;
	}>;

	/**
	 * Release GPU resources. Call before discarding the pipeline (e.g. when
	 * switching projects or on page unload). Optional so test fakes can omit it.
	 */
	dispose?(): void;
}

