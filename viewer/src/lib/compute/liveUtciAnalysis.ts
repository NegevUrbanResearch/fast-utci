import * as THREE from 'three';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { Analysis, AnalysisMetadata, FullDayData, HourStatistics } from '$lib/types/analysis';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import { emitComputeTelemetry } from '$lib/compute/telemetry';

function worldToAnalysisCoords(
	x: number,
	y: number,
	z: number,
	coordinateSystem: 'xy_ground' | 'xz_ground'
): [number, number, number] {
	if (coordinateSystem === 'xy_ground') {
		const Xw = x;
		const Yw = y;
		const Zw = z;
		const x_orig = Xw;
		const z_orig = Yw;
		const y_orig = -Zw;
		return [x_orig, y_orig, z_orig];
	}
	return [x, y, z];
}

export interface LiveUtciAnalysisParams {
	/**
	 * Identifier for logging and debugging; typically matches the .bin analysis id.
	 */
	analysisId: string;
	/**
	 * Metadata from the base .bin-backed analysis.
	 * Used as a template for project, model file, hours, and location fields.
	 */
	baseMetadata: AnalysisMetadata;
	/**
	 * Simplified mesh for grid generation (main-thread path).
	 * Omit when using workerResult (merge + BVH + grid computed in a worker).
	 */
	sampleMesh?: THREE.Mesh;
	/**
	 * Pre-computed grid and BVH from a Web Worker (avoids main-thread freeze on large models).
	 * When set, sampleMesh is not used for init.
	 */
	workerResult?: { gridPoints: Float32Array; serializedBvh: import('$lib/compute/gpu-pipeline').SerializedBvhForGpu };
	/**
	 * Raw EPW file contents for the project location.
	 */
	epwContent: string;
	/**
	 * Target grid resolution in meters. Defaults to baseMetadata.grid_size.
	 */
	gridResolution?: number;
	/**
	 * Height offset above the sampling surface (in meters).
	 * Defaults to 0.9m (parity-aligned debug default).
	 */
	zHeight?: number;
	/**
	 * Optional override for the number of analysis hours.
	 * Defaults to the baseMetadata.hours length when available, otherwise 24.
	 */
	numHours?: number;
	/**
	 * Optional override for the representative month index (1–12).
	 * Defaults to 8 (August) to match existing .bin analyses.
	 */
	startMonth?: number;
}

export interface LiveUtciAnalysisOptions {
	/**
	 * Underlying compute pipeline implementation.
	 * In production this should be a WebGPU-backed pipeline; in tests a fake can be injected.
	 */
	pipeline: UTCIComputePipeline;
	/**
	 * Optional AbortSignal to cancel the run when the user switches project/model.
	 * When aborted, the promise rejects with DOMException('Aborted', 'AbortError').
	 */
	signal?: AbortSignal;
}

function yieldToMain(): Promise<void> {
	if (typeof requestAnimationFrame !== 'undefined') {
		return new Promise((resolve) => requestAnimationFrame(() => resolve()));
	}
	return Promise.resolve();
}

/**
 * Create an Analysis-like object backed by live UTCI values computed via the
 * UTCI compute pipeline. The resulting structure is intentionally compatible
 * with the existing pointCloudService helpers.
 */
export async function createLiveUtciAnalysisFromCompute(
	params: LiveUtciAnalysisParams,
	options: LiveUtciAnalysisOptions
): Promise<Analysis> {
	const {
		analysisId,
		baseMetadata,
		sampleMesh,
		workerResult,
		epwContent,
		gridResolution = baseMetadata.grid_size || 2,
		zHeight = 0.9,
		numHours = baseMetadata.hours?.length ?? 24,
		startMonth = 8 // August 15th to match existing .bin analyses
	} = params;

	const numMonths = 1;

	const computeManager = new ComputeManager(options.pipeline, {
		numMonths,
		numHoursPerDay: numHours,
		startMonth
	});

	const signal = options.signal;
	if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

	// 1. Initialize the compute pipeline (worker path: grid+BVH already done; main-thread: from mesh)
	const initStartedAt = performance.now();
	const result = workerResult
		? await computeManager.initFromModelAndWeather({
				gridPoints: workerResult.gridPoints,
				serializedBvh: workerResult.serializedBvh,
				epwContent,
				gridResolution,
				zHeight,
				signal
			})
		: sampleMesh
			? await computeManager.initFromModelAndWeather({
					mesh: sampleMesh,
					epwContent,
					gridResolution,
					zHeight,
					signal
				})
			: (() => {
					throw new Error('createLiveUtciAnalysisFromCompute: provide sampleMesh or workerResult');
				})();
	emitComputeTelemetry('pipeline.upload.done', {
		ms: performance.now() - initStartedAt,
		data: { numPoints: result.numPoints, numHours, numMonths }
	});

	const { numPoints, gridPoints: gridPointsFlat } = result;
	const effectiveNumPoints = numPoints;
	const positions = new Float32Array(effectiveNumPoints * 3);
	const coordinateSystem =
		(baseMetadata.coordinate_system as 'xy_ground' | 'xz_ground') ?? 'xy_ground';

	// 2. Use the grid points from the compute pipeline (no second grid generation).
	//    Apply normalization offset and convert to analysis coordinates for parity with .bin analyses.
	let normalizationOffset = new THREE.Vector3(0, 0, 0);
	if (isNormalizationEnabled()) {
		const scenarioOrigin = calculateScenarioOrigin(baseMetadata as any);
		const anchorOffset = getAnchorOffset();

		let transformedOrigin: THREE.Vector3;
		if (coordinateSystem === 'xy_ground') {
			transformedOrigin = new THREE.Vector3(
				scenarioOrigin.x,
				scenarioOrigin.z,
				-scenarioOrigin.y
			);
		} else {
			transformedOrigin = scenarioOrigin.clone();
		}

		normalizationOffset = anchorOffset.clone().sub(transformedOrigin);
	}

	for (let i = 0; i < effectiveNumPoints; i++) {
		const worldX = gridPointsFlat[i * 3] - normalizationOffset.x;
		const worldY = gridPointsFlat[i * 3 + 1] - normalizationOffset.y;
		const worldZ = gridPointsFlat[i * 3 + 2] - normalizationOffset.z;
		const [ax, ay, az] = worldToAnalysisCoords(worldX, worldY, worldZ, coordinateSystem);
		positions[i * 3] = ax;
		positions[i * 3 + 1] = ay;
		positions[i * 3 + 2] = az;
	}

	// 3. Read back UTCI slices for each analysis hour.
	const utciByHour: Float32Array[] = [];
	const hourStatistics: HourStatistics[] = [];

	let globalMin = Number.POSITIVE_INFINITY;
	let globalMax = Number.NEGATIVE_INFINITY;

	for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
		if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
		const hourReadStartedAt = performance.now();
		const slice = await computeManager.getUtcisForMonthHour({
			monthIndex: 0,
			hourIndex,
			numPoints: effectiveNumPoints,
			numMonths,
			numHours
		});

		// Clamp slice length if the pipeline returned more data than expected.
		const effectiveSlice =
			slice.length === effectiveNumPoints ? slice : slice.subarray(0, effectiveNumPoints);

		let hourMin = Number.POSITIVE_INFINITY;
		let hourMax = Number.NEGATIVE_INFINITY;
		let sum = 0;
		const CHUNK_SIZE = 20_000;

		for (let i = 0; i < effectiveNumPoints; i++) {
			const value = effectiveSlice[i];
			if (!Number.isFinite(value)) continue;
			if (value < hourMin) hourMin = value;
			if (value > hourMax) hourMax = value;
			sum += value;
			if (i > 0 && i % CHUNK_SIZE === 0) {
				if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
				await yieldToMain();
			}
		}

		if (!Number.isFinite(hourMin) || !Number.isFinite(hourMax)) {
			hourMin = 0;
			hourMax = 0;
		}

		const mean = effectiveNumPoints > 0 ? sum / effectiveNumPoints : 0;

		if (hourMin < globalMin) globalMin = hourMin;
		if (hourMax > globalMax) globalMax = hourMax;

		hourStatistics.push({
			min: hourMin,
			max: hourMax,
			mean
		});

		utciByHour.push(effectiveSlice);
		emitComputeTelemetry('utci.readback.done', {
			ms: performance.now() - hourReadStartedAt,
			data: { hourIndex, numPoints: effectiveNumPoints }
		});
	}

	if (!Number.isFinite(globalMin) || !Number.isFinite(globalMax) || globalMin === globalMax) {
		// Fallback to a small artificial range to keep color mapping stable
		globalMin = 0;
		globalMax = 1;
	}

	const hours =
		baseMetadata.analysis_type === 'full_day' && baseMetadata.hours && baseMetadata.hours.length
			? baseMetadata.hours.slice(0, numHours)
			: Array.from({ length: numHours }, (_, i) =>
					`${i.toString().padStart(2, '0')}:00`
			  );

	const liveMetadata: AnalysisMetadata = {
		...baseMetadata,
		analysis_type: 'full_day',
		num_positions: effectiveNumPoints,
		hours,
		utci_range: {
			min: globalMin,
			max: globalMax
		},
		grid_size: gridResolution,
		coordinate_system: coordinateSystem,
		hour_statistics: hourStatistics,
		has_shading_index: false,
		shading_index_range: undefined
	};

	const liveData: FullDayData = {
		numPositions: effectiveNumPoints,
		numHours,
		positions,
		utciByHour
	};

	const analysis: Analysis = {
		metadata: liveMetadata,
		data: liveData
	};

	// Non-breaking debug marker for downstream consumers if needed.
	(analysis as any).__source = 'webgpu';

	return analysis;
}
