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
	 * Simplified mesh for BVH (main-thread fallback when worker fails).
	 * Grid is always from analysis bounds; mesh is only used to build the BVH for exposure.
	 */
	sampleMesh?: THREE.Mesh;
	/**
	 * BVH from Web Worker (avoids main-thread merge). Grid is built from analysis bounds.
	 * When set, sampleMesh is not used.
	 */
	workerResult?: { serializedBvh: import('$lib/compute/gpu-pipeline').SerializedBvhForGpu; gridPoints?: Float32Array };
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
	/**
	 * Use a rectangular grid from analysis bounds (same-grid-as-.bin). Default true when bounds exist.
	 * @deprecated Grid is always from bounds; this is kept for backward compatibility.
	 */
	useRectangularGridFromBounds?: boolean;
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

	const coordinateSystem = (baseMetadata.coordinate_system as 'xy_ground' | 'xz_ground') ?? 'xy_ground';
	// Normalization offset: displayed model is at this offset in viewer world. Rect grid from bounds is in analysis-origin viewer space; add this so grid matches BVH space when normalization is on.
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
	const gridOriginOffset =
		normalizationOffset.lengthSq() > 0.001
			? { x: normalizationOffset.x, y: normalizationOffset.y, z: normalizationOffset.z }
			: undefined;

	// 1. Initialize the compute pipeline. When using rect grid with normalization, gridOriginOffset shifts grid into BVH (viewer world) space.
	const initStartedAt = performance.now();
	const bounds = baseMetadata.bounds as
		| { x_min: number; x_max: number; y_min: number; y_max: number; z?: number }
		| undefined;

	if (!bounds || (!workerResult && !sampleMesh)) {
		throw new Error(
			'createLiveUtciAnalysisFromCompute: analysis metadata must have bounds and provide workerResult (BVH) or sampleMesh for BVH.'
		);
	}

	const result = await computeManager.initFromModelAndWeather({
		...(workerResult ? { serializedBvh: workerResult.serializedBvh } : {}),
		...(sampleMesh && !workerResult ? { mesh: sampleMesh } : {}),
		useRectangularGridFromBounds: true,
		analysisBounds: bounds,
		coordinateSystem,
		gridOriginOffset,
		epwContent,
		gridResolution,
		zHeight,
		signal
	});
	emitComputeTelemetry('pipeline.upload.done', {
		ms: performance.now() - initStartedAt,
		data: { numPoints: result.numPoints, numHours, numMonths }
	});

	const { numPoints, gridPoints: gridPointsFlat } = result;
	const effectiveNumPoints = numPoints;
	const positions = new Float32Array(effectiveNumPoints * 3);

	// Stored positions: pipeline grid is in viewer world (BVH space). Store analysis = worldToAnalysisCoords(viewer - offset) so display (transform+offset) is correct.
	for (let i = 0; i < effectiveNumPoints; i++) {
		const wx = gridPointsFlat[i * 3] - normalizationOffset.x;
		const wy = gridPointsFlat[i * 3 + 1] - normalizationOffset.y;
		const wz = gridPointsFlat[i * 3 + 2] - normalizationOffset.z;
		const [ax, ay, az] = worldToAnalysisCoords(wx, wy, wz, coordinateSystem);
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
