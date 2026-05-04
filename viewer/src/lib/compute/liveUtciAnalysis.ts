import * as THREE from 'three';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { Analysis, AnalysisMetadata, FullDayData, HourStatistics } from '$lib/types/analysis';
import type { SerializedBvhForGpu, UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import { emitComputeTelemetry } from '$lib/compute/telemetry';

/**
 * Read all UTCI slices in a single GPU→CPU transfer instead of 288 serial mapAsync calls.
 * 
 * Uses the pipeline's readUtciBulk method (single mapAsync instead of 288).
 * This reduces PCIe round-trip overhead from 288 × ~1-2ms to 1 × ~1-2ms.
 * 
 * Falls back to per-slice reading if the pipeline doesn't support bulk access.
 */
async function readAllUtciSlices(
	computeManager: ComputeManager,
	params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
		signal?: AbortSignal;
	}
): Promise<Float32Array> {
	const { numPoints, numHours, numMonths, signal } = params;

	// Try bulk readback first (single mapAsync instead of 288).
	const pipeline = computeManager.getPipeline();
	if (pipeline.readUtciBulk) {
		return pipeline.readUtciBulk({ numPoints, numHours, numMonths });
	}

	// Fallback: per-slice reading
	const totalSlices = numMonths * numHours;
	const allUtci = new Float32Array(totalSlices * numPoints);

	for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
		for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
			if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
			const slice = await computeManager.getUtcisForMonthHour({
				monthIndex: monthOffset,
				hourIndex,
				numPoints,
				numMonths,
				numHours
			});
			const sliceIdx = monthOffset * numHours + hourIndex;
			allUtci.set(slice, sliceIdx * numPoints);
		}
		await yieldToMain();
	}

	return allUtci;
}

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

function buildSunVectorsFixtureFromMetadata(params: {
	baseMetadata: AnalysisMetadata;
	numHours: number;
	numMonths: number;
}): { sunVectors: Float32Array; sunAltitudes: Float32Array } | undefined {
	const { baseMetadata, numHours, numMonths } = params;
	const raw = (baseMetadata as unknown as { sun_positions?: unknown }).sun_positions;
	if (!Array.isArray(raw) || raw.length < numHours) return undefined;

	type SunPositionLike = {
		vector?: [number, number, number];
		altitude?: number;
		is_up?: boolean;
	};
	const firstDay = raw as SunPositionLike[];
	const sunVectors = new Float32Array(numMonths * numHours * 3);
	const sunAltitudes = new Float32Array(numMonths * numHours);
	for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
		for (let hour = 0; hour < numHours; hour++) {
			const src = firstDay[hour];
			const vec = src?.vector;
			const base = (monthOffset * numHours + hour) * 3;
			if (vec && vec.length === 3) {
				// Metadata vectors are Z-up (x east, y north, z up); pipeline expects Y-up.
				sunVectors[base] = vec[0];
				sunVectors[base + 1] = vec[2];
				sunVectors[base + 2] = -vec[1];
			}
			const altitudeDeg = Number.isFinite(src?.altitude) ? (src?.altitude as number) : 0;
			const isUp = src?.is_up === true || altitudeDeg > 0;
			sunAltitudes[monthOffset * numHours + hour] = isUp ? (altitudeDeg * Math.PI) / 180 : 0;
		}
	}
	return { sunVectors, sunAltitudes };
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
	 * Precomputed BVH from worker serialization.
	 * Rectangular parity compute requires this to avoid main-thread mesh fallbacks.
	 */
	workerResult?: { serializedBvh: SerializedBvhForGpu; gridPoints?: Float32Array };
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
	 * Number of representative months to compute (1 = single day, 12 = full year).
	 * When 12, uses startMonth=1 and 15th of each month; omit sunVectorsFixture so
	 * ComputeManager computes per-month sun vectors from EPW.
	 */
	numMonths?: number;
}

const PARITY_SAMPLE_HEIGHT_OFFSET_M = 0.9;

export interface LiveUtciAnalysisOptions {
	/**
	 * Underlying compute pipeline implementation.
	 * In production this should be a WebGPU-backed pipeline; in tests a fake can be injected.
	 */
	pipeline: UTCIComputePipeline;
	/**
	 * Optional progress callback during 12-month readback: (completed, total).
	 */
	onProgress?: (completed: number, total: number) => void;
	/**
	 * Optional phase change callback: (phaseName).
	 */
	onPhase?: (phase: string) => void;
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
		workerResult,
		epwContent,
		gridResolution = baseMetadata.grid_size || 2,
		zHeight = 0.9,
		numHours = baseMetadata.hours?.length ?? 24,
		startMonth: startMonthParam,
		numMonths: numMonthsParam
	} = params;

	const numMonths = numMonthsParam ?? 1;
	const startMonth = numMonths > 1 ? 1 : (startMonthParam ?? 8); // 12-month: Jan; single: Aug to match .bin

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

	if (!bounds || !workerResult) {
		throw new Error(
			'createLiveUtciAnalysisFromCompute: rectangular parity path requires analysis bounds and workerResult.serializedBvh.'
		);
	}
	const baseGridHeight = bounds.z ?? zHeight;
	// Python reference intermediates are computed from human sample points
	// above the grid position (pt_count=1 at height/2 ~= 0.9 m).
	const computeGridHeight = baseGridHeight + PARITY_SAMPLE_HEIGHT_OFFSET_M;
	// When numMonths > 1, omit sunVectorsFixture so ComputeManager computes real sun vectors from EPW per month.
	const sunVectorsFixture =
		numMonths === 1
			? buildSunVectorsFixtureFromMetadata({
					baseMetadata,
					numHours,
					numMonths
				})
			: undefined;

	const result = await computeManager.initFromModelAndWeather({
		serializedBvh: workerResult.serializedBvh,
		sunVectorsFixture,
		useRectangularGridFromBounds: true,
		analysisBounds: bounds,
		coordinateSystem,
		gridOriginOffset,
		epwContent,
		gridResolution,
		zHeight: computeGridHeight,
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
		// Store positions at the analysis grid level (not elevated sample-point level)
		// so .bin and live layers align visually and for topology comparisons.
		const wy =
			gridPointsFlat[i * 3 + 1] - normalizationOffset.y - PARITY_SAMPLE_HEIGHT_OFFSET_M;
		const wz = gridPointsFlat[i * 3 + 2] - normalizationOffset.z;
		const [ax, ay, az] = worldToAnalysisCoords(wx, wy, wz, coordinateSystem);
		positions[i * 3] = ax;
		positions[i * 3 + 1] = ay;
		positions[i * 3 + 2] = az;
	}

	// 3. Read back UTCI slices: batch all into a single large readback.
	if (options.onPhase) options.onPhase('readback');
	// Instead of 288 serial mapAsync calls (each adding ~1-2ms of CPU/PCIe latency),
	// we read the entire UTCI buffer in one shot and quantize on CPU.
	const totalSlices = numMonths * numHours;
	const UTCI_STORAGE_SCALE = 100;
	const utciStorage = new Int16Array(totalSlices * effectiveNumPoints);
	const hourStatistics: HourStatistics[] = [];

	let globalMin = Number.POSITIVE_INFINITY;
	let globalMax = Number.NEGATIVE_INFINITY;

	// Read all UTCI results in one mapAsync call instead of 288 serial calls.
	// This eliminates ~300-600ms of PCIe round-trip latency.
	const allUtci = await readAllUtciSlices(computeManager, {
		numPoints: effectiveNumPoints,
		numHours,
		numMonths,
		signal
	});

	for (let sliceIdx = 0; sliceIdx < totalSlices; sliceIdx++) {
		if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
		const base = sliceIdx * effectiveNumPoints;

		let hourMin = Number.POSITIVE_INFINITY;
		let hourMax = Number.NEGATIVE_INFINITY;
		let sum = 0;

		for (let i = 0; i < effectiveNumPoints; i++) {
			// GPU buffer is point-major: allUtci[pointIdx * totalSlices + sliceIdx]
			const value = allUtci[i * totalSlices + sliceIdx];
			if (!Number.isFinite(value)) continue;
			if (value < hourMin) hourMin = value;
			if (value > hourMax) hourMax = value;
			sum += value;
			const encoded = Math.round(value * UTCI_STORAGE_SCALE);
			utciStorage[base + i] = Math.max(-32768, Math.min(32767, encoded));
		}

		if (!Number.isFinite(hourMin) || !Number.isFinite(hourMax)) {
			hourMin = 0;
			hourMax = 0;
		}

		const mean = effectiveNumPoints > 0 ? sum / effectiveNumPoints : 0;

		if (hourMin < globalMin) globalMin = hourMin;
		if (hourMax > globalMax) globalMax = hourMax;

		hourStatistics.push({ min: hourMin, max: hourMax, mean });

		// Report progress per month and yield to keep UI/spinner responsive
		if ((sliceIdx + 1) % numHours === 0) {
			const monthsDone = (sliceIdx + 1) / numHours;
			options.onProgress?.(monthsDone, numMonths);
			await yieldToMain();
		}
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
		num_months: numMonths,
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
		numHours: totalSlices,
		positions,
		utciStorage: {
			buffer: utciStorage,
			numPoints: effectiveNumPoints,
			numSlices: totalSlices,
			scale: UTCI_STORAGE_SCALE
		}
	};

	const analysis: Analysis = {
		metadata: liveMetadata,
		data: liveData
	};

	// Non-breaking debug marker for downstream consumers if needed.
	(analysis as any).__source = 'webgpu';
	// Debug-only: retain compute-space sample points (Y-up world) used for solar/sky ray origin.
	(analysis as any).__computeGridPointsWorld = Array.from(gridPointsFlat);

	return analysis;
}
