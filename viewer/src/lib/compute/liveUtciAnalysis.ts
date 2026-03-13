import * as THREE from 'three';
import { generateGridFromMesh } from '$lib/compute/grid-generator';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { Analysis, AnalysisMetadata, FullDayData, HourStatistics } from '$lib/types/analysis';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';

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
	 * Simplified mesh representing the sampling surface for grid generation.
	 * This should be aligned with the model's world coordinates.
	 */
	sampleMesh: THREE.Mesh;
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
	 * Defaults to 1.5m for pedestrian head height.
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
		epwContent,
		gridResolution = baseMetadata.grid_size || 2,
		zHeight = 1.5,
		numHours = baseMetadata.hours?.length ?? 24,
		startMonth = 8 // August 15th to match existing .bin analyses
	} = params;

	const numMonths = 1;

	const computeManager = new ComputeManager(options.pipeline, {
		numMonths,
		numHoursPerDay: numHours,
		startMonth
	});

	// 1. Initialize the compute pipeline from the sampling mesh and EPW
	const { numPoints } = await computeManager.initFromModelAndWeather({
		mesh: sampleMesh,
		epwContent,
		gridResolution,
		zHeight
	});

	// 2. Rebuild the grid positions so we can populate Analysis.data.positions.
	//    This mirrors the internal grid generation used by ComputeManager.
	const grid = generateGridFromMesh(sampleMesh, gridResolution, zHeight);
	if (grid.points.length !== numPoints) {
		// This should not happen in normal operation; log and clamp if it does.
		console.warn(
			`[liveUtciAnalysis] Grid/compute mismatch for ${analysisId}: ` +
				`${grid.points.length} points from grid, ${numPoints} from compute. Using the smaller count.`
		);
	}

	const effectiveNumPoints = Math.min(grid.points.length, numPoints);
	const positions = new Float32Array(effectiveNumPoints * 3);
	const coordinateSystem =
		(baseMetadata.coordinate_system as 'xy_ground' | 'xz_ground') ?? 'xy_ground';

	// Live grid points are generated from a model that has already had the
	// normalization offset applied. To keep parity with .bin analyses (which
	// store positions in the original analysis frame and rely on
	// buildUtciGridLayout to apply normalization once), we remove the same
	// normalization offset here before converting back to analysis coordinates.
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
		const p = grid.points[i];
		const worldX = p.x - normalizationOffset.x;
		const worldY = p.y - normalizationOffset.y;
		const worldZ = p.z - normalizationOffset.z;
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

		for (let i = 0; i < effectiveNumPoints; i++) {
			const value = effectiveSlice[i];
			if (!Number.isFinite(value)) continue;
			if (value < hourMin) hourMin = value;
			if (value > hourMax) hourMax = value;
			sum += value;
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

