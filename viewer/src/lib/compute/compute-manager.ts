import type { Analysis } from '$lib/types/analysis';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { parseEPW } from '$lib/compute/epw-parser';
import { getSunVectors } from '$lib/compute/sunpath';
import { getTregenzaDome } from '$lib/compute/tregenza';
import { generateGridFromMesh } from '$lib/compute/grid-generator';
import type * as THREE from 'three';

/**
 * Rotate a direction vector from the Python/ladybug Z-up convention
 * (X = East, Y = North, Z = Up) into the Three.js world frame
 * (X = East, Y = Up, Z = North, Y-up).
 */
function rotateZUpToYUp(vec: [number, number, number]): [number, number, number] {
	const [x, y, z] = vec;
	return [x, z, y];
}

export interface ComputeManagerConfig {
	numMonths: number;
	numHoursPerDay: number;
	/**
	 * Starting month index (1–12) for representative-day sampling.
	 * When numMonths > 1, subsequent months are sampled as startMonth + i.
	 */
	startMonth: number;
	/**
	 * Representative day of month used for sun/EPW sampling.
	 */
	representativeDay: number;
}

const DEFAULT_CONFIG: ComputeManagerConfig = {
	numMonths: 12,
	numHoursPerDay: 24,
	startMonth: 1,
	representativeDay: 15
};

/**
 * High-level orchestrator for preparing inputs for the UTCI GPU pipeline.
 * 
 * This class is intentionally WebGPU-agnostic; it only depends on the
 * UTCIComputePipeline interface so tests can inject a fake implementation.
 */
export class ComputeManager {
	private readonly pipeline: UTCIComputePipeline;
	private readonly config: ComputeManagerConfig;

	constructor(pipeline: UTCIComputePipeline, config: Partial<ComputeManagerConfig> = {}) {
		this.pipeline = pipeline;
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	/**
	 * Prepare GPU buffers and run the full compute pipeline given a Three.js
	 * mesh (for grid generation), an EPW file, and a representative day (15th).
	 * 
	 * The concrete WebGPU pipeline decides how to map these arrays into GPU
	 * buffers; this method only prepares packed Float32Array inputs.
	 */
	async initFromModelAndWeather(params: {
		mesh: THREE.Mesh;
		epwContent: string;
		gridResolution: number;
		zHeight: number;
	}): Promise<{
		numPoints: number;
		numMonths: number;
		numHours: number;
	}> {
		const { mesh, epwContent, gridResolution, zHeight } = params;
		const numMonths = this.config.numMonths;
		const numHours = this.config.numHoursPerDay;
		const startMonth = this.config.startMonth;
		const representativeDay = this.config.representativeDay;

		// 1. Generate grid from the mesh
		const grid = generateGridFromMesh(mesh, gridResolution, zHeight);
		const numPoints = grid.points.length;

		const gridPoints = new Float32Array(numPoints * 3);
		for (let i = 0; i < numPoints; i++) {
			const p = grid.points[i];
			gridPoints[i * 3] = p.x;
			gridPoints[i * 3 + 1] = p.y;
			gridPoints[i * 3 + 2] = p.z;
		}

		// 2. Parse EPW and compute sun vectors for 12 representative days (15th)
		const epw = parseEPW(epwContent);
		const location = {
			lat: epw.location.lat,
			lon: epw.location.lon,
			timezone: epw.location.timezone
		};

		const sunVectors = new Float32Array(numMonths * numHours * 3);
		for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
			const month = Math.min(12, Math.max(1, startMonth + monthOffset));
			const day = representativeDay;
			const { sunVectors: dayVectors } = getSunVectors(location, month, day);
			for (let hour = 0; hour < numHours; hour++) {
				const idx = monthOffset * numHours + hour;
				const v = rotateZUpToYUp(dayVectors[hour]);
				sunVectors[idx * 3] = v[0];
				sunVectors[idx * 3 + 1] = v[1];
				sunVectors[idx * 3 + 2] = v[2];
			}
		}

		// 3. Pack per-hour weather samples (air temp, MRT_lw placeholder, wind, RH)
		// For now we approximate longwave MRT with air temperature; Phase 2 can
		// refine this based on epw.horizInfrared and SolarCal.
		const weatherStride = 4;
		const weather = new Float32Array(numMonths * numHours * weatherStride);
		for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
			const month = Math.min(12, Math.max(1, startMonth + monthOffset));
			const day = representativeDay;
			for (let hour = 0; hour < numHours; hour++) {
				const idx = monthOffset * numHours + hour;
				const hourData = epw.getHourData(month, day, hour + 1);
				if (!hourData) continue;

				const base = idx * weatherStride;
				weather[base] = hourData.dryBulb; // air_temp
				weather[base + 1] = hourData.dryBulb; // mrt_longwave (approx for now)
				weather[base + 2] = hourData.windSpeed;
				weather[base + 3] = hourData.relHumidity;
			}
		}

		// 4. Upload to GPU pipeline (or fake implementation in tests)
		await this.pipeline.uploadStaticData({
			gridPoints,
			sunVectors,
			weather
		});

		await this.pipeline.runAll({
			numPoints,
			numHours,
			numMonths
		});

		return {
			numPoints,
			numMonths,
			numHours
		};
	}

	/**
	 * Read back a UTCI slice for a given (month, hour) combination.
	 * This delegates to the underlying pipeline implementation.
	 */
	async getUtcisForMonthHour(params: {
		monthIndex: number;
		hourIndex: number;
		numPoints: number;
		numMonths: number;
		numHours: number;
	}): Promise<Float32Array> {
		return this.pipeline.readUtcisSlice(params);
	}
}

