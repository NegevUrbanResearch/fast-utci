import type { Analysis } from '$lib/types/analysis';
import type { OnDemandRuntimeDiagnostics } from '$lib/compute/onDemandDiagnostics';
import type {
	ExposurePrecomputeParams,
	OnDemandUtciOutput,
	RunUtciForTimeIndexParams,
	SerializedBvhForGpu,
	UTCIComputePipeline
} from '$lib/compute/gpu-pipeline';
import { parseEPW } from '$lib/compute/epw-parser';
import { getSunVectors } from '$lib/compute/sunpath';
import { getTregenzaDome } from '$lib/compute/tregenza';
import { canonicalGridPoints } from '$lib/parity/gridCanonical';

/**
 * Rotate a direction vector from the Python/ladybug Z-up convention
 * (X = East, Y = North, Z = Up) into the Three.js world frame
 * (X = East, Y = Up, Z = North, Y-up).
 */
function rotateZUpToYUp(vec: [number, number, number]): [number, number, number] {
	const [x, y, z] = vec;
	return [x, z, -y];
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

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

function previousCalendarDay(month: number, day: number): { month: number; day: number } {
	if (day > 1) {
		return { month, day: day - 1 };
	}
	const prevMonth = month === 1 ? 12 : month - 1;
	return { month: prevMonth, day: DAYS_IN_MONTH[prevMonth - 1] };
}

/** Yield to the event loop so the UI can update; avoids long main-thread freezes. */
function yieldToMain(): Promise<void> {
	if (typeof requestAnimationFrame !== 'undefined') {
		return new Promise((resolve) => requestAnimationFrame(() => resolve()));
	}
	return Promise.resolve();
}

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
	 * Prepare GPU buffers and run the full compute pipeline.
	 * Grid is always built from analysis bounds; BVH comes from mesh (main-thread) or serializedBvh (worker).
	 */
	async initFromModelAndWeather(params: {
		serializedBvh?: SerializedBvhForGpu;
		sunVectorsFixture?: {
			sunVectors: Float32Array;
			sunAltitudes?: Float32Array;
		};
		epwContent: string;
		gridResolution: number;
		zHeight: number;
		signal?: AbortSignal;
		useRectangularGridFromBounds: boolean;
		analysisBounds: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number };
		coordinateSystem?: 'xy_ground' | 'xz_ground';
		/** Add to each grid point so rays use the same viewer world as the BVH (displayed model at this offset). */
		gridOriginOffset?: { x: number; y: number; z: number };
	}): Promise<{
		numPoints: number;
		numMonths: number;
		numHours: number;
		gridPoints: Float32Array;
	}> {
		const {
			serializedBvh,
			sunVectorsFixture,
			epwContent,
			gridResolution,
			zHeight,
			signal,
			useRectangularGridFromBounds,
			analysisBounds,
			coordinateSystem = 'xy_ground',
			gridOriginOffset
		} = params;
		const numMonths = this.config.numMonths;
		const numHours = this.config.numHoursPerDay;
		const startMonth = this.config.startMonth;
		const representativeDay = this.config.representativeDay;

		let numPoints: number;
		let gridPoints: Float32Array;

		if (!useRectangularGridFromBounds || !analysisBounds || !serializedBvh) {
			throw new Error(
				'initFromModelAndWeather: rectangular parity path requires useRectangularGridFromBounds=true, analysisBounds, and serializedBvh.'
			);
		}
		const canonicalGrid = canonicalGridPoints({
			bounds: analysisBounds,
			gridSize: gridResolution,
			coordinateSystem,
			zHeight,
			originOffset: gridOriginOffset
		});
		numPoints = canonicalGrid.numPoints;
		if (numPoints === 0) {
			throw new Error('Rectangular grid from bounds produced 0 points; check analysis bounds and gridResolution.');
		}
		gridPoints = canonicalGrid.points;

		// 2. Parse EPW and compute sun vectors for 12 representative days (15th)
		const epw = parseEPW(epwContent);
		const location = {
			lat: epw.location.lat,
			lon: epw.location.lon,
			timezone: epw.location.timezone
		};

		let sunVectors: Float32Array;
		let sunAltitudes: Float32Array;
		if (sunVectorsFixture) {
			const expectedVecLen = numMonths * numHours * 3;
			if (sunVectorsFixture.sunVectors.length !== expectedVecLen) {
				throw new Error(
					`sunVectorsFixture length mismatch: expected ${expectedVecLen}, got ${sunVectorsFixture.sunVectors.length}`
				);
			}
			sunVectors = sunVectorsFixture.sunVectors;
			if (sunVectorsFixture.sunAltitudes) {
				const expectedAltLen = numMonths * numHours;
				if (sunVectorsFixture.sunAltitudes.length !== expectedAltLen) {
					throw new Error(
						`sunAltitudes fixture length mismatch: expected ${expectedAltLen}, got ${sunVectorsFixture.sunAltitudes.length}`
					);
				}
				sunAltitudes = sunVectorsFixture.sunAltitudes;
			} else {
				sunAltitudes = new Float32Array(numMonths * numHours);
			}
		} else {
			sunVectors = new Float32Array(numMonths * numHours * 3);
			sunAltitudes = new Float32Array(numMonths * numHours);
			for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
				const month = Math.min(12, Math.max(1, startMonth + monthOffset));
				const day = representativeDay;
				const { sunVectors: dayVectors, altitudes: dayAltitudes } = getSunVectors(location, month, day);
				if (dayVectors.length < numHours || dayAltitudes.length < numHours) {
					throw new Error(
						`Sun vector contract mismatch for month=${month}, day=${day}: expected at least ${numHours} entries, got vectors=${dayVectors.length}, altitudes=${dayAltitudes.length}`
					);
				}
				for (let hour = 0; hour < numHours; hour++) {
					const idx = monthOffset * numHours + hour;
					const v = rotateZUpToYUp(dayVectors[hour]);
					sunVectors[idx * 3] = v[0];
					sunVectors[idx * 3 + 1] = v[1];
					sunVectors[idx * 3 + 2] = v[2];
					sunAltitudes[idx] = (dayAltitudes[hour] * Math.PI) / 180; // radians for shader
				}
			}
		}

		// 3. Pack per-hour weather samples.
		//
		// Layout (must stay in sync with WeatherSample in mrt_utci.wgsl):
		//   0: air_temp (°C)
		//   1: mrt_longwave (°C) – currently approximated as air temp
		//   2: wind_speed (m/s)
		//   3: rel_humidity (%)
		//   4: direct_normal (W/m²)
		//   5: diffuse_horizontal (W/m²)
		//   6: horiz_infrared (W/m²)
		const weatherStride = 7;
		const weather = new Float32Array(numMonths * numHours * weatherStride);
		for (let monthOffset = 0; monthOffset < numMonths; monthOffset++) {
			const month = Math.min(12, Math.max(1, startMonth + monthOffset));
			const day = representativeDay;
			for (let hour = 0; hour < numHours; hour++) {
				const idx = monthOffset * numHours + hour;
				// Thermal channels match Ladybug AnalysisPeriod(..., 0..23) used by Python baseline:
				// hour 0 uses previous-day EPW hour 24; hours 1..23 use same-day EPW hours 1..23.
				const prev = hour === 0 ? previousCalendarDay(month, day) : null;
				const thermalMonth = prev?.month ?? month;
				const thermalDay = prev?.day ?? day;
				const thermalHour = hour === 0 ? 24 : hour;

				// Solar channels use accumulated previous-hour EnergyPlus convention (shift + 1)
				const solarHour = hour + 1;
				const fallbackData =
					// Short synthetic EPW fixtures in tests may not contain a full-year index.
					// In that case, fall back to contiguous records (monthOffset * numHours + hour).
					(idx < epw.dryBulbTemp.length
						? {
							dryBulb: epw.dryBulbTemp[idx],
							relHumidity: epw.relativeHumidity[idx],
							directNormal: epw.directNormalRad[idx],
							diffuseHoriz: epw.diffuseHorizRad[idx],
							windSpeed: epw.windSpeed[idx],
							horizIR: epw.horizInfrared[idx]
						}
						: undefined);
				const thermalData = epw.getHourData(thermalMonth, thermalDay, thermalHour) ?? fallbackData;
				const solarData = epw.getHourData(month, day, solarHour) ?? fallbackData;
				if (!thermalData || !solarData) {
					throw new Error(
						`Missing EPW hour data for thermal(${thermalMonth}-${thermalDay} h${thermalHour}) or solar(${month}-${day} h${solarHour})`
					);
				}

				const base = idx * weatherStride;
				weather[base] = thermalData.dryBulb; // air_temp
				weather[base + 1] = thermalData.dryBulb; // mrt_longwave (approx for now)
				weather[base + 2] = thermalData.windSpeed;
				weather[base + 3] = thermalData.relHumidity;
				weather[base + 4] = solarData.directNormal;
				weather[base + 5] = solarData.diffuseHoriz;
				weather[base + 6] = thermalData.horizIR;
			}
		}

		// 4. Tregenza dome (Y-up) for GPU sky exposure when pipeline has BVH
		const dome = getTregenzaDome();
		const domeVectors = new Float32Array(dome.vectors.length * 3);
		for (let i = 0; i < dome.vectors.length; i++) {
			const v = rotateZUpToYUp(dome.vectors[i]);
			domeVectors[i * 3] = v[0];
			domeVectors[i * 3 + 1] = v[1];
			domeVectors[i * 3 + 2] = v[2];
		}
		const domeWeights = new Float32Array(dome.weights);

		// 5. Upload to GPU pipeline (grid, sun, weather, serialized BVH, dome for sky exposure)
		await this.pipeline.uploadStaticData({
			gridPoints,
			sunVectors,
			sunAltitudes,
			weather,
			serializedBvh,
			domeVectors,
			domeWeights
		});

		await this.pipeline.runAll({
			numPoints,
			numHours,
			numMonths
		});

		return {
			numPoints,
			numMonths,
			numHours,
			gridPoints
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

	async runExposurePrecompute(params: ExposurePrecomputeParams): Promise<void> {
		if (!this.pipeline.runExposurePrecompute) {
			throw new Error('The configured UTCI pipeline does not support exposure-only precompute.');
		}
		return this.pipeline.runExposurePrecompute(params);
	}

	async runUtciForTimeIndex(params: RunUtciForTimeIndexParams): Promise<OnDemandUtciOutput> {
		if (!this.pipeline.runUtciForTimeIndex) {
			throw new Error('The configured UTCI pipeline does not support one-hour UTCI compute.');
		}
		return this.pipeline.runUtciForTimeIndex(params);
	}

	getOnDemandDiagnostics(): OnDemandRuntimeDiagnostics | undefined {
		return this.pipeline.getOnDemandDiagnostics?.();
	}

	/** Expose pipeline for advanced readback patterns (bulk read). */
	getPipeline(): UTCIComputePipeline {
		return this.pipeline;
	}
}
