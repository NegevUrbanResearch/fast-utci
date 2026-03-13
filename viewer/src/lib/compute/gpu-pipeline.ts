export interface PipelineConfig {
	numPoints: number;
	numHours: number;
	numMonths: number;
	solarExposureBufferSize: number;
	skyExposureBufferSize: number;
	utciResultBufferSize: number;
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
		// f32 per point × hour × month
		solarExposureBufferSize: numPoints * totalTimeSteps * 4,
		// Single sky view factor per point
		skyExposureBufferSize: numPoints * 4,
		// f32 per point × hour × month
		utciResultBufferSize: numPoints * totalTimeSteps * 4
	};
}

export interface DispatchDimensions {
	x: number;
	y: number;
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

/**
 * Minimal interface for the UTCI GPU compute pipeline orchestrator.
 * 
 * The concrete WebGPU implementation lives in browser-only code; tests can
 * provide a fake implementation that conforms to this interface.
 */
export interface UTCIComputePipeline {
	/**
	 * Upload static data required for all dispatches.
	 * These arrays must match the sizes described by PipelineConfig.
	 */
	uploadStaticData(params: {
		gridPoints: Float32Array; // vec3<f32>[numPoints]
		sunVectors: Float32Array; // vec3<f32>[numMonths * numHours]
		weather: Float32Array; // packed per-hour weather structs
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
}


