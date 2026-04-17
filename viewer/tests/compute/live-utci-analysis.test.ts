import { describe, it, expect, vi } from 'vitest';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { createLiveUtciAnalysisFromCompute } from '$lib/compute/liveUtciAnalysis';
import type { AnalysisMetadata } from '$lib/types/analysis';
import {
	clearComputeTelemetryHistory,
	getComputeTelemetryHistory
} from '$lib/compute/telemetry';

// Minimal EPW content: 8 header lines + 24 data lines for a single representative day
const buildMinimalEpw = () => {
	const header = [
		'LOCATION,Beer Sheva,ISR,source,wmo,31.25,34.79,2,300',
		'DESIGN CONDITIONS,dummy',
		'TYPICAL/EXTREME,dummy',
		'GROUND TEMPERATURES,dummy',
		'HOLIDAYS/DAYLIGHT,dummy',
		'COMMENTS 1,dummy',
		'COMMENTS 2,dummy',
		'DATA PERIODS,dummy'
	];

	const lines: string[] = [];
	for (let hour = 1; hour <= 24; hour++) {
		// year, month, day, hour, minute, ..., dryBulb(6), ..., relHum(8), ..., horizIR(12),
		// ..., dirNorm(14), diffHoriz(15), ..., wind(21)
		lines.push(
			`2020,8,15,${hour},0,0,30.0,25.0,50,99999,0,0,400,0,800,200,0,0,0,0,0,3.0`
		);
	}

	return `${header.join('\n')}\n${lines.join('\n')}`;
};

// Full-year EPW for 12-month compute: 366 days * 24 hours
const buildFullYearEpw = () => {
	const header = [
		'LOCATION,Beer Sheva,ISR,source,wmo,31.25,34.79,2,300',
		'DESIGN CONDITIONS,dummy',
		'TYPICAL/EXTREME,dummy',
		'GROUND TEMPERATURES,dummy',
		'HOLIDAYS/DAYLIGHT,dummy',
		'COMMENTS 1,dummy',
		'COMMENTS 2,dummy',
		'DATA PERIODS,dummy'
	];

	const DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
	const lines: string[] = [];
	for (let month = 1; month <= 12; month++) {
		for (let day = 1; day <= DAYS_IN_MONTH[month - 1]; day++) {
			for (let hour = 1; hour <= 24; hour++) {
				lines.push(
					`2020,${month},${day},${hour},0,0,30.0,25.0,50,99999,0,0,400,0,800,200,0,0,0,0,0,3.0`
				);
			}
		}
	}
	return `${header.join('\n')}\n${lines.join('\n')}`;
};

const createSerializedBvhFixture = () => {
	return {
		serializedBvh: {
			bvhNodeBuffer: new ArrayBuffer(0),
			bvhIndexBuffer: new ArrayBuffer(0),
			vertexBuffer: new Float32Array(0),
			indexBuffer: new Uint32Array(0)
		}
	};
};

const createFakePipeline = () => {
	const uploadStaticData = vi.fn().mockResolvedValue(undefined);
	const runAll = vi.fn().mockResolvedValue(undefined);
	const readUtcisSlice = vi
		.fn()
		.mockImplementation(
			async (params: { monthIndex: number; hourIndex: number; numPoints: number }) => {
				const { monthIndex, hourIndex, numPoints } = params;
				const arr = new Float32Array(numPoints);
				for (let i = 0; i < numPoints; i++) {
					// Deterministic pattern for assertions
					arr[i] = monthIndex * 100 + hourIndex + i * 0.5;
				}
				return arr;
			}
		);

	const pipeline: UTCIComputePipeline = {
		uploadStaticData,
		runAll,
		readUtcisSlice
	};

	return { pipeline, uploadStaticData, runAll, readUtcisSlice };
};

describe('liveUtciAnalysis adapter', () => {
	it('emits telemetry stages for pipeline upload and UTCI readback', async () => {
		clearComputeTelemetryHistory();
		const { pipeline } = createFakePipeline();
		const baseMetadata: AnalysisMetadata = {
			analysis_type: 'full_day',
			num_positions: 0,
			hours: ['00:00', '01:00'],
			utci_range: { min: -10, max: 50 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'test.glb',
			bounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 }
		};

		await createLiveUtciAnalysisFromCompute(
			{
				analysisId: 'Test/telemetry',
				baseMetadata,
				workerResult: createSerializedBvhFixture(),
				epwContent: buildMinimalEpw(),
				gridResolution: 2,
				numHours: 2,
				startMonth: 8
			},
			{ pipeline }
		);

		const events = getComputeTelemetryHistory();
		expect(events.some((e) => e.stage === 'pipeline.upload.done')).toBe(true);
		expect(events.some((e) => e.stage === 'utci.readback.done')).toBe(true);
	});

	it('produces an Analysis-like structure for a simple grid', async () => {
		const { pipeline, uploadStaticData, runAll, readUtcisSlice } = createFakePipeline();

		const baseMetadata: AnalysisMetadata = {
			analysis_type: 'full_day',
			num_positions: 0,
			hours: Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`),
			utci_range: { min: -10, max: 50 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'test.glb',
			bounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 }
		};

		const epw = buildMinimalEpw();

		const analysis = await createLiveUtciAnalysisFromCompute(
			{
				analysisId: 'Test/20250815_grid_2m_fullday',
				baseMetadata,
				workerResult: createSerializedBvhFixture(),
				epwContent: epw,
				gridResolution: 2,
				zHeight: 1.5,
				numHours: 24,
				startMonth: 8
			},
			{ pipeline }
		);

		// Basic structural checks
		expect(analysis.data.numPositions).toBeGreaterThan(0);
		expect(analysis.data.positions.length).toBe(analysis.data.numPositions * 3);

		if ('utciStorage' in analysis.data) {
			expect(analysis.data.numHours).toBe(24);
			expect(analysis.data.utciStorage!.numSlices).toBe(24);

			const { getUTCIForHour } = await import('$lib/services/dataLoader');
			for (let h = 0; h < 24; h++) {
				const slice = getUTCIForHour(analysis.data, h);
				expect(slice.length).toBe(analysis.data.numPositions);
				for (let i = 0; i < slice.length; i++) {
					expect(Number.isFinite(slice[i])).toBe(true);
				}
			}
		} else if ('utciByHour' in analysis.data) {
			expect(analysis.data.numHours).toBe(24);
			expect(analysis.data.utciByHour!.length).toBe(24);
			for (const slice of analysis.data.utciByHour!) {
				expect(slice.length).toBe(analysis.data.numPositions);
				for (let i = 0; i < slice.length; i++) {
					expect(Number.isFinite(slice[i])).toBe(true);
				}
			}
		} else {
			throw new Error('Expected full-day data with utciStorage or utciByHour');
		}

		// Metadata expectations
		expect(analysis.metadata.analysis_type).toBe('full_day');
		expect(analysis.metadata.num_positions).toBe(analysis.data.numPositions);
		expect(analysis.metadata.grid_size).toBe(2);
		expect(analysis.metadata.coordinate_system).toBe('xy_ground');
		expect(analysis.metadata.utci_range.min).toBeLessThan(analysis.metadata.utci_range.max);
		expect(analysis.metadata.hours.length).toBe(24);

		// Hour statistics should be populated
		expect(analysis.metadata.hour_statistics).toBeDefined();
		expect(analysis.metadata.hour_statistics?.length).toBe(24);

		// Pipeline interactions
		expect(uploadStaticData).toHaveBeenCalledTimes(1);
		expect(runAll).toHaveBeenCalledTimes(1);
		// Should read back one slice per hour
		expect(readUtcisSlice).toHaveBeenCalledTimes(24);
	});

	it('throws when pipeline returns UTCI slice with unexpected length', async () => {
		const { pipeline } = createFakePipeline();
		const badPipeline: UTCIComputePipeline = {
			...pipeline,
			readUtcisSlice: vi.fn(async () => new Float32Array(1))
		};

		const baseMetadata: AnalysisMetadata = {
			analysis_type: 'full_day',
			num_positions: 0,
			hours: ['00:00', '01:00'],
			utci_range: { min: -10, max: 50 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'test.glb',
			bounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 }
		};

		await expect(
			createLiveUtciAnalysisFromCompute(
				{
					analysisId: 'Test/bad-slice',
					baseMetadata,
					workerResult: createSerializedBvhFixture(),
					epwContent: buildMinimalEpw(),
					gridResolution: 2,
					numHours: 2,
					startMonth: 8
				},
				{ pipeline: badPipeline }
			)
		).rejects.toThrow(/UTCI slice length mismatch/);
	});

	it('produces 288 utciStorage slices when numMonths=12', async () => {
		const { pipeline, readUtcisSlice } = createFakePipeline();

		const baseMetadata: AnalysisMetadata = {
			analysis_type: 'full_day',
			num_positions: 0,
			hours: Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`),
			utci_range: { min: -10, max: 50 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'test.glb',
			bounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 }
		};

		const result = await createLiveUtciAnalysisFromCompute(
			{
				analysisId: 'test',
				baseMetadata,
				workerResult: createSerializedBvhFixture(),
				epwContent: buildFullYearEpw(),
				gridResolution: 2,
				numHours: 24,
				startMonth: 1,
				numMonths: 12
			},
			{ pipeline }
		);

		expect(result.data.utciStorage).toBeDefined();
		expect(result.data.utciStorage!.numSlices).toBe(288);
		expect(result.metadata.num_months).toBe(12);
		expect(result.data.numHours).toBe(288);
		expect(readUtcisSlice).toHaveBeenCalledTimes(288);
	});
});

describe('grid position sanity', () => {
	it('uses parity default zHeight of 0.9m when zHeight is not provided', async () => {
		const { pipeline } = createFakePipeline();

		const baseMetadata: AnalysisMetadata = {
			analysis_type: 'full_day',
			num_positions: 0,
			hours: Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`),
			utci_range: { min: -10, max: 50 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'test.glb',
			bounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 0.9 }
		};

		const analysis = await createLiveUtciAnalysisFromCompute(
			{
				analysisId: 'Test/default-zheight',
				baseMetadata,
				workerResult: createSerializedBvhFixture(),
				epwContent: buildMinimalEpw(),
				gridResolution: 2,
				numHours: 1,
				startMonth: 8
			},
			{ pipeline }
		);

		// For xy_ground mapping, world Y (sensor height) maps to analysis Z.
		const positions = analysis.data.positions;
		let sumZ = 0;
		for (let i = 2; i < positions.length; i += 3) {
			sumZ += positions[i];
		}
		const meanZ = sumZ / (positions.length / 3);
		expect(meanZ).toBeCloseTo(0.9, 3);
	});
});
