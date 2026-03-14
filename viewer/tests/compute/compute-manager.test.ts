import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import * as sunpath from '$lib/compute/sunpath';

// Minimal EPW content: 8 header lines + 24 data lines for month=1, day=15
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
			`2020,1,15,${hour},0,?,25.0,20.0,50,99999,0,0,400,0,800,200,0,0,0,0,0,3.0`
		);
	}

	return `${header.join('\n')}\n${lines.join('\n')}`;
};

const createTestMesh = () => {
	// Simple 4x4 plane at y=0, rotated to XZ ground plane (Y up)
	const geometry = new THREE.PlaneGeometry(4, 4);
	geometry.rotateX(-Math.PI / 2);
	return new THREE.Mesh(geometry);
};

/** Bounds for the 4x4 test plane (analysis xy_ground: x/z span ±2, z = sensor height). */
const TEST_BOUNDS = { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 };

const createFakePipeline = () => {
	const uploadStaticData = vi.fn().mockResolvedValue(undefined);
	const runAll = vi.fn().mockResolvedValue(undefined);
	const readUtcisSlice = vi
		.fn()
		.mockImplementation(
			async (params: { monthIndex: number; hourIndex: number; numPoints: number }) => {
				// Return a simple ramp for determinism in tests
				const arr = new Float32Array(params.numPoints);
				for (let i = 0; i < params.numPoints; i++) {
					arr[i] = params.monthIndex * 100 + params.hourIndex + i * 0.1;
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

describe('ComputeManager', () => {
	it('should prepare grid, sun vectors and weather and call pipeline', async () => {
		const { pipeline, uploadStaticData, runAll } = createFakePipeline();
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		const mesh = createTestMesh();
		const epw = buildMinimalEpw();

		const result = await manager.initFromModelAndWeather({
			mesh,
			epwContent: epw,
			gridResolution: 2,
			zHeight: 1.5,
			useRectangularGridFromBounds: true,
			analysisBounds: TEST_BOUNDS
		});

		// Expect some grid points to have been generated
		expect(result.numPoints).toBeGreaterThan(0);
		expect(result.numMonths).toBe(1);
		expect(result.numHours).toBe(24);

		// Verify pipeline was called with correctly sized arrays and BVH/dome data
		expect(uploadStaticData).toHaveBeenCalledTimes(1);
		const args = uploadStaticData.mock.calls[0][0];
		expect(args.gridPoints).toBeInstanceOf(Float32Array);
		expect(args.sunVectors).toBeInstanceOf(Float32Array);
		expect(args.weather).toBeInstanceOf(Float32Array);
		expect(args.mesh).toBe(mesh);
		expect(args.domeVectors).toBeInstanceOf(Float32Array);
		expect(args.domeWeights).toBeInstanceOf(Float32Array);
		expect(args.domeVectors.length).toBe(145 * 3);
		expect(args.domeWeights.length).toBe(145);

		// sunVectors: numMonths * numHours * 3
		expect(args.sunVectors.length).toBe(1 * 24 * 3);

		// sunAltitudes: numMonths * numHours (radians)
		expect(args.sunAltitudes).toBeInstanceOf(Float32Array);
		expect(args.sunAltitudes!.length).toBe(1 * 24);

		// weather: numMonths * numHours * 7
		// (air, mrt_lw, wind, rh, direct_normal, diffuse_horizontal, horiz_infrared)
		expect(args.weather.length).toBe(1 * 24 * 7);

		// runAll should be called once with matching counts
		expect(runAll).toHaveBeenCalledTimes(1);
		expect(runAll).toHaveBeenCalledWith({
			numPoints: result.numPoints,
			numHours: 24,
			numMonths: 1
		});

		// Basic orientation sanity check: in January at Beer Sheva the highest sun
		// altitude occurs around local noon and should be mostly "up" (+Y) with a
		// relatively small horizontal component in the XZ plane in the Three.js
		// Y-up world frame.
		const packedSun: Float32Array = args.sunVectors;
		let maxY = -Infinity;
		let maxIndex = -1;
		for (let i = 0; i < 24; i++) {
			const vy = packedSun[i * 3 + 1];
			if (vy > maxY) {
				maxY = vy;
				maxIndex = i;
			}
		}
		const vx = packedSun[maxIndex * 3];
		const vy = packedSun[maxIndex * 3 + 1];
		const vz = packedSun[maxIndex * 3 + 2];

		// Mostly vertical with very small east/west component and a non-trivial
		// horizontal component so the sun is not straight overhead.
		expect(vy).toBeGreaterThan(0.5);
		expect(Math.abs(vx)).toBeLessThan(0.3);
		expect(Math.abs(vz)).toBeGreaterThan(0.3);
	});

	it('should delegate UTCI slice reads to the pipeline', async () => {
		const { pipeline, readUtcisSlice } = createFakePipeline();
		const manager = new ComputeManager(pipeline, { numMonths: 12, numHoursPerDay: 24 });

		const slice = await manager.getUtcisForMonthHour({
			monthIndex: 5,
			hourIndex: 12,
			numPoints: 10,
			numMonths: 12,
			numHours: 24
		});

		expect(readUtcisSlice).toHaveBeenCalledTimes(1);
		expect(slice.length).toBe(10);
	});

	it('must not compute UTCI on CPU when reading slices', async () => {
		const { pipeline, readUtcisSlice } = createFakePipeline();
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		// The manager exposes only a GPU-oriented API for UTCI slices; there is
		// no public CPU shortcut. This test locks in that contract by asserting
		// that all reads delegate to the injected pipeline implementation.
		const slice = await manager.getUtcisForMonthHour({
			monthIndex: 0,
			hourIndex: 0,
			numPoints: 5,
			numMonths: 1,
			numHours: 24
		});

		expect(readUtcisSlice).toHaveBeenCalledTimes(1);
		expect(slice.length).toBe(5);

		// If a future refactor tried to introduce a CPU-computed UTCI path here,
		// this assertion would start failing because the fake pipeline would no
		// longer be consulted.
	});

	it('initFromModelAndWeather should return grid points', async () => {
		const { pipeline } = createFakePipeline();
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });
		const mesh = createTestMesh();
		const epw = buildMinimalEpw();

		const result = await manager.initFromModelAndWeather({
			mesh,
			epwContent: epw,
			gridResolution: 2,
			zHeight: 1.5,
			useRectangularGridFromBounds: true,
			analysisBounds: TEST_BOUNDS
		});

		expect(result.gridPoints).toBeDefined();
		expect(result.gridPoints).toBeInstanceOf(Float32Array);
		expect(result.gridPoints!.length).toBe(result.numPoints * 3);
	});

	it('should rotate ENU north vector to negative world Z for XY-ground parity', async () => {
		const { pipeline, uploadStaticData } = createFakePipeline();
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });
		const mesh = createTestMesh();
		const epw = buildMinimalEpw();

		const sunVectors = Array.from({ length: 24 }, () => [0, 1, 0] as [number, number, number]);
		const altitudes = Array.from({ length: 24 }, () => 45);
		const isSunUp = Array.from({ length: 24 }, () => true);

		const spy = vi.spyOn(sunpath, 'getSunVectors').mockReturnValue({
			sunVectors,
			altitudes,
			isSunUp
		});

		try {
			await manager.initFromModelAndWeather({
				mesh,
				epwContent: epw,
				gridResolution: 2,
				zHeight: 1.5,
				useRectangularGridFromBounds: true,
				analysisBounds: TEST_BOUNDS
			});
		} finally {
			spy.mockRestore();
		}

		const args = uploadStaticData.mock.calls[0][0];
		expect(args.sunVectors[0]).toBeCloseTo(0, 6);
		expect(args.sunVectors[1]).toBeCloseTo(0, 6);
		expect(args.sunVectors[2]).toBeCloseTo(-1, 6);
	});

	it('should use fixture sun vectors directly when parity fixture mode is provided', async () => {
		const { pipeline, uploadStaticData } = createFakePipeline();
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });
		const mesh = createTestMesh();
		const epw = buildMinimalEpw();

		const fixtureVectors = new Float32Array(24 * 3);
		const fixtureAltitudes = new Float32Array(24);
		for (let i = 0; i < 24; i++) {
			fixtureVectors[i * 3] = 0.1;
			fixtureVectors[i * 3 + 1] = 0.9;
			fixtureVectors[i * 3 + 2] = -0.2;
			fixtureAltitudes[i] = 0.5;
		}

		const spy = vi.spyOn(sunpath, 'getSunVectors');
		await manager.initFromModelAndWeather({
			mesh,
			epwContent: epw,
			gridResolution: 2,
			zHeight: 1.5,
			useRectangularGridFromBounds: true,
			analysisBounds: TEST_BOUNDS,
			sunVectorsFixture: {
				sunVectors: fixtureVectors,
				sunAltitudes: fixtureAltitudes
			}
		});

		const args = uploadStaticData.mock.calls[0][0];
		expect(args.sunVectors).toBe(fixtureVectors);
		expect(args.sunAltitudes).toBe(fixtureAltitudes);
		expect(spy).not.toHaveBeenCalled();
		spy.mockRestore();
	});
});
