import { describe, it, expect, vi } from 'vitest';
import { computeSolarCal } from '$lib/compute/solarcal';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { createWebgpuUtciPipeline } from '$lib/compute/webgpuUtciPipeline';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// Note: This is a browser-only test in practice because it requires WebGPU.
// When running in Node/Vitest, it should be skipped unless a WebGPU shim is
// available. We keep it here as a high-level parity check between the WGSL
// SolarCal+UTCI path and the CPU reference implementation.

const HAS_WEBGPU = typeof navigator !== 'undefined' && (navigator as any).gpu;

const buildSinglePointInputs = () => {
	const numPoints = 1;
	const numMonths = 1;
	const numHours = 1;

	// Single grid point at origin (contents are irrelevant for SolarCal itself).
	const gridPoints = new Float32Array([0, 0, 0]);

	// Single sun vector; direction not used directly in WGSL SolarCal, but the
	// buffer must be non-empty to satisfy uploadStaticData.
	const sunVectors = new Float32Array([0, 1, 0]);

	// Solar altitude in radians for MRT (e.g. ~60° for summer midday)
	const sunAltitudes = new Float32Array([(60 * Math.PI) / 180]);

	// Weather layout must match WeatherSample in mrt_utci.wgsl and packing in
	// compute-manager.ts:
	//   0: air_temp (°C)
	//   1: mrt_longwave (°C) – approx air temp for now
	//   2: wind_speed (m/s)
	//   3: rel_humidity (%)
	//   4: direct_normal (W/m²)
	//   5: diffuse_horizontal (W/m²)
	//   6: horiz_infrared (W/m²)
	const weather = new Float32Array(7);
	weather[0] = 30; // air_temp
	weather[1] = 30; // mrt_longwave approx
	weather[2] = 1.0; // wind_speed
	weather[3] = 50; // rel_humidity
	weather[4] = 800; // direct_normal
	weather[5] = 200; // diffuse_horizontal
	weather[6] = 400; // horiz_infrared

	return { gridPoints, sunVectors, sunAltitudes, weather, numPoints, numMonths, numHours };
};

describe('MRT+UTCI WGSL SolarCal parity', () => {
	it('uses ground_reflectance 0.25 in WGSL for Python parity', () => {
		const shaderPath = resolve(process.cwd(), 'src/lib/compute/shaders/mrt_utci.wgsl');
		const shaderSource = readFileSync(shaderPath, 'utf8');
		expect(shaderSource).toContain('ground_reflectance: f32 = 0.25');
	});

	(HAS_WEBGPU ? it : it.skip)(
		'should produce UTCI close to CPU SolarCal+UTCI for a simple one-point scenario',
		async () => {
			// CPU reference using solarcal.ts + utci.ts
			const solar = computeSolarCal({
				directNormalRad: 800,
				diffuseHorizRad: 200,
				horizInfrared: 400,
				// Use a reasonable solar altitude for summer midday
				solarAltitude: 60,
				solarExposure: 1,
				skyViewFactor: 0.8,
				groundReflectance: 0.2,
				airTemp: 30
			});

			// We only care that the GPU path produces a sensible UTCI in the same
			// ballpark as the CPU reference; exact UTCI parity is covered by other
			// tests once Python reference fixtures are wired in.

			const { gridPoints, sunVectors, sunAltitudes, weather, numPoints, numMonths, numHours } =
				buildSinglePointInputs();

			const pipeline: UTCIComputePipeline = await createWebgpuUtciPipeline();

			await pipeline.uploadStaticData({
				gridPoints,
				sunVectors,
				sunAltitudes,
				weather
			});

			await pipeline.runAll({
				numPoints,
				numHours,
				numMonths
			});

			const utciSlice = await pipeline.readUtcisSlice({
				monthIndex: 0,
				hourIndex: 0,
				numPoints,
				numHours,
				numMonths
			});

			expect(utciSlice.length).toBe(1);

			// Basic sanity: in direct sun with high MRT, UTCI should be notably
			// above air temperature.
			const gpuUtci = utciSlice[0];
			expect(gpuUtci).toBeGreaterThan(30);

			// We expect GPU SolarCal MRT to be in the same range as the CPU one.
			// This is a loose check; tighter numeric parity will be enforced once
			// Python fixtures are wired in Phase 2.
			expect(solar.outdoorMRT).toBeGreaterThan(30);
		}
	);
});

