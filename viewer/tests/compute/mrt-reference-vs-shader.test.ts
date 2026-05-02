import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
	computeMrtReference,
	computeMrtReferenceComponents,
	type MrtReferenceInputs
} from '$lib/compute/mrtReference';

interface FixtureFile {
	cases: Array<MrtReferenceInputs & { name: string }>;
}

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}

const STANDING_FP_SHARP_135 = [
	0.22, 0.221, 0.222, 0.222, 0.223, 0.223, 0.224, 0.224, 0.224, 0.223, 0.223, 0.223,
	0.222, 0.222, 0.221, 0.22, 0.219, 0.218, 0.217, 0.215, 0.214, 0.212, 0.211, 0.209,
	0.207, 0.206, 0.204, 0.202, 0.2, 0.197, 0.195, 0.193, 0.191, 0.188, 0.186, 0.183,
	0.181, 0.179, 0.176, 0.174, 0.171, 0.169, 0.166, 0.164, 0.161, 0.159, 0.157, 0.154,
	0.152, 0.15, 0.148, 0.146, 0.144, 0.141, 0.139, 0.137, 0.135, 0.133, 0.13, 0.128,
	0.126, 0.123, 0.121, 0.118, 0.116, 0.113, 0.111, 0.108, 0.105, 0.103, 0.1, 0.097,
	0.095, 0.092, 0.09, 0.087, 0.085, 0.082, 0.08, 0.078, 0.076, 0.073, 0.071, 0.069,
	0.068, 0.066, 0.064, 0.063, 0.061, 0.06
] as const;

function projectionFactorSharp135(altitudeRad: number): number {
	const altitudeDeg = (altitudeRad * 180) / Math.PI;
	if (altitudeDeg <= 0) {
		return 0;
	}
	const idx = Math.ceil(clamp(altitudeDeg, 1, 90)) - 1;
	return STANDING_FP_SHARP_135[idx] ?? 0;
}

function computeMrtShaderEquivalent(inputs: MrtReferenceInputs) {
	const fEff = 0.725;
	const radTransCoeff = 6.012;
	const aSw = 0.7;
	const aLw = 0.95;
	const sigma = 5.6697e-8;
	const totalTregenzaWeight = 145.2488;
	const fG = 0.5;
	const groundReflectance = 0.25;

	const skyVf = clamp(inputs.skyExposureRaw / totalTregenzaWeight, 0, 1);
	const solarExp = clamp(inputs.solarExposure, 0, 1);
	const surfaceTempC = Number.isFinite(inputs.surfaceTempC) ? (inputs.surfaceTempC as number) : inputs.airTempC;
	const altitudeDeg = (inputs.solarAltitudeRad * 180) / Math.PI;
	const shortwaveActive = altitudeDeg >= 2;
	const fP = shortwaveActive ? projectionFactorSharp135(inputs.solarAltitudeRad) : 0;

	let iTh = inputs.diffuseHorizontalWm2;
	if (shortwaveActive && inputs.directNormalWm2 > 0) {
		iTh += inputs.directNormalWm2 * Math.sin(inputs.solarAltitudeRad);
	}

	const shortFlux = shortwaveActive
		? fP * solarExp * inputs.directNormalWm2 +
			0.5 * skyVf * fEff * inputs.diffuseHorizontalWm2 +
			fG * skyVf * fEff * iTh * groundReflectance
		: 0;
	const shortErf = shortFlux * (aSw / aLw);
	const shortDmrt = shortErf / (fEff * radTransCoeff);
	const skyTemp = Math.pow(Math.max(0, inputs.horizInfraredWm2) / (aLw * sigma), 0.25) - 273.15;
	const longDmrt = 0.5 * skyVf * (skyTemp - surfaceTempC);
	const longErf = longDmrt * fEff * radTransCoeff;
	return {
		shortErf,
		longErf,
		shortDmrt,
		longDmrt,
		mrt: surfaceTempC + shortDmrt + longDmrt
	};
}

describe('mrt reference vs shader', () => {
	it('keeps shader constants synchronized with TS reference contract', () => {
		const shaderPath = resolve(process.cwd(), 'src/lib/compute/shaders/mrt_utci.wgsl');
		const shaderSource = readFileSync(shaderPath, 'utf8');
		expect(shaderSource).toContain('let total_tregenza_weight: f32 = 145.2488');
		expect(shaderSource).toContain('let ground_reflectance: f32 = 0.25');
		expect(shaderSource).toContain('let f_eff: f32 = 0.725');
		expect(shaderSource).toContain('let rad_trans_coeff: f32 = 6.012');
		expect(shaderSource).toContain('var<storage, read_write> short_erf_results: array<f32>;');
		expect(shaderSource).toContain('var<storage, read_write> long_dmrt_results: array<f32>;');
	});

	it('clamps UTCI averaging to the representative-day boundary', () => {
		const shaderPath = resolve(process.cwd(), 'src/lib/compute/shaders/mrt_utci.wgsl');
		const shaderSource = readFileSync(shaderPath, 'utf8');
		expect(shaderSource).toContain('num_hours_per_day: u32,');
		expect(shaderSource).toContain('let hours_per_day = max(params.num_hours_per_day, 1u);');
		expect(shaderSource).toContain('let day_start = (time_idx / hours_per_day) * hours_per_day;');
		expect(shaderSource).toContain('let day_end = min(day_start + hours_per_day - 1u, params.num_time_steps - 1u);');
		expect(shaderSource).toContain('let next_idx = min(time_idx + 1u, day_end);');
		expect(shaderSource).not.toContain('let next_idx = min(time_idx + 1u, params.num_time_steps - 1u);');
		expect(shaderSource).toContain('utci_results[flat_index] = 0.5 * (utci0 + utci1);');
		expect(shaderSource).not.toContain('fall back to the single-hour UTCI');
	});

	it('shader MRT and components match reference on canonical fixtures', () => {
		const fixturePath = resolve(process.cwd(), 'tests/fixtures/parity/mrt-fixtures.json');
		const fixtures = JSON.parse(readFileSync(fixturePath, 'utf8')) as FixtureFile;

		let maxMrtDiff = 0;
		let maxShortErfDiff = 0;
		let maxLongErfDiff = 0;
		let maxShortDmrtDiff = 0;
		let maxLongDmrtDiff = 0;
		for (const fixtureCase of fixtures.cases) {
			const reference = computeMrtReference(fixtureCase);
			const referenceComponents = computeMrtReferenceComponents(fixtureCase);
			const shaderEquivalent = computeMrtShaderEquivalent(fixtureCase);
			maxMrtDiff = Math.max(maxMrtDiff, Math.abs(reference - shaderEquivalent.mrt));
			maxShortErfDiff = Math.max(
				maxShortErfDiff,
				Math.abs(referenceComponents.shortwaveErfWm2 - shaderEquivalent.shortErf)
			);
			maxLongErfDiff = Math.max(
				maxLongErfDiff,
				Math.abs(referenceComponents.longwaveErfWm2 - shaderEquivalent.longErf)
			);
			maxShortDmrtDiff = Math.max(
				maxShortDmrtDiff,
				Math.abs(referenceComponents.shortwaveDmrtC - shaderEquivalent.shortDmrt)
			);
			maxLongDmrtDiff = Math.max(
				maxLongDmrtDiff,
				Math.abs(referenceComponents.longwaveDmrtC - shaderEquivalent.longDmrt)
			);
		}

		expect(maxMrtDiff).toBeLessThan(1e-5);
		expect(maxShortErfDiff).toBeLessThan(1e-5);
		expect(maxLongErfDiff).toBeLessThan(1e-5);
		expect(maxShortDmrtDiff).toBeLessThan(1e-5);
		expect(maxLongDmrtDiff).toBeLessThan(1e-5);
	});
});
