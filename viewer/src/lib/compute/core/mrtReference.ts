export interface MrtReferenceInputs {
	solarExposure: number;
	skyExposureRaw: number;
	airTempC: number;
	surfaceTempC?: number;
	directNormalWm2: number;
	diffuseHorizontalWm2: number;
	horizInfraredWm2: number;
	solarAltitudeRad: number;
}

export const MRT_REFERENCE_CONSTANTS = {
	fEff: 0.725,
	aSw: 0.7,
	aLw: 0.95,
	sigma: 5.6697e-8,
	radTransCoeff: 6.012,
	totalTregenzaWeight: 145.2488,
	groundViewFactor: 0.5,
	groundReflectance: 0.25,
	standingSharpDeg: 135,
	shortwaveAltitudeCutoffDeg: 2
} as const;

// Ladybug standing projection factors sampled from get_projection_factor(alt, sharp=135)
// for altitude degrees 1..90 (SolarCal spline table path).
const STANDING_PROJECTION_FACTOR_SHARP_135: readonly number[] = [
	0.22, 0.221, 0.222, 0.222, 0.223, 0.223, 0.224, 0.224, 0.224, 0.223,
	0.223, 0.223, 0.222, 0.222, 0.221, 0.22, 0.219, 0.218, 0.217, 0.215,
	0.214, 0.212, 0.211, 0.209, 0.207, 0.206, 0.204, 0.202, 0.2, 0.197,
	0.195, 0.193, 0.191, 0.188, 0.186, 0.183, 0.181, 0.179, 0.176, 0.174,
	0.171, 0.169, 0.166, 0.164, 0.161, 0.159, 0.157, 0.154, 0.152, 0.15,
	0.148, 0.146, 0.144, 0.141, 0.139, 0.137, 0.135, 0.133, 0.13, 0.128,
	0.126, 0.123, 0.121, 0.118, 0.116, 0.113, 0.111, 0.108, 0.105, 0.103,
	0.1, 0.097, 0.095, 0.092, 0.09, 0.087, 0.085, 0.082, 0.08, 0.078,
	0.076, 0.073, 0.071, 0.069, 0.068, 0.066, 0.064, 0.063, 0.061, 0.06
] as const;

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}

function projectionFactorStandingSharp135(altitudeRad: number): number {
	const altitudeDeg = (altitudeRad * 180) / Math.PI;
	if (altitudeDeg <= 0) {
		return 0;
	}
	const idx = Math.ceil(clamp(altitudeDeg, 1, 90)) - 1;
	return STANDING_PROJECTION_FACTOR_SHARP_135[idx] ?? 0;
}

export interface MrtReferenceComponents {
	skyViewFactor: number;
	skyTemperatureC: number;
	mrtLongwaveBaseC: number;
	shortwaveBodyFluxWm2: number;
	shortwaveErfWm2: number;
	shortwaveDmrtC: number;
	longwaveErfWm2: number;
	longwaveDmrtC: number;
	mrtC: number;
}

export function computeMrtReferenceComponents(inputs: MrtReferenceInputs): MrtReferenceComponents {
	const c = MRT_REFERENCE_CONSTANTS;
	const surfaceTempC = Number.isFinite(inputs.surfaceTempC) ? (inputs.surfaceTempC as number) : inputs.airTempC;
	const skyVf = clamp(inputs.skyExposureRaw / c.totalTregenzaWeight, 0, 1);
	const solarExp = clamp(inputs.solarExposure, 0, 1);
	const altRad = inputs.solarAltitudeRad;
	const altDeg = (altRad * 180) / Math.PI;
	const isSolarActive = altDeg >= c.shortwaveAltitudeCutoffDeg;
	const fP = isSolarActive ? projectionFactorStandingSharp135(altRad) : 0;

	let iTh = inputs.diffuseHorizontalWm2;
	if (isSolarActive && inputs.directNormalWm2 > 0) {
		iTh += inputs.directNormalWm2 * Math.sin(altRad);
	}

	const shortwaveBodyFluxWm2 = isSolarActive
		? fP * solarExp * inputs.directNormalWm2 +
			0.5 * skyVf * c.fEff * inputs.diffuseHorizontalWm2 +
			c.groundViewFactor * skyVf * c.fEff * iTh * c.groundReflectance
		: 0;

	const shortwaveErfWm2 = shortwaveBodyFluxWm2 * (c.aSw / c.aLw);
	const shortwaveDmrtC = shortwaveErfWm2 / (c.fEff * c.radTransCoeff);

	const safeHorizIr = Math.max(0, inputs.horizInfraredWm2);
	const skyTemperatureC = Math.pow(safeHorizIr / (c.aLw * c.sigma), 0.25) - 273.15;
	const longwaveDmrtC = 0.5 * skyVf * (skyTemperatureC - surfaceTempC);
	const longwaveErfWm2 = longwaveDmrtC * c.fEff * c.radTransCoeff;

	const mrtLongwaveBaseC = surfaceTempC + longwaveDmrtC;
	const mrtC = surfaceTempC + shortwaveDmrtC + longwaveDmrtC;

	return {
		skyViewFactor: skyVf,
		skyTemperatureC,
		mrtLongwaveBaseC,
		shortwaveBodyFluxWm2,
		shortwaveErfWm2,
		shortwaveDmrtC,
		longwaveErfWm2,
		longwaveDmrtC,
		mrtC
	};
}

export function computeMrtReference(inputs: MrtReferenceInputs): number {
	return computeMrtReferenceComponents(inputs).mrtC;
}
