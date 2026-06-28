export interface SolarCalInput {
  directNormalRad: number;
  diffuseHorizRad: number;
  horizInfrared: number; // Needed for longwave MRT
  solarAltitude: number; // degrees
  solarExposure: number; // 0 to 1
	skyViewFactor: number; // 0 to 1
  groundReflectance: number; // 0 to 1
  airTemp: number; // Celsius
	surfaceTemp?: number; // Celsius, defaults to airTemp
}

export interface SolarCalResult {
	shortwaveErf: number;
	longwaveErf: number;
	shortwaveDeltaMRT: number;
	longwaveDeltaMRT: number;
  outdoorMRT: number;
}

export function computeSolarCal(input: SolarCalInput): SolarCalResult {
  const {
    directNormalRad,
    diffuseHorizRad,
    horizInfrared,
    solarAltitude,
    solarExposure,
    skyViewFactor,
    groundReflectance,
		airTemp
  } = input;
	const surfaceTemp = Number.isFinite(input.surfaceTemp) ? (input.surfaceTemp as number) : airTemp;

	// Constants aligned with ladybug_comfort.solarcal defaults.
	const f_eff = 0.725;
	const rad_trans_coeff = 6.012;
	const a_sw = 0.7;
	const a_lw = 0.95;
  const sigma = 5.6697e-8; // Stefan-Boltzmann constant

	// Standing projection factor curve for SHARP=135 from Ladybug's spline table.
	const fpSharp135 = [
		0.22, 0.221, 0.222, 0.222, 0.223, 0.223, 0.224, 0.224, 0.224, 0.223, 0.223, 0.223,
		0.222, 0.222, 0.221, 0.22, 0.219, 0.218, 0.217, 0.215, 0.214, 0.212, 0.211, 0.209,
		0.207, 0.206, 0.204, 0.202, 0.2, 0.197, 0.195, 0.193, 0.191, 0.188, 0.186, 0.183,
		0.181, 0.179, 0.176, 0.174, 0.171, 0.169, 0.166, 0.164, 0.161, 0.159, 0.157, 0.154,
		0.152, 0.15, 0.148, 0.146, 0.144, 0.141, 0.139, 0.137, 0.135, 0.133, 0.13, 0.128,
		0.126, 0.123, 0.121, 0.118, 0.116, 0.113, 0.111, 0.108, 0.105, 0.103, 0.1, 0.097,
		0.095, 0.092, 0.09, 0.087, 0.085, 0.082, 0.08, 0.078, 0.076, 0.073, 0.071, 0.069,
		0.068, 0.066, 0.064, 0.063, 0.061, 0.06
	] as const;

  const alt_rad = (solarAltitude * Math.PI) / 180;
	const isShortwaveActive = solarAltitude >= 2;
	let f_p = 0;
	if (isShortwaveActive) {
		const idx = Math.ceil(Math.max(1, Math.min(90, solarAltitude))) - 1;
		f_p = fpSharp135[idx] ?? 0;
  }

	// Global horizontal radiation.
  let I_TH = diffuseHorizRad;
	if (isShortwaveActive) {
    I_TH += directNormalRad * Math.sin(alt_rad);
  }

	const shortFlux = isShortwaveActive
		? f_p * solarExposure * directNormalRad +
			0.5 * skyViewFactor * f_eff * diffuseHorizRad +
			0.5 * skyViewFactor * f_eff * I_TH * groundReflectance
		: 0;
	const shortwaveErf = shortFlux * (a_sw / a_lw);
	const shortwaveDeltaMRT = shortwaveErf / (f_eff * rad_trans_coeff);

	const safeHorizIr = Math.max(horizInfrared, 0);
	const skyTemp = Math.pow(safeHorizIr / (a_lw * sigma), 0.25) - 273.15;
	const longwaveDeltaMRT = 0.5 * skyViewFactor * (skyTemp - surfaceTemp);
	const longwaveErf = longwaveDeltaMRT * f_eff * rad_trans_coeff;

	const outdoorMRT = surfaceTemp + shortwaveDeltaMRT + longwaveDeltaMRT;

  return {
		shortwaveErf,
		longwaveErf,
		shortwaveDeltaMRT,
		longwaveDeltaMRT,
    outdoorMRT,
  };
}
