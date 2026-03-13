export interface SolarCalInput {
  directNormalRad: number;
  diffuseHorizRad: number;
  horizInfrared: number; // Needed for longwave MRT
  solarAltitude: number; // degrees
  solarExposure: number; // 0 to 1
  skyViewFactor: number; // 0 to 1
  groundReflectance: number; // 0 to 1
  airTemp: number; // Celsius
}

export interface SolarCalResult {
  erf: number;
  deltaMRT: number;
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
    airTemp,
  } = input;

  // Constants for SolarCal (human body)
  const f_eff = 0.725; // effective fraction of body surface exposed to radiation
  const a_sw = 0.7; // shortwave absorptivity
  const a_lw = 0.95; // longwave emissivity/absorptivity
  const sigma = 5.6697e-8; // Stefan-Boltzmann constant

  // 1. Projected area factor (f_p)
  // Projected area factor for a seated/standing person at solar altitude (alt_deg)
  // Ladybug/ASHRAE 55 polynomial or simplified trig. 
  // simplified: fp = 0.308 * cos(alt) + 0.082 * sin(alt) -- actually just use Ladybug's simplified
  const alt_rad = (solarAltitude * Math.PI) / 180;
  // ASHRAE 55 Standing person fp:
  let f_p = 0.308 * Math.cos(alt_rad); // simple cylinder approx
  
  if (solarAltitude < 0) {
    f_p = 0;
  }

  // 2. Shortwave ERF (Effective Radiant Field)
  // ERF_sw = a_sw * (f_svv * I_diff + f_svv * I_TH * R_floor + f_p * I_dir)
  // Using ladybug formulation:
  
  // Sky Vault View (f_svv): for a person, half the sphere is sky, half is ground.
  // With urban context, f_svv = 0.5 * skyViewFactor.
  const f_svv = 0.5 * skyViewFactor;
  
  // Ground View Factor: typically 0.5
  const f_g = 0.5;

  // Global horizontal radiation (I_TH)
  // I_TH = I_diff + I_dir * sin(alt)
  let I_TH = diffuseHorizRad;
  if (solarAltitude > 0) {
    I_TH += directNormalRad * Math.sin(alt_rad);
  }

  // Diffuse from sky: f_svv * I_diff
  const erf_diffuse = f_svv * diffuseHorizRad;
  
  // Ground reflected: f_g * I_TH * R_floor
  const erf_ground = f_g * I_TH * groundReflectance;
  
  // Direct solar: f_p * I_dir * solarExposure
  const erf_direct = f_p * directNormalRad * solarExposure;

  // Total shortwave ERF (W/m2)
  const erf_sw = (erf_diffuse + erf_ground + erf_direct) * (a_sw / a_lw);

  // 3. Longwave MRT base
  // We need the baseline MRT from longwave exchange.
  // MRT_lw = ( (horizInfrared / sigma) ) ^ 0.25 - 273.15?
  // Actually, ladybug OutdoorSolarCal calculates longwave MRT using sky temp and surface temp.
  // The simplest proxy when only horizIR and T_air are known:
  // Sky Temp T_sky = (horizInfrared / sigma)^0.25 - 273.15
  // MRT_lw = T_sky * f_svv + T_air * (1 - f_svv)
  const T_sky_k = Math.pow(horizInfrared / sigma, 0.25);
  const T_sky = T_sky_k - 273.15;
  
  // Base longwave MRT (Celsius)
  const mrt_lw = T_sky * f_svv + airTemp * (1 - f_svv);

  // 4. Delta MRT from Shortwave
  // delta_MRT = ERF_sw / (f_eff * sigma * a_lw * 4 * (MRT_lw + 273.15)^3) ?
  // wait, the linearization is 4 * sigma * T^3, but standard formula is:
  // MRT = ( (MRT_lw + 273.15)^4 + ERF_sw / (f_eff * sigma * a_lw) )^0.25 - 273.15
  const base_k = mrt_lw + 273.15;
  const mrt_k = Math.pow((Math.pow(base_k, 4) + erf_sw / (f_eff * sigma * a_lw)), 0.25);
  const outdoorMRT = mrt_k - 273.15;
  const deltaMRT = outdoorMRT - mrt_lw;

  return {
    erf: erf_sw,
    deltaMRT,
    outdoorMRT,
  };
}
