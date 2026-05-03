// MRT + UTCI compute shader
// Reads solar and sky exposure plus per-hour weather and writes UTCI values.

struct WeatherSample {
	// Air temperature in °C
	air_temp: f32,
	// Longwave MRT baseline in °C (can be used as a fallback or for
	// validation; current GPU path derives MRT primarily from SolarCal
	// inputs and horizInfrared but keeps this for parity/debugging).
	mrt_longwave: f32,
	// Wind speed in m/s
	wind_speed: f32,
	// Relative humidity in %
	rel_humidity: f32,
	// Direct normal solar radiation (W/m²)
	direct_normal: f32,
	// Diffuse horizontal radiation (W/m²)
	diffuse_horizontal: f32,
	// Horizontal infrared radiation intensity (W/m²)
	horiz_infrared: f32,
};

// Bit-packed solar exposure (read-only in MRT pass).
// Word index = flat_index / 32, bit index = flat_index % 32.
// Bit = 1 means exposed, 0 means occluded.
@group(0) @binding(0)
var<storage, read> solar_exposure: array<u32>;

@group(0) @binding(1)
var<storage, read> sky_exposure: array<f32>;

@group(0) @binding(2)
var<storage, read> weather_data: array<WeatherSample>;

@group(0) @binding(3)
var<storage, read_write> utci_results: array<f32>;

struct MRTParams {
	num_points: u32,
	num_time_steps: u32,
	num_hours_per_day: u32,
	_pad: u32,
};

@group(0) @binding(4)
var<uniform> params: MRTParams;

@group(0) @binding(5)
var<storage, read> sun_altitudes: array<f32>;

@group(0) @binding(6)
var<storage, read_write> mrt_results: array<f32>;

// MRT_COMPONENT_DECLS_START
@group(0) @binding(7)
var<storage, read_write> short_erf_results: array<f32>;

@group(0) @binding(8)
var<storage, read_write> long_erf_results: array<f32>;

@group(0) @binding(9)
var<storage, read_write> short_dmrt_results: array<f32>;

@group(0) @binding(10)
var<storage, read_write> long_dmrt_results: array<f32>;
// MRT_COMPONENT_DECLS_END

const STANDING_FP_SHARP_135: array<f32, 90> = array<f32, 90>(
	0.220, 0.221, 0.222, 0.222, 0.223, 0.223, 0.224, 0.224, 0.224, 0.223,
	0.223, 0.223, 0.222, 0.222, 0.221, 0.220, 0.219, 0.218, 0.217, 0.215,
	0.214, 0.212, 0.211, 0.209, 0.207, 0.206, 0.204, 0.202, 0.200, 0.197,
	0.195, 0.193, 0.191, 0.188, 0.186, 0.183, 0.181, 0.179, 0.176, 0.174,
	0.171, 0.169, 0.166, 0.164, 0.161, 0.159, 0.157, 0.154, 0.152, 0.150,
	0.148, 0.146, 0.144, 0.141, 0.139, 0.137, 0.135, 0.133, 0.130, 0.128,
	0.126, 0.123, 0.121, 0.118, 0.116, 0.113, 0.111, 0.108, 0.105, 0.103,
	0.100, 0.097, 0.095, 0.092, 0.090, 0.087, 0.085, 0.082, 0.080, 0.078,
	0.076, 0.073, 0.071, 0.069, 0.068, 0.066, 0.064, 0.063, 0.061, 0.060
);

struct MrtComponents {
	mrt: f32,
	short_erf: f32,
	long_erf: f32,
	short_dmrt: f32,
	long_dmrt: f32
};

fn projection_factor_standing_sharp_135(alt_rad: f32) -> f32 {
	let alt_deg: f32 = degrees(alt_rad);
	if (alt_deg <= 0.0) {
		return 0.0;
	}
	let clamped: f32 = clamp(alt_deg, 1.0, 90.0);
	let idx: u32 = u32(ceil(clamped) - 1.0);
	return STANDING_FP_SHARP_135[idx];
}

// SolarCal-style MRT computation aligned with Ladybug outdoor_sky_heat_exch terms.
fn compute_outdoor_mrt(
	point_idx: u32,
	time_idx: u32,
	surface_temp_c: f32,
	w: WeatherSample,
	num_time_steps: u32,
) -> MrtComponents {
	// Constants kept in sync with src/lib/compute/mrtReference.ts.
	let f_eff: f32 = 0.725;
	let a_sw: f32 = 0.7;
	let a_lw: f32 = 0.95;
	let sigma: f32 = 5.6697e-8;
	let rad_trans_coeff: f32 = 6.012;
	let total_tregenza_weight: f32 = 145.2488;
	let f_g: f32 = 0.5;
	let ground_reflectance: f32 = 0.25;
	let shortwave_alt_cutoff_deg: f32 = 2.0;

	// Tregenza weights sum to ~145.25; divide raw sky exposure to get [0,1] SVF.
	var sky_vf = clamp(sky_exposure[point_idx] / total_tregenza_weight, 0.0, 1.0);

	// Solar exposure: unpack single bit from bit-packed u32 buffer.
	let flat_index: u32 = point_idx * num_time_steps + time_idx;
	let solar_word: u32 = solar_exposure[flat_index / 32u];
	let solar_bit: u32 = (solar_word >> (flat_index % 32u)) & 1u;
	let solar_exp: f32 = f32(solar_bit);

	// Real solar altitude for this timestep (radians), from CPU sunpath.
	let alt_rad: f32 = sun_altitudes[time_idx];
	let alt_deg: f32 = degrees(alt_rad);
	let shortwave_active: bool = alt_deg >= shortwave_alt_cutoff_deg;

	let f_p: f32 = select(0.0, projection_factor_standing_sharp_135(alt_rad), shortwave_active);

	// Global horizontal radiation approximation
	var i_th: f32 = w.diffuse_horizontal;
	if (shortwave_active && w.direct_normal > 0.0) {
		i_th = i_th + w.direct_normal * sin(alt_rad);
	}

	let short_flux_direct: f32 = f_p * solar_exp * w.direct_normal;
	let short_flux_diff: f32 = 0.5 * sky_vf * f_eff * w.diffuse_horizontal;
	let short_flux_ground: f32 = f_g * sky_vf * f_eff * i_th * ground_reflectance;
	let short_flux_total: f32 = select(
		0.0,
		short_flux_direct + short_flux_diff + short_flux_ground,
		shortwave_active
	);
	let short_erf: f32 = short_flux_total * (a_sw / a_lw);
	let short_dmrt: f32 = short_erf / (f_eff * rad_trans_coeff);

	let safe_horiz_ir: f32 = max(0.0, w.horiz_infrared);
	let sky_temp_c: f32 = pow(safe_horiz_ir / (a_lw * sigma), 0.25) - 273.15;
	let long_dmrt: f32 = 0.5 * sky_vf * (sky_temp_c - surface_temp_c);
	let long_erf: f32 = long_dmrt * f_eff * rad_trans_coeff;

	let mrt: f32 = surface_temp_c + short_dmrt + long_dmrt;
	return MrtComponents(mrt, short_erf, long_erf, short_dmrt, long_dmrt);
}

// UTCI polynomial ported from utci.ts / Bröde et al. (2012)
fn compute_utci(air_temp: f32, mrt: f32, wind_speed: f32, rel_humidity: f32) -> f32 {
	// Domain policy matches TS default ('clamped-domain'):
	// clamp tdb to [-50, 50] and delta(mrt-tdb) to [-30, 70].
	let tdb_clamped = clamp(air_temp, -50.0, 50.0);
	let delta_clamped = clamp(mrt - air_temp, -30.0, 70.0);
	let tdb = tdb_clamped;
	var v = max(0.5, min(17.0, wind_speed));
	let delta_t_tr = delta_clamped;
	let tk = tdb + 273.15;

	// Saturation vapor pressure (Pa) approximation
	let g0 = -2836.5744;
	let g1 = -6028.076559;
	let g2 = 19.54263612;
	let g3 = -0.02737830188;
	let g4 = 0.000016261698;
	let g5 = 7.0229056e-10;
	let g6 = -1.8680009e-13;

	var es = 2.7150305 * log(tk);
	es = es + g0 * pow(tk, -2.0);
	es = es + g1 * pow(tk, -1.0);
	es = es + g2 * pow(tk, 0.0);
	es = es + g3 * pow(tk, 1.0);
	es = es + g4 * pow(tk, 2.0);
	es = es + g5 * pow(tk, 3.0);
	es = es + g6 * pow(tk, 4.0);

	es = exp(es) * 0.01;
	let eh_pa = es * (rel_humidity / 100.0);
	let pa = eh_pa / 10.0;

	let d = delta_t_tr;
	let v2 = v * v;
	let v3 = v2 * v;
	let v4 = v3 * v;
	let v5 = v4 * v;
	let v6 = v5 * v;
	let d2 = d * d;
	let d3 = d2 * d;
	let d4 = d3 * d;
	let d5 = d4 * d;
	let d6 = d5 * d;

	return  tdb + 0.607562052
		+ (-0.0227712343) * tdb
		+ (8.06470249e-4) * tdb * tdb
		+ (-1.54271372e-4) * tdb * tdb * tdb
		+ (-3.24651735e-6) * tdb * tdb * tdb * tdb
		+ (7.32602852e-8) * tdb * tdb * tdb * tdb * tdb
		+ (1.35959073e-9) * tdb * tdb * tdb * tdb * tdb * tdb
		+ (-2.25836520) * v
		+ 0.0880326035 * tdb * v
		+ 0.00216844454 * tdb * tdb * v
		+ (-1.53347087e-5) * tdb * tdb * tdb * v
		+ (-5.72983704e-7) * tdb * tdb * tdb * tdb * v
		+ (-2.55090145e-9) * tdb * tdb * tdb * tdb * tdb * v
		+ (-0.751269505) * v2
		+ (-0.00408350271) * tdb * v2
		+ (-5.21670675e-5) * tdb * tdb * v2
		+ (1.94544667e-6) * tdb * tdb * tdb * v2
		+ (1.14099531e-8) * tdb * tdb * tdb * tdb * v2
		+ 0.158137256 * v3
		+ (-6.57263143e-5) * tdb * v3
		+ (2.22697524e-7) * tdb * tdb * v3
		+ (-4.16117031e-8) * tdb * tdb * tdb * v3
		+ (-0.0127762753) * v4
		+ (9.66891875e-6) * tdb * v4
		+ (2.52785852e-9) * tdb * tdb * v4
		+ (4.56306672e-4) * v5
		+ (-1.74202546e-7) * tdb * v5
		+ (-5.91491269e-6) * v6
		+ 0.398374029 * d
		+ (1.83945314e-4) * tdb * d
		+ (-1.73754510e-4) * tdb * tdb * d
		+ (-7.60781159e-7) * tdb * tdb * tdb * d
		+ (3.77830287e-8) * tdb * tdb * tdb * tdb * d
		+ (5.43079673e-10) * tdb * tdb * tdb * tdb * tdb * d
		+ (-0.0200518269) * v * d
		+ (8.92859837e-4) * tdb * v * d
		+ (3.45433048e-6) * tdb * tdb * v * d
		+ (-3.77925774e-7) * tdb * tdb * tdb * v * d
		+ (-1.69699377e-9) * tdb * tdb * tdb * tdb * v * d
		+ (1.69992415e-4) * v2 * d
		+ (-4.99204314e-5) * tdb * v2 * d
		+ (2.47417178e-7) * tdb * tdb * v2 * d
		+ (1.07596466e-8) * tdb * tdb * tdb * v2 * d
		+ (8.49242932e-5) * v3 * d
		+ (1.35191328e-6) * tdb * v3 * d
		+ (-6.21531254e-9) * tdb * tdb * v3 * d
		+ (-4.99410301e-6) * v4 * d
		+ (-1.89489258e-8) * tdb * v4 * d
		+ (8.15300114e-8) * v5 * d
		+ (7.55043090e-4) * d2
		+ (-5.65095215e-5) * tdb * d2
		+ (-4.52166564e-7) * tdb * tdb * d2
		+ (2.46688878e-8) * tdb * tdb * tdb * d2
		+ (2.42674348e-10) * tdb * tdb * tdb * tdb * d2
		+ (1.54547250e-4) * v * d2
		+ (5.24110970e-6) * tdb * v * d2
		+ (-8.75874982e-8) * tdb * tdb * v * d2
		+ (-1.50743064e-9) * tdb * tdb * tdb * v * d2
		+ (-1.56236307e-5) * v2 * d2
		+ (-1.33895614e-7) * tdb * v2 * d2
		+ (2.49709824e-9) * tdb * tdb * v2 * d2
		+ (6.51711721e-7) * v3 * d2
		+ (1.94960053e-9) * tdb * v3 * d2
		+ (-1.00361113e-8) * v4 * d2
		+ (-1.21206673e-5) * d3
		+ (-2.18203660e-7) * tdb * d3
		+ (7.51269482e-9) * tdb * tdb * d3
		+ (9.79063848e-11) * tdb * tdb * tdb * d3
		+ (1.25006734e-6) * v * d3
		+ (-1.81584736e-9) * tdb * v * d3
		+ (-3.52197671e-10) * tdb * tdb * v * d3
		+ (-3.36514630e-8) * v2 * d3
		+ (1.35908359e-10) * tdb * v2 * d3
		+ (4.17032620e-10) * v3 * d3
		+ (-1.30369025e-9) * d4
		+ (4.13908461e-10) * tdb * d4
		+ (9.22652254e-12) * tdb * tdb * d4
		+ (-5.08220384e-9) * v * d4
		+ (-2.24730961e-11) * tdb * v * d4
		+ (1.17139133e-10) * v2 * d4
		+ (6.62154879e-10) * d5
		+ (4.03863260e-13) * tdb * d5
		+ (1.95087203e-12) * v * d5
		+ (-4.73602469e-12) * d6
		+ 5.12733497 * pa
		+ (-0.312788561) * tdb * pa
		+ (-0.0196701861) * tdb * tdb * pa
		+ (9.99690870e-4) * tdb * tdb * tdb * pa
		+ (9.51738512e-6) * tdb * tdb * tdb * tdb * pa
		+ (-4.66426341e-7) * tdb * tdb * tdb * tdb * tdb * pa
		+ 0.548050612 * v * pa
		+ (-0.00330552823) * tdb * v * pa
		+ (-0.00164119440) * tdb * tdb * v * pa
		+ (-5.16670694e-6) * tdb * tdb * tdb * v * pa
		+ (9.52692432e-7) * tdb * tdb * tdb * tdb * v * pa
		+ (-0.0429223622) * v2 * pa
		+ 0.00500845667 * tdb * v2 * pa
		+ (1.00601257e-6) * tdb * tdb * v2 * pa
		+ (-1.81748644e-6) * tdb * tdb * tdb * v2 * pa
		+ (-1.25813502e-3) * v3 * pa
		+ (-1.79330391e-4) * tdb * v3 * pa
		+ (2.34994441e-6) * tdb * tdb * v3 * pa
		+ (1.29735808e-4) * v4 * pa
		+ (1.29064870e-6) * tdb * v4 * pa
		+ (-2.28558686e-6) * v5 * pa
		+ (-0.0369476348) * d * pa
		+ 0.00162325322 * tdb * d * pa
		+ (-3.14279680e-5) * tdb * tdb * d * pa
		+ (2.59835559e-6) * tdb * tdb * tdb * d * pa
		+ (-4.77136523e-8) * tdb * tdb * tdb * tdb * d * pa
		+ (8.64203390e-3) * v * d * pa
		+ (-6.87405181e-4) * tdb * v * d * pa
		+ (-9.13863872e-6) * tdb * tdb * v * d * pa
		+ (5.15916806e-7) * tdb * tdb * tdb * v * d * pa
		+ (-3.59217476e-5) * v2 * d * pa
		+ (3.28696511e-5) * tdb * v2 * d * pa
		+ (-7.10542454e-7) * tdb * tdb * v2 * d * pa
		+ (-1.24382300e-5) * v3 * d * pa
		+ (-7.38584400e-9) * tdb * v3 * d * pa
		+ (2.20609296e-7) * v4 * d * pa
		+ (-7.32469180e-4) * d2 * pa
		+ (-1.87381964e-5) * tdb * d2 * pa
		+ (4.80925239e-6) * tdb * tdb * d2 * pa
		+ (-8.75492040e-8) * tdb * tdb * tdb * d2 * pa
		+ (2.77862930e-5) * v * d2 * pa
		+ (-5.06004592e-6) * tdb * v * d2 * pa
		+ (1.14325367e-7) * tdb * tdb * v * d2 * pa
		+ (2.53016723e-6) * v2 * d2 * pa
		+ (-1.72857035e-8) * tdb * v2 * d2 * pa
		+ (-3.95079398e-8) * v3 * d2 * pa
		+ (-3.59413173e-7) * d3 * pa
		+ (7.04388046e-7) * tdb * d3 * pa
		+ (-1.89309167e-8) * tdb * tdb * d3 * pa
		+ (-4.79768731e-7) * v * d3 * pa
		+ (7.96079978e-9) * tdb * v * d3 * pa
		+ (1.62897058e-9) * v2 * d3 * pa
		+ (3.94367674e-8) * d4 * pa
		+ (-1.18566247e-9) * tdb * d4 * pa
		+ (3.34678041e-10) * v * d4 * pa
		+ (-1.15606447e-10) * d5 * pa
		+ (-2.80626406) * pa * pa
		+ 0.548712484 * tdb * pa * pa
		+ (-0.00399428410) * tdb * tdb * pa * pa
		+ (-9.54009191e-4) * tdb * tdb * tdb * pa * pa
		+ (1.93090978e-5) * tdb * tdb * tdb * tdb * pa * pa
		+ (-0.308806365) * v * pa * pa
		+ 0.0116952364 * tdb * v * pa * pa
		+ (4.95271903e-4) * tdb * tdb * v * pa * pa
		+ (-1.90710882e-5) * tdb * tdb * tdb * v * pa * pa
		+ 0.00210787756 * v2 * pa * pa
		+ (-6.98445738e-4) * tdb * v2 * pa * pa
		+ (2.30109073e-5) * tdb * tdb * v2 * pa * pa
		+ (4.17856590e-4) * v3 * pa * pa
		+ (-1.27043871e-5) * tdb * v3 * pa * pa
		+ (-3.04620472e-6) * v4 * pa * pa
		+ 0.0514507424 * d * pa * pa
		+ (-0.00432510997) * tdb * d * pa * pa
		+ (8.99281156e-5) * tdb * tdb * d * pa * pa
		+ (-7.14663943e-7) * tdb * tdb * tdb * d * pa * pa
		+ (-2.66016305e-4) * v * d * pa * pa
		+ (2.63789586e-4) * tdb * v * d * pa * pa
		+ (-7.01199003e-6) * tdb * tdb * v * d * pa * pa
		+ (-1.06823306e-4) * v2 * d * pa * pa
		+ (3.61341136e-6) * tdb * v2 * d * pa * pa
		+ (2.29748967e-7) * v3 * d * pa * pa
		+ (3.04788893e-4) * d2 * pa * pa
		+ (-6.42070836e-5) * tdb * d2 * pa * pa
		+ (1.16257971e-6) * tdb * tdb * d2 * pa * pa
		+ (7.68023384e-6) * v * d2 * pa * pa
		+ (-5.47446896e-7) * tdb * v * d2 * pa * pa
		+ (-3.59937910e-8) * v2 * d2 * pa * pa
		+ (-4.36497725e-6) * d3 * pa * pa
		+ (1.68737969e-7) * tdb * d3 * pa * pa
		+ (2.67489271e-8) * v * d3 * pa * pa
		+ (3.23926897e-9) * d4 * pa * pa
		+ (-0.0353874123) * pa * pa * pa
		+ (-0.221201190) * tdb * pa * pa * pa
		+ 0.0155126038 * tdb * tdb * pa * pa * pa
		+ (-2.63917279e-4) * tdb * tdb * tdb * pa * pa * pa
		+ 0.0453433455 * v * pa * pa * pa
		+ (-0.00432943862) * tdb * v * pa * pa * pa
		+ (1.45389826e-4) * tdb * tdb * v * pa * pa * pa
		+ (2.17508610e-4) * v2 * pa * pa * pa
		+ (-6.66724702e-5) * tdb * v2 * pa * pa * pa
		+ (3.33217140e-5) * v3 * pa * pa * pa
		+ (-0.00226921615) * d * pa * pa * pa
		+ (3.80261982e-4) * tdb * d * pa * pa * pa
		+ (-5.45314314e-9) * tdb * tdb * d * pa * pa * pa
		+ (-7.96355448e-4) * v * d * pa * pa * pa
		+ (2.53458034e-5) * tdb * v * d * pa * pa * pa
		+ (-6.31223658e-6) * v2 * d * pa * pa * pa
		+ (3.02122035e-4) * d2 * pa * pa * pa
		+ (-4.77403547e-6) * tdb * d2 * pa * pa * pa
		+ (1.73825715e-6) * v * d2 * pa * pa * pa
		+ (-4.09087898e-7) * d3 * pa * pa * pa
		+ 0.614155345 * pa * pa * pa * pa
		+ (-0.0616755931) * tdb * pa * pa * pa * pa
		+ 0.00133374846 * tdb * tdb * pa * pa * pa * pa
		+ 0.00355375387 * v * pa * pa * pa * pa
		+ (-5.13027851e-4) * tdb * v * pa * pa * pa * pa
		+ (1.02449757e-4) * v2 * pa * pa * pa * pa
		+ (-0.00148526421) * d * pa * pa * pa * pa
		+ (-4.11469183e-5) * tdb * d * pa * pa * pa * pa
		+ (-6.80434415e-6) * v * d * pa * pa * pa * pa
		+ (-9.77675906e-6) * d2 * pa * pa * pa * pa
		+ 0.0882773108 * pa * pa * pa * pa * pa
		+ (-0.00301859306) * tdb * pa * pa * pa * pa * pa
		+ 0.00104452989 * v * pa * pa * pa * pa * pa
		+ (2.47090539e-4) * d * pa * pa * pa * pa * pa
		+ 0.00148348065 * pa * pa * pa * pa * pa * pa;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let point_idx = global_id.x;
	let time_idx = global_id.y;

	if (point_idx >= params.num_points || time_idx >= params.num_time_steps) {
		return;
	}

	let flat_index = point_idx * params.num_time_steps + time_idx;

	let w = weather_data[time_idx];
	let c0 = compute_outdoor_mrt(point_idx, time_idx, w.mrt_longwave, w, params.num_time_steps);
	mrt_results[flat_index] = c0.mrt;
	// MRT_COMPONENT_WRITES_START
	short_erf_results[flat_index] = c0.short_erf;
	long_erf_results[flat_index] = c0.long_erf;
	short_dmrt_results[flat_index] = c0.short_dmrt;
	long_dmrt_results[flat_index] = c0.long_dmrt;
	// MRT_COMPONENT_WRITES_END
	let utci0 = compute_utci(w.air_temp, c0.mrt, w.wind_speed, w.rel_humidity);

	// Boundary-averaged UTCI semantics:
	// Average UTCI at the current boundary with the next boundary, clamping at
	// each representative-day boundary so month/day hour 23 duplicates itself.
	let hours_per_day = max(params.num_hours_per_day, 1u);
	let day_start = (time_idx / hours_per_day) * hours_per_day;
	let day_end = min(day_start + hours_per_day - 1u, params.num_time_steps - 1u);
	let next_idx = min(time_idx + 1u, day_end);
	let w1 = weather_data[next_idx];
	let c1 = compute_outdoor_mrt(point_idx, next_idx, w1.mrt_longwave, w1, params.num_time_steps);
	let utci1 = compute_utci(w1.air_temp, c1.mrt, w1.wind_speed, w1.rel_humidity);

	utci_results[flat_index] = 0.5 * (utci0 + utci1);
}
