const EPSILON: f32 = 1e-6;

struct Params {
	num_points: u32,
	total_time_steps: u32,
	start_time_index: u32,
	time_count: u32,
	point_offset: u32,
	_pad0: u32,
	_pad1: u32,
	_pad2: u32,
}

@group(0) @binding(0)
var<storage, read> packed_solar: array<u32>;

@group(0) @binding(1)
var<storage, read> is_sun_up: array<u32>;

@group(0) @binding(2)
var<storage, read_write> out_shading_index: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn read_solar_exposure(point_idx: u32, time_idx: u32, total_time_steps: u32) -> f32 {
	let flat_index = point_idx * total_time_steps + time_idx;
	let word_index = flat_index / 32u;
	let bit_index = flat_index % 32u;
	let solar_bit = (packed_solar[word_index] >> bit_index) & 1u;
	return select(0.0, 1.0, solar_bit == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let point_idx = id.x + params.point_offset;
	if (point_idx >= params.num_points) {
		return;
	}

	let period_start = params.start_time_index;
	let period_end = min(period_start + params.time_count, params.total_time_steps);
	var sunlight_count: u32 = 0u;
	var shaded_count: u32 = 0u;

	for (var time_idx = period_start; time_idx < period_end; time_idx = time_idx + 1u) {
		if (is_sun_up[time_idx] != 0u) {
			sunlight_count = sunlight_count + 1u;
			if (read_solar_exposure(point_idx, time_idx, params.total_time_steps) <= EPSILON) {
				shaded_count = shaded_count + 1u;
			}
		}
	}

	out_shading_index[point_idx] = select(
		1.0,
		f32(shaded_count) / f32(sunlight_count),
		sunlight_count > 0u
	);
}
