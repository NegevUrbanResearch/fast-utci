// Solar exposure compute shader
// Layout matches the architecture described in the WebGPU migration plan:
// - X dimension: grid points
// - Y dimension: time steps (month × hour)
//
// All positions and directions are expressed in the Three.js world frame:
// X = East/right, Y = Up, Z = North/forward (Y-up). CPU packing code rotates
// sun vectors from the Python Z-up convention into this frame before upload.

struct Vec3F32 {
	x: f32,
	y: f32,
	z: f32,
};

@group(0) @binding(0)
var<storage, read> grid_points: array<Vec3F32>;

@group(0) @binding(1)
var<storage, read> sun_vectors: array<Vec3F32>;

// Bit-packed solar exposure: 1 bit per (point, time_step) result.
// Word index = flat_index / 32, bit index = flat_index % 32.
// Bit = 1 means exposed (sun visible), 0 means occluded.
@group(0) @binding(2)
var<storage, read_write> solar_exposure: array<atomic<u32>>;

struct Params {
	num_points: u32,
	num_time_steps: u32,
	point_offset: u32,
	_pad0: u32,
}

@group(0) @binding(3)
var<uniform> params: Params;

// When this shader is concatenated with bvh_raycast.wgsl, @group(1) and bvh_intersects_any are provided there.
// Set to true to force-write a known bit at (0,0) to verify the compute buffer is the one we read back (debug zeros).
const PROBE_FORCE_WRITE: bool = false;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let point_idx = global_id.x + params.point_offset;
	let time_idx = global_id.y;

	if (point_idx >= params.num_points || time_idx >= params.num_time_steps) {
		return;
	}

	let flat_index = point_idx * params.num_time_steps + time_idx;
	let word_idx = flat_index / 32u;
	let bit_idx = flat_index % 32u;

	if (PROBE_FORCE_WRITE && point_idx == 0u && time_idx == 0u) {
		atomicOr(&solar_exposure[word_idx], 1u << bit_idx);
		return;
	}

	let origin = vec3<f32>(
		grid_points[point_idx].x,
		grid_points[point_idx].y,
		grid_points[point_idx].z
	);

	let sun = vec3<f32>(
		sun_vectors[time_idx].x,
		sun_vectors[time_idx].y,
		sun_vectors[time_idx].z
	);

	// Skip BVH traversal for nighttime/invalid vectors.
	let sun_len2 = dot(sun, sun);
	if (sun_len2 < 1e-10 || sun.y <= 0.0) {
		// Occluded (bit stays 0 from zero-fill). No atomicOr needed.
		return;
	}

	// Match Python semantics: launch rays from the sample point itself.
	let ray_origin = origin;
	let hit = bvh_intersects_any(ray_origin, sun);

	// Exposed (not hit) → set bit to 1; occluded (hit) → leave bit as 0.
	if (!hit) {
		atomicOr(&solar_exposure[word_idx], 1u << bit_idx);
	}
}
