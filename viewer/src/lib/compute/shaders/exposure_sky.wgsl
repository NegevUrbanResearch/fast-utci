// Sky exposure compute shader
// One work item per grid point; inner loop over 145 Tregenza dome directions.
//
// All positions and directions are interpreted in the Three.js world frame:
// X = East/right, Y = Up, Z = North/forward (Y-up). The CPU packing step is
// responsible for rotating the static Tregenza dome vectors from their
// original Z-up definition into this Y-up frame before upload.

struct Vec3F32 {
	x: f32,
	y: f32,
	z: f32,
};

@group(0) @binding(0)
var<storage, read> grid_points: array<Vec3F32>;

@group(0) @binding(1)
var<storage, read> dome_vectors: array<Vec3F32>;

@group(0) @binding(2)
var<storage, read> dome_weights: array<f32>;

@group(0) @binding(3)
var<storage, read_write> sky_exposure: array<f32>;

struct SkyParams {
	num_points: u32,
	num_patches: u32,
}

@group(0) @binding(4)
var<uniform> params: SkyParams;

// When this shader is concatenated with bvh_raycast.wgsl, @group(1) and bvh_intersects_any are provided there.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let point_idx = global_id.x;
	if (point_idx >= params.num_points) {
		return;
	}

	let origin = vec3<f32>(
		grid_points[point_idx].x,
		grid_points[point_idx].y,
		grid_points[point_idx].z
	);

	var sky_view: f32 = 0.0;

	for (var i: u32 = 0u; i < params.num_patches; i = i + 1u) {
		let dir = vec3<f32>(
			dome_vectors[i].x,
			dome_vectors[i].y,
			dome_vectors[i].z
		);
		let weight = dome_weights[i];
		// Match Python semantics: launch rays from the sample point itself.
		let ray_origin = origin;
		let hit = bvh_intersects_any(ray_origin, dir);
		// If there is no hit, this patch contributes its weight
		sky_view = sky_view + select(weight, 0.0, hit);
	}

	sky_exposure[point_idx] = sky_view;
}

