// BVH traversal and ray–triangle intersection for GPU exposure.
// Layout matches three-mesh-bvh serialized format: 32 bytes per node, 8 × u32.
// Consuming shader must concatenate this and declare the same @group(1) bindings
// (or use this file as-is with group 1 bindings below).

const UINT32_PER_NODE: u32 = 8u;
const LEAF_FLAG: u32 = 0xFFFFu;

// Group 1: BVH and geometry (shared by solar and sky exposure shaders)
@group(1) @binding(0)
var<storage, read> bvh_nodes: array<u32>;

@group(1) @binding(1)
var<storage, read> bvh_index: array<u32>;

@group(1) @binding(2)
var<storage, read> vertex: array<f32>;

struct BvhParams {
	num_nodes: u32,
	num_vertices: u32,
	num_indices: u32,
	_pad: u32,
}

@group(1) @binding(3)
var<uniform> bvh_params: BvhParams;

// Möller–Trumbore ray–triangle intersection. Returns true if ray hits triangle.
fn ray_triangle_intersect(
	ray_origin: vec3<f32>,
	ray_dir: vec3<f32>,
	v0: vec3<f32>,
	v1: vec3<f32>,
	v2: vec3<f32>
) -> bool {
	let edge1 = v1 - v0;
	let edge2 = v2 - v0;
	let h = cross(ray_dir, edge2);
	let a = dot(edge1, h);
	let eps: f32 = 1e-6;
	if (abs(a) < eps) {
		return false;
	}
	let f = 1.0 / a;
	let s = ray_origin - v0;
	let u = f * dot(s, h);
	if (u < -eps || u > 1.0 + eps) {
		return false;
	}
	let q = cross(s, edge1);
	let v = f * dot(ray_dir, q);
	if (v < -eps || u + v > 1.0 + eps) {
		return false;
	}
	let t = f * dot(edge2, q);
	return t > eps;
}

fn read_vertex_at(i: u32) -> vec3<f32> {
	let base = i * 3u;
	return vec3<f32>(vertex[base], vertex[base + 1u], vertex[base + 2u]);
}

// Ray–AABB intersection. Returns true if ray hits the box (t in [0, inf)).
fn ray_aabb_intersect(origin: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
	let eps: f32 = 1e-8;
	var t_near: f32 = -1e30;
	var t_far: f32 = 1e30;

	if (abs(dir.x) < eps) {
		if (origin.x < bmin.x || origin.x > bmax.x) {
			return false;
		}
	} else {
		let inv_x = 1.0 / dir.x;
		let tx1 = (bmin.x - origin.x) * inv_x;
		let tx2 = (bmax.x - origin.x) * inv_x;
		t_near = max(t_near, min(tx1, tx2));
		t_far = min(t_far, max(tx1, tx2));
	}

	if (abs(dir.y) < eps) {
		if (origin.y < bmin.y || origin.y > bmax.y) {
			return false;
		}
	} else {
		let inv_y = 1.0 / dir.y;
		let ty1 = (bmin.y - origin.y) * inv_y;
		let ty2 = (bmax.y - origin.y) * inv_y;
		t_near = max(t_near, min(ty1, ty2));
		t_far = min(t_far, max(ty1, ty2));
	}

	if (abs(dir.z) < eps) {
		if (origin.z < bmin.z || origin.z > bmax.z) {
			return false;
		}
	} else {
		let inv_z = 1.0 / dir.z;
		let tz1 = (bmin.z - origin.z) * inv_z;
		let tz2 = (bmax.z - origin.z) * inv_z;
		t_near = max(t_near, min(tz1, tz2));
		t_far = min(t_far, max(tz1, tz2));
	}

	return t_near <= t_far && t_far >= 0.0;
}

// Traverse BVH and test ray against triangles at leaves. Returns true if any hit.
fn bvh_intersects_any(origin: vec3<f32>, direction: vec3<f32>) -> bool {
	let num_nodes = bvh_params.num_nodes;
	if (num_nodes == 0u) {
		return false;
	}
	// Stack of node base indices (u32 offset into bvh_nodes). Max depth 64.
	var stack: array<u32, 64>;
	var stack_size: u32 = 0u;
	// Root node: base index 0 (first 8 u32s)
	stack[0] = 0u;
	stack_size = 1u;

	while (stack_size > 0u) {
		stack_size = stack_size - 1u;
		let base = stack[stack_size];

		let bmin = vec3<f32>(
			bitcast<f32>(bvh_nodes[base]),
			bitcast<f32>(bvh_nodes[base + 1u]),
			bitcast<f32>(bvh_nodes[base + 2u])
		);
		let bmax = vec3<f32>(
			bitcast<f32>(bvh_nodes[base + 3u]),
			bitcast<f32>(bvh_nodes[base + 4u]),
			bitcast<f32>(bvh_nodes[base + 5u])
		);
		if (!ray_aabb_intersect(origin, direction, bmin, bmax)) {
			continue;
		}

		let offset_val = bvh_nodes[base + 6u];
		let count_and_leaf = bvh_nodes[base + 7u];
		let tri_count = count_and_leaf & 0xFFFFu;
		let is_leaf = (count_and_leaf >> 16u) == LEAF_FLAG;

		if (is_leaf) {
			// Test ray against each triangle in this leaf
			let tri_offset = offset_val;
			for (var i: u32 = 0u; i < tri_count; i = i + 1u) {
				let idx = (tri_offset + i) * 3u;
				let i0 = bvh_index[idx];
				let i1 = bvh_index[idx + 1u];
				let i2 = bvh_index[idx + 2u];
				let v0 = read_vertex_at(i0);
				let v1 = read_vertex_at(i1);
				let v2 = read_vertex_at(i2);
				if (ray_triangle_intersect(origin, direction, v0, v1, v2)) {
					return true;
				}
			}
		} else {
			// Internal node: push right then left so we process left first
			let right_base = base + offset_val * UINT32_PER_NODE;
			let left_base = base + UINT32_PER_NODE;
			if (right_base < num_nodes * UINT32_PER_NODE && stack_size < 63u) {
				stack[stack_size] = right_base;
				stack_size = stack_size + 1u;
			}
			if (left_base < num_nodes * UINT32_PER_NODE && stack_size < 63u) {
				stack[stack_size] = left_base;
				stack_size = stack_size + 1u;
			}
		}
	}
	return false;
}
