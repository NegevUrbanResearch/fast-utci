const F32_MAX_VALUE: f32 = 3.4028234663852886e38;

struct Params {
	input_count: u32,
	input_offset: u32,
	output_offset: u32,
	input_stride: u32,
}

struct RangeSummary {
	min_value: f32,
	max_value: f32,
	valid_count: u32,
	_pad: u32,
}

@group(0) @binding(0)
var<storage, read> source_values: array<f32>;

@group(0) @binding(1)
var<storage, read> source_ranges: array<RangeSummary>;

@group(0) @binding(2)
var<storage, read_write> output_ranges: array<RangeSummary>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> workgroup_min: array<f32, 256>;
var<workgroup> workgroup_max: array<f32, 256>;
var<workgroup> workgroup_count: array<u32, 256>;

fn is_valid_value(value: f32) -> bool {
	return value == value && abs(value) <= F32_MAX_VALUE;
}

fn reduce_workgroup(local_id: u32) {
	workgroupBarrier();
	var stride = 128u;
	loop {
		if (local_id < stride) {
			workgroup_min[local_id] = min(workgroup_min[local_id], workgroup_min[local_id + stride]);
			workgroup_max[local_id] = max(workgroup_max[local_id], workgroup_max[local_id + stride]);
			workgroup_count[local_id] = workgroup_count[local_id] + workgroup_count[local_id + stride];
		}
		workgroupBarrier();
		if (stride == 1u) {
			break;
		}
		stride = stride / 2u;
	}
}

fn write_workgroup_summary(output_index: u32) {
	output_ranges[output_index] = RangeSummary(
		workgroup_min[0],
		workgroup_max[0],
		workgroup_count[0],
		0u
	);
}

@compute @workgroup_size(256)
fn reduce_values(
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_id) local_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>
) {
	let lane = local_id.x;
	if (global_id.x < params.input_count) {
		let value_index = params.input_offset + global_id.x * params.input_stride;
		let value = source_values[value_index];
		if (is_valid_value(value)) {
			workgroup_min[lane] = value;
			workgroup_max[lane] = value;
			workgroup_count[lane] = 1u;
		} else {
			workgroup_min[lane] = F32_MAX_VALUE;
			workgroup_max[lane] = -F32_MAX_VALUE;
			workgroup_count[lane] = 0u;
		}
	} else {
		workgroup_min[lane] = F32_MAX_VALUE;
		workgroup_max[lane] = -F32_MAX_VALUE;
		workgroup_count[lane] = 0u;
	}

	reduce_workgroup(lane);
	if (lane == 0u) {
		write_workgroup_summary(params.output_offset + workgroup_id.x);
	}
}

@compute @workgroup_size(256)
fn reduce_ranges(
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_id) local_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>
) {
	let lane = local_id.x;
	if (global_id.x < params.input_count) {
		let range_index = params.input_offset + global_id.x * params.input_stride;
		let summary = source_ranges[range_index];
		if (
			summary.valid_count > 0u &&
			is_valid_value(summary.min_value) &&
			is_valid_value(summary.max_value)
		) {
			workgroup_min[lane] = summary.min_value;
			workgroup_max[lane] = summary.max_value;
			workgroup_count[lane] = summary.valid_count;
		} else {
			workgroup_min[lane] = F32_MAX_VALUE;
			workgroup_max[lane] = -F32_MAX_VALUE;
			workgroup_count[lane] = 0u;
		}
	} else {
		workgroup_min[lane] = F32_MAX_VALUE;
		workgroup_max[lane] = -F32_MAX_VALUE;
		workgroup_count[lane] = 0u;
	}

	reduce_workgroup(lane);
	if (lane == 0u) {
		write_workgroup_summary(params.output_offset + workgroup_id.x);
	}
}
