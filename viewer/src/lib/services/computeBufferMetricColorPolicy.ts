import * as THREE from 'three';
import type { F32MetricType } from '$lib/compute/on-demand/onDemandOutputFormat';
import { mapShadingIndexToColor, mapUTCIToColor } from '$lib/services/colorScale';

export type ComputeBufferMetricValueRange = { min: number; max: number };

export type ComputeBufferMetricColorPolicy = {
	metricType: F32MetricType;
	valueRange: ComputeBufferMetricValueRange;
};

export const COMPUTE_BUFFER_COLOR_LUT_SIZE = 256;
export const COMPUTE_BUFFER_COLOR_LUT_BYTES = COMPUTE_BUFFER_COLOR_LUT_SIZE * 4;

export function resolveComputeBufferMetricColorPolicy(params: {
	metricType?: F32MetricType;
	utciRange: ComputeBufferMetricValueRange;
	valueRange?: ComputeBufferMetricValueRange;
}): ComputeBufferMetricColorPolicy {
	const metricType = params.metricType ?? 'utci';
	const suppliedRange = params.valueRange ?? params.utciRange;
	if (metricType === 'shading_index') {
		return {
			metricType,
			valueRange: { min: 0, max: 1 }
		};
	}

	return {
		metricType,
		valueRange: { ...suppliedRange }
	};
}

export function createComputeBufferMetricColorLutTexture(
	metricType: F32MetricType
): THREE.DataTexture {
	const bytes = new Uint8Array(COMPUTE_BUFFER_COLOR_LUT_SIZE * 4);
	for (let index = 0; index < COMPUTE_BUFFER_COLOR_LUT_SIZE; index += 1) {
		const t = index / (COMPUTE_BUFFER_COLOR_LUT_SIZE - 1);
		const color =
			metricType === 'shading_index'
				? mapShadingIndexToColor(t, 0, 1)
				: mapUTCIToColor(t, 0, 1);
		const offset = index * 4;
		bytes[offset] = Math.round(color.r * 255);
		bytes[offset + 1] = Math.round(color.g * 255);
		bytes[offset + 2] = Math.round(color.b * 255);
		bytes[offset + 3] = 255;
	}

	const lut = new THREE.DataTexture(
		bytes,
		COMPUTE_BUFFER_COLOR_LUT_SIZE,
		1,
		THREE.RGBAFormat,
		THREE.UnsignedByteType
	);
	lut.needsUpdate = true;
	lut.flipY = false;
	lut.generateMipmaps = false;
	lut.magFilter = THREE.LinearFilter;
	lut.minFilter = THREE.LinearFilter;
	lut.wrapS = THREE.ClampToEdgeWrapping;
	lut.wrapT = THREE.ClampToEdgeWrapping;
	lut.colorSpace = THREE.SRGBColorSpace;
	return lut;
}
