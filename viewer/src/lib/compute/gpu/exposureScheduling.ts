import { createPointDispatchChunks, type PointDispatchChunk } from '$lib/compute/gpu/gpu-pipeline';

export type ExposureSchedulingMode = 'single-submit' | 'chunked';

export interface ExposureSchedulingOptions {
	mode: ExposureSchedulingMode;
	maxWorkgroupsPerSlice: number;
	yieldBetweenSlices: boolean;
}

export const DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE = 2048;
export const MIN_EXPOSURE_MAX_WORKGROUPS_PER_SLICE = 1;
export const MAX_EXPOSURE_MAX_WORKGROUPS_PER_SLICE = 65_535;

export const DEFAULT_EXPOSURE_SCHEDULING: ExposureSchedulingOptions = {
	mode: 'chunked',
	maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
	yieldBetweenSlices: true
};

export function parseExposureSchedulingFromSearchParams(
	params: URLSearchParams
): ExposureSchedulingOptions {
	const mode =
		params.get('utciExposureSchedule') === 'single-submit' ? 'single-submit' : 'chunked';
	const rawMaxWorkgroups = Number(params.get('utciExposureMaxWorkgroupsPerSlice'));
	const maxWorkgroupsPerSlice =
		Number.isFinite(rawMaxWorkgroups) &&
		rawMaxWorkgroups >= MIN_EXPOSURE_MAX_WORKGROUPS_PER_SLICE
			? Math.min(Math.floor(rawMaxWorkgroups), MAX_EXPOSURE_MAX_WORKGROUPS_PER_SLICE)
			: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE;
	const yieldBetweenSlices = params.get('utciExposureYieldBetweenSlices') !== '0';

	return {
		mode,
		maxWorkgroupsPerSlice,
		yieldBetweenSlices
	};
}

export function areExposureSchedulingOptionsEqual(
	left: ExposureSchedulingOptions | undefined,
	right: ExposureSchedulingOptions | undefined
): boolean {
	const resolvedLeft = left ?? DEFAULT_EXPOSURE_SCHEDULING;
	const resolvedRight = right ?? DEFAULT_EXPOSURE_SCHEDULING;
	if (resolvedLeft.mode !== resolvedRight.mode) {
		return false;
	}
	if (resolvedLeft.mode === 'single-submit') {
		return true;
	}
	return (
		resolvedLeft.maxWorkgroupsPerSlice === resolvedRight.maxWorkgroupsPerSlice &&
		resolvedLeft.yieldBetweenSlices === resolvedRight.yieldBetweenSlices
	);
}

export function buildExposurePointSlices(params: {
	numPoints: number;
	workgroupSize: number;
	maxWorkgroupsPerSlice: number;
}): PointDispatchChunk[] {
	const { numPoints, workgroupSize, maxWorkgroupsPerSlice } = params;
	if (numPoints <= 0 || workgroupSize <= 0 || maxWorkgroupsPerSlice <= 0) {
		throw new Error('numPoints, workgroupSize, and maxWorkgroupsPerSlice must be positive');
	}
	return createPointDispatchChunks(numPoints, workgroupSize, maxWorkgroupsPerSlice);
}
