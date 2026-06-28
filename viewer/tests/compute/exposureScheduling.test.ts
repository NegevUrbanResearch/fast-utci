import { describe, expect, it } from 'vitest';
import {
	DEFAULT_EXPOSURE_SCHEDULING,
	DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
	MAX_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
	areExposureSchedulingOptionsEqual,
	buildExposurePointSlices,
	parseExposureSchedulingFromSearchParams
} from '../../src/lib/compute/gpu/exposureScheduling';

describe('exposureScheduling', () => {
	it('defaults to chunked exposure scheduling', () => {
		expect(parseExposureSchedulingFromSearchParams(new URLSearchParams(''))).toEqual({
			mode: 'chunked',
			maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
			yieldBetweenSlices: true
		});
	});

	it('parses the explicit single-submit rollback flag', () => {
		const params = new URLSearchParams(
			'utciExposureSchedule=single-submit&utciExposureMaxWorkgroupsPerSlice=8192'
		);

		expect(parseExposureSchedulingFromSearchParams(params)).toEqual({
			mode: 'single-submit',
			maxWorkgroupsPerSlice: 8192,
			yieldBetweenSlices: true
		});
	});

	it('parses the chunked query flag and clamps slice size', () => {
		const params = new URLSearchParams(
			'utciExposureSchedule=chunked&utciExposureMaxWorkgroupsPerSlice=8192'
		);

		expect(parseExposureSchedulingFromSearchParams(params)).toEqual({
			mode: 'chunked',
			maxWorkgroupsPerSlice: 8192,
			yieldBetweenSlices: true
		});
	});

	it('ignores invalid mode and invalid slice sizes', () => {
		const params = new URLSearchParams(
			'utciExposureSchedule=banana&utciExposureMaxWorkgroupsPerSlice=-1'
		);

		expect(parseExposureSchedulingFromSearchParams(params)).toEqual({
			mode: 'chunked',
			maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
			yieldBetweenSlices: true
		});
	});

	it('clamps oversized slice size and parses disabled yielding', () => {
		const params = new URLSearchParams(
			'utciExposureMaxWorkgroupsPerSlice=999999&utciExposureYieldBetweenSlices=0'
		);

		expect(parseExposureSchedulingFromSearchParams(params)).toEqual({
			mode: 'chunked',
			maxWorkgroupsPerSlice: MAX_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
			yieldBetweenSlices: false
		});
	});

	it('floors decimal slice sizes', () => {
		const params = new URLSearchParams('utciExposureMaxWorkgroupsPerSlice=12.75');

		expect(parseExposureSchedulingFromSearchParams(params).maxWorkgroupsPerSlice).toBe(12);
	});

	it('compares exposure scheduling options with mode-aware chunk fields', () => {
		const explicitDefault = {
			mode: 'chunked' as const,
			maxWorkgroupsPerSlice: DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE,
			yieldBetweenSlices: true
		};
		const explicitSingleSubmit = {
			mode: 'single-submit' as const,
			maxWorkgroupsPerSlice: 8192,
			yieldBetweenSlices: true
		};

		expect(areExposureSchedulingOptionsEqual(undefined, DEFAULT_EXPOSURE_SCHEDULING)).toBe(true);
		expect(areExposureSchedulingOptionsEqual(explicitDefault, DEFAULT_EXPOSURE_SCHEDULING)).toBe(
			true
		);
		expect(
			areExposureSchedulingOptionsEqual(explicitSingleSubmit, {
				mode: 'single-submit',
				maxWorkgroupsPerSlice: 128,
				yieldBetweenSlices: false
			})
		).toBe(true);
		expect(areExposureSchedulingOptionsEqual(explicitSingleSubmit, DEFAULT_EXPOSURE_SCHEDULING)).toBe(
			false
		);
		expect(
			areExposureSchedulingOptionsEqual(explicitDefault, explicitSingleSubmit)
		).toBe(false);
		expect(
			areExposureSchedulingOptionsEqual(
				{
					mode: 'chunked',
					maxWorkgroupsPerSlice: 128,
					yieldBetweenSlices: true
				},
				{
					mode: 'chunked',
					maxWorkgroupsPerSlice: 256,
					yieldBetweenSlices: true
				}
			)
		).toBe(false);
		expect(
			areExposureSchedulingOptionsEqual(
				{
					mode: 'chunked',
					maxWorkgroupsPerSlice: 128,
					yieldBetweenSlices: true
				},
				{
					mode: 'chunked',
					maxWorkgroupsPerSlice: 128,
					yieldBetweenSlices: false
				}
			)
		).toBe(false);
	});

	it('builds bounded point slices without dropping points', () => {
		const slices = buildExposurePointSlices({
			numPoints: 1_000,
			workgroupSize: 64,
			maxWorkgroupsPerSlice: 4
		});

		expect(slices).toEqual([
			{ pointOffset: 0, pointCount: 256, workgroupsX: 4 },
			{ pointOffset: 256, pointCount: 256, workgroupsX: 4 },
			{ pointOffset: 512, pointCount: 256, workgroupsX: 4 },
			{ pointOffset: 768, pointCount: 232, workgroupsX: 4 }
		]);
	});

	it('rejects non-positive point slice inputs', () => {
		expect(() =>
			buildExposurePointSlices({ numPoints: 0, workgroupSize: 64, maxWorkgroupsPerSlice: 4 })
		).toThrowError('numPoints, workgroupSize, and maxWorkgroupsPerSlice must be positive');
		expect(() =>
			buildExposurePointSlices({ numPoints: 1_000, workgroupSize: 0, maxWorkgroupsPerSlice: 4 })
		).toThrowError('numPoints, workgroupSize, and maxWorkgroupsPerSlice must be positive');
		expect(() =>
			buildExposurePointSlices({ numPoints: 1_000, workgroupSize: 64, maxWorkgroupsPerSlice: 0 })
		).toThrowError('numPoints, workgroupSize, and maxWorkgroupsPerSlice must be positive');
	});
});
