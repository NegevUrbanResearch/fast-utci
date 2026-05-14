import { describe, expect, it } from 'vitest';
import { buildDebugPerformanceComparisonRows } from '$lib/performance/debugPerformanceComparison';

describe('debugPerformanceComparison', () => {
	it('builds a valid .bin vs WebGPU comparison table from debug diagnostics', () => {
		const rows = buildDebugPerformanceComparisonRows({
			binComparisonEnabled: true,
			binComparisonValid: true,
			pythonBaselineStatus: 'valid',
			pythonSelectedHourMeanUtci: 28.12,
			webgpuSelectedHourMeanUtci: 28.15,
			pythonDerivedOneHourMs: 8127.55,
			timings: {
				firstSelectedHourVisibleMs: 4200,
				oneHourDispatchMs: 86.49,
				debugReadbackMs: 12.25
			}
		});

		expect(rows).toEqual([
			{ metric: 'Mean UTCI', python: '28.12 C', webgpu: '28.15 C', diff: '+0.03 C' },
			{ metric: 'Visible time', python: '8.1 s', webgpu: '4.2 s', diff: '-3.9 s' }
		]);
	});

	it('reports unavailable comparison when the .bin baseline is invalid', () => {
		const rows = buildDebugPerformanceComparisonRows({
			binComparisonEnabled: true,
			binComparisonValid: false,
			pythonBaselineStatus: 'invalid-month',
			timings: {
				oneHourDispatchMs: 86.49,
				firstSelectedHourVisibleMs: 4200
			}
		});

		expect(rows).toEqual([
			{
				metric: 'Mean UTCI',
				python: 'Unavailable for this selection',
				webgpu: '-',
				diff: '-'
			},
			{ metric: 'Visible time', python: '-', webgpu: '4.2 s', diff: '-' }
		]);
	});

	it('formats missing and non-finite values without throwing', () => {
		expect(() =>
			buildDebugPerformanceComparisonRows({
				binComparisonEnabled: true,
				binComparisonValid: true,
				pythonBaselineStatus: 'valid',
				pythonSelectedHourMeanUtci: Number.POSITIVE_INFINITY,
				webgpuSelectedHourMeanUtci: Number.NaN,
				pythonDerivedOneHourMs: Number.NaN,
				timings: {
					oneHourDispatchMs: Number.NaN,
					firstSelectedHourVisibleMs: Number.POSITIVE_INFINITY
				}
			})
		).not.toThrow();

		expect(
			buildDebugPerformanceComparisonRows({
				binComparisonEnabled: true,
				binComparisonValid: true,
				pythonBaselineStatus: 'valid',
				pythonSelectedHourMeanUtci: Number.POSITIVE_INFINITY,
				webgpuSelectedHourMeanUtci: Number.NaN,
				pythonDerivedOneHourMs: Number.NaN,
				timings: {
					oneHourDispatchMs: Number.NaN,
					firstSelectedHourVisibleMs: Number.POSITIVE_INFINITY
				}
			})
		).toEqual([
			{ metric: 'Mean UTCI', python: '-', webgpu: '-', diff: '-' },
			{ metric: 'Visible time', python: 'Measuring', webgpu: 'Measuring', diff: '-' }
		]);
	});
});
