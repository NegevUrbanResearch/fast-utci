import { describe, it, expect } from 'vitest';
import { getInitialAnalysisId, parseMainRouteGridResolution } from '$lib/utils/analysisQuery';

describe('analysis query', () => {
	it('uses default when no query param present', () => {
		expect(getInitialAnalysisId('', 'Ben-Gurion/20250815_grid_2m_fullday')).toBe(
			'Ben-Gurion/20250815_grid_2m_fullday'
		);
	});

	it('uses analysis query param when present', () => {
		expect(getInitialAnalysisId('?analysis=foo', 'default')).toBe('foo');
	});

	it('parses supported main route grid resolution values', () => {
		expect(parseMainRouteGridResolution('?gridResolution=0.5')).toBe(0.5);
		expect(parseMainRouteGridResolution('?gridResolution=1')).toBe(1);
		expect(parseMainRouteGridResolution('?gridResolution=2')).toBe(2);
	});

	it('falls back for unsupported main route grid resolution values', () => {
		expect(parseMainRouteGridResolution('?gridResolution=3')).toBe(2);
		expect(parseMainRouteGridResolution('?gridResolution=bad')).toBe(2);
		expect(parseMainRouteGridResolution('', 4)).toBe(4);
	});
});
