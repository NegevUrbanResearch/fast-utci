import { describe, it, expect } from 'vitest';
import { getInitialAnalysisId } from '$lib/utils/analysisQuery';

describe('analysis query', () => {
	it('uses default when no query param present', () => {
		expect(getInitialAnalysisId('', 'Ben-Gurion/20250815_grid_2m_fullday')).toBe(
			'Ben-Gurion/20250815_grid_2m_fullday'
		);
	});

	it('uses analysis query param when present', () => {
		expect(getInitialAnalysisId('?analysis=foo', 'default')).toBe('foo');
	});
});
