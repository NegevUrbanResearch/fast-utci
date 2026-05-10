import { describe, expect, it } from 'vitest';
import { getMainRouteOverlayGating } from '../../src/routes/mainRouteOverlayGating';

describe('mainRouteOverlayGating', () => {
	it('suppresses the full-screen base live overlay when a visible base surface already exists', () => {
		const result = getMainRouteOverlayGating({
			modelLoading: false,
			useLiveUtciOnMainRoute: true,
			baseLiveLoading: true,
			baseHasVisibleLiveSurface: true,
			isComparing: false,
			comparisonModelLoading: false,
			comparisonLiveLoading: false,
			comparisonHasVisibleLiveSurface: false
		});

		expect(result.baseNeedsLiveComputeOverlay).toBe(false);
		expect(result.showOverlay).toBe(false);
	});

	it('shows the full-screen base live overlay when no visible base surface exists yet', () => {
		const result = getMainRouteOverlayGating({
			modelLoading: false,
			useLiveUtciOnMainRoute: true,
			baseLiveLoading: true,
			baseHasVisibleLiveSurface: false,
			isComparing: false,
			comparisonModelLoading: false,
			comparisonLiveLoading: false,
			comparisonHasVisibleLiveSurface: false
		});

		expect(result.baseNeedsLiveComputeOverlay).toBe(true);
		expect(result.showOverlay).toBe(true);
	});

	it('suppresses the comparison-side full-screen live overlay when a visible comparison surface already exists', () => {
		const result = getMainRouteOverlayGating({
			modelLoading: false,
			useLiveUtciOnMainRoute: true,
			baseLiveLoading: false,
			baseHasVisibleLiveSurface: true,
			isComparing: true,
			comparisonModelLoading: false,
			comparisonLiveLoading: true,
			comparisonHasVisibleLiveSurface: true
		});

		expect(result.comparisonNeedsLiveComputeOverlay).toBe(false);
		expect(result.showOverlay).toBe(false);
		expect(result.showComparisonModeOverlay).toBe(false);
	});

	it('keeps comparison-mode overlay behavior for comparison refreshes without a visible comparison surface', () => {
		const result = getMainRouteOverlayGating({
			modelLoading: false,
			useLiveUtciOnMainRoute: true,
			baseLiveLoading: false,
			baseHasVisibleLiveSurface: true,
			isComparing: true,
			comparisonModelLoading: false,
			comparisonLiveLoading: true,
			comparisonHasVisibleLiveSurface: false
		});

		expect(result.comparisonNeedsLiveComputeOverlay).toBe(true);
		expect(result.showOverlay).toBe(true);
		expect(result.showComparisonModeOverlay).toBe(true);
	});
});
