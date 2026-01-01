/**
 * Comparison Store
 *
 * ABOUTME: Svelte store for managing comparison mode state between base and scenario analyses.
 * Handles curtain position, comparison analysis loading, and comparison lifecycle.
 */

import { writable, derived, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import type { Analysis } from '$lib/types/analysis';
import { loadAnalysis } from '$lib/services/dataLoader';
import { analysisStore } from './analysisStore';

/**
 * Comparison state interface
 */
export interface ComparisonState {
	/** Whether comparison mode is active */
	isComparing: boolean;
	/** The comparison scenario analysis ID */
	comparisonAnalysisId: string | null;
	/** Loaded comparison analysis data */
	comparisonAnalysis: Analysis | null;
	/** Curtain position (0-1), 0 = full base visible, 1 = full comparison visible */
	curtainPosition: number;
	/** Whether comparison analysis is loading */
	isLoading: boolean;
	/** Error message if comparison loading failed */
	error: string | null;
}

/**
 * Default comparison state
 */
const defaultComparisonState: ComparisonState = {
	isComparing: false,
	comparisonAnalysisId: null,
	comparisonAnalysis: null,
	curtainPosition: 0.5,
	isLoading: false,
	error: null
};

/**
 * Comparison store
 */
export const comparisonStore: Writable<ComparisonState> = writable<ComparisonState>(
	defaultComparisonState
);

/**
 * Derived store: is comparison active
 */
export const isComparing: Readable<boolean> = derived(
	comparisonStore,
	($comparison) => $comparison.isComparing
);

/**
 * Derived store: curtain position
 */
export const curtainPosition: Readable<number> = derived(
	comparisonStore,
	($comparison) => $comparison.curtainPosition
);

/**
 * Derived store: comparison analysis
 */
export const comparisonAnalysis: Readable<Analysis | null> = derived(
	comparisonStore,
	($comparison) => $comparison.comparisonAnalysis
);

/**
 * Unified UTCI range interface for comparison mode
 */
export interface UnifiedUtciRange {
	utciMin: number;
	utciMax: number;
}

// Import viewerStore for colorMode awareness
import { viewerStore } from './viewerStore';

/**
 * Derived store: unified UTCI range for comparison mode
 * 
 * When comparing two analyses, this store provides a unified min/max range
 * that encompasses both analyses, ensuring consistent color mapping across both views.
 * Respects the current colorMode - uses per-hour statistics when in 'discrete' mode.
 * Returns null when not in comparison mode.
 */
export const unifiedUtciRange: Readable<UnifiedUtciRange | null> = derived(
	[comparisonStore, analysisStore, viewerStore],
	([$comparison, $baseAnalysis, $viewer]) => {
		// Only calculate unified range when actively comparing with both analyses loaded
		if (!$comparison.isComparing || !$comparison.comparisonAnalysis || !$baseAnalysis) {
			return null;
		}

		const colorMode = $viewer.colorMode;
		const hourIndex = $viewer.currentHour;

		let baseMin: number;
		let baseMax: number;
		let comparisonMin: number;
		let comparisonMax: number;

		if (colorMode === 'discrete') {
			// Per-hour mode: use hour-specific statistics if available
			const baseHourStats = $baseAnalysis.metadata.hour_statistics?.[hourIndex];
			const comparisonHourStats = $comparison.comparisonAnalysis.metadata.hour_statistics?.[hourIndex];

			baseMin = baseHourStats?.min ?? $baseAnalysis.metadata.utci_range.min;
			baseMax = baseHourStats?.max ?? $baseAnalysis.metadata.utci_range.max;
			comparisonMin = comparisonHourStats?.min ?? $comparison.comparisonAnalysis.metadata.utci_range.min;
			comparisonMax = comparisonHourStats?.max ?? $comparison.comparisonAnalysis.metadata.utci_range.max;
		} else {
			// Normalized mode: use full-day range
			baseMin = $baseAnalysis.metadata.utci_range.min;
			baseMax = $baseAnalysis.metadata.utci_range.max;
			comparisonMin = $comparison.comparisonAnalysis.metadata.utci_range.min;
			comparisonMax = $comparison.comparisonAnalysis.metadata.utci_range.max;
		}

		// Calculate unified range that encompasses both analyses
		const unifiedMin = Math.min(baseMin, comparisonMin);
		const unifiedMax = Math.max(baseMax, comparisonMax);

		return {
			utciMin: unifiedMin,
			utciMax: unifiedMax
		};
	}
);

/**
 * Start comparison mode with the given analysis ID
 *
 * @param comparisonAnalysisId - The analysis ID to compare against base
 */
export async function startComparison(comparisonAnalysisId: string): Promise<void> {
	// Check if we're already comparing the same analysis
	const currentState = get(comparisonStore);
	if (currentState.isComparing && currentState.comparisonAnalysisId === comparisonAnalysisId) {
		console.log(`[COMPARISON] Already comparing with ${comparisonAnalysisId}`);
		return;
	}

	// Check if the base analysis is loaded
	const baseAnalysis = get(analysisStore);
	if (!baseAnalysis) {
		console.warn('[COMPARISON] Cannot start comparison: no base analysis loaded');
		return;
	}

	// Check if user is trying to compare base with itself
	// (This would happen if the comparison ID matches the base ID)
	// For now we allow it but log a warning
	console.log(`[COMPARISON] Starting comparison: base vs ${comparisonAnalysisId}`);

	// Set loading state
	comparisonStore.update((state) => ({
		...state,
		isComparing: true,
		isLoading: true,
		error: null,
		comparisonAnalysisId,
		curtainPosition: 0.5 // Reset curtain to center
	}));

	try {
		// Load comparison analysis
		const analysis = await loadAnalysis(comparisonAnalysisId);

		comparisonStore.update((state) => ({
			...state,
			comparisonAnalysis: analysis,
			isLoading: false
		}));

		console.log(`[COMPARISON] Loaded comparison analysis: ${comparisonAnalysisId}`);
	} catch (error) {
		console.error('[COMPARISON] Failed to load comparison analysis:', error);

		comparisonStore.update((state) => ({
			...state,
			isLoading: false,
			error: error instanceof Error ? error.message : 'Failed to load comparison analysis'
		}));
	}
}

/**
 * Stop comparison mode and return to base-only view
 */
export function stopComparison(): void {
	console.log('[COMPARISON] Stopping comparison mode');

	comparisonStore.set({
		...defaultComparisonState,
		curtainPosition: 0.5 // Keep default position for next comparison
	});
}

/**
 * Set curtain position
 *
 * @param position - Position value (0-1), clamped to valid range
 */
export function setCurtainPosition(position: number): void {
	// Clamp position to valid range
	const clampedPosition = Math.max(0, Math.min(1, position));

	comparisonStore.update((state) => ({
		...state,
		curtainPosition: clampedPosition
	}));
}

/**
 * Snap curtain to anchor position
 *
 * @param anchor - Anchor position ('left' = 0, 'center' = 0.5, 'right' = 1)
 */
export function snapCurtainToAnchor(anchor: 'left' | 'center' | 'right'): void {
	const positions: Record<typeof anchor, number> = {
		left: 0,
		center: 0.5,
		right: 1
	};

	setCurtainPosition(positions[anchor]);
	console.log(`[COMPARISON] Curtain snapped to ${anchor}`);
}

/**
 * Nudge curtain position by a small amount
 *
 * @param direction - Direction to nudge ('left' or 'right')
 * @param amount - Amount to nudge (default 0.05)
 */
export function nudgeCurtain(direction: 'left' | 'right', amount: number = 0.05): void {
	const delta = direction === 'left' ? -amount : amount;
	const currentPosition = get(comparisonStore).curtainPosition;
	setCurtainPosition(currentPosition + delta);
}
