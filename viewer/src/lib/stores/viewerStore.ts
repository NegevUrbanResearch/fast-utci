/**
 * Viewer Store
 * 
 * Svelte store for managing viewer state (hour, color mode, sun path visibility, etc.)
 */

import { writable, type Writable } from 'svelte/store';
import type { ViewerState, ColorMode } from '$lib/types/viewer';

/**
 * Viewer state store
 */
export const viewerStore: Writable<ViewerState> = writable<ViewerState>({
	currentHour: 0,
	colorMode: 'normalized',
	utciVisible: true,
	analysisId: null,
	loading: false,
	error: null
});

/**
 * Set current hour index
 * @param hour - Hour index (0-23)
 */
export function setCurrentHour(hour: number): void {
	viewerStore.update((state) => ({ ...state, currentHour: hour }));
}

/**
 * Set color mode
 * @param mode - Color mode ('normalized' or 'discrete')
 */
export function setColorMode(mode: ColorMode): void {
	viewerStore.update((state) => ({ ...state, colorMode: mode }));
}

/**
 * Set UTCI point cloud visibility
 * @param visible - Visibility state
 */
export function setUtciVisible(visible: boolean): void {
	viewerStore.update((state) => ({ ...state, utciVisible: visible }));
}

/**
 * Set analysis ID
 * @param analysisId - Analysis identifier
 */
export function setAnalysisId(analysisId: string | null): void {
	viewerStore.update((state) => ({ ...state, analysisId }));
}

/**
 * Set loading state
 * @param loading - Loading state
 */
export function setLoading(loading: boolean): void {
	viewerStore.update((state) => ({ ...state, loading }));
}

/**
 * Set error message
 * @param error - Error message or null
 */
export function setError(error: string | null): void {
	viewerStore.update((state) => ({ ...state, error }));
}


