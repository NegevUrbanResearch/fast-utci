/**
 * Viewer Store
 *
 * Svelte store for managing viewer state (hour, color mode, sun path visibility, theme, etc.)
 */

import { writable, type Writable } from 'svelte/store';
import type { ViewerState, ColorMode, MetricType } from '$lib/types/viewer';

/**
 * Viewer state store
 */
export const viewerStore: Writable<ViewerState> = writable<ViewerState>({
	currentHour: 0,
	currentMonth: 7,
	colorMode: 'normalized',
	metricType: 'utci',
	utciVisible: true,
	analysisId: null,
	loading: false,
	error: null,
	theme: 'dark'
});

/**
 * Set current hour index
 * @param hour - Hour index (0-23)
 */
export function setCurrentHour(hour: number): void {
	viewerStore.update((state) => ({ ...state, currentHour: hour }));
}

/**
 * Set current month index
 * @param month - Month index (0-11, 0=Jan, 7=Aug)
 */
export function setCurrentMonth(month: number): void {
	const clamped = Math.max(0, Math.min(11, month));
	viewerStore.update((state) => ({ ...state, currentMonth: clamped }));
}

/**
 * Set color mode
 * @param mode - Color mode ('normalized' or 'discrete')
 */
export function setColorMode(mode: ColorMode): void {
	viewerStore.update((state) => ({ ...state, colorMode: mode }));
}

/**
 * Set metric type
 * @param type - Metric type ('utci' or 'shading_index')
 */
export function setMetricType(type: MetricType): void {
	viewerStore.update((state) => ({ ...state, metricType: type }));
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

/**
 * Set theme (light or dark)
 * @param theme - 'light' or 'dark'
 */
export function setTheme(theme: 'dark' | 'light'): void {
	viewerStore.update((state) => ({ ...state, theme }));
}


