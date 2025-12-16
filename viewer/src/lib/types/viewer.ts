/**
 * Type definitions for viewer state
 */

/**
 * Color mode for full day analysis
 */
export type ColorMode = 'normalized' | 'discrete';

/**
 * Metric type for visualization
 */
export type MetricType = 'utci' | 'shading_index';

/**
 * Viewer state
 */
export interface ViewerState {
	currentHour: number;
	colorMode: ColorMode;
	metricType: MetricType;
	utciVisible: boolean;
	analysisId: string | null;
	loading: boolean;
	error: string | null;
	theme: 'dark' | 'light';
}


