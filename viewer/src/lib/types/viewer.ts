/**
 * Type definitions for viewer state
 */

/**
 * Color mode for full day analysis
 */
export type ColorMode = 'normalized' | 'discrete';

/**
 * Viewer state
 */
export interface ViewerState {
	currentHour: number;
	colorMode: ColorMode;
	utciVisible: boolean;
	analysisId: string | null;
	loading: boolean;
	error: string | null;
	theme: 'dark' | 'light';
}


