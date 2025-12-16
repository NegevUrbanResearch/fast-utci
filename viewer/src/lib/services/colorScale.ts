/**
 * Ladybug UTCI Color Scale Implementation for TypeScript
 * 
 * Ported from fast_utci/colors.py - provides the standard Ladybug Tools
 * 11-point UTCI color scale for thermal stress visualization.
 */

// Ladybug "Nuanced" gradient colors (11-point scale)
// Based on official Ladybug Tools Colorset.nuanced() RGB values
export const LADYBUG_NUANCED_COLORS = [
	'#313695', // (49, 54, 149) - Extreme Cold
	'#4575B4', // (69, 117, 180) - Very Strong Cold
	'#74ADD1', // (116, 173, 209) - Strong Cold
	'#ABD9E9', // (171, 217, 233) - Moderate Cold
	'#E0F3F8', // (224, 243, 248) - Slight Cold
	'#FFFFBF', // (255, 255, 191) - Comfortable
	'#FEE090', // (254, 224, 144) - Slight Heat
	'#FDAE61', // (253, 174, 97) - Moderate Heat
	'#F46D43', // (244, 109, 67) - Strong Heat
	'#D73027', // (215, 48, 39) - Very Strong Heat
	'#A50026'  // (165, 0, 38) - Extreme Heat
];

// UTCI thermal stress categories and temperature ranges
export interface UTCICategory {
	value: number;
	range: [number, number];
	label: string;
	abbrev: string;
}

export const UTCI_CATEGORIES: UTCICategory[] = [
	{ value: -5, range: [-Infinity, -40], label: 'Extreme Cold Stress', abbrev: 'Extreme Cold' },
	{ value: -4, range: [-40, -27], label: 'Very Strong Cold Stress', abbrev: 'Very Cold' },
	{ value: -3, range: [-27, -13], label: 'Strong Cold Stress', abbrev: 'Strong Cold' },
	{ value: -2, range: [-13, 0], label: 'Moderate Cold Stress', abbrev: 'Moderate Cold' },
	{ value: -1, range: [0, 9], label: 'Slight Cold Stress', abbrev: 'Slight Cold' },
	{ value: 0, range: [9, 26], label: 'No Thermal Stress', abbrev: 'Comfortable' },
	{ value: 1, range: [26, 28], label: 'Slight Heat Stress', abbrev: 'Slight Heat' },
	{ value: 2, range: [28, 32], label: 'Moderate Heat Stress', abbrev: 'Moderate Heat' },
	{ value: 3, range: [32, 38], label: 'Strong Heat Stress', abbrev: 'Strong Heat' },
	{ value: 4, range: [38, 46], label: 'Very Strong Heat Stress', abbrev: 'Very Strong Heat' },
	{ value: 5, range: [46, Infinity], label: 'Extreme Heat Stress', abbrev: 'Extreme Heat' }
];

/**
 * Get the UTCI thermal stress category for a given temperature
 * @param utciValue - UTCI temperature in Celsius
 * @returns Category value from -5 (extreme cold) to +5 (extreme heat)
 */
export function getUTCICategory(utciValue: number): number {
	for (const category of UTCI_CATEGORIES) {
		const [min, max] = category.range;
		if (utciValue >= min && utciValue < max) {
			return category.value;
		}
	}
	return 0; // Default to comfortable
}

/**
 * Get the Ladybug color for a UTCI value
 * @param utciValue - UTCI temperature in Celsius
 * @returns Hex color string
 */
export function getUTCIColor(utciValue: number): string {
	const category = getUTCICategory(utciValue);
	const index = category + 5; // Map -5 to +5 → 0 to 10
	return LADYBUG_NUANCED_COLORS[index];
}

/**
 * Get the thermal stress label for a UTCI value
 * @param utciValue - UTCI temperature in Celsius
 * @param abbreviated - Return abbreviated label if true
 * @returns Label string
 */
export function getUTCILabel(utciValue: number, abbreviated: boolean = false): string {
	const category = getUTCICategory(utciValue);
	const categoryData = UTCI_CATEGORIES[category + 5]; // Map -5 to +5 → 0 to 10
	return abbreviated ? categoryData.abbrev : categoryData.label;
}

/**
 * RGB color object with normalized values (0-1)
 */
export interface RGBColor {
	r: number;
	g: number;
	b: number;
}

/**
 * Convert hex color to normalized RGB object
 * @param hexColor - Hex color string (e.g., '#313695')
 * @returns RGB color object with normalized values (0-1)
 */
export function hexToThreeColor(hexColor: string): RGBColor {
	// Remove # if present
	const hex = hexColor.replace('#', '');
	const r = parseInt(hex.substr(0, 2), 16) / 255;
	const g = parseInt(hex.substr(2, 2), 16) / 255;
	const b = parseInt(hex.substr(4, 2), 16) / 255;
	return { r, g, b };
}

/**
 * Map UTCI value to color using dynamic colorscale (full spectrum across data range)
 * This matches the Python implementation's dynamic colorscale behavior
 * @param utciValue - UTCI temperature in Celsius
 * @param utciMin - Minimum UTCI in dataset
 * @param utciMax - Maximum UTCI in dataset
 * @returns RGB color object normalized to 0-1
 */
export function mapUTCIToColor(utciValue: number, utciMin: number, utciMax: number): RGBColor {
	// Normalize value to 0-1 range
	const normalized = (utciValue - utciMin) / (utciMax - utciMin);
	
	// Clamp to 0-1
	const clamped = Math.max(0, Math.min(1, normalized));
	
	// Map to color index (0-10 for 11 colors)
	const colorIndex = clamped * 10;
	const lowerIndex = Math.floor(colorIndex);
	const upperIndex = Math.min(10, Math.ceil(colorIndex));
	const fraction = colorIndex - lowerIndex;
	
	// Get colors
	const lowerColor = hexToThreeColor(LADYBUG_NUANCED_COLORS[lowerIndex]);
	const upperColor = hexToThreeColor(LADYBUG_NUANCED_COLORS[upperIndex]);
	
	// Interpolate
	return {
		r: lowerColor.r + (upperColor.r - lowerColor.r) * fraction,
		g: lowerColor.g + (upperColor.g - lowerColor.g) * fraction,
		b: lowerColor.b + (upperColor.b - lowerColor.b) * fraction
	};
}

/**
 * Create a legend data array for the UTCI scale
 * @returns Array of legend items with color and label
 */
export interface LegendItem {
	color: string;
	label: string;
	abbrev: string;
	range: [number, number];
}

export function createLegendData(): LegendItem[] {
	return UTCI_CATEGORIES.map((category, index) => ({
		color: LADYBUG_NUANCED_COLORS[index],
		label: category.label,
		abbrev: category.abbrev,
		range: category.range
	}));
}

// Shading Index color scale
// Categories based on Israeli Shading Metrics Guide:
// 0-0.5: Poor (red) - Less than 50% of time shaded
// 0.5-0.7: Acceptable (yellow/orange) - At least 50% of time shaded
// 0.7-0.9: Good (light green) - At least 70% of time shaded
// 0.9-1.0: Excellent (dark green) - At least 90% of time shaded
export const SHADING_INDEX_COLORS = [
	'#D73027', // Red - Poor (0-0.5)
	'#FEE08B', // Yellow - Acceptable (0.5-0.7)
	'#A6D96A', // Light Green - Good (0.7-0.9)
	'#1A9850'  // Dark Green - Excellent (0.9-1.0)
];

export interface ShadingIndexCategory {
	value: number;
	range: [number, number];
	label: string;
	abbrev: string;
}

export const SHADING_INDEX_CATEGORIES: ShadingIndexCategory[] = [
	{ value: 0, range: [0, 0.5], label: 'Poor Shading', abbrev: 'Poor' },
	{ value: 1, range: [0.5, 0.7], label: 'Acceptable Shading', abbrev: 'Acceptable' },
	{ value: 2, range: [0.7, 0.9], label: 'Good Shading', abbrev: 'Good' },
	{ value: 3, range: [0.9, 1.0], label: 'Excellent Shading', abbrev: 'Excellent' }
];

/**
 * Get the Shading Index category for a given value
 * @param shadingIndex - Shading Index value (0-1)
 * @returns Category value from 0 (poor) to 3 (excellent)
 */
export function getShadingIndexCategory(shadingIndex: number): number {
	for (const category of SHADING_INDEX_CATEGORIES) {
		const [min, max] = category.range;
		if (shadingIndex >= min && shadingIndex < max) {
			return category.value;
		}
	}
	// Handle edge case: exactly 1.0
	if (shadingIndex >= 1.0) {
		return 3; // Excellent
	}
	return 0; // Default to poor
}

/**
 * Get the color for a Shading Index value
 * @param shadingIndex - Shading Index value (0-1)
 * @returns Hex color string
 */
export function getShadingIndexColor(shadingIndex: number): string {
	const category = getShadingIndexCategory(shadingIndex);
	return SHADING_INDEX_COLORS[category];
}

/**
 * Map Shading Index value to color using smooth gradient interpolation
 * @param shadingIndex - Shading Index value (0-1)
 * @param shadingIndexMin - Minimum Shading Index in dataset (typically 0)
 * @param shadingIndexMax - Maximum Shading Index in dataset (typically 1)
 * @returns RGB color object normalized to 0-1
 */
export function mapShadingIndexToColor(
	shadingIndex: number,
	shadingIndexMin: number = 0,
	shadingIndexMax: number = 1
): RGBColor {
	// Normalize value to 0-1 range
	const normalized = (shadingIndex - shadingIndexMin) / (shadingIndexMax - shadingIndexMin);
	
	// Clamp to 0-1
	const clamped = Math.max(0, Math.min(1, normalized));
	
	// Map to color stops:
	// 0.0 -> Red (#D73027)
	// 0.5 -> Yellow (#FEE08B)
	// 0.7 -> Light Green (#A6D96A)
	// 1.0 -> Dark Green (#1A9850)
	
	let color: RGBColor;
	
	if (clamped < 0.5) {
		// Interpolate between red and yellow (0-0.5)
		const t = clamped / 0.5;
		const red = hexToThreeColor(SHADING_INDEX_COLORS[0]);
		const yellow = hexToThreeColor(SHADING_INDEX_COLORS[1]);
		color = {
			r: red.r + (yellow.r - red.r) * t,
			g: red.g + (yellow.g - red.g) * t,
			b: red.b + (yellow.b - red.b) * t
		};
	} else if (clamped < 0.7) {
		// Yellow (0.5-0.7)
		color = hexToThreeColor(SHADING_INDEX_COLORS[1]);
	} else if (clamped < 0.9) {
		// Interpolate between yellow and light green (0.7-0.9)
		const t = (clamped - 0.7) / 0.2;
		const yellow = hexToThreeColor(SHADING_INDEX_COLORS[1]);
		const lightGreen = hexToThreeColor(SHADING_INDEX_COLORS[2]);
		color = {
			r: yellow.r + (lightGreen.r - yellow.r) * t,
			g: yellow.g + (lightGreen.g - yellow.g) * t,
			b: yellow.b + (lightGreen.b - yellow.b) * t
		};
	} else {
		// Interpolate between light green and dark green (0.9-1.0)
		const t = (clamped - 0.9) / 0.1;
		const lightGreen = hexToThreeColor(SHADING_INDEX_COLORS[2]);
		const darkGreen = hexToThreeColor(SHADING_INDEX_COLORS[3]);
		color = {
			r: lightGreen.r + (darkGreen.r - lightGreen.r) * t,
			g: lightGreen.g + (darkGreen.g - lightGreen.g) * t,
			b: lightGreen.b + (darkGreen.b - lightGreen.b) * t
		};
	}
	
	return color;
}

/**
 * Create a legend data array for the Shading Index scale
 * @returns Array of legend items with color and label
 */
export interface ShadingIndexLegendItem {
	color: string;
	label: string;
	abbrev: string;
	range: [number, number];
}

export function createShadingIndexLegendData(): ShadingIndexLegendItem[] {
	return SHADING_INDEX_CATEGORIES.map((category, index) => ({
		color: SHADING_INDEX_COLORS[index],
		label: category.label,
		abbrev: category.abbrev,
		range: category.range
	}));
}


