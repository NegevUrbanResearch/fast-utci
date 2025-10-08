/**
 * Ladybug UTCI Color Scale Implementation for JavaScript
 * 
 * Ported from fast_utci/colors.py - provides the standard Ladybug Tools
 * 11-point UTCI color scale for thermal stress visualization.
 */

// Ladybug "Nuanced" gradient colors (11-point scale)
// Based on official Ladybug Tools Colorset.nuanced() RGB values
export const LADYBUG_NUANCED_COLORS = [
    '#313695',  // (49, 54, 149) - Extreme Cold
    '#4575B4',  // (69, 117, 180) - Very Strong Cold
    '#74ADD1',  // (116, 173, 209) - Strong Cold
    '#ABD9E9',  // (171, 217, 233) - Moderate Cold
    '#E0F3F8',  // (224, 243, 248) - Slight Cold
    '#FFFFBF',  // (255, 255, 191) - Comfortable
    '#FEE090',  // (254, 224, 144) - Slight Heat
    '#FDAE61',  // (253, 174, 97) - Moderate Heat
    '#F46D43',  // (244, 109, 67) - Strong Heat
    '#D73027',  // (215, 48, 39) - Very Strong Heat
    '#A50026'   // (165, 0, 38) - Extreme Heat
];

// UTCI thermal stress categories and temperature ranges
export const UTCI_CATEGORIES = [
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
 * @param {number} utciValue - UTCI temperature in Celsius
 * @returns {number} Category value from -5 (extreme cold) to +5 (extreme heat)
 */
export function getUTCICategory(utciValue) {
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
 * @param {number} utciValue - UTCI temperature in Celsius
 * @returns {string} Hex color string
 */
export function getUTCIColor(utciValue) {
    const category = getUTCICategory(utciValue);
    const index = category + 5; // Map -5 to +5 → 0 to 10
    return LADYBUG_NUANCED_COLORS[index];
}

/**
 * Get the thermal stress label for a UTCI value
 * @param {number} utciValue - UTCI temperature in Celsius
 * @param {boolean} abbreviated - Return abbreviated label if true
 * @returns {string} Label string
 */
export function getUTCILabel(utciValue, abbreviated = false) {
    const category = getUTCICategory(utciValue);
    const categoryData = UTCI_CATEGORIES[category + 5]; // Map -5 to +5 → 0 to 10
    return abbreviated ? categoryData.abbrev : categoryData.label;
}

/**
 * Convert hex color to THREE.js Color object
 * @param {string} hexColor - Hex color string (e.g., '#313695')
 * @returns {THREE.Color} THREE.js Color object
 */
export function hexToThreeColor(hexColor) {
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
 * @param {number} utciValue - UTCI temperature in Celsius
 * @param {number} utciMin - Minimum UTCI in dataset
 * @param {number} utciMax - Maximum UTCI in dataset
 * @returns {object} RGB color object {r, g, b} normalized to 0-1
 */
export function mapUTCIToColor(utciValue, utciMin, utciMax) {
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
 * @returns {Array} Array of legend items with color and label
 */
export function createLegendData() {
    return UTCI_CATEGORIES.map((category, index) => ({
        color: LADYBUG_NUANCED_COLORS[index],
        label: category.label,
        abbrev: category.abbrev,
        range: category.range
    }));
}
